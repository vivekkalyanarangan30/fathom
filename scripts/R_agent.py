"""Agent-session benchmark: real OpenHands coding trajectories on one repository concatenated into one long session; each scan decodes the agent's next step from the same cache. Derived from R_ruler.py.
RULER-lite under sketch-selected top-k decode. One dense HF prefill per sample (sdpa, bf16 KV kept on the GPU), then greedy decode
branched per method from the same cache: every query attends to sink + local + top-K prefill tokens ranked by the method's scan.
Tasks (RULER templates, Wikitext haystack): niah_single, niah_multikey, niah_multivalue, niah_multiquery, vt, fwe. Works for Qwen2 and Qwen3 attention.
planesK:u4:Br = our scan over KLT-rotated keys (flat budget Br; the basis rule for models without QK-norm).
landmark:b = Quest / ShadowKV-style block selection from fp16 block-mean keys (our reimplementation; 16*D/b bits per token).
Score = fraction of gold strings contained in the generation (RULER's partial match).
Usage: python scripts/R_ruler.py --model Qwen/Qwen3-8B --tag Qwen3-8B_T32768 --ctx 32000 --k 128 --ns 50 [--tasks a,b] [--methods ...] [--selfcheck]"""
import os, sys, json, math, time, random, argparse, torch
from common import *
from sketch import build_params, params_to, approx_scores, qblock, scale_bits, SINK, LOCAL, BLK
import importlib
torch.set_grad_enabled(False)

if torch.cuda.is_available():                                                           # a materialised 128k x 128k mask OOMs; one unpadded sequence -> is_causal is exact
    import transformers.integrations.sdpa_attention as _sa
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as _AAF
    _orig_sdpa = _sa.sdpa_attention_forward
    def _sdpa_causal(module, query, key, value, attention_mask, **kw):
        kw["is_causal"] = query.shape[2] > 1 if kw.get("is_causal") is None else kw["is_causal"]
        return _orig_sdpa(module, query, key, value, None, **kw)
    _AAF["sdpa"] = _sdpa_causal; torch.backends.cuda.enable_math_sdp(False)
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-8B"); ap.add_argument("--tag", default="Qwen3-8B_T32768"); ap.add_argument("--ctx", type=int, default=32000)
ap.add_argument("--k", type=int, default=128); ap.add_argument("--ns", type=int, default=50); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tasks", default="niah_single,niah_multikey,niah_multivalue,niah_multiquery,vt,fwe")
ap.add_argument("--methods", default="dense,exact_topk,planesL:u4:48,planesL:u4:64,ds:32:4,sparq:16:4,loki:64:4,thumb:2,landmark:8")
ap.add_argument("--out", default=None); ap.add_argument("--selfcheck", action="store_true", help="compare the decode scorers against sketch.approx_scores and exit")
ap.add_argument("--calib-tag", default=None, help="second calibration tag; methods planesA/dsA use its statistics, planesS calibrates on the session's own keys")
ap.add_argument("--gen-max", type=int, default=300); ap.add_argument("--min-ctx", type=int, default=90000); ap.add_argument("--dataset", default="nebius/SWE-rebench-openhands-trajectories")
a = ap.parse_args()
assert 512 <= a.ctx <= 262144 and 40 <= a.k <= 4096 and 1 <= a.ns <= 1000
dev = device()
T = int(a.tag.split("_T")[-1]); base = a.tag.split("_T")[0]

# ------------------------------------------------------------------ decode-time scorers (batched over kv heads; same arithmetic as sketch.approx_scores)
def codes_u4(K):
    """K [H, n, D] fp32 -> (codes int16 in [0,16), amax [H, nb, D]) with sketch.qblock's per-block absmax mid-rise quantiser."""
    H, n, D = K.shape; nb = math.ceil(n / BLK)
    cb = torch.cat([K, K.new_zeros(H, nb * BLK - n, D)], 1).view(H, nb, BLK, D)
    amax = cb.abs().amax(2, keepdim=True).clamp_min(1e-8)
    return ((cb / (amax / 8)).floor().clamp(-8, 7) + 8).to(torch.int16), amax

def dequant(codes, amax, t):
    """Top-t planes of the 4-bit codes, t [H, D] ints in [0, 4] -> values [H, n_padded, D] fp32 (0 where t == 0)."""
    H, nb, _, D = codes.shape; tt = t.view(H, 1, 1, D).to(torch.int16)
    ct = (codes >> (4 - tt)).float(); half = 2.0 ** (tt.float() - 1)
    val = (ct + 0.5 - half) * amax / half
    return torch.where(tt > 0, val, torch.zeros_like(val)).view(H, nb * BLK, D)

def waterfill_t(g, Br, bmax=4):
    """g [H, D] gain -> integer depth per channel with sum <= Br (bisection on log theta, as in sketch.py)."""
    bf = torch.full_like(g, float(bmax)); lo = torch.full((g.shape[0], 1), -60.0, device=g.device); hi = torch.full_like(lo, 60.0); lg = torch.log2(g.clamp_min(1e-30))
    for _ in range(30):
        mid = (lo + hi) / 2; t = ((lg - mid) / 2).round().clamp(min=0).minimum(bf); tot = t.sum(-1, keepdim=True)
        lo = torch.where(tot > Br, mid, lo); hi = torch.where(tot > Br, hi, mid)
    return ((lg - hi) / 2).round().clamp(min=0).minimum(bf)

def self_klt(Kp):
    """KLT statistics of one layer from the session's own prefill keys Kp [Hkv, n, D] (no offline calibration): mu, eigenbasis, eigenvalues."""
    X = Kp[:, SINK * 4:]; mu = X.mean(1); X = X - mu[:, None]
    lam, V = torch.linalg.eigh(torch.einsum("hnd,hne->hde", X, X) / X.shape[1])
    return {"mu": mu, "Vt": V, "lam": lam.clamp_min(0)}

class Stores:
    """Per-layer query-independent quantised stores for one method over the prefill keys Kp [Hkv, n, D] fp32.
    planesA / dsA: planesK / ds with the statistics of --calib-tag; planesS: planesK calibrated on the session itself."""
    def __init__(self, method, P, nL, get_K, P2=None):
        self.name, *args = method.split(":"); self.args = args; self.S = []
        if self.name in ("planesA", "dsA"):
            assert P2 is not None, "--calib-tag required for planesA/dsA"; P = P2; self.name = {"planesA": "planesK", "dsA": "ds"}[self.name]
        elif self.name == "planesS": P = [self_klt(get_K(L)) for L in range(nL)]; self.name = "planesK"
        self.P = P
        for L in range(nL):
            Kp = get_K(L); p = P[L]; s = {}
            if self.name in ("planesL", "thumb", "sparq"): s["codes"], s["amax"] = codes_u4(Kp)
            elif self.name == "planesK":                                                         # bit planes of the KLT-rotated keys (K - mu) @ Vt, for models without QK-norm
                s["codes"], s["amax"] = codes_u4(torch.einsum("hnd,hde->hne", Kp - p["mu"][:, None], p["Vt"]))
            elif self.name == "ds":
                c = int(args[0]); ch = p["ch_order"][:, :c]                                            # [Hkv, c]
                s["ch"] = ch; s["store"] = torch.stack([qblock(Kp[h][:, ch[h]], torch.full((c,), int(args[1]))) for h in range(Kp.shape[0])])
            elif self.name == "landmark":
                b = int(args[0]); H, n, D = Kp.shape; nb = math.ceil(n / b)
                s["lm"] = torch.cat([Kp, Kp.new_zeros(H, nb * b - n, D)], 1).view(H, nb, b, D).mean(2).half().float(); s["b"] = b
            elif self.name == "loki":
                r = int(args[0]); Vt = p["Vt"][:, :, :r]; s["Vt"] = Vt
                s["store"] = torch.stack([qblock((Kp[h] - p["mu"][h]) @ Vt[h], torch.full((r,), int(args[1]))) for h in range(Kp.shape[0])])
            self.S.append(s)
    def scores(self, q, L, n):
        """q [Hkv, G, D] fp32 -> approximate scores [Hkv, G, n] over the prefill tokens and bits/token (scan bytes per kv head)."""
        p = self.P[L]; s = self.S[L]; H, G, D = q.shape
        if self.name == "planesL":
            Br = p["alloc_planes"][self.args[0]][int(self.args[1])]
            t = waterfill_t(((q ** 2) * p["var_raw"][:, None]).sum(1), Br)
            return torch.einsum("hgd,hnd->hgn", q, dequant(s["codes"], s["amax"], t)[:, :n]), (t.sum(-1) + scale_bits((t > 0).float().sum(-1))).mean().item()
        if self.name == "planesK":                                                             # planesK:u4:Br -- flat budget Br, gains (q Vt)^2 * lambda
            qr = torch.einsum("hgd,hde->hge", q, p["Vt"]); t = waterfill_t(((qr ** 2) * p["lam"][:, None]).sum(1), int(self.args[1]))
            return torch.einsum("hgd,hnd->hgn", qr, dequant(s["codes"], s["amax"], t)[:, :n]), (t.sum(-1) + scale_bits((t > 0).float().sum(-1))).mean().item()
        if self.name == "thumb":
            t = torch.full((H, D), float(self.args[0]), device=q.device)
            return torch.einsum("hgd,hnd->hgn", q, dequant(s["codes"], s["amax"], t)[:, :n]), (t.sum(-1) + scale_bits((t > 0).float().sum(-1))).mean().item()
        if self.name == "sparq":                                                             # SparQ under GQA: top-r of the group's sum |q| per KV head, r channels at 4 bits
            r = int(self.args[0]); top = q.abs().sum(1).topk(r, -1).indices; qm = q * torch.zeros(H, D, device=q.device).scatter(-1, top, 1.0)[:, None]
            t = torch.full((H, D), 4, device=q.device)
            return torch.einsum("hgd,hnd->hgn", qm, dequant(s["codes"], s["amax"], t)[:, :n]), r * 4 + scale_bits(r)
        if self.name == "ds":
            c = s["ch"].shape[1]; qs = torch.gather(q, 2, s["ch"][:, None, :].expand(-1, G, -1))
            return torch.einsum("hgc,hnc->hgn", qs, s["store"]), c * int(self.args[1]) + scale_bits(c)
        if self.name == "landmark":                                          # block score for every token of the block; tiny within-block ramp so top-k fills whole blocks
            b = s["b"]; sb = torch.einsum("hgd,hnd->hgn", q, s["lm"])
            ramp = torch.arange(b, device=q.device, dtype=torch.float32) * 1e-6
            return (sb[..., None] - ramp).reshape(H, G, -1)[..., :n], D * 16 / b
        if self.name == "loki":
            r = s["Vt"].shape[-1]; qp = torch.einsum("hgd,hdr->hgr", q, s["Vt"])
            return torch.einsum("hgr,hnr->hgn", qp, s["store"]), r * int(self.args[1]) + scale_bits(r)
        raise ValueError(self.name)

def selfcheck(P, G):
    """Decode scorers must reproduce sketch.approx_scores on random keys (fp32, per head) before any GPU hour is spent."""
    torch.manual_seed(0); L = 0; Hkv, D = P[L]["var_raw"].shape; n = 700
    K = torch.randn(Hkv, n, D, device=dev); q = torch.randn(Hkv, G, D, device=dev)
    for m in ["planesL:u4:48", "planesL:u4:64", "ds:32:4", "sparq:16:4", "loki:64:4"]:
        st = Stores(m, P, 1, lambda L: K); sc, bits = st.scores(q, 0, n); rb = []
        for h in range(Hkv):
            ref, rbits = approx_scores(m, q[h][:, None], K[h], P[L], h); rb.append(rbits)
            d = (sc[h] - ref[:, 0]).abs().max().item(); rel = d / ref.abs().max().item()
            assert rel < 1e-4, (m, h, rel)
        assert abs(bits - sum(rb) / len(rb)) < 1e-6, (m, bits, sum(rb) / len(rb))              # bits are a mean over KV heads (active-channel scales differ per head)
        print(f"selfcheck {m:14s} ok (max rel diff over heads < 1e-4, bits {bits:.1f})", flush=True)

# ------------------------------------------------------------------ model + patched decode attention
tok = AutoTokenizer.from_pretrained(a.model)
model, _ = load_model(a.model, attn="sdpa"); cfg = model.config
Hq, Hkv, nL = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.num_hidden_layers; D = getattr(cfg, "head_dim", None) or cfg.hidden_size // Hq; G = Hq // Hkv
P = params_to(build_params(f"{RES}/caps_{base}_train_T{T}.safetensors", G), dev)
assert P[0]["alloc_planes"] is not None, f"missing {RES}/final_alloc_{a.tag}.json"
P2 = params_to(build_params(f"{RES}/caps_{a.calib_tag.split('_T')[0]}_train_T{int(a.calib_tag.split('_T')[-1])}.safetensors", G), dev) if a.calib_tag else None
CAP = a.ctx + 400
ST = {"mode": "prefill", "method": "dense", "n": 0, "len": 0, "stores": None, "bits": []}
KC = [torch.empty(1, Hkv, CAP, D, dtype=model.dtype, device=dev) for _ in range(nL)]; VC = [torch.empty_like(KC[0]) for _ in range(nL)]
ATT = type(model.model.layers[0].self_attn); mq = importlib.import_module(ATT.__module__); orig_forward = ATT.forward

def decode_attn(q, L, m):
    """q [1, Hq, 1, D] -> attention over sink + local + top-K of the n prefill tokens (plus exact scoring of the m - n generated tokens)."""
    n = ST["n"]; K = a.k; k = KC[L][0, :, :m]; v = VC[L][0, :, :m]; qf = q.view(Hkv, G, D).float()
    kf = k.float(); exact = torch.einsum("hgd,hnd->hgn", qf, kf) * D ** -0.5
    if ST["method"] == "dense": return torch.einsum("hgn,hnd->hgd", torch.softmax(exact, -1), v.float()).to(q.dtype).view(1, Hq, 1, D)
    if ST["method"] == "exact_topk": approx = exact[..., :n].clone()
    else:
        approx, bits = ST["stores"].scores(qf, L, n); ST["bits"].append(bits)
    approx[..., :SINK] = -1e9; approx[..., max(n - LOCAL, 0):] = -1e9
    top = approx.topk(K - SINK - LOCAL, -1).indices
    sel = torch.zeros(Hkv, G, m, dtype=torch.bool, device=dev).scatter(-1, top, True); sel[..., :SINK] = True; sel[..., m - LOCAL:] = True
    pw = torch.softmax(exact.masked_fill(~sel, float("-inf")), -1)
    return torch.einsum("hgn,hnd->hgd", pw, v.float()).to(q.dtype).view(1, Hq, 1, D)

def patched_forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, cache_position=None, **kw):
    if ST["mode"] == "prefill": return orig_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values, cache_position, **kw)
    B, Tq, _ = hidden_states.shape; L = self.layer_idx
    qn = getattr(self, "q_norm", None) or (lambda x: x); kn = getattr(self, "k_norm", None) or (lambda x: x)
    q = qn(self.q_proj(hidden_states).view(B, Tq, -1, D)).transpose(1, 2); k = kn(self.k_proj(hidden_states).view(B, Tq, -1, D)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(B, Tq, -1, D).transpose(1, 2); cos, sin = position_embeddings; q, k = mq.apply_rotary_pos_emb(q, k, cos, sin)
    m = ST["len"]; KC[L][:, :, m] = k[:, :, 0]; VC[L][:, :, m] = v[:, :, 0]
    o = decode_attn(q, L, m + 1)
    return self.o_proj(o.transpose(1, 2).reshape(B, Tq, -1)), None
ATT.forward = patched_forward

def prefill(ids):
    """Dense HF prefill; copies the post-RoPE K/V of every layer into the decode cache. Returns the last-token logits."""
    ST["mode"] = "prefill"; n = ids.shape[1]; assert n <= CAP
    out = model(ids.to(dev), use_cache=True, logits_to_keep=1); pk = out.past_key_values
    for L in range(nL):
        kl, vl = (pk.layers[L].keys, pk.layers[L].values) if hasattr(pk, "layers") else (pk.key_cache[L], pk.value_cache[L])
        KC[L][:, :, :n] = kl; VC[L][:, :, :n] = vl
    ST["n"] = n; del pk, out.past_key_values; return out.logits[0, -1].clone()

def generate(first_logits, max_new, stop_ids, want_tokens=False):
    ST["mode"] = "decode"; ST["len"] = ST["n"]; toks = []; nxt = first_logits.argmax().item()
    for _ in range(max_new):
        if nxt in stop_ids: break
        toks.append(nxt)
        logits = model(torch.tensor([[nxt]], device=dev), position_ids=torch.tensor([[ST["len"]]], device=dev), use_cache=False).logits[0, -1]
        ST["len"] += 1; nxt = logits.argmax().item()
    return (tok.decode(toks), toks) if want_tokens else tok.decode(toks)

def check_dense(ids, steps=8):
    """Our dense decode path must reproduce HF's own greedy generation (cached, unpatched attention) token for token."""
    ST["mode"] = "prefill"; ref = model.generate(ids.to(dev), max_new_tokens=steps, do_sample=False, min_new_tokens=steps)[0, ids.shape[1]:].tolist()
    ST["method"] = "dense"; ours = generate(prefill(ids), steps, set())
    ours_ids = tok(ours, add_special_tokens=False).input_ids
    print(f"check_dense: hf={tok.decode(ref)!r} ours={ours!r}", flush=True); assert tok.decode(ref) == ours, "dense decode path differs from HF generate"

# ------------------------------------------------------------------ sessions from real agent trajectories
import re, random
rng = random.Random(a.seed)
from datasets import load_dataset
def ntok(s): return len(tok(s, add_special_tokens=False).input_ids)

def load_sessions(ns):
    """Deterministic sessions: trajectories of one repo in trajectory_id order, concatenated until the context reaches [min_ctx, ctx];
    the target is the next assistant step of the last trajectory. Returns list of (prefix_text, gold_text, meta)."""
    ds = load_dataset(a.dataset, split="train")
    by_repo = {}
    for i, r in enumerate(ds): by_repo.setdefault(r["repo"], []).append(i)
    repos = sorted(r for r, ix in by_repo.items() if len(ix) >= 2); rng.shuffle(repos)
    out = []
    for repo in repos:
        if len(out) >= ns: break
        ix = sorted(by_repo[repo], key=lambda i: ds[i]["trajectory_id"]); rendered = []; sys_done = False
        for i in ix:
            tr = ds[i]["trajectory"]
            if isinstance(tr, str): tr = json.loads(tr)
            msgs = [m for m in tr if not (m.get("role") == "system" and sys_done)]
            if msgs and msgs[0].get("role") == "system": sys_done = True
            rendered.append([render_msg(m) for m in msgs])
        # candidate: last trajectory j with an assistant step t (having a tool call) such that prefix length fits
        toks = [[ntok(x) for x in traj] for traj in rendered]
        for j in range(len(rendered) - 1, -1, -1):
            traj = rendered[j]; tr = ds[ix[j]]["trajectory"]; tr = json.loads(tr) if isinstance(tr, str) else tr
            tr = [m for m in tr if not (m.get("role") == "system" and j > 0)]
            ass = [t for t, m in enumerate(tr) if m.get("role") == "assistant" and (m.get("tool_calls") or "<tool_call>" in (m.get("content") or "")) and t >= 4]
            for t in reversed(ass):
                prefix_len = sum(toks[j][:t]); jj = j - 1
                while jj >= 0 and prefix_len + sum(toks[jj]) <= a.ctx - 64: prefix_len += sum(toks[jj]); jj -= 1
                if prefix_len >= a.min_ctx:
                    parts = [x for q in range(jj + 1, j) for x in rendered[q]] + traj[:t]
                    prefix = "".join(parts) + f"{IM_S}assistant\n"
                    out.append((prefix, tr[t], dict(repo=repo, n_traj=j - jj, step=t, instance_id=ds[ix[j]]["instance_id"]))); break
            else: continue
            break
    return out

def first_call(text):
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    if not m: return None
    try: d = json.loads(m.group(1)); args = d.get("arguments", {}); args = json.loads(args) if isinstance(args, str) else args
    except Exception: return None
    key = next((k for k in ("path", "file_path", "filename", "command") if isinstance(args, dict) and k in args), None)
    return (d.get("name"), args.get(key) if key else None)
def gold_text(m):
    c = m.get("content") or ""
    for cl in m.get("tool_calls") or []:
        f = cl.get("function", cl); args = f.get("arguments", "")
        c += f"\n<tool_call>\n{json.dumps({'name': f.get('name', ''), 'arguments': args if isinstance(args, str) else json.dumps(args)})}\n</tool_call>"
    return c
def agree(gen_toks, ref_toks):
    k = 0
    for x, y in zip(gen_toks, ref_toks):
        if x != y: break
        k += 1
    return k / max(1, len(ref_toks))

if a.selfcheck:
    selfcheck(P, G); check_dense(tok(IM_S + "user\nhello" + IM_E + "\n" + IM_S + "assistant\n", return_tensors="pt", add_special_tokens=False).input_ids); print("selfcheck passed", flush=True); sys.exit(0)

# ------------------------------------------------------------------ main loop (resumable)
out = a.out or f"{RES}/agent_{a.tag}_K{a.k}_ctx{a.ctx}.json"
res = json.load(open(out)) if os.path.exists(out) else {}
res["_meta"] = dict(model=a.model, tag=a.tag, calib_tag=a.calib_tag, ctx=a.ctx, min_ctx=a.min_ctx, K=a.k, ns=a.ns, seed=a.seed, sink=SINK, local=LOCAL, dataset=a.dataset, gen_max=a.gen_max,
                    protocol="real OpenHands trajectories of one repo concatenated; dense prefill of the session prefix; each method greedily decodes the agent's next step from the same cache")
methods = a.methods.split(","); assert methods[0] == "dense", "dense must be the first method (agreement reference)"
stop_ids = {tok.eos_token_id} | set(tok(IM_E, add_special_tokens=False).input_ids)
sessions = load_sessions(a.ns); print(f"{len(sessions)} sessions built", flush=True)
res.setdefault("sessions", []); done = len(res["sessions"])
for i, (prefix, gold_msg, meta) in enumerate(sessions):
    if i < done: continue
    ids = tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids[0][-a.ctx:]
    gold = gold_text(gold_msg); gcall = first_call(gold); rec = dict(meta=meta, n_tokens=len(ids), gold=gold[:2000], gold_call=gcall, methods={})
    t0 = time.time(); first = prefill(ids[None]); tp = time.time() - t0; ref_toks = None
    for m in methods:
        ST["method"] = m; ST["bits"] = []
        ST["stores"] = Stores(m, P, nL, lambda L: KC[L][0, :, :ST["n"]].float(), P2) if m not in ("dense", "exact_topk") else None
        t1 = time.time(); text, toks = generate(first, a.gen_max, stop_ids, want_tokens=True); ST["stores"] = None
        if m == "dense": ref_toks, ref_text = toks, text
        call = first_call(text)
        rec["methods"][m] = dict(gen=text[:2000], n_gen=len(toks), agree=agree(toks, ref_toks), exact=int(text == ref_text), call_match_dense=int(call is not None and call == first_call(ref_text)),
                                 call_match_gold=int(call is not None and call == gcall), bits=sum(ST["bits"]) / max(1, len(ST["bits"])), sec=time.time() - t1)
    res["sessions"].append(rec); json.dump(res, open(out, "w"), indent=1); torch.cuda.empty_cache()
    print(f"session {i + 1}/{len(sessions)} {meta['repo']} ({len(ids)} tok, prefill {tp:.1f}s): " + "  ".join(f"{m}={rec['methods'][m]['agree']:.2f}/{rec['methods'][m]['call_match_dense']}" for m in methods), flush=True)
print("done ->", out)
