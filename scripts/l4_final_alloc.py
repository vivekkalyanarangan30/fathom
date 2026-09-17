"""GPU port of 16_final_alloc.py for the L4 phase. Same sweep (bit-plane stores rd128/u4/u8 x per-query water-filled read budgets,
greedy per-layer allocation on calibration errors) plus: K as a CLI argument, Loki r=32 / Double-Sparsity c=32 4-bit rows (ported from
12_allayers.py), top-k recall vs the exact top-k, optional task-E rows (--extras: FFD 2-bit thumbnail + 4-bit re-score, 1-bit sign index),
and evaluation of per-layer plans calibrated at another T (--alloc-from TAG). Note `u4/planes48` is the flat-48-bits-in-every-layer ablation.
Usage: python scripts/l4_final_alloc.py Qwen3-8B_T16384 256 [--extras] [--alloc-from Qwen3-8B_T16384]"""
import os, sys, json, math, time, argparse, torch
from capsio import safe_open   # lazy pread reader (no mmap)
from common import RES
from sketch import waterfill, scale_bits, SINK, LOCAL, BLK
ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("K", type=int); ap.add_argument("--tq", type=int, default=128)
ap.add_argument("--extras", action="store_true"); ap.add_argument("--alloc-from", default=None); ap.add_argument("--out", default=None)
ap.add_argument("--layers", default=None, help="comma list, for quick tests")
ap.add_argument("--klt", action="store_true", help="add the u4klt store: uniform 4-bit planes of the KLT-rotated keys (K - mu) @ Vt (task C)")
ap.add_argument("--stores", default="rd128,u4,u8", help="subset of stores to sweep")
ap.add_argument("--test-tag", default=None, help="measure error on this tag's test capture (calibration still from `tag`)")
args = ap.parse_args(); torch.set_grad_enabled(False); dev = torch.device(os.environ.get("NQ_DEVICE", "cuda"))
tag, K, TQ = args.tag, args.K, args.tq
BUDGETS = (24, 32, 40, 48, 64, 80, 96, 128)
STORES = {"rd128": ("rd", 128, 8), "u4": ("uniform", 4, 4), "u8": ("uniform", 8, 8)}
if "--klt" in sys.argv: STORES["u4klt"] = ("uniform_klt", 4, 4)
STORES = {k: v for k, v in STORES.items() if k in args.stores.split(",") or k == "u4klt"}
base, Ttag = tag.split("_T")[0], tag.split("_")[-1]
tbase, tTtag = (args.test_tag or tag).split("_T")[0], (args.test_tag or tag).split("_")[-1]
ftr = safe_open(f"{RES}/caps_{base}_train_{Ttag}.safetensors", "pt"); fte = safe_open(f"{RES}/caps_{tbase}_test_{tTtag}.safetensors", "pt")
Hq, T, D = ftr.get_slice("q0").get_shape(); Hkv = ftr.get_slice("k0").get_shape()[0]; G = Hq // Hkv
nL = len([k for k in ftr.keys() if k.startswith("q")]); LAYERS = list(range(nL)) if args.layers is None else [int(x) for x in args.layers.split(",")]
pos = torch.arange(T - TQ, T, device=dev); mask = torch.arange(T, device=dev)[None] <= pos[:, None]
keep = torch.zeros(TQ, T, dtype=torch.bool, device=dev); keep[:, :SINK] = True
for i, p in enumerate(pos.tolist()): keep[i, p - LOCAL + 1:p + 1] = True
keep_idx = keep.nonzero()[:, 1].view(TQ, -1); kk = K - SINK - LOCAL
n_valid = (mask & ~keep).float().sum(-1).mean().item()          # candidate tokens per query (for FFD re-score fraction)
print(f"{tag}: Hq={Hq} Hkv={Hkv} G={G} T={T} D={D} nL={nL} K={K} TQ={TQ} kk={kk}", flush=True)

def evaluate(approx, exact, lse, V, o_ref, ex_top):
    approx = approx.masked_fill(~mask | keep, -1e9)
    top = approx.topk(kk, -1).indices
    sel = torch.cat([top, keep_idx.expand(approx.shape[0], -1, -1)], -1)
    s_sel = exact.gather(-1, sel); mass = torch.exp(s_sel - lse).sum(-1).mean().item()
    o = torch.einsum("hqk,hqkd->hqd", torch.softmax(s_sel, -1), V[sel])
    err = ((o - o_ref).norm(dim=-1) / o_ref.norm(dim=-1)).mean().item()
    hit = torch.zeros(approx.shape, dtype=torch.bool, device=dev).scatter(-1, ex_top, True).gather(-1, top)
    return mass, err, hit.float().mean().item()

def qblock_dev(c, bits):
    n, r = c.shape; nb = math.ceil(n / BLK); cb = torch.cat([c, c.new_zeros(nb * BLK - n, r)]).view(nb, BLK, r)
    amax = cb.abs().amax(1, keepdim=True).clamp_min(1e-8); half = 2.0 ** (bits.clamp_min(1) - 1).float(); s = amax / half
    out = ((cb / s).floor().clamp(-half, half - 1) + 0.5) * s; out[..., bits <= 0] = 0
    return out.view(-1, r)[:n]

def plane_store(Ke, bits, bmax):
    n, r = Ke.shape; nb = math.ceil(n / BLK); cb = torch.cat([Ke, Ke.new_zeros(nb * BLK - n, r)]).view(nb, BLK, r)
    amax = cb.abs().amax(1, keepdim=True).clamp_min(1e-8); out = []
    for t in range(1, bmax + 1):
        half = 2.0 ** (t - 1); s = amax / half
        q = ((cb / s).floor().clamp(-half, half - 1) + 0.5) * s; q[..., bits < t] = 0
        out.append(q.view(-1, r)[:n])
    return torch.stack(out)

def waterfill_read(g, bits, Br, iters=30):
    bf = bits.float(); lo = torch.full((g.shape[0], 1), -60.0, device=dev); hi = torch.full((g.shape[0], 1), 60.0, device=dev); lg = torch.log2(g.clamp_min(1e-30))
    for _ in range(iters):
        mid = (lo + hi) / 2; t = ((lg - mid) / 2).round().clamp(min=0).minimum(bf); tot = t.sum(-1, keepdim=True)
        lo = torch.where(tot > Br, mid, lo); hi = torch.where(tot > Br, hi, mid)
    return ((lg - hi) / 2).round().clamp(min=0).minimum(bf)

def score_with_depths(Q, t, planes):
    out = torch.zeros(Q.shape[0], Q.shape[1], planes.shape[1], device=dev)
    for lvl in range(1, planes.shape[0] + 1):
        m = (t == lvl).float()
        if m.sum() == 0: continue
        out += (Q * m[None]) @ planes[lvl - 1].T
    return out

err = {"cal": {}, "test": {}}; bits_used = {}; mass_t = {}; rec_t = {}; act_t = {}
def rec(split, name, L, e, b, m, r, act=None):
    err[split].setdefault(name, {}).setdefault(L, []).append(e); bits_used.setdefault(name, []).append(b)
    if act is not None and split == "test": act_t.setdefault(name, []).append(act)
    if split == "test": mass_t.setdefault(name, []).append(m); rec_t.setdefault(name, []).append(r)

t_start = time.time()
for L in LAYERS:
    for hk in range(Hkv):
        Kc = ftr.get_slice(f"k{L}")[hk].to(dev).float(); Kfit = Kc[SINK * 4:]; var = Kfit.var(0); mu = Kfit.mean(0)
        Qcal_all = ftr.get_slice(f"q{L}")[hk * G:(hk + 1) * G].to(dev).float(); Qcal = Qcal_all.reshape(-1, D); w = (Qcal ** 2).mean(0) * var
        Kf = Kfit - mu; cov = Kf.T @ Kf / len(Kf); lam, V = torch.linalg.eigh(cov); Vt_full = V[:, lam.argsort(descending=True)]
        wk = ((Qcal @ Vt_full) ** 2).mean(0) * lam.sort(descending=True).values; Vt_full = Vt_full[:, wk.argsort(descending=True)]   # KLT, score-variance ordered
        var_klt = (Kf @ Vt_full).var(0)
        allocs = {}
        for sname, (kind, B, bmax) in STORES.items():
            allocs[sname] = ((waterfill(w.cpu(), B, bmax=bmax) if kind == "rd" else torch.full((D,), B)).to(dev), bmax)
        for split, fk in (("cal", ftr), ("test", fte)):
            Ke = fk.get_slice(f"k{L}")[hk].to(dev).float(); Ve = fk.get_slice(f"v{L}")[hk].to(dev).float()
            Q = fk.get_slice(f"q{L}")[hk * G:(hk + 1) * G, T - TQ:].to(dev).float()
            exact = (Q @ Ke.T).masked_fill(~mask, -1e9); lse = torch.logsumexp(exact, -1, keepdim=True); o_ref = torch.softmax(exact, -1) @ Ve
            ex_top = exact.masked_fill(keep, -1e9).topk(kk, -1).indices
            g = ((Q ** 2) * var).sum(0); Qr = Q @ Vt_full; g_klt = ((Qr ** 2) * var_klt).sum(0)
            for sname, (bits, bmax) in allocs.items():
                klt = sname.endswith("klt")
                planes = plane_store((Ke - mu) @ Vt_full if klt else Ke, bits, bmax)
                for Br in BUDGETS:
                    t = waterfill_read(g_klt if klt else g, bits, Br); m, e, r = evaluate(score_with_depths(Qr if klt else Q, t, planes), exact, lse, Ve, o_ref, ex_top)
                    nact = (t > 0).float().sum(-1); rec(split, f"{sname}/planes{Br}", L, e, (t.sum(-1) + scale_bits(nact)).mean().item(), m, r, nact.mean().item())
                del planes
            if split == "test":
                Kq4 = qblock_dev(Ke, torch.full((D,), 4, device=dev))
                for r_ in (16, 32):
                    # SparQ under GQA (Ribar et al. Sec. 5): top-r channels of the group-summed |q|, one pick per KV head
                    topg = Q.abs().sum(0).topk(r_, -1).indices; Qg = Q * torch.zeros(TQ, D, device=dev).scatter(-1, topg, 1.0)[None]
                    m, e, r = evaluate(Qg @ Kq4.T, exact, lse, Ve, o_ref, ex_top); rec(split, f"sparq_r{r_}_4bit", L, e, r_ * 4 + scale_bits(r_), m, r)
                    # ablation only: independent per-head picks, kernel reads their union (not SparQ's published rule)
                    top = Q.abs().topk(r_, -1).indices; Qm = torch.zeros_like(Q).scatter(-1, top, Q.gather(-1, top)); nun = torch.zeros_like(Q, dtype=torch.bool).scatter(-1, top, True).any(0).float().sum(-1)
                    m, e, r = evaluate(Qm @ Kq4.T, exact, lse, Ve, o_ref, ex_top); rec(split, f"sparq_r{r_}_perhead_union_variant", L, e, (nun * 4 + scale_bits(nun)).mean().item(), m, r)
                # Double Sparsity c=32: offline channels by E|q| E|k| on calibration text
                ch = (Qcal.abs().mean(0) * Kc.abs().mean(0)).argsort(descending=True)[:32]
                m, e, r = evaluate(Q[..., ch] @ Kq4[:, ch].T, exact, lse, Ve, o_ref, ex_top); rec(split, "dsparsity_c32_4bit", L, e, 128 + scale_bits(32), m, r)
                # Loki r=32: KLT of calibration keys, components ordered by calibration score variance, 4-bit coordinates
                Vt = Vt_full[:, :32]
                m, e, r = evaluate((Q @ Vt) @ qblock_dev((Ke - mu) @ Vt, torch.full((32,), 4, device=dev)).T, exact, lse, Ve, o_ref, ex_top); rec(split, "loki_r32_4bit", L, e, 128 + scale_bits(32), m, r)
                # controls for the Loki collapse: fp16 coordinates (no quantisation) and rank 64 at 4 bits
                m, e, r = evaluate((Q @ Vt) @ ((Ke - mu) @ Vt).T, exact, lse, Ve, o_ref, ex_top); rec(split, "loki_r32_fp16", L, e, 32 * 16.0, m, r)
                Vt64 = Vt_full[:, :64]
                m, e, r = evaluate((Q @ Vt64) @ qblock_dev((Ke - mu) @ Vt64, torch.full((64,), 4, device=dev)).T, exact, lse, Ve, o_ref, ex_top); rec(split, "loki_r64_4bit", L, e, 256 + scale_bits(64), m, r)
                # true full 4-bit scan: all 4 planes of all 128 channels
                m, e, r = evaluate(Q @ Kq4.T, exact, lse, Ve, o_ref, ex_top); rec(split, "full_4bit_scan", L, e, 4 * D + scale_bits(D), m, r)
                m, e, r = evaluate(Q @ qblock_dev(Ke, torch.full((D,), 2, device=dev)).T, exact, lse, Ve, o_ref, ex_top); rec(split, "thumb_2bit_all", L, e, 2 * D + scale_bits(D), m, r)
                if args.extras:
                    # FFD-style: 2-bit thumbnail scan of all channels, then re-score the top-delta with the 4-bit K
                    s2 = (Q @ qblock_dev(Ke, torch.full((D,), 2, device=dev)).T).masked_fill(~mask | keep, -1e9)
                    for mult in (2, 4):
                        delta = min(mult * K, T); short = s2.topk(delta, -1).indices
                        s4 = torch.einsum("hqd,hqmd->hqm", Q, Kq4[short]); out = torch.full_like(s2, -1e9).scatter(-1, short, s4)
                        m, e, r = evaluate(out, exact, lse, Ve, o_ref, ex_top)
                        rec(split, f"ffd_2bit_rescore{mult}k", L, e, 2 * D + scale_bits(D) + (delta / n_valid) * (4 * D + scale_bits(D)), m, r)
                    # Self-Indexing-style 1-bit sign index: sign(k - mu) scored with E|k - mu| weights
                    ek = (Kfit - mu).abs().mean(0); S = torch.sign(Ke - mu) * ek
                    m, e, r = evaluate(Q @ S.T, exact, lse, Ve, o_ref, ex_top); rec(split, "sign1bit_128", L, e, float(D), m, r)
                del Kq4
            del Ke, Ve, Q, exact
        del Kc, Qcal_all
    print(f"layer {L} done ({time.time()-t_start:.0f}s)", flush=True)

def layer_mean(split, name): return {L: sum(v) / len(v) for L, v in err[split][name].items()}
def allocate(prefix, mean_target):
    tbl = {B: layer_mean("cal", f"{prefix}/planes{B}") for B in BUDGETS}; idx = {L: 0 for L in LAYERS}
    while sum(BUDGETS[i] for i in idx.values()) / len(LAYERS) < mean_target:
        best, bestL = -1, None
        for L in LAYERS:
            i = idx[L]
            if i + 1 >= len(BUDGETS): continue
            d = (tbl[BUDGETS[i]][L] - tbl[BUDGETS[i + 1]][L]) / (BUDGETS[i + 1] - BUDGETS[i])
            if d > best: best, bestL = d, L
        if bestL is None: break
        idx[bestL] += 1
    return {L: BUDGETS[i] for L, i in idx.items()}
def alloc_row(sname, alloc):
    nm = lambda L: f"{sname}/planes{alloc[L]}"; mean = lambda d: sum(d) / len(d)
    e = sum(layer_mean("test", nm(L))[L] for L in LAYERS) / len(LAYERS)
    b = sum(mean(bits_used[nm(L)]) for L in LAYERS) / len(LAYERS); m = sum(mean(mass_t[nm(L)]) for L in LAYERS) / len(LAYERS); r = sum(mean(rec_t[nm(L)]) for L in LAYERS) / len(LAYERS)
    a = sum(mean(act_t[nm(L)]) for L in LAYERS) / len(LAYERS)
    return {"bits": b, "err": e, "mass": m, "recall": r, "active": a, "alloc": {str(L): alloc[L] for L in LAYERS}}

summary = {"_meta": {"tag": tag, "T": T, "K": K, "TQ": TQ, "G": G, "Hkv": Hkv, "nL": nL, "layers": LAYERS, "extras": args.extras, "alloc_from": args.alloc_from}}
for name in sorted(err["test"]):
    e = layer_mean("test", name)
    summary[name] = {"bits": sum(bits_used[name]) / len(bits_used[name]), "err": sum(e.values()) / len(LAYERS), "mass": sum(mass_t[name]) / len(mass_t[name]),
                     "recall": sum(rec_t[name]) / len(rec_t[name]), "per_layer": {str(L): v for L, v in e.items()}}
    if name in act_t: summary[name]["active"] = sum(act_t[name]) / len(act_t[name])
for sname in STORES:
    for target in (32, 40, 48, 64, 80): summary[f"{sname}/layeralloc_mean{target}"] = alloc_row(sname, allocate(sname, target))
if args.alloc_from:
    other = json.load(open(f"{RES}/final_alloc_{args.alloc_from}.json"))
    for sname in [s_ for s_ in ("u4", "rd128") if s_ in STORES and f"{s_}/layeralloc_mean48" in other]:
        for target in (48, 64):
            al = {int(L): int(b) for L, b in other[f"{sname}/layeralloc_mean{target}"]["alloc"].items() if int(L) in LAYERS}
            summary[f"{sname}/layeralloc_mean{target}@from_{args.alloc_from}"] = alloc_row(sname, al)
print(f"\n{'variant':44s} {'bits':>6s} {'err':>7s} {'mass':>6s} {'recall':>6s}")
for n, d in sorted(((k, v) for k, v in summary.items() if k != "_meta"), key=lambda x: x[1]["bits"]):
    print(f"{n:44s} {d['bits']:6.1f} {d['err']:7.4f} {d['mass']:6.3f} {d['recall']:6.3f}")
out = args.out or f"{RES}/final_alloc_{tag}.json"
json.dump(summary, open(out, "w"), indent=1); print("wrote", out, f"({time.time()-t_start:.0f}s)")
