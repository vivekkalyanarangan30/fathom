"""Key-sketch construction and approximate-score functions shared by the frontier sim and the PPL harness."""
import math, torch
from capsio import load_file   # lazy per-tensor reads (no mmap; 32k caps do not fit in RAM)

BLK = 64
SINK, LOCAL = 4, 32

def qblock(c, bits, blk=BLK):
    """Per-(block of blk tokens, column) absmax mid-rise uniform quantizer; bits [r] ints, 0 drops the column."""
    n, r = c.shape; nb = math.ceil(n / blk); pad = nb * blk - n
    cb = torch.cat([c, c.new_zeros(pad, r)]).view(nb, blk, r)
    amax = cb.abs().amax(1, keepdim=True).clamp_min(1e-8)
    bits = bits.to(c.device); half = 2.0 ** (bits.clamp_min(1) - 1).to(c.dtype)
    s = amax / half
    out = ((cb / s).floor().clamp(-half, half - 1) + 0.5) * s
    out[..., bits <= 0] = 0
    return out.view(-1, r)[:n]

def waterfill(w, budget, bmax=8):
    """Greedy reverse water-filling: integer bits per component minimising sum_j w_j 4^-b_j at total budget."""
    b = torch.zeros(len(w), dtype=torch.long)
    for _ in range(int(budget)):
        gain = w * 4.0 ** (-b.float()); gain[b >= bmax] = -1
        b[gain.argmax()] += 1
    return b

def scale_bits(ncols, blk=BLK):
    """Bits per token for the fp16 block scale of ncols active channels (the timing harness moves exactly this)."""
    return ncols * 16 / blk

def build_params(cal_path, G, budgets=(16, 24, 32, 48, 64, 96, 128, 192)):
    """Per layer / kv-head: score-variance-ordered PCA basis from calibration keys, bit allocations, DS channel order."""
    import os, json
    tr = load_file(cal_path)
    lb_path = cal_path.replace("caps_", "layer_budget_").replace("_train", "").replace(".safetensors", ".json")
    LB = json.load(open(lb_path)) if os.path.exists(lb_path) else None   # calibrated per-layer read budgets (13_layer_budget.py)
    fa_path = cal_path.replace("caps_", "final_alloc_").replace("_train", "").replace(".safetensors", ".json")
    FA = json.load(open(fa_path)) if os.path.exists(fa_path) else None   # per-layer budgets for bit-plane reads (16_final_alloc.py)
    nL = len([k for k in tr if k.startswith("q")]); D = tr["k0"].shape[-1]; Hkv = tr["k0"].shape[0]
    P = []
    for L in range(nL):
        Vts, mus, lams, ws, chs = [], [], [], [], []
        for hk in range(Hkv):
            Kc = tr[f"k{L}"][hk].float(); Kfit = Kc[SINK * 4:]
            mu = Kfit.mean(0); _, S, Vh = torch.linalg.svd(Kfit - mu, full_matrices=False)
            lam = S ** 2 / len(Kfit); Vt = Vh.T
            Qcal = tr[f"q{L}"][hk * G:(hk + 1) * G].reshape(-1, D).float()
            w = ((Qcal @ Vt) ** 2).mean(0) * lam
            order = w.argsort(descending=True)
            Vts.append(Vt[:, order]); mus.append(mu); lams.append(lam[order]); ws.append(w[order])
            chs.append((Qcal.abs().mean(0) * Kc.abs().mean(0)).argsort(descending=True))
        var_raw = torch.stack([tr[f"k{L}"][hk].float()[SINK * 4:].var(0) for hk in range(Hkv)])
        w_raw = torch.stack([(tr[f"q{L}"][hk * G:(hk + 1) * G].reshape(-1, D).float() ** 2).mean(0) for hk in range(Hkv)]) * var_raw
        P.append({"Vt": torch.stack(Vts), "mu": torch.stack(mus), "lam": torch.stack(lams), "w": torch.stack(ws),
                  "ch_order": torch.stack(chs), "var_raw": var_raw,
                  "bits": {B: torch.stack([waterfill(w, B) for w in ws]) for B in budgets},
                  "bits_raw": {B: torch.stack([waterfill(w, B) for w in w_raw]) for B in budgets},
                  "alloc": {m: LB[f"rd/layeralloc_mean{m}"]["alloc"][str(L)] for m in (48, 64, 80)} if LB else None,
                  "alloc_planes": {st: {m: FA[f"{st}/layeralloc_mean{m}"]["alloc"][str(L)] for m in (32, 48, 64, 80)} for st in ("rd128", "u4", "u8") if f"{st}/layeralloc_mean48" in FA} if FA else None})
    return P

def approx_scores(method, q, k, p, hk):
    """Approximate q.k^T for one kv head. q [Hg,Tq,D] (group heads), k [T,D] fp32. Returns scores [Hg,Tq,T] and bits/token."""
    D = k.shape[-1]; name, *args = method.split(":")
    if name == "dense":
        return q @ k.T, D * 16
    if name == "dense4":
        return q @ qblock(k, torch.full((D,), 4)).T, D * 4 + scale_bits(D)
    if name == "loki":            # loki:r:bits
        r, b = int(args[0]), int(args[1]); Vt = p["Vt"][hk][:, :r]
        c = (k - p["mu"][hk]) @ Vt
        cq = c if b >= 16 else qblock(c, torch.full((r,), b))
        return (q @ Vt) @ cq.T, r * b + (scale_bits(r) if b < 16 else 0)
    if name == "ds":              # ds:c:bits  (Double Sparsity label cache)
        c, b = int(args[0]), int(args[1]); ch = p["ch_order"][hk][:c]
        kq = k[:, ch] if b >= 16 else qblock(k[:, ch], torch.full((c,), b))
        return q[..., ch] @ kq.T, c * b + (scale_bits(c) if b < 16 else 0)
    if name == "sparq":           # sparq:r:bits  SparQ under GQA (Ribar et al. Sec. 5): top-r channels of the group's sum |q|, one pick per KV head
        r, b = int(args[0]), int(args[1])
        top = q.abs().sum(0).topk(r, -1).indices; qm = q * torch.zeros(q.shape[1], D, device=q.device).scatter(-1, top, 1.0)[None]
        kq = k if b >= 16 else qblock(k, torch.full((D,), b))
        return qm @ kq.T, r * b + (scale_bits(r) if b < 16 else 0)
    if name == "ours":            # ours:B  fixed read of the water-filled store
        B = int(args[0]); bits = p["bits"][B][hk]; nz = bits > 0
        Vt = p["Vt"][hk][:, nz]; c = qblock((k - p["mu"][hk]) @ Vt, bits[nz])
        return (q @ Vt) @ c.T, B + scale_bits(int(nz.sum()))
    if name == "adapt":           # adapt:Bstore:Bread  per-query component selection from a fixed store
        Bs, Br = int(args[0]), int(args[1]); bits = p["bits"][Bs][hk].float(); nz = bits > 0
        Vt = p["Vt"][hk]; c = qblock((k - p["mu"][hk]) @ Vt, bits.long())
        qp = q @ Vt
        gain = (qp ** 2) * p["lam"][hk] * (1 - 4.0 ** (-bits)) / bits.clamp_min(1)
        gain[..., ~nz] = -1
        srt = gain.argsort(-1, descending=True); cum = bits[srt].cumsum(-1)
        pick = torch.zeros_like(gain, dtype=torch.bool).scatter(-1, srt, cum <= Br) & nz
        return (qp * pick) @ c.T, (pick.float() * bits).sum(-1).mean().item() + scale_bits(int(nz.sum()))
    if name in ("rdchan", "adaptg", "rdchanL"):   # rdchan:Bs:Br raw-channel RD store, group-shared per-query read; adaptg = KLT basis; rdchanL:Bs:mean = per-layer budgets
        Bs, Br = int(args[0]), int(args[1])
        if name == "rdchanL": Br = p["alloc"][Br]
        if name in ("rdchan", "rdchanL"):
            bits = p["bits_raw"][Bs][hk]; lam = p["var_raw"][hk]; kk = k; qp = q
        else:
            bits = p["bits"][Bs][hk]; lam = p["lam"][hk]; kk = (k - p["mu"][hk]) @ p["Vt"][hk]; qp = q @ p["Vt"][hk]
        bf = bits.float().to(q.device); nz = bits.to(q.device) > 0
        store = qblock(kk, bits)
        gain = ((qp ** 2) * lam * (1 - 4.0 ** (-bf))).sum(0, keepdim=True)          # group-summed over the G query heads
        ratio = gain / bf.clamp_min(1); ratio[..., ~nz] = -1
        srt = ratio.argsort(-1, descending=True); cum = bf[srt].cumsum(-1)
        pick = torch.zeros_like(gain, dtype=torch.bool).scatter(-1, srt, cum <= Br).expand_as(qp) & nz
        return (qp * pick) @ store.T, (pick[0].float() * bf).sum(-1).mean().item() + scale_bits(int(nz.sum()))
    if name in ("planes", "planesL"):   # planes:store:Br | planesL:store:mean -- bit-plane store, per-query water-filled read depth per channel
        st, Br = args[0], int(args[1])
        if name == "planesL": Br = p["alloc_planes"][st][Br]
        var = p["var_raw"][hk]
        bits = p["bits_raw"][128][hk] if st == "rd128" else torch.full((D,), 4 if st == "u4" else 8, dtype=torch.long)
        bits = bits.to(q.device); bmax = int(bits.max().item()); nz = bits > 0
        g = ((q ** 2) * var).sum(0)                                             # [Tq, D] group-summed gain
        bf = bits.float(); lo = torch.full((g.shape[0], 1), -60.0, device=q.device); hi = torch.full_like(lo, 60.0); lg = torch.log2(g.clamp_min(1e-30))
        for _ in range(30):
            mid = (lo + hi) / 2; t = ((lg - mid) / 2).round().clamp(min=0).minimum(bf); tot = t.sum(-1, keepdim=True)
            lo = torch.where(tot > Br, mid, lo); hi = torch.where(tot > Br, hi, mid)
        t = ((lg - hi) / 2).round().clamp(min=0).minimum(bf)                    # bits read per (query, channel)
        n, r = k.shape; nb = math.ceil(n / BLK); cb = torch.cat([k, k.new_zeros(nb * BLK - n, r)]).view(nb, BLK, r)
        amax = cb.abs().amax(1, keepdim=True).clamp_min(1e-8)
        out = torch.zeros(q.shape[0], q.shape[1], n, device=q.device)
        for lvl in range(1, bmax + 1):
            m = (t == lvl).float()
            if m.sum() == 0: continue
            half = 2.0 ** (lvl - 1); sc = amax / half
            plane = (((cb / sc).floor().clamp(-half, half - 1) + 0.5) * sc).view(-1, r)[:n]
            out += (q * m[None]) @ plane.T
        return out, (t.sum(-1) + scale_bits((t > 0).float().sum(-1))).mean().item()
    if name == "cascade":         # cascade:B1:mult:B2  coarse scan, shortlist mult*k, refine from richer store
        B1, mult, B2 = int(args[0]), int(args[1]), int(args[2])
        s1, bits1 = approx_scores(f"ours:{B1}", q, k, p, hk)
        return ("cascade", s1, bits1, mult, B2), None
    raise ValueError(method)

def cascade_refine(s1, q, k, p, hk, B2, m, mask):
    """Second stage: re-score the shortlist of m tokens per query using the B2 store. Returns full-shape scores + extra bits/token."""
    bits = p["bits"][B2][hk]; nz = bits > 0
    Vt = p["Vt"][hk][:, nz]; c = qblock((k - p["mu"][hk]) @ Vt, bits[nz]); qp = q @ Vt
    m = min(m, s1.shape[-1]); short = s1.masked_fill(~mask, -1e9).topk(m, -1).indices
    s2 = torch.einsum("hqd,hqmd->hqm", qp, c[short])
    out = torch.full_like(s1, -1e9).scatter(-1, short, s2)
    frac = (m / mask.float().sum(-1)).mean().item()
    return out, frac * (B2 + scale_bits(int(nz.sum())))

def params_to(P, dev):
    mv = lambda v: v.to(dev) if torch.is_tensor(v) else ({b: mv(t) for b, t in v.items()} if isinstance(v, dict) else v)
    return [{k: mv(v) for k, v in p.items()} for p in P]
