"""Bit-Plane Key Scan: store packing + Triton kernels (L4 phase, task B).

Store semantics (per 64-token block, per channel j): 4-bit mid-rise offset-binary code c in [0,16) with a per-block per-channel absmax
scale s: value = (c + 0.5 - 8) * s / 8. Bit p (0 = MSB) of the 64 tokens' codes is one 64-bit word ("plane"). Reading the top t planes gives
c_t = c >> (4 - t) and value_t = (c_t + 0.5 - 2^(t-1)) * s / 2^(t-1) == the t-bit mid-rise quantiser with the same scale (sketch.qblock).

Memory order. The brief's logical layout is planes[nblk][128][4] uint64 (one channel's 4 planes adjacent = one 32 B DRAM sector), which
would make a 1-plane read cost the same DRAM traffic as a 4-plane read. The kernels therefore use a *superblock* order: 16 consecutive blocks
(1024 tokens) are stored as planes_k[nsb][128][4][16] uint64 so each (channel, plane) unit is 128 contiguous bytes (4 full sectors) and
scales_k[nsb][128][16] fp16 (32 B per channel per superblock). Codes, scales and byte counts are unchanged; only the order differs.
The 4-bit channel-major baseline (chan4) gets the same superblock treatment: chan4_k[nsb][128][16][8] uint32 (64 nibbles per block).

Kernels: plan_kernel (per-query bisection -> planes per channel + compacted (channel, plane) pair list), scan_planes_kernel (kernel 1),
scan_chan4_kernel (kernel 2 = Loki / Double Sparsity / SparQ class and the full-dim 4-bit scan at r = 128).
"""
import math, torch, triton, triton.language as tl
import triton.language.extra.libdevice as libdevice

BLK, SUB, D = 64, 16, 128
SBT = BLK * SUB                     # tokens per superblock

# ------------------------------------------------------------------ quantiser + packing (torch, offline) ------------------------------------------------------------------
def quantise_u4(K, scale_dtype=torch.float16):
    """K [..., n, D] fp32 -> codes uint8 [..., n, D] in [0,16) and scales [..., nblk, D] (absmax per block per channel, stored in scale_dtype).
    Uses exactly sketch.qblock's arithmetic: code = floor(x / (amax/8)) + 8 clamped to [0,15]. With fp16 scales the code is computed from the
    fp16-rounded amax so that dequantisation with the stored scale is self-consistent."""
    *lead, n, d = K.shape; nb = math.ceil(n / BLK); pad = nb * BLK - n
    cb = torch.cat([K, K.new_zeros(*lead, pad, d)], -2).reshape(*lead, nb, BLK, d).float()
    amax = cb.abs().amax(-2, keepdim=True).clamp_min(1e-8).to(scale_dtype).float()
    s = amax / 8
    code = ((cb / s).floor().clamp(-8, 7) + 8).to(torch.uint8)
    return code.reshape(*lead, nb * BLK, d)[..., :n, :], amax.squeeze(-2).to(scale_dtype)

def dequant_planes(code, scales, t):
    """Reference dequantisation of the top-t planes (t int tensor [D] or scalar): [..., n, D] fp32."""
    n = code.shape[-2]; nb = math.ceil(n / BLK); pad = nb * BLK - n
    cb = torch.cat([code, code.new_zeros(*code.shape[:-2], pad, code.shape[-1])], -2).reshape(*code.shape[:-2], nb, BLK, -1).to(torch.int64)
    t = torch.as_tensor(t, device=code.device).to(torch.int64); half = (2.0 ** (t.float() - 1))
    ct = (cb >> (4 - t)).float()
    val = (ct + 0.5 - half) * scales.float().unsqueeze(-2) / half
    val = torch.where(t > 0, val, torch.zeros_like(val))
    return val.reshape(*code.shape[:-2], nb * BLK, -1)[..., :n, :]

def pack_planes(code):
    """codes uint8 [..., n, D] -> logical planes int64 [..., nblk, D, 4] (bit i of planes[b, j, p] = bit p (0 = MSB) of token 64b+i's code)."""
    *lead, n, d = code.shape; nb = math.ceil(n / BLK); pad = nb * BLK - n
    cb = torch.cat([code, code.new_zeros(*lead, pad, d)], -2).reshape(*lead, nb, BLK, d).to(torch.int64)
    lane = torch.arange(BLK, device=code.device, dtype=torch.int64).view(*([1] * len(lead)), 1, BLK, 1)
    planes = [(((cb >> (3 - p)) & 1) << lane).sum(-2) for p in range(4)]            # each [..., nb, d]
    return torch.stack(planes, -1)                                                     # [..., nb, d, 4]

def unpack_planes(planes, n):
    """Inverse of pack_planes: int64 [..., nblk, D, 4] -> codes uint8 [..., n, D]."""
    *lead, nb, d, _ = planes.shape
    lane = torch.arange(BLK, device=planes.device, dtype=torch.int64).view(*([1] * len(lead)), 1, BLK, 1, 1)
    bits = (planes.unsqueeze(-3) >> lane) & 1                                          # [..., nb, 64, d, 4]
    code = sum(bits[..., p] << (3 - p) for p in range(4))
    return code.reshape(*lead, nb * BLK, d)[..., :n, :].to(torch.uint8)

def to_superblocks(planes, scales):
    """Logical [..., nblk, D, 4] / [..., nblk, D] -> kernel order [..., nsb, D, 4, SUB] int64 / [..., nsb, D, SUB] fp16 (padded to whole superblocks)."""
    *lead, nb, d, _ = planes.shape; nsb = math.ceil(nb / SUB); pad = nsb * SUB - nb
    pl = torch.cat([planes, planes.new_zeros(*lead, pad, d, 4)], -3).reshape(*lead, nsb, SUB, d, 4).permute(*range(len(lead)), -4, -2, -1, -3).contiguous()
    sc = torch.cat([scales, scales.new_zeros(*lead, pad, d)], -2).reshape(*lead, nsb, SUB, d).transpose(-1, -2).contiguous()
    return pl, sc

def pack_chan4(code):
    """codes uint8 [..., n, D] -> kernel-order channel-major 4-bit store int32 [..., nsb, D, SUB, 8]: word m nibble k = token 8m+k of the block."""
    *lead, n, d = code.shape; nb = math.ceil(n / BLK); nsb = math.ceil(nb / SUB); pad = nsb * SBT - n
    cb = torch.cat([code, code.new_zeros(*lead, pad, d)], -2).reshape(*lead, nsb, SUB, 8, 8, d).to(torch.int64)
    k = torch.arange(8, device=code.device).view(*([1] * (len(lead) + 3)), 8, 1)
    words = (cb << (4 * k)).sum(-2)                                                    # [..., nsb, SUB, 8, d]
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)       # two's complement into int32
    return words.permute(*range(len(lead)), -4, -1, -3, -2).contiguous()               # [..., nsb, d, SUB, 8]

def bytes_planes(t, n):
    """Bytes a scan_planes call moves for plans t [BH, D] (planes per channel) over n tokens: sum_j 8*t_j + 2*#active per 64-token block."""
    nb = math.ceil(n / BLK); return float(((t.float().sum(-1) * 8 + (t > 0).float().sum(-1) * 2) * nb).sum())
def bytes_chan4(nact, n):
    nb = math.ceil(n / BLK); return float((nact.float() * (32 + 2) * nb).sum())

# ------------------------------------------------------------------ plan kernel ------------------------------------------------------------------
@triton.jit
def plan_kernel(g_ptr, budget_ptr, t_ptr, pairs_ptr, npairs_ptr, act_ptr, nact_ptr, cls_ptr, clsn_ptr, BMAX: tl.constexpr, D: tl.constexpr, ITERS: tl.constexpr):
    """One program per (batch, kv-head): t_j = clip(round(log4(g_j / theta)), 0, BMAX), theta bisected (ITERS steps) so sum_j t_j <= budget.
    Writes t int8 [D], the compacted active-channel list, and the compacted (channel*4 + plane) pair list the scan kernel iterates over."""
    bh = tl.program_id(0); ch = tl.arange(0, D)
    g = tl.load(g_ptr + bh * D + ch); Br = tl.load(budget_ptr + bh).to(tl.float32)
    lg = tl.log2(tl.maximum(g, 1e-30)); bmax = BMAX * 1.0
    lo = tl.full([], -60.0, tl.float32); hi = tl.full([], 60.0, tl.float32)
    for _ in range(ITERS):
        mid = (lo + hi) * 0.5
        t = tl.minimum(tl.maximum(libdevice.rint((lg - mid) * 0.5), 0.0), bmax); tot = tl.sum(t, 0)
        lo = tl.where(tot > Br, mid, lo); hi = tl.where(tot > Br, hi, mid)
    t = tl.minimum(tl.maximum(libdevice.rint((lg - hi) * 0.5), 0.0), bmax).to(tl.int32)
    tl.store(t_ptr + bh * D + ch, t.to(tl.int8))
    active = t > 0; ai = active.to(tl.int32)
    pos = tl.cumsum(ai, 0) - ai
    tl.store(act_ptr + bh * D + pos, ch, mask=active); tl.store(nact_ptr + bh, tl.sum(ai, 0))
    off = tl.cumsum(t, 0) - t
    for p in tl.static_range(BMAX):
        tl.store(pairs_ptr + bh * (D * BMAX) + off + p, ch * BMAX + p, mask=p < t)
    tl.store(npairs_ptr + bh, tl.sum(t, 0))
    for c in tl.static_range(1, BMAX + 1):                                   # per-depth channel lists for scan_planes_cls_kernel
        m = t == c; mi = m.to(tl.int32); posc = tl.cumsum(mi, 0) - mi
        tl.store(cls_ptr + (bh * BMAX + c - 1) * D + posc, ch, mask=m); tl.store(clsn_ptr + bh * BMAX + c - 1, tl.sum(mi, 0))

def make_plan(g, budget, bmax=4, iters=30):
    """g [BH, D] fp32 group-summed gain q_j^2 Var(k_j); budget scalar or [BH] (bits per token). Returns dict of device tensors."""
    BH, d = g.shape; dev = g.device
    if not torch.is_tensor(budget): budget = torch.full((BH,), float(budget), device=dev)
    t = torch.empty(BH, d, dtype=torch.int8, device=dev); pairs = torch.zeros(BH, d * bmax, dtype=torch.int32, device=dev)
    npairs = torch.empty(BH, dtype=torch.int32, device=dev); act = torch.zeros(BH, d, dtype=torch.int32, device=dev); nact = torch.empty(BH, dtype=torch.int32, device=dev)
    cls = torch.zeros(BH, bmax, d, dtype=torch.int32, device=dev); clsn = torch.empty(BH, bmax, dtype=torch.int32, device=dev)
    plan_kernel[(BH,)](g.contiguous().float(), budget.float().contiguous(), t, pairs, npairs, act, nact, cls, clsn, BMAX=bmax, D=d, ITERS=iters)
    return {"t": t, "pairs": pairs, "npairs": npairs, "act": act, "nact": nact, "cls": cls, "clsn": clsn}

def make_plan_torch(g, budget, bmax=4, iters=30):
    """Reference: sketch.py's waterfill read (same bisection)."""
    bf = torch.full_like(g, float(bmax)); lo = torch.full((g.shape[0], 1), -60.0, device=g.device); hi = torch.full_like(lo, 60.0); lg = torch.log2(g.clamp_min(1e-30))
    Br = budget if torch.is_tensor(budget) else torch.tensor(float(budget), device=g.device)
    Br = Br.view(-1, 1) if Br.dim() else Br
    for _ in range(iters):
        mid = (lo + hi) / 2; t = ((lg - mid) / 2).round().clamp(min=0).minimum(bf); tot = t.sum(-1, keepdim=True)
        lo = torch.where(tot > Br, mid, lo); hi = torch.where(tot > Br, hi, mid)
    return ((lg - hi) / 2).round().clamp(min=0).minimum(bf)

def channel_list(active):
    """active bool [BH, D] -> compacted act int32 [BH, D], nact int32 [BH] (for scan_chan4)."""
    BH, d = active.shape; nact = active.sum(-1).to(torch.int32)
    order = torch.argsort((~active).to(torch.int8), dim=-1, stable=True)                # active channels first, ascending
    return order.to(torch.int32).contiguous(), nact.contiguous()

# ------------------------------------------------------------------ scan kernels ------------------------------------------------------------------
_cfgs = [triton.Config({"NB": nb, "KK": kk}, num_warps=w, num_stages=st) for nb in (1, 2, 4) for kk in (8, 16) for w in (2, 4, 8) for st in (2, 3)
         if nb * kk * w <= 128]

@triton.jit
def scan_planes_kernel(q_ptr, pairs_ptr, npairs_ptr, act_ptr, nact_ptr, t_ptr, planes_ptr, scales_ptr, out_ptr, n, BH, nsb,
                       G: tl.constexpr, D: tl.constexpr, NB: tl.constexpr, KK: tl.constexpr, SUB: tl.constexpr):
    """scores[bh, g, tok] = sum_j q[bh, g, j] * value_{t_j}(tok, j). One program per (NB blocks of 64 tokens, bh). Iterates over the compacted
    (channel, plane) pair list KK pairs at a time: one gather of KK x NB plane words (as 2 x int32 halves), 64 lane bits extracted with a
    shift+and, every bit used G times (FMA against q_j * scale_j(block) * 2^(3-p) / 8 for the G heads). The constant part of the dequantiser,
    sum_j q_j scale_j (2^-t_j - 1), is added once per active channel (second, shorter loop)."""
    pid = tl.program_id(0); bh = tl.program_id(1)
    blk0 = pid * NB; sb = blk0 // SUB; b0 = blk0 - sb * SUB
    gi = tl.arange(0, G); ki = tl.arange(0, KK); wi = tl.arange(0, NB * 2); lane = tl.arange(0, 32)
    blk_of_w = wi // 2; half_of_w = wi - blk_of_w * 2                                                   # word index -> (block, int32 half)
    acc = tl.zeros([G, NB * 2, 32], dtype=tl.float32)
    pbase = pairs_ptr + bh * (D * 4); plane_base = planes_ptr + (bh * nsb + sb) * (D * 4 * SUB * 2)
    scale_base = scales_ptr + (bh * nsb + sb) * (D * SUB); qb = q_ptr + bh * G * D
    npairs = tl.load(npairs_ptr + bh)
    for i in range(0, npairs, KK):
        pm = i + ki < npairs
        pr = tl.load(pbase + i + ki, mask=pm, other=0); ch = pr // 4; p = pr - ch * 4                          # [KK]
        w = tl.load(plane_base + ((ch * 4 + p)[:, None] * SUB + b0 + blk_of_w[None, :]) * 2 + half_of_w[None, :], mask=pm[:, None], other=0)   # [KK, NB*2] int32
        bits = ((w[:, :, None] >> lane[None, None, :]) & 1).to(tl.float32)                                        # [KK, NB*2, 32]
        sc = tl.load(scale_base + ch[:, None] * SUB + b0 + blk_of_w[None, :], mask=pm[:, None], other=0.0).to(tl.float32)   # [KK, NB*2]
        qv = tl.load(qb + gi[:, None] * D + ch[None, :], mask=pm[None, :], other=0.0)                              # [G, KK]
        f = tl.exp2((3 - p).to(tl.float32)) * 0.125                                                                # [KK]
        wgt = qv[:, :, None] * (sc * f[:, None])[None, :, :]                                                       # [G, KK, NB*2]
        acc += tl.sum(wgt[:, :, :, None] * bits[None, :, :, :], axis=1)
    nact = tl.load(nact_ptr + bh)
    for i in range(0, nact, KK):
        am = i + ki < nact
        ch = tl.load(act_ptr + bh * D + i + ki, mask=am, other=0); t = tl.load(t_ptr + bh * D + ch, mask=am, other=0).to(tl.float32)
        sc = tl.load(scale_base + ch[:, None] * SUB + b0 + blk_of_w[None, :], mask=am[:, None], other=0.0).to(tl.float32)
        qv = tl.load(qb + gi[:, None] * D + ch[None, :], mask=am[None, :], other=0.0)
        cst = tl.sum(qv[:, :, None] * (sc * (tl.exp2(-t) - 1.0)[:, None])[None, :, :], axis=1)                    # [G, NB*2]
        acc += cst[:, :, None]
    tok = (blk0 + blk_of_w)[:, None] * 64 + half_of_w[:, None] * 32 + lane[None, :]                              # [NB*2, 32]
    optr = out_ptr + (bh * G + gi)[:, None, None] * n + tok[None]
    tl.store(optr, acc, mask=(tok < n)[None])

@triton.jit
def scan_chan4_kernel(q_ptr, act_ptr, nact_ptr, chan_ptr, scales_ptr, out_ptr, n, BH, nsb, blk_start,
                      G: tl.constexpr, D: tl.constexpr, NB: tl.constexpr, KK: tl.constexpr, SUB: tl.constexpr):
    """Baseline: channel-major 4-bit K (2 codes per byte), reads the nact listed channels at full 4 bits, KK channels per iteration.
    Bytes = (32 + 2) * nact per block."""
    pid = tl.program_id(0); bh = tl.program_id(1)
    blk0 = blk_start + pid * NB; sb = blk0 // SUB; b0 = blk0 - sb * SUB
    gi = tl.arange(0, G); ki = tl.arange(0, KK); wi = tl.arange(0, NB * 8); nib = tl.arange(0, 8)
    blk_of_w = wi // 8; m_of_w = wi - blk_of_w * 8
    acc = tl.zeros([G, NB * 8, 8], dtype=tl.float32)
    chan_base = chan_ptr + (bh * nsb + sb) * (D * SUB * 8); scale_base = scales_ptr + (bh * nsb + sb) * (D * SUB); qb = q_ptr + bh * G * D
    nact = tl.load(nact_ptr + bh)
    for i in range(0, nact, KK):
        am = i + ki < nact
        ch = tl.load(act_ptr + bh * D + i + ki, mask=am, other=0)
        w = tl.load(chan_base + (ch[:, None] * SUB + b0 + blk_of_w[None, :]) * 8 + m_of_w[None, :], mask=am[:, None], other=0)   # [KK, NB*8] words
        codes = ((w[:, :, None] >> (4 * nib)[None, None, :]) & 15).to(tl.float32) - 7.5                             # [KK, NB*8, 8]
        sc = tl.load(scale_base + ch[:, None] * SUB + b0 + blk_of_w[None, :], mask=am[:, None], other=0.0).to(tl.float32) * 0.125
        qv = tl.load(qb + gi[:, None] * D + ch[None, :], mask=am[None, :], other=0.0)                                # [G, KK]
        wgt = qv[:, :, None] * sc[None, :, :]                                                                        # [G, KK, NB*8]
        acc += tl.sum(wgt[:, :, :, None] * codes[None, :, :, :], axis=1)
    tok = (blk0 + blk_of_w)[:, None] * 64 + m_of_w[:, None] * 8 + nib[None, :]
    optr = out_ptr + (bh * G + gi)[:, None, None] * n + tok[None]
    tl.store(optr, acc, mask=(tok < n)[None])

_cfgs_dot = [triton.Config({"NB": nb, "KK": kk}, num_warps=w, num_stages=st) for nb in (1, 2, 4) for kk in (16, 32) for w in (2, 4, 8) for st in (2, 3)
             if nb * kk * w <= 256]

@triton.autotune(configs=_cfgs_dot, key=["n", "BH"])
@triton.jit
def scan_planes_dot_kernel(q_ptr, pairs_ptr, npairs_ptr, act_ptr, nact_ptr, t_ptr, planes_ptr, scales_ptr, out_ptr, n, BH, nsb,
                           G: tl.constexpr, D: tl.constexpr, NB: tl.constexpr, KK: tl.constexpr, SUB: tl.constexpr):
    """Tensor-core variant of scan_planes_kernel. Per chunk of KK (channel, plane) pairs: B[pair, tok] = bit ? scale_j(block) : 0 (fp16, exact),
    A[g, pair] = q_gj * 2^(3-p) / 8 (fp16), scores += A @ B on tensor cores (fp32 accumulate). The CUDA cores only do the bit extraction
    (shift + sign test + select). G is padded to 16 rows (tl.dot minimum)."""
    pid = tl.program_id(0); bh = tl.program_id(1)
    blk0 = pid * NB; sb = blk0 // SUB; b0 = blk0 - sb * SUB
    gi = tl.arange(0, 16); ki = tl.arange(0, KK); wi = tl.arange(0, NB * 2); lane = tl.arange(0, 32)
    blk_of_w = wi // 2; half_of_w = wi - blk_of_w * 2; sh = (31 - lane)
    acc = tl.zeros([16, NB * 64], dtype=tl.float32); cst = tl.zeros([16, NB * 2], dtype=tl.float32)
    pbase = pairs_ptr + bh * (D * 4); plane_base = planes_ptr + (bh * nsb + sb) * (D * 4 * SUB * 2)
    scale_base = scales_ptr + (bh * nsb + sb) * (D * SUB); qb = q_ptr + bh * G * D; gm = gi < G
    npairs = tl.load(npairs_ptr + bh)
    for i in range(0, npairs, KK):
        pm = i + ki < npairs
        pr = tl.load(pbase + i + ki, mask=pm, other=0); ch = pr // 4; p = pr - ch * 4
        w = tl.load(plane_base + ((ch * 4 + p)[:, None] * SUB + b0 + blk_of_w[None, :]) * 2 + half_of_w[None, :], mask=pm[:, None], other=0)   # [KK, NB*2]
        sc = tl.load(scale_base + ch[:, None] * SUB + b0 + blk_of_w[None, :], mask=pm[:, None], other=0.0)                                     # [KK, NB*2] fp16
        bit = (w[:, :, None] << sh[None, None, :]) < 0                                                                                        # [KK, NB*2, 32]
        Bm = tl.where(bit, sc[:, :, None], tl.zeros_like(sc)[:, :, None])
        B2 = tl.reshape(Bm, [KK, NB * 64])
        qv = tl.load(qb + gi[:, None] * D + ch[None, :], mask=gm[:, None] & pm[None, :], other=0.0)                                         # [16, KK]
        A = (qv * (tl.exp2((3 - p).to(tl.float32)) * 0.125)[None, :]).to(tl.float16)
        acc = tl.dot(A, B2, acc)
    nact = tl.load(nact_ptr + bh)
    for i in range(0, nact, KK):
        am = i + ki < nact
        ch = tl.load(act_ptr + bh * D + i + ki, mask=am, other=0); t = tl.load(t_ptr + bh * D + ch, mask=am, other=0).to(tl.float32)
        sc = tl.load(scale_base + ch[:, None] * SUB + b0 + blk_of_w[None, :], mask=am[:, None], other=0.0).to(tl.float32)
        qv = tl.load(qb + gi[:, None] * D + ch[None, :], mask=gm[:, None] & am[None, :], other=0.0)
        cst += tl.sum(qv[:, :, None] * (sc * (tl.exp2(-t) - 1.0)[:, None])[None, :, :], axis=1)
    acc3 = tl.reshape(acc, [16, NB * 2, 32]) + cst[:, :, None]
    tok = (blk0 + blk_of_w)[:, None] * 64 + half_of_w[:, None] * 32 + lane[None, :]
    optr = out_ptr + (bh * G + gi)[:, None, None] * n + tok[None]
    tl.store(optr, acc3, mask=gm[:, None, None] & (tok < n)[None])

@triton.jit
def scan_planes_bdot_kernel(q_ptr, pairs_ptr, npairs_ptr, act_ptr, nact_ptr, t_ptr, planes_ptr, scales_ptr, out_ptr, n, BH, nsb,
                            G: tl.constexpr, D: tl.constexpr, NB: tl.constexpr, KK: tl.constexpr, SUB: tl.constexpr):
    """Batched tensor-core variant: batch = the NB*2 int32 half-words (32 tokens each), A_b[g, pair] = q_gj 2^(3-p)/8 * scale_j(block b) (fp16),
    B_b[pair, lane] = bit (fp16 0/1); acc[b, g, lane] += A_b @ B_b. Words are loaded directly as [NB*2, KK], so no reshapes or transposes."""
    pid = tl.program_id(0); bh = tl.program_id(1)
    blk0 = pid * NB; sb = blk0 // SUB; b0 = blk0 - sb * SUB
    gi = tl.arange(0, 16); ki = tl.arange(0, KK); wi = tl.arange(0, NB * 2); lane = tl.arange(0, 32)
    blk_of_w = wi // 2; half_of_w = wi - blk_of_w * 2; sh = 31 - lane
    acc = tl.zeros([NB * 2, 16, 32], dtype=tl.float32); cst = tl.zeros([NB * 2, 16], dtype=tl.float32)
    pbase = pairs_ptr + bh * (D * 4); plane_base = planes_ptr + (bh * nsb + sb) * (D * 4 * SUB * 2)
    scale_base = scales_ptr + (bh * nsb + sb) * (D * SUB); qb = q_ptr + bh * G * D; gm = gi < G
    npairs = tl.load(npairs_ptr + bh)
    for i in range(0, npairs, KK):
        pm = i + ki < npairs
        pr = tl.load(pbase + i + ki, mask=pm, other=0); ch = pr // 4; p = pr - ch * 4
        w = tl.load(plane_base + ((ch * 4 + p)[None, :] * SUB + b0 + blk_of_w[:, None]) * 2 + half_of_w[:, None], mask=pm[None, :], other=0)   # [NB*2, KK]
        sc = tl.load(scale_base + ch[None, :] * SUB + b0 + blk_of_w[:, None], mask=pm[None, :], other=0.0).to(tl.float32)                     # [NB*2, KK]
        B = tl.where((w[:, :, None] << sh[None, None, :]) < 0, 1.0, 0.0).to(tl.float16)                                                       # [NB*2, KK, 32]
        qv = tl.load(qb + gi[:, None] * D + ch[None, :], mask=gm[:, None] & pm[None, :], other=0.0)                                          # [16, KK]
        A = (qv[None, :, :] * (sc * (tl.exp2((3 - p).to(tl.float32)) * 0.125)[None, :])[:, None, :]).to(tl.float16)                         # [NB*2, 16, KK]
        acc = tl.dot(A, B, acc)
    nact = tl.load(nact_ptr + bh)
    for i in range(0, nact, KK):
        am = i + ki < nact
        ch = tl.load(act_ptr + bh * D + i + ki, mask=am, other=0); t = tl.load(t_ptr + bh * D + ch, mask=am, other=0).to(tl.float32)
        sc = tl.load(scale_base + ch[None, :] * SUB + b0 + blk_of_w[:, None], mask=am[None, :], other=0.0).to(tl.float32)                    # [NB*2, KK]
        qv = tl.load(qb + gi[:, None] * D + ch[None, :], mask=gm[:, None] & am[None, :], other=0.0)                                          # [16, KK]
        cst += tl.sum(qv[None, :, :] * (sc * (tl.exp2(-t) - 1.0)[None, :])[:, None, :], axis=2)                                             # [NB*2, 16]
    acc = acc + cst[:, :, None]
    tok = (blk0 + blk_of_w)[:, None] * 64 + half_of_w[:, None] * 32 + lane[None, :]                                                           # [NB*2, 32]
    optr = out_ptr + (bh * G + gi)[None, :, None] * n + tok[:, None, :]
    tl.store(optr, acc, mask=gm[None, :, None] & (tok < n)[:, None, :])

@triton.jit
def scan_planes_cls_kernel(q_ptr, cls_ptr, clsn_ptr, planes_ptr, scales_ptr, out_ptr, n, BH, nsb, blk_start,
                           G: tl.constexpr, D: tl.constexpr, NB: tl.constexpr, KK: tl.constexpr, SUB: tl.constexpr):
    """Depth-class variant of scan_planes: channels are grouped by their plane count c = 1..4 (lists from plan_kernel). For a class-c chunk the
    c plane words are loaded and the c-bit code is assembled in the integer domain (code = 2*code + bit), converted once, and weighted once per
    head: value_c = code * scale/2^(c-1) + scale*(2^-c*... ) -> acc += q_g * scale/2^(c-1) * code; the constant part goes to cst.
    ALU work per (channel, token) ~ 3c + 9 vs 11c for the per-bit kernel."""
    pid = tl.program_id(0); bh = tl.program_id(1)
    blk0 = blk_start + pid * NB; sb = blk0 // SUB; b0 = blk0 - sb * SUB
    gi = tl.arange(0, G); ki = tl.arange(0, KK); wi = tl.arange(0, NB * 2); lane = tl.arange(0, 32)
    blk_of_w = wi // 2; half_of_w = wi - blk_of_w * 2
    acc = tl.zeros([G, NB * 2, 32], dtype=tl.float32); cst = tl.zeros([G, NB * 2], dtype=tl.float32)
    plane_base = planes_ptr + (bh * nsb + sb) * (D * 4 * SUB * 2); scale_base = scales_ptr + (bh * nsb + sb) * (D * SUB); qb = q_ptr + bh * G * D
    for c in tl.static_range(1, 5):
        nc = tl.load(clsn_ptr + bh * 4 + c - 1); lbase = cls_ptr + (bh * 4 + c - 1) * D
        inv_half = 1.0 / (2.0 ** (c - 1)); cadd = 0.5 / (2.0 ** (c - 1)) - 1.0
        for i in range(0, nc, KK):
            am = i + ki < nc
            ch = tl.load(lbase + i + ki, mask=am, other=0)
            sc = tl.load(scale_base + ch[:, None] * SUB + b0 + blk_of_w[None, :], mask=am[:, None], other=0.0).to(tl.float32)        # [KK, NB*2]
            qv = tl.load(qb + gi[:, None] * D + ch[None, :], mask=am[None, :], other=0.0)                                            # [G, KK]
            code = tl.zeros([KK, NB * 2, 32], dtype=tl.int32)
            for p in tl.static_range(c):
                w = tl.load(plane_base + ((ch * 4 + p)[:, None] * SUB + b0 + blk_of_w[None, :]) * 2 + half_of_w[None, :], mask=am[:, None], other=0)
                code = code * 2 + ((w[:, :, None] >> lane[None, None, :]) & 1)
            codef = (code | 0x4B000000).to(tl.float32, bitcast=True) - 8388608.0                                                      # exact int -> float without I2F
            wgt = qv[:, :, None] * (sc * inv_half)[None, :, :]                                                                     # [G, KK, NB*2]
            acc += tl.sum(wgt[:, :, :, None] * codef[None, :, :, :], axis=1)
            cst += tl.sum(qv[:, :, None] * (sc * cadd)[None, :, :], axis=1)
    acc = acc + cst[:, :, None]
    tok = (blk0 + blk_of_w)[:, None] * 64 + half_of_w[:, None] * 32 + lane[None, :]
    optr = out_ptr + (bh * G + gi)[:, None, None] * n + tok[None]
    tl.store(optr, acc, mask=(tok < n)[None])

# ------------------------------------------------------------------ host wrappers ------------------------------------------------------------------
class PlaneStore:
    """K [BH, n, D] (any float dtype) -> superblock-ordered bit-plane store on the same device."""
    def __init__(self, K, scale_dtype=torch.float16):
        self.BH, self.n, d = K.shape; assert d == D
        code, sc = quantise_u4(K.float(), scale_dtype); self.code, self.scales_blk = code, sc
        self.planes_k, self.scales_k = to_superblocks(pack_planes(code), sc.to(torch.float16))    # [BH, nsb, D, 4, SUB] int64, [BH, nsb, D, SUB] fp16
        self.nsb = self.planes_k.shape[1]; self.planes32 = self.planes_k.view(torch.int32)         # [BH, nsb, D, 4, SUB, 2]
    kernel = "cls"      # "cls" (depth-class FMA, default, exact) | "fma" (per-bit FMA, exact) | "dot" (tensor-core, fp16 weights); set PlaneStore.kernel
    fma_cfg = {"NB": 2, "KK": 8, "num_warps": 1, "num_stages": 1}
    def scan(self, q, plan, out=None, blocks=None):
        """q [BH, G, D] fp32, plan from make_plan -> scores fp32 [BH, G, n]; blocks=(b0, b1) scans only 64-token blocks [b0, b1) (cls kernel)."""
        BH, G, d = q.shape; assert BH == self.BH
        out = torch.empty(BH, G, self.n, device=q.device, dtype=torch.float32) if out is None else out
        cfg = PlaneStore.cls_cfg if PlaneStore.kernel == "cls" else PlaneStore.fma_cfg
        grid = (lambda meta: (triton.cdiv(self.n, meta["NB"] * BLK), BH)) if PlaneStore.kernel == "dot" else (triton.cdiv(self.n, cfg["NB"] * BLK), BH)
        if PlaneStore.kernel == "cls":
            if blocks is not None:
                b0, b1 = blocks; grid = (triton.cdiv(b1 - b0, cfg["NB"]), BH)
                scan_planes_cls_kernel[grid](q.contiguous(), plan["cls"], plan["clsn"], self.planes32, self.scales_k, out, self.n, BH, self.nsb, b0, G=G, D=d, SUB=SUB, **PlaneStore.cls_cfg)
                return out
            return self._scan_cls(q, plan, out, grid, G, d)
        if PlaneStore.kernel == "dot":
            scan_planes_dot_kernel[grid](q.contiguous(), plan["pairs"], plan["npairs"], plan["act"], plan["nact"], plan["t"], self.planes32, self.scales_k, out, self.n, BH, self.nsb, G=G, D=d, SUB=SUB)
        else:
            scan_planes_kernel[grid](q.contiguous(), plan["pairs"], plan["npairs"], plan["act"], plan["nact"], plan["t"], self.planes32, self.scales_k, out, self.n, BH, self.nsb, G=G, D=d, SUB=SUB, **PlaneStore.fma_cfg)
        return out
    def _scan_cls(self, q, plan, out, grid, G, d):
        scan_planes_cls_kernel[grid](q.contiguous(), plan["cls"], plan["clsn"], self.planes32, self.scales_k, out, self.n, q.shape[0], self.nsb, 0, G=G, D=d, SUB=SUB, **PlaneStore.cls_cfg)
        return out
    cls_cfg = {"NB": 4, "KK": 4, "num_warps": 1, "num_stages": 1}
    def nbytes(self): return self.planes_k.numel() * 8 + self.scales_k.numel() * 2

class Chan4Store:
    def __init__(self, K, scale_dtype=torch.float16):
        self.BH, self.n, d = K.shape
        code, sc = quantise_u4(K.float(), scale_dtype); self.code, self.scales_blk = code, sc
        self.chan_k = pack_chan4(code); _, self.scales_k = to_superblocks(pack_planes(code[:, :BLK]).new_zeros(self.BH, math.ceil(self.n / BLK), D, 4), sc.to(torch.float16))
        self.nsb = self.chan_k.shape[1]
    def scan(self, q, act, nact, out=None, blocks=None):
        BH, G, d = q.shape
        out = torch.empty(BH, G, self.n, device=q.device, dtype=torch.float32) if out is None else out
        b0, b1 = (0, math.ceil(self.n / BLK)) if blocks is None else blocks
        grid = (triton.cdiv(b1 - b0, Chan4Store.cfg["NB"]), BH)
        scan_chan4_kernel[grid](q.contiguous(), act, nact, self.chan_k, self.scales_k, out, self.n, BH, self.nsb, b0, G=G, D=d, SUB=SUB, **Chan4Store.cfg)
        return out
    cfg = {"NB": 2, "KK": 1, "num_warps": 1, "num_stages": 1}

def scan_planes_reference(q, code, scales, t):
    """Torch reference of scan_planes semantics: q [BH,G,D], code uint8 [BH,n,D], scales [BH,nblk,D], t [BH,D] int -> [BH,G,n]."""
    out = []
    for b in range(q.shape[0]):
        val = dequant_planes(code[b], scales[b], t[b].long())                     # [n, D]
        out.append(q[b].float() @ val.T)
    return torch.stack(out)

def scan_chan4_reference(q, code, scales, active):
    out = []
    for b in range(q.shape[0]):
        val = dequant_planes(code[b], scales[b], torch.full((D,), 4, device=q.device)) * active[b].float()[None]
        out.append(q[b].float() @ val.T)
    return torch.stack(out)

def sink_local_topk(scores, k, sink=4, local=32):
    """scores [BH, G, n] -> selected indices [BH, G, k] = top-(k - sink - local) among non-sink/non-local tokens + sinks + last `local` tokens."""
    BH, G, n = scores.shape; kk = k - sink - local
    s = scores.clone(); s[..., :sink] = -float("inf"); s[..., n - local:] = -float("inf")
    top = s.topk(kk, -1).indices
    fixed = torch.cat([torch.arange(sink, device=scores.device), torch.arange(n - local, n, device=scores.device)])
    return torch.cat([top, fixed.expand(BH, G, -1)], -1)
