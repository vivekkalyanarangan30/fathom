"""End-to-end decode with the KV cache OFFLOADED to host memory (the regime of vLLM/SGLang CPU offload, HF OffloadedCache, InfiniGen, FlexGen).
After a dense GPU prefill, K/V caches and the scan stores are moved to pinned, GPU-mapped host memory; the GPU keeps only the weights and the
last STEPS generated tokens. Per decode step and layer:
  dense_offload      : prefetch the whole K/V of the layer host->device (cudaMemcpyAsync from pinned memory, what offload engines do), then sdpa;
  chan4_r32 / chan4_r16 (SparQ r=32 / r=16 under its GQA rule; r=32 = Loki / DS bytes): zero-copy scan of the channel-major 4-bit store in host memory, top-k, zero-copy gather of the k selected bf16 K/V rows;
  planes_mean48/64   : zero-copy scan of the bit-plane store in host memory, top-k, same row gather.
  planes_thumb2      : uniform 2-plane read of every channel (FFD-class 2-bit thumbnail scan) over the same bit-plane store.
  landmark8          : Quest / ShadowKV-style block selection: fp16 block-mean keys (block 8, 32 B/token), top blocks' rows fetched (our reimplementation).
--index-hbm keeps every method's scan store in GPU memory and offloads only the bf16 rows (the fair setting whenever the index fits).
Reports wall ms/step, GPU-kernel+memcpy ms/step (profiler, launch-free), PCIe bytes per step. Usage: python scripts/B_e2e_offload.py --ctxs 16384,32768 --batches 1,2 --w4"""
import sys, os, json, time, math, ctypes, glob, argparse, torch, triton, triton.language as tl
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
from capsio import safe_open
from common import *
from sketch import SINK, LOCAL
from bitplane import *
import bitplane as bp
import transformers.models.qwen3.modeling_qwen3 as mq
torch.set_grad_enabled(False)
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen3-8B"); ap.add_argument("--tag", default="Qwen3-8B_T16384")
ap.add_argument("--ctxs", default="16384,32768"); ap.add_argument("--batches", default="1,2"); ap.add_argument("--k", type=int, default=512); ap.add_argument("--steps", type=int, default=12)
ap.add_argument("--w4", action="store_true"); ap.add_argument("--out", default=None); ap.add_argument("--methods", default="dense_offload,chan4_r32,chan4_r16,planes_mean48,planes_mean64")
ap.add_argument("--synthetic-kv", action="store_true", help="skip the prefill: fill the host KV with calibration keys tiled to ctx and random values (timing only; needed beyond the model's prefill reach, e.g. 1M tokens)")
ap.add_argument("--copy-index", action="store_true", help="planes: whole-sequence channel-major host store; per query, cudaMemcpyAsync the first t_j planes of each active channel (one contiguous run) into GPU staging, then scan in HBM")
ap.add_argument("--index-hbm", action="store_true", help="keep the scan stores in GPU memory (only the bf16 K/V rows are offloaded): the fair comparison when the index fits HBM")
ap.add_argument("--pipeline", type=int, default=0, help="copy-index mode: split the sequence into this many block chunks and overlap the host->GPU gather of chunk i+1 with the scan of chunk i on a second stream (all scan methods and the landmark copy)"); ap.add_argument("--prefill-chunk", type=int, default=2048); ap.add_argument("--profile-steps", type=int, default=1); ap.add_argument("--trace-dir", default=None); ap.add_argument("--host-gb", type=float, default=12.0, help="pinned host memory budget")
args = ap.parse_args(); dev = torch.device("cuda"); PIPE = args.pipeline; PROF = os.environ.get("PLANES_PROFILE") == "1"; CHECK = os.environ.get("PLANES_CHECK") == "1"; K = args.k; STEPS = args.steps
CTXS = [int(x) for x in args.ctxs.split(",")]; BATCHES = [int(x) for x in args.batches.split(",")]; METHODS = args.methods.split(",")
rt = ctypes.CDLL((glob.glob(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "cuda_runtime", "lib", "libcudart.so*")) or glob.glob("/usr/local/cuda/lib64/libcudart.so*"))[0])
class DevPtr:
    def __init__(self, ptr, dtype): self.ptr, self.dtype = ptr, dtype
    def data_ptr(self): return self.ptr
def host_mapped(t):
    """Copy tensor t into pinned, GPU-mapped host memory. Returns (cpu view tensor, DevPtr for kernels, host ptr)."""
    nb = t.numel() * t.element_size(); hp = ctypes.c_void_p(); r = rt.cudaHostAlloc(ctypes.byref(hp), ctypes.c_size_t(nb), 3)
    if r != 0: raise MemoryError(f"cudaHostAlloc({nb/1e9:.1f} GB) failed: {r}")
    dp = ctypes.c_void_p(); assert rt.cudaHostGetDevicePointer(ctypes.byref(dp), hp, 0) == 0
    view = torch.frombuffer((ctypes.c_byte * nb).from_address(hp.value), dtype=t.dtype).view(t.shape); view.copy_(t.cpu()); return view, DevPtr(dp.value, t.dtype), hp.value
def free_host(hp): rt.cudaFreeHost(ctypes.c_void_p(hp))

def build_store(kind, K, chunk_tokens=65536):
    """PlaneStore / Chan4Store over K [BH, n, D] built in token chunks (the int64 packing transients of a 1M-token build exceed 80 GB otherwise)."""
    BH, n, _ = K.shape; nblk = math.ceil(n / BLK); whole = bp.SUB == nblk                    # copy-index mode: one superblock = the whole sequence
    st = (PlaneStore if kind == "planes" else Chan4Store).__new__(PlaneStore if kind == "planes" else Chan4Store); st.BH, st.n = BH, n
    parts, scs = [], []
    for c0 in range(0, n, chunk_tokens):
        Kc = K[:, c0:c0 + chunk_tokens]; nbc = math.ceil(Kc.shape[1] / BLK)
        if whole: bp.SUB, bp.SBT = nbc, BLK * nbc
        code, sc = quantise_u4(Kc.float()); sc = sc.to(torch.float16)
        if kind == "planes": pk, sk = to_superblocks(pack_planes(code), sc)                    # [BH, nsb_c, D, 4, SUB] / [BH, nsb_c, D, SUB]
        else: pk = pack_chan4(code); _, sk = to_superblocks(pack_planes(code[:, :BLK]).new_zeros(BH, nbc, D, 4), sc)
        parts.append(pk); scs.append(sk); del code, sc
    if whole:
        bp.SUB, bp.SBT = nblk, BLK * nblk
        pk = torch.cat(parts, -1 if kind == "planes" else -2); sk = torch.cat(scs, -1)         # concatenate along the block axis inside the single superblock
    else: pk = torch.cat(parts, 1); sk = torch.cat(scs, 1)
    st.scales_k = sk.contiguous(); st.nsb = pk.shape[1]
    if kind == "planes": st.planes_k = pk.contiguous(); st.planes32 = st.planes_k.view(torch.int32)
    else: st.chan_k = pk.contiguous()
    return st

@triton.jit
def gather_runs_kernel(src_ptr, dst_ptr, off_ptr, len_ptr, nrun_ptr, CH: tl.constexpr):
    """Copy run r (int32 words [off[r], off[r]+len[r])) from the mapped host store into the device staging at the same offsets; wide coalesced loads."""
    r = tl.program_id(0); c0 = tl.program_id(1) * CH
    if r < tl.load(nrun_ptr):
        off = tl.load(off_ptr + r).to(tl.int64); ln = tl.load(len_ptr + r); i = c0 + tl.arange(0, CH); m = i < ln
        tl.store(dst_ptr + off + i, tl.load(src_ptr + off + i, mask=m, other=0), mask=m)
def gather_runs(src, dst, off, ln, nrun, max_len, CH=4096):
    """off/ln int32 [R] device tensors (words), nrun int32 [1] device; no host sync."""
    gather_runs_kernel[(off.shape[0], triton.cdiv(max_len, CH))](src, dst, off, ln, nrun, CH=CH)

@triton.jit
def gather_rows_kernel(src_ptr, idx_ptr, dst_ptr, cap, R, D: tl.constexpr, RB: tl.constexpr):
    """dst[bh, r, :] = src[bh, idx[bh, r], :] for bf16 rows of D; src may live in mapped host memory (zero-copy over PCIe)."""
    bh = tl.program_id(1); r0 = tl.program_id(0) * RB; ri = r0 + tl.arange(0, RB); di = tl.arange(0, D)
    idx = tl.load(idx_ptr + bh * R + ri, mask=ri < R, other=0)
    ok = (ri < R) & (idx >= 0)
    rows = tl.load(src_ptr + (bh * cap + idx)[:, None] * D + di[None, :], mask=ok[:, None], other=0.0)
    tl.store(dst_ptr + (bh * R + ri)[:, None] * D + di[None, :], rows, mask=(ri < R)[:, None])
def gather_rows(src, cap, idx, BH, dtype=torch.bfloat16):
    R = idx.shape[1]; dst = torch.empty(BH, R, D, device=dev, dtype=dtype)
    gather_rows_kernel[(triton.cdiv(R, 64), BH)](src, idx.contiguous(), dst, cap, R, D=D, RB=64); return dst

tok = AutoTokenizer.from_pretrained(args.model)
if args.w4:
    from transformers import BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16), dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()
else: model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()
cfg = model.config; Hq, Hkv, Dh = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim; G = Hq // Hkv; nL = cfg.num_hidden_layers
weight_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9; print(f"{args.model} w4={args.w4}: weights {weight_gb:.1f} GB", flush=True)
base, Ttag = args.tag.split("_T")[0], args.tag.split("_")[-1]
ftr = safe_open(f"{RES}/caps_{base}_train_{Ttag}.safetensors"); FA = json.load(open(f"{RES}/final_alloc_{args.tag}.json"))
VAR = torch.stack([ftr.get_slice(f"k{L}")[:].float()[:, SINK * 4:].var(1) for L in range(nL)]).to(dev); BUD = {m: [FA[f"u4/layeralloc_mean{m}"]["alloc"][str(L)] for L in range(nL)] for m in (40, 48, 64) if f"u4/layeralloc_mean{m}" in FA}
ST = {"mode": "prefill", "method": "dense_offload", "cache": None, "B": 1, "n": 0, "events": [], "bytes": 0.0, "scan_bytes": 0.0, "row_bytes": 0.0}
ARANGE_BHD = torch.arange(64 * 128, dtype=torch.int32, device=dev); NRUN_ALL = torch.zeros(1, dtype=torch.int32, device=dev)
def _bytes_planes_t(t, n):                                                       # tensor-valued (no host sync); reduced once per step
    nb = math.ceil(n / BLK); return ((t.float().sum(-1) * 8 + (t > 0).float().sum(-1) * 2) * nb).sum()
def _bytes_chan4_t(nact, n):
    nb = math.ceil(n / BLK); return (nact.float() * (32 + 2) * nb).sum()
bytes_planes, bytes_chan4 = _bytes_planes_t, _bytes_chan4_t

S_COPY, S_SCAN = torch.cuda.Stream(), torch.cuda.Stream()
STATIC = {}
def chunk_bounds(nblk, parts, align=64):
    """Block boundaries [0 = b_0 < ... < b_parts = nblk], interior boundaries multiples of align (scale runs are fp16, 2 per int32 word)."""
    bs = [0] + [min(nblk, ((nblk * i // parts) // align) * align) for i in range(1, parts)] + [nblk]
    return [(a, b) for a, b in zip(bs[:-1], bs[1:]) if b > a]
def static_offsets(BH, nblk, key):
    """Per-slot word offsets into the copy-index stores: planes slot (bh, ch, plane) run of nblk*2 words; scales slot (bh, ch) run of nblk/2 words; chan4 slot (bh, ch) run of nblk*8 words."""
    k = (BH, nblk, key)
    if k not in STATIC:
        STATIC[k] = {"pl": torch.arange(BH * D * 4, dtype=torch.int32, device=dev) * (nblk * 2), "sc": torch.arange(BH * D, dtype=torch.int32, device=dev) * (nblk // 2),
                     "ch": torch.arange(BH * D, dtype=torch.int32, device=dev) * (nblk * 8), "n_pl": torch.full((1,), BH * D * 4, dtype=torch.int32, device=dev),
                     "n_ch": torch.full((1,), BH * D, dtype=torch.int32, device=dev), "p4": torch.arange(4, dtype=torch.int8, device=dev)}
    return STATIC[k]

def stream_planes(st, qf, plan, n, BH):
    """Chunked gather of the first t planes of every active channel (plus scales) into HBM staging, overlapped with the scan of the previous chunk."""
    nblk = st.planes_k.shape[-1]; so = static_offsets(BH, nblk, "p"); cur = torch.cuda.current_stream()
    on_p = (so["p4"][None, None, :] < plan["t"][:, :, None]).to(torch.int32).flatten()                     # [BH*D*4] plane (bh, ch, p) is read
    on_s = (plan["t"] > 0).to(torch.int32).flatten()                                                        # [BH*D] channel active
    scores = torch.empty(BH, qf.shape[1], n, device=dev, dtype=torch.float32)
    ev0 = torch.cuda.Event(); ev0.record(cur); S_COPY.wait_event(ev0); S_SCAN.wait_event(ev0)
    for (b0, b1) in chunk_bounds(nblk, PIPE):
        with torch.cuda.stream(S_COPY):
            gather_runs(st.planes_hostdev, st.planes32.view(-1), so["pl"] + b0 * 2, on_p * ((b1 - b0) * 2), so["n_pl"], (b1 - b0) * 2)
            gather_runs(st.scales_hostdev, st.scales_k.view(torch.int32).view(-1), so["sc"] + b0 // 2, on_s * ((b1 - b0) // 2), so["n_ch"], (b1 - b0) // 2)
            ev = torch.cuda.Event(); ev.record(S_COPY)
        with torch.cuda.stream(S_SCAN):
            S_SCAN.wait_event(ev); st.scan(qf, plan, out=scores, blocks=(b0, b1))
    ev1 = torch.cuda.Event(); ev1.record(S_SCAN); cur.wait_event(ev1)
    return scores

def stream_chan4(st, qf, a, act, nact, n, BH):
    nblk = st.chan_k.shape[-2]; so = static_offsets(BH, nblk, "c"); cur = torch.cuda.current_stream()
    on = a.to(torch.int32).flatten()
    scores = torch.empty(BH, qf.shape[1], n, device=dev, dtype=torch.float32)
    ev0 = torch.cuda.Event(); ev0.record(cur); S_COPY.wait_event(ev0); S_SCAN.wait_event(ev0)
    for (b0, b1) in chunk_bounds(nblk, PIPE):
        with torch.cuda.stream(S_COPY):
            gather_runs(st.chan_hostdev, st.chan_k.view(-1), so["ch"] + b0 * 8, on * ((b1 - b0) * 8), so["n_ch"], (b1 - b0) * 8)
            gather_runs(st.scales_hostdev, st.scales_k.view(torch.int32).view(-1), so["sc"] + b0 // 2, on * ((b1 - b0) // 2), so["n_ch"], (b1 - b0) // 2)
            ev = torch.cuda.Event(); ev.record(S_COPY)
        with torch.cuda.stream(S_SCAN):
            S_SCAN.wait_event(ev); st.scan(qf, act, nact, out=scores, blocks=(b0, b1))
    ev1 = torch.cuda.Event(); ev1.record(S_SCAN); cur.wait_event(ev1)
    return scores

def stream_landmark(store, qf):
    """Chunked copy of the fp16 block-mean index (stored block-major [nb, BH, D] so each chunk is one contiguous memcpy) overlapped with its scoring."""
    lm, lmh = store["lm_dev_t"], store["lm_host_t"]; nb = lm.shape[0]; cur = torch.cuda.current_stream()
    sb = torch.empty(lm.shape[1], qf.shape[1], nb, device=dev, dtype=torch.float32)
    ev0 = torch.cuda.Event(); ev0.record(cur); S_COPY.wait_event(ev0); S_SCAN.wait_event(ev0)
    for (b0, b1) in chunk_bounds(nb, PIPE, align=8):
        with torch.cuda.stream(S_COPY):
            lm[b0:b1].copy_(lmh[b0:b1], non_blocking=True); ev = torch.cuda.Event(); ev.record(S_COPY)
        with torch.cuda.stream(S_SCAN):
            S_SCAN.wait_event(ev); sb[:, :, b0:b1] = torch.einsum("bgd,nbd->bgn", qf, lm[b0:b1].float())
    ev1 = torch.cuda.Event(); ev1.record(S_SCAN); cur.wait_event(ev1)
    return sb

def offload_attn(q, c, L, m):
    """q [B,Hq,1,D]; host-resident prefill cache (n tokens) + device-resident recent tokens (positions n..m-1)."""
    B = q.shape[0]; BH = B * Hkv; n = ST["n"]; method = ST["method"]; qf = q.view(BH, G, D).float(); nrec = m - n
    if method == "dense_offload":                                                   # prefetch the whole layer K/V from pinned host memory, then sdpa
        Kd = c["Kdev"]; Vd = c["Vdev"]; Kd[:, :, :n].copy_(c["Khost"], non_blocking=True); Vd[:, :, :n].copy_(c["Vhost"], non_blocking=True)
        Kd[:, :, n:m] = c["Krec"][:, :, :nrec]; Vd[:, :, n:m] = c["Vrec"][:, :, :nrec]; ST["bytes"] += 2 * B * Hkv * n * D * 2
        return torch.nn.functional.scaled_dot_product_attention(q, Kd[:, :, :m], Vd[:, :, :m], enable_gqa=True)
    if method == "landmark8":
        lm = c["store"]["lm_dev"]
        if not args.index_hbm and PIPE: sb = stream_landmark(c["store"], qf); ST["bytes"] += lm.numel() * 2
        else:
            if not args.index_hbm: lm.copy_(c["store"]["lm_host"], non_blocking=True); ST["bytes"] += lm.numel() * 2
            sb = torch.einsum("bgd,bnd->bgn", qf, lm.float())
        ST["scan_bytes"] += lm.numel() * 2; nbk = sb.shape[-1]
        sb[..., :1] = -float("inf"); sb[..., max((m - LOCAL) // 8, 0):] = -float("inf")
        kk = K - SINK - LOCAL; tb = sb.topk(math.ceil(kk / 8), -1).indices                                              # [BH, G, ceil(kk/8)] blocks
        top = (tb[..., None] * 8 + torch.arange(8, device=dev)).reshape(BH, G, -1)[..., :kk].clamp(max=n - 1)         # exactly kk tokens
    elif method.startswith("planes"):
        g = ((qf ** 2) * VAR[L].repeat(B, 1)[:, None]).sum(1); plan = make_plan(torch.ones_like(g), 2.0 * D) if method == "planes_thumb2" else make_plan(g, float(BUD[int(method[-2:])][L]))
        if args.copy_index and PIPE:
            st = c["store"]
            if CHECK and L == 0: st.planes_k.zero_(); st.scales_k.zero_()
            scores = stream_planes(st, qf, plan, n, BH)
            if CHECK and L == 0:
                p32, sk = st.planes32, st.scales_k; st.planes32, st.scales_k = st.planes_ref.view(torch.int32), st.scales_ref
                ref = st.scan(qf, plan).clone(); st.planes32, st.scales_k = p32, sk; torch.cuda.synchronize()
                d = (scores - ref).abs().max().item(); print(f"pipeline check (planes): max |streamed - pristine| = {d:.3e} ({'OK' if d < 1e-3 else 'MISMATCH'})", flush=True); assert d < 1e-3
        elif args.copy_index:                                                          # gather the first t planes of every active channel (one contiguous run each) + its scales from host into HBM staging
            st = c["store"]; nblk = st.planes_k.shape[-1]; t = plan["t"].to(torch.int32).flatten(); R = BH * D
            if PROF and L == 0: torch.cuda.synchronize(); _t0 = time.perf_counter()
            slot = ARANGE_BHD[:R]; nrun = NRUN_ALL[:1] * 0 + R                                                  # every (bh, channel) slot is a run; inactive slots have length 0 -> no host sync
            offp = slot * (4 * nblk * 2); lenp = t * (nblk * 2); offs = slot * (nblk // 2); lens = (t > 0).to(torch.int32) * (nblk // 2)
            if PROF and L == 0: torch.cuda.synchronize(); _t1 = time.perf_counter()
            if CHECK and L == 0: st.planes_k.zero_(); st.scales_k.zero_()           # staging must be fully rebuilt by the gather for the check to mean anything
            gather_runs(st.planes_hostdev, st.planes32.view(-1), offp, lenp, nrun, 4 * nblk * 2); gather_runs(st.scales_hostdev, st.scales_k.view(torch.int32).view(-1), offs, lens, nrun, nblk // 2)
            if PROF and L == 0: torch.cuda.synchronize(); _t2 = time.perf_counter()
            if CHECK and L == 0:                                                      # gathered staging must reproduce the scan over the pristine GPU store
                got = st.scan(qf, plan).clone(); p32, sk = st.planes32, st.scales_k; st.planes32, st.scales_k = st.planes_ref.view(torch.int32), st.scales_ref
                ref = st.scan(qf, plan).clone(); st.planes32, st.scales_k = p32, sk; torch.cuda.synchronize()
                d = (got - ref).abs().max().item(); print(f"copy-index check: max |gathered - pristine| = {d:.3e} over {int(R)} runs ({'OK' if d < 1e-3 else 'MISMATCH'})", flush=True); assert d < 1e-3
        if not (args.copy_index and PIPE): scores = c["store"].scan(qf, plan)
        ST["bytes"] += bytes_planes(plan["t"], n); ST["scan_bytes"] += bytes_planes(plan["t"], n)
        if args.copy_index and PROF and L == 0: torch.cuda.synchronize(); print(f"[prof L0] runlist {1e3*(_t1-_t0):.2f} ms  gather {1e3*(_t2-_t1):.2f} ms  scan {1e3*(time.perf_counter()-_t2):.2f} ms", flush=True)
    else:
        r_ = int(method.split("_r")[1]); a = torch.zeros(BH, D, dtype=torch.bool, device=dev); a.scatter_(1, qf.abs().sum(1).topk(r_, -1).indices, True)   # SparQ GQA rule: top-r of the group's sum |q|; r=32 also matches the Loki / DS byte count
        act, nact = channel_list(a)
        if args.copy_index and PIPE:
            st = c["store"]
            if CHECK and L == 0: st.chan_k.zero_(); st.scales_k.zero_()
            scores = stream_chan4(st, qf, a, act, nact, n, BH)
            if CHECK and L == 0:
                ck, sk = st.chan_k, st.scales_k; st.chan_k, st.scales_k = st.chan_ref, st.scales_ref
                ref = st.scan(qf, act, nact).clone(); st.chan_k, st.scales_k = ck, sk; torch.cuda.synchronize()
                d = (scores - ref).abs().max().item(); print(f"pipeline check (chan4): max |streamed - pristine| = {d:.3e} ({'OK' if d < 1e-3 else 'MISMATCH'})", flush=True); assert d < 1e-3
        elif args.copy_index:                                                          # same treatment for the 4-bit channel store: one contiguous run per active channel + its scales
            st = c["store"]; nblk = st.chan_k.shape[-2]; R = BH * D; slot = ARANGE_BHD[:R]; nrun = NRUN_ALL[:1] * 0 + R; on = a.flatten().to(torch.int32)
            offp = slot * (nblk * 8); lenp = on * (nblk * 8); offs = slot * (nblk // 2); lens = on * (nblk // 2)
            gather_runs(st.chan_hostdev, st.chan_k.view(-1), offp, lenp, nrun, nblk * 8); gather_runs(st.scales_hostdev, st.scales_k.view(torch.int32).view(-1), offs, lens, nrun, nblk // 2)
        if not (args.copy_index and PIPE): scores = c["store"].scan(qf, act, nact)
        ST["bytes"] += bytes_chan4(nact, n); ST["scan_bytes"] += bytes_chan4(nact, n)
    if method != "landmark8":
        scores[..., :SINK] = -float("inf"); scores[..., max(m - LOCAL, 0):] = -float("inf")
        top = scores.topk(K - SINK - LOCAL, -1).indices                                                      # [BH, G, kk] indices < n
    fixed = torch.cat([torch.arange(SINK, device=dev), torch.arange(m - LOCAL, m, device=dev)])              # sinks + local window (may include recent tokens >= n)
    idx = torch.cat([top, fixed.expand(BH, G, -1)], -1)                                                       # [BH, G, K]
    flat = idx.reshape(BH, -1)
    # fetch each unique token of the GQA group once: sort, mark first occurrences, masked (no-traffic) loads for duplicates, forward-fill on device
    srt, order = flat.sort(-1); first = torch.cat([torch.ones(BH, 1, dtype=torch.bool, device=dev), srt[:, 1:] != srt[:, :-1]], 1)
    host_idx = torch.where(first & (srt < n), srt, torch.full_like(srt, -1)).to(torch.int32)                   # -1 = skip (duplicate or recent token)
    n_unique_host = (host_idx >= 0).sum()
    Ks = gather_rows(c["Khost_dev"], c["cap"], host_idx, BH); Vs = gather_rows(c["Vhost_dev"], c["cap"], host_idx, BH); ST["bytes"] += 2 * n_unique_host * D * 2; ST["row_bytes"] += 2 * n_unique_host * D * 2
    pos = torch.cummax(torch.where(first, torch.arange(flat.shape[1], device=dev)[None].expand(BH, -1), torch.zeros_like(flat)), 1).values   # position of the first occurrence
    Ks = torch.gather(Ks, 1, pos.unsqueeze(-1).expand(-1, -1, D)); Vs = torch.gather(Vs, 1, pos.unsqueeze(-1).expand(-1, -1, D))
    if nrec > 0:                                                                                                # recent tokens come from the device buffer
        rec = srt >= n; ridx = (srt - n).clamp(min=0, max=nrec - 1)
        Kr = torch.gather(c["Krec"].view(BH, -1, D), 1, ridx.unsqueeze(-1).expand(-1, -1, D)); Vr = torch.gather(c["Vrec"].view(BH, -1, D), 1, ridx.unsqueeze(-1).expand(-1, -1, D))
        Ks = torch.where(rec.unsqueeze(-1), Kr, Ks); Vs = torch.where(rec.unsqueeze(-1), Vr, Vs)
    inv = torch.empty_like(order); inv.scatter_(1, order, torch.arange(flat.shape[1], device=dev)[None].expand(BH, -1))                     # unsort
    Ks = torch.gather(Ks, 1, inv.unsqueeze(-1).expand(-1, -1, D)); Vs = torch.gather(Vs, 1, inv.unsqueeze(-1).expand(-1, -1, D))
    Ks = Ks.view(BH, G, K, D).float(); Vs = Vs.view(BH, G, K, D).float()
    s = torch.einsum("bgd,bgkd->bgk", qf, Ks) * D ** -0.5
    return torch.einsum("bgk,bgkd->bgd", torch.softmax(s, -1), Vs).to(q.dtype).view(B, Hq, 1, D)

def patched_forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, cache_position=None, **kw):
    B, Tq, _ = hidden_states.shape; L = self.layer_idx; c = ST["cache"][L]
    q = self.q_norm(self.q_proj(hidden_states).view(B, Tq, -1, D)).transpose(1, 2); k = self.k_norm(self.k_proj(hidden_states).view(B, Tq, -1, D)).transpose(1, 2); v = self.v_proj(hidden_states).view(B, Tq, -1, D).transpose(1, 2)
    cos, sin = position_embeddings; q, k = mq.apply_rotary_pos_emb(q, k, cos, sin)
    if ST["mode"] == "prefill":
        m0 = c["len"]; c["Kdev"][:, :, m0:m0 + Tq] = k; c["Vdev"][:, :, m0:m0 + Tq] = v; c["len"] = m = m0 + Tq
        mask = torch.arange(m, device=dev)[None] <= (m0 + torch.arange(Tq, device=dev))[:, None]
        o = torch.nn.functional.scaled_dot_product_attention(q, c["Kdev"][:, :, :m].repeat_interleave(G, 1), c["Vdev"][:, :, :m].repeat_interleave(G, 1), attn_mask=mask)
    else:
        r = c["len"] - ST["n"]; c["Krec"][:, :, r] = k[:, :, 0]; c["Vrec"][:, :, r] = v[:, :, 0]; c["len"] += 1
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record(); _c0 = time.perf_counter(); o = offload_attn(q, c, L, c["len"]); _c1 = time.perf_counter(); e1.record(); ST["events"].append((e0, e1))
        if PROF: ST.setdefault("cpu_attn", []).append(_c1 - _c0)
    return self.o_proj(o.transpose(1, 2).reshape(B, Tq, -1)), None
mq.Qwen3Attention.forward = patched_forward

res = []; out_path = args.out or f"{RES}/B_e2e_offload{'_w4' if args.w4 else ''}{'_indexhbm' if args.index_hbm else ''}{'_synth' if args.synthetic_kv else ''}{'_copyidx' if args.copy_index else ''}.json"
for n in CTXS:
    for B in BATCHES:
        kv_gb = nL * Hkv * n * D * 2 * 2 * B / 1e9; store_gb = nL * Hkv * math.ceil(n / SBT) * SBT * 64 * B / 1e9
        if kv_gb + store_gb > args.host_gb: print(f"SKIP ctx={n} B={B}: host KV {kv_gb:.1f} GB + store {store_gb:.1f} GB > {args.host_gb} GB budget", flush=True); res.append(dict(ctx=n, batch=B, skipped=f"host {kv_gb+store_gb:.1f} GB")); continue
        ids = torch.randint(100, 10000, (B, n), device=dev)
        if args.copy_index: bp.SUB = math.ceil(n / BLK); bp.SBT = BLK * bp.SUB          # whole sequence = one superblock: every (channel, plane) run is contiguous
        for method in METHODS:
            hps = []
            try:
                mk = (lambda: {"len": 0}) if args.synthetic_kv else (lambda: {"Kdev": torch.empty(B, Hkv, n + STEPS + args.profile_steps + 1, D, dtype=torch.bfloat16, device=dev), "Vdev": torch.empty(B, Hkv, n + STEPS + args.profile_steps + 1, D, dtype=torch.bfloat16, device=dev), "len": 0})
                ST.update(cache=[mk() for _ in range(nL)], mode="prefill", method=method, B=B, n=n)   # synthetic mode never holds the full KV on the GPU (147 GB at 1M)
                t0 = time.time()
                if args.synthetic_kv:
                    if method == "dense_offload":                                    # one shared prefetch target (the layers run sequentially)
                        Kd = torch.empty(B, Hkv, n + STEPS + args.profile_steps + 1, D, dtype=torch.bfloat16, device=dev); Vd = torch.empty_like(Kd)
                        for c in ST["cache"]: c["Kdev"], c["Vdev"] = Kd, Vd
                else:
                    for c0 in range(0, n, args.prefill_chunk):
                        c1 = min(n, c0 + args.prefill_chunk); model(ids[:, c0:c1], position_ids=torch.arange(c0, c1, device=dev)[None].expand(B, -1), use_cache=False, logits_to_keep=1)
                torch.cuda.synchronize(); t_prefill = time.time() - t0; t0 = time.time()
                for L, c in enumerate(ST["cache"]):                                   # offload: K/V (and the scan store) to pinned host memory; keep only a recent-token buffer on the GPU
                    if args.synthetic_kv:
                        Kt = ftr.get_slice(f"k{L}")[:].to(dev).to(torch.bfloat16); reps = math.ceil(n / Kt.shape[1])
                        Kn = Kt.repeat(B, reps, 1)[:, :n].contiguous().view(B, Hkv, n, D); del Kt; Vn = torch.randn_like(Kn); c["len"] = n
                    else: Kn = c["Kdev"][:, :, :n].contiguous(); Vn = c["Vdev"][:, :, :n].contiguous()
                    c["Khost"], c["Khost_dev"], hp = host_mapped(Kn.view(B * Hkv, n, D)); hps.append(hp); c["Vhost"], c["Vhost_dev"], hp = host_mapped(Vn.view(B * Hkv, n, D)); hps.append(hp); c["cap"] = n
                    c["Krec"] = torch.zeros(B, Hkv, STEPS + args.profile_steps + 1, D, dtype=torch.bfloat16, device=dev); c["Vrec"] = torch.zeros_like(c["Krec"])
                    if method == "landmark8":
                        nb = math.ceil(n / 8); lm = torch.cat([Kn.view(B * Hkv, n, D), Kn.new_zeros(B * Hkv, nb * 8 - n, D)], 1).view(B * Hkv, nb, 8, D).float().mean(2).half()
                        c["store"] = {"lm_dev": lm if args.index_hbm else torch.empty_like(lm), "lm_host": None if args.index_hbm else torch.empty(lm.shape, dtype=lm.dtype, pin_memory=True).copy_(lm)}
                        if PIPE and not args.index_hbm:                                          # block-major copy for the chunked pipeline
                            lmt = lm.transpose(0, 1).contiguous(); c["store"]["lm_dev_t"] = torch.empty_like(lmt); c["store"]["lm_host_t"] = torch.empty(lmt.shape, dtype=lmt.dtype, pin_memory=True).copy_(lmt); del lmt
                        if not args.synthetic_kv: del c["Kdev"], c["Vdev"]
                    elif method != "dense_offload":
                        st = build_store("planes" if method.startswith("planes") else "chan4", Kn.view(B * Hkv, n, D))
                        if args.index_hbm: pass
                        elif args.copy_index and method.startswith("planes"):
                            _, st.planes_hostdev, hp = host_mapped(st.planes_k); hps.append(hp); st.planes_hostdev = DevPtr(st.planes_hostdev.ptr, torch.int32)   # GPU tensors stay as staging
                            if CHECK and L == 0: st.planes_ref, st.scales_ref = st.planes_k.clone(), st.scales_k.clone()
                            _, st.scales_hostdev, hp = host_mapped(st.scales_k); hps.append(hp); st.scales_hostdev = DevPtr(st.scales_hostdev.ptr, torch.int32)
                        elif args.copy_index:
                            if CHECK and L == 0: st.chan_ref, st.scales_ref = st.chan_k.clone(), st.scales_k.clone()
                            _, st.chan_hostdev, hp = host_mapped(st.chan_k); hps.append(hp); st.chan_hostdev = DevPtr(st.chan_hostdev.ptr, torch.int32)
                            _, st.scales_hostdev, hp = host_mapped(st.scales_k); hps.append(hp); st.scales_hostdev = DevPtr(st.scales_hostdev.ptr, torch.int32)
                        elif method.startswith("planes"):
                            _, st.planes32, hp = host_mapped(st.planes_k); hps.append(hp); st.planes32 = DevPtr(st.planes32.ptr, torch.int32); del st.planes_k
                        else: _, st.chan_k, hp = host_mapped(st.chan_k); hps.append(hp)
                        if not args.index_hbm and not args.copy_index: _, st.scales_k, hp = host_mapped(st.scales_k); hps.append(hp)
                        c["store"] = st
                        if not args.synthetic_kv: del c["Kdev"], c["Vdev"]            # the GPU no longer holds the prefill KV
                    else: c["Khost"] = c["Khost"].view(B, Hkv, n, D); c["Vhost"] = c["Vhost"].view(B, Hkv, n, D)   # device buffers kept as the prefetch target
                    del Kn, Vn
                    if os.environ.get("DEBUG_MEM") == "1" and L % 6 == 0: print(f"[mem] L{L} allocated {torch.cuda.memory_allocated()/1e9:.2f} GB reserved {torch.cuda.memory_reserved()/1e9:.2f} GB", flush=True)
                torch.cuda.empty_cache(); t_off = time.time() - t0
                ST["mode"] = "decode"; nxt = ids[:, -1:]; steps, attns = [], []
                for s_ in range(STEPS):
                    pos = torch.full((B, 1), n + s_, device=dev); ST["events"] = []; ST["bytes"] = 0.0; ST["scan_bytes"] = 0.0; ST["row_bytes"] = 0.0
                    torch.cuda.synchronize(); t0 = time.perf_counter(); ST["cpu_attn"] = []; logits = model(nxt, position_ids=pos, use_cache=False).logits; _cpu_model = time.perf_counter() - t0; torch.cuda.synchronize(); dt = (time.perf_counter() - t0) * 1e3
                    if PROF: print(f"[prof step {s_}] wall {dt:.1f} ms; CPU issue for whole model {1e3*_cpu_model:.1f} ms; CPU inside attention {1e3*sum(ST['cpu_attn']):.1f} ms over {len(ST['cpu_attn'])} layers", flush=True)
                    nxt = logits[:, -1:].argmax(-1)
                    if s_ >= 2: steps.append(dt); attns.append(sum(a.elapsed_time(b) for a, b in ST["events"]))
                torch.cuda.synchronize(); pcie_bytes = float(ST["bytes"]); scan_b = float(ST["scan_bytes"]); row_b = float(ST["row_bytes"])
                from torch.profiler import profile, ProfilerActivity
                import statistics, gzip
                gms, gsum = [], []
                for ps in range(args.profile_steps):
                    ST["events"] = []; torch.cuda.synchronize()
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        model(nxt, position_ids=torch.full((B, 1), n + STEPS + ps, device=dev), use_cache=False); torch.cuda.synchronize()
                    kev = [e for e in prof.events() if e.device_type.name == "CUDA"]; gsum.append(sum(e.device_time for e in kev) / 1e3)
                    iv = sorted((e.time_range.start, e.time_range.end) for e in kev); tot = 0.0; cs = ce = None
                    for a_, b_ in iv:
                        if cs is None or a_ > ce:
                            if cs is not None: tot += ce - cs
                            cs, ce = a_, b_
                        else: ce = max(ce, b_)
                    if cs is not None: tot += ce - cs
                    gms.append(tot / 1e3)
                if args.trace_dir:
                    os.makedirs(args.trace_dir, exist_ok=True); tp = f"{args.trace_dir}/trace_offload_{args.model.split('/')[-1]}_ctx{n}_B{B}_{method}.json"
                    prof.export_chrome_trace(tp); open(tp + ".gz", "wb").write(gzip.compress(open(tp, "rb").read())); os.remove(tp)
                gpu_ms = statistics.mean(gms); gpu_std = statistics.pstdev(gms) if len(gms) > 1 else 0.0; gpu_sum_ms = statistics.mean(gsum)
                step = sorted(steps)[len(steps) // 2]; attn = sorted(attns)[len(attns) // 2]
                r = dict(model=args.model, w4=args.w4, index_hbm=args.index_hbm, copy_index=args.copy_index, synthetic_kv=args.synthetic_kv, ctx=n, batch=B, method=method, k=K, step_ms=step, attn_ms=attn, tok_per_s=B * 1000 / step, step_gpu_ms=gpu_ms, step_gpu_sum_ms=gpu_sum_ms, pipeline=PIPE, step_gpu_ms_std=gpu_std, step_gpu_ms_all=gms, tok_per_s_gpu=B * 1000 / gpu_ms,
                         pcie_GB_per_step=pcie_bytes / 1e9, scan_MB_per_step=scan_b / 1e6, rows_MB_per_step=row_b / 1e6, pcie_GBps_effective=pcie_bytes / attn / 1e6, prefill_s=t_prefill, offload_s=t_off, host_kv_GB=kv_gb, weight_GB=weight_gb, peak_gpu_GB=torch.cuda.max_memory_allocated() / 1e9)
                res.append(r); print(f"ctx={n:6d} B={B} {method:20s} step={step:7.1f} ms (GPU+memcpy {gpu_ms:6.1f}+-{gpu_std:.1f} over {len(gms)})  attn={attn:7.1f} ms  {r['tok_per_s']:5.1f} tok/s ({r['tok_per_s_gpu']:5.1f} launch-free)  PCIe {r['pcie_GB_per_step']*1e3:7.1f} MB/step (scan {r['scan_MB_per_step']:6.1f} + rows {r['rows_MB_per_step']:5.1f}) @ {r['pcie_GBps_effective']:.1f} GB/s  peak GPU {r['peak_gpu_GB']:.1f} GB", flush=True)
            except (torch.OutOfMemoryError, MemoryError) as e:
                print(f"OOM ctx={n} B={B} {method}: {str(e)[:100]}", flush=True); res.append(dict(ctx=n, batch=B, method=method, oom=True))
            ST["cache"] = None; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            for hp in hps: free_host(hp)
        json.dump(res, open(out_path, "w"), indent=1)
print("done ->", out_path)
