"""Task B microbenchmark: real Triton scan kernels vs the 4-bit channel-major baseline, with real Qwen3-8B plans and queries.
For each (layer, n, batch): plan -> scan -> top-k -> gather + exact attention, per method. Each stage is captured in a CUDA graph and
replayed (CUDA events around each replay => GPU time without Triton's Python launch overhead, which is 0.1-0.3 ms per call on this 4-vCPU VM);
the eager wall time of the whole pipeline is also recorded. Methods interleaved per repetition, median of REPS after warm-up.
Reports bytes moved, achieved GB/s, predicted time at peak bandwidth, and the scan's share of the pipeline.
Usage: python scripts/B_kernel_bench.py --tag Qwen3-8B_T16384 --ns 8192,16384,32768,65536,131072 --batches 1,4,8 --layers 0,1,2,18,30 --k 512"""
import sys, os, json, math, time, argparse, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
from capsio import safe_open   # lazy pread reader (no mmap)
from common import RES
from sketch import SINK, LOCAL
from bitplane import *
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="Qwen3-8B_T16384"); ap.add_argument("--ns", default="8192,16384,32768,65536,131072")
ap.add_argument("--batches", default="1,4,8"); ap.add_argument("--layers", default="0,1,2,18,30"); ap.add_argument("--k", type=int, default=512)
ap.add_argument("--reps", type=int, default=30); ap.add_argument("--out", default=None); ap.add_argument("--peak-gbps", type=float, default=300.0)
ap.add_argument("--single-head-too", action="store_true", default=True)
args = ap.parse_args(); torch.set_grad_enabled(False); dev = torch.device("cuda")
base, Ttag = args.tag.split("_T")[0], args.tag.split("_")[-1]
ftr = safe_open(f"{RES}/caps_{base}_train_{Ttag}.safetensors", "pt"); fte = safe_open(f"{RES}/caps_{base}_test_{Ttag}.safetensors", "pt")
FA = json.load(open(f"{RES}/final_alloc_{args.tag}.json"))
Hq, T, Dd = fte.get_slice("q0").get_shape(); Hkv = fte.get_slice("k0").get_shape()[0]; G = Hq // Hkv; assert Dd == D
NS = [int(x) for x in args.ns.split(",")]; BATCHES = [int(x) for x in args.batches.split(",")]; LAYERS = [int(x) for x in args.layers.split(",")]; K = args.k
print(f"{args.tag}: Hkv={Hkv} G={G} T={T}; n={NS} batches={BATCHES} layers={LAYERS} k={K} reps={args.reps}", flush=True)

def ev(): e = torch.cuda.Event(enable_timing=True); e.record(); return e
def gather_attn(q, Kc, Vc, idx):
    """q [BH,G,D] fp32, Kc/Vc [BH,n,D] bf16, idx [BH,G,k] -> exact attention over the selected rows (bf16 rows, fp32 math)."""
    BH, Gh, k = idx.shape; flat = idx.reshape(BH, Gh * k)
    Ks = torch.gather(Kc, 1, flat.unsqueeze(-1).expand(-1, -1, D)).view(BH, Gh, k, D).float(); Vs = torch.gather(Vc, 1, flat.unsqueeze(-1).expand(-1, -1, D)).view(BH, Gh, k, D).float()
    s = torch.einsum("bgd,bgkd->bgk", q, Ks) * D ** -0.5
    return torch.einsum("bgk,bgkd->bgd", torch.softmax(s, -1), Vs)
def dense_attn(q, Kc, Vc):
    return torch.nn.functional.scaled_dot_product_attention(q.to(Kc.dtype).unsqueeze(2), Kc.unsqueeze(1), Vc.unsqueeze(1), enable_gqa=True)

rows = []
for L in LAYERS:
    Kfull = fte.get_slice(f"k{L}")[:].to(dev).float()                                   # [Hkv, T, D]
    Qall = fte.get_slice(f"q{L}")[:, T - 8:].to(dev).float()                              # [Hq, 8, D] last 8 positions (one per batch element)
    Ktr = ftr.get_slice(f"k{L}")[:].to(dev).float(); var = Ktr[:, SINK * 4:].var(1)         # [Hkv, D] calibration variance
    del Ktr
    budgets = {48: FA["u4/layeralloc_mean48"]["alloc"][str(L)], 64: FA["u4/layeralloc_mean64"]["alloc"][str(L)]}
    for n in NS:
        reps_needed = math.ceil(n / T); Kn = Kfull.repeat(1, reps_needed, 1)[:, :n]        # [Hkv, n, D] real keys tiled to n
        configs = [(1, 1)] + [(b, Hkv) for b in BATCHES]                                  # (batch, heads) -> BH
        for batch, heads in configs:
            BH = batch * heads
            try:
                Kb = Kn[:heads].repeat(batch, 1, 1).contiguous()                             # [BH, n, D]
                q = torch.stack([Qall[hk * G:(hk + 1) * G, b % 8] for b in range(batch) for hk in range(heads)])   # [BH, G, D]
                v = var[:heads].repeat(batch, 1)                                             # [BH, D]
                store = PlaneStore(Kb); cs = Chan4Store(Kb)
                Kc = Kb.to(torch.bfloat16); Vc = torch.randn_like(Kc)
            except torch.OutOfMemoryError:
                print(f"OOM building stores at L{L} n={n} BH={BH}; skipping", flush=True); torch.cuda.empty_cache(); continue
            g = ((q ** 2) * v[:, None]).sum(1)
            # channel sets for the chan4 baselines
            abs_sum = q.abs().sum(1)
            def act_topr(r):                                                                  # SparQ GQA rule: top-r of sum_g |q_g|
                a = torch.zeros(BH, D, dtype=torch.bool, device=dev); a.scatter_(1, abs_sum.topk(r, -1).indices, True); return a
            contig = torch.zeros(BH, D, dtype=torch.bool, device=dev); contig[:, :32] = True                        # a Loki / Double-Sparsity store: 32 stored coordinates, contiguous
            chan_sets = {"chan4_r128_fulldim": torch.ones(BH, D, dtype=torch.bool, device=dev), "chan4_r32_loki_ds_sparq32": act_topr(32), "chan4_r32_contiguous_loki_ds_store": contig,
                         "chan4_r16_sparq16": act_topr(16)}
            methods = ["dense_sdpa_bf16"] + list(chan_sets) + ["planes_u4_mean48", "planes_u4_mean64"]
            def stages(m):
                """Return the 4 stage closures (plan, scan, topk, attn) for method m, sharing state through a dict."""
                st = {}
                if m == "dense_sdpa_bf16":
                    return [lambda: None, lambda: None, lambda: None, lambda: st.__setitem__("o", dense_attn(q, Kc, Vc))]
                if m.startswith("chan4"):
                    a = chan_sets[m]
                    return [lambda: st.__setitem__("al", channel_list(a)), lambda: st.__setitem__("sc", cs.scan(q, *st["al"])),
                            lambda: st.__setitem__("idx", sink_local_topk(st["sc"], K, SINK, LOCAL)), lambda: st.__setitem__("o", gather_attn(q, Kc, Vc, st["idx"]))]
                bud = float(budgets[int(m[-2:])])
                return [lambda: st.__setitem__("plan", make_plan(g, bud)), lambda: st.__setitem__("sc", store.scan(q, st["plan"])),
                        lambda: st.__setitem__("idx", sink_local_topk(st["sc"], K, SINK, LOCAL)), lambda: st.__setitem__("o", gather_attn(q, Kc, Vc, st["idx"]))]
            def graph_of(fn):
                for _ in range(3): fn()
                torch.cuda.synchronize(); gph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gph): fn()
                torch.cuda.synchronize(); return gph
            times = {m: {s: [] for s in ("plan", "scan", "topk", "attn")} for m in methods}; wall = {m: [] for m in methods}; info = {}; graphs = {}
            for m in methods:
                fns = stages(m)
                for f in fns: f()                                                                   # eager warm-up (Triton compile)
                if m.startswith("chan4"): al = channel_list(chan_sets[m]); info[m] = {"bytes": bytes_chan4(al[1], n), "mean_active": al[1].float().mean().item()}
                elif m.startswith("planes"):
                    plan = make_plan(g, float(budgets[int(m[-2:])]))
                    info[m] = {"bytes": bytes_planes(plan["t"], n), "mean_active": plan["nact"].float().mean().item(), "mean_planes_per_active": (plan["t"].float().sum(-1) / plan["nact"].float()).mean().item(),
                               "depth_class_sizes": plan["clsn"].float().mean(0).tolist()}
                graphs[m] = [graph_of(f) for f in fns]
            for rep in range(args.reps + 3):
                for m in methods:
                    evs = [ev()]
                    for gph in graphs[m]: gph.replay(); evs.append(ev())
                    torch.cuda.synchronize()
                    if rep >= 3:
                        for s_, (a, b) in zip(("plan", "scan", "topk", "attn"), zip(evs[:-1], evs[1:])): times[m][s_].append(a.elapsed_time(b))
                    e0 = ev()
                    for f in stages(m): f()
                    e1 = ev(); torch.cuda.synchronize()
                    if rep >= 3: wall[m].append(e0.elapsed_time(e1))
            info["dense_sdpa_bf16"] = {"bytes": float(BH * n * D * 2 * 2), "mean_active": D}
            med = lambda a: sorted(a)[len(a) // 2]
            for m in methods:
                t = {s: med(v) for s, v in times[m].items()}; tot = sum(t.values()); by = info[m]["bytes"]
                r = dict(layer=L, n=n, batch=batch, heads=heads, BH=BH, method=m, plan_ms=t["plan"], scan_ms=t["scan"], topk_ms=t["topk"], attn_ms=t["attn"], total_ms=tot, eager_wall_ms=med(wall[m]),
                         bytes=by, bits_per_token=by * 8 / (n * BH), scan_GBps=by / t["scan"] / 1e6 if t["scan"] > 0 else None, pred_scan_ms_at_peak=by / args.peak_gbps / 1e6,
                         scan_share=t["scan"] / tot, topk_share=t["topk"] / tot, mean_active=info[m]["mean_active"], mean_planes_per_active=info[m].get("mean_planes_per_active"),
                         budget=budgets.get(int(m[-2:])) if m.startswith("planes") else None, k=K, depth_class_sizes=info[m].get("depth_class_sizes"), planes_kernel=PlaneStore.kernel)
                rows.append(r)
                print(f"L{L:2d} n={n:6d} B={batch} H={heads} {m:28s} plan={t['plan']:.3f} scan={t['scan']:7.3f} topk={t['topk']:6.3f} attn={t['attn']:6.3f} tot={tot:7.3f} ms (eager {r['eager_wall_ms']:6.2f})  "
                      f"{r['bits_per_token']:6.1f} b/tok  {r['scan_GBps'] or 0:6.1f} GB/s  scan {100*r['scan_share']:4.1f}% topk {100*r['topk_share']:4.1f}%", flush=True)
            del store, cs, Kc, Vc, Kb; torch.cuda.empty_cache()
            json.dump({"meta": vars(args) | {"G": G, "Hkv": Hkv, "T": T, "gpu": torch.cuda.get_device_name(0)}, "rows": rows}, open(args.out or f"{RES}/B_kernel_bench.json", "w"), indent=1)
print("done")
