"""Sweep Triton launch configs of the plane and channel scan kernels on this GPU at long context (both methods get the same tuning).
Usage: python scripts/bench_scan_cfg.py --n 1048576 --bh 8 --budget 48 --out results/scan_cfg.json"""
import sys, os, json, time, argparse, itertools, torch, triton
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
from bitplane import *
import bitplane as bp
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1048576); ap.add_argument("--bh", type=int, default=8); ap.add_argument("--g", type=int, default=4)
ap.add_argument("--budget", type=float, default=48); ap.add_argument("--reps", type=int, default=5); ap.add_argument("--out", default=None)
a = ap.parse_args(); dev = torch.device("cuda"); torch.manual_seed(0)
bp.SUB = math.ceil(a.n / BLK); bp.SBT = BLK * bp.SUB                                  # copy-index layout: whole sequence = one superblock
K = torch.randn(a.bh, a.n, D, device=dev) * torch.rand(1, 1, D, device=dev)          # unequal channel spreads, as in real keys
q = torch.randn(a.bh, a.g, D, device=dev); g = ((q ** 2) * K.var(1)[:, None]).sum(1)
plan = make_plan(g, a.budget); act, nact = channel_list(torch.zeros(a.bh, D, dtype=torch.bool, device=dev).scatter_(1, q.abs().sum(1).topk(16, -1).indices, True))
act32, nact32 = channel_list(torch.zeros(a.bh, D, dtype=torch.bool, device=dev).scatter_(1, q.abs().sum(1).topk(32, -1).indices, True))
ps = PlaneStore(K); cs = Chan4Store(K); del K; torch.cuda.synchronize()
def timeit(fn):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(a.reps):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record(); fn(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    return sorted(ts)[len(ts) // 2]
res = {"n": a.n, "bh": a.bh, "budget": a.budget, "gpu": torch.cuda.get_device_name(0), "planes_cls": [], "planes_dot": [], "chan4_r16": [], "chan4_r32": []}
ref = ps.scan(q, plan).clone(); base = timeit(lambda: ps.scan(q, plan)); print(f"planes cls default {PlaneStore.cls_cfg}: {base:.2f} ms", flush=True)
for NB, KK, W in itertools.product((1, 2, 4, 8, 16), (1, 2, 4, 8), (1, 2, 4, 8)):
    if NB * 2 * 32 * KK * W > 65536 * 4: continue
    PlaneStore.cls_cfg = {"NB": NB, "KK": KK, "num_warps": W, "num_stages": 1}
    try:
        out = ps.scan(q, plan); err = (out - ref).abs().max().item(); t = timeit(lambda: ps.scan(q, plan))
        res["planes_cls"].append({"NB": NB, "KK": KK, "warps": W, "ms": t, "max_abs_err": err}); print(f"  cls NB={NB} KK={KK} W={W}: {t:.2f} ms err {err:.1e}", flush=True)
    except Exception as e: print(f"  cls NB={NB} KK={KK} W={W}: fail {type(e).__name__}", flush=True)
PlaneStore.cls_cfg = {"NB": 4, "KK": 4, "num_warps": 1, "num_stages": 1}
try:
    PlaneStore.kernel = "dot"; out = ps.scan(q, plan); err = (out - ref).abs().max().item(); rel = err / ref.abs().max().item(); t = timeit(lambda: ps.scan(q, plan))
    res["planes_dot"].append({"ms": t, "max_abs_err": err, "rel": rel}); print(f"planes dot (tensor-core, autotuned): {t:.2f} ms, max abs err {err:.2e} (rel {rel:.1e})", flush=True)
except Exception as e: print("dot kernel failed:", type(e).__name__, str(e)[:120], flush=True)
PlaneStore.kernel = "cls"
for name, (ac, na) in (("chan4_r16", (act, nact)), ("chan4_r32", (act32, nact32))):
    refc = cs.scan(q, ac, na).clone(); print(f"{name} default {Chan4Store.cfg}: {timeit(lambda: cs.scan(q, ac, na)):.2f} ms", flush=True)
    for NB, KK, W in itertools.product((1, 2, 4, 8), (1, 2, 4, 8), (1, 2, 4, 8)):
        Chan4Store.cfg = {"NB": NB, "KK": KK, "num_warps": W, "num_stages": 1}
        try:
            out = cs.scan(q, ac, na); err = (out - refc).abs().max().item(); t = timeit(lambda: cs.scan(q, ac, na))
            res[name].append({"NB": NB, "KK": KK, "warps": W, "ms": t, "max_abs_err": err}); print(f"  {name} NB={NB} KK={KK} W={W}: {t:.2f} ms err {err:.1e}", flush=True)
        except Exception as e: print(f"  {name} NB={NB} KK={KK} W={W}: fail {type(e).__name__}", flush=True)
    Chan4Store.cfg = {"NB": 2, "KK": 1, "num_warps": 1, "num_stages": 1}
for k in ("planes_cls", "chan4_r16", "chan4_r32"):
    ok = [r for r in res[k] if r["max_abs_err"] < 1e-3]; best = min(ok, key=lambda r: r["ms"]); print(f"BEST {k}: {best}", flush=True); res[k + "_best"] = best
json.dump(res, open(a.out or "results/scan_cfg.json", "w"), indent=1)
