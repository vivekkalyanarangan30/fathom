"""Top-k over [BH*G, n] fp32 scores at 1M: torch.topk vs chunked two-stage top-k vs fp16; exactness of the selected set."""
import torch, time, sys
dev = torch.device("cuda"); n = int(sys.argv[1]) if len(sys.argv) > 1 else 1048576; k = 476; rows = 32
s = torch.randn(rows, n, device=dev)
def timeit(fn, reps=10):
    fn(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / reps * 1e3
ref = s.topk(k, -1).indices
print(f"torch.topk fp32 [{rows},{n}]: {timeit(lambda: s.topk(k, -1)):.2f} ms")
for C in (8, 16, 32, 64):
    def two_stage():
        v, i = s.view(rows, C, n // C).topk(k, -1); v2, j = v.reshape(rows, -1).topk(k, -1)
        return torch.gather((i + (torch.arange(C, device=dev) * (n // C))[None, :, None]).reshape(rows, -1), 1, j)
    idx = two_stage(); same = (idx.sort(-1).values == ref.sort(-1).values).all().item()
    print(f"two-stage C={C}: {timeit(two_stage):.2f} ms exact-set {same}")
sh = s.half(); print(f"torch.topk fp16: {timeit(lambda: sh.topk(k, -1)):.2f} ms")
def thresh():
    smp = s[:, ::256]; th = smp.topk(max(1, int(k * 2 / 256)), -1).values[:, -1:]           # sample-based threshold, 2x margin
    m = s >= th; cand = torch.where(m, s, torch.full_like(s, -float("inf")))
    return cand.topk(k, -1).indices
print(f"threshold+topk (not exact-safe): {timeit(thresh):.2f} ms")
