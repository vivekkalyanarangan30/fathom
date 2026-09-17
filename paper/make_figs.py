"""Figures for the Bit-Plane Key Scan paper. Reads the project's result JSONs and writes vector PDFs into paper/figs/."""
import json, os, math, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R13 = f"{ROOT}/results/a100_20260913"
LLAMA = f"{ROOT}/results/l4/final_alloc_Meta-Llama-3.1-8B_T4096_K128.json"
OUT = f"{ROOT}/paper/figs"
os.makedirs(OUT, exist_ok=True)

IND = "#4A3FA8"; IND2 = "#8C84D6"
GREYS = ["#4D4D5C", "#7A7A8C", "#A8A8B8"]; ORANGE = "#B8682A"; ORANGE2 = "#D9A06B"; RED = "#B23A3A"; GREEN = "#2E7D5B"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
SC, DC = 3.3, 6.8

def load(p):
    return json.load(open(p))

def rows(d, prefix):
    """(bits, err) pairs for variants named prefix + budget, sorted by bits."""
    pts = [(v["bits"], v["err"]) for k, v in d.items() if k.startswith(prefix) and k[len(prefix):].isdigit()]
    return sorted(pts)

def save(fig, name):
    fig.savefig(f"{OUT}/{name}", bbox_inches="tight"); plt.close(fig); print("wrote", name)

BASE_STYLE = {"dsparsity_c32_4bit": ("Double Sparsity c=32", "s", GREYS[0]), "loki_r32_4bit": ("Loki r=32", "^", GREYS[1]), "loki_r64_4bit": ("Loki r=64", "v", GREYS[1]),
              "sparq_r16_4bit": ("SparQ r=16", "d", ORANGE), "sparq_r32_4bit": ("SparQ r=32", "D", ORANGE), "thumb_2bit_all": ("2-bit thumbnail", "X", GREYS[2])}   # full 4-bit scan (err ~1e-7) is off-scale and stated in the caption

def baselines(ax, d, keys=BASE_STYLE, label=True):
    for k, (nm, mk, col) in keys.items():
        if k in d and d[k]["err"] > 0:
            ax.scatter(d[k]["bits"], d[k]["err"], marker=mk, s=28, color=col, zorder=3, label=nm if label else None, edgecolor="white", linewidth=0.4)

# 1. frontier ------------------------------------------------------------------------------------------------------------
def fig_frontier():
    panels = [(f"{R13}/final_alloc_Qwen3-8B_T16384.json", "Qwen3-8B, 16k, K=256", "u4"),
              (f"{R13}/final_alloc_Qwen3-8B_T32768.json", "Qwen3-8B, 32k, K=512", "u4"),
              (f"{R13}/final_alloc_Qwen2.5-7B-Instruct-1M_T131072_K2048.json", "Qwen2.5-7B-1M, 128k, K=2048", "u4klt")]
    fig, axs = plt.subplots(1, 3, figsize=(DC, 2.3), sharey=False)
    for ax, (p, title, st) in zip(axs, panels):
        d = load(p)
        pts = rows(d, f"{st}/planes"); ax.plot([b for b, _ in pts], [e for _, e in pts], "-o", color=IND, ms=3.5, lw=1.4, label="Fathom, flat budget (basis per rule)", zorder=4)
        la = sorted((v["bits"], v["err"]) for k, v in d.items() if k.startswith(f"{st}/layeralloc_mean") and "@" not in k)
        if la: ax.plot([b for b, _ in la], [e for _, e in la], "--o", color=IND2, ms=3, lw=1, label="Fathom, per-layer plan", zorder=4)
        other = "u4klt" if st == "u4" else "u4"; kl = rows(d, f"{other}/planes")
        if kl: ax.plot([b for b, _ in kl], [e for _, e in kl], ":o", color=GREEN, ms=2.5, lw=1, label="Fathom, other basis", zorder=3)
        baselines(ax, d, label=(ax is axs[0]))
        ax.set_yscale("log"); ax.set_xlabel("scan bits / token"); ax.set_title(title, loc="left", fontsize=8); ax.grid(alpha=.25, which="both", lw=.4)
        ax.set_xlim(20, 300); ax.set_ylim(5e-5, min(1.0, ax.get_ylim()[1]))
    axs[0].set_ylabel("attention-output relative error")
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); save(fig, "fig_frontier.pdf")

# 2. selection ratio -----------------------------------------------------------------------------------------------------
def fig_ratio():
    srcs = [(128, f"{R13}/final_alloc_Qwen2.5-7B-Instruct-1M_T131072.json"), (512, f"{R13}/final_alloc_Qwen2.5-7B-Instruct-1M_T131072_K512.json"), (2048, f"{R13}/final_alloc_Qwen2.5-7B-Instruct-1M_T131072_K2048.json")]
    fig, ax = plt.subplots(figsize=(SC, 3.1)); cols = ["#2A2358", IND, IND2]
    for (K, p), c in zip(srcs, cols):
        d = load(p); pts = rows(d, "u4klt/planes")
        ax.plot([b for b, _ in pts], [e for _, e in pts], "-o", color=c, ms=3, lw=1.3, label=f"Fathom, K={K} ({K/131072*100:.1f}%)")
        ax.scatter(d["dsparsity_c32_4bit"]["bits"], d["dsparsity_c32_4bit"]["err"], marker="s", facecolor="none", edgecolor=c, s=30, zorder=3)
        ax.scatter(d["sparq_r32_4bit"]["bits"], d["sparq_r32_4bit"]["err"], marker="D", facecolor="none", edgecolor=c, s=30, zorder=3)
    ax.scatter([], [], marker="s", facecolor="none", edgecolor="k", s=30, label="hollow square: Double Sparsity c=32 (136 b), colour = same K")
    ax.scatter([], [], marker="D", facecolor="none", edgecolor="k", s=30, label="hollow diamond: SparQ r=32 (136 b), colour = same K")
    ax.set_yscale("log"); ax.set_xlabel("scan bits / token"); ax.set_ylabel("attention-output relative error"); ax.grid(alpha=.25, which="both", lw=.4)
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=1)
    fig.tight_layout(); save(fig, "fig_ratio.pdf")

# 3. basis rule ----------------------------------------------------------------------------------------------------------
def fig_basis():
    srcs = [("Qwen3-8B\n16k, K=256", f"{R13}/final_alloc_Qwen3-8B_T16384.json"), ("Qwen3-4B\n16k, K=256", f"{R13}/final_alloc_Qwen3-4B_T16384.json"), ("Llama-3.1-8B\n4k, K=128", LLAMA),
            ("Qwen2.5-7B\n32k, K=512", f"{R13}/final_alloc_Qwen2.5-7B_T32768.json"), ("Qwen2.5-7B-1M\n32k, K=512", f"{R13}/final_alloc_Qwen2.5-7B-Instruct-1M_T32768.json"), ("Qwen2.5-7B-1M\n128k, K=2048", f"{R13}/final_alloc_Qwen2.5-7B-Instruct-1M_T131072_K2048.json")]
    fig, ax = plt.subplots(figsize=(DC, 2.7)); w = 0.17; xs = range(len(srcs))
    for i, (nm, p) in enumerate(srcs):
        d = load(p)
        vals = [d["u4/planes48"]["err"], d["u4klt/planes48"]["err"], d["u4/planes64"]["err"], d["u4klt/planes64"]["err"]]
        cols = [IND2, GREEN, IND, "#1E5A40"]; offs = [-1.5, -0.5, 0.5, 1.5]
        for v, c, o in zip(vals, cols, offs): ax.bar(i + o * w, v, w, color=c)
        ax.hlines(d["dsparsity_c32_4bit"]["err"], i - 2 * w, i + 2 * w, color=GREYS[0], lw=1, ls="--")
    handles = [Patch(color=c, label=l) for c, l in zip([IND2, GREEN, IND, "#1E5A40"], ["raw, mean 48", "KLT, mean 48", "raw, mean 64", "KLT, mean 64"])] + [Line2D([], [], color=GREYS[0], lw=1, ls="--", label="Double Sparsity, 136 b")]
    ax.set_xticks(list(xs)); ax.set_xticklabels([s[0] for s in srcs], fontsize=5.6); ax.set_xlim(-0.55, len(srcs) - 0.45); ax.set_yscale("log"); ax.set_ylabel("attention-output relative error")
    ax.legend(handles=handles, frameon=False, fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.grid(axis="y", alpha=.25, which="both", lw=.4); fig.tight_layout(); save(fig, "fig_basis.pdf")

# 4. RULER ---------------------------------------------------------------------------------------------------------------
RULER_NAMES = {"dense": "dense", "exact_topk": "exact top-k", "planesL:u4:48": "Fathom 48", "planesL:u4:64": "Fathom 64", "planesK:u4:48": "Fathom 48 KLT", "planesK:u4:64": "Fathom 64 KLT",
               "ds:32:4": "Double Sparsity", "sparq:16:4": "SparQ r16", "sparq:32:4": "SparQ r32", "loki:64:4": "Loki r64", "thumb:2": "2-bit thumbnail", "landmark:8": "block landmark"}
def ruler_means(p):
    d = load(p); ts = [t for t in d if t != "_meta"]; ms = [m for m in d[ts[0]] if all(m in d[t] and d[t][m]["scores"] for t in ts)]
    # error bar = standard error of the pooled per-sample scores across all tasks
    out = {}
    for m in ms:
        sc = [s for t in ts for s in d[t][m]["scores"]]; mean = sum(d[t][m]["mean"] for t in ts) / len(ts)
        se = statistics.pstdev(sc) / math.sqrt(len(sc)); out[m] = (mean, se, d[ts[-1]][m]["mean_bits"])
    return out
def fig_ruler():
    srcs = [(f"{R13}/ruler_Qwen3-8B_T32768_K128_ctx32000.json", "Qwen3-8B, 32k, K=128"), (f"{R13}/ruler_Qwen2.5-7B-Instruct-1M_T131072_K128_ctx128000.json", "Qwen2.5-7B-1M, 128k, K=128")]
    srcs = [(p, t) for p, t in srcs if os.path.exists(p)]
    fig, axs = plt.subplots(len(srcs), 1, figsize=(SC, 2.6 * len(srcs)), squeeze=False); axs = axs[:, 0]
    for ax, (p, title) in zip(axs, srcs):
        r = ruler_means(p); ms = [m for m in r if m not in ("dense", "exact_topk")]
        xs = range(len(ms)); cols = [IND if m.startswith("planes") else (RED if m.startswith("landmark") else GREYS[1]) for m in ms]
        ax.bar(xs, [r[m][0] for m in ms], color=cols, yerr=[r[m][1] for m in ms], error_kw={"lw": .6, "capsize": 1.5})
        ax.axhline(r["dense"][0], ls="--", color="k", lw=.8, label="dense"); ax.axhline(r["exact_topk"][0], ls=":", color="k", lw=.8, label="exact top-k")
        ax.set_xticks(list(xs)); ax.set_xticklabels([f"{RULER_NAMES.get(m, m)}\n({r[m][2]:.0f} b)" for m in ms], rotation=70, ha="right", fontsize=5.4)
        lo = min(r[m][0] for m in ms); ax.set_ylim(max(0, lo - 0.15), min(1.0, max(r[m][0] for m in r) + 0.05)); ax.set_title(title, loc="left", fontsize=8); ax.grid(axis="y", alpha=.25, lw=.4)
    for ax in axs: ax.set_ylabel("mean RULER score")
    axs[-1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.75), ncol=2, fontsize=6.5)
    fig.tight_layout(); save(fig, "fig_ruler.pdf")

# 5. offload ---------------------------------------------------------------------------------------------------------------
OFF_STYLE = {"planes_mean48": ("Fathom 48", IND, "-o"), "planes_mean64": ("Fathom 64", IND2, "-o"), "chan4_r32": ("32-channel scan (SparQ r32 / DS / Loki bytes)", GREYS[0], "-s"),
             "chan4_r16": ("SparQ r16", ORANGE, "-D"), "landmark8": ("block landmark", RED, "-^"), "planes_thumb2": ("2-bit thumbnail", GREYS[2], "-v")}
def fig_offload():
    d = [r for r in load(f"{R13}/B_e2e_offload_synth_copyidx.json") if "method" in r and not r.get("oom")]
    fig, axs = plt.subplots(1, 2, figsize=(DC, 2.5), sharex=True)
    for m, (nm, c, st) in OFF_STYLE.items():
        rs = sorted([r for r in d if r["method"] == m], key=lambda r: r["ctx"])
        axs[0].plot([r["ctx"] / 1024 for r in rs], [r["step_ms"] for r in rs], st, color=c, ms=3.5, lw=1.2, label=nm)
        axs[1].plot([r["ctx"] / 1024 for r in rs], [r["step_gpu_ms"] for r in rs], st, color=c, ms=3.5, lw=1.2)
    for ax, yl in zip(axs, ["decode step, wall-clock (ms)", "decode step, GPU time (ms)"]):
        ax.set_xscale("log", base=2); ax.set_xticks([32, 256, 512, 1024]); ax.set_xticklabels(["32k", "256k", "512k", "1M"]); ax.set_xlabel("context length (tokens)"); ax.set_ylabel(yl); ax.grid(alpha=.25, which="both", lw=.4); ax.set_ylim(0)
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=6.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06)); fig.tight_layout(rect=(0, 0.1, 1, 1)); save(fig, "fig_offload.pdf")
    fig, ax = plt.subplots(figsize=(SC, 2.4))
    for m, (nm, c, st) in OFF_STYLE.items():
        rs = sorted([r for r in d if r["method"] == m], key=lambda r: r["ctx"])
        ax.plot([r["ctx"] / 1024 for r in rs], [r["pcie_GB_per_step"] for r in rs], st, color=c, ms=3.5, lw=1.2, label=nm)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks([32, 256, 512, 1024]); ax.set_xticklabels(["32k", "256k", "512k", "1M"]); ax.set_xlabel("context length (tokens)"); ax.set_ylabel("PCIe traffic per decode step (GB)")
    ax.grid(alpha=.25, which="both", lw=.4); ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2); fig.tight_layout(); save(fig, "fig_offload_bytes.pdf")

# 6. index in HBM control ----------------------------------------------------------------------------------------------------
def fig_hbm():
    host = {(r["ctx"], r["method"]): r for r in load(f"{R13}/B_e2e_offload_real_copyidx.json") if "method" in r and not r.get("oom") and r["batch"] == 1}
    hbm = {(r["ctx"], r["method"]): r for r in load(f"{R13}/B_e2e_offload_real_hbm.json") if "method" in r and not r.get("oom") and r["batch"] == 1}
    ms = ["chan4_r32", "chan4_r16", "planes_mean48", "planes_thumb2", "landmark8"]; ctxs = [32768, 65536, 131072]
    fig, ax = plt.subplots(figsize=(SC, 3.0)); w = 0.1
    for i, ctx in enumerate(ctxs):
        for j, m in enumerate(ms):
            c = OFF_STYLE[m][1]; x0 = i + (j - 2) * 2.2 * w
            ax.bar(x0 - w / 2, host[(ctx, m)]["step_gpu_ms"], w, color=c, alpha=.45); ax.bar(x0 + w / 2, hbm[(ctx, m)]["step_gpu_ms"], w, color=c)
    handles = [Patch(color=OFF_STYLE[m][1], label=OFF_STYLE[m][0]) for m in ms] + [Patch(color="grey", alpha=.45, label="index in host memory"), Patch(color="grey", label="index in HBM")]
    ax.set_xticks(range(len(ctxs))); ax.set_xticklabels(["32k", "64k", "128k"]); ax.set_xlabel("context length (tokens), batch 1"); ax.set_ylabel("decode step, GPU time (ms)")
    ax.legend(handles=handles, frameon=False, fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22)); ax.grid(axis="y", alpha=.25, lw=.4); fig.tight_layout(); save(fig, "fig_hbm.pdf")

# 7. per-layer budgets ---------------------------------------------------------------------------------------------------------
def fig_layers():
    fig, ax = plt.subplots(figsize=(SC, 3.2))
    for p, nm, c in [(f"{R13}/final_alloc_Qwen3-8B_T16384.json", "Qwen3-8B, plan calibrated at 16k", IND2), (f"{R13}/final_alloc_Qwen3-8B_T32768.json", "Qwen3-8B, plan calibrated at 32k", IND)]:
        al = load(p)["u4/layeralloc_mean48"]["alloc"]; L = sorted(int(k) for k in al)
        ax.step(L, [al[str(l)] for l in L], where="mid", color=c, lw=1.3, label=nm)
    ax.axhline(48, color=GREYS[2], ls=":", lw=.8, label="mean budget 48"); ax.set_xlabel("layer"); ax.set_ylabel("read budget (code bits / token)"); ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1); ax.grid(alpha=.25, lw=.4)
    fig.tight_layout(); save(fig, "fig_layers.pdf")

# 8. L4 kernel bandwidth ---------------------------------------------------------------------------------------------------------
def fig_kernel():
    rows_ = load(f"{R13}/B_kernel_bench_a100.json")["rows"]
    ms = [("planes_u4_mean48", "Fathom 48", IND), ("planes_u4_mean64", "Fathom 64", IND2), ("chan4_r32_loki_ds_sparq32", "4-bit 32-channel scan", GREYS[0]), ("chan4_r128_fulldim", "full 4-bit scan (128 channels)", GREYS[2])]
    fig, ax = plt.subplots(figsize=(SC, 2.4)); w = 0.2; ns = [32768, 131072]
    for i, n in enumerate(ns):
        for j, (m, nm, c) in enumerate(ms):
            v = [r["scan_GBps"] for r in rows_ if r["method"] == m and r["n"] == n and r["batch"] == 4]     # mean over the 5 measured layers (0,1,2,18,30)
            ax.bar(i + (j - 1.5) * w, sum(v) / len(v), w, color=c, label=nm if i == 0 else None)
    ax.set_xticks(range(len(ns))); ax.set_xticklabels(["32k", "128k"]); ax.set_xlabel("context length, batch 4 × 8 KV heads")
    ax.set_ylabel("achieved scan bandwidth (GB/s)"); ax.set_ylim(0, 320); ax.legend(frameon=False, fontsize=6, loc="upper right", bbox_to_anchor=(1, 0.9)); ax.grid(axis="y", alpha=.25, lw=.4); fig.tight_layout(); save(fig, "fig_kernel.pdf")

# 9. PCIe transfer bandwidth (measured on pod 2 A100, pcie_gather_bench.py) ------------------------------------------------------
def fig_pcie():
    """Parse pcie_bench.log: lines 'run <KB> KB x <R> <name> ...: <GB/s> GB/s' and the memcpy / per-run lines."""
    import re
    log = f"{R13}/pcie_bench.log"
    if not os.path.exists(log): print("skip fig_pcie (no log)"); return
    ser = {}
    for line in open(log):
        m = re.match(r"run\s+(\d+) KB x\s+\d+\s+(contiguous|scattered\(4x stride\))\s+CH=\d+ warps=\d+:\s+([\d.]+) GB/s", line)
        if m: k = "contiguous" if m.group(2) == "contiguous" else "scattered"; ser.setdefault(k, {}); ser[k][int(m.group(1))] = max(ser[k].get(int(m.group(1)), 0), float(m.group(3))); continue
        m = re.match(r"run\s+(\d+) KB\s+one cudaMemcpyAsync of \d+ MB:\s+([\d.]+) GB/s", line)
        if m: ser.setdefault("memcpy", {})[int(m.group(1))] = float(m.group(2)); continue
        m = re.match(r"run\s+(\d+) KB\s+\d+ separate copy_ calls:\s+([\d.]+) GB/s", line)
        if m: ser.setdefault("percall", {})[int(m.group(1))] = float(m.group(2))
    if not ser: print("skip fig_pcie (log has no data)"); return
    json.dump(ser, open(f"{ROOT}/paper/tables/pcie_series.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(SC, 3.0))
    for key, (lab, c, st) in {"contiguous": ("Triton gather, contiguous runs (best block size)", IND, "-o"), "scattered": ("Triton gather, runs at 4x stride", IND2, "--o"), "memcpy": ("one cudaMemcpyAsync of the same bytes", GREYS[0], "-s"), "percall": ("one copy call per run", ORANGE, "-^")}.items():
        if key in ser: kb = sorted(ser[key]); ax.plot(kb, [ser[key][k] for k in kb], st, color=c, ms=3.5, lw=1.2, label=lab)
    ax.set_xscale("log", base=2); ax.set_xticks([4, 32, 128, 512]); ax.set_xticklabels(["4 KB\n(32k, t=1)", "32 KB", "128 KB\n(1M, t=1)", "512 KB\n(1M, t=4)"], fontsize=6.5); ax.set_xlabel("run size"); ax.set_ylabel("host to GPU bandwidth (GB/s)"); ax.set_ylim(0, 30)
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=1); ax.grid(alpha=.25, which="both", lw=.4); fig.tight_layout(); save(fig, "fig_pcie.pdf")

# 10. method schematic -----------------------------------------------------------------------------------------------------------
def grid4(ax, x0, y0, cells, size=0.22, hatch=None):
    """4x4 bit-cell grid at (x0, y0); cells = set of (col, row) shaded; row 0 = MSB at top."""
    for c in range(4):
        for r in range(4):
            on = (c, r) in cells
            ax.add_patch(Rectangle((x0 + c * size, y0 + (3 - r) * size), size * .92, size * .92, facecolor=(IND if on else "#E4E4EE"), edgecolor="white", lw=.5, hatch=(hatch if on and hatch else None)))
def fig_method():
    fig, axs = plt.subplots(1, 3, figsize=(DC, 2.3), gridspec_kw={"width_ratios": [1.2, 1.6, 1.1]})
    ax = axs[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    for p in range(4):                                                       # four plane words for one channel and one 64-token block
        ax.add_patch(Rectangle((0.8, 4.3 - p * 1.0), 8.4, 0.75, facecolor=(IND if p < 2 else "#E4E4EE"), edgecolor="white"))
        ax.text(0.6, 4.65 - p, f"plane {p}" + (" (MSB)" if p == 0 else " (LSB)" if p == 3 else ""), ha="right", va="center", fontsize=6.5)
        for i in range(0, 64, 2): ax.plot([0.8 + 8.4 * i / 64] * 2, [4.3 - p * 1.0, 5.05 - p * 1.0], color="white", lw=.2)
    ax.text(5, 5.6, "one channel, one 64-token block: 4 × 8-byte words", ha="center", fontsize=6.5); ax.text(5, 0.45, "depth t = 2 reads the first two words, contiguous", ha="center", fontsize=6.5, color=IND)
    ax = axs[1]
    depths = sorted([4] * 2 + [3] * 4 + [2] * 6 + [1] * 16 + [0] * 100, reverse=True)                 # a plausible mean-48 plan: 28 active channels, 48 bits
    ax.bar(range(128), depths, width=1.0, color=[IND if d > 0 else "#E4E4EE" for d in depths])
    ax.set_xlabel("channel, sorted by importance $g_j$"); ax.set_ylabel("planes read $t_j$"); ax.set_yticks([0, 1, 2, 3, 4]); ax.set_xlim(-1, 128); ax.spines["left"].set_position(("outward", 2))
    ax.text(84, 3.6, f"illustrative plan: Σ t = {sum(depths)} code bits\n{sum(1 for d in depths if d)} active channels", ha="center", fontsize=6.5)
    ax = axs[2]; ax.set_xlim(0, 4.0); ax.set_ylim(-0.2, 1.7); ax.axis("off")
    grid4(ax, 0.1, 0.5, {(c, r) for c in (0, 1) for r in range(4)}); ax.text(0.54, 0.35, "channel\npick", ha="center", va="top", fontsize=6.5)
    grid4(ax, 1.5, 0.5, {(c, r) for c in range(4) for r in (0, 1)}); ax.text(1.94, 0.35, "uniform\ndepth", ha="center", va="top", fontsize=6.5)
    grid4(ax, 2.9, 0.5, {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (2, 0)}); ax.text(3.34, 0.35, "Fathom", ha="center", va="top", fontsize=6.5, color=IND)
    ax.text(2.0, 1.6, "columns = channels, rows = bit planes", ha="center", fontsize=6.5)
    fig.tight_layout(); save(fig, "fig_method.pdf")

# 11. toy grids ------------------------------------------------------------------------------------------------------------------
def fig_toy():
    cols = lambda cs: {(c, r) for c in cs for r in range(4)}
    items = [("Full 4-bit\n16 b", cols(range(4)), None, "✓ ✓"), ("Loki r=2\n8 b", cols((0, 1)), "////", "✓ ✗"), ("DS c=2\n8 b", cols((0, 1)), None, "✓ ✗"),
             ("SparQ r=2\n8 b", cols((0, 1)), None, "✓ ✗"), ("Thumbnail\n8 b", {(c, r) for c in range(4) for r in (0, 1)}, None, "✗ ✗"), ("Fathom\n8 b", {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (2, 0)}, None, "✓ ✓")]
    fig, ax = plt.subplots(figsize=(SC, 1.7)); ax.set_xlim(0, 6.3); ax.set_ylim(-0.1, 1.9); ax.axis("off")
    for i, (nm, cells, hatch, verdict) in enumerate(items):
        x0 = 0.15 + i * 1.03; grid4(ax, x0, 0.55, cells, size=0.2, hatch=hatch)
        ax.text(x0 + 0.4, 1.45, nm, ha="center", va="bottom", fontsize=6.2, color=(IND if nm.startswith("Fathom") else "black"))
        ax.text(x0 + 0.4, 0.3, verdict, ha="center", fontsize=7, color=(GREEN if verdict == "✓ ✓" else RED))
    ax.text(3.15, 0.02, "verdict: head A / head B top-2 recovered", ha="center", fontsize=6.2, color=GREYS[1])
    fig.tight_layout(); save(fig, "fig_toy.pdf")

if __name__ == "__main__":
    import sys, traceback
    only = sys.argv[1:]
    for f in (fig_frontier, fig_ratio, fig_basis, fig_ruler, fig_offload, fig_hbm, fig_layers, fig_kernel, fig_pcie, fig_method, fig_toy):
        if only and f.__name__ not in only: continue
        try: f()
        except Exception as e: print(f"skip {f.__name__}: {type(e).__name__}: {str(e)[:120]}")
