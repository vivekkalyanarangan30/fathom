"""Generate every table and every prose number of the paper from the result JSONs (paper/tables/*.tex, paper/macros.tex)."""
import json, os, glob, math, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R13 = f"{ROOT}/results/a100_20260913"; R14 = f"{ROOT}/results/a100_20260914"; OUT = f"{ROOT}/paper/tables"
def pick(name):
    """Newest run that has the file (the 14 Sep rerun adds the 40-bit budget; identical code otherwise)."""
    return f"{R14}/{name}" if os.path.exists(f"{R14}/{name}") else f"{R13}/{name}"
os.makedirs(OUT, exist_ok=True)
MACROS = {}

def load(p): return json.load(open(p)) if os.path.exists(p) else None
def fmt_e(e): return "$<$0.0001" if e < 5e-5 else f"{e:.4f}"
def fmt_b(b): return f"{b:.0f}"
def write(name, body):
    open(f"{OUT}/{name}.tex", "w").write(body); print("wrote", name)
def macro(name, val): MACROS[name] = val

# (model, ctx, K, file, qk_norm)
SETTINGS = [("Qwen3-8B", "16k", 256, pick("final_alloc_Qwen3-8B_T16384.json"), True),
            ("Qwen3-8B", "32k", 512, pick("final_alloc_Qwen3-8B_T32768.json"), True),
            ("Qwen3-8B", "32k", 128, pick("final_alloc_Qwen3-8B_T32768_K128.json"), True),
            ("Qwen3-4B", "16k", 256, pick("final_alloc_Qwen3-4B_T16384.json"), True),
            ("Llama-3.1-8B", "4k", 128, f"{ROOT}/results/l4/final_alloc_Meta-Llama-3.1-8B_T4096_K128.json", False),
            ("Qwen2.5-7B", "32k", 512, pick("final_alloc_Qwen2.5-7B_T32768.json"), False),
            ("Qwen2.5-7B", "32k", 128, pick("final_alloc_Qwen2.5-7B_T32768_K128.json"), False),
            ("Qwen2.5-7B-1M", "32k", 512, pick("final_alloc_Qwen2.5-7B-Instruct-1M_T32768.json"), False),
            ("Qwen2.5-7B-1M", "128k", 128, pick("final_alloc_Qwen2.5-7B-Instruct-1M_T131072.json"), False),
            ("Qwen2.5-7B-1M", "128k", 512, pick("final_alloc_Qwen2.5-7B-Instruct-1M_T131072_K512.json"), False),
            ("Qwen2.5-7B-1M", "128k", 2048, pick("final_alloc_Qwen2.5-7B-Instruct-1M_T131072_K2048.json"), False)]
HEADLINE = [s for s in SETTINGS if (s[0], s[1], s[2]) in {("Qwen3-8B", "16k", 256), ("Qwen3-8B", "32k", 512), ("Qwen3-4B", "16k", 256), ("Llama-3.1-8B", "4k", 128),
                                                          ("Qwen2.5-7B", "32k", 512), ("Qwen2.5-7B-1M", "32k", 512), ("Qwen2.5-7B-1M", "128k", 2048)}]
BUD = (24, 32, 40, 48, 64, 80, 96, 128)

def store(qk): return "u4" if qk else "u4klt"
def curve(d, st):
    """Flat-budget points [(budget, bits, err)] and per-layer points for store st."""
    flat = [(B, d[f"{st}/planes{B}"]["bits"], d[f"{st}/planes{B}"]["err"], "flat") for B in BUD if f"{st}/planes{B}" in d]
    pl = [(B, d[f"{st}/layeralloc_mean{B}"]["bits"], d[f"{st}/layeralloc_mean{B}"]["err"], "per-layer") for B in (32, 40, 48, 64, 80) if f"{st}/layeralloc_mean{B}" in d]
    return flat, pl
def match(points, target):
    """Smallest-bits point whose error is at or below target, else None."""
    ok = [p for p in points if p[2] <= target]
    return min(ok, key=lambda p: (round(p[1]), p[3] != "flat", p[2])) if ok else None

# ---------------------------------------------------------------- Table: bits to match the 136-bit scans (headline)
def tab_equalerror():
    rows = []; ds_bits = []; sq_bits = []
    for model, ctx, K, f, qk in HEADLINE:
        d = load(f)
        if d is None: rows.append(f"{model} & {ctx}/{K} & \\multicolumn{{8}}{{c}}{{(pending)}} \\\\"); continue
        flat, pl = curve(d, store(qk)); allpts = flat + pl
        ds, sq = d["dsparsity_c32_4bit"], d["sparq_r32_4bit"]
        m_ds, m_sq = match(allpts, ds["err"]), match(allpts, sq["err"])
        def cell(m):
            if m is None: return "$>$146 & -- & --"
            return f"\\textbf{{{fmt_b(m[1])}}} & {fmt_e(m[2])} & {m[3]}"
        if m_ds: ds_bits.append(m_ds[1])
        if m_sq: sq_bits.append(m_sq[1])
        basis = "raw" if qk else "KLT"
        rows.append(f"{model} & {ctx}/{K} & {basis} & {fmt_e(ds['err'])} & {cell(m_ds)} & {fmt_e(sq['err'])} & {cell(m_sq)} \\\\")
    body = "\\begin{tabular}{llllrlllrll}\n\\toprule\n & & & \\multicolumn{4}{c}{Double Sparsity $c{=}32$ (136 b)} & \\multicolumn{4}{c}{SparQ $r{=}32$ (136 b)} \\\\\n" \
           "Model & ctx/$k$ & basis & error & Fathom bits & Fathom error & plan & error & Fathom bits & Fathom error & plan \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}"
    write("tab_equalerror", body)
    if ds_bits: macro("matchDSmin", fmt_b(min(ds_bits))); macro("matchDSmax", fmt_b(max(ds_bits))); macro("matchDSratioMin", f"{136/max(ds_bits):.1f}"); macro("matchDSratioMax", f"{136/min(ds_bits):.1f}")
    if sq_bits: macro("matchSQmin", fmt_b(min(sq_bits))); macro("matchSQmax", fmt_b(max(sq_bits))); macro("matchSQratioMin", f"{136/max(sq_bits):.1f}"); macro("matchSQratioMax", f"{136/min(sq_bits):.1f}")
    macro("nSettings", str(len(HEADLINE))); macro("nSettingsMatched", str(len(ds_bits)))
    lk = []
    for model, ctx, K, f, qk in HEADLINE:
        d = load(f)
        if d is None or d["loki_r32_4bit"]["err"] > 0.01: continue
        flat, pl = curve(d, store(qk)); m = match(flat + pl, d["loki_r32_4bit"]["err"])
        if m: lk.append(m[1])
    if lk: macro("matchLokiMin", fmt_b(min(lk))); macro("matchLokiMax", fmt_b(max(lk))); macro("matchLokiN", str(len(lk)))

# ---------------------------------------------------------------- Table: full rows for Qwen3-8B at 16k and 32k
ROWS_MAIN = [("Fathom flat 40", "u4/planes40"), ("Fathom flat 48", "u4/planes48"), ("Fathom flat 64", "u4/planes64"), ("Fathom flat 80", "u4/planes80"), ("Fathom flat 128", "u4/planes128"),
             ("Fathom per-layer 40", "u4/layeralloc_mean40"), ("Fathom per-layer 48", "u4/layeralloc_mean48"), ("Fathom per-layer 64", "u4/layeralloc_mean64"),
             ("Fathom per-layer 48, plan from 16k", "u4/layeralloc_mean48@from_Qwen3-8B_T16384"), ("Fathom per-layer 64, plan from 16k", "u4/layeralloc_mean64@from_Qwen3-8B_T16384"),
             ("SparQ $r{=}16$", "sparq_r16_4bit"), ("SparQ $r{=}32$", "sparq_r32_4bit"), ("Double Sparsity $c{=}32$", "dsparsity_c32_4bit"),
             ("Loki $r{=}32$", "loki_r32_4bit"), ("Loki $r{=}64$", "loki_r64_4bit"), ("2-bit thumbnail", "thumb_2bit_all"), ("full 4-bit scan", "full_4bit_scan")]
def tab_rows(name, settings, rows=ROWS_MAIN, caption_cols=None):
    ds = [load(f) for _, _, _, f, _ in settings]
    for d, (_, _, _, f, _) in zip(ds, settings):                                   # the 14 Sep rerun did not repeat the cross-context plan rows; take them from the 13 Sep run of the same code
        old = load(f.replace(R14, R13))
        if d is not None and old is not None:
            for k in old:
                if "@from_" in k and k not in d: d[k] = old[k]
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{m} {c}/$K{{=}}{K}$}}" for (m, c, K, _, _) in settings)
    sub = " & ".join("bits & error" for _ in settings)
    lines = []
    for label, key in rows:
        cells = []
        for d in ds:
            cells.append("-- & --" if d is None or key not in d else f"{fmt_b(d[key]['bits'])} & {fmt_e(d[key]['err'])}")
        lab = f"\\textbf{{{label}}}" if label.startswith(("ours", "Fathom")) else label
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    col = "l" + "rl" * len(settings)
    write(name, f"\\begin{{tabular}}{{{col}}}\n\\toprule\n & {head} \\\\\n & {sub} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")

# ---------------------------------------------------------------- Table: selection ratio (Qwen2.5-7B-1M 128k)
def tab_kratio():
    sets = [s for s in SETTINGS if s[0] == "Qwen2.5-7B-1M" and s[1] == "128k"]
    d = load(sets[0][3]); d2 = load(sets[-1][3])
    if d: macro("kratioRawFortyEight", fmt_e(d["u4/planes48"]["err"])); macro("kratioKltFortyEight", fmt_e(d["u4klt/planes48"]["err"]))
    if d and d2:
        macro("riseSparq", f"{d['sparq_r16_4bit']['err']/d2['sparq_r16_4bit']['err']:.0f}"); macro("riseDS", f"{d['dsparsity_c32_4bit']['err']/d2['dsparsity_c32_4bit']['err']:.0f}")
        macro("riseOurs", f"{d['u4klt/planes48']['err']/d2['u4klt/planes48']['err']:.0f}"); macro("riseFull", f"{d['full_4bit_scan']['err']/max(d2['full_4bit_scan']['err'],1e-9):.0f}")
        macro("marginLowRatio", f"{d['sparq_r16_4bit']['err']/d['u4klt/planes48']['err']:.1f}"); macro("marginHighRatio", f"{d2['sparq_r16_4bit']['err']/d2['u4klt/planes48']['err']:.1f}")
    rows = [("Fathom KLT 48", "u4klt/planes48"), ("Fathom KLT 64", "u4klt/planes64"), ("Fathom KLT 80", "u4klt/planes80"), ("Fathom raw 48", "u4/planes48"), ("Fathom raw 64", "u4/planes64"),
            ("SparQ $r{=}16$", "sparq_r16_4bit"), ("SparQ $r{=}32$", "sparq_r32_4bit"), ("Double Sparsity", "dsparsity_c32_4bit"), ("Loki $r{=}32$", "loki_r32_4bit"),
            ("Loki $r{=}64$", "loki_r64_4bit"), ("2-bit thumbnail", "thumb_2bit_all"), ("full 4-bit scan", "full_4bit_scan")]
    tab_rows("tab_kratio", sets, rows)

# ---------------------------------------------------------------- Ablation tables
def tab_ablations():
    # A1 store precision, Qwen3-8B 16k
    d = load(f"{R13}/final_alloc_Qwen3-8B_T16384.json")
    if d:
        lines = []
        for B in (48, 64, 80):
            cells = [f"{fmt_b(d[f'{s}/planes{B}']['bits'])} & {fmt_e(d[f'{s}/planes{B}']['err'])}" for s in ("u4", "u8", "rd128")]
            lines.append(f"mean {B} & " + " & ".join(cells) + " \\\\")
        write("tab_store", "\\begin{tabular}{lrlrlrl}\n\\toprule\n & \\multicolumn{2}{c}{4-bit store} & \\multicolumn{2}{c}{8-bit store} & \\multicolumn{2}{c}{rate-allocated store} \\\\\n"
              "read budget & bits & error & bits & error & bits & error \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    # A2 flat vs per-layer, A3 raw vs KLT, A8 SparQ variant: one row per setting
    l2, l3, l8 = [], [], []
    for model, ctx, K, f, qk in SETTINGS:
        d = load(f)
        if d is None: continue
        st = store(qk); lab = f"{model} & {ctx}/{K}"
        c2 = []
        for B in (48, 64):
            fl, pl = d.get(f"{st}/planes{B}"), d.get(f"{st}/layeralloc_mean{B}")
            c2.append(f"{fmt_e(fl['err'])} & {fmt_e(pl['err'])}" if fl and pl else "-- & --")
        l2.append(f"{lab} & {'raw' if qk else 'KLT'} & " + " & ".join(c2) + " \\\\")
        c3 = []
        for B in (48, 64):
            r, k = d.get(f"u4/planes{B}"), d.get(f"u4klt/planes{B}")
            c3.append(f"{fmt_e(r['err'])} & {fmt_e(k['err'])}" if r and k else "-- & --")
        if "u4klt/planes48" in d: l3.append(f"{lab} & " + " & ".join(c3) + " \\\\")
        c8 = []
        for r_ in (16, 32):
            p, v = d.get(f"sparq_r{r_}_4bit"), d.get(f"sparq_r{r_}_perhead_union_variant")
            c8.append(f"{fmt_b(p['bits'])} & {fmt_e(p['err'])} & {fmt_b(v['bits'])} & {fmt_e(v['err'])}" if p and v else "-- & -- & -- & --")
        l8.append(f"{lab} & " + " & ".join(c8) + " \\\\")
    write("tab_plan", "\\begin{tabular}{lllllll}\n\\toprule\n & & & \\multicolumn{2}{c}{mean 48} & \\multicolumn{2}{c}{mean 64} \\\\\nModel & ctx/$k$ & basis & flat & per-layer & flat & per-layer \\\\\n\\midrule\n" + "\n".join(l2) + "\n\\bottomrule\n\\end{tabular}")
    write("tab_basis", "\\begin{tabular}{llllll}\n\\toprule\n & & \\multicolumn{2}{c}{mean 48} & \\multicolumn{2}{c}{mean 64} \\\\\nModel & ctx/$k$ & raw & KLT & raw & KLT \\\\\n\\midrule\n" + "\n".join(l3) + "\n\\bottomrule\n\\end{tabular}")
    write("tab_sparqvariant", "\\begin{tabular}{llrlrlrlrl}\n\\toprule\n & & \\multicolumn{4}{c}{$r{=}16$} & \\multicolumn{4}{c}{$r{=}32$} \\\\\n & & \\multicolumn{2}{c}{published rule} & \\multicolumn{2}{c}{per-head variant} & \\multicolumn{2}{c}{published rule} & \\multicolumn{2}{c}{per-head variant} \\\\\n"
          "Model & ctx/$k$ & bits & error & bits & error & bits & error & bits & error \\\\\n\\midrule\n" + "\n".join(l8) + "\n\\bottomrule\n\\end{tabular}")
    # A5 Loki rank on Qwen3-8B
    l5 = []
    for model, ctx, K, f, qk in SETTINGS:
        d = load(f)
        if d is None or model not in ("Qwen3-8B", "Llama-3.1-8B"): continue
        l5.append(f"{model} & {ctx}/{K} & {fmt_e(d['loki_r32_fp16']['err'])} & {fmt_e(d['loki_r32_4bit']['err'])} & {fmt_e(d['loki_r64_4bit']['err'])} \\\\")
    write("tab_loki", "\\begin{tabular}{lllll}\n\\toprule\nModel & ctx/$k$ & $r{=}32$ fp16 (512 b) & $r{=}32$ 4-bit (136 b) & $r{=}64$ 4-bit (272 b) \\\\\n\\midrule\n" + "\n".join(l5) + "\n\\bottomrule\n\\end{tabular}")
    # active channels per budget (headline settings)
    la = []
    for model, ctx, K, f, qk in HEADLINE:
        d = load(f)
        if d is None: continue
        st = store(qk); cells = [f"{d[f'{st}/planes{B}']['active']:.0f}" if f"{st}/planes{B}" in d else "--" for B in (32, 48, 64, 80, 128)]
        la.append(f"{model} & {ctx}/{K} & " + " & ".join(cells) + " \\\\")
    write("tab_active", "\\begin{tabular}{llrrrrr}\n\\toprule\nModel & ctx/$k$ & mean 32 & mean 48 & mean 64 & mean 80 & mean 128 \\\\\n\\midrule\n" + "\n".join(la) + "\n\\bottomrule\n\\end{tabular}")
    acts = [load(f)[f"{store(qk)}/planes48"]["active"] for _, _, _, f, qk in HEADLINE if load(f)]
    if acts: macro("activeFortyEightMin", f"{min(acts):.0f}"); macro("activeFortyEightMax", f"{max(acts):.0f}")
    b48 = [load(f)[f"{store(qk)}/planes48"]["bits"] for _, _, _, f, qk in HEADLINE if load(f)]; b64 = [load(f)[f"{store(qk)}/planes64"]["bits"] for _, _, _, f, qk in HEADLINE if load(f)]
    if b48: macro("bitsFortyEight", f"{statistics.mean(b48):.0f}"); macro("bitsSixtyFour", f"{statistics.mean(b64):.0f}")

# ---------------------------------------------------------------- RULER
def tab_ruler():
    files = [("Qwen3-8B 32k, $k{=}128$", f"{R13}/ruler_Qwen3-8B_T32768_K128_ctx32000.json"), ("Qwen2.5-7B-1M 128k, $k{=}128$", f"{R13}/ruler_Qwen2.5-7B-Instruct-1M_T131072_K128_ctx128000.json")]
    names = {"dense": "dense", "exact_topk": "exact top-$k$ oracle", "planesL:u4:48": "Fathom 48", "planesL:u4:64": "Fathom 64", "planesK:u4:48": "Fathom KLT 48", "planesK:u4:64": "Fathom KLT 64",
             "ds:32:4": "Double Sparsity $c{=}32$", "sparq:16:4": "SparQ $r{=}16$", "sparq:32:4": "SparQ $r{=}32$", "loki:64:4": "Loki $r{=}64$", "thumb:2": "2-bit thumbnail", "landmark:8": "block landmark"}
    ds = [(t, load(f)) for t, f in files]
    if all(d is None for _, d in ds): return
    def summ(d, m):
        tasks = [t for t in d if t != "_meta" and m in d[t]]
        if not tasks: return None
        sc = [s for t in tasks for s in d[t][m]["scores"]]; bits = statistics.mean(d[t][m]["mean_bits"] for t in tasks)
        mean = statistics.mean(statistics.mean(d[t][m]["scores"]) for t in tasks); se = statistics.pstdev(sc) / math.sqrt(len(sc)) if len(sc) > 1 else 0
        return bits, mean, se, len(tasks), len(sc) // len(tasks)
    lines = []
    for m, lab in names.items():
        cells = []
        for _, d in ds:
            s = summ(d, m) if d else None
            cells.append("-- & --" if s is None else f"{fmt_b(s[0]) if s[0] > 0 else '--'} & {s[1]:.3f} $\\pm$ {s[2]:.3f}")
        labb = ("\\textbf{" + lab + "}") if m.startswith("planes") else lab
        if any(c != "-- & --" for c in cells): lines.append(f"{labb} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{t}}}" for t, _ in files)
    write("tab_ruler", "\\begin{tabular}{lrlrl}\n\\toprule\n & " + head + " \\\\\nmethod & bits & score & bits & score \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    for (t, d), tag in zip(ds, ("ruler32k", "ruler128k")):
        if d is None: continue
        tasks = [x for x in d if x != "_meta"]; ms = [m for m in names if m in d[tasks[0]]]
        rows = [f"{names[m]} & " + " & ".join(f"{statistics.mean(d[x][m]['scores']):.3f}" for x in tasks) + " \\\\" for m in ms]
        write(f"tab_{tag}_tasks", "\\begin{tabular}{l" + "r" * len(tasks) + "}\n\\toprule\nmethod & " + " & ".join(x.replace("_", "\\_") for x in tasks) + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
        ns = len(d[tasks[0]][ms[0]]["scores"]); sfx = "Short" if tag == "ruler32k" else "Long"; macro("rulerN" if tag == "ruler32k" else "rulerLongN", str(ns))
        S = {m: summ(d, m) for m in ms}; scans = [m for m in ms if m not in ("dense", "exact_topk", "landmark:8")]
        orc, dn = S["exact_topk"][1], S["dense"][1]
        macro(f"rulerSE{sfx}", f"{statistics.mean(S[m][2] for m in scans):.3f}")
        macro(f"rulerScanDev{sfx}", f"{max(abs(S[m][1] - orc) for m in scans):.3f}")
        macro(f"rulerOracleGap{sfx}", f"{dn - orc:.3f}")
        if "landmark:8" in S: macro(f"rulerLandmarkGap{sfx}", f"{orc - S['landmark:8'][1]:.3f}")
        worst = max(scans, key=lambda m: abs(S[m][1] - orc)); print(f"RULER {tag}: oracle {orc:.3f} dense {dn:.3f}; scans within {max(abs(S[m][1]-orc) for m in scans):.3f} (worst {worst}); SE ~{statistics.mean(S[m][2] for m in scans):.3f}; ours48 {S.get('planesL:u4:48', (0,0))[1]:.3f} ours64 {S.get('planesL:u4:64', (0,0))[1]:.3f}")
        for m in scans:
            if abs(S[m][1] - orc) > 2 * S[m][2]: print(f"  NOTE {tag}: {m} differs from oracle by more than 2 SE ({S[m][1]:.3f} vs {orc:.3f}, SE {S[m][2]:.3f})")

# ---------------------------------------------------------------- Offload timing
OFF_NAMES = {"planes_mean40": "Fathom 40", "planes_mean48": "Fathom 48", "planes_mean64": "Fathom 64", "chan4_r32": "32-channel scan", "chan4_r16": "SparQ $r{=}16$", "landmark8": "block landmark", "planes_thumb2": "2-bit thumbnail", "dense_offload": "dense (all rows)"}
def offrows(f):
    d = load(f)
    if d is None: return None
    out = {}
    for r in d:
        out[(r["ctx"], r["batch"], r["method"])] = r
    return out
def bits_of(r): return r["scan_MB_per_step"] * 8e6 / (r["ctx"] * 288 / 36 * 36) if r["ctx"] else 0   # per token per KV head per layer: MB*8e6 / (ctx * 8 heads * 36 layers)
def tab_offload():
    o = offrows(pick("B_e2e_offload_synth_copyidx.json"))
    if o is None: return
    ctxs = [262144, 524288, 1048576]; meths = [m for m in ["planes_mean40", "planes_mean48", "planes_mean64", "chan4_r16", "chan4_r32", "landmark8", "planes_thumb2"] if any((c, 1, m) in o for c in ctxs)]
    ref = {c: o.get((c, 1, "planes_mean48")) for c in ctxs}
    lines = []
    best_gpu = {c: min(o[(c, 1, m)]["step_gpu_ms"] for m in meths if (c, 1, m) in o) for c in ctxs}
    best_gb = min(o[(1048576, 1, m)]["pcie_GB_per_step"] for m in meths if (1048576, 1, m) in o)
    for m in meths:
        cells = []
        for c in ctxs:
            r = o.get((c, 1, m))
            g = "" if r is None else (f"\\textbf{{{r['step_gpu_ms']:.0f}}}" if r["step_gpu_ms"] <= best_gpu[c] * 1.01 else f"{r['step_gpu_ms']:.0f}")
            cells.append("-- & --" if r is None else f"{r['step_ms']:.0f} & {g}")
        r1 = o.get((1048576, 1, m)); rr = ref[1048576]
        gb = "" if r1 is None else (f"\\textbf{{{r1['pcie_GB_per_step']:.2f}}}" if r1["pcie_GB_per_step"] <= best_gb * 1.001 else f"{r1['pcie_GB_per_step']:.2f}")
        tail = "-- & -- & --" if r1 is None or rr is None else f"{gb} & {r1['step_ms']/rr['step_ms']:.2f} & {r1['step_gpu_ms']/rr['step_gpu_ms']:.2f}"
        b = next((o[(c, 1, m)] for c in ctxs if (c, 1, m) in o), None)
        bits = f"{b['scan_MB_per_step']*8e6/(b['ctx']*8*36):.0f}" if b else "--"
        nm = f"\\textbf{{{OFF_NAMES[m]}}}" if m.startswith("planes_mean") else OFF_NAMES[m]
        lines.append(f"{nm} & {bits} & " + " & ".join(cells) + f" & {tail} \\\\")
        if r1 and rr and m != "planes_mean48": macro("ratioGPU" + m.replace("_", ""), f"{r1['step_gpu_ms']/rr['step_gpu_ms']:.2f}"); macro("ratioWall" + m.replace("_", ""), f"{r1['step_ms']/rr['step_ms']:.2f}")
    write("tab_offload", "\\begin{tabular}{lrrrrrrrrrr}\n\\toprule\n & & \\multicolumn{2}{c}{256k} & \\multicolumn{2}{c}{512k} & \\multicolumn{2}{c}{1M} & \\multicolumn{3}{c}{1M relative to Fathom 48} \\\\\n"
          "method & bits & wall & GPU & wall & GPU & wall & GPU & GB/step & wall & GPU \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    r256, o256 = o.get((262144, 1, "planes_mean48")), o.get((262144, 1, "chan4_r32"))
    if r256 and o256: macro("ratioGPUchanAt256k", f"{o256['step_gpu_ms']/r256['step_gpu_ms']:.2f}")
    rows256 = [o[(262144, 1, m)]["rows_MB_per_step"] for m in meths if (262144, 1, m) in o]
    if rows256: macro("rowsMBmin", f"{min(rows256):.0f}"); macro("rowsMBmax", f"{max(rows256):.0f}")
    b2 = offrows(pick("B_e2e_offload_synth_b2_copyidx.json"))
    if b2:
        for c, tag in ((262144, "Short"), (524288, "Long")):
            a2, c2 = b2.get((c, 2, "planes_mean48")), b2.get((c, 2, "chan4_r32"))
            if a2 and c2: macro(f"bTwoRatio{tag}", f"{c2['step_gpu_ms']/a2['step_gpu_ms']:.2f}")
    a, b = o.get((1048576, 1, "planes_mean48")), o.get((1048576, 1, "chan4_r16"))
    if a and b: macro("sparqSixteenBytesMorePct", f"{100*(b['scan_MB_per_step']/a['scan_MB_per_step']-1):.0f}"); macro("oursGBstep", f"{a['pcie_GB_per_step']:.2f}"); macro("rowsMBstep", f"{a['rows_MB_per_step']:.0f}")
    if a and b: macro("fortyEightBytesFewerPct", f"{100*(1-a['scan_MB_per_step']/b['scan_MB_per_step']):.0f}"); macro("wallGapSparqPct", f"{100*(a['step_ms']/b['step_ms']-1):.0f}")
    f40, c32 = o.get((1048576, 1, "planes_mean40")), o.get((1048576, 1, "chan4_r32"))
    if f40 and b: macro("ratioSparqSixteenOverForty", f"{b['step_gpu_ms']/f40['step_gpu_ms']:.2f}"); macro("fortyBytesFewerPct", f"{100*(1-f40['scan_MB_per_step']/b['scan_MB_per_step']):.0f}")
    if f40 and c32: macro("ratioChanOverForty", f"{c32['step_gpu_ms']/f40['step_gpu_ms']:.2f}")
    # wall minus GPU per method at 1M (issue cost) -> macros
    for m in meths:
        r1 = o.get((1048576, 1, m))
        if r1: macro("issue" + m.replace("_", ""), f"{r1['step_ms']-r1['step_gpu_ms']:.0f}")
def tab_real():
    host, hbm = offrows(pick("B_e2e_offload_real_copyidx.json")), offrows(pick("B_e2e_offload_real_hbm.json"))
    if host is None and hbm is None: return
    ctxs = [32768, 65536, 131072]; meths = [m for m in ["planes_mean40", "planes_mean48", "planes_mean64", "chan4_r16", "chan4_r32", "landmark8", "planes_thumb2", "dense_offload"] if any((c, 1, m) in (host or hbm) for c in ctxs)]
    lines = []
    for m in meths:
        cells = []
        for o in (host, hbm):
            for c in ctxs:
                r = o.get((c, 1, m)) if o else None
                cells.append("--" if r is None else f"{r['step_gpu_ms']:.0f}")
        lines.append(f"{OFF_NAMES[m]} & " + " & ".join(cells) + " \\\\")
    sparse = ["planes_mean48", "planes_mean64", "chan4_r16", "chan4_r32", "landmark8", "planes_thumb2"]
    if hbm:
        v = [hbm[(131072, 1, m)]["step_gpu_ms"] for m in sparse if (131072, 1, m) in hbm]
        if v: macro("hbmTieMin", f"{min(v):.0f}"); macro("hbmTieMax", f"{max(v):.0f}")
        sc = [hbm[(131072, 1, m)]["step_gpu_ms"] for m in ("planes_mean48", "planes_mean64", "chan4_r16", "chan4_r32") if (131072, 1, m) in hbm]
        if sc: macro("hbmScanMin", f"{min(sc):.0f}"); macro("hbmScanMax", f"{max(sc):.0f}")
        if (131072, 1, "landmark8") in hbm: macro("hbmLandmark", f"{hbm[(131072, 1, 'landmark8')]['step_gpu_ms']:.0f}")
        if (131072, 1, "planes_thumb2") in hbm: macro("hbmThumb", f"{hbm[(131072, 1, 'planes_thumb2')]['step_gpu_ms']:.0f}")
        v32 = [hbm[(32768, 1, m)]["step_gpu_ms"] for m in sparse if (32768, 1, m) in hbm]
        if v32: macro("hbmTieMinShort", f"{min(v32):.0f}"); macro("hbmTieMaxShort", f"{max(v32):.0f}")
    if host and (131072, 1, "planes_mean48") in host and (131072, 1, "chan4_r32") in host:
        macro("realRatio128k", f"{host[(131072, 1, 'chan4_r32')]['step_gpu_ms']/host[(131072, 1, 'planes_mean48')]['step_gpu_ms']:.2f}")
        macro("realRatio128kSparq16", f"{host[(131072, 1, 'chan4_r16')]['step_gpu_ms']/host[(131072, 1, 'planes_mean48')]['step_gpu_ms']:.2f}")
    write("tab_real", "\\begin{tabular}{lrrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{index in host memory} & \\multicolumn{3}{c}{index in HBM} \\\\\nmethod & 32k & 64k & 128k & 32k & 64k & 128k \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")

def tab_pcie():
    p = f"{OUT}/pcie_series.json"
    if not os.path.exists(p): return
    ser = json.load(open(p))
    for k, nm in (("contiguous", "pcieGather"), ("memcpy", "pcieMemcpy"), ("percall", "pciePercall")):
        if k in ser: v = list(ser[k].values()); macro(nm + "Min", f"{min(v):.1f}"); macro(nm + "Max", f"{max(v):.1f}")
    import re
    worst = [float(m.group(1)) for line in open(f"{R13}/pcie_bench.log") for m in [re.match(r"run\s+4 KB x\s+\d+\s+contiguous\s+CH=\d+ warps=\d+:\s+([\d.]+) GB/s", line)] if m]
    if worst: macro("pcieGatherWorst", f"{min(worst):.1f}")

def tab_headtohead():
    rows = []; ratios48 = []; ratios40 = []; wins40 = 0; n = 0
    for model, ctx, K, f, qk in HEADLINE:
        d = load(f)
        if d is None: continue
        st = store(qk); s16 = d["sparq_r16_4bit"]["err"]; n += 1
        b = lambda x, win: (f"\\textbf{{{fmt_e(x)}}}" if win else fmt_e(x))
        cells = []
        for B in (40, 48):
            fl, pl = d.get(f"{st}/planes{B}"), d.get(f"{st}/layeralloc_mean{B}")
            if fl is None: cells.append("-- & -- & --"); continue
            best = min(fl["err"], pl["err"] if pl else fl["err"]); (ratios40 if B == 40 else ratios48).append(s16 / best)
            if B == 40: wins40 += best < s16
            cells.append(f"{b(fl['err'], fl['err'] < s16)} & {b(pl['err'], pl['err'] < s16) if pl else '--'} & {s16/best:.1f}$\\times$")
        rows.append(f"{model} & {ctx}/{K} & {fmt_e(s16)} & " + " & ".join(cells) + " \\\\")
        if (model, ctx, K) == ("Qwen2.5-7B-1M", "128k", 2048) and ratios48: macro("hthRatioLong", f"{ratios48[-1]:.1f}")
    write("tab_headtohead", "\\begin{tabular}{llrrrrrrr}\n\\toprule\n & & SparQ $r{=}16$ & \\multicolumn{3}{c}{Fathom, mean 40 ($\\approx$47 bits)} & \\multicolumn{3}{c}{Fathom, mean 48 ($\\approx$56 bits)} \\\\\nModel & ctx/$k$ & 68 bits & flat & per-layer & ratio & flat & per-layer & ratio \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    if ratios48: macro("hthRatioMin", f"{min(ratios48):.1f}"); macro("hthRatioMax", f"{max(ratios48):.1f}")
    if ratios40: macro("hthFortyRatioMin", f"{min(ratios40):.1f}"); macro("hthFortyRatioMax", f"{max(ratios40):.1f}"); macro("hthFortyWins", str(wins40)); macro("hthFortyN", str(n))

def tab_breakdown():
    """GPU time per step by component at 1M, from the profiler traces of the host-index run."""
    p = f"{R14}/trace_breakdown_host.json"
    if not os.path.exists(p): return
    d = json.load(open(p)); comps = ["scan transfer", "scan kernel", "top-k", "row fetch", "weights and attention GEMMs", "other", "total"]
    names = [("planes_mean40", "Fathom 40"), ("planes_mean48", "Fathom 48"), ("planes_mean64", "Fathom 64"), ("chan4_r16", "SparQ $r{=}16$"), ("chan4_r32", "32-channel scan"), ("landmark8", "block landmark"), ("planes_thumb2", "2-bit thumbnail")]
    lines = []
    for m, lab in names:
        k = f"{m}@1048576x1"
        if k not in d: continue
        v = d[k]; scan = v.get("scan transfer", 0) + v.get("scan kernel", 0)
        lines.append((("\\textbf{" + lab + "}") if m.startswith("planes_mean") else lab) + " & " + " & ".join(f"{v.get(c, 0):.0f}" for c in comps[:-1]) + f" & {scan:.0f} & {v['total']:.0f} \\\\")
        macro("scanCost" + m.replace("_", ""), f"{scan:.0f}")
        if m in ("planes_mean48", "chan4_r32"): macro("sharedPct" + ("Ours" if m == "planes_mean48" else "Chan"), f"{100*(v.get('top-k', 0) + v.get('row fetch', 0) + v.get('weights and attention GEMMs', 0))/v['total']:.0f}")
    write("tab_breakdown", "\\begin{tabular}{lrrrrrrrr}\n\\toprule\nmethod & scan transfer & scan kernel & top-$k$ & row fetch & GEMMs & other & scan total & step \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")

AGENT_TAGS = {"dense": "dense", "exact_topk": "exacttopk", "planesK:u4:40": "planesKforty", "planesK:u4:48": "planesK", "planesK:u4:64": "planesKsixtyfour", "planesK:u4:80": "planesKeighty",
              "ds:32:4": "ds", "sparq:16:4": "sparq", "sparq:32:4": "sparqthirtytwo", "landmark:8": "landmark",
              "planesA:u4:48": "planesAfortyeight", "planesA:u4:64": "planesAsixtyfour", "planesA:u4:80": "planesAeighty",
              "planesS:u4:48": "planesSfortyeight", "planesS:u4:64": "planesSsixtyfour", "planesS:u4:80": "planesSeighty", "dsA:32:4": "dsA"}
AGENT_NAMES = {"dense": "dense", "exact_topk": "exact top-$k$ oracle", "planesK:u4:40": "Fathom 40", "planesK:u4:48": "Fathom 48", "planesK:u4:64": "Fathom 64", "planesK:u4:80": "Fathom 80",
               "ds:32:4": "Double Sparsity $c{=}32$", "sparq:16:4": "SparQ $r{=}16$", "sparq:32:4": "SparQ $r{=}32$", "landmark:8": "block landmark",
               "planesA:u4:48": "Fathom 48, agent-calibrated", "planesA:u4:64": "Fathom 64, agent-calibrated", "planesA:u4:80": "Fathom 80, agent-calibrated",
               "planesS:u4:48": "Fathom 48, session-calibrated", "planesS:u4:64": "Fathom 64, session-calibrated", "planesS:u4:80": "Fathom 80, session-calibrated", "dsA:32:4": "Double Sparsity $c{=}32$, agent-calibrated"}
MAIN_ORDER = ["exact_topk", "planesK:u4:40", "planesK:u4:48", "planesK:u4:64", "planesK:u4:80", "ds:32:4", "sparq:16:4", "sparq:32:4", "landmark:8"]
CALIB_ORDER = ["planesK:u4:48", "planesA:u4:48", "planesS:u4:48", "planesK:u4:64", "planesA:u4:64", "planesS:u4:64", "planesK:u4:80", "planesA:u4:80", "planesS:u4:80", "ds:32:4", "dsA:32:4", "sparq:32:4"]
def tab_agent():
    for K, sfx, name in ((512, "", "tab_agent"), (2048, "Big", "tab_agent_kbig")): agent_one(K, sfx, name)
def agent_sessions(K):
    """Sessions of the main agent run, with the methods of the calibration run merged in (same sessions, checked by instance id and oracle step)."""
    p = f"{R14}/agent_Qwen2.5-7B-Instruct-1M_T131072_K{K}_ctx100000.json"
    if not os.path.exists(p): return None
    S = json.load(open(p))["sessions"]; p2 = f"{R14}/agent_calib_K{K}_ctx100000.json"
    if os.path.exists(p2):
        S2 = json.load(open(p2))["sessions"]; same = 0
        for a, b in zip(S, S2):
            assert a["meta"]["instance_id"] == b["meta"]["instance_id"], (a["meta"], b["meta"])
            same += a["methods"]["exact_topk"]["gen"] == b["methods"]["exact_topk"]["gen"]
            for m, v in b["methods"].items():
                if m not in ("dense", "exact_topk"): a["methods"][m] = v
        macro("agentCalibN", str(len(S2))); macro("agentCalibOracleSame", str(same))
        S = S[:len(S2)] if len(S2) < len(S) else S
    return S
def agent_one(K, sfx, name):
    S = agent_sessions(K)
    if S is None: return
    import difflib
    def sim(a, b): return difflib.SequenceMatcher(None, a.split(), b.split(), autojunk=False).ratio()
    def pw(a, b):
        a, b = a.split(), b.split(); k = 0
        for x, y in zip(a, b):
            if x != y: break
            k += 1
        return k / max(1, len(b))
    ms = list(S[0]["methods"].keys()); n = len(S); vals = {}; row = {}
    for m in ms:
        so = [sim(s["methods"][m]["gen"], s["methods"]["exact_topk"]["gen"]) for s in S]; po = [pw(s["methods"][m]["gen"], s["methods"]["exact_topk"]["gen"]) for s in S]
        sd = [sim(s["methods"][m]["gen"], s["methods"]["dense"]["gen"]) for s in S]; vals[m] = so
        mso, seo = statistics.mean(so), statistics.pstdev(so) / math.sqrt(n); bits = statistics.mean(x["bits"] for x in (s["methods"][m] for s in S))
        lab = ("\\textbf{" + AGENT_NAMES.get(m, m) + "}") if m.startswith("planes") else AGENT_NAMES.get(m, m)
        cell = f"\\textbf{{{mso:.2f}}} $\\pm$ {seo:.2f}" if m.startswith("planes") else f"{mso:.2f} $\\pm$ {seo:.2f}"
        row[m] = f"{lab} & {fmt_b(bits) if bits > 0 else '--'} & {cell} \\\\"
        tag = AGENT_TAGS.get(m)
        if tag: macro("agentOsim" + sfx + tag, f"{mso:.2f}"); macro("agentOpre" + sfx + tag, f"{statistics.mean(po):.2f}"); macro("agentDsim" + sfx + tag, f"{statistics.mean(sd):.2f}")
    pairs = (("planesK:u4:48", "sparq:16:4", "Sparq"), ("planesK:u4:48", "landmark:8", "Landmark"), ("planesK:u4:80", "sparq:32:4", "EightySparqThirtyTwo"), ("planesK:u4:64", "sparq:32:4", "SixtyFourSparqThirtyTwo"),
             ("planesS:u4:80", "sparq:32:4", "SelfEightySparqThirtyTwo"), ("planesA:u4:80", "sparq:32:4", "AgentEightySparqThirtyTwo"), ("planesS:u4:48", "planesK:u4:48", "SelfFortyEightWiki"),
             ("planesA:u4:48", "planesK:u4:48", "AgentFortyEightWiki"), ("planesA:u4:64", "planesK:u4:64", "AgentSixtyFourWiki"), ("planesA:u4:80", "planesK:u4:80", "AgentEightyWiki"), ("planesS:u4:64", "planesK:u4:64", "SelfSixtyFourWiki"), ("planesS:u4:80", "planesK:u4:80", "SelfEightyWiki"), ("dsA:32:4", "ds:32:4", "DsAgentWiki"))
    for a, b, nm in pairs:
        if a in vals and b in vals:
            dv = [x - y for x, y in zip(vals[a], vals[b])]; macro("agentPaired" + sfx + nm, f"{statistics.mean(dv):+.2f}"); macro("agentPairedSE" + sfx + nm, f"{statistics.pstdev(dv)/math.sqrt(n):.2f}"); macro("agentWins" + sfx + nm, str(sum(x > 0 for x in dv)))
    macro("agentN" + sfx, str(n)); macro("agentCtxMean" + sfx, f"{statistics.mean(s['n_tokens'] for s in S)/1000:.0f}")
    hdr = "\\begin{tabular}{lrl}\n\\toprule\nmethod & bits & step agreement with exact top-$k$, mean $\\pm$ s.e. \\\\\n\\midrule\n"
    write(name, hdr + "\n".join(row[m] for m in MAIN_ORDER if m in row) + "\n\\bottomrule\n\\end{tabular}")
    if sfx == "" and any(m in row for m in CALIB_ORDER if m.startswith(("planesA", "planesS"))):
        write("tab_agent_calib", hdr + "\n".join(row[m] for m in CALIB_ORDER if m in row) + "\n\\bottomrule\n\\end{tabular}")

def tab_calib_fid():
    """Attention error on agent-transcript keys (128k, Qwen2.5-7B-1M): Fathom and Double Sparsity calibrated on Wikitext versus on agent transcripts; SparQ needs none."""
    files = {(K, c): load(f"{R14}/final_alloc_{c}_on_agent_K{K}.json") for K in (512, 2048) for c in ("wikical", "agentcal")}
    if not all(files.values()): return
    rows = [("Fathom KLT 48", "u4klt/planes48"), ("Fathom KLT 64", "u4klt/planes64"), ("Fathom KLT 80", "u4klt/planes80"), ("SparQ $r{=}16$", "sparq_r16_4bit"), ("SparQ $r{=}32$", "sparq_r32_4bit"), ("Double Sparsity", "dsparsity_c32_4bit")]
    lines = []
    for lab, key in rows:
        cells = [f"{fmt_e(files[(K, c)][key]['err'])}" for K in (512, 2048) for c in ("wikical", "agentcal")]
        b = fmt_b(files[(512, "wikical")][key]["bits"]); lines.append((("\\textbf{" + lab + "}") if lab.startswith("Fathom") else lab) + f" & {b} & " + " & ".join(cells) + " \\\\")
    for K, kn in ((512, "FiveTwelve"), (2048, "TwoK")):
        for key, mn in (("u4klt/planes48", "FortyEight"), ("u4klt/planes64", "SixtyFour"), ("u4klt/planes80", "Eighty")):
            macro(f"calibWiki{mn}{kn}", fmt_e(files[(K, "wikical")][key]["err"])); macro(f"calibAgent{mn}{kn}", fmt_e(files[(K, "agentcal")][key]["err"]))
        macro(f"calibSparqThirtyTwo{kn}", fmt_e(files[(K, "wikical")]["sparq_r32_4bit"]["err"])); macro(f"calibDsWiki{kn}", fmt_e(files[(K, "wikical")]["dsparsity_c32_4bit"]["err"])); macro(f"calibDsAgent{kn}", fmt_e(files[(K, "agentcal")]["dsparsity_c32_4bit"]["err"]))
    write("tab_calib_fid", "\\begin{tabular}{lrllll}\n\\toprule\n & & \\multicolumn{2}{c}{$k{=}512$ (0.4\\%)} & \\multicolumn{2}{c}{$k{=}2048$ (1.6\\%)} \\\\\n"
          "method & bits & Wikitext calib. & agent calib. & Wikitext calib. & agent calib. \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")

def checks():
    """Claims made in prose that must hold in the data; print loudly if not."""
    flat_wins, any_wins, n = 0, 0, 0
    for model, ctx, K, f, qk in HEADLINE:
        d = load(f)
        if d is None: continue
        st = store(qk); o_flat = d[f"{st}/planes48"]["err"]; o_pl = d[f"{st}/layeralloc_mean48"]["err"]; s16 = d["sparq_r16_4bit"]["err"]; n += 1
        flat_wins += s16 > o_flat; any_wins += s16 > min(o_flat, o_pl)
        print(f"check SparQ16(68b) {s16:.4f} vs ours56 flat {o_flat:.4f} per-layer {o_pl:.4f} on {model} {ctx}/{K}: {'OK' if s16 > min(o_flat, o_pl) else 'VIOLATED'}")
    macro("sparqSixteenFlatWins", str(flat_wins)); macro("sparqSixteenBeatenAll", "yes" if any_wins == n else "NO")
    tm = []; unmatched = []
    for model, ctx, K, f, qk in HEADLINE:
        d = load(f)
        if d is None: continue
        flat, pl = curve(d, store(qk)); m = match(flat + pl, d["thumb_2bit_all"]["err"])
        (tm.append(m[1]) if m else unmatched.append(f"{model} {ctx}/{K}"))
    if tm: macro("thumbMatchMin", fmt_b(min(tm))); macro("thumbMatchMax", fmt_b(max(tm))); macro("thumbMatchedN", str(len(tm)))
    print("thumbnail unmatched below 146 bits:", unmatched)

def tab_kernel():
    d = load(f"{R13}/B_kernel_bench_a100.json")
    if d is None: return
    rows = d["rows"]; names = [("planes_u4_mean48", "Fathom 48"), ("planes_u4_mean64", "Fathom 64"), ("chan4_r16_sparq16", "SparQ $r{=}16$ (16 channels)"), ("chan4_r32_loki_ds_sparq32", "32-channel scan"), ("chan4_r128_fulldim", "full 4-bit scan")]
    lines = []
    for m, lab in names:
        cells = []
        for n in (32768, 131072):
            rs = [r for r in rows if r["method"] == m and r["n"] == n and r["batch"] == 4]
            if not rs: cells.append("-- & -- & --"); continue
            cells.append(f"{statistics.mean(r['bits_per_token'] for r in rs):.0f} & {statistics.mean(r['scan_ms'] for r in rs):.3f} & {statistics.mean(r['scan_GBps'] for r in rs):.0f}")
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    write("tab_kernel", "\\begin{tabular}{lrrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{32k tokens} & \\multicolumn{3}{c}{128k tokens} \\\\\nmethod & bits & scan ms & GB/s & bits & scan ms & GB/s \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    def mean_scan(m, n): 
        rs = [r for r in rows if r["method"] == m and r["n"] == n and r["batch"] == 4]; return statistics.mean(r["scan_ms"] for r in rs) if rs else None
    for n, tag in ((32768, "32k"), (131072, "128k")):
        a, b = mean_scan("planes_u4_mean48", n), mean_scan("chan4_r32_loki_ds_sparq32", n)
        if a and b: macro(f"kernelRatio{tag}", f"{b/a:.2f}")
        sh = [r["scan_share"] for r in rows if r["method"] == "chan4_r32_loki_ds_sparq32" and r["n"] == n and r["batch"] == 4]
        if sh: macro(f"scanShare{tag}", f"{100*statistics.mean(sh):.0f}")
        sho = [r["scan_share"] for r in rows if r["method"] == "planes_u4_mean48" and r["n"] == n and r["batch"] == 4]
        if sho: macro(f"scanShareOurs{tag}", f"{100*statistics.mean(sho):.0f}")
    bo = [r["bits_per_token"] for r in rows if r["method"] == "planes_u4_mean48" and r["batch"] == 4]; bc = [r["bits_per_token"] for r in rows if r["method"] == "chan4_r32_loki_ds_sparq32" and r["batch"] == 4]
    if bo and bc: macro("kernelBytesFewerPct", f"{100*(1-statistics.mean(bo)/statistics.mean(bc)):.0f}")

if __name__ == "__main__":
    for k in ("rulerN", "rulerLongN", "realRatio128k", "realRatio128kSparq16", "hbmTieMin", "hbmTieMax", "hbmTieMinShort", "hbmTieMaxShort", "kernelRatio32k", "kernelRatio128k", "scanShare32k", "scanShare128k",
              "sparqSixteenBytesMorePct", "oursGBstep", "rowsMBstep", "ratioGPUchanAt256k", "pcieGatherMin", "pcieGatherMax", "pcieMemcpyMin", "pcieMemcpyMax", "pciePercallMin", "pciePercallMax", "pcieGatherWorst", "scanShareOurs32k", "scanShareOurs128k", "kernelBytesFewerPct", "rowsMBmin", "rowsMBmax", "bTwoRatioShort", "bTwoRatioLong", "hbmScanMin", "hbmScanMax", "hbmLandmark", "hbmThumb", "kratioRawFortyEight", "kratioKltFortyEight", "hthRatioMin", "hthRatioMax", "rulerSEShort", "rulerSELong", "rulerScanDevShort", "rulerScanDevLong", "rulerOracleGapShort", "rulerOracleGapLong", "rulerLandmarkGapShort", "rulerLandmarkGapLong") + tuple(f"{a}{m}" for a in ("ratioGPU", "ratioWall", "issue") for m in ("planesmean48", "planesmean64", "chan4r16", "chan4r32", "landmark8", "planesthumb2")) + ("hthFortyRatioMin", "hthFortyRatioMax", "hthFortyWins", "hthFortyN", "agentN", "agentCtxMean", "ratioGPUplanesmean40", "ratioWallplanesmean40", "issueplanesmean40", "ratioSparqSixteenOverForty", "ratioChanOverForty", "fortyBytesFewerPct") + ("agentNBig", "agentCtxMeanBig", "fortyEightBytesFewerPct", "wallGapSparqPct", "sharedPctOurs", "sharedPctChan", "matchLokiMin", "matchLokiMax", "matchLokiN", "riseSparq", "riseDS", "riseOurs", "riseFull", "marginLowRatio", "marginHighRatio", "hthRatioLong") + tuple(p + sf + q for p in ("agentPaired", "agentPairedSE", "agentWins") for sf in ("", "Big") for q in ("Sparq", "Landmark")) + tuple(p + sf + v for p in ("agentAgree", "agentCall", "agentSim", "agentMed", "agentOsim", "agentOpre", "agentDsim") for sf in ("", "Big") for v in ("dense", "exacttopk", "planesKforty", "planesK", "planesKsixtyfour", "ds", "sparq", "sparqthirtytwo", "landmark")):
        macro(k, "--")
    tab_equalerror()
    tab_rows("tab_qwen3_8b", [s for s in SETTINGS if s[0] == "Qwen3-8B" and s[2] in (256, 512)])
    tab_rows("tab_others", [s for s in SETTINGS if (s[0], s[2]) in (("Qwen3-4B", 256), ("Llama-3.1-8B", 128), ("Qwen2.5-7B", 512), ("Qwen2.5-7B-1M", 512)) and s[1] != "128k"],
             rows=[(l, k) for l, k in ROWS_MAIN if "per-layer" not in l] + [("Fathom KLT 40", "u4klt/planes40"), ("Fathom KLT 48", "u4klt/planes48"), ("Fathom KLT 64", "u4klt/planes64"), ("Fathom KLT 80", "u4klt/planes80")])
    tab_kratio(); tab_ablations(); tab_ruler(); tab_offload(); tab_real(); tab_kernel(); tab_pcie(); tab_headtohead(); tab_breakdown(); tab_agent(); tab_calib_fid(); checks()
    TR = str.maketrans({"0": "zero", "1": "one", "2": "two", "3": "three", "4": "", "5": "five", "6": "six", "7": "seven", "8": "", "9": "nine"})
    def texname(k):
        k = k.replace("chan4r32", "chanr").replace("chan4r16", "chanrsixteen").replace("planesmean40", "planesmeanforty").replace("planesmean48", "planesmean").replace("planesmean64", "planesmeansixtyfour").replace("landmark8", "landmark").replace("planesthumb2", "planesthumb").replace("ratioGPUchanAt256k", "ratioGPUchanAt").replace("realRatio128kSparq16", "realRatioSparq").replace("realRatio128k", "realRatio").replace("kernelRatio32k", "kernelRatioShort").replace("kernelRatio128k", "kernelRatioLong").replace("scanShareOurs32k", "scanShareOursShort").replace("scanShareOurs128k", "scanShareOursLong").replace("scanShare32k", "scanShareShort").replace("scanShare128k", "scanShareLong")
        assert k.isalpha(), k; return k
    open(f"{ROOT}/paper/macros.tex", "w").write("".join(f"\\newcommand{{\\{texname(k)}}}{{{v}}}\n" for k, v in sorted(MACROS.items())))
    print("macros:", MACROS)
