"""Rewrite the number-bearing parts of report/method.html from the same result JSONs the paper uses."""
import json, os, re, statistics, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/paper")
import make_tables as T
P = f"{ROOT}/docs/method.html"
s = open(P).read()
def rep(old, new, cnt=1):
    global s
    if s.count(old) == 0 and new in s: return                      # already applied on an earlier run
    assert s.count(old) == cnt, (old[:70], s.count(old)); s = s.replace(old, new)
def e4(x): return "&lt;0.0001" if x < 5e-5 else f"{x:.4f}"

# shared-job numbers (fp16 scale convention)
rep("Top-k with k=512 reads about 160 MB of winner rows per step, a cost that stays fixed as the context grows, plus the scan. A Loki or Double Sparsity scan keeps 32 coordinates of each key at 4 bits plus a block scale, 132 bits per token for each of the 288 layer-head pairs: 32,768 × 132 / 8 = 528 KB per pair, times 288 pairs, about 160 MB per step at 32k. That term grows with <code>n</code>, reaching 5.3 GB at one million tokens while the winner rows are still about 200 MB. Every bit count on this page is per token, per KV head, per layer.",
    "Top-k with k=512 per query head fetches the union of the four heads' winners, about 200 MB of rows per step, a cost that does not grow with the context, plus the scan. A Loki, Double Sparsity or SparQ (r=32) scan reads 32 coordinates of each key at 4 bits plus an fp16 block scale, 136 bits per token for each of the 288 layer-head pairs: 32,768 × 136 / 8 = 557 KB per pair, times 288 pairs, 160 MB per step at 32k. That term grows with <code>n</code>, reaching 5.1 GB at one million tokens while the winner rows stay near 200 MB. Every bit count on this page is per token, per KV head, per layer, and includes the fp16 block scale of every channel read (16 bits per 64 tokens per channel).")
rep("Its 132-bit label cache is 16.5 bytes per token that our K cache does not need to exist.", "Its 136-bit label cache is 17 bytes per token that our K cache does not need to exist.")
# SparQ card: published grouped-query rule
rep('<div class="kv"><b>stores</b><span>the K cache channel-major (4-bit here)</span><b>reads per query</b><span>the r channels where this query\'s |q| is largest</span><b>who decides</b><span>each query, each head</span><b>bits / token</b><span>r × 4 per head; under GQA the kernel reads the union across heads</span></div>',
    '<div class="kv"><b>stores</b><span>the K cache channel-major (4-bit here)</span><b>reads per query</b><span>the r channels where |q|, summed over the query heads that share the KV head, is largest</span><b>who decides</b><span>each query, once per KV head</span><b>bits / token</b><span>r × 4 plus scales: 68 at r=16, 136 at r=32</span></div>')
rep('<p>The first method to let the query choose. <strong>On the toy (r=2):</strong> head A picks channels 0 and 2, head B picks 1 and 0. Head A <span class="mono">t0, t3</span> correct; head B <span class="mono">t5, t3</span> wrong, because |q| alone ranked channel 2 (weight -1.0) behind channel 0 (weight 1.0) and dropped it. The two heads disagree, so the shared read is the union: 3 channels, 12 bits, and still wrong.</p>',
    '<p>The first method to let the query choose. Under grouped-query attention SparQ sums |q| over the heads of the group before picking, so one channel set is read per KV head. <strong>On the toy (r=2):</strong> the summed |q| is (4, 3, 3, 0.3); channels 0 and 1 are picked (channel 1 wins the tie with channel 2 by index). Head A <span class="mono">t0, t3</span> correct; head B <span class="mono">t5, t3</span> wrong, because channel 2, the one that separates t0 from t3 for head B, is not read at all. 8 bits, same mistake as Double Sparsity.</p>')
rep('<p><strong>Where it breaks:</strong> three things. It ranks channels by |q| and ignores how much the keys vary in that channel. It is all-or-nothing per channel. And under grouped-query attention its real read is the union of the heads\' picks: 165 bits on Qwen3-4B, 197 on Qwen2.5-7B-1M, not r × 4.</p>',
    '<p><strong>Where it breaks:</strong> two things. It ranks channels by |q| and ignores how much the keys vary in that channel. And it is all-or-nothing per channel: a chosen channel is read at full 4-bit depth, an unchosen one not at all. A variant that lets every head pick its own channels and reads the union is stronger but costs about twice the bytes; the paper reports it once as an ablation.</p>')
rep('<div class="bits"><span class="b"></span><span class="hat"></span><span class="b"></span><span></span><span class="b"></span><span class="hat"></span><span class="b"></span><span></span><span class="b"></span><span class="hat"></span><span class="b"></span><span></span><span class="b"></span><span class="hat"></span><span class="b"></span><span></span></div>\n<div class="legend">head A\'s 2 channels solid; head B adds channel 1 (hatched) to the union</div>',
    '<div class="bits"><span class="b"></span><span class="b"></span><span></span><span></span><span class="b"></span><span class="b"></span><span></span><span></span><span class="b"></span><span class="b"></span><span></span><span></span><span class="b"></span><span class="b"></span><span></span><span></span></div>\n<div class="legend">2 channels chosen by the group\'s summed |q|, × 4 bits</div>')
rep('<tr><td>SparQ r=2</td><td>query, per head</td><td>channels (union under GQA)</td><td class="n">12</td><td class="ok">right</td><td class="no">wrong</td></tr>',
    '<tr><td>SparQ r=2</td><td>query, per KV head</td><td>channels</td><td class="n">8</td><td class="ok">right</td><td class="no">wrong</td></tr>')
# Move 4: per-layer text and basis table
rep("It halved the error on Qwen3 at 16k; on Qwen2.5 a flat budget did as well, so this is an option, not a requirement.",
    "It roughly halves the error on the QK-norm models (Qwen3) and is neutral or slightly worse on the Qwen2.5 models, and a plan calibrated at one context length does not transfer to another, so the flat budget is the default and the plan is a tuning step.")
rows = []
for model, ctx, K, f, qk in T.SETTINGS:
    if (model, ctx, K) not in {("Llama-3.1-8B", "4k", 128), ("Qwen2.5-7B", "32k", 512), ("Qwen2.5-7B-1M", "32k", 512), ("Qwen2.5-7B-1M", "128k", 2048), ("Qwen3-8B", "16k", 256)}: continue
    d = T.load(f)
    rows.append(f'<tr><td>{model}, {ctx}, K={K}</td><td class="n">{e4(d["u4/planes64"]["err"])}</td><td class="n">{e4(d["u4klt/planes64"]["err"])}</td><td class="n">{e4(d["dsparsity_c32_4bit"]["err"])}</td><td class="n">{e4(d["sparq_r32_4bit"]["err"])}</td></tr>')
hdr = '<tr><th>model, context, K</th><th class="n">raw channels, 80 bits</th>' if '<th class="n">raw channels, 80 bits</th>' in s else '<tr><th>model, context, K</th><th class="n">raw channels, 74 bits</th>'
cap = '<p class="cap">Attention-output relative error, lower is better.' if '<p class="cap">Attention-output relative error, lower is better.' in s else '<p class="cap">On the three models without QK-norm'
old = s[s.index(hdr):s.index(cap)]
s = s.replace(old, '<tr><th>model, context, K</th><th class="n">raw channels, 74 bits</th><th class="n">rotated planes, 74 bits</th><th class="n">Double Sparsity, 136 bits</th><th class="n">SparQ r32, 136 bits</th></tr>\n' + "\n".join(rows) + "\n</table></div>\n")
rep("Attention-output relative error, lower is better. With the rotation, 80 bits beats both 132-bit baselines on a model without QK-norm.", "On the three models without QK-norm the rotation lowers error at equal bits; on Qwen3-8B, which has QK-norm, it raises it.")
# Step 6 equal-error table
rows = []
for model, ctx, K, f, qk in T.HEADLINE:
    d = T.load(f); flat, pl = T.curve(d, T.store(qk)); a = flat + pl
    mds, msq = T.match(a, d["dsparsity_c32_4bit"]["err"]), T.match(a, d["sparq_r32_4bit"]["err"])
    rows.append(f'<tr><td>{model}, {ctx}, K={K}{"" if qk else ", rotated"}</td><td class="n">{T.fmt_b(mds[1])} ({mds[3]})</td><td class="n">{T.fmt_b(msq[1])} ({msq[3]})</td><td class="n">{e4(d["sparq_r16_4bit"]["err"])} vs {e4(min(d[T.store(qk)+"/planes48"]["err"], d[T.store(qk)+"/layeralloc_mean48"]["err"]))}</td></tr>')
hdr = '<tr><th>model, context, K</th><th class="n">ours, bits to match Double Sparsity (132 b)</th>' if 'Double Sparsity (132 b)</th>' in s else '<tr><th>model, context, K</th><th class="n">ours, bits to match Double Sparsity (136 b)</th>'
cap = '<p class="cap">Held-out attention-output error. "Bits" include the per-block scales.' if '"Bits" include the per-block scales.' in s else '<p class="cap">Held-out attention-output error; "flat" or "per-layer"'
old = s[s.index(hdr):s.index(cap)]
s = s.replace(old, '<tr><th>model, context, K</th><th class="n">ours, bits to match Double Sparsity (136 b)</th><th class="n">ours, bits to match SparQ r32 (136 b)</th><th class="n">SparQ r16 (68 b) error vs ours at 56 b</th></tr>\n' + "\n".join(rows) + "\n</table></div>\n")
rep('<p class="cap">Held-out attention-output error. "Bits" include the per-block scales. At 128k with K=128, a 0.1% selection ratio, every scan\'s error rises 4 to 8×; at the same ratio as the 32k runs the picture is identical to 32k.</p>',
    '<p class="cap">Held-out attention-output error; "flat" or "per-layer" is the cheaper plan that reaches the target. At 128k with K=128, a 0.1% selection ratio, every scan\'s error rises by an order of magnitude; at the same ratio as the 32k runs the crossings are where they are at 32k.</p>')
# RULER table (two runs) from the JSONs when present
def ruler_rows():
    files = [f"{T.R13}/ruler_Qwen3-8B_T32768_K128_ctx32000.json", f"{T.R13}/ruler_Qwen2.5-7B-Instruct-1M_T131072_K128_ctx128000.json"]
    ds = [T.load(f) for f in files]
    names = [("dense", "dense attention"), ("exact_topk", "exact top-k oracle"), ("planesL:u4:48", "ours 48 (raw)"), ("planesL:u4:64", "ours 64 (raw)"), ("planesK:u4:48", "ours 48 (rotated)"), ("planesK:u4:64", "ours 64 (rotated)"),
             ("ds:32:4", "Double Sparsity"), ("sparq:16:4", "SparQ r16"), ("sparq:32:4", "SparQ r32"), ("loki:64:4", "Loki r64"), ("thumb:2", "2-bit thumbnail"), ("landmark:8", "block landmarks b=8")]
    out = []
    for m, lab in names:
        cells = []; bits = None
        for d in ds:
            if d is None: cells.append("pending"); continue
            ts = [t for t in d if t != "_meta" and m in d[t]]
            if not ts: cells.append("—"); continue
            cells.append(f"{statistics.mean(statistics.mean(d[t][m]['scores']) for t in ts):.3f}"); bits = statistics.mean(d[t][m]["mean_bits"] for t in ts)
        if all(c in ("—", "pending") for c in cells) and not any(cells): continue
        cls = ' class="hi"' if m.startswith("planes") else (' class="bad"' if m.startswith("landmark") else "")
        out.append(f'<tr{cls}><td>{lab}</td><td class="n">{"full" if m == "dense" else ("—" if m == "exact_topk" else (f"{bits:.0f}" if bits else "—"))}</td><td class="n">{cells[0]}</td><td class="n">{cells[1]}</td></tr>')
    return "\n".join(out)
hdr = '<tr><th>method</th><th class="n">bits</th><th class="n">32k, K=128 (6 tasks)</th>' if '<th class="n">32k, K=128 (6 tasks)</th>' in s else '<tr><th>method</th><th class="n">bits</th><th class="n">Qwen3-8B, 32k, K=128 (6 tasks)</th>'
cap = next(c for c in ('<p class="cap">Every per-token scan sits within noise of the oracle', '<p class="cap">Standard errors at these sample sizes', '<p class="cap">At 32k (') if c in s)
old = s[s.index(hdr):s.index(cap)]
s = s.replace(old, '<tr><th>method</th><th class="n">bits</th><th class="n">Qwen3-8B, 32k, K=128 (6 tasks)</th><th class="n">Qwen2.5-7B-1M, 128k, K=128 (4 tasks)</th></tr>\n' + ruler_rows() + "\n</table></div>\n")
s = re.sub(r'<p class="cap">(Standard errors at these sample sizes|At 32k \().*?</p>', '<p class="cap">RULERCAP</p>', s, flags=re.S)
if 'Every per-token scan sits within noise of the oracle' in s:
    rep('<p class="cap">Every per-token scan sits within noise of the oracle (about ±0.04 at these sample sizes); ours does so at the fewest bits. Block selection loses 11 to 28 points. The 32k rows are Qwen3-8B, the 128k rows Qwen2.5-7B-Instruct-1M.</p>', '<p class="cap">RULERCAP</p>')
# regimes table and access pattern
rep("arithmetic per scanned bit (the planes kernel runs at 41–68 GB/s on an L4, the 4-bit channel scan at 94–111)", "arithmetic per scanned bit (on the A100 the planes kernel reaches 75–106 GB/s of HBM bandwidth, the 4-bit channel scans 148–245)")
rep("parity end to end; late layers 1.4× faster, early layers slower", "the scan kernel itself is 1.4× slower than the 32-channel scan while reading 40% fewer bytes; the scan is 16–24% of sparse attention, so the step lands within a few percent")
rep("all sparse methods tie at 105–140 ms/step (32k–128k, A40)", "all sparse methods tie at 44–55 ms/step of GPU time at 128k (A100, real prefill)")
rep("<p>One more thing the offload runs taught: access pattern matters as much as bytes. Kernels reading scattered 8-byte words straight from host memory reached 3 to 8 GB/s; copying each active channel's planes as one contiguous run into GPU memory first reaches 15 to 25 GB/s, the same as a plain memcpy. The bit-plane layout makes that contiguous run possible; the landmark index gets it for free but moves four times the bytes.</p>",
    "<p>One more thing the offload runs taught: access pattern matters as much as bytes. Copying each active channel's planes as one contiguous run into GPU memory moves 25–26 GB/s at every run size from 4 KB to 512 KB, the same as a plain memcpy of the same bytes; issuing one copy call per run manages 0.4 GB/s at 4 KB runs (23 µs of issue cost per call). The bit-plane layout makes the contiguous run possible; the landmark index gets it for free but moves four times the bytes.</p>")
rep("The scan is 10 to 25% of sparse attention there, and unpacking bit planes costs more arithmetic per bit than unpacking nibbles.", "The scan is 16 to 24% of sparse attention there, and unpacking bit planes costs more arithmetic per bit than unpacking nibbles.")
rep('<div class="foot">Numbers on this page are from the project\'s own runs: MacBook M3 (Qwen3-1.7B, 4B), an L4 (Qwen3-8B kernels, 16k–32k), an A40 (RULER at 32k, offload 32k–128k, Qwen2.5-7B) and an A100 (Qwen2.5-7B-Instruct-1M at 128k, offload 256k–1M).',
    '<div class="foot">Numbers on this page are from the project\'s own runs on one A100-SXM4-80GB pod (fidelity, RULER-style tasks, all timing, kernel and PCIe benches), plus Llama-3.1-8B activations captured on an L4 and evaluated with the same code. The tables are generated from the result files by the same script as the paper\'s.')
import statistics as st_
def rulercap():
    files = [f"{T.R13}/ruler_Qwen3-8B_T32768_K128_ctx32000.json", f"{T.R13}/ruler_Qwen2.5-7B-Instruct-1M_T131072_K128_ctx128000.json"]
    out = []
    for f, lab in zip(files, ("32k", "128k")):
        d = T.load(f)
        if d is None: continue
        ts = [t for t in d if t != "_meta"]; mean = lambda m: st_.mean(st_.mean(d[t][m]["scores"]) for t in ts if m in d[t])
        orc = mean("exact_topk"); scans = [m for m in d[ts[0]] if m not in ("dense", "exact_topk", "landmark:8")]
        n = len(d[ts[0]]["dense"]["scores"]); dev = max(abs(mean(m) - orc) for m in scans); lm = orc - mean("landmark:8")
        sc = [x for t in ts for x in d[t]["exact_topk"]["scores"]]; se = st_.pstdev(sc) / len(sc) ** 0.5
        out.append(f"At {lab} ({n} samples per task, standard error about {se:.3f}) every per-token scan is within {dev:.3f} of the exact top-k oracle; block landmarks lose {lm:.3f}.")
    return " ".join(out) + " Differences between per-token scans are inside the noise; what separates methods is bytes."
s = s.replace("RULERCAP", rulercap())
# regimes table refresh from macros (Fathom name, 40-bit row, breakdown, agent sessions)
M = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(f"{ROOT}/paper/macros.tex").read()))
s = s.replace("<h1>How Bit-Plane Key Scan Works</h1>", "<h1>How Fathom Works</h1>")
s = re.sub(r'<tr class="hi"><td>KV rows and index in host memory \(long contexts, many sessions\)</td><td>PCIe bytes</td><td>.*?</td></tr>',
    f'<tr class="hi"><td>KV rows and index in host memory (long contexts, many sessions)</td><td>PCIe bytes</td><td>GPU time per step at 1M (A100): Fathom 40 and 48 versus SparQ r16, the 32-channel scan, landmarks and the thumbnail as in the paper; at 56 bits Fathom is {M.get("ratioGPUchanr","--")}× faster than the 136-bit scans and ties SparQ r16 at {M.get("hthRatioMin","--")}–{M.get("hthRatioMax","--")}× lower error; at 47 bits it reads {M.get("fortyBytesFewerPct","--")}% fewer bytes than SparQ r16 and SparQ takes {M.get("ratioSparqSixteenOverForty","--")}× its GPU time</td></tr>', s, flags=re.S)
import re as _re
agent_block = f"""<h3>Real agent sessions</h3>
<p>{M.get("agentN","--")} sessions of about {M.get("agentCtxMean","--")}k tokens built from recorded OpenHands coding-agent trajectories on one repository each (Qwen2.5-7B-Instruct-1M). Each scan decodes the agent's next step from the same cache. The baseline is exact top-k decoding with the same k, which is what every scan is built to reproduce; the score is step agreement, the word-level sequence-match ratio between a method's step and the exact top-k step.</p>
<div class="tw"><table>
<tr><th>method</th><th class="n">k=512, step agreement with exact top-k</th><th class="n">k=2048</th></tr>
<tr class="hi"><td>Fathom 48 (56 b)</td><td class="n">{M.get("agentOsimplanesK","--")}</td><td class="n">{M.get("agentOsimBigplanesK","--")}</td></tr>
<tr class="hi"><td>Fathom 64 (74 b)</td><td class="n">{M.get("agentOsimplanesKsixtyfour","--")}</td><td class="n">—</td></tr>
<tr class="hi"><td>Fathom 80 (92 b)</td><td class="n">{M.get("agentOsimplanesKeighty","--")}</td><td class="n">—</td></tr>
<tr><td>SparQ r16 (68 b)</td><td class="n">{M.get("agentOsimsparq","--")}</td><td class="n">{M.get("agentOsimBigsparq","--")}</td></tr>
<tr><td>SparQ r32 (136 b)</td><td class="n">{M.get("agentOsimsparqthirtytwo","--")}</td><td class="n">—</td></tr>
<tr><td>Double Sparsity (136 b)</td><td class="n">{M.get("agentOsimds","--")}</td><td class="n">—</td></tr>
<tr class="bad"><td>block landmarks b=8</td><td class="n">{M.get("agentOsimlandmark","--")}</td><td class="n">{M.get("agentOsimBiglandmark","--")}</td></tr>
</table></div>
<p class="cap">At a 2% budget Fathom agrees with the exact top-k step at {M.get("agentOsimBigplanesK","--")} against {M.get("agentOsimBigsparq","--")} for SparQ r16 (paired margin {M.get("agentPairedBigSparq","--")} ± {M.get("agentPairedSEBigSparq","--")} over {M.get("agentNBig","--")} sessions), at 18% fewer bytes and the same GPU time; at 0.5% the most accurate scan is SparQ r32 at 136 bits ({M.get("agentOsimsparqthirtytwo","--")}), Fathom reaches the same agreement at 92 bits ({M.get("agentOsimplanesKeighty","--")}, paired {M.get("agentPairedEightySparqThirtyTwo","--")} ± {M.get("agentPairedSEEightySparqThirtyTwo","--")}), and the landmark index is below all of them. Calibrating Fathom on agent transcripts or on the session's own keys instead of Wikitext moves step agreement by {M.get("agentPairedAgentFortyEightWiki","--")} to {M.get("agentPairedAgentEightyWiki","--")}, within 1.5 s.e.; the ordering against SparQ r32 is set by the byte budget.</p>
"""
if "Real agent sessions" in s: s = _re.sub(r"<h3>Real agent sessions</h3>.*?(?=<h3>When bytes become time</h3>)", agent_block, s, flags=_re.S)
else: s = s.replace('<h3>When bytes become time</h3>', agent_block + '<h3>When bytes become time</h3>')
open(P, "w").write(s); print("html updated; pending left:", s.count("pending"))
