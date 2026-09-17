# Fathom: per-query read depth for sparse decoding over offloaded KV caches

Code, measurements and paper source for

> Vivek Kalyanarangan. **Fathom: Per-Query Read Depth for Sparse Decoding over Offloaded KV Caches.** arXiv:2609.17652, 2026.
> [arxiv.org/abs/2609.17652](https://arxiv.org/abs/2609.17652) · [PDF](paper/main.pdf) · [plain-language explainer](docs/method.html)

Fathom is a key scan for top-k sparse decoding when the KV cache and the index that ranks it live in host memory. The 4-bit K cache is stored channel-major as bit planes, so reading the first *t* planes of a channel is exactly that channel's *t*-bit quantizer, and each query decides per channel how many planes to read by reverse water-filling over its channel importances. Every number in the paper is produced from the JSON files in `results/` by `paper/make_tables.py` and `paper/make_figs.py`.

| regime (A100, Qwen3-8B, k = 512) | what binds | Fathom, 56-bit read unless noted |
|---|---|---|
| KV rows and index in host memory, 1M tokens | PCIe bytes | 1.67x faster in GPU time than the 136-bit scans (Double Sparsity, Loki, SparQ r=32), 2.50x than a landmark index; same GPU time as SparQ r=16 at 18% fewer bytes and lower error |
| same, 128k tokens, real prefill | PCIe bytes | 1.26x faster than the 136-bit scans |
| real coding-agent sessions, 100k tokens, k = 2048 | scan fidelity | step agreement with exact top-k 0.67 vs 0.49 for SparQ r=16 |
| real coding-agent sessions, k = 512 | scan fidelity | 0.60 at 92 bits, equal to SparQ r=32 at 136 bits |
| rows in host memory, index in HBM | shared row fetch | not faster |
| everything in HBM | arithmetic per scanned bit | not faster |

## Layout

```
kernels/bitplane.py      bit-plane store packing, water-filling plan, Triton scan kernels, top-k with sink/local  (Sec. 3)
scripts/sketch.py        quantizers and approximate-score functions for every method (Fathom, SparQ, Double Sparsity, Loki, thumbnail, landmark)
scripts/l4_capture.py    capture post-RoPE q/k/v of a model on Wikitext-103 windows into a safetensors file
scripts/l4_final_alloc.py  attention-output error vs scan bits for every method; per-layer plans; KLT basis  (Sec. 5.3, 5.4, App. B, C)
scripts/B_e2e_offload_v2.py  decode step with K/V rows and scan index in pinned host memory or HBM  (Sec. 5.1, 5.2, 7)
scripts/B_kernel_bench.py    HBM-resident scan kernel bandwidth  (Sec. 7)
scripts/R_ruler.py       RULER-style tasks decoded under each scan  (Sec. 5.5, App. B)
scripts/R_agent.py       OpenHands coding-agent sessions decoded under each scan; calibration variants  (Sec. 5.6, App. D B5)
scripts/agent_calib_text.py  agent-transcript calibration text disjoint from the benchmark sessions  (App. D B5)
scripts/bench_scan_cfg.py, topk_bench.py  launch-config sweep and top-k variants  (Sec. 6 A6)
chains/                  the shell chains that ran the experiments on RunPod A100 pods, in order, with resume-on-restart
results/                 the JSON outputs those chains produced (a100_20260913 and a100_20260914 runs; results/l4 holds the Llama-3.1-8B fidelity run)
paper/                   main.tex, macros.tex, refs.bib, tables/, figs/, and the two generators
docs/method.html         the explainer, regenerated from the same JSONs by docs/update_method_html.py
```

## Paper section to code and data

| paper | what it reports | script | result files |
|---|---|---|---|
| Sec. 3 Method, App. A worked example | store, quantizer, water-filling, per-layer plan, basis rule, gather | `kernels/bitplane.py` (`quantise_u4`, `pack_planes`, `make_plan`, `PlaneStore.scan`, `sink_local_topk`), `scripts/sketch.py` (`qblock`, `waterfill`) | `paper/make_figs.py::fig_toy` reproduces the worked example |
| Sec. 5.1 target regime, Tables 2, 3, 8, Fig. 2 | decode step vs context, host index vs HBM index, component breakdown | `scripts/B_e2e_offload_v2.py` | `results/a100_20260913/B_e2e_offload_synth_copyidx.json`, `B_e2e_offload_synth_hbm.json`, `B_e2e_offload_real_copyidx.json`, `B_e2e_offload_real_hbm.json`, `B_e2e_offload_synth_b2_copyidx.json`; `results/a100_20260914/trace_breakdown_host.json` |
| Sec. 5.2 matched GPU time, Table 4 | Fathom 47/56 b vs SparQ r=16 at equal step time | `scripts/l4_final_alloc.py` | the seven `final_alloc_*.json` fidelity files |
| Sec. 5.3 equal error, Table 5, Fig. 3, App. C Tables 12, 13 | bits at which Fathom reaches each baseline's error | `scripts/l4_final_alloc.py` | `results/a100_20260914/final_alloc_Qwen3-8B_T16384.json`, `..._T32768.json`, `final_alloc_Qwen3-4B_T16384.json`, `final_alloc_Qwen2.5-7B_T32768.json`, `final_alloc_Qwen2.5-7B-Instruct-1M_T32768.json`, `..._T131072.json`, `..._T131072_K2048.json`; `results/a100_20260913/*_K128.json`, `*_K512.json`; `results/l4/final_alloc_Meta-Llama-3.1-8B_T4096_K128.json` |
| Sec. 5.4 selection ratio, Table 14, Fig. 6 | error vs bits at k = 128, 512, 2048 on 128k | `scripts/l4_final_alloc.py` | the three Qwen2.5-7B-Instruct-1M 128k files |
| Sec. 5.5 downstream, Table 6, Fig. 7, App. B | RULER-style tasks | `scripts/R_ruler.py` | `results/a100_20260913/ruler_Qwen3-8B_T32768_K128_ctx32000.json`, `ruler_Qwen2.5-7B-Instruct-1M_T131072_K128_ctx128000.json` |
| Sec. 5.6 agent sessions, Tables 7, 11 | step agreement with exact top-k | `scripts/R_agent.py` | `results/a100_20260914/agent_Qwen2.5-7B-Instruct-1M_T131072_K512_ctx100000.json`, `..._K2048_...json` |
| Sec. 6 ablations, Tables 16 to 18, Figs. 5, 11 | per-layer plan, basis, thumbnail, SparQ variant, measurement controls | `scripts/l4_final_alloc.py --klt --stores rd128,u4,u8`, `scripts/bench_scan_cfg.py`, `B_e2e_offload_v2.py --pipeline` | fidelity files above; `results/a100_20260914/scan_cfg_1m.json`, `shake_pipe4.json`, `shake_pipe8.json` |
| Sec. 7 analysis, Table 15, Figs. 8 to 10 | HBM kernel, arithmetic per byte, PCIe access pattern | `scripts/B_kernel_bench.py` | `results/a100_20260913/B_kernel_bench_a100.json`, `pcie_bench.log` |
| App. D B5 calibration domain, Tables 22, 23 | Wikitext vs agent-transcript vs session calibration | `scripts/agent_calib_text.py`, `l4_capture.py --text-dir`, `l4_final_alloc.py --test-tag`, `R_agent.py --calib-tag` | `results/a100_20260914/final_alloc_{wikical,agentcal}_on_agent_K{512,2048}.json`, `agent_calib_K512_ctx100000.json` |

Table and figure numbers refer to the arXiv v1 PDF.

## Reproducing

Everything below ran on one RunPod A100-SXM4-80GB pod (PCIe 4.0, 2 TB host RAM, `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`). `chains/rp_bootstrap.sh` installs the pinned dependencies; `requirements.txt` lists them. The chains assume the repository is at `/workspace/new_quant` and write to `results/`; edit the `cd` line for another location. Models are downloaded from Hugging Face; the agent benchmark reads `nebius/SWE-rebench-openhands-trajectories`.

1. Capture activations and measure fidelity (Sec. 5.3, 5.4, App. B, C): `chains/rp3_run_all.sh` phase A, which is `l4_capture.py` then `l4_final_alloc.py` per model and context. A 32k capture of an 8B model is about 34 GB on disk.
2. Offload timing (Sec. 5.1, 5.2, 7): `chains/rp4.sh` phase C, which is `B_e2e_offload_v2.py` with `--copy-index` (index in host memory) and `--index-hbm`, then `B_kernel_bench.py`. Contexts beyond 128k use `--synthetic-kv` (timing only), as stated in the paper.
3. RULER-style tasks (Sec. 5.5): `chains/rp4b.sh`, which runs `R_ruler.py --selfcheck` (the dense path must reproduce Hugging Face greedy decoding token for token) and then the two task runs.
4. Agent sessions (Sec. 5.6): `python scripts/R_agent.py --model Qwen/Qwen2.5-7B-Instruct-1M --tag Qwen2.5-7B-Instruct-1M_T131072 --ctx 100000 --k 512 --ns 40 --min-ctx 80000`, and the same with `--k 2048 --ns 20`.
5. Calibration domain (App. D B5): `chains/rp5.sh`.
6. Paper: `python paper/make_tables.py && python paper/make_figs.py` regenerates `paper/tables/*.tex`, `paper/macros.tex` and `paper/figs/*.pdf` from `results/` (byte-identical to the committed files), then compile `paper/main.tex` with pdflatex or tectonic.

Two things are not in this repository. The captured activations (`caps_*.safetensors`, 17 to 34 GB per setting) are regenerated by step 1. The PCIe gather microbenchmark script that produced `results/a100_20260913/pcie_bench.log` (Fig. 10, Table 10) was run on the pod and not saved; the log holds its full output, and `B_e2e_offload_v2.py` contains the same Triton gather.

## Method names in the code

`planes_mean48`, `planesK:u4:48` and the like are Fathom (flat budget or per-layer plan; `planesK` is the KLT-rotated store, `planesL` the raw one). `chan4_r32` / `chan4_r16` are SparQ under its published grouped-query rule at 4 bits, which is also the byte count of Double Sparsity c=32 and Loki r=32 in the timing harness. `ds`, `loki`, `planes_thumb2` and `landmark8` are Double Sparsity, Loki, the 2-bit thumbnail and the ShadowKV-style block-mean index. The bit convention everywhere is code bits plus 16 bits per 64 tokens for the block scale of every active channel.

## Citation

```bibtex
@article{kalyanarangan2026fathom,
  title   = {Fathom: Per-Query Read Depth for Sparse Decoding over Offloaded KV Caches},
  author  = {Kalyanarangan, Vivek},
  journal = {arXiv preprint arXiv:2609.17652},
  year    = {2026}
}
```

MIT license.
