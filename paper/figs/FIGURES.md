# Figures

Every PDF here is written by `python paper/make_figs.py` from the files under `results/`.

| figure | content | sources |
|---|---|---|
| fig_method.pdf | schematic of the bit-plane store and a per-query plane read | drawn, no data |
| fig_toy.pdf | the worked example of Appendix A: six keys, four channels, two heads | drawn from the values in the appendix |
| fig_frontier.pdf | attention-output error vs scan bits, three panels | results/a100_20260913/final_alloc_Qwen3-8B_T16384.json, final_alloc_Qwen3-8B_T32768.json, final_alloc_Qwen2.5-7B-Instruct-1M_T131072_K2048.json |
| fig_ratio.pdf | error vs bits at k = 128, 512, 2048 on Qwen2.5-7B-Instruct-1M at 128k | results/a100_20260913/final_alloc_Qwen2.5-7B-Instruct-1M_T131072.json, the _K512 and _K2048 files |
| fig_basis.pdf | raw vs KLT-rotated planes at means 48 and 64 on six settings | the seven final_alloc files in results/a100_20260913 and results/l4 |
| fig_ruler.pdf | RULER-style mean score per method, 32k and 128k | results/a100_20260913/ruler_Qwen3-8B_T32768_K128_ctx32000.json, ruler_Qwen2.5-7B-Instruct-1M_T131072_K128_ctx128000.json |
| fig_offload.pdf | decode step wall-clock and GPU time vs context, index in host memory | results/a100_20260913/B_e2e_offload_synth_copyidx.json |
| fig_hbm.pdf | real prefill at 32k to 128k, index in host memory vs index in HBM | results/a100_20260913/B_e2e_offload_real_copyidx.json, B_e2e_offload_real_hbm.json |
| fig_layers.pdf | per-layer bit budgets of the greedy plans | results/a100_20260913/final_alloc_Qwen3-8B_T16384.json, final_alloc_Qwen3-8B_T32768.json |
| fig_kernel.pdf | achieved HBM bandwidth of the scan kernels | results/a100_20260913/B_kernel_bench_a100.json |
| fig_pcie.pdf | PCIe transfer rate vs run size and stride | results/a100_20260913/pcie_bench.log |
