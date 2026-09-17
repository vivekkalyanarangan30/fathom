# RunPod runbook (driven from the laptop)

Goal of this GPU session: the three pieces of evidence a workshop paper still lacks, at minimum cost.
No H100 run: the HBM "flip" is ruled out on paper (scan is 10-25% of sparse attention; H100 has 5x less ALU per byte than the L4).

| # | what | why | GPU time |
|---|---|---|---|
| 1 | RULER-lite (6 tasks, 50 samples) at 32k, k = 128 and 256, 8 methods, Qwen3-8B | first downstream metric where scans can separate (passkey/multi-needle saturated at k = 512) | ~3 h |
| 2 | Offload decode, wall-clock, 32k/64k/128k, batch 1-2, index in host memory AND index in HBM | replaces the profiler-modelled 1.32x; the HBM-index run is the fair setting and is expected to show parity | ~1 h |
| 3 | Qwen2.5-7B (no QK-norm) fidelity at 32k, K = 512 and 128, with FFD-thumbnail / sign-index extras | third model family; tests whether the raw-channel basis result holds without QK-norm | ~1.5 h |

Pod: Secure Cloud, 1x A40 48 GB (fallback L40S 48 GB), template "RunPod PyTorch 2.8" (CUDA 12.8), volume 150 GB at /workspace, SSH enabled.
Budget: about 6 GPU-hours; stop the pod as soon as `results/STATUS.md` says finished.

## Steps
1. Deploy the pod; copy the SSH command from the pod's Connect panel.
2. From the laptop: `tar czf /tmp/nq.tgz -C ~/new_quant runpod && scp -P <port> /tmp/nq.tgz root@<ip>:/workspace/ && ssh -p <port> root@<ip> 'cd /workspace && tar xzf nq.tgz && mv runpod new_quant && bash new_quant/rp_bootstrap.sh'`
3. `ssh ... 'cd /workspace/new_quant && nohup bash rp_run_all.sh > results/run_all.log 2>&1 &'`
4. Poll: `ssh ... 'cat /workspace/new_quant/results/STATUS.md; tail -3 /workspace/new_quant/results/*.log'`
5. Pull: `scp -P <port> -r root@<ip>:/workspace/new_quant/results ~/new_quant/runpod_results/<date>/` (exclude caps_*.safetensors).
6. Stop the pod, then terminate it once results are verified locally.

## Outputs
- `results/ruler_Qwen3-8B_T32768_K128_ctx32000.json`, `..._K256_...` (per task, per method: mean score, bits/token, generations)
- `results/B_e2e_offload.json`, `results/B_e2e_offload_indexhbm.json`
- `results/final_alloc_Qwen2.5-7B_T32768.json` (K = 512, extras) and the K = 128 variant
- `results/STATUS.md`, one log per step
