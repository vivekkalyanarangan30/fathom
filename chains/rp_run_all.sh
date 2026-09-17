#!/usr/bin/env bash
# Whole RunPod assignment, in priority order; every step logs to results/STATUS.md and its own log; rerunnable (each script resumes).
set -uo pipefail
cd /workspace/new_quant; export HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ST=results/STATUS.md; touch "$ST"
HOST_GB=$(python -c "import os;print(int(os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')/1e9*0.6))")
step() { local name=$1; shift; grep -q "DONE  $name\$" "$ST" && { echo "skip $name"; return 0; }
  echo "$(date -u +%FT%TZ) START $name" >> "$ST"
  if "$@" > "results/$name.log" 2>&1; then echo "$(date -u +%FT%TZ) DONE  $name" >> "$ST"; else echo "$(date -u +%FT%TZ) FAIL  $name: $(grep -m1 -E 'Error|error' results/$name.log | cut -c1-160)" >> "$ST"; return 1; fi; }

# 0. calibration capture for the 32k plans (train split only; ~34 GB on disk)
step cap_8b_32k_train python scripts/l4_capture.py Qwen/Qwen3-8B 32768 train
step selfcheck python scripts/R_ruler.py --tag Qwen3-8B_T32768 --selfcheck
# 1. RULER-lite where methods can separate: k = 128 at 32k (main table), then k = 256
step ruler_8b_32k_K128 python scripts/R_ruler.py --tag Qwen3-8B_T32768 --ctx 32000 --k 128 --ns 50
step ruler_8b_32k_K256 python scripts/R_ruler.py --tag Qwen3-8B_T32768 --ctx 32000 --k 256 --ns 25
# 2. offload regime, wall-clock: index in host memory (bytes = time) AND the fair index-in-HBM setting
M=dense_offload,chan4_r32,chan4_r16,planes_thumb2,planes_mean48,planes_mean64
step offload_hostindex python scripts/B_e2e_offload_v2.py --tag Qwen3-8B_T32768 --ctxs 32768,65536,131072 --batches 1,2 --steps 30 --methods $M --host-gb $HOST_GB
step offload_hbmindex  python scripts/B_e2e_offload_v2.py --tag Qwen3-8B_T32768 --ctxs 32768,65536,131072 --batches 1,2 --steps 30 --methods $M --host-gb $HOST_GB --index-hbm
# 3. third model family (no QK-norm): fidelity sweep at 32k with the FFD / sign-index extras
step cap_q25_32k python scripts/l4_capture.py Qwen/Qwen2.5-7B 32768 train,test
step fid_q25_32k_K512 python scripts/l4_final_alloc.py Qwen2.5-7B_T32768 512 --extras
step fid_q25_32k_K128 python scripts/l4_final_alloc.py Qwen2.5-7B_T32768 128
echo "$(date -u +%FT%TZ) rp_run_all.sh finished" >> "$ST"
