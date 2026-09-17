#!/usr/bin/env bash
# Pod 2 (A100 80 GB): the long-context evidence. Logs to results/STATUS.md; every step resumes/skips if already done.
set -uo pipefail
cd /workspace/new_quant; export HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ST=results/STATUS.md; touch "$ST"
HOST_GB=${HOST_GB:-$(python -c "import os;print(int(os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')/1e9*0.7))")}
step() { local name=$1; shift; grep -q "DONE  $name\$" "$ST" && { echo "skip $name"; return 0; }
  echo "$(date -u +%FT%TZ) START $name" >> "$ST"
  if "$@" > "results/$name.log" 2>&1; then echo "$(date -u +%FT%TZ) DONE  $name" >> "$ST"; else echo "$(date -u +%FT%TZ) FAIL  $name: $(grep -m1 -E 'Error|error' results/$name.log | cut -c1-160)" >> "$ST"; return 1; fi; }
M1M=chan4_r32,chan4_r16,planes_thumb2,planes_mean48,planes_mean64,landmark8

# 0. Qwen3-8B calibration (for the offload plans) + a 3-minute shakedown of the synthetic-KV offload path before anything long
step cap_8b_32k_train python scripts/l4_capture.py Qwen/Qwen3-8B 32768 train
step offload_synth_shakedown python scripts/B_e2e_offload_v2.py --tag Qwen3-8B_T32768 --ctxs 32768 --batches 1 --steps 4 --synthetic-kv --methods $M1M --host-gb $HOST_GB --out results/shakedown.json
# 1. long-context quality: Qwen2.5-7B-Instruct-1M at 128k, k = 128 (4 tasks x 30 samples, 9 methods incl. the landmark baseline)
step cap_q25_1m_128k python scripts/l4_capture.py Qwen/Qwen2.5-7B-Instruct-1M 131072 train,test
step fid_q25_1m_128k_K128 python scripts/l4_final_alloc.py Qwen2.5-7B-Instruct-1M_T131072 128 --stores u4 --tq 64
step selfcheck_q25_1m python scripts/R_ruler.py --model Qwen/Qwen2.5-7B-Instruct-1M --tag Qwen2.5-7B-Instruct-1M_T131072 --ctx 128000 --selfcheck
step ruler_q25_1m_128k_K128 python scripts/R_ruler.py --model Qwen/Qwen2.5-7B-Instruct-1M --tag Qwen2.5-7B-Instruct-1M_T131072 --ctx 128000 --k 128 --ns 30 --tasks niah_multikey,niah_multiquery,vt,fwe
# 2. offload decode at agentic context lengths (synthetic KV: timing only), index in host memory and index in HBM
step offload_1m_hostindex python scripts/B_e2e_offload_v2.py --tag Qwen3-8B_T32768 --ctxs 262144,524288,1048576 --batches 1 --steps 20 --synthetic-kv --methods $M1M --host-gb $HOST_GB
step offload_1m_hbmindex  python scripts/B_e2e_offload_v2.py --tag Qwen3-8B_T32768 --ctxs 262144,524288,1048576 --batches 1 --steps 20 --synthetic-kv --methods $M1M --host-gb $HOST_GB --index-hbm
step offload_1m_b2_hostindex python scripts/B_e2e_offload_v2.py --tag Qwen3-8B_T32768 --ctxs 262144,524288 --batches 2 --steps 20 --synthetic-kv --methods $M1M --host-gb $HOST_GB --out results/B_e2e_offload_synth_b2.json
echo "$(date -u +%FT%TZ) rp2_run_all.sh finished" >> "$ST"
