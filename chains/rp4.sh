#!/usr/bin/env bash
# Remaining chain on the A100: redo the failed 8B 32k fidelity, the 128k captures and fidelities, all timing, then RULER (never concurrent with timing).
set -uo pipefail
cd /workspace/new_quant; export HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ST=results/STATUS.md; touch "$ST"
step() { local name=$1; shift; grep -q "DONE  $name\$" "$ST" && { echo "skip $name"; return 0; }
  echo "$(date -u +%FT%TZ) START $name" >> "$ST"
  if "$@" > "results/$name.log" 2>&1; then echo "$(date -u +%FT%TZ) DONE  $name" >> "$ST"; else echo "$(date -u +%FT%TZ) FAIL  $name: $(grep -m1 -E 'Error|error' results/$name.log | cut -c1-160)" >> "$ST"; return 1; fi; }
FID="python scripts/l4_final_alloc.py"; CAP="python scripts/l4_capture.py"; OFF="python scripts/B_e2e_offload_v2.py --tag Qwen3-8B_T32768 --host-gb 200"
M1M=chan4_r32,chan4_r16,planes_thumb2,planes_mean48,planes_mean64,landmark8
M32=dense_offload,$M1M
step fid_8b_32k_K512  $FID Qwen3-8B_T32768 512 --klt --stores u4 --alloc-from Qwen3-8B_T16384
step cap_q25_1m_128k $CAP Qwen/Qwen2.5-7B-Instruct-1M 131072 train,test
step fid_q25_1m_128k_K128  $FID Qwen2.5-7B-Instruct-1M_T131072 128  --klt --stores u4 --tq 64
step fid_q25_1m_128k_K512  $FID Qwen2.5-7B-Instruct-1M_T131072 512  --klt --stores u4 --tq 64 --out results/final_alloc_Qwen2.5-7B-Instruct-1M_T131072_K512.json
step fid_q25_1m_128k_K2048 $FID Qwen2.5-7B-Instruct-1M_T131072 2048 --klt --stores u4 --tq 64 --out results/final_alloc_Qwen2.5-7B-Instruct-1M_T131072_K2048.json
echo "$(date -u +%FT%TZ) phase A finished" >> "$ST"
step offload_synth_hostindex $OFF --ctxs 32768,262144,524288,1048576 --batches 1 --steps 20 --synthetic-kv --copy-index --methods $M1M --out results/B_e2e_offload_synth_copyidx.json
step offload_synth_hbmindex  $OFF --ctxs 32768,262144,524288,1048576 --batches 1 --steps 20 --synthetic-kv --index-hbm   --methods $M1M --out results/B_e2e_offload_synth_hbm.json
step offload_real_hostindex  $OFF --ctxs 32768,65536,131072 --batches 1 --steps 30 --copy-index --methods $M32 --out results/B_e2e_offload_real_copyidx.json
step offload_real_hbmindex   $OFF --ctxs 32768,65536,131072 --batches 1 --steps 30 --index-hbm  --methods $M32 --out results/B_e2e_offload_real_hbm.json
step offload_synth_b2        $OFF --ctxs 262144,524288 --batches 2 --steps 20 --synthetic-kv --copy-index --methods $M1M --out results/B_e2e_offload_synth_b2_copyidx.json
step kernel_bench python scripts/B_kernel_bench.py --tag Qwen3-8B_T32768 --ns 32768,131072 --batches 1,4 --layers 0,1,2,18,30 --k 512 --peak-gbps 2039 --out results/B_kernel_bench_a100.json
step pcie_bench python pcie_gather_bench.py
echo "$(date -u +%FT%TZ) phase C finished" >> "$ST"
M8=dense,exact_topk,planesL:u4:48,planesL:u4:64,ds:32:4,sparq:16:4,sparq:32:4,loki:64:4,thumb:2,landmark:8
MQ=dense,exact_topk,planesL:u4:48,planesL:u4:64,planesK:u4:48,planesK:u4:64,ds:32:4,sparq:16:4,sparq:32:4,loki:64:4,thumb:2,landmark:8
step selfcheck_8b python scripts/R_ruler.py --tag Qwen3-8B_T32768 --selfcheck
step ruler_8b_32k_K128 python scripts/R_ruler.py --tag Qwen3-8B_T32768 --ctx 32000 --k 128 --ns 50 --methods $M8
step selfcheck_q25_1m python scripts/R_ruler.py --model Qwen/Qwen2.5-7B-Instruct-1M --tag Qwen2.5-7B-Instruct-1M_T131072 --ctx 128000 --selfcheck
step ruler_q25_1m_128k_K128 python scripts/R_ruler.py --model Qwen/Qwen2.5-7B-Instruct-1M --tag Qwen2.5-7B-Instruct-1M_T131072 --ctx 128000 --k 128 --ns 30 --tasks niah_multikey,niah_multiquery,vt,fwe --methods $MQ
echo "$(date -u +%FT%TZ) rp4 finished" >> "$ST"
