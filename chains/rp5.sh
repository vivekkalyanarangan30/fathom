#!/usr/bin/env bash
# Calibration study on the A100: agent-domain and self-calibrated Fathom on the k=512 agent sessions, plus in-domain and cross-domain fidelity.
set -uo pipefail
cd /workspace/new_quant; export HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
ST=results/STATUS.md; touch "$ST"
step() { local name=$1; shift; grep -q "DONE  $name\$" "$ST" && { echo "skip $name"; return 0; }
  echo "$(date -u +%FT%TZ) START $name" >> "$ST"
  if "$@" > "results/$name.log" 2>&1; then echo "$(date -u +%FT%TZ) DONE  $name" >> "$ST"; else echo "$(date -u +%FT%TZ) FAIL  $name: $(grep -m1 -E 'Error|error' results/$name.log | cut -c1-160)" >> "$ST"; return 1; fi; }
MODEL=Qwen/Qwen2.5-7B-Instruct-1M; TAG=Qwen2.5-7B-Instruct-1M_T131072; CAL=Qwen2.5-7B-Instruct-1M-agentcal_T131072
FID="python scripts/l4_final_alloc.py"
step calib_text python scripts/agent_calib_text.py --exclude results/agent_prev_K512.json --out-dir results/agentcal
step cap_agentcal python scripts/l4_capture.py $MODEL 131072 train,test --text-dir results/agentcal --suffix agentcal
step cap_wiki_train python scripts/l4_capture.py $MODEL 131072 train
step fid_agentcal_K512  $FID $CAL 512  --klt --stores u4 --tq 64 --out results/final_alloc_agentcal_on_agent_K512.json
step fid_agentcal_K2048 $FID $CAL 2048 --klt --stores u4 --tq 64 --out results/final_alloc_agentcal_on_agent_K2048.json
step fid_wikical_K512   $FID $TAG 512  --klt --stores u4 --tq 64 --test-tag $CAL --out results/final_alloc_wikical_on_agent_K512.json
step fid_wikical_K2048  $FID $TAG 2048 --klt --stores u4 --tq 64 --test-tag $CAL --out results/final_alloc_wikical_on_agent_K2048.json
rm -f results/caps_Qwen2.5-7B-Instruct-1M-agentcal_test_T131072.safetensors
M=dense,exact_topk,planesK:u4:80,planesA:u4:48,planesA:u4:64,planesA:u4:80,planesS:u4:48,planesS:u4:64,planesS:u4:80,dsA:32:4
AG="python scripts/R_agent.py --model $MODEL --tag $TAG --calib-tag $CAL --ctx 100000 --min-ctx 80000 --k 512"
step agent_shake2 $AG --ns 1 --gen-max 16 --methods $M --out results/agent_shake2.json
step agent_calib_K512 $AG --ns 40 --methods $M --out results/agent_calib_K512_ctx100000.json
echo "$(date -u +%FT%TZ) rp5 finished" >> "$ST"
