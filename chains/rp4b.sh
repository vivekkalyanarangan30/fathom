#!/usr/bin/env bash
# Take over from rp4.sh once phase C is done: stop it before its RULER steps and run RULER with the budgeted sample counts.
set -uo pipefail
cd /workspace/new_quant; export HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ST=results/STATUS.md
while ! grep -q "phase C finished" $ST; do sleep 20; done
P=$(pgrep -f "^bash rp4.sh" | head -1)
if [ -n "$P" ]; then for c in $(pgrep -P "$P"); do kill "$c" 2>/dev/null; done; kill "$P" 2>/dev/null; sleep 3; fi
sed -i '/START selfcheck_8b$/d; /START ruler_8b_32k_K128$/d' $ST
step() { local name=$1; shift; grep -q "DONE  $name\$" "$ST" && { echo "skip $name"; return 0; }
  echo "$(date -u +%FT%TZ) START $name" >> "$ST"
  if "$@" > "results/$name.log" 2>&1; then echo "$(date -u +%FT%TZ) DONE  $name" >> "$ST"; else echo "$(date -u +%FT%TZ) FAIL  $name: $(grep -m1 -E 'Error|error' results/$name.log | cut -c1-160)" >> "$ST"; return 1; fi; }
M8=dense,exact_topk,planesL:u4:48,planesL:u4:64,ds:32:4,sparq:16:4,sparq:32:4,loki:64:4,thumb:2,landmark:8
MQ=dense,exact_topk,planesL:u4:48,planesL:u4:64,planesK:u4:48,planesK:u4:64,ds:32:4,sparq:16:4,sparq:32:4,loki:64:4,thumb:2,landmark:8
step selfcheck_8b python scripts/R_ruler.py --tag Qwen3-8B_T32768 --selfcheck
step ruler_8b_32k_K128 python scripts/R_ruler.py --tag Qwen3-8B_T32768 --ctx 32000 --k 128 --ns 40 --methods $M8
step selfcheck_q25_1m python scripts/R_ruler.py --model Qwen/Qwen2.5-7B-Instruct-1M --tag Qwen2.5-7B-Instruct-1M_T131072 --ctx 128000 --selfcheck
step ruler_q25_1m_128k_K128 python scripts/R_ruler.py --model Qwen/Qwen2.5-7B-Instruct-1M --tag Qwen2.5-7B-Instruct-1M_T131072 --ctx 128000 --k 128 --ns 30 --tasks niah_multikey,niah_multiquery,vt,fwe --methods $MQ
echo "$(date -u +%FT%TZ) rp4 finished" >> "$ST"
