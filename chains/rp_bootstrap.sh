#!/usr/bin/env bash
# RunPod bootstrap: run once inside the pod after the project tarball is in /workspace/new_quant (scp from the laptop).
# Keeps the template's torch/triton; installs the pinned rest; caches HF models on the persistent volume.
set -euo pipefail
cd /workspace/new_quant
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME" results
python -m pip install -q --break-system-packages --upgrade pip
python -m pip install -q --break-system-packages "transformers==4.56.1" "datasets==4.0.0" "safetensors==0.6.2" "accelerate==1.10.1" "huggingface-hub==0.34.4" "numpy==2.3.2" "scipy==1.16.1" hf_transfer bitsandbytes
python - <<'PY'
import torch, triton, transformers, os
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__} triton {triton.__version__} transformers {transformers.__version__}")
print(f"GPU {p.name} {p.total_memory/1e9:.0f} GB, SMs {p.multi_processor_count}; host RAM {os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')/1e9:.0f} GB")
PY
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem --format=csv
df -h /workspace | tail -1
echo "bootstrap done. Next: nohup bash rp_run_all.sh > results/run_all.log 2>&1 &"
