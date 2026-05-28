#!/usr/bin/env bash
set -euo pipefail

# modified by zhoujiwen: one-command launcher for SSL-style feature distillation on GPUs 4-7.
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}:."
export CUDA_VISIBLE_DEVICES=4,5,6,7  # modified by zhoujiwen

echo "[ssl-feature-distill] workspace: $(pwd)"
echo "[ssl-feature-distill] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
test -f checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth || { echo "[ssl-feature-distill][ERROR] missing teacher checkpoint"; exit 1; }
test -d /mnt/ht2-nas2/00-model/00-hulj/Dinov3/workspace/data_eurosat_train/train/pseudo/ || { echo "[ssl-feature-distill][ERROR] missing dataset dir"; exit 1; }

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Refusing to run on CPU.")
print(f"CUDA devices: {torch.cuda.device_count()} | current: {torch.cuda.get_device_name(0)}")
PY

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" dinov3/train/train.py \
  --config-file dinov3/configs/train/swin_base_feature_distill_vitl16.yaml \
  --output-dir outputs/swin_base_vitl16_ssl_feature_distill "$@"
