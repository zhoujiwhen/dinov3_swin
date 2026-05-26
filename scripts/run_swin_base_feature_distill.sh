#!/usr/bin/env bash
set -euo pipefail

# modified by zhoujiwen: one-command launcher for A40 CUDA feature distillation.
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}:."
export CUDA_VISIBLE_DEVICES=4  # modified by zhoujiwen: use GPU 4 by default.

echo "[feature-distill] workspace: $(pwd)"
echo "[feature-distill] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
test -f checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth || { echo "[feature-distill][ERROR] missing teacher checkpoint: checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"; exit 1; }
test -d /mnt/ht2-nas2/00-model/00-hulj/Dinov3/workspace/data_eurosat_train/train/pseudo/ || { echo "[feature-distill][ERROR] missing dataset dir: /mnt/ht2-nas2/00-model/00-hulj/Dinov3/workspace/data_eurosat_train/train/pseudo/"; exit 1; }

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Refusing to run feature distillation on CPU.")
print(f"CUDA devices: {torch.cuda.device_count()} | current: {torch.cuda.get_device_name(0)}")
PY

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" dinov3/train/train.py \
  --config-file dinov3/configs/train/swin_base_feature_distill_vitl16.yaml \
  --output-dir outputs/swin_base_vitl16_feature_distill "$@"
