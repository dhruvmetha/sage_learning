#!/bin/bash
# 2-epoch smoke training run for the cross-attn diffusion mask model on the
# v3 tight-crop H5 (partial — built from data already on disk).
#
# Goal: verify the dataloader/model/wandb pipeline is healthy. Logs go to
# wandb project sage_learning.
#
#SBATCH --job-name=v3-smoke-train
#SBATCH --partition=gpu,gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/dm1487/logs/v3-smoke-train-%j.out
#SBATCH --error=/scratch/dm1487/logs/v3-smoke-train-%j.err

set -euo pipefail
NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
SAGE_REPO=${SAGE_REPO:-/cache/home/dm1487/projects/namo/sage_learning}
source "$NAMO_REPO/scripts/amarel/activate.sh"

cd "$SAGE_REPO"
RUN_NAME="v3_smoke_$(date +%Y%m%d_%H%M%S)"
OUT_DIR=/scratch/dm1487/sage_outputs/v3_smoke

echo "host=$(hostname)  job=${SLURM_JOB_ID:-N/A}  run=$RUN_NAME"
echo "start=$(date -Is)"
nvidia-smi | head -20 || true

# Train: cross-attn diffusion mask model on local_tight crop, 2 epochs.
PYTHONPATH="$SAGE_REPO" python "$SAGE_REPO/src/train_generative.py" \
    --config-name=train_cropped_diffusion_crossattn \
    name="$RUN_NAME" \
    data_dir=/scratch/dm1487/h5/v3_smoke_train_data \
    output_dir="$OUT_DIR/${RUN_NAME}" \
    max_epochs=2 \
    check_val_every_n_epoch=1 \
    num_workers=4 \
    batch_size=128 \
    use_h5=true \
    weighted_sampling=none \
    crop_prefix=local_tight \
    crop_size=64 \
    context_size=64

echo "end=$(date -Is)"
