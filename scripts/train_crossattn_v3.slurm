#!/bin/bash
# Production training run on the v3 LZF/tight balanced H5.
# Cross-attn diffusion mask model on local_tight 64x64 crop.
#
# Run name = v3_main_<timestamp> (so we can see it in wandb easily).
# Auto-stops at max_epochs OR when SLURM time budget runs out.
#
#SBATCH --job-name=v3-train
#SBATCH --partition=gpu,gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=09:00:00
#SBATCH --output=/scratch/dm1487/logs/v3-train-%j.out
#SBATCH --error=/scratch/dm1487/logs/v3-train-%j.err

set -euo pipefail
NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
SAGE_REPO=${SAGE_REPO:-/cache/home/dm1487/projects/namo/sage_learning}
source "$NAMO_REPO/scripts/amarel/activate.sh"

cd "$SAGE_REPO"
RUN_NAME="v3_main_$(date +%Y%m%d_%H%M%S)"
OUT_DIR=/scratch/dm1487/sage_outputs/v3_main

echo "host=$(hostname)  job=${SLURM_JOB_ID:-N/A}  run=$RUN_NAME"
echo "start=$(date -Is)"
nvidia-smi | head -20 || true

# Train: cross-attn diffusion mask model on local_tight crop.
# - 200 epochs nominal max; SLURM 9h time limit cuts it sooner if needed.
# - val every epoch so we can monitor progress live in wandb.
# - weighted_sampling=none because the H5 is ALREADY balanced 1:1 (per the picker).
PYTHONPATH="$SAGE_REPO" python "$SAGE_REPO/src/train_generative.py" \
    --config-name=train_cropped_diffusion_crossattn \
    name="$RUN_NAME" \
    data_dir=/scratch/dm1487/h5/v3_balanced_1to1_lzf_tight_data \
    output_dir="$OUT_DIR/${RUN_NAME}" \
    max_epochs=200 \
    check_val_every_n_epoch=1 \
    num_workers=8 \
    batch_size=128 \
    use_h5=true \
    weighted_sampling=none \
    crop_prefix=local_tight \
    crop_size=64 \
    context_size=64

echo "end=$(date -Is)"
