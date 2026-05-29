#!/bin/bash
# Multi-GPU DDP training on the v3 LZF/tight balanced H5.
# Cross-attn diffusion mask model on local_tight 64x64 crop.
#
# Required env vars at submit:
#   GPU_TAG   identifier for wandb run name (e.g. "a100_2gpu", "l40_2gpu")
#
# Submit example (2x A100 4h):
#   GPU_TAG=a100_2gpu sbatch --partition=gpu,gpu-redhat --constraint=ampere \
#     --gres=gpu:2 --cpus-per-task=16 --mem=120G --time=04:00:00 \
#     scripts/train_crossattn_v3_multigpu.slurm
#
#SBATCH --job-name=v3-train-mg
#SBATCH --output=/scratch/dm1487/logs/v3-train-mg-%j.out
#SBATCH --error=/scratch/dm1487/logs/v3-train-mg-%j.err

set -euo pipefail
NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
SAGE_REPO=${SAGE_REPO:-/cache/home/dm1487/projects/namo/sage_learning}
source "$NAMO_REPO/scripts/amarel/activate.sh"

cd "$SAGE_REPO"
TAG=${GPU_TAG:-multigpu}
RUN_NAME="v3_${TAG}_$(date +%Y%m%d_%H%M%S)"
OUT_DIR=/scratch/dm1487/sage_outputs/v3_main

echo "host=$(hostname)  job=${SLURM_JOB_ID:-N/A}  run=$RUN_NAME"
echo "gpus_on_node=${SLURM_GPUS_ON_NODE:-?}"
echo "start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null | head || true

# Cross-attn diffusion mask model, local_tight 64x64.
# Multi-GPU DDP via trainer=multi_gpu (devices=auto → uses all visible GPUs).
# Linear LR scaling: 2 GPUs → 2× base_lr.
PYTHONPATH="$SAGE_REPO" python "$SAGE_REPO/src/train_generative.py" \
    --config-name=train_cropped_diffusion_crossattn \
    trainer=multi_gpu \
    name="$RUN_NAME" \
    data_dir=/scratch/dm1487/h5/v3_balanced_1to1_lzf_tight_data \
    output_dir="$OUT_DIR/${RUN_NAME}" \
    max_epochs=200 \
    check_val_every_n_epoch=1 \
    num_workers=8 \
    batch_size=128 \
    base_lr=0.0008 \
    use_h5=true \
    weighted_sampling=none \
    crop_prefix=local_tight \
    crop_size=64 \
    context_size=64

echo "end=$(date -Is)"
