#!/bin/bash
# Parameterized stability-sweep arm for the cross-attn diffusion mask model.
# Each arm probes a different fix for the mid-training collapse.
#
# Required env vars at submit:
#   ARM_TAG    short label -> wandb run name + log file
#   LR         base_lr (e.g. 0.0001)
#   BETA2      AdamW beta2 (e.g. 0.95)
#   PRECISION  lightning precision: "bf16-mixed" or "32-true"
#   WARMUP     warmup_steps (e.g. 1000, 5000, 10000)
#
# All arms share: no Min-SNR, local_tight crop 64, batch 128, the v3 LZF H5.
#
#SBATCH --job-name=v3-sweep
#SBATCH --output=/scratch/dm1487/logs/v3-sweep-%j.out
#SBATCH --error=/scratch/dm1487/logs/v3-sweep-%j.err

set -euo pipefail
NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
SAGE_REPO=${SAGE_REPO:-/cache/home/dm1487/projects/namo/sage_learning}
source "$NAMO_REPO/scripts/amarel/activate.sh"

: "${ARM_TAG:?}"; : "${LR:?}"; : "${BETA2:?}"; : "${PRECISION:?}"; : "${WARMUP:?}"

cd "$SAGE_REPO"
RUN_NAME="v3_${ARM_TAG}_$(date +%Y%m%d_%H%M%S)"
OUT_DIR=/scratch/dm1487/sage_outputs/v3_sweep

echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} run=$RUN_NAME"
echo "ARM: lr=$LR beta2=$BETA2 precision=$PRECISION warmup=$WARMUP"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

PYTHONPATH="$SAGE_REPO" python "$SAGE_REPO/src/train_generative.py" \
    --config-name=train_cropped_diffusion_crossattn \
    name="$RUN_NAME" \
    data_dir=/scratch/dm1487/h5/v3_balanced_1to1_lzf_tight_data \
    output_dir="$OUT_DIR/${RUN_NAME}" \
    max_epochs=200 \
    check_val_every_n_epoch=1 \
    num_workers=8 batch_size=128 use_h5=true weighted_sampling=none \
    crop_prefix=local_tight crop_size=64 context_size=64 \
    model.use_min_snr=false \
    base_lr=$LR \
    warmup_steps=$WARMUP \
    "+model.optimizer.betas=[0.9,$BETA2]" \
    trainer.precision=$PRECISION

echo "end=$(date -Is)"
