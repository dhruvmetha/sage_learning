#!/bin/bash
# SE(2) diffusion training on a single A100 (~500 epochs).
#
# Required env vars (set via --export at sbatch time):
#   H5             /scratch/dm1487/h5/car_v2_dualcrop_balanced_*.h5
#   RUN_NAME       wandb run name
#
# Optional:
#   MAX_EPOCHS     default 500
#   BATCH_SIZE     default 512
#
#SBATCH --job-name=se2-train
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/dm1487/logs/se2-train-%j.out
#SBATCH --error=/scratch/dm1487/logs/se2-train-%j.err

set -euo pipefail
NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
SAGE_REPO=${SAGE_REPO:-/cache/home/dm1487/projects/namo/sage_learning}
source "$NAMO_REPO/scripts/amarel/activate.sh"

: "${H5:?must set H5}"
: "${RUN_NAME:?must set RUN_NAME}"

MAX_EPOCHS=${MAX_EPOCHS:-500}
BATCH_SIZE=${BATCH_SIZE:-512}
ENV_FAMILY_FILTER=${ENV_FAMILY_FILTER:-}

mkdir -p /scratch/dm1487/logs

echo "host=$(hostname)  job=${SLURM_JOB_ID:-N/A}"
echo "H5: $H5"
echo "RUN_NAME: $RUN_NAME"
echo "EPOCHS: $MAX_EPOCHS  BATCH: $BATCH_SIZE"
echo "start: $(date -Is)"

cd "$SAGE_REPO"
EXTRA_ARGS=()
if [ -n "$ENV_FAMILY_FILTER" ]; then
    EXTRA_ARGS+=(--env-family-filter "$ENV_FAMILY_FILTER")
fi

PYTHONPATH="$SAGE_REPO" python scripts/train_se2.py \
    --h5 "$H5" \
    --max-epochs "$MAX_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers 8 \
    --gpus 1 \
    --precision bf16-mixed \
    --warmup-steps 500 \
    --wandb-project namo-se2 \
    --wandb-run-name "$RUN_NAME" \
    "${EXTRA_ARGS[@]}"

echo "end: $(date -Is)"
