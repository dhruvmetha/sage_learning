#!/bin/bash
#SBATCH --job-name=qboot
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=/common/users/dm1487/scratch_namo/logs/qboot-%x-%j.out
#SBATCH --error=/common/users/dm1487/scratch_namo/logs/qboot-%x-%j.err
#
# STAGE 1 (ilab) — single-Q BOOTSTRAPPED value, Horizon DROPPED. ilab twin of
#   namo/scripts/amarel/launch_bootstrap.sh + sage scripts/train_h5_sampling.slurm (lines 56-62).
# Machine-agnostic: every path comes from <parent>/.env via namo scripts/env/activate.sh.
# mix = m2b (1-push openers+dead, value 1.0/0) + ExIt-v4 finish (s1 openers) + bootstrap-SETUP
#       (s0, target = gamma*V_GT(s1)). NoHorizon flags (budget_cond off = single Q). From scratch
#       (grounded targets => stable, no online divergence). VSUMMARY=density (Stage 1) | depth (Stage 3 ctrl).
#   VSUMMARY=density sbatch --job-name=qbden  scripts/train_bootstrap_ilab.slurm
#   VSUMMARY=depth   sbatch --job-name=qbdep  scripts/train_bootstrap_ilab.slurm
#   EPOCHS=2 ...  -> pipeline smoke.

# env first (activate.sh is sourced-safe; do NOT run it under set -e)
source /common/home/dm1487/robotics_research/ktamp/namo/scripts/env/activate.sh
set -euo pipefail
export WANDB_MODE=offline
export WANDB_DIR=/common/users/dm1487/scratch_namo/wandb

H5=/common/users/dm1487/fresh_start/projects/namo/h5
VSUMMARY=${VSUMMARY:-density}
SEED=${SEED:-1}
EPOCHS=${EPOCHS:-200}
NWORKERS=${NWORKERS:-32}   # ilab nodes = 64 cores; ctx is LZF-compressed => dataloading is CPU-bound (8 workers starves the GPU)

M2B=$H5/v4_hq_m2b_scorer/data.h5
EXIT_JOINED=$(ls $H5/v4_hq_exit_finish_v4/shard_*.h5 | sort -V | paste -sd ';' -)
BOOT_JOINED=$(ls $H5/v4_hq_boot_setup_${VSUMMARY}/shard_*.h5 | sort -V | paste -sd ';' -)
NB=$(echo "$BOOT_JOINED" | tr ';' '\n' | grep -c .)
NEX=$(echo "$EXIT_JOINED" | tr ';' '\n' | grep -c .)
[ -f "$M2B" ] && [ "$NB" -ge 1 ] && [ "$NEX" -ge 1 ] || { echo "MISSING data (M2B=$M2B boot=$NB exit=$NEX)"; exit 1; }
DATA_DIR="$M2B;$EXIT_JOINED;$BOOT_JOINED"

RUN="qboot_${VSUMMARY}_s${SEED}"
[ "$EPOCHS" -le 5 ] && RUN="${RUN}_smoke"
OUT="$NAMO_OUTPUTS/scorer/$RUN"
mkdir -p "$OUT" "$NAMO_LOGS"

echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} run=$RUN vsummary=$VSUMMARY epochs=$EPOCHS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || true
echo "boot_shards=$NB exit_shards=$NEX"
echo "start=$(date -Is)"

cd "$NAMO_SAGE"
PYTHONPATH="$NAMO_SAGE" python src/train_classifier.py \
    --config-name=train_scorer_edge \
    name="$RUN" \
    data_dir="'$DATA_DIR'" \
    output_dir="$OUT" \
    max_epochs="$EPOCHS" \
    num_workers="$NWORKERS" \
    +seed="$SEED" \
    +data.sample_seed="$SEED" \
    +data.sample_k=30 \
    +model.bce_reachable_only=true \
    +network.pos_fourier=true \
    +network.use_edge_embed=true \
    +data.budget_h=false +model.head_mode=hl_gauss +network.value_bins=51

echo "end=$(date -Is)"
ls "$OUT"/*/*/checkpoints/*.ckpt 2>/dev/null | tail -3 || true
