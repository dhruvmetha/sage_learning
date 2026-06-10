#!/bin/bash
# H2 — EDGE SELF-ATTENTION ablation (policy_framework journal), 1-push only.
# Question [USER]: does inter-edge self-attention help or hurt the score map — and does the answer
# FLIP under masked/sampled labels (sparse supervision should favor independent edges)?
#
# 2x2 design, but the self-attn=ON arms are H5's runs (h5samp_Aexh_s{1..3} = ON x exhaustive,
# h5samp_B15_s{1..3} = ON x masked-15 — same recipe/seeds). This script trains ONLY the OFF arms:
#   cond 0  h2_noattn_exh : edge_self_attn=false, exhaustive labels
#   cond 1  h2_noattn_k15 : edge_self_attn=false, sampled-15 masked labels
# array idx -> cond = idx/3, seed = idx%3+1 (6 runs).  SMOKE=1 -> 2 epochs.
# All runs bce_reachable_only=true (same confound control as H5).
#
#SBATCH --job-name=h2attn
#SBATCH --partition=gpu,gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/dm1487/logs/h2attn-%A_%a.out
#SBATCH --error=/scratch/dm1487/logs/h2attn-%A_%a.err

set -euo pipefail
NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
SAGE_REPO=${SAGE_REPO:-/cache/home/dm1487/projects/namo/sage_learning}
source "$NAMO_REPO/scripts/amarel/activate.sh"
cd "$SAGE_REPO"

IDX=${SLURM_ARRAY_TASK_ID:-0}
COND=$(( IDX / 3 ))
SEED=$(( IDX % 3 + 1 ))

case $COND in
  0) NAME="h2_noattn_exh"; EXTRA="+data.sample_k=0" ;;
  1) NAME="h2_noattn_k15"; EXTRA="+data.sample_k=15" ;;
esac
RUN="${NAME}_s${SEED}"
EPOCHS=${EPOCHS:-200}
if [ "${SMOKE:-0}" = "1" ]; then RUN="${RUN}_smoke"; EPOCHS=2; fi
OUT=/scratch/dm1487/sage_outputs/scorer/${RUN}
mkdir -p "$OUT" /scratch/dm1487/logs

echo "host=$(hostname) job=${SLURM_JOB_ID:-NA}_${IDX} run=$RUN cond=$COND seed=$SEED"
echo "start=$(date -Is)"

PYTHONPATH="$SAGE_REPO" python "$SAGE_REPO/src/train_classifier.py" \
    --config-name=train_scorer_edge \
    name="$RUN" \
    data_dir=/scratch/dm1487/h5/v3_scorer_e4_data \
    output_dir="$OUT" \
    max_epochs="$EPOCHS" \
    +seed="$SEED" \
    +data.sample_seed="$SEED" \
    +model.bce_reachable_only=true \
    +network.pos_fourier=true \
    +network.use_edge_embed=true \
    +network.edge_self_attn=false \
    $EXTRA

echo "end=$(date -Is)"
ls "$OUT"/*/*/checkpoints/*.ckpt 2>/dev/null | tail -2 || true
