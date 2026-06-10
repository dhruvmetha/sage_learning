#!/bin/bash
# H1 FRAMING ablation (policy_framework journal): is the (60,5) head a per-cell VALUE (sigmoid-BCE,
# the scorer) or a POLICY (softmax-CE over reachable cells)? × sharp vs soft targets.
# 2x2 x 3 matched seeds = 12 runs, champion sharp recipe + E4 data; ONLY the head/target package differs:
#   cond 0  C1 sig_sharp : sigmoid-BCE+Dice, hard f_grid      (champion replica — the control)
#   cond 1  C2 sig_soft  : sigmoid-BCE+Dice, Gaussian-soft    (re-test of the rejected soft)
#   cond 2  C3 pol_sharp : softmax-CE, normalize(f_grid)      (policy, uniform over solutions)
#   cond 3  C4 pol_soft  : softmax-CE, normalized soft blur   (hypothesized winner)
# array idx -> cond = idx/3, seed = idx%3+1.  SMOKE=1 -> 2 epochs.
# NOTE [journaled]: C1/C2 keep the champion loss package (BCE all-600 + Dice); C3/C4 are softmax over
# reachable (legal-move masking) — H1 compares PACKAGES, not isolated terms (red-team sigma-sweep caveat
# also journaled; sigma fixed at 1.0/1.0 this round).
#
#SBATCH --job-name=h1frame
#SBATCH --partition=gpu,gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-11
#SBATCH --output=/scratch/dm1487/logs/h1frame-%A_%a.out
#SBATCH --error=/scratch/dm1487/logs/h1frame-%A_%a.err

set -euo pipefail
NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
SAGE_REPO=${SAGE_REPO:-/cache/home/dm1487/projects/namo/sage_learning}
source "$NAMO_REPO/scripts/amarel/activate.sh"
cd "$SAGE_REPO"

IDX=${SLURM_ARRAY_TASK_ID:-0}
COND=$(( IDX / 3 ))
SEED=$(( IDX % 3 + 1 ))

case $COND in
  0) NAME="h1_sig_sharp"; EXTRA="" ;;
  1) NAME="h1_sig_soft";  EXTRA="soft_edge_sigma=1.0 soft_depth_sigma=1.0" ;;
  2) NAME="h1_pol_sharp"; EXTRA="+model.head_mode=softmax_ce" ;;
  3) NAME="h1_pol_soft";  EXTRA="+model.head_mode=softmax_ce soft_edge_sigma=1.0 soft_depth_sigma=1.0" ;;
esac
RUN="${NAME}_s${SEED}"
EPOCHS=${EPOCHS:-200}
if [ "${SMOKE:-0}" = "1" ]; then RUN="${RUN}_smoke"; EPOCHS=2; fi
OUT=/scratch/dm1487/sage_outputs/scorer/${RUN}
mkdir -p "$OUT" /scratch/dm1487/logs

echo "host=$(hostname) job=${SLURM_JOB_ID:-NA}_${IDX} run=$RUN cond=$COND seed=$SEED"
echo "start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true

PYTHONPATH="$SAGE_REPO" python "$SAGE_REPO/src/train_classifier.py" \
    --config-name=train_scorer_edge \
    name="$RUN" \
    data_dir=/scratch/dm1487/h5/v3_scorer_e4_data \
    output_dir="$OUT" \
    max_epochs="$EPOCHS" \
    +seed="$SEED" \
    +network.pos_fourier=true \
    +network.use_edge_embed=true \
    $EXTRA

echo "end=$(date -Is)"
ls "$OUT"/*/*/checkpoints/*.ckpt 2>/dev/null | tail -2 || true
