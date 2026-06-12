#!/bin/bash
# H5 SAMPLING ablation (policy_framework journal, hypothesis H5):
#   can we train the scorer on SAMPLED+MASKED labels (non-exhaustive collection — the only thing
#   possible for 2-push) instead of the exhaustive f_grid, and how much does k matter?
#
# 5 conditions x 3 seeds = 15 single-GPU runs (champion sharp recipe, ONLY the loss mask/labels differ;
# all use bce_reachable_only=true incl. the exhaustive control, so BCE scope is NOT a confound):
#   cond 0  Aexh : exhaustive labels (ceiling control)
#   cond 1  B5   : 5  sampled cells/scene, masked loss      (unknown cells excluded)
#   cond 2  B15  : 15 sampled cells/scene, masked loss
#   cond 3  B30  : 30 sampled cells/scene, masked loss
#   cond 4  C15  : 15 sampled cells/scene, unsampled-as-NEGATIVE (the PU-bug baseline)
# array idx -> cond = idx/3, seed = idx%3+1.  SMOKE=1 runs 2 epochs (pipeline check).
# Eval after: scripts/eval_scorer.py (namo_cpp) on namo_testset_v1; verdict via paired seeds.
#
#SBATCH --job-name=h5samp
#SBATCH --partition=gpu,gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-14
#SBATCH --output=/scratch/dm1487/logs/h5samp-%A_%a.out
#SBATCH --error=/scratch/dm1487/logs/h5samp-%A_%a.err

set -euo pipefail
NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
SAGE_REPO=${SAGE_REPO:-/cache/home/dm1487/projects/namo/sage_learning}
source "$NAMO_REPO/scripts/amarel/activate.sh"
cd "$SAGE_REPO"

IDX=${SLURM_ARRAY_TASK_ID:-0}
COND=$(( IDX / 3 ))
SEED=$(( IDX % 3 + 1 ))

case $COND in
  0) NAME="h5samp_Aexh"; EXTRA="+data.sample_k=0" ;;
  1) NAME="h5samp_B5";   EXTRA="+data.sample_k=5" ;;
  2) NAME="h5samp_B15";  EXTRA="+data.sample_k=15" ;;
  3) NAME="h5samp_B30";  EXTRA="+data.sample_k=30" ;;
  4) NAME="h5samp_C15";  EXTRA="+data.sample_k=15 +data.unsampled_negative=true" ;;
esac
RUN="${RUN_PREFIX:-$NAME}_s${SEED}"   # RUN_PREFIX: reuse the recipe on NEW data without colliding with the h5samp registry (e.g. m1_v4hq)
EPOCHS=${EPOCHS:-200}
if [ "${SMOKE:-0}" = "1" ]; then RUN="${RUN}_smoke"; EPOCHS=${SMOKE_EPOCHS:-2}; fi   # SMOKE_EPOCHS: keep the real LR horizon (e.g. 200) for schedule-matched smokes; cap wall-time to stop early
OUT=/scratch/dm1487/sage_outputs/scorer/${RUN}
mkdir -p "$OUT" /scratch/dm1487/logs

echo "host=$(hostname) job=${SLURM_JOB_ID:-NA}_${IDX} run=$RUN cond=$COND seed=$SEED"
echo "start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || true

# Champion sharp recipe (wandb 9yizg6i8) + bce_reachable_only for ALL conditions (no BCE-scope confound).
# sample_seed=SEED so seed variance includes sampling variance (honest for "is k enough").
PYTHONPATH="$SAGE_REPO" python "$SAGE_REPO/src/train_classifier.py" \
    --config-name=train_scorer_edge \
    name="$RUN" \
    data_dir="${DATA_DIR:-/scratch/dm1487/h5/v3_scorer_e4_data}" \
    output_dir="$OUT" \
    max_epochs="$EPOCHS" \
    +seed="$SEED" \
    +data.sample_seed="$SEED" \
    +model.bce_reachable_only=true \
    +network.pos_fourier=true \
    +network.use_edge_embed=true \
    $EXTRA ${EXTRA_OVERRIDES:-}

echo "end=$(date -Is)"
ls "$OUT"/*/*/checkpoints/*.ckpt 2>/dev/null | tail -3 || true
