#!/bin/bash
# _step_penalty experiment — NoHorizon retrain with the SIGNED target scheme, on ARRAKIS (direct GPUs, no SLURM).
#
# SAME v3 (car) data mix + SAME recipe as the registered NoHz-v3 (qfull_nohz_v3_v4hq); the ONLY delta is the
# target relabel: immediate open +1 / valid setup 0 / never -1  (was 1 / 0.9 / 0), which needs the HL-Gauss
# range widened to [-1,1]. Implemented via the two new flags:
#     +data.target_scheme=signed +model.value_vmin=-1.0 +model.value_vmax=1.0
# 3 seeds pinned to GPUs 1,2,3 (GPU 0 is in use by another job; 4 left spare). Logs + ckpts under $OUT.
#   bash scripts/train_steppen_arrakis.sh
set -euo pipefail

SAGE=/common/home/dm1487/robotics_research/ktamp/sage_learning
H5=/common/users/dm1487/scratch_namo/h5
OUTROOT=/common/users/dm1487/scratch_namo/sage_outputs/scorer
PY=/common/users/dm1487/envs/mjxrl/bin/python
GPUS=(1 2 3)                 # one GPU per seed; seed s uses GPUS[s-1]
NWORKERS=${NWORKERS:-10}     # 3 concurrent x 10 = 30 of 32 cores (ctx is LZF => dataloader-bound)
EPOCHS=${EPOCHS:-200}        # early-stops ~ep12-25 (patience 25); NoHz-v3 best-val was ep12

# ---- v3 (car) data mix — exact order recovered from the NoHz-v3 wandb data_dir ----
M2B=$H5/v4_hq_m2b_scorer/data.h5
H2=$H5/v4_hq_h2_scorer/data.h5
AUG=$H5/v4_hq_onepush_h2_aug/data.h5
EXIT=$(ls $H5/v4_hq_exit_finish/shard_*.h5 | sort -V | paste -sd ';' -)
EXITV=$(ls $H5/v4_hq_exit_finish_valid/shard_*.h5 | sort -V | paste -sd ';' -)
for f in "$M2B" "$H2" "$AUG"; do [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }; done
[ -n "$EXIT" ] && [ -n "$EXITV" ] || { echo "MISSING exit shards"; exit 1; }
DATA_DIR="$M2B;$H2;$AUG;$EXIT;$EXITV"

cd "$SAGE"
mkdir -p "$OUTROOT"
for SEED in 1 2 3; do
  GPU=${GPUS[$((SEED-1))]}
  RUN="qfull_nohz_steppen_v3_s${SEED}"
  OUT="$OUTROOT/$RUN"
  mkdir -p "$OUT"
  echo "launch $RUN on GPU $GPU -> $OUT"
  CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=offline WANDB_DIR="$OUT" PYTHONPATH="$SAGE" \
    nohup "$PY" src/train_classifier.py \
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
      +data.budget_h=false +model.head_mode=hl_gauss +network.value_bins=51 \
      +data.target_scheme=signed +model.value_vmin=-1.0 +model.value_vmax=1.0 \
      > "$OUT/train.log" 2>&1 &
  echo "  pid=$! log=$OUT/train.log"
  sleep 2
done
echo "all 3 seeds launched (nohup, detached — survive shell exit)"
