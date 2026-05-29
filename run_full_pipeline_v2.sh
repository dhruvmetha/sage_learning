#!/bin/bash
# Full pipeline: wait for rlab7 collection → generate masks → train → eval
# Run in screen on westeros: screen -S full_pipeline bash run_full_pipeline_v2.sh

set -e

PYTHON=/common/users/dm1487/envs/mjxrl/bin/python
NAMO_DIR=/common/home/dm1487/robotics_research/ktamp/namo
SAGE_DIR=/common/home/dm1487/robotics_research/ktamp/sage_learning

TRAIN_PKL_DIR_RLAB5=/common/users/dm1487/namo_data/f_characterization/1_push_exhaustive_train/modular_data_rlab5
TRAIN_PKL_DIR_RLAB6=/common/users/dm1487/namo_data/f_characterization/1_push_exhaustive_train/modular_data_rlab6
TRAIN_PKL_DIR_RLAB7=/common/users/dm1487/namo_data/f_characterization/1_push_exhaustive_train/modular_data_rlab7
TRAIN_NPZ_DIR=/common/users/dm1487/namo_data/f_characterization/classifier_train_npz
TEST_NPZ_DIR=/common/users/dm1487/namo_data/f_characterization/classifier_test_npz
EVAL_OUT=$NAMO_DIR/docs/f_characterization/eval_results_trainset

export CUDA_VISIBLE_DEVICES=0

echo "============================================================"
echo "FULL CLASSIFIER PIPELINE v2"
echo "Started: $(date)"
echo "============================================================"

# ── Step 1: Wait for rlab7 training data collection ──
echo ""
echo "[Step 1/4] Waiting for training data collection on rlab7..."

while true; do
    rlab7_running=$(ssh rlab7.cs.rutgers.edu "screen -ls 2>/dev/null | grep f_char_train | grep -v Dead | wc -l" 2>/dev/null || echo "1")

    rlab5_count=$(ls "$TRAIN_PKL_DIR_RLAB5"/*_results.pkl 2>/dev/null | wc -l || echo "0")
    rlab6_count=$(ls "$TRAIN_PKL_DIR_RLAB6"/*_results.pkl 2>/dev/null | wc -l || echo "0")
    rlab7_count=$(ls "$TRAIN_PKL_DIR_RLAB7"/*_results.pkl 2>/dev/null | wc -l || echo "0")
    total=$((rlab5_count + rlab6_count + rlab7_count))

    echo "  $(date +%H:%M) — rlab5=$rlab5_count rlab6=$rlab6_count rlab7=$rlab7_count total=$total (rlab7 running=$rlab7_running)"

    if [ "$rlab7_running" = "0" ]; then
        echo "  rlab7 collection finished!"
        break
    fi

    sleep 300
done

echo "  Final: rlab5=$rlab5_count rlab6=$rlab6_count rlab7=$rlab7_count total=$total"

# ── Step 2: Generate training NPZ masks ──
echo ""
echo "[Step 2/4] Generating classifier training NPZ..."
echo "  Output: $TRAIN_NPZ_DIR"
echo "  Started: $(date)"

cd "$NAMO_DIR"

# Collect from all three machines' data
INPUT_DIRS=""
[ -d "$TRAIN_PKL_DIR_RLAB5" ] && INPUT_DIRS="$INPUT_DIRS $TRAIN_PKL_DIR_RLAB5"
[ -d "$TRAIN_PKL_DIR_RLAB6" ] && INPUT_DIRS="$INPUT_DIRS $TRAIN_PKL_DIR_RLAB6"
[ -d "$TRAIN_PKL_DIR_RLAB7" ] && INPUT_DIRS="$INPUT_DIRS $TRAIN_PKL_DIR_RLAB7"

$PYTHON -m namo.visualization.mask_generation.batch_collection_classifier \
    --input-dir $INPUT_DIRS \
    --output-dir "$TRAIN_NPZ_DIR" \
    --workers 16

train_npz_count=$(find "$TRAIN_NPZ_DIR" -name "*.npz" | wc -l)
echo "  Generated $train_npz_count training NPZ files"
echo "  Finished: $(date)"

# ── Step 3: Train classifier ──
echo ""
echo "[Step 3/4] Training classifier on training set..."
echo "  Data: $TRAIN_NPZ_DIR ($train_npz_count files)"
echo "  Started: $(date)"

cd "$SAGE_DIR"
$PYTHON src/train_classifier.py \
    --config-name=train_classifier \
    name=classifier_trainset \
    data_dir="$TRAIN_NPZ_DIR" \
    output_dir=/common/users/dm1487/namo_data/outputs/classifier/classifier_trainset \
    batch_size=64 \
    max_epochs=200 \
    num_workers=4

echo "  Training finished: $(date)"

# ── Step 4: Evaluate on test set ──
echo ""
echo "[Step 4/4] Evaluating on test set..."
echo "  Test data: $TEST_NPZ_DIR"
echo "  Output: $EVAL_OUT"
echo "  Started: $(date)"

CKPT_DIR=/common/users/dm1487/namo_data/outputs/classifier/classifier_trainset
BEST_CKPT=$(find "$CKPT_DIR" -name "epoch*val_loss*.ckpt" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
LAST_CKPT=$(find "$CKPT_DIR" -name "last.ckpt" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT="$LAST_CKPT"
fi

echo "  Using checkpoint: $BEST_CKPT"

$PYTHON src/eval_classifier.py \
    --checkpoint "$BEST_CKPT" \
    --data-dir "$TEST_NPZ_DIR" \
    --output-dir "$EVAL_OUT" \
    --n-heatmaps 5

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "Finished: $(date)"
echo ""
echo "Results:"
echo "  Prototype (trained on test):  $NAMO_DIR/docs/f_characterization/eval_results/"
echo "  Proper (trained on train):    $EVAL_OUT"
echo "============================================================"
