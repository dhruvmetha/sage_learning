#!/bin/bash
# Wait for training to finish, then run evaluation
# This script monitors the training process and auto-runs eval

set -e
cd /common/home/dm1487/robotics_research/ktamp/sage_learning

PYTHON=/common/users/dm1487/envs/mjxrl/bin/python
CKPT_DIR=/common/users/dm1487/namo_data/outputs/classifier/primitive_classifier/namo-classifier
DATA_DIR=/common/users/dm1487/namo_data/f_characterization/classifier_test_npz
EVAL_OUT=/common/home/dm1487/robotics_research/ktamp/namo/docs/f_characterization/eval_results

echo "Waiting for training to finish..."
echo "Monitoring: $CKPT_DIR"

# Wait for training process to exit
while pgrep -f "train_classifier" > /dev/null 2>&1; do
    sleep 30
done

echo "Training finished at $(date)"
echo ""

# Find the best checkpoint
BEST_CKPT=$(find "$CKPT_DIR" -name "epoch*val_loss*.ckpt" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
LAST_CKPT=$(find "$CKPT_DIR" -name "last.ckpt" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT="$LAST_CKPT"
fi

echo "Using checkpoint: $BEST_CKPT"
echo "Eval output: $EVAL_OUT"
echo ""

export CUDA_VISIBLE_DEVICES=0

$PYTHON src/eval_classifier.py \
    --checkpoint "$BEST_CKPT" \
    --data-dir "$DATA_DIR" \
    --output-dir "$EVAL_OUT" \
    --n-heatmaps 5

echo ""
echo "Evaluation complete at $(date)"
echo "Results in: $EVAL_OUT"
