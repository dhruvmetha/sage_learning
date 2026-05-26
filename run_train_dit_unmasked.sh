#!/bin/bash
# Train DiT classifier with BCE+Dice unmasked loss on training set
# Then auto-eval on test set
set -e
cd /common/home/dm1487/robotics_research/ktamp/sage_learning

PYTHON=/common/users/dm1487/envs/mjxrl/bin/python
TRAIN_NPZ=/common/users/dm1487/namo_data/f_characterization/classifier_train_npz
TEST_NPZ=/common/users/dm1487/namo_data/f_characterization/classifier_test_npz
OUT=/common/users/dm1487/namo_data/outputs/classifier/classifier_trainset_dit_dice_unmasked
EVAL_OUT=/common/home/dm1487/robotics_research/ktamp/namo/docs/f_characterization/eval_trainset_dit_dice_unmasked

echo "=== Training DiT + BCE+Dice (unmasked) ==="
echo "Train data: $TRAIN_NPZ"
echo "Start: $(date)"

export WANDB_DIR=$OUT
export WANDB_CACHE_DIR=$OUT/wandb_cache
export TMPDIR=/common/users/dm1487/tmp
mkdir -p $TMPDIR $WANDB_CACHE_DIR

srun --gres=gpu:1 --time=06:00:00 --mem=16G --export=ALL,WANDB_DIR=$OUT,WANDB_CACHE_DIR=$OUT/wandb_cache,TMPDIR=/common/users/dm1487/tmp,WANDB_MODE=offline bash -c "
cd /common/home/dm1487/robotics_research/ktamp/sage_learning
$PYTHON src/train_classifier.py \
    --config-name=train_classifier \
    name=classifier_trainset_dit_dice_unmasked \
    data_dir=$TRAIN_NPZ \
    output_dir=$OUT \
    batch_size=64 \
    num_workers=4 \
    hydra.run.dir=$OUT/hydra
"

echo ""
echo "Training done: $(date)"
echo ""

# Find best checkpoint
BEST_CKPT=\$(find $OUT -name "epoch*val_loss*.ckpt" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
LAST_CKPT=\$(find $OUT -name "last.ckpt" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
[ -z "\$BEST_CKPT" ] && BEST_CKPT="\$LAST_CKPT"

echo "=== Evaluating on test set ==="
echo "Checkpoint: \$BEST_CKPT"
echo "Test data: $TEST_NPZ"

srun --gres=gpu:1 --time=00:30:00 --mem=8G bash -c "
cd /common/home/dm1487/robotics_research/ktamp/sage_learning
$PYTHON src/eval_classifier_detailed.py \
    --checkpoint \$BEST_CKPT \
    --data-dir $TEST_NPZ \
    --output-dir $EVAL_OUT \
    --n-heatmaps 5
"

echo ""
echo "=== All done: $(date) ==="
echo "Results: $EVAL_OUT"
