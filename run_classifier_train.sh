#!/bin/bash
# Train primitive classifier on test set data (prototype)
# Run in screen: screen -S classifier bash run_classifier_train.sh

set -e

cd /common/home/dm1487/robotics_research/ktamp/sage_learning

export CUDA_VISIBLE_DEVICES=0

echo "Training primitive classifier"
echo "Data: /common/users/dm1487/namo_data/f_characterization/classifier_test_npz"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo ""

/common/users/dm1487/envs/mjxrl/bin/python src/train_classifier.py \
    --config-name=train_classifier \
    data_dir=/common/users/dm1487/namo_data/f_characterization/classifier_test_npz \
    batch_size=64 \
    max_epochs=200 \
    num_workers=4

echo ""
echo "Finished: $(date)"
