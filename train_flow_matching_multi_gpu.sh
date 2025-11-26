#!/bin/bash
#SBATCH --job-name=fm_multi
#SBATCH --output=slurm_logs/fm_multi_%j.out
#SBATCH --error=slurm_logs/fm_multi_%j.err
#SBATCH --time=24:00:00
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Flow Matching Training - Multi GPU
# Usage: sbatch train_flow_matching_multi_gpu.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/common/users/shared/robot_learning/dm1487/namo/datasets/images/nov22/train/1_push"

echo "=== Flow Matching Training (Multi GPU) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo ""

mkdir -p "${SCRIPT_DIR}/slurm_logs"

# Activate conda environment
source /common/home/dm1487/.bashrc
conda activate /common/users/dm1487/envs/mjxrl

echo "Environment: $(which python)"

cd "${SCRIPT_DIR}"
python src/train_generative.py \
    --config-name=train_fb_flow_matching \
    trainer=multi_gpu \
    data.data_dir="${DATA_DIR}" \
    max_epochs=1000

echo "Training completed"
