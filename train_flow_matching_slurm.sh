#!/bin/bash
#SBATCH --job-name=flow_match_1k
#SBATCH --output=/common/users/tdn39/Robotics/Mujoco/sage_learning/slurm_logs/flow_match_%j.out
#SBATCH --error=/common/users/tdn39/Robotics/Mujoco/sage_learning/slurm_logs/flow_match_%j.err
#SBATCH --time=12:00:00
#SBATCH --gpus=a4500:6
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G

# Script to run full training for 1000 epochs
# Usage: sbatch train_flow_matching_slurm.sh

set -e  # Exit on error

# Configuration
SCRIPT_DIR="/common/users/tdn39/Robotics/Mujoco/sage_learning"
DATA_DIR="/common/users/shared/robot_learning/dm1487/namo/datasets/images/nov22/train/1_push"

echo "=== Starting Training Job ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Data Directory: ${DATA_DIR}"
echo ""

# Create logs directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/slurm_logs"

# Set up temp directory to avoid NFS issues
MYTMP="/common/users/tdn39/.tmp"
mkdir -p "${MYTMP}"
export TMPDIR="${MYTMP}"
export TEMP="${MYTMP}"
export TMP="${MYTMP}"

# Activate virtualenv
if [ -f "${HOME}/.local/bin/virtualenvwrapper.sh" ]; then
    source "${HOME}/.local/bin/virtualenvwrapper.sh"
    workon mujoco
elif [ -f "${HOME}/.virtualenvs/mujoco/bin/activate" ]; then
    source "${HOME}/.virtualenvs/mujoco/bin/activate"
else
    echo "Error: Cannot find virtualenv"
    exit 1
fi
echo "✓ Virtualenv activated: ${VIRTUAL_ENV}"

# Run training
# Note: Multi-GPU is configured in the config files (devices='auto', strategy='ddp')
# We override max_epochs and data_dir here.
echo "Running training command..."
python "${SCRIPT_DIR}/src/train_generative.py" \
    --config-name=train_flow_matching \
    data.data_dir="${DATA_DIR}" \
    max_epochs=1000

echo "✓ Training completed"
