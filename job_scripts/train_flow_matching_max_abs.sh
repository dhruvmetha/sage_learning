#!/bin/bash
#SBATCH --job-name=flowmatch_maxabs
#SBATCH --output=slurm_logs/flowmatch_maxabs_%j.out
#SBATCH --error=slurm_logs/flowmatch_maxabs_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:a4000:6
#SBATCH --mem=64G

# Flow Matching Training Job (max_abs normalization)
# Usage: sbatch train_flow_matching_max_abs.sh

set -e

source /common/home/tdn39/.virtualenvs/mujoco/bin/activate
export PYTHONPATH=$PYTHONPATH:/common/users/tdn39/Robotics/Mujoco/namo_cpp/build_python

OUTDIR_BASE="/common/users/tdn39/Robotics/Mujoco/sage_learning/outputs"
NORM_MODE="max_abs"
STAMP=$(date +"%Y%m%d_%H%M%S")
OUTDIR="$OUTDIR_BASE/2025-12-27/${NORM_MODE}"
mkdir -p "$OUTDIR"

cd /common/users/tdn39/Robotics/Mujoco/sage_learning

python src/train_generative.py \
  --config-name=train_flow_matching \
  model.norm_mode=max_abs \
  model.pose_stats_file=/common/users/tdn39/Robotics/Mujoco/sage_learning/config/stats_max_abs.json \
  model.overfit_mode=false \
  trainer.max_epochs=1000 \
  data.batch_size=32 \
  data.num_workers=5 \
  wandb_name="flowmatch_max_abs_${STAMP}" \
  > "$OUTDIR/train.log" 2>&1
