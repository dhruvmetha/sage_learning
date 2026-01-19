#!/bin/bash
#SBATCH --job-name=flowmatch_meanstd
#SBATCH --output=slurm_logs/flowmatch_meanstd_%j.out
#SBATCH --error=slurm_logs/flowmatch_meanstd_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a4000:4
#SBATCH --mem=128G
#SBATCH --ntasks=1

# Flow Matching Training Job (mean_std normalization)
# Usage: sbatch train_flow_matching_mean_std.sh
# Resume: sbatch train_flow_matching_mean_std.sh /path/to/checkpoint.ckpt
# 
# ============================================================================
# RESOURCE SCALING RATIOS (for future reference):
# ============================================================================
# Per GPU:
#   - CPUs: 8 per GPU (e.g., 4 GPUs = 32 CPUs)
#   - Memory: 32GB per GPU (e.g., 4 GPUs = 128GB)
#   - Workers: 2-4 per GPU (set in data.num_workers, Lightning shares across GPUs)
#
# Examples:
#   1 GPU:  --cpus-per-task=8,  --mem=32G,  num_workers=4
#   2 GPUs: --cpus-per-task=16, --mem=64G,  num_workers=4
#   4 GPUs: --cpus-per-task=32, --mem=128G, num_workers=4
#   6 GPUs: --cpus-per-task=48, --mem=192G, num_workers=4
# ============================================================================

set -e

source /common/home/tdn39/.virtualenvs/mujoco/bin/activate
export PYTHONPATH=$PYTHONPATH:/common/users/tdn39/Robotics/Mujoco/namo_cpp/build_python

# ============================================================================
# NCCL Settings for Stability
# ============================================================================
export NCCL_DEBUG=WARN                    # Set to INFO for verbose debugging
export NCCL_TIMEOUT=1800                  # 30 min timeout (in seconds)
export NCCL_IB_DISABLE=1                  # Disable InfiniBand (use ethernet)
export NCCL_P2P_DISABLE=0                 # Keep P2P enabled for multi-GPU
export TORCH_NCCL_BLOCKING_WAIT=1         # Better error messages on timeout
export CUDA_LAUNCH_BLOCKING=0             # Keep async for performance

# Handle resume from checkpoint
CHECKPOINT_PATH="${1:-}"
RESUME_ARG=""
if [ -n "$CHECKPOINT_PATH" ]; then
  if [ -f "$CHECKPOINT_PATH" ]; then
    echo "Resuming from checkpoint: $CHECKPOINT_PATH"
    RESUME_ARG="+ckpt_path='$CHECKPOINT_PATH'"
  else
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
  fi
fi

OUTDIR_BASE="/common/users/tdn39/Robotics/Mujoco/sage_learning/outputs"
NORM_MODE="mean_std"
STAMP=$(date +"%Y%m%d_%H%M%S")
OUTDIR="$OUTDIR_BASE/$(date +%Y-%m-%d)/${NORM_MODE}"
mkdir -p "$OUTDIR"

cd /common/users/tdn39/Robotics/Mujoco/sage_learning

# Use srun for proper SLURM + DDP integration
srun python src/train_generative.py \
  --config-name=train_flow_matching \
  model.norm_mode=$NORM_MODE \
  model.pose_stats_file=/common/users/tdn39/Robotics/Mujoco/sage_learning/config/stats_mean_std.json \
  model.overfit_mode=false \
  trainer.max_epochs=1000 \
  data.batch_size=32 \
  wandb_name="flowmatch_mean_std_${STAMP}" \
  hydra.run.dir="$OUTDIR" \
  $RESUME_ARG \
  2>&1 | tee "$OUTDIR/train.log"
