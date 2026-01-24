#!/bin/bash
#SBATCH --job-name=flowmatch_msxattn_dec2_2push_eval
#SBATCH --output=/common/users/tdn39/Robotics/Mujoco/sage_learning/job_scripts/slurm_logs/flowmatch_msxattn_dec2_2push_eval_%j.out
#SBATCH --error=/common/users/tdn39/Robotics/Mujoco/sage_learning/job_scripts/slurm_logs/flowmatch_msxattn_dec2_2push_eval_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a4000:4
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

# Train (Multi-Scale Cross-Attention) on dec2 2-push -> Evaluate on 2-push filtered manifest.
# Usage: sbatch train_eval_flow_matching_multiscale_cross_attention_dec2_2push.sh
# Resume: sbatch train_eval_flow_matching_multiscale_cross_attention_dec2_2push.sh /path/to/checkpoint.ckpt

set -euo pipefail

# venv activation script references PYTHONPATH; with `set -u` this can error if unset.
set +u
source /common/home/tdn39/.virtualenvs/mujoco/bin/activate
set -u

SAGE_ROOT="/common/users/tdn39/Robotics/Mujoco/sage_learning"
NAMO_ROOT="/common/users/tdn39/Robotics/Mujoco/namo_cpp"

export PYTHONPATH="$SAGE_ROOT:$NAMO_ROOT/python:$NAMO_ROOT/python/namo/visualization:$NAMO_ROOT/build_python:${PYTHONPATH:-}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export HDF5_USE_FILE_LOCKING=FALSE

export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_LAUNCH_BLOCKING=0

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

H5_ROOT="/common/users/shared/robot_learning/dm1487/namo/datasets/h5_files/se2/dec2/aug9_envs/2_push_train_srcsplit"
H5_FILE="${H5_ROOT}/training_data.h5"
STATS_FILE="${H5_ROOT}/stats_max_abs.json"
MANIFEST_FILE="/common/users/shared/robot_learning/dm1487/namo/manifests/aug9_medium/manifest_2push_test_minus_1push_test_filtered.txt"

OUTDIR_BASE="${SAGE_ROOT}/outputs"
ARCH_NAME="multiscale_cross_attn"
NORM_MODE="max_abs_dec2_2push_srcsplit"
STAMP=$(date +"%Y%m%d_%H%M%S")
OUTDIR="${OUTDIR_BASE}/$(date +%Y-%m-%d)/${ARCH_NAME}_${NORM_MODE}_${STAMP}"
mkdir -p "$OUTDIR"

BATCH_SIZE="${BATCH_SIZE:-32}"
# HDF5-backed dataset + DDP can be fragile with many DataLoader workers on NFS.
# Default to single-process loading for stability; override if you want more throughput.
NUM_WORKERS="${NUM_WORKERS:-0}"

cd "$SAGE_ROOT"

srun --ntasks=${SLURM_NTASKS:-4} --kill-on-bad-exit=1 python src/train_generative.py \
  --config-name=train_flow_matching_multiscale_cross_attention \
  trainer.devices=4 \
  trainer.strategy=ddp \
  model.norm_mode=max_abs \
  model.pose_stats_file="$STATS_FILE" \
  model.overfit_mode=false \
  trainer.max_epochs=1000 \
  data.h5_file="$H5_FILE" \
  data.batch_size=${BATCH_SIZE} \
  data.num_workers=${NUM_WORKERS} \
  wandb_name="flowmatch_${ARCH_NAME}_${NORM_MODE}_${STAMP}" \
  hydra.run.dir="$OUTDIR" \
  $RESUME_ARG \
  2>&1 | tee "$OUTDIR/train.log"

if [[ ! -d "$OUTDIR/checkpoints" ]] || ! compgen -G "$OUTDIR/checkpoints/*.ckpt" >/dev/null; then
  echo "ERROR: no checkpoints found under $OUTDIR/checkpoints (training likely failed). Aborting eval." >&2
  exit 1
fi

EVAL_OUTDIR="${OUTDIR}/eval_manifest_2push_${STAMP}"
mkdir -p "$EVAL_OUTDIR"

if [ ! -f "$MANIFEST_FILE" ]; then
  echo "ERROR: Manifest file not found: $MANIFEST_FILE"
  exit 1
fi

END_IDX=$(wc -l < "$MANIFEST_FILE")
END_IDX=${END_IDX//[[:space:]]/}
if [ -z "$END_IDX" ] || [ "$END_IDX" -le 0 ]; then
  echo "ERROR: Manifest file appears empty: $MANIFEST_FILE"
  exit 1
fi

cd "$NAMO_ROOT"

if [ -n "${SLURM_JOB_GPUS:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
fi

EVAL_CONFIG="python/namo/data_collection/eval_vector_model_2push.yaml"
if [[ "${EVAL_HYBRID:-0}" == "1" ]]; then
  EVAL_CONFIG="python/namo/data_collection/eval_vector_model_2push_hybrid.yaml"
fi

python python/namo/data_collection/sequential_ml_collection.py \
  --config-yaml "$EVAL_CONFIG" \
  --output-dir "$EVAL_OUTDIR" \
  --start-idx 0 \
  --end-idx "$END_IDX" \
  --ml-goal-model "$OUTDIR" \
  --ml-sampler-method euler \
  --manifest "$MANIFEST_FILE" \
  --config-file "$NAMO_ROOT/config/namo_config_complete_skill15.yaml" \
  --primitive-data-dir "$NAMO_ROOT/data" \
  --xml-dir "/common/users/shared/robot_learning/dm1487/namo/mj_env_configs/aug9/medium" \
  --ml-device "cuda:0" \
  2>&1 | tee "$EVAL_OUTDIR/eval.log"
