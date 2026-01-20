#!/bin/bash
#SBATCH --job-name=flowmatch_fusion_xattn_cg_dec2_2push_eval
#SBATCH --output=/common/users/tdn39/Robotics/Mujoco/sage_learning/job_scripts/slurm_logs/flowmatch_fusion_xattn_cg_dec2_2push_eval_%j.out
#SBATCH --error=/common/users/tdn39/Robotics/Mujoco/sage_learning/job_scripts/slurm_logs/flowmatch_fusion_xattn_cg_dec2_2push_eval_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a4000:4
#SBATCH --mem=128G
#SBATCH --ntasks=1

# Train (Fusion Cross-Attention + Coord Grid) on dec2 2-push -> Evaluate on 2-push filtered manifest.
# Usage: sbatch train_eval_flow_matching_fusion_cross_attention_coordgrid_dec2_2push.sh

set -e

source /common/home/tdn39/.virtualenvs/mujoco/bin/activate

SAGE_ROOT="/common/users/tdn39/Robotics/Mujoco/sage_learning"
NAMO_ROOT="/common/users/tdn39/Robotics/Mujoco/namo_cpp"

export PYTHONPATH="$SAGE_ROOT:$NAMO_ROOT/python:$NAMO_ROOT/python/namo/visualization:$NAMO_ROOT/build_python:${PYTHONPATH:-}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_LAUNCH_BLOCKING=0

H5_ROOT="/common/users/shared/robot_learning/dm1487/namo/datasets/h5_files/se2/dec2/aug9_envs/2_push_train_srcsplit"
H5_FILE="${H5_ROOT}/training_data.h5"
STATS_FILE="${H5_ROOT}/stats_max_abs.json"
MANIFEST_FILE="/common/users/shared/robot_learning/dm1487/namo/manifests/aug9_medium/manifest_2push_test_minus_1push_test_filtered.txt"

OUTDIR_BASE="${SAGE_ROOT}/outputs"
ARCH_NAME="fusion_xattn_coordgrid"
NORM_MODE="max_abs_dec2_2push_srcsplit"
STAMP=$(date +"%Y%m%d_%H%M%S")
OUTDIR="${OUTDIR_BASE}/$(date +%Y-%m-%d)/${ARCH_NAME}_${NORM_MODE}_${STAMP}"
mkdir -p "$OUTDIR"

cd "$SAGE_ROOT"

srun python src/train_generative.py \
  --config-name=train_flow_matching_fusion_cross_attention \
  model.norm_mode=max_abs \
  model.pose_stats_file="$STATS_FILE" \
  model.overfit_mode=false \
  data.h5_file="$H5_FILE" \
  data.batch_size=32 \
  data.num_workers=8 \
  data.use_coord_grid=true \
  model.network.image_channels=7 \
  wandb_name="flowmatch_${ARCH_NAME}_${NORM_MODE}_${STAMP}" \
  hydra.run.dir="$OUTDIR" \
  2>&1 | tee "$OUTDIR/train.log"

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
export CUDA_VISIBLE_DEVICES=0

python python/namo/data_collection/sequential_ml_collection.py \
  --config-yaml python/namo/data_collection/eval_vector_model_2push.yaml \
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
