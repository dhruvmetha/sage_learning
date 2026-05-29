#!/bin/bash
# Generic diffusion training launcher for v2 data.
# Set USE_MIN_SNR=true/false, DATA_H5, OUTPUT_DIR, NAME via env vars at submit.
#SBATCH --partition=gpu-redhat
#SBATCH --constraint=ampere
#SBATCH --exclude=gpu017,gpu018          # skip 3090 nodes; keeps us on A100s (the ampere GPUs we have)
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/dm1487/slurm_logs/diffusion_v2_%j.out
#SBATCH --error=/scratch/dm1487/slurm_logs/diffusion_v2_%j.err

set -e
mkdir -p /scratch/dm1487/slurm_logs

: "${USE_MIN_SNR:?must set USE_MIN_SNR=true|false}"
: "${DATA_H5:?must set DATA_H5=/scratch/dm1487/h5/<name> (without .h5 — loader will find <name>.h5)}"
: "${OUTPUT_DIR:?must set OUTPUT_DIR}"
: "${NAME:?must set NAME}"

echo "=== Job info ==="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "GPUs visible:  $CUDA_VISIBLE_DEVICES"
echo "USE_MIN_SNR:   $USE_MIN_SNR"
echo "DATA_H5:       $DATA_H5"
echo "OUTPUT_DIR:    $OUTPUT_DIR"
echo "NAME:          $NAME"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo

source /cache/home/dm1487/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/dm1487/envs/namo
echo "Python: $(which python)"
echo

cd /cache/home/dm1487/projects/namo/sage_learning

python src/train_generative.py \
    --config-name=train_cropped_diffusion_crossattn \
    trainer=multi_gpu \
    data_dir="$DATA_H5" \
    output_dir="$OUTPUT_DIR" \
    name="$NAME" \
    crop_size=64 \
    batch_size=64 \
    model.use_min_snr="$USE_MIN_SNR" \
    max_epochs=500 \
    check_val_every_n_epoch=2

echo "=== Done ==="
