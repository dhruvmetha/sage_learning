#!/bin/bash
#SBATCH --job-name=crossattn_warmup_crop64_nominsnr
#SBATCH --partition=gpu-redhat
#SBATCH --constraint=ampere
#SBATCH --exclude=gpu017,gpu018          # skip RTX 3090 nodes
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/dm1487/slurm_logs/crossattn_warmup_%j.out
#SBATCH --error=/scratch/dm1487/slurm_logs/crossattn_warmup_%j.err

set -e
mkdir -p /scratch/dm1487/slurm_logs

echo "=== Job info ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPUs:    $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo

# Conda env activation
source /cache/home/dm1487/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/dm1487/envs/namo
echo "Python: $(which python)"
echo

cd /cache/home/dm1487/projects/namo/sage_learning

python src/train_generative.py \
    --config-name=train_cropped_diffusion_crossattn \
    trainer=multi_gpu \
    data_dir=/scratch/dm1487/h5/car_v1_aug9_depth1_p2_5 \
    output_dir=/scratch/dm1487/runs/car_v1_aug9_depth1_p2_5_crop64_nominsnr_500ep \
    name=warmup_car_v1_aug9_depth1_p2_5_crop64_nominsnr_500ep \
    crop_size=64 \
    batch_size=64 \
    model.use_min_snr=false \
    max_epochs=500 \
    check_val_every_n_epoch=2

echo "=== Done ==="
