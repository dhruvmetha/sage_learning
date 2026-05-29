#!/bin/bash
# Build the v3 combined feb+aug9 balanced H5.
# Two steps: (1) pick a 1:1 NPZ stage with primitive_trial_log difficulty
# weighting on the 1-push side; (2) convert that stage to a single H5.
#
# Submit AFTER feb NPZs and aug9 NPZs both exist (chain as a dependency
# of the aug9 NPZ driver):
#
#   sbatch --dependency=afterok:<aug9_npz_driver_jid> \
#       scripts/build_v3_balanced_h5.slurm
#
#SBATCH --job-name=v3-balanced-h5
#SBATCH --partition=main-redhat
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/dm1487/logs/v3-balanced-h5-%j.out
#SBATCH --error=/scratch/dm1487/logs/v3-balanced-h5-%j.err

set -euo pipefail
NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
SAGE_REPO=${SAGE_REPO:-/cache/home/dm1487/projects/namo/sage_learning}
source "$NAMO_REPO/scripts/amarel/activate.sh"

STAGE=/scratch/dm1487/h5/v3_balanced_1to1_npz
OUT_H5=/scratch/dm1487/h5/v3_balanced_1to1.h5

# Clean stage dir (picker errors if it exists)
rm -rf "$STAGE"
rm -f "$OUT_H5"

echo "host=$(hostname)  job=${SLURM_JOB_ID:-N/A}"
echo "stage=$STAGE  out=$OUT_H5"
echo "start=$(date -Is)"

# ---- 1. Pick & symlink ----
PYTHONPATH="$SAGE_REPO" python "$SAGE_REPO/scripts/pick_balanced_npz_v3.py" \
    --pkl-roots \
        /scratch/dm1487/outputs/v3_phase1 \
        /scratch/dm1487/outputs/v3_aug9_phase1 \
    --npz-roots-1push \
        /scratch/dm1487/outputs/v3_phase1_masks \
        /scratch/dm1487/outputs/v3_aug9_phase1_masks \
        /scratch/dm1487/outputs/v3_phase4_masks \
        /scratch/dm1487/outputs/v3_aug9_phase4_masks \
    --npz-roots-2push \
        /scratch/dm1487/outputs/v3_phase2_masks \
        /scratch/dm1487/outputs/v3_phase3_masks \
        /scratch/dm1487/outputs/v3_phase4_masks \
        /scratch/dm1487/outputs/v3_phase5_masks \
        /scratch/dm1487/outputs/v3_aug9_phase2_masks \
        /scratch/dm1487/outputs/v3_aug9_phase3_masks \
        /scratch/dm1487/outputs/v3_aug9_phase4_masks \
        /scratch/dm1487/outputs/v3_aug9_phase5A_masks \
        /scratch/dm1487/outputs/v3_aug9_phase5B_masks \
        /scratch/dm1487/outputs/v3_aug9_phase5C_masks \
        /scratch/dm1487/outputs/v3_aug9_phase5D_masks \
        /scratch/dm1487/outputs/v3_aug9_phase5E_masks \
    --output-dir "$STAGE" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --hard-thresh 0.05 --med-thresh 0.30 \
    --target-hard 0.55 --target-med 0.30 --target-easy 0.15 \
    --seed 42

echo ""
echo "stage size: $(ls "$STAGE" | wc -l) symlinks"
echo "stage stats:"
ls "$STAGE" | awk -F_goal 'NR>0 {if (NF>1 && $2 ~ /_step_/) chain++; else one++} END {print "  *_goal.npz (1-push): " one; print "  *_goal_step_*.npz (2-push): " chain}'

# ---- 2. Convert to H5 (minimal: all the wide+tight keys, no full-res globals) ----
PYTHONPATH="$SAGE_REPO" python "$SAGE_REPO/scripts/convert_to_hdf5.py" \
    "$STAGE" "$OUT_H5" --minimal

echo ""
echo "end=$(date -Is)"
ls -lh "$OUT_H5"
