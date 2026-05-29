#!/bin/bash
# Build a balanced H5 from data CURRENTLY on disk (don't wait for aug9 phase 5
# to land). Output lives in its own dir so DataModule.find_h5_file resolves
# cleanly via data_dir/data.h5.
#
# Output: /scratch/dm1487/h5/v3_smoke_train_data/data.h5
#
#SBATCH --job-name=v3-balanced-h5-partial
#SBATCH --partition=main-redhat
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/dm1487/logs/v3-balanced-h5-partial-%j.out
#SBATCH --error=/scratch/dm1487/logs/v3-balanced-h5-partial-%j.err

set -euo pipefail
NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
SAGE_REPO=${SAGE_REPO:-/cache/home/dm1487/projects/namo/sage_learning}
source "$NAMO_REPO/scripts/amarel/activate.sh"

STAGE=/scratch/dm1487/h5/v3_balanced_1to1_partial_npz
OUT_DIR=/scratch/dm1487/h5/v3_smoke_train_data
OUT_H5="$OUT_DIR/data.h5"
mkdir -p "$OUT_DIR"
rm -rf "$STAGE" "$OUT_H5"

echo "host=$(hostname)  job=${SLURM_JOB_ID:-N/A}"
echo "stage=$STAGE"
echo "out=$OUT_H5"
echo "start=$(date -Is)"

PYTHONPATH="$SAGE_REPO" python "$SAGE_REPO/scripts/pick_balanced_npz_v3.py" \
    --pkl-roots \
        /scratch/dm1487/outputs/v3_phase1 \
        /scratch/dm1487/outputs/v3_aug9_phase1 \
    --npz-roots-1push \
        /scratch/dm1487/outputs/v3_phase1_masks \
        /scratch/dm1487/outputs/v3_aug9_phase1_masks \
        /scratch/dm1487/outputs/v3_phase4_masks \
    --npz-roots-2push \
        /scratch/dm1487/outputs/v3_phase2_masks \
        /scratch/dm1487/outputs/v3_phase3_masks \
        /scratch/dm1487/outputs/v3_phase4_masks \
        /scratch/dm1487/outputs/v3_phase5_masks \
        /scratch/dm1487/outputs/v3_aug9_phase2_masks \
        /scratch/dm1487/outputs/v3_aug9_phase3_masks \
        /scratch/dm1487/outputs/v3_aug9_phase4_masks \
    --output-dir "$STAGE" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --hard-thresh 0.05 --med-thresh 0.30 \
    --hard-weight 2.0  --med-weight 2.5  --easy-weight 0.5 \
    --seed 42

echo "stage size: $(ls "$STAGE" | wc -l) symlinks"
PYTHONPATH="$SAGE_REPO" python "$SAGE_REPO/scripts/convert_to_hdf5.py" \
    "$STAGE" "$OUT_H5" --minimal
echo "end=$(date -Is)"
ls -lh "$OUT_H5"

# Quick sanity check
python -c "
import h5py, numpy as np
f = h5py.File('$OUT_H5','r')
n = f[list(f.keys())[0]].shape[0]
print(f'total samples: {n}')
print(f'solution_depth: {dict(zip(*np.unique(f[\"solution_depth\"][:], return_counts=True)))}')
print(f'keys (showing tight + key meta only):')
for k in sorted(f.keys()):
    if 'local_tight' in k or k in ('solution_depth','solutions_total','solutions_found','edge_idx_a1','depth_idx_a1','se2_target_a1'):
        print(f'  {k:38s} shape={f[k].shape} dtype={f[k].dtype}')
"
