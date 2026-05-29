#!/bin/bash
# Parallel-fast variant: reuse the symlink stage that the slow build already
# produced (/scratch/dm1487/h5/v3_balanced_1to1_npz) and re-run convert_to_hdf5
# with:
#   - LZF compression (5-10x faster writes than gzip-4)
#   - --tight-only (drops 10 of 19 large mask arrays we don't train on)
#
# Output is a SEPARATE file so it doesn't clobber the slow build's output:
#   /scratch/dm1487/h5/v3_balanced_1to1_lzf_tight.h5
#
# Picker is skipped entirely (stage dir already exists). Only the H5 write happens.
#
#SBATCH --job-name=v3-h5-lzf-tight
#SBATCH --partition=main-redhat
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/dm1487/logs/v3-h5-lzf-tight-%j.out
#SBATCH --error=/scratch/dm1487/logs/v3-h5-lzf-tight-%j.err

set -euo pipefail
NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
SAGE_REPO=${SAGE_REPO:-/cache/home/dm1487/projects/namo/sage_learning}
source "$NAMO_REPO/scripts/amarel/activate.sh"

STAGE=/scratch/dm1487/h5/v3_balanced_1to1_npz
OUT_DIR=/scratch/dm1487/h5/v3_balanced_1to1_lzf_tight_data
OUT_H5="$OUT_DIR/data.h5"
mkdir -p "$OUT_DIR"
rm -f "$OUT_H5"

if [ ! -d "$STAGE" ]; then
    echo "ERROR: stage dir $STAGE not found. The slow build must have created it." >&2
    exit 1
fi

echo "host=$(hostname)  job=${SLURM_JOB_ID:-N/A}"
echo "reusing stage:  $STAGE  ($(ls "$STAGE" | wc -l) symlinks)"
echo "writing to:     $OUT_H5"
echo "compression:    lzf"
echo "tight-only:     yes (drops local_wide_* keys)"
echo "start=$(date -Is)"

PYTHONPATH="$SAGE_REPO" python "$SAGE_REPO/scripts/convert_to_hdf5.py" \
    "$STAGE" "$OUT_H5" --minimal --tight-only --compression lzf

echo "end=$(date -Is)"
ls -lh "$OUT_H5"

# Quick stats
python -c "
import h5py, numpy as np
f = h5py.File('$OUT_H5','r')
n = f[list(f.keys())[0]].shape[0]
print(f'total samples: {n}')
sd = f['solution_depth'][:]
print(f'solution_depth: {dict(zip(*np.unique(sd, return_counts=True)))}')
print(f'datasets: {len(list(f.keys()))} (tight-only filter applied)')
print(f'  has se2_target_a1? {\"se2_target_a1\" in f}')
print(f'  has pre_pose_a1?   {\"pre_pose_a1\" in f}')
print(f'  has any local_wide_*? {any(k.startswith(\"local_wide_\") for k in f.keys())}')
"
