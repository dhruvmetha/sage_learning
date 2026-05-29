#!/bin/bash
# Tiny smoke H5: 100 1-push + 100 2-push NPZs sampled per phase mask dir.
# Fast (~2 min), enough to validate the dataloader/model/wandb pipeline.
#
# Output: /scratch/dm1487/h5/v3_smoke_train_data/data.h5
#
#SBATCH --job-name=v3-tiny-smoke-h5
#SBATCH --partition=main-redhat
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/dm1487/logs/v3-tiny-smoke-h5-%j.out
#SBATCH --error=/scratch/dm1487/logs/v3-tiny-smoke-h5-%j.err

set -euo pipefail
NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
SAGE_REPO=${SAGE_REPO:-/cache/home/dm1487/projects/namo/sage_learning}
source "$NAMO_REPO/scripts/amarel/activate.sh"

STAGE=/scratch/dm1487/h5/v3_tiny_smoke_npz
OUT_DIR=/scratch/dm1487/h5/v3_smoke_train_data
OUT_H5="$OUT_DIR/data.h5"
mkdir -p "$OUT_DIR"
rm -rf "$STAGE" "$OUT_H5"
mkdir -p "$STAGE"

echo "host=$(hostname)  job=${SLURM_JOB_ID:-N/A}"
echo "stage=$STAGE  out=$OUT_H5"
echo "start=$(date -Is)"

# Stage 100 1-push + 100 2-push (when available) per phase mask dir.
PYTHONPATH="$SAGE_REPO" python - <<'PY'
import glob, os, random
random.seed(42)
STAGE = "/scratch/dm1487/h5/v3_tiny_smoke_npz"
DIRS = [
    "/scratch/dm1487/outputs/v3_phase1_masks",
    "/scratch/dm1487/outputs/v3_phase2_masks",
    "/scratch/dm1487/outputs/v3_phase3_masks",
    "/scratch/dm1487/outputs/v3_phase4_masks",
    "/scratch/dm1487/outputs/v3_phase5_masks",
    "/scratch/dm1487/outputs/v3_aug9_phase1_masks",
    "/scratch/dm1487/outputs/v3_aug9_phase2_masks",
    "/scratch/dm1487/outputs/v3_aug9_phase3_masks",
    "/scratch/dm1487/outputs/v3_aug9_phase4_masks",
]
idx = 0
for d in DIRS:
    if not os.path.isdir(d):
        print(f"  {d}: missing, skip"); continue
    npzs = glob.glob(f"{d}/**/*.npz", recursive=True)
    one_push  = [p for p in npzs if p.endswith("_goal.npz")]
    two_push  = [p for p in npzs if "_goal_step_" in os.path.basename(p)]
    picks = (random.sample(one_push, min(100, len(one_push)))
             + random.sample(two_push, min(100, len(two_push))))
    print(f"  {os.path.basename(d):30s}  picked {len(picks):3d}  (1-push avail {len(one_push)}, 2-push avail {len(two_push)})")
    for src in picks:
        link = os.path.join(STAGE, f"{idx:06d}_{os.path.basename(src)}")
        os.symlink(src, link)
        idx += 1
print(f"\nTOTAL staged: {idx}")
PY

echo ""
echo "stage size: $(ls "$STAGE" | wc -l) symlinks"
PYTHONPATH="$SAGE_REPO" python "$SAGE_REPO/scripts/convert_to_hdf5.py" \
    "$STAGE" "$OUT_H5" --minimal

echo ""; echo "end=$(date -Is)"
ls -lh "$OUT_H5"

# Sanity stats
python -c "
import h5py, numpy as np
f = h5py.File('$OUT_H5','r')
n = f[list(f.keys())[0]].shape[0]
print(f'total samples: {n}')
sd = f['solution_depth'][:]
print(f'solution_depth: {dict(zip(*np.unique(sd, return_counts=True)))}')
"
