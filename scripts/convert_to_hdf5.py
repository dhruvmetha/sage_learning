#!/usr/bin/env python3
"""
Convert directory of NPZ files to a single HDF5 file for faster training.

Usage:
    python scripts/convert_to_hdf5.py /path/to/npz_dir /path/to/output.h5

    # Minimal mode: only local masks + xml_file (smaller file)
    python scripts/convert_to_hdf5.py /path/to/npz_dir /path/to/output.h5 --minimal

This consolidates 100k+ individual NPZ files into a single HDF5 file,
dramatically reducing startup time and improving I/O performance.
"""

import argparse
import glob
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


def get_npz_keys(sample_file: str) -> list:
    """Get the keys from a sample NPZ file."""
    with np.load(sample_file) as data:
        return list(data.keys())


def find_file_with_max_keys(npz_files: list, target_keys: list = None, sample_size: int = 100) -> str:
    """Find an NPZ file that has the most keys (or all target keys).

    For multi-horizon data, step_0 files have more goal masks than step_1 files.
    This ensures we find a file with the full schema.

    Args:
        npz_files: List of NPZ file paths
        target_keys: Optional list of keys we want to find
        sample_size: Number of files to sample

    Returns:
        Path to NPZ file with most keys
    """
    # First, try to find a step_0 file (these have all goal horizons)
    for f in npz_files[:sample_size]:
        if '_step_0.' in f or '_step_0_' in f:
            keys = get_npz_keys(f)
            if target_keys is None or all(k in keys for k in target_keys if 'goal_mask' not in k):
                return f

    # Fallback: sample files and find one with most keys
    best_file = npz_files[0]
    best_key_count = 0

    sample_indices = np.linspace(0, len(npz_files) - 1, min(sample_size, len(npz_files)), dtype=int)
    for idx in sample_indices:
        f = npz_files[idx]
        keys = get_npz_keys(f)
        if len(keys) > best_key_count:
            best_key_count = len(keys)
            best_file = f

    return best_file


def is_string_dtype(dtype) -> bool:
    """Check if dtype is a string type."""
    return dtype.kind in ('U', 'S', 'O')  # Unicode, byte string, or object


MINIMAL_KEYS = [
    # === Wide crop (1.2 m) — mask-prediction supervision ===
    'local_wide_static',
    'local_wide_movable',
    'local_wide_target_object',
    'local_wide_target_goal',
    'local_wide_robot_region',
    'local_wide_goal_sample_region',
    'local_wide_goal_mask_a1',     # single-horizon target
    'local_wide_robot',            # sampled robot position in robot_region
    'local_wide_goal',             # sampled goal position in goal_sample_region
    'local_wide_goal_samples',     # multiple sampled goal positions
    'local_wide_crop_size_meters',
    'local_wide_resolution',
    'local_wide_object_center',
    'local_wide_object_theta',
    'local_wide_bounds',

    # === Tight crop (0.5 m) — SE(2)/index-prediction supervision ===
    'local_tight_static',
    'local_tight_movable',
    'local_tight_target_object',
    'local_tight_target_goal',
    'local_tight_robot_region',
    'local_tight_goal_sample_region',
    'local_tight_robot',
    'local_tight_goal',
    'local_tight_goal_samples',
    'local_tight_crop_size_meters',
    'local_tight_resolution',
    'local_tight_object_center',
    'local_tight_object_theta',
    'local_tight_bounds',

    # === Crop-independent supervision targets ===
    'se2_target_a1',         # world-frame Δ from pre_pose_a1
    'pre_pose_a1',            # (px, py, p_theta) crop center / Δ anchor
    'edge_idx_a1',            # primitive direction (0..59)
    'depth_idx_a1',           # primitive length step (0..9)
    'target_object_size',     # (sx, sy, sz) — loader picks .dat shape

    # === Metadata ===
    'xml_file',
    'solution_depth',
    'solutions_found',
    'solutions_total',
    'pushes_total',
]


def convert_npz_to_hdf5(input_dir: str, output_file: str, compression: str = "gzip",
                        compression_opts=4, minimal: bool = False,
                        tight_only: bool = False):
    """Convert directory of NPZ files to single HDF5 file.

    Args:
        input_dir: Directory containing NPZ files
        output_file: Output HDF5 file path
        compression: Compression algorithm — 'gzip', 'lzf', or None.
        compression_opts: Only used for 'gzip' (1-9, ignored for 'lzf').
        minimal: If True, only keep the MINIMAL_KEYS list.
        tight_only: If True (and minimal), drop local_wide_* keys.
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)

    # Find all NPZ files
    print(f"Scanning {input_path} for .npz files...")
    npz_files = sorted(glob.glob(f"{input_path}/**/*.npz", recursive=True))
    n_samples = len(npz_files)
    print(f"Found {n_samples} NPZ files")

    if n_samples == 0:
        raise ValueError(f"No NPZ files found in {input_dir}")

    # Find a representative file with all keys (important for multi-horizon data)
    print("Analyzing data structure...")
    if minimal:
        # In minimal mode, use MINIMAL_KEYS and find a file that has them
        sample_file = find_file_with_max_keys(npz_files, target_keys=MINIMAL_KEYS)
        print(f"Using sample file: {sample_file}")
        available_keys = get_npz_keys(sample_file)
        print(f"Keys in sample file: {available_keys}")
        # Use MINIMAL_KEYS but only if they exist in at least one file.
        # tight_only drops local_wide_* keys (dataloader reads only tight).
        keys = [k for k in MINIMAL_KEYS if k in available_keys]
        if tight_only:
            keys = [k for k in keys if not k.startswith("local_wide_")]
            print(f"Tight-only mode: dropped local_wide_* keys, kept {len(keys)}")
        print(f"Minimal mode: keeping only {keys}")
    else:
        sample_file = find_file_with_max_keys(npz_files)
        print(f"Using sample file: {sample_file}")
        keys = get_npz_keys(sample_file)
        print(f"Keys found in NPZ: {keys}")

    # Get shapes for each key, separate string vs numeric fields
    shapes = {}
    dtypes = {}
    string_keys = []
    numeric_keys = []

    with np.load(sample_file) as data:
        for key in keys:
            arr = data[key]
            shapes[key] = arr.shape
            dtypes[key] = arr.dtype

            if is_string_dtype(arr.dtype):
                string_keys.append(key)
                print(f"  {key}: shape={arr.shape}, dtype={arr.dtype} (string)")
            else:
                numeric_keys.append(key)
                print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

    # Create HDF5 file with datasets
    print(f"\nCreating HDF5 file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as h5f:
        datasets = {}

        # Create numeric datasets with chunking
        for key in numeric_keys:
            shape = (n_samples,) + shapes[key]
            chunks = (1,) + shapes[key]

            if compression == "lzf":
                ds = h5f.create_dataset(
                    key, shape=shape, dtype=dtypes[key],
                    chunks=chunks, compression="lzf"
                )
            elif compression:
                ds = h5f.create_dataset(
                    key, shape=shape, dtype=dtypes[key],
                    chunks=chunks, compression=compression, compression_opts=compression_opts
                )
            else:
                ds = h5f.create_dataset(
                    key, shape=shape, dtype=dtypes[key], chunks=chunks
                )
            datasets[key] = ds
            print(f"  Created dataset '{key}': {shape}")

        # Create string datasets with variable-length string type
        dt_str = h5py.special_dtype(vlen=str)
        for key in string_keys:
            shape = (n_samples,) + shapes[key]
            ds = h5f.create_dataset(key, shape=shape, dtype=dt_str)
            datasets[key] = ds
            print(f"  Created string dataset '{key}': {shape}")

        # Store file paths as metadata (useful for debugging)
        h5f.attrs['source_dir'] = str(input_path)
        h5f.attrs['n_samples'] = n_samples

        # Copy data from NPZ files
        print(f"\nConverting {n_samples} files...")
        for i, npz_file in enumerate(tqdm(npz_files, desc="Converting")):
            try:
                with np.load(npz_file) as data:
                    for key in numeric_keys:
                        if key in data:
                            datasets[key][i] = data[key]
                        else:
                            datasets[key][i] = np.zeros(shapes[key], dtype=dtypes[key])

                    for key in string_keys:
                        if key in data:
                            # Convert numpy string array to Python strings
                            arr = data[key]
                            if arr.ndim == 0:
                                datasets[key][i] = str(arr)
                            else:
                                datasets[key][i] = [str(x) for x in arr.flat]
                        else:
                            datasets[key][i] = [''] * int(np.prod(shapes[key]))

            except Exception as e:
                print(f"\nError processing {npz_file}: {e}")
                for key in numeric_keys:
                    datasets[key][i] = np.zeros(shapes[key], dtype=dtypes[key])
                for key in string_keys:
                    datasets[key][i] = [''] * int(np.prod(shapes[key]))

    # Report file size
    output_size = output_path.stat().st_size / (1024**3)
    print(f"\nDone! Output file: {output_path}")
    print(f"File size: {output_size:.2f} GB")
    print(f"Samples: {n_samples}")


def main():
    parser = argparse.ArgumentParser(description="Convert NPZ files to HDF5")
    parser.add_argument("input_dir", help="Directory containing NPZ files")
    parser.add_argument("output_file", help="Output HDF5 file path")
    parser.add_argument("--no-compression", action="store_true",
                        help="Disable compression (faster writes, larger file)")
    parser.add_argument("--compression", choices=("gzip", "lzf"), default="gzip",
                        help="Compression algorithm (overridden by --no-compression).")
    parser.add_argument("--compression-opts", type=int, default=4,
                        help="gzip level 1-9 (ignored for lzf). Default 4.")
    parser.add_argument("--tight-only", action="store_true",
                        help="In --minimal mode, drop local_wide_* keys.")
    parser.add_argument("--minimal", action="store_true",
                        help="Only keep local masks and xml_file (smaller file)")
    args = parser.parse_args()

    compression = None if args.no_compression else args.compression
    convert_npz_to_hdf5(args.input_dir, args.output_file, compression,
                        compression_opts=args.compression_opts,
                        minimal=args.minimal, tight_only=args.tight_only)


if __name__ == "__main__":
    main()
