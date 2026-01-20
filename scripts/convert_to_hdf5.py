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


def is_string_dtype(dtype) -> bool:
    """Check if dtype is a string type."""
    return dtype.kind in ('U', 'S', 'O')  # Unicode, byte string, or object


CANONICAL_NUMERIC_SHAPES = {
    # For 2-push (and n-push) datasets, these can be (k, 3) where k is the number of
    # remaining actions. Sage learning trains a single-step model, so we canonicalize
    # to the first delta with a fixed shape (1, 3) to avoid HDF5 shape mismatches.
    "target_goal_pose_deltas_obj": (1, 3),
    "target_goal_pose_deltas_world": (1, 3),
    # Raw target pose/corners metadata can also be stored per-horizon.
    "target_goal_poses": (1, 3),
    "target_goal_corners_world": (1, 4, 2),
    "target_goal_corners_px": (1, 4, 2),
    # Action metadata is stored as remaining action sequences for trajectory-suffix
    # samples. Canonicalize to the first action to keep examples 1-step.
    "action_targets": (1, 3),
    # Corner deltas follow the same trajectory-suffix convention: (k, 4, 2).
    "target_goal_corner_deltas_obj": (1, 4, 2),
    "target_goal_corner_deltas_world": (1, 4, 2),
    "target_goal_corner_deltas_px": (1, 4, 2),
}

CANONICAL_STRING_SHAPES = {
    # Stored as (k,) for trajectory-suffix samples; keep only the first action.
    "action_object_ids": (1,),
}


def canonicalize_numeric_array(key: str, arr: np.ndarray, *, strict: bool = False) -> np.ndarray:
    """Canonicalize arrays with variable shapes to a fixed per-key shape."""
    if key not in CANONICAL_NUMERIC_SHAPES:
        return arr

    expected = CANONICAL_NUMERIC_SHAPES[key]
    arr = np.asarray(arr)

    if arr.size == 0:
        return np.zeros(expected, dtype=arr.dtype if arr.dtype != object else np.float32)

    expected_tail = expected[1:]
    tail_size = int(np.prod(expected_tail)) if expected_tail else 1

    # If the array is missing the leading "horizon" dimension (e.g., (3,) or (4,2)),
    # treat it as a single-step sample.
    if arr.shape == expected_tail:
        reshaped = arr.reshape((1,) + expected_tail)
    else:
        if arr.size % tail_size != 0:
            if strict:
                raise ValueError(
                    f"{key}: cannot reshape array of size {arr.size} into (-1, {expected_tail})"
                )
            # Can't safely interpret the array; return zeros but keep the sample.
            return np.zeros(expected, dtype=arr.dtype if arr.dtype != object else np.float32)
        reshaped = arr.reshape((-1,) + expected_tail)

    out = reshaped[: expected[0]]
    if out.shape == expected:
        return out

    padded = np.zeros(expected, dtype=reshaped.dtype)
    padded[: out.shape[0]] = out
    return padded


def canonicalize_string_array(key: str, arr: np.ndarray, *, strict: bool = False) -> np.ndarray:
    """Canonicalize string arrays with variable lengths to a fixed per-key shape."""
    if key not in CANONICAL_STRING_SHAPES:
        return arr

    expected = CANONICAL_STRING_SHAPES[key]
    arr = np.asarray(arr)

    if arr.size == 0:
        if strict:
            raise ValueError(f"{key}: cannot canonicalize empty string array")
        return np.full(expected, "", dtype="U")

    if arr.ndim == 0:
        flat = np.array([str(arr)], dtype="U")
    else:
        flat = np.array([str(x) for x in arr.reshape(-1)], dtype="U")

    out = flat[: expected[0]]
    if out.shape == expected:
        return out

    padded = np.full(expected, "", dtype=flat.dtype)
    padded[: out.shape[0]] = out
    return padded


MINIMAL_KEYS = [
    # Local masks
    'local_target_object',
    'local_target_goal',
    'local_static',
    'local_movable',
    'local_robot_region',
    'local_goal_sample_region',
    'target_goal_pose_deltas_obj',
    # Object pose metadata (needed for visualization)
    'local_object_theta',
    # Metadata needed for training
    'xml_file',
]


def convert_npz_to_hdf5(input_dir: str, output_file: str, compression: str = "gzip",
                        minimal: bool = False, strict: bool = False):
    """Convert directory of NPZ files to single HDF5 file.

    Args:
        input_dir: Directory containing NPZ files
        output_file: Output HDF5 file path
        compression: Compression algorithm ('gzip' or None)
        minimal: If True, only keep local masks and xml_file
        strict: If True, fail fast on missing keys or conversion errors.
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

    # Get keys and shapes from first file
    print("Analyzing data structure...")
    keys = get_npz_keys(npz_files[0])
    print(f"Keys found in NPZ: {keys}")

    # Filter keys if minimal mode
    if minimal:
        keys = [k for k in keys if k in MINIMAL_KEYS]
        print(f"Minimal mode: keeping only {keys}")

    # Get shapes for each key, separate string vs numeric fields
    shapes = {}
    dtypes = {}
    string_keys = []
    numeric_keys = []

    with np.load(npz_files[0]) as data:
        for key in keys:
            arr = data[key]
            if is_string_dtype(arr.dtype):
                arr = canonicalize_string_array(key, arr, strict=strict)
            else:
                arr = canonicalize_numeric_array(key, arr, strict=strict)
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

            if compression:
                ds = h5f.create_dataset(
                    key, shape=shape, dtype=dtypes[key],
                    chunks=chunks, compression=compression, compression_opts=4
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
        load_errors = 0
        missing_key_errors = 0
        write_errors = 0
        for i, npz_file in enumerate(tqdm(npz_files, desc="Converting")):
            try:
                data = np.load(npz_file)
            except Exception as e:
                print(f"\nError loading {npz_file}: {e}")
                load_errors += 1
                if strict:
                    raise
                for key in numeric_keys:
                    datasets[key][i] = np.zeros(shapes[key], dtype=dtypes[key])
                for key in string_keys:
                    datasets[key][i] = [''] * int(np.prod(shapes[key]))
                continue

            with data:
                for key in numeric_keys:
                    if key not in datasets:
                        continue

                    if key not in data:
                        missing_key_errors += 1
                        if strict:
                            raise KeyError(f"Missing key '{key}' in {npz_file}")
                        datasets[key][i] = np.zeros(shapes[key], dtype=dtypes[key])
                        continue

                    arr = canonicalize_numeric_array(key, data[key], strict=strict)
                    try:
                        datasets[key][i] = arr
                    except Exception as e:
                        print(f"\nError writing {key} for {npz_file}: {e}")
                        write_errors += 1
                        if strict:
                            raise
                        datasets[key][i] = np.zeros(shapes[key], dtype=dtypes[key])

                for key in string_keys:
                    if key not in datasets:
                        continue

                    if key not in data:
                        missing_key_errors += 1
                        if strict:
                            raise KeyError(f"Missing key '{key}' in {npz_file}")
                        datasets[key][i] = [''] * int(np.prod(shapes[key]))
                        continue

                    arr = canonicalize_string_array(key, data[key], strict=strict)
                    try:
                        if arr.ndim == 0:
                            datasets[key][i] = str(arr)
                        else:
                            datasets[key][i] = [str(x) for x in arr.flat]
                    except Exception as e:
                        print(f"\nError writing {key} for {npz_file}: {e}")
                        write_errors += 1
                        if strict:
                            raise
                        datasets[key][i] = [''] * int(np.prod(shapes[key]))

        if load_errors or missing_key_errors or write_errors:
            print("\nConversion warnings:")
            print(f"  load_errors={load_errors}")
            print(f"  missing_key_errors={missing_key_errors}")
            print(f"  write_errors={write_errors}")

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
    parser.add_argument("--minimal", action="store_true",
                        help="Only keep local masks and xml_file (smaller file)")
    parser.add_argument("--strict", action="store_true",
                        help="Fail fast if any file is missing keys or cannot be written cleanly")
    args = parser.parse_args()

    compression = None if args.no_compression else "gzip"
    convert_npz_to_hdf5(
        args.input_dir,
        args.output_file,
        compression,
        minimal=args.minimal,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
