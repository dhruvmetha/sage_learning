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


MINIMAL_KEYS = [
    # Local masks
    'local_target_object',
    'local_target_goal',
    'local_static',
    'local_movable',
    'local_robot_region',
    'local_goal_sample_region',
    # Metadata needed for training
    'xml_file',
]


def convert_npz_to_hdf5(input_dir: str, output_file: str, compression: str = "gzip",
                        minimal: bool = False):
    """Convert directory of NPZ files to single HDF5 file.

    Args:
        input_dir: Directory containing NPZ files
        output_file: Output HDF5 file path
        compression: Compression algorithm ('gzip' or None)
        minimal: If True, only keep local masks and xml_file
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
    parser.add_argument("--minimal", action="store_true",
                        help="Only keep local masks and xml_file (smaller file)")
    args = parser.parse_args()

    compression = None if args.no_compression else "gzip"
    convert_npz_to_hdf5(args.input_dir, args.output_file, compression, minimal=args.minimal)


if __name__ == "__main__":
    main()
