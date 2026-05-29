#!/usr/bin/env python3
"""
Filter an HDF5 dataset by per-scenario push-step count and copy the survivors.

The source H5 produced by ``convert_to_hdf5.py --minimal`` does not keep
``episode_id`` or ``task_id``, so this script supports reconstructing chain
membership from the original NPZ tree. It assumes the H5 row order matches the
sorted recursive NPZ order used by the converter.

For multi-step chains produced by mask generation, each original solution chain
is expanded into trajectory suffix rows:

  step_0 -> solution_depth = N
  step_1 -> solution_depth = N - 1
  ...
  step_(N-1) -> solution_depth = 1

This script groups rows by the base episode name with any ``_step_<k>`` suffix
removed, computes ``max(solution_depth)`` for each scenario, and drops whole
scenarios whose push-step count exceeds the configured threshold.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np


STEP_SUFFIX_RE = re.compile(r"_step_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy an H5 while filtering out scenarios with too many push steps."
    )
    parser.add_argument("input_h5", help="Source HDF5 file")
    parser.add_argument("output_h5", help="Filtered output HDF5 file")
    parser.add_argument(
        "--npz-root",
        help=(
            "Root directory of the original NPZ tree. Required for minimal H5s "
            "that do not contain an episode_id dataset."
        ),
    )
    parser.add_argument(
        "--max-push-steps",
        type=int,
        default=5,
        help=(
            "Keep only scenarios whose max solution_depth is <= this value. "
            "Since solution_depth counts remaining pushes, this is the maximum "
            "number of push steps allowed in any scenario. Default: 5."
        ),
    )
    parser.add_argument(
        "--max-chain-depth",
        type=int,
        dest="max_push_steps",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--summary-json",
        help="Optional path for a JSON summary. Defaults to <output_h5>.summary.json.",
    )
    parser.add_argument(
        "--copy-batch-size",
        type=int,
        default=256,
        help="Number of kept rows to copy per HDF5 read/write batch.",
    )
    return parser.parse_args()


def decode_scalar_string(value: object) -> str:
    """Convert an HDF5/NumPy string cell into a Python str."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        return decode_scalar_string(value.flat[0])
    return str(value)


def strip_step_suffix(name: str) -> str:
    return STEP_SUFFIX_RE.sub("", name)


def chain_key_from_npz_path(npz_root: Path, npz_path: str) -> str:
    path = Path(npz_path)
    rel = path.relative_to(npz_root)
    stem = strip_step_suffix(str(rel.with_suffix("")))
    return stem


def chain_key_from_episode_id(episode_id: str) -> str:
    return strip_step_suffix(episode_id)


def find_row_count(h5f: h5py.File) -> int:
    if "n_samples" in h5f.attrs:
        return int(h5f.attrs["n_samples"])

    for obj in h5f.values():
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 1:
            return int(obj.shape[0])

    raise ValueError("Could not determine row count from HDF5 file")


def load_solution_depths(h5f: h5py.File, n_rows: int) -> np.ndarray:
    if "solution_depth" not in h5f:
        raise ValueError("HDF5 file does not contain a solution_depth dataset")

    depths = np.asarray(h5f["solution_depth"][:]).reshape(-1)
    if len(depths) != n_rows:
        raise ValueError(
            f"solution_depth length mismatch: expected {n_rows}, found {len(depths)}"
        )
    return depths.astype(np.int32, copy=False)


def load_row_chain_ids_from_h5(h5f: h5py.File, n_rows: int) -> Optional[List[str]]:
    if "episode_id" not in h5f:
        return None

    raw_ids = h5f["episode_id"][:]
    row_chain_ids = [chain_key_from_episode_id(decode_scalar_string(v)) for v in raw_ids]
    if len(row_chain_ids) != n_rows:
        raise ValueError(
            f"episode_id length mismatch: expected {n_rows}, found {len(row_chain_ids)}"
        )
    return row_chain_ids


def load_row_chain_ids_from_npz(npz_root: Path, n_rows: int) -> List[str]:
    npz_pattern = str(npz_root / "**" / "*.npz")
    npz_files = sorted(glob.glob(npz_pattern, recursive=True))
    if len(npz_files) != n_rows:
        raise ValueError(
            "NPZ count mismatch with H5 rows: "
            f"expected {n_rows}, found {len(npz_files)} under {npz_root}"
        )

    return [chain_key_from_npz_path(npz_root, path) for path in npz_files]


def load_row_chain_ids(
    h5f: h5py.File, n_rows: int, npz_root: Optional[Path]
) -> Tuple[List[str], str]:
    row_chain_ids = load_row_chain_ids_from_h5(h5f, n_rows)
    if row_chain_ids is not None:
        return row_chain_ids, "h5:episode_id"

    if npz_root is None:
        raise ValueError(
            "This H5 does not contain episode_id. Please provide --npz-root so "
            "chain ids can be reconstructed from the original NPZ ordering."
        )

    return load_row_chain_ids_from_npz(npz_root, n_rows), "npz:sorted_recursive_glob"


def build_chain_stats(
    row_chain_ids: Sequence[str], solution_depths: np.ndarray, max_push_steps: int
) -> Tuple[Dict[str, int], np.ndarray, np.ndarray]:
    chain_depths: Dict[str, int] = {}
    for chain_id, depth in zip(row_chain_ids, solution_depths):
        depth_i = int(depth)
        chain_depths[chain_id] = max(depth_i, chain_depths.get(chain_id, depth_i))

    keep_mask = np.fromiter(
        (chain_depths[chain_id] <= max_push_steps for chain_id in row_chain_ids),
        dtype=bool,
        count=len(row_chain_ids),
    )
    keep_indices = np.flatnonzero(keep_mask)
    return chain_depths, keep_mask, keep_indices


def histogram_to_json(counter: Counter) -> Dict[str, int]:
    return {str(depth): int(counter[depth]) for depth in sorted(counter)}


def count_rows_for_chain_depth(
    row_chain_ids: Sequence[str], chain_depths: Dict[str, int], target_depth: int
) -> int:
    return int(sum(1 for chain_id in row_chain_ids if chain_depths[chain_id] == target_depth))


def summarize(
    input_h5: Path,
    output_h5: Path,
    chain_source: str,
    npz_root: Optional[Path],
    max_push_steps: int,
    row_chain_ids: Sequence[str],
    solution_depths: np.ndarray,
    chain_depths: Dict[str, int],
    keep_mask: np.ndarray,
) -> Dict[str, object]:
    chain_depth_hist = Counter(chain_depths.values())
    kept_chain_depth_hist = Counter(
        depth for depth in chain_depths.values() if depth <= max_push_steps
    )

    kept_chain_ids = {chain_id for chain_id, depth in chain_depths.items() if depth <= max_push_steps}
    kept_chain_count = len(kept_chain_ids)
    filtered_chain_count = len(chain_depths) - kept_chain_count

    kept_rows = int(keep_mask.sum())
    filtered_rows = len(keep_mask) - kept_rows

    two_push_chain_count = int(chain_depth_hist.get(2, 0))
    two_push_chain_count_kept = int(kept_chain_depth_hist.get(2, 0))
    two_push_row_count = count_rows_for_chain_depth(row_chain_ids, chain_depths, 2)
    two_push_row_count_kept = int(
        sum(
            1
            for row_idx, chain_id in enumerate(row_chain_ids)
            if keep_mask[row_idx] and chain_depths[chain_id] == 2
        )
    )

    return {
        "input_h5": str(input_h5),
        "output_h5": str(output_h5),
        "chain_source": chain_source,
        "npz_root": str(npz_root) if npz_root is not None else None,
        "max_push_steps": int(max_push_steps),
        "source_rows": int(len(row_chain_ids)),
        "kept_rows": kept_rows,
        "filtered_rows": int(filtered_rows),
        "source_chains": int(len(chain_depths)),
        "kept_chains": int(kept_chain_count),
        "filtered_chains": int(filtered_chain_count),
        "source_rows_with_solution_depth_2": int(np.count_nonzero(solution_depths == 2)),
        "kept_rows_with_solution_depth_2": int(np.count_nonzero(solution_depths[keep_mask] == 2)),
        "source_two_push_chains": two_push_chain_count,
        "kept_two_push_chains": two_push_chain_count_kept,
        "source_rows_from_two_push_chains": two_push_row_count,
        "kept_rows_from_two_push_chains": two_push_row_count_kept,
        "source_chain_depth_histogram": histogram_to_json(chain_depth_hist),
        "kept_chain_depth_histogram": histogram_to_json(kept_chain_depth_hist),
    }


def copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def dataset_create_kwargs(src_ds: h5py.Dataset, shape: Tuple[int, ...], row_filtered: bool) -> Dict[str, object]:
    kwargs: Dict[str, object] = {"shape": shape, "dtype": src_ds.dtype}

    if src_ds.chunks is not None:
        chunks = list(src_ds.chunks)
        if row_filtered and chunks:
            chunks[0] = max(1, min(chunks[0], max(1, shape[0])))
        kwargs["chunks"] = tuple(chunks)

    if src_ds.compression is not None:
        kwargs["compression"] = src_ds.compression
    if src_ds.compression_opts is not None:
        kwargs["compression_opts"] = src_ds.compression_opts
    if src_ds.shuffle:
        kwargs["shuffle"] = True
    if src_ds.fletcher32:
        kwargs["fletcher32"] = True
    if src_ds.scaleoffset is not None:
        kwargs["scaleoffset"] = src_ds.scaleoffset

    return kwargs


def copy_dataset(
    src_ds: h5py.Dataset,
    dst_group: h5py.Group,
    keep_indices: np.ndarray,
    n_rows: int,
    copy_batch_size: int,
) -> None:
    row_filtered = src_ds.ndim >= 1 and src_ds.shape[0] == n_rows
    shape = (len(keep_indices),) + src_ds.shape[1:] if row_filtered else src_ds.shape

    dst_ds = dst_group.create_dataset(
        src_ds.name.rsplit("/", 1)[-1],
        **dataset_create_kwargs(src_ds, shape, row_filtered),
    )
    copy_attrs(src_ds.attrs, dst_ds.attrs)

    if not row_filtered:
        dst_ds[...] = src_ds[...]
        return

    out_start = 0
    for batch_start in range(0, len(keep_indices), copy_batch_size):
        batch_indices = keep_indices[batch_start : batch_start + copy_batch_size]
        batch = src_ds[batch_indices]
        out_end = out_start + len(batch_indices)
        dst_ds[out_start:out_end] = batch
        out_start = out_end


def copy_group_filtered(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    keep_indices: np.ndarray,
    n_rows: int,
    copy_batch_size: int,
) -> None:
    copy_attrs(src_group.attrs, dst_group.attrs)

    for name, obj in src_group.items():
        if isinstance(obj, h5py.Group):
            child = dst_group.create_group(name)
            copy_group_filtered(obj, child, keep_indices, n_rows, copy_batch_size)
        elif isinstance(obj, h5py.Dataset):
            print(f"Copying dataset {obj.name} ...", flush=True)
            copy_dataset(obj, dst_group, keep_indices, n_rows, copy_batch_size)
        else:
            raise TypeError(f"Unsupported HDF5 object type for {obj.name}: {type(obj)}")


def main() -> None:
    args = parse_args()

    input_h5 = Path(args.input_h5).resolve()
    output_h5 = Path(args.output_h5).resolve()
    npz_root = Path(args.npz_root).resolve() if args.npz_root else None
    summary_json = (
        Path(args.summary_json).resolve()
        if args.summary_json
        else output_h5.with_suffix(output_h5.suffix + ".summary.json")
    )

    print(f"input_h5:        {input_h5}", flush=True)
    print(f"output_h5:       {output_h5}", flush=True)
    print(f"npz_root:        {npz_root if npz_root else '<not used>'}", flush=True)
    print(f"max_push_steps:  {args.max_push_steps}", flush=True)
    print(f"copy_batch_size: {args.copy_batch_size}", flush=True)

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_h5, "r") as src_h5:
        n_rows = find_row_count(src_h5)
        print(f"source_rows:     {n_rows}", flush=True)

        solution_depths = load_solution_depths(src_h5, n_rows)
        row_chain_ids, chain_source = load_row_chain_ids(src_h5, n_rows, npz_root)
        print(f"chain_source:    {chain_source}", flush=True)

        chain_depths, keep_mask, keep_indices = build_chain_stats(
            row_chain_ids=row_chain_ids,
            solution_depths=solution_depths,
            max_push_steps=args.max_push_steps,
        )
        summary = summarize(
            input_h5=input_h5,
            output_h5=output_h5,
            chain_source=chain_source,
            npz_root=npz_root,
            max_push_steps=args.max_push_steps,
            row_chain_ids=row_chain_ids,
            solution_depths=solution_depths,
            chain_depths=chain_depths,
            keep_mask=keep_mask,
        )

        print(
            "keeping "
            f"{summary['kept_rows']}/{summary['source_rows']} rows across "
            f"{summary['kept_chains']}/{summary['source_chains']} chains",
            flush=True,
        )
        print(
            "2-push chains "
            f"{summary['source_two_push_chains']} source / "
            f"{summary['kept_two_push_chains']} kept",
            flush=True,
        )

        with h5py.File(output_h5, "w") as dst_h5:
            copy_group_filtered(
                src_group=src_h5,
                dst_group=dst_h5,
                keep_indices=keep_indices,
                n_rows=n_rows,
                copy_batch_size=args.copy_batch_size,
            )

            dst_h5.attrs["source_h5"] = str(input_h5)
            dst_h5.attrs["chain_source"] = chain_source
            dst_h5.attrs["max_push_steps"] = int(args.max_push_steps)
            dst_h5.attrs["source_rows"] = int(summary["source_rows"])
            dst_h5.attrs["kept_rows"] = int(summary["kept_rows"])
            dst_h5.attrs["source_chains"] = int(summary["source_chains"])
            dst_h5.attrs["kept_chains"] = int(summary["kept_chains"])
            dst_h5.attrs["source_two_push_chains"] = int(summary["source_two_push_chains"])
            dst_h5.attrs["kept_two_push_chains"] = int(summary["kept_two_push_chains"])

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"summary_json:    {summary_json}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
