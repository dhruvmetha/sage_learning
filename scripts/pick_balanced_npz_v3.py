#!/usr/bin/env python3
"""Pick a 1:1 balanced NPZ stage for the v3 combined (feb+aug9) H5.

Plan
----
* 1-push side: phase-1 NPZs (single-shot). Stratified by per-episode
  *solve_rate* = (#successful trials) / (#trials in primitive_trial_log).
  Buckets — hard: <5%, medium: 5-30%, easy: >=30%. Inverse-weight sampling
  upsamples hard+medium.
* 2-push side: every step_0/step_1 NPZ from phase 2-5 (chain training data).
  Already the scarce class; we keep all and let the count drive the 1-push
  side via the 1:1 ratio.

primitive_trial_log lives in the PKL (per-episode algorithm_stats), not the
NPZ. We scan PKLs once to build a {episode_id -> solve_rate} map, then walk
the NPZ corpora and assign each 1-push NPZ a bucket via its episode_id.

Output: a stage dir of symlinks named `<numeric_idx>_<orig_npz_name>` —
matches the v2 convention. Feed it to `convert_to_hdf5.py --minimal`.

Usage:
    python pick_balanced_npz_v3.py \\
        --pkl-roots /scratch/dm1487/outputs/v3_phase1 \\
                    /scratch/dm1487/outputs/v3_aug9_phase1 \\
        --npz-roots-1push  /scratch/dm1487/outputs/v3_phase1_masks \\
                           /scratch/dm1487/outputs/v3_aug9_phase1_masks \\
        --npz-roots-2push  /scratch/dm1487/outputs/v3_phase2_masks \\
                           /scratch/dm1487/outputs/v3_phase3_masks \\
                           /scratch/dm1487/outputs/v3_phase4_masks \\
                           /scratch/dm1487/outputs/v3_phase5_masks \\
                           /scratch/dm1487/outputs/v3_aug9_phase2_masks \\
                           /scratch/dm1487/outputs/v3_aug9_phase3_masks \\
                           /scratch/dm1487/outputs/v3_aug9_phase4_masks \\
                           /scratch/dm1487/outputs/v3_aug9_phase5A_masks \\
                           ... \\
        --output-dir /scratch/dm1487/h5/v3_balanced_1to1_npz \\
        --workers 32 \\
        --hard-weight 2.0 --med-weight 2.5 --easy-weight 0.5 \\
        --seed 42
"""
import argparse
import glob
import os
import pickle
import random
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np


def _scan_pkl(pkl_path):
    """Worker: return {canonical_xml_path: (tried, solved)} pairs from a PKL.

    Keyed by realpath(xml_file) so phase-1 PKLs match NPZs generated in later
    phases (each phase shard creates fresh symlinks → different episode_id
    hosts, but the symlink targets are the same canonical XML).
    """
    out = {}
    try:
        d = pickle.load(open(pkl_path, "rb"))
    except Exception:
        return out
    for ep in d.get("episode_results") or []:
        xml = ep.get("xml_file") or ""
        if not xml:
            continue
        try:
            key = os.path.realpath(xml)
        except Exception:
            continue
        log = (ep.get("algorithm_stats") or {}).get("primitive_trial_log") or []
        if not log:
            continue
        tried = len(log)
        solved = sum(1 for t in log if t.get("success"))
        # If multiple episodes (per-neighbor) share the same xml, keep the
        # max-attempts one (best signal); fold solved similarly.
        prev = out.get(key, (0, 0))
        out[key] = (max(prev[0], tried), max(prev[1], solved))
    return out


def build_solve_rate_map(pkl_roots, workers):
    """Walk all PKLs under pkl_roots (sharded layout aware) and return
    {episode_id: solve_rate} for episodes whose primitive_trial_log is non-empty."""
    pkls = []
    for root in pkl_roots:
        pkls += glob.glob(f"{root}/modular_data_*/*_results.pkl")
        pkls += glob.glob(f"{root}/shard_*/pkls/modular_data_*/*_results.pkl")
    pkls = sorted(set(pkls))
    print(f"  scanning {len(pkls)} PKLs across {len(pkl_roots)} roots with {workers} workers",
          file=sys.stderr)
    merged = {}
    with Pool(processes=workers) as pool:
        for chunk in pool.imap_unordered(_scan_pkl, pkls, chunksize=64):
            merged.update(chunk)
    return {eid: solved / max(tried, 1) for eid, (tried, solved) in merged.items()}


def _npz_xml_realpath(npz_path):
    """Worker: load NPZ, return (npz_path, canonical_xml_path) so we can match
    to the PKL rate_map (also keyed by realpath xml). Falls back to None on
    error/missing field."""
    try:
        with np.load(npz_path) as d:
            if "xml_file" not in d.files:
                return (npz_path, None)
            xml = str(d["xml_file"][0])
            return (npz_path, os.path.realpath(xml))
    except Exception:
        return (npz_path, None)


def collect_npzs(roots, suffix_filter):
    """Walk roots, return NPZs matching the filter. suffix_filter is a callable
    taking the basename and returning True/False."""
    out = []
    for root in roots:
        out += [f for f in glob.glob(f"{root}/**/*.npz", recursive=True)
                if suffix_filter(os.path.basename(f))]
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl-roots", nargs="+", required=True,
                    help="Phase-1 PKL output dirs (where primitive_trial_log lives)")
    ap.add_argument("--npz-roots-1push", nargs="+", required=True,
                    help="NPZ dirs for 1-push examples (*_goal.npz, no _step_)")
    ap.add_argument("--npz-roots-2push", nargs="+", required=True,
                    help="NPZ dirs for 2-push chain examples (*_goal_step_*.npz)")
    ap.add_argument("--output-dir", required=True, help="Stage dir of symlinks")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--hard-thresh", type=float, default=0.05,
                    help="solve_rate strictly below this -> Hard")
    ap.add_argument("--med-thresh", type=float, default=0.30,
                    help="solve_rate strictly below this -> Medium (above hard)")
    ap.add_argument("--hard-weight", type=float, default=2.0,
                    help="(legacy) used only when --target-* fractions not set")
    ap.add_argument("--med-weight", type=float, default=2.5,
                    help="(legacy)")
    ap.add_argument("--easy-weight", type=float, default=0.5,
                    help="(legacy)")
    # Explicit-quota mode (preferred): exact target fractions of 1-push pool.
    ap.add_argument("--target-hard", type=float, default=0.55,
                    help="Fraction of 1-push pool that should be hard (default 0.55)")
    ap.add_argument("--target-med", type=float, default=0.30,
                    help="Fraction of 1-push pool that should be medium (default 0.30)")
    ap.add_argument("--target-easy", type=float, default=0.15,
                    help="Fraction of 1-push pool that should be easy (default 0.15)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.output_dir)
    if out.exists():
        print(f"ERROR: --output-dir {out} exists; remove first.", file=sys.stderr)
        return 1
    out.mkdir(parents=True)

    # 1. Build solve_rate map from phase-1 PKLs.
    print("[1/4] Building solve_rate map from PKLs", file=sys.stderr)
    rate_map = build_solve_rate_map(args.pkl_roots, args.workers)
    print(f"      {len(rate_map)} episodes with non-empty primitive_trial_log", file=sys.stderr)

    # 2. Collect NPZs by class.
    print("[2/4] Collecting NPZs", file=sys.stderr)
    one_push = collect_npzs(
        args.npz_roots_1push,
        lambda b: b.endswith("_goal.npz"),
    )
    two_push = collect_npzs(
        args.npz_roots_2push,
        lambda b: "_goal_step_" in b,
    )
    print(f"      1-push NPZs found: {len(one_push)}", file=sys.stderr)
    print(f"      2-push NPZs found: {len(two_push)}", file=sys.stderr)

    # 3. Bucket 1-push NPZs by solve_rate via xml_file matching (parallel).
    #    Store (path, rate) so we can sort easy bucket by rate later (for the
    #    "hardest easies" fallback rule).
    print(f"[3/4] Bucketing 1-push NPZs (xml_file matching, {args.workers} workers)",
          file=sys.stderr)
    buckets = {"hard": [], "med": [], "easy": [], "unknown": []}
    with Pool(processes=args.workers) as pool:
        for npz_path, xml_real in pool.imap_unordered(_npz_xml_realpath, one_push, chunksize=128):
            rate = rate_map.get(xml_real) if xml_real else None
            if rate is None:
                buckets["unknown"].append((npz_path, float("nan")))
            elif rate < args.hard_thresh:
                buckets["hard"].append((npz_path, rate))
            elif rate < args.med_thresh:
                buckets["med"].append((npz_path, rate))
            else:
                buckets["easy"].append((npz_path, rate))
    for k, v in buckets.items():
        print(f"      {k}: {len(v)}", file=sys.stderr)

    # Pre-sort the easy bucket by ascending solve_rate so the "hardest easies"
    # (lowest rate, just above the med threshold) sit at the front. This is
    # what we tap into when medium fallback also runs short.
    buckets["easy"].sort(key=lambda t: t[1])

    # 4. Sample 1-push to match |two_push| with EXPLICIT quotas (55/30/15)
    #    and STRICT PRIORITY FALLBACK:
    #      - Hard quota fills from: hard -> medium -> hardest-easies.
    #      - Medium quota fills from: medium (residual) -> hardest-easies.
    #      - Easy quota fills from: easy (residual).
    target_1push = len(two_push)
    if target_1push == 0:
        print("ERROR: no 2-push NPZs found; can't balance.", file=sys.stderr)
        return 1

    tot = args.target_hard + args.target_med + args.target_easy
    th, tm, te = args.target_hard / tot, args.target_med / tot, args.target_easy / tot
    q_hard = int(round(target_1push * th))
    q_med  = int(round(target_1push * tm))
    q_easy = target_1push - q_hard - q_med

    print(f"[4/4] Sampling — quotas (target {target_1push}):", file=sys.stderr)
    print(f"      hard desired={q_hard} avail={len(buckets['hard'])}", file=sys.stderr)
    print(f"      med  desired={q_med}  avail={len(buckets['med'])}", file=sys.stderr)
    print(f"      easy desired={q_easy} avail={len(buckets['easy'])}", file=sys.stderr)

    # We CONSUME buckets as we sample. Easy is consumed in sorted order
    # (hardest first) when it's a fallback source; in random order when it's
    # filling its own quota from whatever remains.
    hard_pool = list(buckets["hard"])
    med_pool  = list(buckets["med"])
    easy_pool = list(buckets["easy"])   # sorted asc by rate

    rng.shuffle(hard_pool)
    rng.shuffle(med_pool)
    # easy_pool stays sorted — we'll slice "hardest" from the head when needed

    picked_1push = []
    stats = {"hard_from_hard": 0, "hard_from_med": 0, "hard_from_easy": 0,
             "med_from_med": 0,  "med_from_easy": 0,
             "easy_from_easy": 0,
             "unfulfilled_hard": 0, "unfulfilled_med": 0, "unfulfilled_easy": 0}

    # --- HARD quota ---
    need = q_hard
    take = min(need, len(hard_pool))
    picked_1push.extend(p for p, _ in hard_pool[:take])
    hard_pool = hard_pool[take:]
    need -= take
    stats["hard_from_hard"] = take
    if need > 0:
        take = min(need, len(med_pool))
        picked_1push.extend(p for p, _ in med_pool[:take])
        med_pool = med_pool[take:]
        need -= take
        stats["hard_from_med"] = take
    if need > 0:
        # Take from the HEAD of easy (hardest easies first)
        take = min(need, len(easy_pool))
        picked_1push.extend(p for p, _ in easy_pool[:take])
        easy_pool = easy_pool[take:]
        need -= take
        stats["hard_from_easy"] = take
    stats["unfulfilled_hard"] = need

    # --- MEDIUM quota (from remaining med pool, then hardest easies) ---
    need = q_med
    take = min(need, len(med_pool))
    picked_1push.extend(p for p, _ in med_pool[:take])
    med_pool = med_pool[take:]
    need -= take
    stats["med_from_med"] = take
    if need > 0:
        take = min(need, len(easy_pool))
        picked_1push.extend(p for p, _ in easy_pool[:take])
        easy_pool = easy_pool[take:]
        need -= take
        stats["med_from_easy"] = take
    stats["unfulfilled_med"] = need

    # --- EASY quota (random sample of what's left in easy_pool) ---
    rng.shuffle(easy_pool)
    need = q_easy
    take = min(need, len(easy_pool))
    picked_1push.extend(p for p, _ in easy_pool[:take])
    need -= take
    stats["easy_from_easy"] = take
    stats["unfulfilled_easy"] = need

    print("      sampling breakdown:", file=sys.stderr)
    for k, v in stats.items():
        if v: print(f"        {k}: {v}", file=sys.stderr)
    total_unfulfilled = stats["unfulfilled_hard"] + stats["unfulfilled_med"] + stats["unfulfilled_easy"]
    if total_unfulfilled:
        print(f"      WARN: {total_unfulfilled} total 1-push slots unfilled (1-push pool exhausted)",
              file=sys.stderr)
    rng.shuffle(picked_1push)
    rng.shuffle(two_push)
    all_picked = picked_1push + two_push
    rng.shuffle(all_picked)
    print(f"      total picked: {len(all_picked)} (1-push={len(picked_1push)}, 2-push={len(two_push)})",
          file=sys.stderr)

    width = max(8, len(str(len(all_picked))))
    for i, src in enumerate(all_picked):
        link_name = f"{i:0{width}d}_{os.path.basename(src)}"
        os.symlink(src, out / link_name)
        if (i + 1) % 50000 == 0:
            print(f"      symlinked {i+1}/{len(all_picked)}", file=sys.stderr)

    print(f"\nDONE. Stage: {out}  total={len(all_picked)}", file=sys.stderr)
    print(f"  1-push: {len(picked_1push)}  (target hard/med/easy quotas: "
          f"{q_hard}/{q_med}/{q_easy})", file=sys.stderr)
    print(f"  2-push: {len(two_push)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
