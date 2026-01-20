import h5py
import numpy as np
import json
import argparse
from pathlib import Path
import math

def compute_stats(h5_path, output_path, use_local=True, mode="mean_std"):
    print(f"Opening {h5_path} (Mode: {mode})...")
    
    with h5py.File(h5_path, 'r') as f:
        # 1. Select the correct key based on mode
        key = 'target_goal_pose_deltas_obj' if use_local else 'target_goal_pose_deltas_world'
        
        if key not in f:
            raise ValueError(f"Key '{key}' not found in HDF5 file.")
            
        print(f"Reading {key}...")
        data = f[key][:] 
        
        # For n-push datasets, pose deltas may be stored as (N, k, 3) where k is the
        # number of remaining actions (or canonicalized to (N, 1, 3)). We train a
        # single-step model, so compute stats from the first delta only.
        if data.ndim == 3:
            data = data[:, 0, :]

        stats = {"mode": mode, "n_samples": len(data)}

        if mode == "max_abs":
            # Shared XY Norm to preserve aspect ratio
            max_xy = np.max(np.abs(data[:, 0:2]))
            xy_norm = float(max_xy)
            
            theta_norm = math.pi

            stats.update({"xy_norm": xy_norm, "theta_norm": theta_norm})

        elif mode == "mean_std":
            x_mean = float(np.mean(data[:, 0:1]))
            y_mean = float(np.mean(data[:, 1:2]))
            shared_std = float(np.std(data[:, 0:2])) + 1e-6

            theta_mean = 0
            theta_std = math.pi

            stats.update({
                "mean": [x_mean, y_mean, theta_mean],
                "std":  [shared_std, shared_std, theta_std]
            })

    # Save logic
    out = Path(output_path)
    if out.is_dir():
        out = out / f"stats_{mode}.json"
    
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print("="*40)
    print(f"Stats computed for {mode}:")
    for k, v in stats.items():
        if k != "n_samples":
            print(f"  {k}: {v}")
    print("="*40)
    print(f"Saved to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_file", type=str, required=True, help="Path to absolute H5 file")
    parser.add_argument("--output", type=str, default="config/stats/", help="Output directory or file")
    parser.add_argument("--mode", type=str, choices=["max_abs", "mean_std"], default="mean_std")
    parser.add_argument("--global_mode", action="store_true", help="Use global keys instead of local")
    args = parser.parse_args()
    
    compute_stats(args.h5_file, args.output, use_local=not args.global_mode, mode=args.mode)
