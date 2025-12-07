import h5py
import numpy as np
import json
import argparse
from pathlib import Path

def compute_stats(h5_path, output_path, use_local=True):
    print(f"Opening {h5_path}...")
    
    with h5py.File(h5_path, 'r') as f:
        # 1. Select the correct key based on mode
        key = 'target_goal_pose_deltas_obj' if use_local else 'target_goal_pose_deltas_world'
        
        if key not in f:
            raise ValueError(f"Key '{key}' not found in HDF5 file.")
            
        print(f"Reading {key}...")
        # Load data (Shape: N x 1 x 3 or N x 3)
        data = f[key][:] 
        
        # Squeeze if necessary (N, 1, 3) -> (N, 3)
        if data.ndim == 3:
            data = data.squeeze(1)
            
        # 2. Compute XY Norm
        # We want the maximum absolute value to ensure everything fits in [-1, 1]
        # We add a small 5% buffer to be safe against outliers in test sets
        max_x = np.max(np.abs(data[:, 0]))
        max_y = np.max(np.abs(data[:, 1]))
        xy_norm = float(max(max_x, max_y)) * 1.05
        
        # 3. Compute Theta Norm
        # Usually Pi, but let's check the data
        max_theta = np.max(np.abs(data[:, 2]))
        theta_norm = float(max_theta)
        
        # If theta is roughly Pi (within 10%), just use Pi to keep it standard
        if abs(theta_norm - np.pi) < 0.3:
            theta_norm = float(np.pi)
        else:
            theta_norm = theta_norm * 1.05 # Add buffer if it's weird data

    stats = {
        "xy_norm": xy_norm,
        "dtheta_norm": theta_norm,
        "n_samples": len(data)
    }
    
    print("="*40)
    print(f"Stats computed:")
    print(f"XY Norm (Max abs + 5%): {xy_norm:.4f} m")
    print(f"Theta Norm:             {theta_norm:.4f} rad")
    print("="*40)
    
    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Saved to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_file", type=str, required=True, help="Path to absolute H5 file")
    parser.add_argument("--output", type=str, default="config/stats/dataset_stats.json")
    parser.add_argument("--global_mode", action="store_true", help="Use global keys instead of local")
    args = parser.parse_args()
    
    compute_stats(args.h5_file, args.output, use_local=not args.global_mode)