#!/usr/bin/env python3
"""
Create a local filelist cache to avoid NFS stalls during training.
This script scans the dataset directory and creates a cache file in ~/.cache/sage_learning/
"""

import os
import json
import hashlib
from collections import defaultdict
from tqdm import tqdm

def create_cache(data_dir):
    """Create a local cache of the file list for the given data directory."""
    
    print(f"Scanning data directory: {data_dir}")
    
    # Create local cache directory
    dir_hash = hashlib.md5(data_dir.encode()).hexdigest()[:16]
    local_cache_dir = os.path.expanduser("~/.cache/sage_learning")
    os.makedirs(local_cache_dir, exist_ok=True)
    cache_path = os.path.join(local_cache_dir, f"filelist_{dir_hash}.json")
    
    print(f"Cache will be saved to: {cache_path}")
    
    # Check if data directory exists
    if not os.path.isdir(data_dir):
        print(f"ERROR: Data directory does not exist: {data_dir}")
        return
    
    # First, check the structure
    try:
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    except Exception as e:
        print(f"ERROR: Cannot list data directory: {e}")
        return
    
    print(f"Found {len(subdirs)} environment directories")
    
    if not subdirs:
        print("ERROR: No subdirectories found in data directory")
        return
    
    # Sample a few directories
    print("\nSample environment directories:")
    for d in subdirs[:5]:
        print(f"  {d}")
    
    # Build the cache
    data = defaultdict(list)
    total_files = 0
    
    print("\nScanning environment directories...")
    for env_dir in tqdm(subdirs, desc="Processing environments"):
        env_path = os.path.join(data_dir, env_dir)
        try:
            files = [f for f in os.listdir(env_path) if f.endswith('.npz')]
            if files:
                # Store just the filenames (relative to env_dir)
                data[env_dir] = files
                total_files += len(files)
        except Exception as e:
            print(f"Warning: Error reading {env_dir}: {e}")
    
    print(f"\nTotal: {total_files} files across {len(data)} environments")
    
    # Write cache to temporary file first
    tmp_cache = cache_path + ".tmp"
    print(f"\nWriting cache to {tmp_cache}...")
    
    try:
        with open(tmp_cache, 'w') as f:
            json.dump(data, f)
        
        # Verify the cache file
        file_size = os.path.getsize(tmp_cache)
        print(f"Cache file size: {file_size} bytes")
        
        if file_size < 100:
            print("ERROR: Cache file is too small, something went wrong")
            os.remove(tmp_cache)
            return
        
        # Move to final location (this is a local operation, should be fast)
        os.replace(tmp_cache, cache_path)
        print(f"\nCache created successfully!")
        print(f"Location: {cache_path}")
        
        # Verify
        with open(cache_path, 'r') as f:
            loaded = json.load(f)
        print(f"Verified: {sum(len(v) for v in loaded.values())} files across {len(loaded)} environments")
        
    except Exception as e:
        print(f"ERROR: Failed to write cache: {e}")
        if os.path.exists(tmp_cache):
            os.remove(tmp_cache)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        # Default data directory
        data_dir = "/common/users/shared/robot_learning/dm1487/namo/datasets/images/oct26/train/1push"
    
    create_cache(data_dir)
