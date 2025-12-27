import os
import json
from tqdm import tqdm

def recreate_splits(data_dir):
    """
    Recreate training and validation splits based on the deterministic logic in the setup method.
    """
    unique_envs = set()
    datafiles = []
    env_to_files = {}

    # Check if data_dir contains subdirectories (new structure)
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    if subdirs:
        # New structure: data_dir/env_dirs/*.npz
        print(f"Found {len(subdirs)} environment directories")
        for env_dir in tqdm(subdirs, desc="Loading data from subdirectories"):
            env_path = os.path.join(data_dir, env_dir)
            try:
                env_files = [os.path.join(env_path, f) for f in os.listdir(env_path) if f.endswith('.npz')]
            except Exception:
                env_files = []

            if env_files:
                datafiles.extend(env_files)
                unique_envs.add(env_dir)
                env_to_files[env_dir] = env_files
    else:
        # Old structure: data_dir/*.npz with "final_state" pattern
        for f in tqdm(os.listdir(data_dir), desc="Loading data"):
            if "final_state" in f:
                datafiles.append(os.path.join(data_dir, f))
                unique_envs.add(int(f.split("env_config_")[-1].split("_")[0]))

    print(f"Total files: {len(datafiles)}, Unique environments: {len(unique_envs)}")

    train_datafiles = []
    val_datafiles = []

    if env_to_files:
        # New structure: split by environment directories
        sorted_envs = sorted(list(unique_envs))
        num_val_envs = max(min(100, len(sorted_envs) // 5), len(sorted_envs) // 20)

        print(f"Using {len(sorted_envs) - num_val_envs} environments for training, {num_val_envs} for validation")

        for env in tqdm(sorted_envs[:-num_val_envs], desc="Setting up train data"):
            train_datafiles.extend(env_to_files[env])

        for env in tqdm(sorted_envs[-num_val_envs:], desc="Setting up val data"):
            val_datafiles.extend(env_to_files[env])
    else:
        # Old structure: split by env_config number
        sorted_envs = sorted(list(unique_envs))

        for env in tqdm(sorted_envs[:-20], desc="Setting up train data"):
            train_datafiles.extend([f for f in datafiles if ("env_config_" + str(env) + "_") in f])

        for env in tqdm(sorted_envs[-20:], desc="Setting up val data"):
            val_datafiles.extend([f for f in datafiles if ("env_config_" + str(env) + "_") in f])

    return train_datafiles, val_datafiles

if __name__ == "__main__":
    data_dir = input("Enter the path to the data directory: ").strip()
    train_files, val_files = recreate_splits(data_dir)

    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")

    with open("train_files.json", "w") as f:
        json.dump(train_files, f, indent=4)

    with open("val_files.json", "w") as f:
        json.dump(val_files, f, indent=4)

    print("Training and validation splits saved to 'train_files.json' and 'val_files.json'")