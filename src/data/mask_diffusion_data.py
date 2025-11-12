from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Any
from torchvision import transforms
from tqdm import tqdm
import os
import torch
import lightning.pytorch as pl
import numpy as np
import random
import json

class MaskDiffusionDataset(Dataset):
    def __init__(self, datafiles: List[str], transform=None, split="train", use_coord_grid=False):
        """
        Args:
            data_dir: path to data directory
            transform: pytorch transforms for data augmentation
            split: one of 'train', 'val', or 'test'
        """
        self.transform = transform
        self.split = split
        self.use_coord_grid = use_coord_grid
        
        
        # split the datafiles into train, val, test 
        # if self.split == "train":
        #     sample_files = datafiles[:int(0.8*len(datafiles))]
        # elif self.split == "val":
        #     sample_files = datafiles[int(0.9*len(datafiles)):]
        # elif self.split == "test":
        #     sample_files = datafiles[int(0.9*len(datafiles)):]
            
        # print(self.split, len(sample_files))
        
        self.samples = datafiles
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Use context manager to ensure file is closed
        with np.load(sample) as data:
            # Support both old and new key naming conventions
            # Copy data to avoid issues after file is closed
            robot_image = (data.get('robot_image') if 'robot_image' in data else data['robot']).copy()
            goal_image = (data.get('goal_image') if 'goal_image' in data else data['goal']).copy()
            movable_objects_image = (data.get('movable_objects_image') if 'movable_objects_image' in data else data['movable']).copy()
            static_objects_image = (data.get('static_objects_image') if 'static_objects_image' in data else data['static']).copy()
            reachable_objects_image = (data.get('reachable_objects_image') if 'reachable_objects_image' in data else data['reachable']).copy()
            
            # For target_object and target_goal (new dataset)
            target_object = data.get('target_object', None)
            if target_object is not None:
                target_object = target_object.copy()
            target_goal = data.get('target_goal', None)
            if target_goal is not None:
                target_goal = target_goal.copy()
            
            # Legacy keys (old dataset)
            object_mask = data.get('object_mask', None)
            if object_mask is not None:
                object_mask = object_mask.copy()
            goal_mask = data.get('goal_mask', None)
            if goal_mask is not None:
                goal_mask = goal_mask.copy()
        
        image_size = robot_image.shape[0]
        
        if self.use_coord_grid:
            ys, xs = np.meshgrid(np.linspace(0, 1, image_size), 
                                 np.linspace(0, 1, image_size), 
                                 indexing='ij')
            coord_grid = np.stack([xs, ys], axis=-1)
            coord_grid = coord_grid.reshape(image_size, image_size, 2).astype(np.float32)
            
        
        if self.transform:
            ret = {
                "robot": self.transform(robot_image),
                "goal": self.transform(goal_image),
                "movable_objects": self.transform(movable_objects_image),
                "static_objects": self.transform(static_objects_image),
                "reachable_objects": self.transform(reachable_objects_image),
            }
            
            # Add target_object if available
            if target_object is not None:
                ret["target_object"] = self.transform(target_object)
                
            # Add target_goal if available
            if target_goal is not None:
                ret["target_goal"] = self.transform(target_goal)
                
            # Add legacy masks if available (for backward compatibility)
            if object_mask is not None:
                ret["object_mask"] = self.transform(object_mask)
            if goal_mask is not None:
                ret["goal_mask"] = self.transform(goal_mask)
                
            if self.use_coord_grid:
                ret["coord_grid"] = self.transform(coord_grid)
            return ret
        
        ret = {
            "robot": robot_image,
            "goal": goal_image,
            "movable_objects": movable_objects_image,
            "static_objects": static_objects_image,
            "reachable_objects": reachable_objects_image,
        }
        
        if target_object is not None:
            ret["target_object"] = target_object
        if target_goal is not None:
            ret["target_goal"] = target_goal
        if object_mask is not None:
            ret["object_mask"] = object_mask
        if goal_mask is not None:
            ret["goal_mask"] = goal_mask
        if self.use_coord_grid:
            ret["coord_grid"] = coord_grid
        return ret

class MaskDiffusionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        image_size: int = 64,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_coord_grid: bool = False,
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_size = image_size
        self.use_coord_grid = use_coord_grid
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self.datafiles = []
        
    def prepare_data(self):
        """
        preprocess data if needed
        """
        pass
        
        
    def setup(self, stage: Optional[str] = None):
        """
        Load data. Set variables: self.train_dataset, self.val_dataset, self.test_dataset
        """
        
        unique_envs = set()
        datafiles = []
        env_to_files = {}
        
        # Store cache locally to avoid NFS stalls
        if isinstance(self.data_dir, str):
            import hashlib
            # Create a unique cache filename based on the data directory path
            dir_hash = hashlib.md5(self.data_dir.encode()).hexdigest()[:16]
            local_cache_dir = os.path.expanduser(f"~/.cache/sage_learning")
            os.makedirs(local_cache_dir, exist_ok=True)
            cache_path = os.path.join(local_cache_dir, f"filelist_{dir_hash}.json")
        else:
            cache_path = None

        # Try to load a cached mapping of env -> filelist to avoid expensive NFS scans
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    raw = json.load(f)

                # raw is expected to be a dict env -> list of relative file paths
                for env, files in raw.items():
                    env_files = []
                    for p in files:
                        # if path is absolute, keep it, otherwise join with data_dir/env
                        if os.path.isabs(p):
                            env_files.append(p)
                        else:
                            env_files.append(os.path.join(self.data_dir, env, p))
                    if env_files:
                        env_to_files[env] = env_files
                        datafiles.extend(env_files)
                        unique_envs.add(env)

                if datafiles:
                    print(f"Loaded filelist cache from {cache_path}: {len(datafiles)} files across {len(unique_envs)} envs")
            except Exception as e:
                print(f"Failed to load filelist cache {cache_path}: {e}; falling back to directory scan")
        
        # Only scan directories if we didn't load from cache
        if not datafiles:
            if isinstance(self.data_dir, list):
                for folder in self.data_dir:
                    datafiles.extend([os.path.join(folder, f) for f in os.listdir(folder)])
            else:
                # Check if data_dir contains subdirectories (new oct26 structure)
                subdirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
                
                if subdirs:
                    # New structure: data_dir/env_dirs/*.npz
                    print(f"Found {len(subdirs)} environment directories")
                    for env_dir in tqdm(subdirs, desc="Loading data from subdirectories"):
                        # skip if we already loaded from cache
                        if env_dir in env_to_files:
                            continue
                        env_path = os.path.join(self.data_dir, env_dir)
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
                    for f in tqdm(os.listdir(self.data_dir), desc="Loading data"):
                        if "final_state" in f:
                            datafiles.append(os.path.join(self.data_dir, f))
                            unique_envs.add(int(f.split("env_config_")[-1].split("_")[0]))
        
        print(f"Total files: {len(datafiles)}, Unique environments: {len(unique_envs)}")

        # If we performed a scan and have a mapping, write a cache for future runs to avoid NFS stalls.
        # NOTE: In DDP training, create the cache BEFORE training using create_local_cache.py
        # to avoid race conditions. This code path should only run in single-GPU or non-DDP scenarios.
        try:
            if cache_path and env_to_files and not os.path.exists(cache_path):
                serializable = {}
                for env, files in env_to_files.items():
                    rels = []
                    for p in files:
                        # store relative path under the env directory when possible
                        try:
                            rel = os.path.relpath(p, os.path.join(self.data_dir, env))
                        except Exception:
                            rel = p
                        rels.append(rel)
                    serializable[env] = rels

                tmp_cache = cache_path + '.tmp'
                with open(tmp_cache, 'w') as f:
                    json.dump(serializable, f)
                os.replace(tmp_cache, cache_path)
                print(f"Wrote filelist cache to {cache_path}")
        except Exception as e:
            print(f"Failed to write filelist cache {cache_path}: {e}")
        
        train_datafiles = []
        val_datafiles = []
        test_datafiles = []
        
        if env_to_files:
            # New structure: split by environment directories
            sorted_envs = sorted(list(unique_envs))
            # ~5% for validation, minimum 100 envs, but not more than 20% of total
            num_val_envs = max(min(100, len(sorted_envs) // 5), len(sorted_envs) // 20)
            
            print(f"Using {len(sorted_envs) - num_val_envs} environments for training, {num_val_envs} for validation")
            
            for env in tqdm(sorted_envs[:-num_val_envs], desc="Setting up train data"):
                train_datafiles.extend(env_to_files[env])
                
            for env in tqdm(sorted_envs[-num_val_envs:], desc="Setting up val data"):
                val_datafiles.extend(env_to_files[env])
        else:
            # Old structure: split by env_config number
            sorted_envs = sorted(list(unique_envs))
            
            for env in tqdm(sorted_envs[:-20], desc="setting up train data"):
                train_datafiles.extend([f for f in datafiles if ("env_config_" + str(env) + "_") in f])
                
            for env in tqdm(sorted_envs[-20:], desc="setting up val data"):
                val_datafiles.extend([f for f in datafiles if ("env_config_" + str(env) + "_") in f])
            
        # train_datafiles = train_datafiles[:64]
        # val_datafiles = train_datafiles[:64]
        
        # valid_datafiles = datafiles
        # pbar = tqdm(range(len(datafiles)), desc="Preparing data")
        # for idx in pbar:
        #     file = datafiles[idx]
        #     npz_file = np.load(file)
        #     object_mask = npz_file['object_mask']
        #     if np.sum(object_mask) > 0:
        #         valid_datafiles.append(file)
        #     npz_file.close()
        
        # train_datafiles = train_datafiles
        # val_datafiles = val_datafiles
        print("train_datafiles:", len(train_datafiles), "val_datafiles:", len(val_datafiles))
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Lambda(lambda x: x * 2 - 1),
        ])
        
        if stage == "fit" or stage is None:
            self.train_dataset = MaskDiffusionDataset(
                train_datafiles, split="train", transform=transform, use_coord_grid=self.use_coord_grid
            )
            self.val_dataset = MaskDiffusionDataset(
                val_datafiles, split="val", transform=transform, use_coord_grid=self.use_coord_grid
            )
            
        if stage == "test" or stage is None:
            self.test_dataset = MaskDiffusionDataset(
                val_datafiles, split="test", transform=transform, use_coord_grid=self.use_coord_grid
            )
    
    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            persistent_workers=True if self.num_workers > 0 else False,  # Keep workers alive between epochs
        )
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,  # Don't shuffle validation data
            persistent_workers=True if self.num_workers > 0 else False,
        )
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,  # Don't shuffle test data
            persistent_workers=True if self.num_workers > 0 else False,
        )
        return loader