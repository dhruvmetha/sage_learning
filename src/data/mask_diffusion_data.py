from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Union, Dict
from torchvision import transforms
from pathlib import Path
import lightning.pytorch as pl
import numpy as np
import random
import glob
import torch
import os

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

class MaskDiffusionDataset(Dataset):
    def __init__(self, datafiles: List[str], transform=None, use_coord_grid=False, use_local=True):
        self.transform = transform
        self.use_coord_grid = use_coord_grid
        self.use_local = use_local
        self.samples = datafiles

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ret = {}
        
        # We assume files are valid and contain the required keys
        with np.load(sample) as data:
            if self.use_local:
                # -----------------------------------------------------------
                # LOCAL MODE (Assumes keys exist)
                # Map 'local_X' -> 'X' for the model
                # -----------------------------------------------------------
                ret['static'] = data['local_static']
                ret['movable'] = data['local_movable']
                ret['target_object'] = data['local_target_object']
                ret['robot_region'] = data['local_robot_region']
                ret['goal_sample_region'] = data['local_goal_sample_region']
                
                # Target Vector: (dx, dy, dtheta) in OBJECT frame
                # Shape can be (1, 3) for 1-push samples, or (k, 3) for n-push
                # trajectory-suffix samples. We always train a single-step model,
                # so take only the first delta.
                deltas = data['target_goal_pose_deltas_obj']
                ret['target_goal'] = deltas[0] if getattr(deltas, "ndim", 0) > 1 else deltas
                # local_target_goal is already (224, 224), no indexing needed
                ret['target_goal_mask'] = data['local_target_goal']
                
                # Object theta (orientation) - needed for visualization
                # This is the current object's theta in world frame
                if 'local_object_theta' in data:
                    theta_arr = data['local_object_theta']
                    ret['object_theta'] = theta_arr[0] if getattr(theta_arr, "ndim", 0) > 0 else theta_arr
                else:
                    ret['object_theta'] = np.float32(0.0)  # Fallback

            else:
                # -----------------------------------------------------------
                # GLOBAL MODE (Legacy support if needed)
                # -----------------------------------------------------------
                ret['robot'] = data['robot_image'] if 'robot_image' in data else data['robot']
                ret['goal'] = data['goal_image'] if 'goal_image' in data else data['goal']
                ret['movable'] = data['movable_objects_image'] if 'movable_objects_image' in data else data['movable']
                ret['static'] = data['static_objects_image'] if 'static_objects_image' in data else data['static']
                ret['target_object'] = data['target_object']
                deltas = data['target_goal_pose_deltas_world']
                ret['target_goal'] = deltas[0] if getattr(deltas, "ndim", 0) > 1 else deltas
                # target_goal mask is already (224, 224), no indexing needed
                ret['target_goal_mask'] = data['target_goal']

            # -----------------------------------------------------------
            # Coordinate Grid (Optional)
            # -----------------------------------------------------------
            if self.use_coord_grid:
                # Infer size from the first image loaded
                image_size = ret['static'].shape[0] if self.use_local else ret['robot'].shape[0]
                
                ys, xs = np.meshgrid(np.linspace(0, 1, image_size),
                                     np.linspace(0, 1, image_size),
                                     indexing='ij')
                coord_grid = np.stack([xs, ys], axis=-1).astype(np.float32)
                ret["coord_grid"] = coord_grid

            # -----------------------------------------------------------
            # Transforms
            # -----------------------------------------------------------
            if self.transform:
                for k, v in ret.items():
                    if k == "target_goal":
                        # Vectors: Just tensorify, don't image-transform
                        ret[k] = torch.tensor(v, dtype=torch.float32)
                    elif k == "object_theta":
                        # Scalar: Just tensorify
                        ret[k] = torch.tensor(v, dtype=torch.float32)
                    else:
                        # Images: Apply transform (Resize, ToTensor, Normalize)
                        ret[k] = self.transform(v)
            else:
                # Minimal conversion if no transform provided
                if not isinstance(ret['target_goal'], torch.Tensor):
                     ret['target_goal'] = torch.tensor(ret['target_goal'], dtype=torch.float32)
                if 'object_theta' in ret and not isinstance(ret['object_theta'], torch.Tensor):
                     ret['object_theta'] = torch.tensor(ret['object_theta'], dtype=torch.float32)

            return ret


class MaskDiffusionHDF5Dataset(Dataset):
    """HDF5-backed dataset."""
    def __init__(self, h5_path: str, indices: List[int], transform=None, use_coord_grid=False, use_local=True):
        self.h5_path = h5_path
        self.indices = indices
        self.transform = transform
        self.use_coord_grid = use_coord_grid
        self.use_local = use_local
        self._h5_file = None
        self._pid = None

    def _get_h5_file(self):

        current_pid = os.getpid()

        if self._h5_file is None or self._pid != current_pid:
            if self._h5_file is not None:
                self._h5_file.close()

            # Re-open file
            self._h5_file = h5py.File(self.h5_path, 'r')
            self._pid = current_pid

        return self._h5_file

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        h5f = self._get_h5_file()
        real_idx = self.indices[idx]
        ret = {}

        if self.use_local:
            # Direct Key Access (Assumes presence)
            ret['static'] = h5f['local_static'][real_idx]
            ret['movable'] = h5f['local_movable'][real_idx]
            ret['target_object'] = h5f['local_target_object'][real_idx]
            ret['robot_region'] = h5f['local_robot_region'][real_idx]
            ret['goal_sample_region'] = h5f['local_goal_sample_region'][real_idx]
            # Supports both canonical (N, 1, 3) storage and legacy (N, k, 3).
            deltas = h5f['target_goal_pose_deltas_obj'][real_idx]
            ret['target_goal'] = deltas[0] if getattr(deltas, "ndim", 0) > 1 else deltas
            # local_target_goal is already (224, 224) per sample, no extra indexing needed
            ret['target_goal_mask'] = h5f['local_target_goal'][real_idx]
            
            # Object theta (orientation) - needed for visualization
            if 'local_object_theta' in h5f:
                theta_arr = h5f['local_object_theta'][real_idx]
                ret['object_theta'] = theta_arr[0] if getattr(theta_arr, "ndim", 0) > 0 else theta_arr
            else:
                ret['object_theta'] = np.float32(0.0)  # Fallback
        else:
            # Global Key Access
            ret['robot'] = h5f['robot_image'][real_idx]
            ret['goal'] = h5f['goal_image'][real_idx]
            ret['movable'] = h5f['movable_objects_image'][real_idx]
            ret['static'] = h5f['static_objects_image'][real_idx]
            ret['target_object'] = h5f['target_object'][real_idx]
            deltas = h5f['target_goal_pose_deltas_world'][real_idx]
            ret['target_goal'] = deltas[0] if getattr(deltas, "ndim", 0) > 1 else deltas
            # target_goal mask is already (224, 224) per sample, no extra indexing needed
            ret['target_goal_mask'] = h5f['target_goal'][real_idx]

        # Coordinate Grid
        if self.use_coord_grid:
            # Infer size from static channel
            img_shape = ret['static'].shape if self.use_local else ret['static'].shape
            ys, xs = np.meshgrid(np.linspace(0, 1, img_shape[0]), 
                                 np.linspace(0, 1, img_shape[0]), 
                                 indexing='ij')
            ret["coord_grid"] = np.stack([xs, ys], axis=-1).astype(np.float32)

        # Transforms
        if self.transform:
            for k, v in ret.items():
                if k == "target_goal":
                    ret[k] = torch.tensor(v, dtype=torch.float32)
                elif k == "object_theta":
                    # Scalar: Just tensorify
                    ret[k] = torch.tensor(v, dtype=torch.float32)
                else:
                    ret[k] = self.transform(v)
        else:
             if not isinstance(ret['target_goal'], torch.Tensor):
                 ret['target_goal'] = torch.tensor(ret['target_goal'], dtype=torch.float32)
             if 'object_theta' in ret and not isinstance(ret['object_theta'], torch.Tensor):
                 ret['object_theta'] = torch.tensor(ret['object_theta'], dtype=torch.float32)
        
        return ret
    
    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()


class MaskDiffusionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Optional[str] = None,
        h5_file: Optional[str] = None,
        image_size: int = 224, # Updated default to 224
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_coord_grid: bool = False,
        use_local: bool = True, # Default to True per instructions
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters() # Handled by Lightning
        
        self.data_dir = data_dir
        self.h5_file = h5_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_size = image_size
        self.use_coord_grid = use_coord_grid
        self.use_local = use_local
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        # Backward compatibility: if splits sum > 1.0, cap val_split and zero test_split
        # This handles old configs that only specified train_split
        total = train_split + val_split + test_split
        if total > 1.0:
            # Assume old 2-way split: val_split = 1 - train_split, test_split = 0
            self.val_split = 1.0 - train_split
            self.test_split = 0.0
            total = 1.0

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def _normalized_roots(self) -> List[Path]:
        if self.data_dir is None:
            return []
        if isinstance(self.data_dir, (list, tuple)):
            roots = self.data_dir
        else:
            roots = [self.data_dir]
        return [Path(r).expanduser() for r in roots]

    def _collect_files(self) -> List[str]:
        # Simple glob collection
        roots = self._normalized_roots()
        datafiles = []
        for root in roots:
            if root.is_file() and str(root).endswith('.npz'):
                datafiles.append(str(root))
            else:
                datafiles.extend(glob.glob(f"{root}/**/*.npz", recursive=True))
        return sorted(list(set(datafiles)))

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return
        if self.h5_file is None and self.data_dir is None:
             raise ValueError("Both 'h5_file' and 'data_dir' are None. Please provide at least one.")

        # 1. Define Transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size)),
            # Normalize to [-1, 1] for Diffusion/Flow Matching
            transforms.Lambda(lambda x: x * 2 - 1),
        ])

        # 2. Check for HDF5, fallback if not provided.
        h5_path = None
        if self.h5_file is not None:
            p = Path(self.h5_file)
            if p.exists():
                h5_path = str(p)
            else:
                raise FileNotFoundError(f"Provided h5_file not found: {self.h5_file}")
        elif self.data_dir is not None:
            roots = self._normalized_roots()
            h5_path = None
            for root in roots:
                possible = root / "data.h5"
                if possible.exists():
                    h5_path = str(possible)
                    break
        
        # 3. Instantiate Datasets
        if h5_path and HAS_H5PY:
            print(f"Using HDF5: {h5_path}")
            with h5py.File(h5_path, 'r') as h5f:
                # Assumes local_static is representative of dataset length
                n_samples = len(h5f['local_static']) if self.use_local else len(h5f['robot_image'])
            
            indices = list(range(n_samples))
            random.Random(42).shuffle(indices)
            
            # 3-way split: train / val / test
            train_end = int(n_samples * self.train_split)
            val_end = int(n_samples * (self.train_split + self.val_split))
            
            train_idx = indices[:train_end]
            val_idx = indices[train_end:val_end]
            test_idx = indices[val_end:]
            
            print(f"Dataset split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
            
            self.train_dataset = MaskDiffusionHDF5Dataset(h5_path, train_idx, transform, self.use_coord_grid, self.use_local)
            self.val_dataset = MaskDiffusionHDF5Dataset(h5_path, val_idx, transform, self.use_coord_grid, self.use_local)
            self.test_dataset = MaskDiffusionHDF5Dataset(h5_path, test_idx, transform, self.use_coord_grid, self.use_local)

        else:
            print("Using NPZ files")
            files = self._collect_files()
            random.Random(42).shuffle(files)
            
            # 3-way split: train / val / test
            n_files = len(files)
            train_end = int(n_files * self.train_split)
            val_end = int(n_files * (self.train_split + self.val_split))
            
            train_files = files[:train_end]
            val_files = files[train_end:val_end]
            test_files = files[val_end:]
            
            print(f"Dataset split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
            
            self.train_dataset = MaskDiffusionDataset(train_files, transform, self.use_coord_grid, self.use_local)
            self.val_dataset = MaskDiffusionDataset(val_files, transform, self.use_coord_grid, self.use_local)
            self.test_dataset = MaskDiffusionDataset(test_files, transform, self.use_coord_grid, self.use_local)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=(self.num_workers > 0))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=(self.num_workers > 0))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=self.pin_memory)
