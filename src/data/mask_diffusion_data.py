from torch.utils.data import Dataset, DataLoader
from typing import Optional, List
from torchvision import transforms
from pathlib import Path
import lightning.pytorch as pl
import numpy as np
import random

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
        train_split: float = 0.9,
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_size = image_size
        self.use_coord_grid = use_coord_grid
        self.train_split = train_split
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self.datafiles = []
        
    def prepare_data(self):
        """
        preprocess data if needed
        """
        pass

    def _normalized_roots(self) -> List[Path]:
        if isinstance(self.data_dir, (list, tuple)):
            roots = self.data_dir
        else:
            roots = [self.data_dir]

        resolved_roots = []
        for root in roots:
            path = Path(root).expanduser()
            resolved_roots.append(path)
        return resolved_roots

    def _collect_npz_files(self) -> List[str]:
        datafiles = []
        for root in self._normalized_roots():
            if not root.exists():
                print(f"Warning: data path {root} does not exist, skipping")
                continue
            if root.is_file():
                if root.suffix == ".npz":
                    datafiles.append(str(root))
                continue
            for npz_file in root.rglob("*.npz"):
                datafiles.append(str(npz_file))

        # Remove duplicates while keeping deterministic order
        unique_files = sorted(set(datafiles))
        return unique_files
        
        
    def setup(self, stage: Optional[str] = None):
        """
        Load data. Set variables: self.train_dataset, self.val_dataset, self.test_dataset
        """
        all_datafiles = self._collect_npz_files()
        if not all_datafiles:
            raise RuntimeError(f"No .npz files found under {self.data_dir}")

        rng = random.Random(0)
        rng.shuffle(all_datafiles)

        if len(all_datafiles) == 1:
            train_datafiles = all_datafiles
            val_datafiles = all_datafiles
        else:
            split_idx = int(len(all_datafiles) * self.train_split)
            split_idx = max(1, min(split_idx, len(all_datafiles) - 1))
            train_datafiles = all_datafiles[:split_idx]
            val_datafiles = all_datafiles[split_idx:]
            
        train_datafiles = train_datafiles[:10]
        val_datafiles = val_datafiles[:10]

        print(
            f"Total .npz files: {len(all_datafiles)} | "
            f"train: {len(train_datafiles)} | test: {len(val_datafiles)}"
        )
        
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