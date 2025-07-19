from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Any
from torchvision import transforms
from tqdm import tqdm
import os
import torch
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
        
        data = np.load(sample)
        robot_image = data['robot_image']
        goal_image = data['goal_image']
        movable_objects_image = data['movable_objects_image']
        static_objects_image = data['static_objects_image']
        reachable_objects_image = data['reachable_objects_image']
        
        object_mask = data['object_mask']
        goal_mask = data['goal_mask']
        
        image_size = robot_image.shape[1]
        
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
                "object_mask": self.transform(object_mask),
                "goal_mask": self.transform(goal_mask),
            }
            if self.use_coord_grid:
                ret["coord_grid"] = self.transform(coord_grid)
            return ret
        
        ret = {
            "robot": robot_image,
            "goal": goal_image,
            "movable_objects": movable_objects_image,
            "static_objects": static_objects_image,
            "reachable_objects": reachable_objects_image,
            "object_mask": object_mask,
            "goal_mask": goal_mask,
        }
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
        if isinstance(self.data_dir, list):
            for folder in self.data_dir:
                datafiles.extend([os.path.join(folder, f) for f in os.listdir(folder)])
        else:
            for f in tqdm(os.listdir(self.data_dir), desc="Loading data"):
                if "final_state" in f:
                    datafiles.append(os.path.join(self.data_dir, f))
                    unique_envs.add(int(f.split("env_config_")[-1].split("_")[0]))
        
        print("Sorting envs")
        sorted_envs = sorted(list(unique_envs))
        
        train_datafiles = []
        val_datafiles = []
        test_datafiles = []
        
        # datafiles = datafiles[:100]
    
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
        )
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
        return loader