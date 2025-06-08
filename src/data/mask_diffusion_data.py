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
    def __init__(self, datafiles: List[str], transform=None, split="train"):
        """
        Args:
            data_dir: path to data directory
            transform: pytorch transforms for data augmentation
            split: one of 'train', 'val', or 'test'
        """
        self.transform = transform
        self.split = split
        
        # split the datafiles into train, val, test 
        if self.split == "train":
            sample_files = datafiles[:int(0.8*len(datafiles))]
        elif self.split == "val":
            sample_files = datafiles[int(0.8*len(datafiles)):int(0.9*len(datafiles))]
        elif self.split == "test":
            sample_files = datafiles[int(0.9*len(datafiles)):]
            
        print(self.split, len(sample_files))
        
        self.samples = sample_files
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        data = np.load(sample)
        scene = data['scene']
        
        object_mask = data['object_mask']
        goal_mask = data['goal_mask']
        reachable_objects_mask = data['reachable_objects_image'][:, :, :1]
        
        inp = np.concatenate([scene, reachable_objects_mask], axis=-1)
        tgt = np.concatenate([object_mask, goal_mask], axis=-1)
        
        if self.transform:
            inp = self.transform(inp)
            tgt = self.transform(tgt)
        
        return inp, tgt

class MaskDiffusionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        image_size: int = 64,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_size = image_size
        
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
        datafiles = []
        if isinstance(self.data_dir, list):
            for folder in self.data_dir:
                datafiles.extend([os.path.join(folder, f) for f in os.listdir(folder)])
        else:
            datafiles = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir)]
        
        valid_datafiles = []
        pbar = tqdm(range(len(datafiles)), desc="Preparing data")
        for idx in pbar:
            file = datafiles[idx]
            npz_file = np.load(file)
            object_mask = npz_file['object_mask']
            if np.sum(object_mask) > 0:
                valid_datafiles.append(file)
            npz_file.close()
            
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Lambda(lambda x: x * 2 - 1),
        ])
        
        if stage == "fit" or stage is None:
            self.train_dataset = MaskDiffusionDataset(
                valid_datafiles, split="train", transform=transform
            )
            self.val_dataset = MaskDiffusionDataset(
                valid_datafiles, split="val", transform=transform
            )
            
        if stage == "test" or stage is None:
            self.test_dataset = MaskDiffusionDataset(
                valid_datafiles, split="test", transform=transform
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
            shuffle=False,
        )
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        return loader