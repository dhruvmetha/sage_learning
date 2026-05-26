"""
Dataset and DataModule for F-characterization classifier training.

Loads NPZ files produced by batch_collection_classifier.py.
Each sample contains scene masks (same as SAGE) + F grid (60x10 primitive labels).
"""

from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Union
from torchvision import transforms
from pathlib import Path
import lightning.pytorch as pl
import numpy as np
import torch
import random
import glob


class ClassifierDataset(Dataset):
    """Dataset for primitive feasibility classification.

    Each sample returns:
        - context: scene masks (5 channels, image_size x image_size)
        - f_grid: (60, 10) float32, 1=success, 0=fail, nan=unreachable
        - r_mask: (60, 10) float32, 1=reachable, 0=unreachable
        - metadata: dict with F, R, ratio, xml_file, object_id, region
    """

    def __init__(self, datafiles: List[str], transform=None,
                 use_local: bool = True, use_coord_grid: bool = False):
        self.samples = datafiles
        self.transform = transform
        self.use_local = use_local
        self.use_coord_grid = use_coord_grid

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        with np.load(sample) as data:
            # Load F grid and reachability mask (always present)
            f_grid = data['f_grid'].copy()        # (60, 10) with nan
            r_mask = data['r_mask'].copy()         # (60, 10) binary, per-(edge, depth)

            # Replace nan with 0 in f_grid for loss masking
            f_labels = np.nan_to_num(f_grid, nan=0.0).astype(np.float32)

            # Contact-point-level reachability: if ANY depth is reachable,
            # the contact point is reachable (robot can reach the object there).
            # This is what's known at inference time from wavefront BFS.
            cp_reachable = np.zeros((60, 10), dtype=np.float32)
            for ei in range(60):
                if r_mask[ei, :].sum() > 0:
                    cp_reachable[ei, :] = 1.0  # all depths are candidates

            # Metadata
            F = int(data['F'][0])
            R = int(data['R'][0])
            ratio = float(data['f_ratio'][0])

            # Load scene masks
            if self.use_local:
                static = self._load_mask(data, 'local_static', 'static')
                movable = self._load_mask(data, 'local_movable', 'movable')
                target_obj = self._load_mask(data, 'local_target_object', 'target_object')
                robot_region = self._load_mask(data, 'local_robot_region', 'robot_region')
                goal_region = self._load_mask(data, 'local_goal_sample_region', 'goal_sample_region')
                masks = [static, movable, target_obj, robot_region, goal_region]
            else:
                robot = self._load_mask(data, 'robot', 'robot_image')
                goal = self._load_mask(data, 'goal', 'goal_image')
                movable = self._load_mask(data, 'movable', 'movable_objects_image')
                static = self._load_mask(data, 'static', 'static_objects_image')
                target_obj = self._load_mask(data, 'target_object')
                masks = [robot, goal, movable, static, target_obj]

            # Apply transforms to each mask
            if self.transform:
                masks = [self.transform(m) for m in masks]

            # Coord grid
            if self.use_coord_grid:
                img_size = masks[0].shape[-1]
                ys, xs = np.meshgrid(
                    np.linspace(0, 1, img_size),
                    np.linspace(0, 1, img_size),
                    indexing='ij')
                coord_grid = np.stack([xs, ys], axis=-1).astype(np.float32)
                if self.transform:
                    coord_grid = self.transform(coord_grid)
                masks.append(coord_grid)

            context = torch.cat(masks, dim=0)  # (C, H, W)

        return {
            'context': context,
            'f_labels': torch.from_numpy(f_labels),         # (60, 10) ground truth
            'r_mask': torch.from_numpy(r_mask),              # (60, 10) exact per-depth reachability (for training loss)
            'cp_reachable': torch.from_numpy(cp_reachable),  # (60, 10) contact-point-level reachability (for inference masking)
            'F': F,
            'R': R,
            'ratio': ratio,
        }

    def _load_mask(self, data, *keys):
        """Load mask by trying multiple key names (for backward compatibility)."""
        for key in keys:
            if key in data:
                mask = data[key]
                if mask is not None:
                    return mask.copy()
        # Fallback: zero mask
        return np.zeros((224, 224), dtype=np.float32)


class ClassifierDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, List[str]],
        image_size: int = 64,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_coord_grid: bool = False,
        use_local: bool = True,
        train_split: float = 0.9,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_size = image_size
        self.use_coord_grid = use_coord_grid
        self.use_local = use_local
        self.train_split = train_split

        self.train_dataset = None
        self.val_dataset = None

    def _normalized_roots(self) -> List[Path]:
        if isinstance(self.data_dir, (list, tuple)):
            roots = self.data_dir
        else:
            roots = [self.data_dir]
        return [Path(root).expanduser() for root in roots]

    def _collect_npz_files(self) -> List[str]:
        roots = self._normalized_roots()
        cache_path = roots[0] / ".classifier_file_cache.txt" if roots else None

        if cache_path and cache_path.exists():
            with open(cache_path, 'r') as f:
                datafiles = [line.strip() for line in f if line.strip()]
            if datafiles and all(Path(datafiles[i]).exists() for i in [0, len(datafiles)//2, -1]):
                print(f"Loaded {len(datafiles)} files from cache")
                return datafiles

        datafiles = []
        for root in roots:
            if not root.exists():
                print(f"Warning: {root} does not exist, skipping")
                continue
            files = glob.glob(f"{root}/**/*.npz", recursive=True)
            print(f"Found {len(files)} files in {root.name}")
            datafiles.extend(files)

        unique_files = sorted(set(datafiles))
        print(f"Total unique files: {len(unique_files)}")

        if cache_path and unique_files:
            with open(cache_path, 'w') as f:
                f.write('\n'.join(unique_files))

        return unique_files

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Lambda(lambda x: x * 2 - 1),
        ])

        all_datafiles = self._collect_npz_files()
        if not all_datafiles:
            raise RuntimeError(f"No .npz files found under {self.data_dir}")

        rng = random.Random(0)
        rng.shuffle(all_datafiles)

        split_idx = int(len(all_datafiles) * self.train_split)
        split_idx = max(1, min(split_idx, len(all_datafiles) - 1))
        train_files = all_datafiles[:split_idx]
        val_files = all_datafiles[split_idx:]

        print(f"Train: {len(train_files)}, Val: {len(val_files)}")

        if stage == "fit" or stage is None:
            self.train_dataset = ClassifierDataset(
                train_files, transform=transform,
                use_local=self.use_local, use_coord_grid=self.use_coord_grid
            )
            self.val_dataset = ClassifierDataset(
                val_files, transform=transform,
                use_local=self.use_local, use_coord_grid=self.use_coord_grid
            )

        if stage == "test" or stage is None:
            self.test_dataset = ClassifierDataset(
                val_files, transform=transform,
                use_local=self.use_local, use_coord_grid=self.use_coord_grid
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.num_workers > 0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
