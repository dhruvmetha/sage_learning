"""
Data module for cropped target prediction.

This module loads full-resolution context but center-crops the target
to a smaller size for more focused prediction.

Context channels (64x64): static, movable, target_object, robot_region, goal_sample_region
Target (24x24): center-cropped target_goal
"""

from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Union
from torchvision import transforms
from pathlib import Path
import lightning.pytorch as pl
import numpy as np
import random
import glob
from tqdm import tqdm

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def center_crop(arr: np.ndarray, crop_size: int) -> np.ndarray:
    """Center crop a 2D array to crop_size x crop_size."""
    h, w = arr.shape[-2:]
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    if arr.ndim == 2:
        return arr[start_h:start_h + crop_size, start_w:start_w + crop_size]
    else:
        return arr[..., start_h:start_h + crop_size, start_w:start_w + crop_size]


class MaskDiffusionCroppedDataset(Dataset):
    """Dataset that returns full context and center-cropped target."""

    def __init__(
        self,
        datafiles: List[str],
        context_size: int = 64,
        crop_size: int = 24,
        split: str = "train",
    ):
        """
        Args:
            datafiles: List of .npz file paths
            context_size: Size to resize context channels to (default 64)
            crop_size: Size of center crop for target (default 24)
            split: Dataset split name
        """
        self.datafiles = datafiles
        self.context_size = context_size
        self.crop_size = crop_size
        self.split = split

        # Transform for context (full resolution)
        self.context_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((context_size, context_size)),
            transforms.Lambda(lambda x: x * 2 - 1),  # Scale to [-1, 1]
        ])

        # Transform for target (resize to context_size first, then crop will happen)
        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((context_size, context_size)),
            transforms.CenterCrop(crop_size),
            transforms.Lambda(lambda x: x * 2 - 1),
        ])

    def __len__(self):
        return len(self.datafiles)

    def __getitem__(self, idx):
        sample_path = self.datafiles[idx]

        with np.load(sample_path) as data:
            # Load local masks (object-centered)
            static = data.get('local_static')
            movable = data.get('local_movable')
            target_object = data.get('local_target_object')
            robot_region = data.get('local_robot_region')
            goal_sample_region = data.get('local_goal_sample_region')

            # Load target (use local_goal_mask_a1 or fallback)
            target_goal = data.get('local_goal_mask_a1')
            if target_goal is None:
                target_goal = data.get('local_target_goal')

            # Copy arrays
            static = static.copy() if static is not None else np.zeros((224, 224))
            movable = movable.copy() if movable is not None else np.zeros_like(static)
            target_object = target_object.copy() if target_object is not None else np.zeros_like(static)
            robot_region = robot_region.copy() if robot_region is not None else np.zeros_like(static)
            goal_sample_region = goal_sample_region.copy() if goal_sample_region is not None else np.zeros_like(static)
            target_goal = target_goal.copy() if target_goal is not None else np.zeros_like(static)

        # Apply transforms
        # Context channels at full resolution
        context = {
            'static': self.context_transform(static),
            'movable': self.context_transform(movable),
            'target_object': self.context_transform(target_object),
            'robot_region': self.context_transform(robot_region),
            'goal_sample_region': self.context_transform(goal_sample_region),
        }

        # Target center-cropped
        target = self.target_transform(target_goal)

        return {
            'context': context,
            'target_goal': target,
        }


class MaskDiffusionCroppedHDF5Dataset(Dataset):
    """HDF5-backed dataset for cropped target prediction."""

    def __init__(
        self,
        h5_path: str,
        indices: List[int],
        context_size: int = 64,
        crop_size: int = 24,
    ):
        if not HAS_H5PY:
            raise ImportError("h5py required for HDF5 dataset")

        self.h5_path = h5_path
        self.indices = indices
        self.context_size = context_size
        self.crop_size = crop_size
        self._h5_file = None

        # Transforms
        self.context_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((context_size, context_size)),
            transforms.Lambda(lambda x: x * 2 - 1),
        ])

        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((context_size, context_size)),
            transforms.CenterCrop(crop_size),
            transforms.Lambda(lambda x: x * 2 - 1),
        ])

    def _get_h5_file(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        h5f = self._get_h5_file()
        real_idx = self.indices[idx]

        # Load local masks
        static = h5f['local_static'][real_idx] if 'local_static' in h5f else None
        movable = h5f['local_movable'][real_idx] if 'local_movable' in h5f else None
        target_object = h5f['local_target_object'][real_idx] if 'local_target_object' in h5f else None
        robot_region = h5f['local_robot_region'][real_idx] if 'local_robot_region' in h5f else None
        goal_sample_region = h5f['local_goal_sample_region'][real_idx] if 'local_goal_sample_region' in h5f else None

        # Target
        target_goal = h5f['local_goal_mask_a1'][real_idx] if 'local_goal_mask_a1' in h5f else None
        if target_goal is None:
            target_goal = h5f['local_target_goal'][real_idx] if 'local_target_goal' in h5f else None

        # Handle missing data
        if static is None:
            static = np.zeros((224, 224), dtype=np.float32)
        if movable is None:
            movable = np.zeros_like(static)
        if target_object is None:
            target_object = np.zeros_like(static)
        if robot_region is None:
            robot_region = np.zeros_like(static)
        if goal_sample_region is None:
            goal_sample_region = np.zeros_like(static)
        if target_goal is None:
            target_goal = np.zeros_like(static)

        # Apply transforms
        context = {
            'static': self.context_transform(static),
            'movable': self.context_transform(movable),
            'target_object': self.context_transform(target_object),
            'robot_region': self.context_transform(robot_region),
            'goal_sample_region': self.context_transform(goal_sample_region),
        }

        target = self.target_transform(target_goal)

        return {
            'context': context,
            'target_goal': target,
        }

    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()


class MaskDiffusionCroppedDataModule(pl.LightningDataModule):
    """
    Data module for cropped target prediction.

    Returns batches with:
    - context: dict of 5 channels at context_size x context_size
    - target_goal: tensor at crop_size x crop_size (center cropped)
    """

    def __init__(
        self,
        data_dir: Union[str, List[str]],
        context_size: int = 64,
        crop_size: int = 24,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_split: float = 1.0,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.context_size = context_size
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_split = train_split

        self.train_dataset = None
        self.val_dataset = None

    def _normalized_roots(self) -> List[Path]:
        if isinstance(self.data_dir, (list, tuple)):
            roots = self.data_dir
        else:
            roots = [self.data_dir]
        return [Path(r).expanduser() for r in roots]

    def _collect_npz_files(self) -> List[str]:
        roots = self._normalized_roots()
        cache_path = roots[0] / ".file_list_cache.txt" if roots else None

        if cache_path and cache_path.exists():
            print(f"Loading cached file list from {cache_path}")
            with open(cache_path, 'r') as f:
                datafiles = [line.strip() for line in f if line.strip()]
            if datafiles and all(Path(datafiles[i]).exists() for i in [0, len(datafiles)//2, -1]):
                print(f"Loaded {len(datafiles)} files from cache")
                return datafiles
            print("Cache invalid, rescanning...")

        datafiles = []
        for root in tqdm(roots, desc="Scanning data roots"):
            if not root.exists():
                print(f"Warning: {root} does not exist, skipping")
                continue
            if root.is_file() and str(root).endswith('.npz'):
                datafiles.append(str(root))
                continue
            files = glob.glob(f"{root}/**/*.npz", recursive=True)
            print(f"  Found {len(files)} files in {root.name}")
            datafiles.extend(files)

        unique_files = sorted(set(datafiles))
        print(f"Total unique files: {len(unique_files)}")

        if cache_path and unique_files:
            with open(cache_path, 'w') as f:
                f.write('\n'.join(unique_files))

        return unique_files

    def _find_h5_file(self) -> Optional[str]:
        roots = self._normalized_roots()
        for root in roots:
            h5_path = root.parent / f"{root.name}.h5"
            if h5_path.exists():
                return str(h5_path)
            h5_path = root / "data.h5"
            if h5_path.exists():
                return str(h5_path)
            h5_files = list(root.glob("*.h5"))
            if h5_files:
                return str(h5_files[0])
        return None

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return

        print(f"[setup] Context size: {self.context_size}, Crop size: {self.crop_size}")

        h5_path = self._find_h5_file()

        if h5_path and HAS_H5PY:
            print(f"[setup] Using HDF5: {h5_path}")

            with h5py.File(h5_path, 'r') as h5f:
                n_samples = h5f.attrs.get('n_samples', len(h5f[list(h5f.keys())[0]]))

            rng = random.Random(0)
            all_indices = list(range(n_samples))
            rng.shuffle(all_indices)

            if n_samples == 1:
                train_indices = all_indices
                val_indices = all_indices
            else:
                split_idx = int(n_samples * self.train_split)
                split_idx = max(1, min(split_idx, n_samples - 1))
                train_indices = all_indices[:split_idx]
                val_indices = all_indices[split_idx:]

            print(f"[setup] Train: {len(train_indices)}, Val: {len(val_indices)}")

            if stage == "fit" or stage is None:
                self.train_dataset = MaskDiffusionCroppedHDF5Dataset(
                    h5_path, train_indices,
                    context_size=self.context_size, crop_size=self.crop_size
                )
                self.val_dataset = MaskDiffusionCroppedHDF5Dataset(
                    h5_path, val_indices,
                    context_size=self.context_size, crop_size=self.crop_size
                )
        else:
            print("[setup] Using NPZ files")
            all_datafiles = self._collect_npz_files()
            if not all_datafiles:
                raise RuntimeError(f"No .npz files found under {self.data_dir}")

            rng = random.Random(0)
            rng.shuffle(all_datafiles)

            if len(all_datafiles) == 1:
                train_files = all_datafiles
                val_files = all_datafiles
            else:
                split_idx = int(len(all_datafiles) * self.train_split)
                split_idx = max(1, min(split_idx, len(all_datafiles) - 1))
                train_files = all_datafiles[:split_idx]
                val_files = all_datafiles[split_idx:]

            print(f"[setup] Train: {len(train_files)}, Val: {len(val_files)}")

            if stage == "fit" or stage is None:
                self.train_dataset = MaskDiffusionCroppedDataset(
                    train_files,
                    context_size=self.context_size, crop_size=self.crop_size,
                    split="train"
                )
                self.val_dataset = MaskDiffusionCroppedDataset(
                    val_files,
                    context_size=self.context_size, crop_size=self.crop_size,
                    split="val"
                )

        print("[setup] Complete!")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )


# =============================================================================
# Multi-Horizon Cropped Classes (for 2-push training)
# =============================================================================


class MaskDiffusionCroppedMultiHorizonDataset(Dataset):
    """Dataset that returns full context and center-cropped multi-horizon targets."""

    def __init__(
        self,
        datafiles: List[str],
        context_size: int = 64,
        crop_size: int = 24,
        split: str = "train",
    ):
        """
        Args:
            datafiles: List of .npz file paths
            context_size: Size to resize context channels to (default 64)
            crop_size: Size of center crop for targets (default 24)
            split: Dataset split name
        """
        self.datafiles = datafiles
        self.context_size = context_size
        self.crop_size = crop_size
        self.split = split

        # Transform for context (full resolution)
        self.context_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((context_size, context_size)),
            transforms.Lambda(lambda x: x * 2 - 1),  # Scale to [-1, 1]
        ])

        # Transform for target (resize to context_size first, then crop)
        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((context_size, context_size)),
            transforms.CenterCrop(crop_size),
            transforms.Lambda(lambda x: x * 2 - 1),
        ])

    def __len__(self):
        return len(self.datafiles)

    def __getitem__(self, idx):
        sample_path = self.datafiles[idx]

        with np.load(sample_path) as data:
            # Load local masks (object-centered)
            static = data.get('local_static')
            movable = data.get('local_movable')
            target_object = data.get('local_target_object')
            robot_region = data.get('local_robot_region')
            goal_sample_region = data.get('local_goal_sample_region')

            # Load multi-horizon targets
            goal_mask_a1 = data.get('local_goal_mask_a1')
            if goal_mask_a1 is None:
                goal_mask_a1 = data.get('local_target_goal')
            goal_mask_a2 = data.get('local_goal_mask_a2')

            # Load solution_depth
            solution_depth = data.get('solution_depth', np.array([1]))
            if isinstance(solution_depth, np.ndarray):
                solution_depth = int(solution_depth.flat[0])

            # Copy arrays
            static = static.copy() if static is not None else np.zeros((224, 224))
            movable = movable.copy() if movable is not None else np.zeros_like(static)
            target_object = target_object.copy() if target_object is not None else np.zeros_like(static)
            robot_region = robot_region.copy() if robot_region is not None else np.zeros_like(static)
            goal_sample_region = goal_sample_region.copy() if goal_sample_region is not None else np.zeros_like(static)
            goal_mask_a1 = goal_mask_a1.copy() if goal_mask_a1 is not None else np.zeros_like(static)
            goal_mask_a2 = goal_mask_a2.copy() if goal_mask_a2 is not None else np.zeros_like(static)

        # Apply transforms
        # Context channels at full resolution
        context = {
            'static': self.context_transform(static),
            'movable': self.context_transform(movable),
            'target_object': self.context_transform(target_object),
            'robot_region': self.context_transform(robot_region),
            'goal_sample_region': self.context_transform(goal_sample_region),
        }

        # Targets center-cropped - apply transform to each, then stack
        target_a1 = self.target_transform(goal_mask_a1)
        target_a2 = self.target_transform(goal_mask_a2)
        import torch
        target_goals = torch.cat([target_a1, target_a2], dim=0)  # [2, crop_size, crop_size]

        return {
            'context': context,
            'target_goals': target_goals,
            'solution_depth': solution_depth,
        }


class MaskDiffusionCroppedMultiHorizonHDF5Dataset(Dataset):
    """HDF5-backed dataset for cropped multi-horizon target prediction."""

    def __init__(
        self,
        h5_path: str,
        indices: List[int],
        context_size: int = 64,
        crop_size: int = 24,
    ):
        if not HAS_H5PY:
            raise ImportError("h5py required for HDF5 dataset")

        self.h5_path = h5_path
        self.indices = indices
        self.context_size = context_size
        self.crop_size = crop_size
        self._h5_file = None

        # Transforms
        self.context_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((context_size, context_size)),
            transforms.Lambda(lambda x: x * 2 - 1),
        ])

        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((context_size, context_size)),
            transforms.CenterCrop(crop_size),
            transforms.Lambda(lambda x: x * 2 - 1),
        ])

    def _get_h5_file(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        h5f = self._get_h5_file()
        real_idx = self.indices[idx]

        # Load local masks
        static = h5f['local_static'][real_idx] if 'local_static' in h5f else None
        movable = h5f['local_movable'][real_idx] if 'local_movable' in h5f else None
        target_object = h5f['local_target_object'][real_idx] if 'local_target_object' in h5f else None
        robot_region = h5f['local_robot_region'][real_idx] if 'local_robot_region' in h5f else None
        goal_sample_region = h5f['local_goal_sample_region'][real_idx] if 'local_goal_sample_region' in h5f else None

        # Multi-horizon targets
        goal_mask_a1 = h5f['local_goal_mask_a1'][real_idx] if 'local_goal_mask_a1' in h5f else None
        if goal_mask_a1 is None:
            goal_mask_a1 = h5f['local_target_goal'][real_idx] if 'local_target_goal' in h5f else None
        goal_mask_a2 = h5f['local_goal_mask_a2'][real_idx] if 'local_goal_mask_a2' in h5f else None

        # Solution depth
        if 'solution_depth' in h5f:
            solution_depth = int(h5f['solution_depth'][real_idx])
        else:
            solution_depth = 1

        # Handle missing data
        if static is None:
            static = np.zeros((224, 224), dtype=np.float32)
        if movable is None:
            movable = np.zeros_like(static)
        if target_object is None:
            target_object = np.zeros_like(static)
        if robot_region is None:
            robot_region = np.zeros_like(static)
        if goal_sample_region is None:
            goal_sample_region = np.zeros_like(static)
        if goal_mask_a1 is None:
            goal_mask_a1 = np.zeros_like(static)
        if goal_mask_a2 is None:
            goal_mask_a2 = np.zeros_like(static)

        # Apply transforms
        context = {
            'static': self.context_transform(static),
            'movable': self.context_transform(movable),
            'target_object': self.context_transform(target_object),
            'robot_region': self.context_transform(robot_region),
            'goal_sample_region': self.context_transform(goal_sample_region),
        }

        # Targets center-cropped
        import torch
        target_a1 = self.target_transform(goal_mask_a1)
        target_a2 = self.target_transform(goal_mask_a2)
        target_goals = torch.cat([target_a1, target_a2], dim=0)  # [2, crop_size, crop_size]

        return {
            'context': context,
            'target_goals': target_goals,
            'solution_depth': solution_depth,
        }

    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()


class MaskDiffusionCroppedMultiHorizonDataModule(pl.LightningDataModule):
    """
    Data module for cropped multi-horizon target prediction.

    Returns batches with:
    - context: dict of 5 channels at context_size x context_size
    - target_goals: tensor [2, crop_size, crop_size] (center cropped a1 and a2)
    - solution_depth: int (1 or 2)
    """

    def __init__(
        self,
        data_dir: Union[str, List[str]],
        context_size: int = 64,
        crop_size: int = 24,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_split: float = 0.9,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.context_size = context_size
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_split = train_split

        self.train_dataset = None
        self.val_dataset = None

    def _normalized_roots(self) -> List[Path]:
        if isinstance(self.data_dir, (list, tuple)):
            roots = self.data_dir
        else:
            roots = [self.data_dir]
        return [Path(r).expanduser() for r in roots]

    def _collect_npz_files(self) -> List[str]:
        roots = self._normalized_roots()
        cache_path = roots[0] / ".file_list_cache_cropped_mh.txt" if roots else None

        if cache_path and cache_path.exists():
            print(f"Loading cached file list from {cache_path}")
            with open(cache_path, 'r') as f:
                datafiles = [line.strip() for line in f if line.strip()]
            if datafiles and all(Path(datafiles[i]).exists() for i in [0, len(datafiles)//2, -1]):
                print(f"Loaded {len(datafiles)} files from cache")
                return datafiles
            print("Cache invalid, rescanning...")

        datafiles = []
        for root in tqdm(roots, desc="Scanning data roots"):
            if not root.exists():
                print(f"Warning: {root} does not exist, skipping")
                continue
            if root.is_file() and str(root).endswith('.npz'):
                datafiles.append(str(root))
                continue
            files = glob.glob(f"{root}/**/*.npz", recursive=True)
            print(f"  Found {len(files)} files in {root.name}")
            datafiles.extend(files)

        unique_files = sorted(set(datafiles))
        print(f"Total unique files: {len(unique_files)}")

        if cache_path and unique_files:
            with open(cache_path, 'w') as f:
                f.write('\n'.join(unique_files))

        return unique_files

    def _find_h5_file(self) -> Optional[str]:
        roots = self._normalized_roots()
        for root in roots:
            # Check if root itself is an h5 file
            if root.is_file() and str(root).endswith('.h5'):
                return str(root)
            # Check for adjacent h5 file
            h5_path = root.parent / f"{root.name}.h5"
            if h5_path.exists():
                return str(h5_path)
            # Check for data.h5 inside directory
            h5_path = root / "data.h5"
            if h5_path.exists():
                return str(h5_path)
            # Check for any h5 file
            if root.is_dir():
                h5_files = list(root.glob("*.h5"))
                if h5_files:
                    return str(h5_files[0])
        return None

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return

        print(f"[CroppedMultiHorizon] Context size: {self.context_size}, Crop size: {self.crop_size}")

        h5_path = self._find_h5_file()

        if h5_path and HAS_H5PY:
            print(f"[CroppedMultiHorizon] Using HDF5: {h5_path}")

            with h5py.File(h5_path, 'r') as h5f:
                n_samples = h5f.attrs.get('n_samples', len(h5f[list(h5f.keys())[0]]))

            rng = random.Random(0)
            all_indices = list(range(n_samples))
            rng.shuffle(all_indices)

            if n_samples == 1:
                train_indices = all_indices
                val_indices = all_indices
            else:
                split_idx = int(n_samples * self.train_split)
                split_idx = max(1, min(split_idx, n_samples - 1))
                train_indices = all_indices[:split_idx]
                val_indices = all_indices[split_idx:]

            print(f"[CroppedMultiHorizon] Train: {len(train_indices)}, Val: {len(val_indices)}")

            if stage == "fit" or stage is None:
                self.train_dataset = MaskDiffusionCroppedMultiHorizonHDF5Dataset(
                    h5_path, train_indices,
                    context_size=self.context_size, crop_size=self.crop_size
                )
                self.val_dataset = MaskDiffusionCroppedMultiHorizonHDF5Dataset(
                    h5_path, val_indices,
                    context_size=self.context_size, crop_size=self.crop_size
                )
        else:
            print("[CroppedMultiHorizon] Using NPZ files")
            all_datafiles = self._collect_npz_files()
            if not all_datafiles:
                raise RuntimeError(f"No .npz files found under {self.data_dir}")

            rng = random.Random(0)
            rng.shuffle(all_datafiles)

            if len(all_datafiles) == 1:
                train_files = all_datafiles
                val_files = all_datafiles
            else:
                split_idx = int(len(all_datafiles) * self.train_split)
                split_idx = max(1, min(split_idx, len(all_datafiles) - 1))
                train_files = all_datafiles[:split_idx]
                val_files = all_datafiles[split_idx:]

            print(f"[CroppedMultiHorizon] Train: {len(train_files)}, Val: {len(val_files)}")

            if stage == "fit" or stage is None:
                self.train_dataset = MaskDiffusionCroppedMultiHorizonDataset(
                    train_files,
                    context_size=self.context_size, crop_size=self.crop_size,
                    split="train"
                )
                self.val_dataset = MaskDiffusionCroppedMultiHorizonDataset(
                    val_files,
                    context_size=self.context_size, crop_size=self.crop_size,
                    split="val"
                )

        print("[CroppedMultiHorizon] Setup complete!")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )