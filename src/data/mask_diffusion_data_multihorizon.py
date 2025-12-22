"""
Multi-Horizon Mask Diffusion Data Module

This data loader supports multi-horizon prediction where the model predicts
both goal_mask_a1 (next action) and goal_mask_a2 (second action if exists).

Key differences from mask_diffusion_data.py:
- Returns target_goals as [2, H, W] instead of single [H, W]
- Returns solution_depth for loss masking
- goal_mask_a2 is zeros when solution_depth == 1
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
from tqdm import tqdm

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


class MultiHorizonMaskDataset(Dataset):
    """NPZ-backed dataset for multi-horizon mask prediction."""

    def __init__(self, datafiles: List[str], transform=None, use_coord_grid=False):
        """
        Args:
            datafiles: List of paths to NPZ files
            transform: PyTorch transforms for data augmentation
            use_coord_grid: Whether to add coordinate grid channels
        """
        self.transform = transform
        self.use_coord_grid = use_coord_grid
        self.samples = datafiles

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        with np.load(sample) as data:
            # Load context masks
            static_image = data.get('local_static')
            movable_image = data.get('local_movable')
            target_object = data.get('local_target_object')
            robot_region = data.get('local_robot_region')
            goal_sample_region = data.get('local_goal_sample_region')

            # Load multi-horizon goal masks
            goal_mask_a1 = data.get('local_goal_mask_a1')
            goal_mask_a2 = data.get('local_goal_mask_a2')

            # Fallback for goal_mask_a1
            if goal_mask_a1 is None:
                goal_mask_a1 = data.get('local_target_goal')

            # Load solution_depth
            solution_depth = data.get('solution_depth', np.array([1]))
            if isinstance(solution_depth, np.ndarray):
                solution_depth = int(solution_depth.flat[0])

            # Handle missing masks
            if static_image is None:
                raise ValueError(f"Missing local_static in {sample}")

            static_image = static_image.copy()
            movable_image = movable_image.copy() if movable_image is not None else np.zeros_like(static_image)
            target_object = target_object.copy() if target_object is not None else np.zeros_like(static_image)
            robot_region = robot_region.copy() if robot_region is not None else np.zeros_like(static_image)
            goal_sample_region = goal_sample_region.copy() if goal_sample_region is not None else np.zeros_like(static_image)
            goal_mask_a1 = goal_mask_a1.copy() if goal_mask_a1 is not None else np.zeros_like(static_image)

            # goal_mask_a2 is zeros if doesn't exist (1-push case)
            if goal_mask_a2 is not None:
                goal_mask_a2 = goal_mask_a2.copy()
            else:
                goal_mask_a2 = np.zeros_like(static_image)

        image_size = static_image.shape[0]

        # Build coordinate grid
        if self.use_coord_grid:
            ys, xs = np.meshgrid(
                np.linspace(0, 1, image_size),
                np.linspace(0, 1, image_size),
                indexing='ij'
            )
            coord_grid = np.stack([xs, ys], axis=-1)
            coord_grid = coord_grid.reshape(image_size, image_size, 2).astype(np.float32)

        # Stack goal masks as [2, H, W]
        target_goals = np.stack([goal_mask_a1, goal_mask_a2], axis=0).astype(np.float32)

        if self.transform:
            ret = {
                "static": self.transform(static_image),
                "movable": self.transform(movable_image),
                "target_object": self.transform(target_object),
                "robot_region": self.transform(robot_region),
                "goal_sample_region": self.transform(goal_sample_region),
                "target_goals": torch.from_numpy(target_goals),  # [2, H, W] - transform separately
                "solution_depth": solution_depth,
            }
            # Apply transform to target_goals (resize)
            # Note: target_goals is [2, H, W], transform expects [H, W] or [C, H, W]
            # We need to handle this carefully
            tg_transformed = []
            for i in range(2):
                tg_transformed.append(self.transform(target_goals[i]))
            ret["target_goals"] = torch.stack(tg_transformed, dim=0).squeeze(1)  # [2, H, W]

            if self.use_coord_grid:
                ret["coord_grid"] = self.transform(coord_grid)
            return ret

        ret = {
            "static": static_image,
            "movable": movable_image,
            "target_object": target_object,
            "robot_region": robot_region,
            "goal_sample_region": goal_sample_region,
            "target_goals": target_goals,  # [2, H, W]
            "solution_depth": solution_depth,
        }
        if self.use_coord_grid:
            ret["coord_grid"] = coord_grid
        return ret


class MultiHorizonMaskHDF5Dataset(Dataset):
    """HDF5-backed dataset for multi-horizon mask prediction."""

    def __init__(
        self,
        h5_path: str,
        indices: List[int],
        transform=None,
        use_coord_grid: bool = False,
    ):
        if not HAS_H5PY:
            raise ImportError("h5py required for HDF5 dataset. Install with: pip install h5py")

        self.h5_path = h5_path
        self.indices = indices
        self.transform = transform
        self.use_coord_grid = use_coord_grid
        self._h5_file = None

    def _get_h5_file(self):
        """Lazy open HDF5 file (needed for multiprocessing)."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        h5f = self._get_h5_file()
        real_idx = self.indices[idx]

        # Load context masks
        static_image = h5f['local_static'][real_idx] if 'local_static' in h5f else None
        movable_image = h5f['local_movable'][real_idx] if 'local_movable' in h5f else None
        target_object = h5f['local_target_object'][real_idx] if 'local_target_object' in h5f else None
        robot_region = h5f['local_robot_region'][real_idx] if 'local_robot_region' in h5f else None
        goal_sample_region = h5f['local_goal_sample_region'][real_idx] if 'local_goal_sample_region' in h5f else None

        # Load multi-horizon goal masks
        goal_mask_a1 = h5f['local_goal_mask_a1'][real_idx] if 'local_goal_mask_a1' in h5f else None
        goal_mask_a2 = h5f['local_goal_mask_a2'][real_idx] if 'local_goal_mask_a2' in h5f else None

        # Fallback for goal_mask_a1
        if goal_mask_a1 is None:
            goal_mask_a1 = h5f['local_target_goal'][real_idx] if 'local_target_goal' in h5f else None

        # Load solution_depth
        if 'solution_depth' in h5f:
            solution_depth = int(h5f['solution_depth'][real_idx])
        else:
            solution_depth = 1

        # Handle missing masks
        if static_image is None:
            raise ValueError(f"Missing local_static at index {real_idx}")

        if movable_image is None:
            movable_image = np.zeros_like(static_image)
        if target_object is None:
            target_object = np.zeros_like(static_image)
        if robot_region is None:
            robot_region = np.zeros_like(static_image)
        if goal_sample_region is None:
            goal_sample_region = np.zeros_like(static_image)
        if goal_mask_a1 is None:
            goal_mask_a1 = np.zeros_like(static_image)
        if goal_mask_a2 is None:
            goal_mask_a2 = np.zeros_like(static_image)

        image_size = static_image.shape[0]

        # Build coordinate grid
        if self.use_coord_grid:
            ys, xs = np.meshgrid(
                np.linspace(0, 1, image_size),
                np.linspace(0, 1, image_size),
                indexing='ij'
            )
            coord_grid = np.stack([xs, ys], axis=-1).astype(np.float32)

        # Stack goal masks as [2, H, W]
        target_goals = np.stack([goal_mask_a1, goal_mask_a2], axis=0).astype(np.float32)

        ret = {
            "static": static_image,
            "movable": movable_image,
            "target_object": target_object,
            "robot_region": robot_region,
            "goal_sample_region": goal_sample_region,
            "target_goals": target_goals,  # [2, H, W]
            "solution_depth": solution_depth,
        }

        if self.use_coord_grid:
            ret["coord_grid"] = coord_grid

        if self.transform:
            # Transform context masks
            ret["static"] = self.transform(ret["static"])
            ret["movable"] = self.transform(ret["movable"])
            ret["target_object"] = self.transform(ret["target_object"])
            ret["robot_region"] = self.transform(ret["robot_region"])
            ret["goal_sample_region"] = self.transform(ret["goal_sample_region"])

            # Transform target_goals [2, H, W] - each channel separately
            tg_transformed = []
            for i in range(2):
                tg_transformed.append(self.transform(target_goals[i]))
            ret["target_goals"] = torch.stack(tg_transformed, dim=0).squeeze(1)  # [2, H, W]

            if self.use_coord_grid:
                ret["coord_grid"] = self.transform(ret["coord_grid"])

        return ret

    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()


class MultiHorizonDataModule(pl.LightningDataModule):
    """Lightning DataModule for multi-horizon mask prediction."""

    def __init__(
        self,
        data_dir: Union[str, List[str]],
        image_size: int = 64,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_coord_grid: bool = False,
        train_split: float = 0.9,
        use_h5: bool = True,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_size = image_size
        self.use_coord_grid = use_coord_grid
        self.train_split = train_split
        self.use_h5 = use_h5

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

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
        roots = self._normalized_roots()
        cache_path = roots[0] / ".file_list_cache_multihorizon.txt" if roots else None

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
                print(f"Warning: data path {root} does not exist, skipping")
                continue
            if root.is_file():
                if str(root).endswith('.npz'):
                    datafiles.append(str(root))
                continue
            print(f"Scanning {root} for .npz files")
            files = glob.glob(f"{root}/**/*.npz", recursive=True)
            print(f"  Found {len(files)} files in {root.name}")
            datafiles.extend(files)

        unique_files = sorted(set(datafiles))
        print(f"Total unique files: {len(unique_files)}")

        if cache_path and unique_files:
            print(f"Caching file list to {cache_path}")
            with open(cache_path, 'w') as f:
                f.write('\n'.join(unique_files))

        return unique_files

    def _find_h5_file(self) -> Optional[str]:
        """Check if an HDF5 file exists in the data directory."""
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
        """Set up train/val/test datasets."""
        if self.train_dataset is not None:
            return

        print(f"[MultiHorizon] Starting setup with stage={stage}")
        print(f"[MultiHorizon] Data dir: {self.data_dir}")

        h5_path = self._find_h5_file() if self.use_h5 else None

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Lambda(lambda x: x * 2 - 1),  # Normalize to [-1, 1]
        ])

        if h5_path and HAS_H5PY:
            print(f"[MultiHorizon] Found HDF5 file: {h5_path}")

            with h5py.File(h5_path, 'r') as h5f:
                n_samples = h5f.attrs.get('n_samples', len(h5f[list(h5f.keys())[0]]))
            print(f"[MultiHorizon] Total samples in HDF5: {n_samples}")

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

            print(f"[MultiHorizon] Train/val split: {self.train_split:.0%} train, {1-self.train_split:.0%} val")
            print(f"[MultiHorizon] Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

            if stage == "fit" or stage is None:
                self.train_dataset = MultiHorizonMaskHDF5Dataset(
                    h5_path, train_indices, transform=transform,
                    use_coord_grid=self.use_coord_grid
                )
                self.val_dataset = MultiHorizonMaskHDF5Dataset(
                    h5_path, val_indices, transform=transform,
                    use_coord_grid=self.use_coord_grid
                )

            if stage == "test" or stage is None:
                self.test_dataset = MultiHorizonMaskHDF5Dataset(
                    h5_path, val_indices, transform=transform,
                    use_coord_grid=self.use_coord_grid
                )
        else:
            print(f"[MultiHorizon] Collecting .npz files...")

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

            print(f"[MultiHorizon] Train/val split: {self.train_split:.0%} train, {1-self.train_split:.0%} val")
            print(f"[MultiHorizon] Train files: {len(train_datafiles)}, Val files: {len(val_datafiles)}")

            if stage == "fit" or stage is None:
                self.train_dataset = MultiHorizonMaskDataset(
                    train_datafiles, transform=transform,
                    use_coord_grid=self.use_coord_grid
                )
                self.val_dataset = MultiHorizonMaskDataset(
                    val_datafiles, transform=transform,
                    use_coord_grid=self.use_coord_grid
                )

            if stage == "test" or stage is None:
                self.test_dataset = MultiHorizonMaskDataset(
                    val_datafiles, transform=transform,
                    use_coord_grid=self.use_coord_grid
                )

        print(f"[MultiHorizon] Batch size: {self.batch_size}")
        print(f"[MultiHorizon] Image size: {self.image_size}")
        print(f"[MultiHorizon] use_coord_grid: {self.use_coord_grid}")
        print(f"[MultiHorizon] Setup complete!")

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

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )
