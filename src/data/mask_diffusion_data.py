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

class MaskDiffusionDataset(Dataset):
    def __init__(self, datafiles: List[str], transform=None, split="train", use_coord_grid=False, use_local=False):
        """
        Args:
            data_dir: path to data directory
            transform: pytorch transforms for data augmentation
            split: one of 'train', 'val', or 'test'
            use_coord_grid: whether to add coordinate grid channels
            use_local: whether to use local (object-centered) masks instead of global masks
        """
        self.transform = transform
        self.split = split
        self.use_coord_grid = use_coord_grid
        self.use_local = use_local


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
            if self.use_local:
                # Load local (object-centered) masks
                # Input channels: local_static, local_movable, local_target_object, local_robot_region, local_goal_sample_region
                # Target: local_target_goal
                static_image = data.get('local_static')
                movable_image = data.get('local_movable')
                target_object = data.get('local_target_object')
                robot_region = data.get('local_robot_region')
                goal_sample_region = data.get('local_goal_sample_region')
                target_goal = data.get('local_target_goal')

                # Skip samples without local masks
                if static_image is None or target_object is None:
                    # Fallback to global if local not available
                    static_image = (data.get('static_objects_image') if 'static_objects_image' in data else data.get('static', np.zeros((224, 224)))).copy()
                    movable_image = (data.get('movable_objects_image') if 'movable_objects_image' in data else data.get('movable', np.zeros((224, 224)))).copy()
                    target_object = data.get('target_object', np.zeros((224, 224)))
                    if target_object is not None:
                        target_object = target_object.copy()
                    robot_region = data.get('robot_region', np.zeros((224, 224)))
                    if robot_region is not None:
                        robot_region = robot_region.copy()
                    goal_sample_region = data.get('goal_sample_region', np.zeros((224, 224)))
                    if goal_sample_region is not None:
                        goal_sample_region = goal_sample_region.copy()
                    target_goal = data.get('target_goal')
                    if target_goal is not None:
                        target_goal = target_goal.copy()
                else:
                    static_image = static_image.copy()
                    movable_image = movable_image.copy() if movable_image is not None else np.zeros_like(static_image)
                    target_object = target_object.copy()
                    robot_region = robot_region.copy() if robot_region is not None else np.zeros_like(static_image)
                    goal_sample_region = goal_sample_region.copy() if goal_sample_region is not None else np.zeros_like(static_image)
                    target_goal = target_goal.copy() if target_goal is not None else None

                image_size = static_image.shape[0]

                if self.use_coord_grid:
                    ys, xs = np.meshgrid(np.linspace(0, 1, image_size),
                                         np.linspace(0, 1, image_size),
                                         indexing='ij')
                    coord_grid = np.stack([xs, ys], axis=-1)
                    coord_grid = coord_grid.reshape(image_size, image_size, 2).astype(np.float32)

                if self.transform:
                    ret = {
                        "static": self.transform(static_image),
                        "movable": self.transform(movable_image),
                        "target_object": self.transform(target_object),
                        "robot_region": self.transform(robot_region),
                        "goal_sample_region": self.transform(goal_sample_region),
                    }
                    if target_goal is not None:
                        ret["target_goal"] = self.transform(target_goal)
                    if self.use_coord_grid:
                        ret["coord_grid"] = self.transform(coord_grid)
                    return ret

                ret = {
                    "static": static_image,
                    "movable": movable_image,
                    "target_object": target_object,
                    "robot_region": robot_region,
                    "goal_sample_region": goal_sample_region,
                }
                if target_goal is not None:
                    ret["target_goal"] = target_goal
                if self.use_coord_grid:
                    ret["coord_grid"] = coord_grid
                return ret

            else:
                # Load global masks (original behavior)
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
                "movable": self.transform(movable_objects_image),
                "static": self.transform(static_objects_image),
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
            "movable": movable_objects_image,
            "static": static_objects_image,
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

class MaskDiffusionHDF5Dataset(Dataset):
    """HDF5-backed dataset for much faster loading with large datasets."""

    def __init__(
        self,
        h5_path: str,
        indices: List[int],
        transform=None,
        use_coord_grid: bool = False,
        use_local: bool = False,
    ):
        if not HAS_H5PY:
            raise ImportError("h5py required for HDF5 dataset. Install with: pip install h5py")

        self.h5_path = h5_path
        self.indices = indices
        self.transform = transform
        self.use_coord_grid = use_coord_grid
        self.use_local = use_local
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

        if self.use_local:
            static_image = h5f['local_static'][real_idx] if 'local_static' in h5f else None
            movable_image = h5f['local_movable'][real_idx] if 'local_movable' in h5f else None
            target_object = h5f['local_target_object'][real_idx] if 'local_target_object' in h5f else None
            robot_region = h5f['local_robot_region'][real_idx] if 'local_robot_region' in h5f else None
            goal_sample_region = h5f['local_goal_sample_region'][real_idx] if 'local_goal_sample_region' in h5f else None
            target_goal = h5f['local_target_goal'][real_idx] if 'local_target_goal' in h5f else None

            if static_image is None:
                # Fallback to global
                static_image = h5f.get('static_objects_image', h5f.get('static'))[real_idx]
                movable_image = h5f.get('movable_objects_image', h5f.get('movable'))[real_idx]
                target_object = h5f['target_object'][real_idx] if 'target_object' in h5f else np.zeros_like(static_image)
                robot_region = h5f['robot_region'][real_idx] if 'robot_region' in h5f else np.zeros_like(static_image)
                goal_sample_region = h5f['goal_sample_region'][real_idx] if 'goal_sample_region' in h5f else np.zeros_like(static_image)
                target_goal = h5f['target_goal'][real_idx] if 'target_goal' in h5f else None
            else:
                if movable_image is None:
                    movable_image = np.zeros_like(static_image)
                if robot_region is None:
                    robot_region = np.zeros_like(static_image)
                if goal_sample_region is None:
                    goal_sample_region = np.zeros_like(static_image)

            image_size = static_image.shape[0]
            ret = {
                "static": static_image,
                "movable": movable_image,
                "target_object": target_object,
                "robot_region": robot_region,
                "goal_sample_region": goal_sample_region,
            }
            if target_goal is not None:
                ret["target_goal"] = target_goal

        else:
            robot_image = h5f.get('robot_image', h5f.get('robot'))[real_idx]
            goal_image = h5f.get('goal_image', h5f.get('goal'))[real_idx]
            movable_image = h5f.get('movable_objects_image', h5f.get('movable'))[real_idx]
            static_image = h5f.get('static_objects_image', h5f.get('static'))[real_idx]
            target_object = h5f['target_object'][real_idx] if 'target_object' in h5f else None
            target_goal = h5f['target_goal'][real_idx] if 'target_goal' in h5f else None

            image_size = robot_image.shape[0]
            ret = {
                "robot": robot_image,
                "goal": goal_image,
                "movable": movable_image,
                "static": static_image,
            }
            if target_object is not None:
                ret["target_object"] = target_object
            if target_goal is not None:
                ret["target_goal"] = target_goal

        if self.use_coord_grid:
            ys, xs = np.meshgrid(
                np.linspace(0, 1, image_size),
                np.linspace(0, 1, image_size),
                indexing='ij'
            )
            coord_grid = np.stack([xs, ys], axis=-1).astype(np.float32)
            ret["coord_grid"] = coord_grid

        if self.transform:
            ret = {k: self.transform(v) for k, v in ret.items()}

        return ret

    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()


class MaskDiffusionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, List[str]],
        image_size: int = 64,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_coord_grid: bool = False,
        use_local: bool = False,
        train_split: float = 1, # already separated testing data
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
        # Try to load cached file list first
        roots = self._normalized_roots()
        cache_path = roots[0] / ".file_list_cache.txt" if roots else None

        if cache_path and cache_path.exists():
            print(f"Loading cached file list from {cache_path}")
            with open(cache_path, 'r') as f:
                datafiles = [line.strip() for line in f if line.strip()]
            # Verify cache is still valid (check a few random files exist)
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

        # Remove duplicates while keeping deterministic order
        unique_files = sorted(set(datafiles))
        print(f"Total unique files: {len(unique_files)}")

        # Cache the file list for next time
        if cache_path and unique_files:
            print(f"Caching file list to {cache_path}")
            with open(cache_path, 'w') as f:
                f.write('\n'.join(unique_files))

        return unique_files
        
        
    def _find_h5_file(self) -> Optional[str]:
        """Check if an HDF5 file exists in the data directory."""
        roots = self._normalized_roots()
        for root in roots:
            # Check for .h5 file with same name as directory
            h5_path = root.parent / f"{root.name}.h5"
            if h5_path.exists():
                return str(h5_path)
            # Check for data.h5 inside directory
            h5_path = root / "data.h5"
            if h5_path.exists():
                return str(h5_path)
            # Check for any .h5 file in directory
            h5_files = list(root.glob("*.h5"))
            if h5_files:
                return str(h5_files[0])
        return None

    def setup(self, stage: Optional[str] = None):
        """
        Load data. Set variables: self.train_dataset, self.val_dataset, self.test_dataset
        """
        # Skip if already setup (avoid duplicate work from multiple workers)
        if self.train_dataset is not None:
            return

        print(f"[setup] Starting setup with stage={stage}")
        print(f"[setup] Data dir: {self.data_dir}")

        # Check for HDF5 file first (much faster for large datasets)
        h5_path = self._find_h5_file()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Lambda(lambda x: x * 2 - 1),
        ])

        if h5_path and HAS_H5PY:
            print(f"[setup] Found HDF5 file: {h5_path}")
            print(f"[setup] Using fast HDF5 data loading")

            # Get number of samples from HDF5 file
            with h5py.File(h5_path, 'r') as h5f:
                n_samples = h5f.attrs.get('n_samples', len(h5f[list(h5f.keys())[0]]))
            print(f"[setup] Total samples in HDF5: {n_samples}")

            # Create shuffled indices
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

            print(f"[setup] Train/val split: {self.train_split:.0%} train, {1-self.train_split:.0%} val")
            print(f"[setup] Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

            if stage == "fit" or stage is None:
                self.train_dataset = MaskDiffusionHDF5Dataset(
                    h5_path, train_indices, transform=transform,
                    use_coord_grid=self.use_coord_grid, use_local=self.use_local
                )
                self.val_dataset = MaskDiffusionHDF5Dataset(
                    h5_path, val_indices, transform=transform,
                    use_coord_grid=self.use_coord_grid, use_local=self.use_local
                )

            if stage == "test" or stage is None:
                self.test_dataset = MaskDiffusionHDF5Dataset(
                    h5_path, val_indices, transform=transform,
                    use_coord_grid=self.use_coord_grid, use_local=self.use_local
                )
        else:
            # Fall back to NPZ file loading
            print(f"[setup] Collecting .npz files...")

            all_datafiles = self._collect_npz_files()
            if not all_datafiles:
                raise RuntimeError(f"No .npz files found under {self.data_dir}")

            print(f"[setup] Shuffling {len(all_datafiles)} files with seed=0")
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

            print(f"[setup] Train/val split: {self.train_split:.0%} train, {1-self.train_split:.0%} val")
            print(f"[setup] Train files: {len(train_datafiles)}, Val files: {len(val_datafiles)}")

            if stage == "fit" or stage is None:
                print(f"[setup] Creating train dataset...")
                self.train_dataset = MaskDiffusionDataset(
                    train_datafiles, split="train", transform=transform,
                    use_coord_grid=self.use_coord_grid, use_local=self.use_local
                )
                print(f"[setup] Creating val dataset...")
                self.val_dataset = MaskDiffusionDataset(
                    val_datafiles, split="val", transform=transform,
                    use_coord_grid=self.use_coord_grid, use_local=self.use_local
                )

            if stage == "test" or stage is None:
                print(f"[setup] Creating test dataset...")
                self.test_dataset = MaskDiffusionDataset(
                    val_datafiles, split="test", transform=transform,
                    use_coord_grid=self.use_coord_grid, use_local=self.use_local
                )

        print(f"[setup] Batch size: {self.batch_size}")
        print(f"[setup] Image size: {self.image_size}")
        print(f"[setup] use_local: {self.use_local}, use_coord_grid: {self.use_coord_grid}")
        print(f"[setup] Setup complete!")
    
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