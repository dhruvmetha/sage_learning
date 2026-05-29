"""Data module for SE(2) delta prediction with local_tight conditioning.

Single-horizon: each NPZ contributes ONE prediction sample.

Reads from the v2_dualcrop NPZ schema:
  - context (5 channels @ 64x64 after resize): local_tight_static, local_tight_movable,
    local_tight_target_object, local_tight_robot_region, local_tight_goal_sample_region
  - target (3-vec): (Δx, Δy, Δθ) where Δ = action.target − pre_pose_a1, all world-frame
  - aux (loader passes through, model conditions on / inference uses):
    pre_pose_a1, edge_idx_a1, depth_idx_a1, target_object_size

Δθ is stored raw (radians) — for our primitive data |Δθ| ≤ ~75°, well within
[-π, π], so the wrap discontinuity at ±π is never engaged. Plain Euclidean
3-vec is the simplest correct choice.
"""

import glob
import random
from pathlib import Path
from typing import List, Optional, Union

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def _encode_target(se2_target_a1: np.ndarray) -> np.ndarray:
    """Pass-through to float32 (Δx, Δy, Δθ).

    Plain Euclidean 3-vec — for our primitive data |Δθ| stays well within
    [-π, π], so direct MSE works correctly.
    """
    return np.asarray(se2_target_a1, dtype=np.float32)


class SE2CroppedDataset(Dataset):
    """NPZ-backed single-horizon SE(2) dataset.

    Returns a dict per sample:
        'context':  (5, context_size, context_size) float tensor in [-1, 1]
                    (from local_tight_* channels at 224→context_size)
        'target':   (3,) float tensor — (Δx, Δy, Δθ) world-frame
        'pre_pose': (3,) float tensor — crop center anchor (px, py, pθ)
        'edge_idx': (1,) int tensor — primitive direction (-1 if absent)
        'depth_idx':(1,) int tensor — primitive length step (-1 if absent)
        'obj_size': (3,) float tensor — (sx, sy, sz)
    """

    CONTEXT_KEYS = [
        'local_tight_static',
        'local_tight_movable',
        'local_tight_target_object',
        'local_tight_robot_region',
        'local_tight_goal_sample_region',
    ]

    def __init__(self, datafiles: List[str], context_size: int = 64, split: str = "train"):
        self.datafiles = datafiles
        self.context_size = context_size
        self.split = split

        self.context_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((context_size, context_size)),
            transforms.Lambda(lambda x: x * 2 - 1),  # [0,1] → [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.datafiles)

    def __getitem__(self, idx: int):
        with np.load(self.datafiles[idx]) as data:
            channels = []
            ref_shape = None
            for k in self.CONTEXT_KEYS:
                arr = data.get(k)
                if arr is None:
                    if ref_shape is None:
                        arr = np.zeros((224, 224), dtype=np.float32)
                    else:
                        arr = np.zeros(ref_shape, dtype=np.float32)
                else:
                    arr = arr.copy()
                    ref_shape = arr.shape
                channels.append(arr)

            se2_target = data['se2_target_a1']
            pre_pose = data['pre_pose_a1']
            edge_idx = int(data['edge_idx_a1'][0]) if 'edge_idx_a1' in data.files else -1
            depth_idx = int(data['depth_idx_a1'][0]) if 'depth_idx_a1' in data.files else -1
            obj_size = data['target_object_size'] if 'target_object_size' in data.files else np.zeros(3, dtype=np.float32)

        # Stack the 5 channels through the per-channel transform
        ctx_tensor = torch.stack([self.context_transform(c)[0] for c in channels], dim=0)

        target_4d = torch.from_numpy(_encode_target(se2_target))
        return {
            'context': ctx_tensor,
            'target': target_4d,
            'pre_pose': torch.from_numpy(pre_pose.astype(np.float32)),
            'edge_idx': torch.tensor([edge_idx], dtype=torch.long),
            'depth_idx': torch.tensor([depth_idx], dtype=torch.long),
            'obj_size': torch.from_numpy(obj_size.astype(np.float32)),
        }


class SE2CroppedHDF5Dataset(Dataset):
    """HDF5-backed equivalent of SE2CroppedDataset."""

    CONTEXT_KEYS = SE2CroppedDataset.CONTEXT_KEYS

    def __init__(self, h5_path: str, indices: List[int], context_size: int = 64):
        if not HAS_H5PY:
            raise ImportError("h5py required for HDF5 dataset")
        self.h5_path = h5_path
        self.indices = indices
        self.context_size = context_size
        self._h5 = None

        self.context_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((context_size, context_size)),
            transforms.Lambda(lambda x: x * 2 - 1),
        ])

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        h5 = self._open()
        real_idx = self.indices[idx]

        channels = []
        for k in self.CONTEXT_KEYS:
            if k in h5:
                arr = h5[k][real_idx]
            else:
                arr = np.zeros((224, 224), dtype=np.float32)
            channels.append(arr)

        se2_target = h5['se2_target_a1'][real_idx]
        pre_pose = h5['pre_pose_a1'][real_idx]
        edge_idx = int(h5['edge_idx_a1'][real_idx][0]) if 'edge_idx_a1' in h5 else -1
        depth_idx = int(h5['depth_idx_a1'][real_idx][0]) if 'depth_idx_a1' in h5 else -1
        obj_size = h5['target_object_size'][real_idx] if 'target_object_size' in h5 else np.zeros(3, dtype=np.float32)

        ctx_tensor = torch.stack([self.context_transform(c)[0] for c in channels], dim=0)
        target_4d = torch.from_numpy(_encode_target(se2_target))

        return {
            'context': ctx_tensor,
            'target': target_4d,
            'pre_pose': torch.from_numpy(pre_pose.astype(np.float32)),
            'edge_idx': torch.tensor([edge_idx], dtype=torch.long),
            'depth_idx': torch.tensor([depth_idx], dtype=torch.long),
            'obj_size': torch.from_numpy(obj_size.astype(np.float32)),
        }

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


class SE2CroppedDataModule(pl.LightningDataModule):
    """LightningDataModule for SE(2) delta diffusion training.

    Auto-detects whether to use an .h5 file (recommended) or scan a dir of NPZs.
    Same root-dir conventions as MaskDiffusionCroppedDataModule.
    """

    def __init__(
        self,
        data_dir: Union[str, List[str]],
        context_size: int = 64,
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_split: float = 0.95,
        use_h5: bool = True,
        h5_path: Optional[str] = None,
        env_family_filter: Optional[str] = None,
    ):
        """
        Args:
            env_family_filter: if set, keep only samples whose xml_file
                contains this substring (e.g. "feb_car" excludes aug9_car).
                None (default) → all families.
        """
        super().__init__()
        self.data_dir = data_dir
        self.context_size = context_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_split = train_split
        self.use_h5 = use_h5
        self.h5_path = h5_path
        self.env_family_filter = env_family_filter
        self.train_dataset = None
        self.val_dataset = None

    def _roots(self) -> List[Path]:
        roots = self.data_dir if isinstance(self.data_dir, (list, tuple)) else [self.data_dir]
        return [Path(r).expanduser() for r in roots]

    def _find_h5(self) -> Optional[str]:
        for root in self._roots():
            for candidate in (root.parent / f"{root.name}.h5", root / "data.h5"):
                if candidate.exists():
                    return str(candidate)
            for h5 in root.glob("*.h5"):
                return str(h5)
        return None

    def _collect_npz(self) -> List[str]:
        out = []
        for root in tqdm(self._roots(), desc="Scanning data roots"):
            if not root.exists():
                continue
            if root.is_file() and str(root).endswith('.npz'):
                out.append(str(root))
                continue
            out.extend(glob.glob(f"{root}/**/*.npz", recursive=True))
        return sorted(set(out))

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return

        h5_path = self.h5_path if self.h5_path else (self._find_h5() if self.use_h5 else None)
        if h5_path and HAS_H5PY:
            print(f"[se2 setup] Using HDF5: {h5_path}")
            with h5py.File(h5_path, 'r') as h5:
                n_samples = h5.attrs.get('n_samples', len(h5[list(h5.keys())[0]]))
                # Apply env_family_filter via xml_file substring match.
                if self.env_family_filter:
                    print(f"[se2 setup] Filtering by xml_file substring: '{self.env_family_filter}'")
                    xml_col = h5['xml_file'][:]
                    keep = []
                    for i in range(n_samples):
                        v = xml_col[i][0] if xml_col[i].shape else xml_col[i]
                        if isinstance(v, bytes):
                            v = v.decode('utf-8', errors='ignore')
                        if self.env_family_filter in str(v):
                            keep.append(i)
                    print(f"[se2 setup] Kept {len(keep)}/{n_samples} samples after filter")
                    indices = keep
                else:
                    indices = list(range(n_samples))
            print(f"[se2 setup] n_samples (post-filter): {len(indices)}")
            rng = random.Random(0)
            rng.shuffle(indices)
            split = max(1, min(int(len(indices) * self.train_split), len(indices) - 1))
            train_idx, val_idx = indices[:split], indices[split:]
            print(f"[se2 setup] Train: {len(train_idx)}  Val: {len(val_idx)}")
            self.train_dataset = SE2CroppedHDF5Dataset(h5_path, train_idx, context_size=self.context_size)
            self.val_dataset = SE2CroppedHDF5Dataset(h5_path, val_idx, context_size=self.context_size)
        else:
            print("[se2 setup] Using NPZ files")
            files = self._collect_npz()
            if not files:
                raise RuntimeError(f"No NPZ files found under {self.data_dir}")
            if self.env_family_filter:
                print(f"[se2 setup] Filtering NPZs by xml_file substring: '{self.env_family_filter}'")
                kept = []
                for f in files:
                    with np.load(f) as d:
                        xml = str(d['xml_file'][0])
                    if self.env_family_filter in xml:
                        kept.append(f)
                print(f"[se2 setup] Kept {len(kept)}/{len(files)} NPZs after filter")
                files = kept
            rng = random.Random(0)
            rng.shuffle(files)
            split = max(1, min(int(len(files) * self.train_split), len(files) - 1))
            self.train_dataset = SE2CroppedDataset(files[:split], context_size=self.context_size, split="train")
            self.val_dataset = SE2CroppedDataset(files[split:], context_size=self.context_size, split="val")
            print(f"[se2 setup] Train: {len(self.train_dataset)}  Val: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
        )
