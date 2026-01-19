from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def _resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Resize mask to (size, size, 1) using area interpolation."""
    if mask.ndim == 2:
        mask = mask[:, :, None]
    if mask.shape[0] == size and mask.shape[1] == size:
        return mask.astype(np.float32)
    if not HAS_CV2:
        raise RuntimeError("cv2 is required for resizing cropped masks.")
    resized = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)
    if resized.ndim == 2:
        resized = resized[:, :, None]
    return resized.astype(np.float32)


def _to_tensor(mask: np.ndarray) -> torch.Tensor:
    """Convert HWC mask in [0,1] to CHW tensor in [-1,1]."""
    if mask.ndim == 2:
        mask = mask[:, :, None]
    tensor = torch.from_numpy(mask).permute(2, 0, 1).float()
    return tensor * 2.0 - 1.0


class MaskDiffusionCroppedHDF5Dataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        indices: List[int],
        context_size: int,
        crop_size: int,
        use_local: bool = True,
    ):
        if not HAS_H5PY:
            raise ImportError("h5py is required for HDF5 datasets.")
        self.h5_path = h5_path
        self.indices = indices
        self.context_size = context_size
        self.crop_size = crop_size
        self.use_local = use_local
        self._h5_file = None
        self._pid = None

    def _get_h5_file(self):
        import os
        current_pid = os.getpid()
        if self._h5_file is None or self._pid != current_pid:
            if self._h5_file is not None:
                self._h5_file.close()
            self._h5_file = h5py.File(self.h5_path, "r")
            self._pid = current_pid
        return self._h5_file

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        h5f = self._get_h5_file()
        real_idx = self.indices[idx]

        ret = {}
        if self.use_local:
            ret["static"] = h5f["local_static"][real_idx]
            ret["movable"] = h5f["local_movable"][real_idx]
            ret["target_object"] = h5f["local_target_object"][real_idx]
            ret["robot_region"] = h5f.get("local_robot_region", None)
            ret["goal_sample_region"] = h5f.get("local_goal_sample_region", None)
            ret["target_goal_mask"] = h5f["local_target_goal"][real_idx]
            if ret["robot_region"] is not None:
                ret["robot_region"] = ret["robot_region"][real_idx]
            else:
                ret["robot_region"] = np.zeros_like(ret["static"])
            if ret["goal_sample_region"] is not None:
                ret["goal_sample_region"] = ret["goal_sample_region"][real_idx]
            else:
                ret["goal_sample_region"] = np.zeros_like(ret["static"])
        else:
            ret["robot"] = h5f["robot_image"][real_idx]
            ret["goal"] = h5f["goal_image"][real_idx]
            ret["movable"] = h5f["movable_objects_image"][real_idx]
            ret["static"] = h5f["static_objects_image"][real_idx]
            ret["target_object"] = h5f["target_object"][real_idx]
            ret["target_goal_mask"] = h5f["target_goal"][real_idx]

        # Resize context channels and target separately
        context_keys = [
            "static",
            "movable",
            "target_object",
            "robot_region",
            "goal_sample_region",
        ]
        for key in context_keys:
            if key in ret:
                ret[key] = _resize_mask(ret[key], self.context_size)

        ret["target_goal_mask"] = _resize_mask(ret["target_goal_mask"], self.crop_size)

        # Convert to tensors in [-1,1]
        for key, value in list(ret.items()):
            if isinstance(value, np.ndarray):
                ret[key] = _to_tensor(value)

        return ret

    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()


@dataclass
class _DatasetSplit:
    train_idx: List[int]
    val_idx: List[int]
    test_idx: List[int]


def _split_indices(n: int, train_split: float, val_split: float, test_split: float) -> _DatasetSplit:
    indices = np.arange(n)
    np.random.shuffle(indices)
    train_end = int(train_split * n)
    val_end = train_end + int(val_split * n)
    return _DatasetSplit(
        train_idx=indices[:train_end].tolist(),
        val_idx=indices[train_end:val_end].tolist(),
        test_idx=indices[val_end:].tolist(),
    )


class MaskDiffusionCroppedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Optional[str] = None,
        h5_file: Optional[str] = None,
        context_size: int = 64,
        crop_size: int = 32,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        use_local: bool = True,
        train_split: float = 0.9,
        val_split: float = 0.1,
        test_split: float = 0.0,
        use_h5: bool = True,
        weighted_sampling: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.h5_file = h5_file
        self.context_size = context_size
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_local = use_local
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.use_h5 = use_h5
        self.weighted_sampling = weighted_sampling

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._train_sampler = None

    def _resolve_h5_path(self) -> str:
        if self.h5_file:
            return self.h5_file
        if self.data_dir is None:
            raise ValueError("data_dir or h5_file must be provided.")
        data_path = Path(self.data_dir)
        if data_path.is_file():
            return str(data_path)
        h5_candidates = sorted(data_path.glob("*.h5"))
        if not h5_candidates:
            h5_candidates = sorted(data_path.glob("**/*.h5"))
        if not h5_candidates:
            raise FileNotFoundError(f"No .h5 files found under {data_path}")
        return str(h5_candidates[0])

    def setup(self, stage: Optional[str] = None):
        if not self.use_h5:
            raise NotImplementedError("Cropped data module currently supports HDF5 only.")

        h5_path = self._resolve_h5_path()
        with h5py.File(h5_path, "r") as h5f:
            dataset_len = h5f["local_target_goal"].shape[0] if self.use_local else h5f["target_goal"].shape[0]

        splits = _split_indices(dataset_len, self.train_split, self.val_split, self.test_split)

        self.train_dataset = MaskDiffusionCroppedHDF5Dataset(
            h5_path=h5_path,
            indices=splits.train_idx,
            context_size=self.context_size,
            crop_size=self.crop_size,
            use_local=self.use_local,
        )
        self.val_dataset = MaskDiffusionCroppedHDF5Dataset(
            h5_path=h5_path,
            indices=splits.val_idx,
            context_size=self.context_size,
            crop_size=self.crop_size,
            use_local=self.use_local,
        )
        self.test_dataset = MaskDiffusionCroppedHDF5Dataset(
            h5_path=h5_path,
            indices=splits.test_idx,
            context_size=self.context_size,
            crop_size=self.crop_size,
            use_local=self.use_local,
        )

        if self.weighted_sampling == "inverse_solutions":
            self._train_sampler = self._build_inverse_solution_sampler(h5_path, splits.train_idx)

    def _build_inverse_solution_sampler(self, h5_path: str, indices: List[int]) -> Optional[WeightedRandomSampler]:
        candidate_keys = [
            "solutions_per_env",
            "solutions_count",
            "solution_count",
            "num_solutions",
        ]
        with h5py.File(h5_path, "r") as h5f:
            for key in candidate_keys:
                if key in h5f:
                    counts = h5f[key][:]
                    counts = counts.astype(np.float32)
                    weights = 1.0 / (counts + 1e-6)
                    weights = weights[indices]
                    return WeightedRandomSampler(weights, num_samples=len(indices), replacement=True)
        return None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self._train_sampler,
            shuffle=self._train_sampler is None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
