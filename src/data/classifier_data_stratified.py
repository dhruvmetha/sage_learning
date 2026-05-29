"""Difficulty-stratified variant of ClassifierDataModule.

Forces each training batch to be ~50% hard (very_hard + hard) and 50% easy
(medium + easy + very_easy). Uses WeightedRandomSampler on the train set.

Validation set uses default random sampling (we want unbiased val metrics).

Drop-in replacement for ClassifierDataModule. The config can swap them via
_target_.
"""

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from pathlib import Path
from typing import List, Optional, Union
import lightning.pytorch as pl
import numpy as np
import random
import glob

from src.data.classifier_data import ClassifierDataset


def _difficulty_bucket(ratio: float) -> str:
    if ratio < 0.05:  return "very_hard"
    if ratio < 0.15:  return "hard"
    if ratio < 0.40:  return "medium"
    if ratio < 0.70:  return "easy"
    return "very_easy"


def _scan_difficulties(npz_files: List[str]) -> List[str]:
    """Read f_ratio from each NPZ and bucket it. Cached on the first NPZ dir."""
    buckets = []
    for p in npz_files:
        with np.load(p) as d:
            ratio = float(d['f_ratio'][0])
        buckets.append(_difficulty_bucket(ratio))
    return buckets


class ClassifierDataModuleStratified(pl.LightningDataModule):
    """Same as ClassifierDataModule but training uses WeightedRandomSampler
    targeting 50% hard (very_hard+hard) and 50% easy (medium+easy+very_easy).
    """

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
        hard_share: float = 0.5,
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
        self.hard_share = hard_share

        self.train_dataset = None
        self.val_dataset = None
        self._train_sampler = None

    def _collect_files(self) -> List[str]:
        roots = self.data_dir if isinstance(self.data_dir, (list, tuple)) else [self.data_dir]
        files = []
        for root in roots:
            files.extend(glob.glob(f"{root}/**/*.npz", recursive=True))
        files = sorted(set(files))
        # Cache file (compat with the original DataModule cache)
        return files

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Lambda(lambda x: x * 2 - 1),
        ])

        all_files = self._collect_files()
        if not all_files:
            raise RuntimeError(f"No NPZ files in {self.data_dir}")
        print(f"Stratified DataModule: total NPZ files = {len(all_files)}", flush=True)

        rng = random.Random(0)
        rng.shuffle(all_files)
        split_idx = max(1, min(int(len(all_files) * self.train_split), len(all_files) - 1))
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]
        print(f"Train: {len(train_files)}, Val: {len(val_files)}", flush=True)

        # Bucket every training file. Cache to avoid re-scanning on resume.
        cache_path = Path(roots := (self.data_dir if isinstance(self.data_dir, str)
                                    else self.data_dir[0])) / ".classifier_difficulty_cache.txt"
        bucket_map = {}
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    for ln in f:
                        ln = ln.strip()
                        if not ln: continue
                        parts = ln.split("\t")
                        if len(parts) == 2:
                            bucket_map[parts[0]] = parts[1]
                print(f"Loaded difficulty cache: {len(bucket_map)} entries", flush=True)
            except Exception as e:
                print(f"Difficulty cache invalid ({e}); rebuilding", flush=True)
                bucket_map = {}

        train_buckets = []
        new_entries = []
        for i, p in enumerate(train_files):
            if p in bucket_map:
                b = bucket_map[p]
            else:
                with np.load(p) as d:
                    ratio = float(d['f_ratio'][0])
                b = _difficulty_bucket(ratio)
                new_entries.append((p, b))
            train_buckets.append(b)
            if (i + 1) % 1000 == 0:
                print(f"  bucketed {i+1}/{len(train_files)}", flush=True)
        if new_entries:
            with open(cache_path, "a") as f:
                for p, b in new_entries:
                    f.write(f"{p}\t{b}\n")
            print(f"Added {len(new_entries)} new entries to difficulty cache", flush=True)

        # Compute per-instance sampling weights
        HARD = {"very_hard", "hard"}
        n_hard = sum(1 for b in train_buckets if b in HARD)
        n_easy = len(train_buckets) - n_hard
        from collections import Counter
        bc = Counter(train_buckets)
        print(f"Training difficulty distribution:", flush=True)
        for b in ("very_hard", "hard", "medium", "easy", "very_easy"):
            print(f"  {b:>11s}: {bc[b]:>5d}", flush=True)
        print(f"  hard pool (vh+h):  {n_hard}", flush=True)
        print(f"  easy pool (m+e+ve):{n_easy}", flush=True)

        hard_w = self.hard_share / max(1, n_hard)
        easy_w = (1.0 - self.hard_share) / max(1, n_easy)
        weights = [(hard_w if b in HARD else easy_w) for b in train_buckets]
        print(f"Per-sample weights: hard={hard_w:.6g}, easy={easy_w:.6g}", flush=True)
        print(f"Hard oversampling factor: {hard_w / easy_w:.1f}x", flush=True)

        num_samples_per_epoch = len(train_files)  # keep epoch length similar to default
        self._train_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples_per_epoch,
            replacement=True,
        )

        if stage == "fit" or stage is None:
            self.train_dataset = ClassifierDataset(
                train_files, transform=transform,
                use_local=self.use_local, use_coord_grid=self.use_coord_grid,
            )
            self.val_dataset = ClassifierDataset(
                val_files, transform=transform,
                use_local=self.use_local, use_coord_grid=self.use_coord_grid,
            )
        if stage == "test" or stage is None:
            self.test_dataset = ClassifierDataset(
                val_files, transform=transform,
                use_local=self.use_local, use_coord_grid=self.use_coord_grid,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self._train_sampler,        # ← key change vs default
            shuffle=False,                      # sampler handles ordering
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=self.pin_memory, persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
