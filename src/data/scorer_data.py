"""Datamodule for the 1-push SCORER (HACMan-critic) — reads the joined scorer H5
(scripts/sandbox/build_scorer_dataset.py): per-episode car masks + 60x5 f_grid + reachable mask.

Two things this gets right that the legacy classifier_data did not:
  1. Reads a single H5 (ctx, f_grid, r_mask, xml, ratio), not thousands of NPZs.
  2. Splits train/val BY ROOM (xml), never by row — a scene with several pushed-object episodes
     never straddles the split (docs/pipeline/multi_episode_rooms.md). Mirrors the se2 fix.

Returns the exact dict the existing ClassifierModule expects:
  context (5,H,W), f_labels (60,nd), r_mask (60,nd), cp_reachable (60,nd), ratio.
"""
import random
from typing import List, Optional

import h5py
import numpy as np
import torch
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader


class ScorerH5Dataset(Dataset):
    def __init__(self, h5_path: str, indices: List[int],
                 sample_k: int = 0, unsampled_negative: bool = False, sample_seed: int = 0,
                 budget_h: bool = False):
        """sample_k / unsampled_negative: the H5 sampling ablation (policy_framework journal).

        Simulates NON-exhaustive collection: per row, only `sample_k` of the reachable cells were
        "tried" (deterministic per (sample_seed,row)); the rest are UNKNOWN, not negative.
          sample_k=0 (default)            -> exhaustive (current behavior); loss_mask = r_mask.
          sample_k>0, masked              -> loss_mask = the k sampled cells (labels untouched).
          sample_k>0 + unsampled_negative -> the PU-bug baseline: unsampled cells stay IN the loss
                                             (loss_mask = r_mask) with label forced to 0 (false negs).
        Train-time only (the datamodule never passes sample_k to the val split). Such runs MUST set
        bce_reachable_only=true so the BCE respects loss_mask (the all-600 BCE would leak labels).

        budget_h (horizon-Q): emit batch['H'] = the row's remaining push budget — from the H5's 'H'
        dataset when present (mixed-H training sets), else 1 (pure H=1 sets, where the gamma target
        equals f_grid). Default off = batch dict unchanged."""
        self.h5_path = h5_path
        self.indices = indices
        self.sample_k = int(sample_k)
        self.unsampled_negative = bool(unsampled_negative)
        self.sample_seed = int(sample_seed)
        self.budget_h = bool(budget_h)
        self._h5 = None  # opened lazily per worker (h5py is not fork-safe)

    def __len__(self):
        return len(self.indices)

    def _f(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, k):
        i = self.indices[k]
        f = self._f()
        ctx = torch.from_numpy(f["ctx"][i].astype(np.float32))           # (5, H, W)
        f_grid = f["f_grid"][i].astype(np.float32)                        # (60, nd) 1=valid 0=fail-or-unreach
        r_mask = f["r_mask"][i].astype(np.float32)                        # (60, nd) 1=reachable
        # contact-point-level reachability: if ANY depth of an edge is reachable, all depths are
        # candidates at inference (robot can reach that contact point). Used by the realistic eval mask.
        cp = np.zeros_like(r_mask)
        cp[(r_mask.sum(axis=1) > 0)] = 1.0
        loss_mask = r_mask
        if self.sample_k > 0:
            # deterministic per (sample_seed, row) so every epoch sees the SAME sampled subset
            rng = np.random.default_rng(self.sample_seed * 1_000_003 + i)
            reach = np.argwhere(r_mask > 0)
            kk = min(self.sample_k, len(reach))
            smask = np.zeros_like(r_mask)
            if kk > 0:
                pick = reach[rng.choice(len(reach), size=kk, replace=False)]
                smask[pick[:, 0], pick[:, 1]] = 1.0
            if self.unsampled_negative:
                f_grid = f_grid * smask     # unsampled positives become 0 = FALSE negatives (the bug)
                loss_mask = r_mask          # ...and stay in the loss
            else:
                loss_mask = smask           # masked: loss only on what was actually tried
        out = {
            "context": ctx,
            "f_labels": torch.from_numpy(f_grid),
            "r_mask": torch.from_numpy(r_mask),
            "loss_mask": torch.from_numpy(loss_mask),
            "cp_reachable": torch.from_numpy(cp),
            "ratio": float(f["ratio"][i]),
        }
        if self.budget_h:       # horizon-Q: remaining push budget for this row (H-conditioned forward)
            h_val = int(f["H"][i]) if "H" in f else 1
            out["H"] = torch.tensor(h_val, dtype=torch.long)
        if "contact_px" in f:   # (60,2) pixel coords of each edge's contact point (for per-edge models)
            out["contact_px"] = torch.from_numpy(f["contact_px"][i].astype(np.float32))
        if "context_zoom" in f:  # dual-crop: tight object crop + its contact pixels (for use_zoom models)
            out["context_zoom"] = torch.from_numpy(f["context_zoom"][i].astype(np.float32))
            out["contact_px_zoom"] = torch.from_numpy(f["contact_px_zoom"][i].astype(np.float32))
        return out


class ScorerDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 64, num_workers: int = 4,
                 image_size: int = 64, train_split: float = 0.9, pin_memory: bool = True,
                 sample_k: int = 0, unsampled_negative: bool = False, sample_seed: int = 0,
                 budget_h: bool = False, **_):
        super().__init__()
        self.h5_path = data_dir if data_dir.endswith(".h5") else f"{data_dir}/data.h5"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.pin_memory = pin_memory
        # H5 sampling ablation — applied to the TRAIN split only (val stays exhaustive so
        # val_loss/val metrics are comparable across sampling conditions).
        self.sample_k = sample_k
        self.unsampled_negative = unsampled_negative
        self.sample_seed = sample_seed
        self.budget_h = budget_h   # horizon-Q: emit batch['H'] (train AND val — the model is H-conditioned)
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        with h5py.File(self.h5_path, "r") as h5:
            n = int(h5.attrs.get("n_samples", h5["f_grid"].shape[0]))
            xml = [x.decode() if isinstance(x, bytes) else str(x) for x in h5["xml"][:]]
        # group rows by ROOM, shuffle rooms, fill train to the target fraction of SAMPLES
        groups = {}
        for i in range(n):
            groups.setdefault(xml[i], []).append(i)
        keys = sorted(groups)
        random.Random(0).shuffle(keys)
        target = int(n * self.train_split)
        train_idx, val_idx, cum = [], [], 0
        for k in keys:
            if cum < target:
                train_idx += groups[k]; cum += len(groups[k])
            else:
                val_idx += groups[k]
        print(f"[scorer setup] n={n} rooms={len(groups)} train={len(train_idx)} val={len(val_idx)} "
              f"(room-grouped, 0 scenes straddle)", flush=True)
        if self.sample_k > 0:
            print(f"[scorer setup] H5-SAMPLING ablation: sample_k={self.sample_k} "
                  f"unsampled_negative={self.unsampled_negative} sample_seed={self.sample_seed} "
                  f"(train split only)", flush=True)
        self.train_dataset = ScorerH5Dataset(self.h5_path, train_idx,
                                             sample_k=self.sample_k,
                                             unsampled_negative=self.unsampled_negative,
                                             sample_seed=self.sample_seed,
                                             budget_h=self.budget_h)
        self.val_dataset = ScorerH5Dataset(self.h5_path, val_idx, budget_h=self.budget_h)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)
