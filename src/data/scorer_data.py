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
    def __init__(self, h5_path, indices: List,
                 sample_k: int = 0, unsampled_negative: bool = False, sample_seed: int = 0,
                 budget_h: bool = False, unreachable_k: int = 0, emit_reach_flag: bool = False):
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
        equals f_grid). Default off = batch dict unchanged.

        unreachable_k (M2c, [USER] hypothesis: reachability supervision sharpens the encoder): ADD k
        uniformly-sampled UNREACHABLE cells (r_mask==0) to the loss mask with target 0 — executability is
        a KNOWN fact (f_grid already stores 0 there). Applied AFTER sample_k, so the reachable-side
        supervision is byte-identical to the base recipe (e.g. B30): mask = S30 ∪ S20. Deterministic per
        (sample_seed,row) like B30. ONLY valid when tried == reachable (exhaustive-over-reachable rows);
        under sampled rows the complement holds reachable-but-untried cells (zeroing = the C15 bug).
        Distinct from unsampled_negative (which falsely zeroes sampled-REACHABLE cells — the bug arm).

        emit_reach_flag (M2d): emit batch['reach_edges'] (60,) long — 1 iff ANY depth of the edge is in
        r_mask (contact point reachable) — for the reachability-input-flag network variant."""
        # MIXED-H training (Q-full): h5_path may be a LIST of H5s; indices are then (file_idx, row)
        # pairs. Single-path callers are unchanged (str -> [str], int indices -> file 0).
        self.h5_paths = [h5_path] if isinstance(h5_path, str) else list(h5_path)
        self.indices = [(0, i) if isinstance(i, (int, np.integer)) else tuple(i) for i in indices]
        self.sample_k = int(sample_k)
        self.unsampled_negative = bool(unsampled_negative)
        self.sample_seed = int(sample_seed)
        self.budget_h = bool(budget_h)
        self.unreachable_k = int(unreachable_k)
        self.emit_reach_flag = bool(emit_reach_flag)
        self._h5 = None  # opened lazily per worker (h5py is not fork-safe)

    def __len__(self):
        return len(self.indices)

    def _f(self, fi=0):
        if self._h5 is None:
            self._h5 = [None] * len(self.h5_paths)
        if self._h5[fi] is None:
            self._h5[fi] = h5py.File(self.h5_paths[fi], "r")
        return self._h5[fi]

    def __getitem__(self, k):
        fi, i = self.indices[k]
        f = self._f(fi)
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
        if self.unreachable_k > 0:
            # M2c: union in k sampled UNREACHABLE cells (target 0) — after sample_k, so S30 stays identical
            rng_u = np.random.default_rng(self.sample_seed * 2_000_003 + i)
            unreach = np.argwhere(r_mask <= 0)
            ku = min(self.unreachable_k, len(unreach))
            if ku > 0:
                pick = unreach[rng_u.choice(len(unreach), size=ku, replace=False)]
                loss_mask = loss_mask.copy()
                loss_mask[pick[:, 0], pick[:, 1]] = 1.0
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
        if self.emit_reach_flag:  # M2d: per-edge contact-point reachability bit
            out["reach_edges"] = torch.from_numpy((r_mask.sum(axis=1) > 0).astype(np.int64))
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
                 budget_h: bool = False, unreachable_k: int = 0, emit_reach_flag: bool = False, **_):
        super().__init__()
        # MIXED-H (Q-full): data_dir may be ';'-separated H5 paths/dirs — rooms are grouped ACROSS
        # files (same scene in two files lands on the same split side; xml paths are realpath-normalized
        # because H=1 and H=2 renders reference the same scenes through different shard symlinks).
        self.h5_paths = [(d if d.endswith(".h5") else f"{d}/data.h5") for d in str(data_dir).split(";")]
        self.h5_path = self.h5_paths[0]   # legacy single-file attr
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
        self.unreachable_k = unreachable_k       # M2c: TRAIN-only loss-mask change (val stays full-R, comparable)
        self.emit_reach_flag = emit_reach_flag   # M2d: input feature — train AND val (the model consumes it)
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        import os as _os
        groups = {}
        n = 0
        _rp = {}
        for fi, path in enumerate(self.h5_paths):
            with h5py.File(path, "r") as h5:
                nf = int(h5.attrs.get("n_samples", h5["f_grid"].shape[0]))
                xml = [x.decode() if isinstance(x, bytes) else str(x) for x in h5["xml"][:]]
            for i in range(nf):
                k = xml[i]
                if k not in _rp:
                    _rp[k] = _os.path.realpath(k)
                groups.setdefault(_rp[k], []).append((fi, i))
            n += nf
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
        self.train_dataset = ScorerH5Dataset(self.h5_paths, train_idx,
                                             sample_k=self.sample_k,
                                             unsampled_negative=self.unsampled_negative,
                                             sample_seed=self.sample_seed,
                                             budget_h=self.budget_h,
                                             unreachable_k=self.unreachable_k,
                                             emit_reach_flag=self.emit_reach_flag)
        self.val_dataset = ScorerH5Dataset(self.h5_paths, val_idx, budget_h=self.budget_h,
                                           emit_reach_flag=self.emit_reach_flag)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)
