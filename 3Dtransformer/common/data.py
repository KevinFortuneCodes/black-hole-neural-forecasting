# Copyright (c) 2026 Kevin Fortune. All rights reserved, except for portions adapted from
# code by Sachin Venkatesh (the TRON model), which remain his. See LICENSE.
"""Dump listing, train/validation split and the unrolled forecast dataset.

HAMRForecastDataset is copied from uno_simple_ds_v3.py (sachin branch, fall26/,
commit ca0288555c99); its warning now goes through `logging`, it wraps each loaded
array with torch.from_numpy instead of copying it, and it retries a read that fails with
an OSError (Delta's shared filesystem gave a transient I/O error once). list_dumps and
split_train_val reproduce the file ordering and 90/10 split from v3's main().
"""
import glob
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

N_PHYS_CHANNELS = 8
N_CUT_DIAG_CHANNELS = 10
READ_RETRIES = 4          # waits 1, 2, 4, 8 s between attempts


def load_dump(path):
    """np.load with retries: a transient filesystem error should not end a multi-day run."""
    for attempt in range(READ_RETRIES + 1):
        try:
            return np.load(path)
        except OSError as e:
            if attempt == READ_RETRIES:
                raise
            logger.warning(f"reading {path} failed ({e}); retry {attempt + 1}/{READ_RETRIES}")
            time.sleep(2 ** attempt)


def dump_index(path):
    """dump_123.npy -> 123"""
    return int(path.split('_')[-1].split('.')[0])


def list_dumps(data_dir):
    """All dump_*.npy files in data_dir, in numeric order (dump_2 before dump_10)."""
    return sorted(glob.glob(os.path.join(data_dir, "dump_*.npy")), key=dump_index)


def split_train_val(files, val_frac=0.1):
    """Chronological split: the first (1 - val_frac) of the dumps train, the rest validate."""
    n_train = int((1.0 - val_frac) * len(files))
    return files[:n_train], files[n_train:]


class HAMRForecastDataset(Dataset):
    def __init__(self, files, unroll_steps=2):
        self.files = sorted(files, key=dump_index)
        self.unroll_steps = unroll_steps

    def __len__(self):
        return len(self.files) - self.unroll_steps

    def __getitem__(self, i):
        sequence = []
        for step in range(self.unroll_steps + 1):
            a = load_dump(self.files[i + step])
            if a.shape[0] == N_CUT_DIAG_CHANNELS:
                logger.warning(f"dump {self.files[i + step]}: {a.shape[0]} channels "
                               "(preprocess_cut layout: 8 physical + A_theta/A_phi). "
                               f"Keeping the first {N_PHYS_CHANNELS}.")
                a = a[:N_PHYS_CHANNELS]
            elif a.shape[0] != N_PHYS_CHANNELS:
                raise ValueError(
                    f"dump {self.files[i + step]}: expected {N_PHYS_CHANNELS} physical "
                    f"channels [rho, ug, utilde1, utilde2, utilde3, B1, B2, B3] "
                    f"(first in every preprocess layout), got {a.shape[0]}. "
                    "Re-preprocess with preprocess_forecast_ds.py if the layout changed.")
            sequence.append(torch.from_numpy(a).float())   # shares a's memory (no second copy)
        return sequence
