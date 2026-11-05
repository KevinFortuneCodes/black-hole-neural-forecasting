# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.
"""Experiments registry, paths and output naming for the ViT baseline.

Every training run is one experiment, a named entry in EXPERIMENTS that holds all of its
hyperparameters, and one seed. Outputs are named after both:

    Fall26/outputs/<experiment>/seed<N>/                         training run
        config.json, train.log, metrics.csv, latest.pth, best.pth
    Fall26/outputs/<experiment>/seed<N>/eval/<checkpoint>_<split>/  evaluation of one checkpoint
        e.g. eval/best_blind/ = best.pth scored on the blind test set

Never edit an experiment once it has runs; add a new entry instead. Training saves each
run's full configuration to config.json and refuses to resume a run whose experiment has
changed since.

Design decisions are recorded in the design doc ("Transformer Baseline for GRMHD Forecasting").
Physics settings (floors, loss weights) live in common/settings.py.
"""
import os
from pathlib import Path

FALL26_DIR = Path(__file__).resolve().parents[1]
# The preprocessed datasets live under GRMHD_DATA_ROOT. Keep them on fast (NVMe) storage: on Delta
# it loads a batch in ~0.1 s against ~3.4 s from the HDD file system, where some GPU nodes also
# returned I/O errors (2026-10-19). Fall26/docs/rc300_preprocessing.md
DATA_ROOT = os.environ.get("GRMHD_DATA_ROOT", "/path/to/fast_storage")
DATA_5000RUN = f"{DATA_ROOT}/5000run/processed_forecast"
DATA_RC300 = f"{DATA_ROOT}/rc300_reduced"

# Where and how to run; these do not change what is trained. (The data is part of each experiment.)
RUNTIME = {
    # Fall26/outputs is a link to /path/to/work/fall26_outputs on Delta.
    "out_dir": str(FALL26_DIR / "outputs"),
    "num_workers": 4,           # data-loading processes per GPU
    "log_every": 50,            # batches between progress lines
    "max_batches": None,        # limit batches per epoch (smoke tests only)
    "stop_after": None,         # stop this job after N epochs, as if it hit its time limit (tests only)
}

DEFAULT_EXPERIMENT = "vit_baseline_v1"

EXPERIMENTS = {
    "vit_baseline_v1": {
        "description": "ViT baseline: one token per variable and 8x8x4 patch (55,296 tokens), "
                       "12 blocks, d=256, MLP ratio 8, 16.8M parameters. One-step prediction, "
                       "5-epoch warmup then cosine decay over 100 epochs.",
        # Model
        "token_mode": "per_variable",   # one token per (variable, patch); "per_patch": one per patch
        "patch": (8, 8, 4),         # r x theta x phi; halving phi makes equatorial patches near-cubic
        "d_model": 256,
        "depth": 12,
        "n_heads": 8,
        "mlp_ratio": 8,             # attention dominates compute, so a wider MLP adds parameters for ~4% cost
        "use_amp": True,            # bf16 autocast around the transformer blocks only
        "grad_checkpoint": True,    # recompute each block in the backward pass to save memory
        # Normalization (common/normalizer.py); it sets the model's channel count:
        # "arcsinh" = TRON's, 8 channels; "magdir" = log10 magnitude + unit direction per vector, 10 channels
        "normalization": "arcsinh",
        # Data
        "data_dir": DATA_5000RUN,
        "geometry_file": f"{DATA_5000RUN}/geometry.npz",
        "val_frac": 0.1,            # chronological 90/10 split
        "unroll_steps": 1,          # one-step prediction (TRON trains on 2 detached unrolled steps)
        "batch_size": 8,            # per GPU
        # Optimization: fixed budget, warmup then cosine decay
        "epochs": 100,              # about 16,000 optimizer steps at a global batch of 32
        "learning_rate": 5e-4,      # peak
        "warmup_epochs": 5,         # linear warmup from 0, stepped every optimizer step
        "min_lr": 1e-6,             # cosine decays to this by the last step
        "weight_decay": 0.05,       # AdamW; not applied to biases or LayerNorm weights
        "grad_clip": 1.0,
        # Physics loss phase-in: reconstruction terms only, then divB and mass ramped in
        "recon_only_epochs": 10,
        "curriculum_ramp_epochs": 20,
        # Best checkpoint and early stopping on validation reconstruction loss (Fluid + Mag),
        # tracked once the physics terms are fully on. Early stopping is a safety net: under
        # cosine decay validation usually keeps improving until the last epoch.
        "stable_epoch": 30,         # recon_only_epochs + curriculum_ramp_epochs
        "early_stop_patience": 20,
    },
}

EXPERIMENTS["vit_rc300_v1"] = {
    **EXPERIMENTS["vit_baseline_v1"],
    "description": "ViT on the rc300 reduced run: one token per 4x4x4 patch holding all 8 variables "
                   "(8,064 tokens), width 512 with 8 heads of 64, 12 blocks, MLP ratio 8 (~63M parameters), "
                   "vit_baseline_v1's schedule. 112x48x96 grid (r 1.1-1,048), dumps 3000-11671 (test "
                   "11672-12635 in blind/). Physics loss: div B error vs the truth's, unweighted mass shells.",
    "token_mode": "per_patch",  # attention 64x cheaper than per_variable; variables mixed in the embedding
    "d_model": 512,             # = values per token (8 variables x 64 cells): no forced compression at the input
    "grad_checkpoint": False,   # 21 GiB of 40 at batch 8 without it, and 20% faster per step
    "n_heads": 8,               # 64 dimensions per head, the standard
    "patch": (4, 4, 4),         # rc300 cells are near-cubic (0.061, 0.064, 0.065), so equal counts give cubic patches
    "data_dir": DATA_RC300,
    "geometry_file": f"{DATA_RC300}/geometry.npz",
    "val_frac": 0.11116236162361623,   # from splits.json: 7,708 train / 964 val (10% of all dumps each)
}

EXPERIMENTS["vit_rc300_magdir_v1"] = {
    **EXPERIMENTS["vit_rc300_v1"],
    "description": "vit_rc300_v1 with the magdir normalization: ũ and B each as log10 of the magnitude "
                   "and the unit direction (10 channels). Under arcsinh the outer field and velocities "
                   "sit in its linear regime, where the loss barely sees them (vit_rc300_v1's relative "
                   "error of ~10 on B at r ~ 800 after one step). d_model stays 512 (now 640 values "
                   "per token) so the normalization is the only change.",
    "normalization": "magdir",
}

# Settings added after earlier runs, with the value those runs used. Their config.json lacks the key.
ADDED_SETTINGS = {"normalization": "arcsinh"}


def experiment_config(name=DEFAULT_EXPERIMENT):
    """Runtime settings plus all of one experiment's hyperparameters."""
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name!r}; known: {', '.join(EXPERIMENTS)}")
    return {**RUNTIME, **EXPERIMENTS[name], "experiment": name}


def run_dir(experiment, seed, out_dir=None):
    """Fall26/outputs/<experiment>/seed<N>/"""
    return Path(out_dir or RUNTIME["out_dir"]) / experiment / f"seed{seed}"


def eval_dir(experiment, seed, checkpoint="best", split="blind", out_dir=None):
    """Fall26/outputs/<experiment>/seed<N>/eval/<checkpoint>_<split>/, e.g. eval/best_blind/"""
    return run_dir(experiment, seed, out_dir) / "eval" / f"{checkpoint}_{split}"


CONFIG = experiment_config()   # the default experiment, for scripts that don't choose one
