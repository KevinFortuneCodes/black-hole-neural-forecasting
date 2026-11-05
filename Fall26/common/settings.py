# Copyright (c) 2026 Kevin Fortune. All rights reserved, except for portions adapted from
# code by Sachin Venkatesh (the TRON model), which remain his. See LICENSE.
"""Physics settings shared by the normalizer and the physics loss.

Copied from TRON's CONFIG in uno_simple_ds_v3.py (sachin branch, fall26/, commit
ca0288555c99, 2026-10-05). Keep these identical to TRON's so both models are trained
and scored against the same physics, except:
  - b_mag_threshold is 7.0 (TRON: 5.0, the default in its loss). The saturation guard penalizes
    predictions with |arcsinh(B)| above it, i.e. |B| > 74 at 5.0, but the true rc300 data exceeds
    that in 2-4% of cells (r = 1.1-1.7, at the horizon; max |B| ~ 270). 7.0 allows |B| up to 548.
    check_normalizer_constants warns if a dataset's B exceeds the threshold.
  - floor_vec is new (not in TRON): the smallest vector magnitude MagDirNormalizer takes a log of.
"""

SOURCE = "sachin:fall26/uno_simple_ds_v3.py @ ca0288555c99"

PHYSICS = {
    # Physics constraints
    "max_gamma": 50.0,  # matches H-AMR's own GAMMAMAX_RAD / GAMMAMAX_NU
    "floor_rho": 1e-20,
    "ceiling_rho": 1e5,
    "floor_ug": 1e-22,
    "ceiling_ug": 1e5,
    "b_mag_threshold": 7.0,   # saturation guard on |arcsinh(B)| (see module docstring; TRON: 5.0)
    "floor_vec": 1e-10,       # magnitude floor for the "magdir" normalization; rc300's smallest |ũ|, |B| ~1e-7

    # Geometry metadata overrides (None = read from geometry.npz)
    "eos_gamma": None,
    "dt_dump": None,

    # Fixed loss weights
    "loss_weight_data": 1.0,
    "loss_weight_grad_smooth": 0.10,
    "loss_weight_grad_turb": 0.03,
    "loss_weight_spec": 0.01,
    "loss_weight_divb": 0.10,
    "loss_weight_mass": 0.03,
    "loss_weight_guards": 0.10,

    # Curriculum: reconstruction only, then physics terms ramped in, stable from epoch 150.
    # vit/train.py replaces the epoch counts with its own (vit/config.py).
    "recon_only_epochs": 50,
    "curriculum_ramp_epochs": 100,
    "curriculum_stable_epoch": 150,

    "detect_anomaly": False,
}
