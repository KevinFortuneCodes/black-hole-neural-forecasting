# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.
"""Preprocess the rc300 H-AMR run into Fall26 training data.

Reads H-AMR's raw output directly with numpy (pp is not needed), assembles the
adaptive-mesh blocks onto one uniform grid at the finest refinement level,
optionally coarsens and crops it, and writes what Fall26's training code reads:

  <out_dir>/dump_<N>.npy            train and validation dumps, float32 (8, Nr, Ntheta, Nphi):
                                    rho, ug, utilde1-3, B1-3
  <out_dir>/blind/dump_<N>.npy      test dumps (the last part of the run)
  <out_dir>/geometry.npz            coordinates, metric, dx and run metadata
  <out_dir>/channel_stats.npz       per-channel min/max/mean/std over the dumps processed
  <out_dir>/splits.json             each split's dump range, each dump's time, and the
                                    val_frac to set in the training experiment
  <out_dir>/checks.json             validation results
  <out_dir>/preprocess_config.json  the settings used

Every decision is a setting in PREPROCESS; override any with --set KEY=VALUE (JSON).
The raw format, the checks and the reasons for each setting are in
Fall26/docs/rc300_preprocessing.md.

Test on a few dumps first (login node):
    python Fall26/data/preprocess_rc300.py --set out_dir=$SCRATCH/rc300_test \
        --dumps 3000 3001 12635 --workers 3
Full run: sbatch Fall26/scripts/preprocess_rc300_perlmutter.sbatch
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np

PREPROCESS = {
    # Source: the reduced run (decided)
    "run_dir": "/path/to/ml_data_rc300/reduced",
    "first_dump": 3000,        # skip the early unstable phase (team suggestion); no further settling cut
    "last_dump": None,         # None = last available (12635)

    # Resolution: no coarsening (decided) -> finest grid 224 x 48 x 96 before cropping
    "lowres": [1, 1, 1],       # k > 1 averages k x k x k cells (per direction)
    "fill_coarse": "copy",     # cells of unrefined blocks (r < 6.11): "copy" or "interpolate" (linear)

    # Crop (decided): r up to ~1,050, all theta and phi (doc section 6). None = keep the whole range.
    "r_min": None,             # code units (M); None = inner boundary (r = 1.1)
    "r_max": 1040.0,           # radial cell 112 (edge r = 1,048): 98-99% of the one-step change (doc section 6)
    "theta_deg": None,         # [min, max] in degrees; None = all
    "phi_deg": None,           # [min, max] in degrees; None = all
    "align_cells": [8, 8, 4],  # widen each cropped dimension to a multiple of this (the ViT patch)

    # Splits: chronological over first_dump..last_dump; test is the last part of the run (decided)
    "split_fracs": {"train": 0.8, "val": 0.1, "test": 0.1},

    # Output (decided)
    "out_dir": "$SCRATCH/rc300_reduced",                     # staging on Perlmutter
    "delta_dir": "/path/to/work/rc300_reduced",    # where it was first copied (not used here); now
                                                             # on /work/nvme (doc section 9). Value kept so reruns
                                                             # into the existing output match its saved settings.
}

CHANNEL_NAMES = ["rho", "ug", "utilde1", "utilde2", "utilde3", "B1", "B2", "B3"]
NPR = 9                                    # values per cell: rho, ug, u0, v1-v3, B1-B3
U0 = 2                                     # raw value index of u^0
RAW_CHANNELS = [0, 1, 3, 4, 5, 6, 7, 8]    # raw value index of each output channel

# Bump when the code changes what it writes for the same settings: reruns into an older folder
# are then refused instead of mixing old and new dumps.
FORMAT_VERSION = 2   # 2: coordinate velocities converted to utilde; values at their sample positions

# Block-table fields (pp.py rblock_new)
AMR_LEVEL, AMR_COORD1, AMR_LEVEL1 = 1, 3, 110

# Checks
COORD_TOL = 1e-3          # gdump cell coordinates vs the uniform grid, in units of dx
MAPPING_TOL = 1e-6        # r = exp(x1), theta = pi (x2 + 1) / 2, phi = x3
VEL_TOL = 1e-4            # per-cell relative velocity residual (real data: worst cell ~1e-7)
VEL_BAD_FRACTION = 1e-5   # a convention fits if at most this fraction of cells exceed VEL_TOL


# ---------------------------------------------------------------- settings and splits

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Preprocess the rc300 H-AMR run into Fall26 training data.")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a PREPROCESS setting; VALUE is JSON, e.g. r_max=null or theta_deg=[30,150]")
    p.add_argument("--dumps", type=int, nargs="+",
                   help="process only these dumps (testing); splits still come from the full range")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel processes (default: SLURM_CPUS_PER_TASK, else 1)")
    p.add_argument("--overwrite", action="store_true",
                   help="reprocess dumps that already exist and accept changed settings")
    return p.parse_args(argv)


def build_settings(overrides):
    s = json.loads(json.dumps(PREPROCESS))          # deep copy, JSON types
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or key not in s:
            raise SystemExit(f"--set {item!r}: expected KEY=VALUE with KEY one of {sorted(s)}")
        try:
            s[key] = json.loads(value)
        except json.JSONDecodeError:
            s[key] = value
    for key in ("run_dir", "out_dir"):
        s[key] = os.path.expandvars(os.path.expanduser(s[key]))
        if "$" in s[key]:
            raise SystemExit(f"{key} = {s[key]!r} has an unset environment variable")
    if s["fill_coarse"] not in ("copy", "interpolate"):
        raise SystemExit(f"fill_coarse must be 'copy' or 'interpolate', got {s['fill_coarse']!r}")
    fr = s["split_fracs"]
    if set(fr) != {"train", "val", "test"} or abs(sum(fr.values()) - 1.0) > 1e-9:
        raise SystemExit(f"split_fracs must have train, val and test summing to 1, got {fr}")
    if len(s["lowres"]) != 3 or len(s["align_cells"]) != 3:
        raise SystemExit("lowres and align_cells need 3 values (r, theta, phi)")
    return s


def last_available_dump(run_dir):
    nums = [int(n[5:]) for n in os.listdir(run_dir) if n.startswith("dumps") and n[5:].isdigit()]
    if not nums:
        raise SystemExit(f"no dumps<N> folders in {run_dir}")
    return max(nums)


def plan_splits(s):
    """Chronological train/val/test over first_dump..last_dump."""
    first = s["first_dump"]
    last = s["last_dump"] if s["last_dump"] is not None else last_available_dump(s["run_dir"])
    dumps = list(range(first, last + 1))
    missing = [d for d in dumps if not os.path.isfile(os.path.join(s["run_dir"], f"dumps{d}", "parameters"))]
    if missing:
        raise SystemExit(f"{len(missing)} dumps in {first}-{last} have no parameters file, e.g. {missing[:10]}")
    n = len(dumps)
    n_test = round(s["split_fracs"]["test"] * n)
    n_val = round(s["split_fracs"]["val"] * n)
    n_train = n - n_val - n_test
    split = {}
    for i, d in enumerate(dumps):
        split[d] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    # Fall26/common/data.py keeps int((1 - val_frac) * n) of the train+val folder for training
    m = n_train + n_val
    val_frac = n_val / m
    if int((1.0 - val_frac) * m) != n_train:
        val_frac = (n_val - 0.5) / m
    ranges = {}
    for name, (a, b) in (("train", (0, n_train)), ("val", (n_train, m)), ("test", (m, n))):
        ranges[name] = {"first": dumps[a] if b > a else None, "last": dumps[b - 1] if b > a else None,
                        "count": b - a}
    return dict(dumps=dumps, split=split, ranges=ranges, val_frac_for_training=val_frac)


def output_path(out_dir, dump, split):
    folder = os.path.join(out_dir, "blind") if split == "test" else out_dir
    return os.path.join(folder, f"dump_{dump}.npy")


def record_settings(out_dir, s, overwrite):
    """Refuse to mix outputs made with different settings in one folder."""
    path = os.path.join(out_dir, "preprocess_config.json")
    record = dict(s, format_version=FORMAT_VERSION)
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old != record and not overwrite:
            changed = sorted(k for k in set(old) | set(record) if old.get(k) != record.get(k))
            raise SystemExit(f"{out_dir} was made with different settings or code ({', '.join(changed)}). "
                             "Use another out_dir, or --overwrite to reprocess everything.")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)


# ---------------------------------------------------------------- raw H-AMR files

PARAMETERS_HEADER = np.dtype([
    ("t", "<f8"), ("n_active", "<i4"), ("n_active_total", "<i4"), ("nstep", "<i4"),
    ("Dtd", "<f8"), ("Dtl", "<f8"), ("Dtr", "<f8"), ("dump_cnt", "<i4"), ("rdump_cnt", "<i4"),
    ("dt", "<f8"), ("failed", "<i4"),
    ("bs", "<i4", 3), ("nmax", "<i4"), ("nb", "<i4", 3),
    ("startx", "<f8", 3), ("dx", "<f8", 3),
    ("tf", "<f8"), ("a", "<f8"), ("gam", "<f8"), ("cour", "<f8"),
    ("Rin", "<f8"), ("Rout", "<f8"), ("R0", "<f8"), ("density_scale", "<f8"),
    ("unused", "<i4", 13), ("flags", "<i4"), ("unused2", "<i4"),
])
assert PARAMETERS_HEADER.itemsize == 66 * 4


def read_parameters(run_dir, dump):
    """dumps<N>/parameters (pp.py rpar_new). Values are raw: dx is wrong for the reduced run (see grid_axes)."""
    with open(os.path.join(run_dir, f"dumps{dump}", "parameters"), "rb") as f:
        raw = f.read()
    h = np.frombuffer(raw, PARAMETERS_HEADER, count=1)[0]
    nat = int(h["n_active_total"])
    if len(raw) < PARAMETERS_HEADER.itemsize + 12 * nat:
        raise ValueError(f"dump {dump}: parameters has no (n_ord, TIMELEVEL, NODE) table; older format not supported")
    table = np.frombuffer(raw, "<i4", count=3 * nat, offset=PARAMETERS_HEADER.itemsize).reshape(nat, 3)
    par = {k: h[k] for k in ("t", "Dtd", "tf", "a", "gam", "Rin", "Rout", "R0", "density_scale")}
    par = {k: float(v) for k, v in par.items()}
    par.update(bs=h["bs"].astype(int), nb=h["nb"].astype(int), startx=h["startx"].astype(float),
               dx=h["dx"].astype(float), flags=int(h["flags"]), n_ord=table[:, 0].copy(),
               node=table[:, 2].copy())
    return par


def read_block_table(run_dir):
    """gdumps/grid (pp.py rblock_new): one row of NV int32 fields per block ID."""
    path = os.path.join(run_dir, "gdumps", "grid")
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        nmax = int(np.fromfile(f, "<i4", 1)[0])
        nv = (size - 1) // nmax // 4
        return np.fromfile(f, "<i4", nv * nmax).reshape((nv, nmax), order="F").T


def read_blocks(run_dir, dump, par, bs):
    """All active blocks of one dump, float32 (n_blocks, bs1, bs2, bs3, 9), sorted by block ID.

    new_dump0, new_dump1, ... hold the blocks of task 0, 1, ... in turn; together they are the
    blocks in parameter-table order n = 0, 1, ..., and block n has ID n_ord[n]. That order
    changes from dump to dump, so sorting by ID is what makes dumps comparable.
    """
    folder = os.path.join(run_dir, f"dumps{dump}")
    names = sorted((n for n in os.listdir(folder) if n.startswith("new_dump") and n[8:].isdigit()),
                   key=lambda n: int(n[8:]))
    if not names and os.path.isfile(os.path.join(folder, "new_dump")):
        names = ["new_dump"]
    node = par["node"]
    if names != ["new_dump"]:
        if np.any(np.diff(node) < 0):
            raise ValueError(f"dump {dump}: parameter table is not grouped by task (NODE); "
                             "file order would not match n_ord")
        counts = np.bincount(node, minlength=len(names))
        if len(counts) != len(names):
            raise ValueError(f"dump {dump}: NODE refers to {len(counts)} tasks but there are {len(names)} files")
    cell = int(np.prod(bs)) * NPR
    parts = []
    for k, name in enumerate(names):
        a = np.fromfile(os.path.join(folder, name), "<f4")
        if a.size % cell:
            raise ValueError(f"dump {dump}: {name} is not a whole number of blocks")
        if names != ["new_dump"] and a.size // cell != counts[k]:
            raise ValueError(f"dump {dump}: {name} holds {a.size // cell} blocks, parameters say {counts[k]}")
        parts.append(a)
    data = np.concatenate(parts)
    nat = len(par["n_ord"])
    if data.size != nat * cell:
        raise ValueError(f"dump {dump}: {data.size // cell} blocks in files, {nat} in parameters")
    return data.reshape(nat, bs[0], bs[1], bs[2], NPR)[np.argsort(par["n_ord"])]


def read_gdumps(run_dir, ids, bs):
    """Per-block geometry from gdumps/gdump<ID> (axisymmetric layout, functions.c kernel_rgdump_new).

    Returns coords (n, bs1, bs2, bs3, 9) [.., x1, x2, x3, r, h, ph at 3..8] and
    metric (n, bs1, bs2, 49) [gcov 0:16, gcon 16:32, gdet 32, dxdxp 33:49].
    """
    b1, b2, b3 = bs
    n_cell, n_axi = 9 * b1 * b2 * b3, 49 * b1 * b2
    coords = np.empty((len(ids), b1, b2, b3, 9))
    metric = np.empty((len(ids), b1, b2, 49))
    for k, bid in enumerate(ids):
        a = np.fromfile(os.path.join(run_dir, "gdumps", f"gdump{bid}"), "<f8")
        if a.size != n_cell + n_axi:
            raise NotImplementedError(f"gdump{bid} has {a.size} values; only the axisymmetric layout "
                                      f"({n_cell + n_axi}) is supported")
        coords[k] = a[:n_cell].reshape(b1, b2, b3, 9)
        metric[k] = a[n_cell:].reshape(b1, b2, 49)
    return coords, metric


# ---------------------------------------------------------------- assembling blocks

class Layout:
    """Where every active block sits on the uniform grid at the finest level."""

    def __init__(self, par, table):
        self.ids = np.sort(par["n_ord"])
        self.bs, self.nb = par["bs"], par["nb"]
        if table.shape[1] > AMR_LEVEL1 + 2:
            lev = table[self.ids, AMR_LEVEL1:AMR_LEVEL1 + 3]
        else:
            lev = np.repeat(table[self.ids, AMR_LEVEL:AMR_LEVEL + 1], 3, axis=1)
        self.lev = lev.astype(int)
        self.coord = table[self.ids, AMR_COORD1:AMR_COORD1 + 3].astype(int)
        self.lmax = self.lev.max(axis=0)
        self.shape = tuple(int(x) for x in self.nb * self.bs * 2 ** self.lmax)
        self.factor = 2 ** (self.lmax - self.lev)              # finest cells per block cell
        self.start = self.coord * self.bs * self.factor        # first finest cell of each block
        self.coarse_mask = self._check_coverage()
        self.sample_frac = 0.5     # where each value sits inside its cell; set from grid_axes

    def _check_coverage(self):
        """Every finest cell must belong to exactly one block. Returns cells of unrefined blocks."""
        count = np.zeros(self.shape, np.uint8)
        coarse = np.zeros(self.shape, bool)
        for s, f in zip(self.start, self.factor):
            sl = tuple(slice(s[d], s[d] + self.bs[d] * f[d]) for d in range(3))
            count[sl] += 1
            if (f > 1).any():
                coarse[sl] = True
        if not (count == 1).all():
            raise ValueError(f"blocks do not tile the {self.shape} grid: "
                             f"{(count == 0).sum()} cells uncovered, {(count > 1).sum()} covered twice")
        return coarse

    def duplicated_mask(self, lowres):
        """Finest cells whose block is coarser than the output grid, so its values are spread
        over several output cells (copied or interpolated) instead of resolved."""
        mask = np.zeros(self.shape, bool)
        for s, f in zip(self.start, self.factor):
            if (f > np.asarray(lowres)).any():
                mask[tuple(slice(s[d], s[d] + self.bs[d] * f[d]) for d in range(3))] = True
        return mask

    def check_static(self, par, dump):
        if not np.array_equal(np.sort(par["n_ord"]), self.ids):
            raise ValueError(f"dump {dump}: the set of active blocks differs from the first dump "
                             "(the mesh changed); one fixed output grid no longer applies")

    def place(self, vals, fill="copy", axisym=False):
        """Blocks (n, b1, b2, b3, C) in ID order -> finest grid (N1, N2, N3, C).

        axisym: vals are (n, b1, b2, 1, C), independent of phi; the result is (N1, N2, 1, C).
        """
        n3 = 1 if axisym else self.shape[2]
        out = np.empty(self.shape[:2] + (n3,) + vals.shape[4:], vals.dtype)
        for v, s, f in zip(vals, self.start, self.factor):
            f3, s3 = (1, 0) if axisym else (f[2], s[2])
            v = np.repeat(np.repeat(np.repeat(v, f[0], 0), f[1], 1), f3, 2)
            out[s[0]:s[0] + v.shape[0], s[1]:s[1] + v.shape[1], s3:s3 + v.shape[2]] = v
        if fill == "interpolate" and self.coarse_mask.any():
            fine = self._interpolate_coarse(vals, axisym)
            mask = self.coarse_mask.any(axis=2, keepdims=True) if axisym else self.coarse_mask
            out[mask] = fine[mask]
        return out

    def _interpolate_coarse(self, vals, axisym):
        """Linear interpolation from the next-coarser level, at the true sample positions.

        A value sits sample_frac of the way into its cell, at every level: coarse cell k at
        2k + 2 frac finest cells, fine cell j at j + frac. Refined blocks are first interpolated
        onto the coarse sample positions, then every coarse value back onto the fine ones.
        """
        if not ((self.lev == self.lev[:, :1]).all() and (self.lmax == self.lmax[0]).all()
                and set(np.unique(self.lev)) <= {self.lmax[0] - 1, self.lmax[0]}):
            raise NotImplementedError("fill_coarse='interpolate' needs isotropic levels one apart")
        b = self.bs
        n3 = 1 if axisym else self.shape[2] // 2
        coarse = np.empty((self.shape[0] // 2, self.shape[1] // 2, n3) + vals.shape[4:])
        for v, s, f in zip(vals, self.start, self.factor):
            v = v.astype(np.float64)
            if f[0] == 1:                                     # refined block: onto the coarser level's samples
                for axis in (0, 1) if axisym else (0, 1, 2):
                    v = _restrict2(v, axis, self.sample_frac)
            c = s // 2
            c3 = 0 if axisym else c[2]
            coarse[c[0]:c[0] + v.shape[0], c[1]:c[1] + v.shape[1], c3:c3 + v.shape[2]] = v
        fine = _upsample2(coarse, 0, False, self.sample_frac)
        fine = _upsample2(fine, 1, False, self.sample_frac)
        if not axisym:
            fine = _upsample2(fine, 2, True, self.sample_frac)
        return fine.astype(vals.dtype)


def _restrict2(a, axis, frac):
    """Fine samples at 2k + frac and 2k + 1 + frac -> linear value at the coarse sample 2k + 2 frac."""
    a = np.moveaxis(a, axis, 0)
    return np.moveaxis((1 - frac) * a[0::2] + frac * a[1::2], 0, axis)


def _upsample2(a, axis, periodic, frac):
    """Coarse samples (at 2k + 2 frac) -> linear values at the fine samples 2k + frac and 2k + 1 + frac.
    For centred values (frac = 1/2) the children sit 1/4 of a coarse cell either side of the parent."""
    a = np.moveaxis(a, axis, 0)
    if periodic:
        lo, hi = np.roll(a, 1, 0), np.roll(a, -1, 0)
    else:
        lo = np.concatenate([a[:1], a[:-1]])
        hi = np.concatenate([a[1:], a[-1:]])
    w_lo, w_hi = frac / 2, (1 - frac) / 2
    out = np.empty((2 * a.shape[0],) + a.shape[1:], a.dtype)
    out[0::2] = (1 - w_lo) * a + w_lo * lo
    out[1::2] = (1 - w_hi) * a + w_hi * hi
    return np.moveaxis(out, 0, axis)


def coarsen(a, k):
    """Average k[d] cells per direction of a (N1, N2, N3, ...) array (N3 may be 1 for axisym)."""
    k = [k[0], k[1], k[2] if a.shape[2] > 1 else 1]
    if k == [1, 1, 1]:
        return a
    n = a.shape[:3]
    if any(n[d] % k[d] for d in range(3)):
        raise ValueError(f"grid {n} is not divisible by lowres {k}")
    shape = (n[0] // k[0], k[0], n[1] // k[1], k[1], n[2] // k[2], k[2]) + a.shape[3:]
    return a.reshape(shape).mean(axis=(1, 3, 5), dtype=np.float64).astype(a.dtype)


# ---------------------------------------------------------------- grid, crop and checks

def grid_axes(par, layout, lowres):
    """Code-coordinate spacing and sample positions of the output grid (finest level, then lowres).

    The spacing comes from Rin/Rout for x1. The reduced run was written by pp's downscale(),
    which keeps every 4th cell (the first of each group of 4) and leaves the full run's dx in
    parameters. So the file's dx is rescaled by that factor D in all three directions, and each
    value sits 0.5/D of the way into its cell (1/8 for the reduced run, the centre for D = 1).
    """
    bs, nb = layout.bs, layout.nb
    dx_base1 = (np.log(par["Rout"]) - np.log(par["Rin"])) / (nb[0] * bs[0])
    scale = dx_base1 / par["dx"][0]
    if round(scale) < 1 or abs(scale - round(scale)) > 1e-6 * scale:
        raise ValueError(f"file dx1 {par['dx'][0]} is not a whole fraction of (ln Rout - ln Rin)/(nb1 bs1) = {dx_base1}")
    scale = int(round(scale))
    frac = 0.5 / scale
    dx_fine = par["dx"] * scale / 2.0 ** layout.lmax
    fine_samples = [par["startx"][d] + (np.arange(layout.shape[d]) + frac) * dx_fine[d] for d in range(3)]
    dx = dx_fine * np.asarray(lowres)
    n = [layout.shape[d] // lowres[d] for d in range(3)]
    edges = [par["startx"][d] + np.arange(n[d] + 1) * dx[d] for d in range(3)]
    return dict(file_dx_scale=scale, sample_frac=frac, startx=par["startx"], dx_fine=dx_fine,
                fine_samples=fine_samples, dx=dx, edges=edges,
                samples=[fine_samples[d].reshape(n[d], lowres[d]).mean(axis=1) for d in range(3)],
                phi_periodic=bool(abs(layout.shape[2] * dx_fine[2] - 2 * np.pi) < 1e-6))


def to_r(x1):
    return np.exp(x1)


def to_theta(x2):
    return 0.5 * np.pi * (x2 + 1.0)


def check_block_coordinates(layout, coords, axes):
    """gdump cell coordinates must match the uniform grid (values 0.5/D into each cell, at the
    block's own level), and r, theta, phi the assumed mapping."""
    report = {}
    frac = axes["sample_frac"]
    for d, name in enumerate(("x1", "x2", "x3")):
        x0, dx = axes["startx"][d], axes["dx_fine"][d]
        worst = 0.0
        for n in range(len(layout.ids)):
            f, s, b = layout.factor[n, d], layout.start[n, d], layout.bs[d]
            expect = x0 + (s + (np.arange(b) + frac) * f) * dx
            got = np.moveaxis(coords[n, ..., 3 + d], d, 0)
            worst = max(worst, float(np.abs(got - expect.reshape((b,) + (1,) * 2)).max() / dx))
        report[f"{name}_max_error_in_dx"] = worst
        if worst > COORD_TOL:
            raise ValueError(f"gdump {name} differs from the uniform grid by {worst:.3g} cells; "
                             "the block placement or the grid spacing is wrong")
    x1, x2, x3 = coords[..., 3], coords[..., 4], coords[..., 5]
    r, h, ph = coords[..., 6], coords[..., 7], coords[..., 8]
    errs = {"r_vs_exp_x1_rel": float(np.abs(r / to_r(x1) - 1).max()),
            "theta_vs_x2_abs": float(np.abs(h - to_theta(x2)).max()),
            "phi_vs_x3_abs": float(np.abs(ph - x3).max())}
    report.update(errs)
    bad = {k: v for k, v in errs.items() if v > MAPPING_TOL}
    if bad:
        sample = np.stack([x1.ravel()[:3], r.ravel()[:3], x2.ravel()[:3], h.ravel()[:3]], axis=1)
        raise ValueError(f"coordinate mapping is not r = exp(x1), theta = pi (x2 + 1) / 2, phi = x3: {bad}. "
                         f"Sample rows (x1, r, x2, theta): {sample.tolist()}")
    return report


def axis_range(edges, lo, hi, align, name):
    """Cells overlapping [lo, hi), widened to a multiple of align. edges ascending."""
    n = len(edges) - 1
    i0 = 0 if lo is None else int(np.searchsorted(edges[1:], lo, side="right"))
    i1 = n if hi is None else int(np.searchsorted(edges[:-1], hi, side="left"))
    if i1 <= i0:
        raise SystemExit(f"{name} crop [{lo}, {hi}] keeps no cells (grid {edges[0]:.4g}-{edges[-1]:.4g})")
    target = min(n, math.ceil((i1 - i0) / align) * align)
    i1 = min(n, i0 + target)
    i0 = max(0, i1 - target)
    return i0, i1


def crop_ranges(s, axes):
    e1, e2, e3 = axes["edges"]
    r_edges, th_edges, ph_edges = to_r(e1), np.degrees(to_theta(e2)), np.degrees(e3)
    th = s["theta_deg"] or [None, None]
    ph = s["phi_deg"] or [None, None]
    a = s["align_cells"]
    return [axis_range(r_edges, s["r_min"], s["r_max"], a[0], "r"),
            axis_range(th_edges, th[0], th[1], a[1], "theta"),
            axis_range(ph_edges, ph[0], ph[1], a[2], "phi")]


def crop(a, ranges, axisym=False):
    sl = [slice(*ranges[0]), slice(*ranges[1]), slice(None) if axisym else slice(*ranges[2])]
    return a[tuple(sl)]


def utilde_residual(v, u0, gcov, gcon):
    """Per cell |sqrt(1 + g_ij v^i v^j) - alpha u^0| / (alpha u^0): zero if v is HARM's relative
    4-velocity utilde. gcov, gcon: (n, b1, b2, 4, 4), axisymmetric (broadcast over phi)."""
    v, u0 = v.astype(np.float64), u0.astype(np.float64)
    alpha = 1.0 / np.sqrt(-gcon[:, :, :, None, 0, 0])
    q = np.einsum("...i,...ij,...j->...", v, gcov[:, :, :, None, 1:, 1:], v)
    return np.abs(np.sqrt(1.0 + np.maximum(q, 0.0)) - alpha * u0) / (alpha * u0)


def coordinate_residual(v, u0, gcov):
    """Per cell |g_mn u^m u^n + 1| / sum |g_mn u^m u^n|, with u = (u^0, v): zero if v is the
    coordinate 4-velocity u^i. Relative to the size of the terms, like utilde_residual."""
    u4 = np.concatenate([u0[..., None], v], axis=-1).astype(np.float64)
    g = gcov[:, :, :, None]
    norm = np.einsum("...m,...mn,...n->...", u4, g, u4)
    scale = np.einsum("...m,...mn,...n->...", np.abs(u4), np.abs(g), np.abs(u4))
    return np.abs(norm + 1.0) / scale


def to_utilde(blocks, gcon, convention):
    """Output velocity channels as utilde^i. From coordinate u^i: utilde^i = u^i + u^0 beta^i with
    beta^i = -g^{0i}/g^{00} (the inverse of DataNormalizer.ucon_of)."""
    v = blocks[..., 3:6]
    if convention == "utilde":
        return v
    beta = -gcon[:, :, :, None, 0, 1:] / gcon[:, :, :, None, 0, 0:1]
    return (v + blocks[..., U0:U0 + 1] * beta).astype(np.float32)


def divb_ratio(b, gdet, dx, phi_periodic):
    """|div B| relative to the sum of its terms' magnitudes, per cell, as PhysicsLoss3D.calc_divB
    differentiates (central differences; edges copied in r and theta, wrapped in phi)."""
    f = [gdet * b[i].astype(np.float64) for i in range(3)]
    p1 = np.concatenate([f[0][:1], f[0], f[0][-1:]], axis=0)
    p2 = np.concatenate([f[1][:, :1], f[1], f[1][:, -1:]], axis=1)
    p3 = (np.concatenate([f[2][:, :, -1:], f[2], f[2][:, :, :1]], axis=2) if phi_periodic
          else np.concatenate([f[2][:, :, :1], f[2], f[2][:, :, -1:]], axis=2))
    d = [(p1[2:] - p1[:-2]) / (2 * dx[0]), (p2[:, 2:] - p2[:, :-2]) / (2 * dx[1]),
         (p3[:, :, 2:] - p3[:, :, :-2]) / (2 * dx[2])]
    return np.abs(d[0] + d[1] + d[2]) / (np.abs(d[0]) + np.abs(d[1]) + np.abs(d[2]) + 1e-300)


# ---------------------------------------------------------------- geometry

def build_geometry(s, plan):
    run_dir, first = s["run_dir"], plan["dumps"][0]
    par = read_parameters(run_dir, first)
    if par["flags"] != 0:
        raise NotImplementedError(f"flags = {par['flags']}: extra variables per cell are not supported")
    layout = Layout(par, read_block_table(run_dir))
    coords, metric = read_gdumps(run_dir, layout.ids, layout.bs)
    axes = grid_axes(par, layout, s["lowres"])
    layout.sample_frac = axes["sample_frac"]
    checks = {"finest_grid": list(layout.shape), "levels": sorted(set(layout.lev.ravel().tolist())),
              "unrefined_cells_fraction": float(layout.coarse_mask.mean()),
              "file_dx_scale": axes["file_dx_scale"], "sample_position_in_cell": axes["sample_frac"],
              "phi_periodic": axes["phi_periodic"]}
    checks.update(check_block_coordinates(layout, coords, axes))

    gcov_b = metric[..., 0:16].reshape(metric.shape[:3] + (4, 4))
    gcon_b = metric[..., 16:32].reshape(metric.shape[:3] + (4, 4))

    # Velocity convention, decided once from the first dump. Far from the hole the shift is tiny and
    # both readings fit, so judge by the share of cells that fail, which the active region decides.
    blocks = read_blocks(run_dir, first, par, layout.bs)
    v, u0 = blocks[..., 3:6], blocks[..., U0]
    res = {"utilde": utilde_residual(v, u0, gcov_b, gcon_b), "coordinate": coordinate_residual(v, u0, gcov_b)}
    bad = {k: float((r > VEL_TOL).mean()) for k, r in res.items()}
    worst = {k: float(r.max()) for k, r in res.items()}
    fits = [k for k in res if bad[k] <= VEL_BAD_FRACTION]
    if not fits:
        raise ValueError(f"stored velocities fit neither convention (share of cells with residual > {VEL_TOL}: "
                         f"{bad}); see doc section 4.5")
    convention = min(fits, key=lambda k: worst[k])
    checks.update(velocity_convention=convention, velocity_convention_ambiguous=len(fits) > 1,
                  velocity_bad_fraction=bad, velocity_max_residual=worst)
    if len(fits) > 1:
        print(f"  WARNING: both velocity conventions fit dump {first} (no active region?); chose {convention}",
              flush=True)

    ranges = crop_ranges(s, axes)
    fill, low = s["fill_coarse"], s["lowres"]
    # Metric is axisymmetric: (n, b1, b2, 1, C) -> (Nr, Ntheta, 1, C)
    met = crop(coarsen(layout.place(metric[:, :, :, None, 0:33], fill, axisym=True), low), ranges, axisym=True)
    nr, nt = met.shape[:2]
    gcov = met[..., 0:16].reshape(nr, nt, 1, 4, 4).transpose(3, 4, 0, 1, 2)
    gcon = met[..., 16:32].reshape(nr, nt, 1, 4, 4).transpose(3, 4, 0, 1, 2)
    gdet = met[..., 32]

    c1, c2, c3 = (axes["samples"][d][slice(*ranges[d])] for d in range(3))
    r = np.broadcast_to(to_r(c1)[:, None, None], (len(c1), len(c2), len(c3)))
    h = np.broadcast_to(to_theta(c2)[None, :, None], r.shape)
    ph = np.broadcast_to(c3[None, None, :], r.shape)
    resolved = crop(coarsen((~layout.duplicated_mask(low)).astype(np.float32)[..., None], low), ranges)[..., 0] == 1.0

    t_next = read_parameters(run_dir, plan["dumps"][1])["t"] if len(plan["dumps"]) > 1 else par["t"] + par["Dtd"]
    dt_exact = t_next - par["t"]
    # t advances in whole time steps, so the measured gap slightly overshoots a round cadence (100.0027)
    dt_dump = float(round(dt_exact)) if abs(dt_exact - round(dt_exact)) < 1e-3 * dt_exact else dt_exact
    geometry = dict(
        r=np.ascontiguousarray(r), h=np.ascontiguousarray(h), ph=np.ascontiguousarray(ph),
        gcov=gcov.astype(np.float32), gcon=gcon.astype(np.float32), gdet=gdet.astype(np.float32),
        dx=axes["dx"], resolved=resolved,
        eos_gamma=par["gam"], dt_dump=dt_dump, dt_dump_exact=dt_exact, dtd=par["Dtd"],
        spin=par["a"], density_scale=par["density_scale"], rin=par["Rin"], rout=par["Rout"],
        r0=par["R0"], tf=par["tf"], t_start=par["t"],
        crop_index=np.array(ranges), finest_grid=np.array(layout.shape), lowres=np.array(low),
        sample_frac=axes["sample_frac"], format_version=FORMAT_VERSION,
    )
    out_shape = (len(CHANNEL_NAMES),) + r.shape
    checks.update(output_grid=list(r.shape),
                  crop_r=[float(to_r(axes["edges"][0][ranges[0][0]])), float(to_r(axes["edges"][0][ranges[0][1]]))],
                  crop_theta_deg=[float(np.degrees(to_theta(axes["edges"][1][i]))) for i in ranges[1]],
                  crop_phi_deg=[float(np.degrees(axes["edges"][2][i])) for i in ranges[2]],
                  dx=axes["dx"].tolist(), resolved_fraction=float(resolved.mean()),
                  phi_full_circle=bool(axes["phi_periodic"] and ranges[2] == (0, len(axes["samples"][2]))))
    worker = dict(layout=layout, gcon_b=gcon_b, gcov_b=gcov_b, convention=convention,
                  ranges=ranges, fill=fill, lowres=low, out_shape=out_shape)
    return geometry, checks, worker


# ---------------------------------------------------------------- per-dump work

_CTX = {}


def _init_worker(ctx):
    _CTX.update(ctx)


def _stats(a):
    d = a.reshape(a.shape[0], -1).astype(np.float64)
    return dict(min=d.min(axis=1), max=d.max(axis=1), sum=d.sum(axis=1),
                sumsq=(d * d).sum(axis=1), count=d.shape[1])


def process_dump(dump):
    c = _CTX
    path = c["paths"][dump]
    try:
        if not c["overwrite"] and os.path.exists(path):
            try:
                a = np.load(path, mmap_mode="r")
                if a.shape == c["out_shape"] and a.dtype == np.float32:
                    t = read_parameters(c["run_dir"], dump)["t"]
                    return dict(dump=dump, t=t, skipped=True, stats=_stats(np.asarray(a)))
            except (ValueError, OSError):
                pass
        layout = c["layout"]
        par = read_parameters(c["run_dir"], dump)
        layout.check_static(par, dump)
        blocks = read_blocks(c["run_dir"], dump, par, layout.bs)

        problems = []
        n_bad = int((~np.isfinite(blocks)).sum())
        rho_min, ug_min = float(blocks[..., 0].min()), float(blocks[..., 1].min())
        u0_min = float(blocks[..., U0].min())
        if n_bad:
            problems.append(f"{n_bad} non-finite values")
        if rho_min <= 0 or ug_min <= 0:
            problems.append(f"non-positive rho ({rho_min:.3g}) or ug ({ug_min:.3g})")
        if u0_min < 1.0 - 1e-5:
            problems.append(f"u0 below 1 ({u0_min:.6g})")
        # Check the velocities actually written: utilde must satisfy the Lorentz-factor identity in every cell
        vt = to_utilde(blocks, c["gcon_b"], c["convention"])
        res = utilde_residual(vt, blocks[..., U0], c["gcov_b"], c["gcon_b"])
        vel_p99, vel_max = float(np.percentile(res, 99)), float(res.max())
        n_vel_bad = int((res > VEL_TOL).sum())
        if n_vel_bad:
            problems.append(f"{n_vel_bad} cells with velocity residual > {VEL_TOL} (max {vel_max:.3g})")

        vals = blocks[..., RAW_CHANNELS]
        vals[..., 2:5] = vt
        grid = crop(coarsen(layout.place(vals, c["fill"]), c["lowres"]), c["ranges"])
        out = np.ascontiguousarray(np.moveaxis(grid, -1, 0), dtype=np.float32)
        if out.shape != c["out_shape"]:
            raise ValueError(f"output {out.shape} != expected {c['out_shape']}")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.save(f, out)
        os.replace(tmp, path)
        return dict(dump=dump, t=par["t"], skipped=False, stats=_stats(out), problems=problems,
                    checks=dict(rho_min=rho_min, ug_min=ug_min, u0_min=u0_min,
                                velocity_residual_p99=vel_p99, velocity_residual_max=vel_max))
    except Exception:
        return dict(dump=dump, error=traceback.format_exc())


# ---------------------------------------------------------------- main

def main(argv=None):
    args = parse_args(argv)
    s = build_settings(args.set)
    out_dir = s["out_dir"]
    t0 = time.time()

    plan = plan_splits(s)
    targets = args.dumps or plan["dumps"]
    unknown = [d for d in targets if d not in plan["split"]]
    if unknown:
        raise SystemExit(f"dumps {unknown} are outside {plan['dumps'][0]}-{plan['dumps'][-1]}")
    os.makedirs(out_dir, exist_ok=True)
    record_settings(out_dir, s, args.overwrite)
    r = plan["ranges"]
    print(f"splits over dumps {plan['dumps'][0]}-{plan['dumps'][-1]}: "
          + ", ".join(f"{k} {v['first']}-{v['last']} ({v['count']})" for k, v in r.items()), flush=True)

    print("building geometry ...", flush=True)
    geometry, geo_checks, ctx = build_geometry(s, plan)
    np.savez_compressed(os.path.join(out_dir, "geometry.npz"), **geometry)
    print(f"  finest grid {geo_checks['finest_grid']}, output grid {geo_checks['output_grid']}, "
          f"r {geo_checks['crop_r'][0]:.4g}-{geo_checks['crop_r'][1]:.4g}, "
          f"theta {geo_checks['crop_theta_deg'][0]:.3f}-{geo_checks['crop_theta_deg'][1]:.3f} deg, "
          f"phi {geo_checks['crop_phi_deg'][0]:.1f}-{geo_checks['crop_phi_deg'][1]:.1f} deg", flush=True)
    print(f"  dx {geo_checks['dx']}, velocities stored as {geo_checks['velocity_convention']} "
          f"(share of cells failing as utilde {geo_checks['velocity_bad_fraction']['utilde']:.2%}, "
          f"as coordinate {geo_checks['velocity_bad_fraction']['coordinate']:.2%}), "
          f"{100 * geo_checks['resolved_fraction']:.1f}% of output cells resolved (the rest filled from "
          f"unrefined blocks)", flush=True)

    ctx.update(run_dir=s["run_dir"], overwrite=args.overwrite,
               paths={d: output_path(out_dir, d, plan["split"][d]) for d in targets})
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    workers = max(1, min(workers, len(targets)))
    print(f"processing {len(targets)} dumps with {workers} workers ...", flush=True)

    results = []
    if workers == 1:
        _init_worker(ctx)
        it = map(process_dump, targets)
    else:
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        pool = mp.get_context(method).Pool(workers, initializer=_init_worker, initargs=(ctx,))
        it = pool.imap_unordered(process_dump, targets, chunksize=4)
    every = max(1, len(targets) // 20)
    for k, rec in enumerate(it, 1):
        results.append(rec)
        if "error" in rec:
            print(f"  dump {rec['dump']} FAILED:\n{rec['error']}", flush=True)
        if k % every == 0 or k == len(targets):
            print(f"  {k}/{len(targets)} done ({time.time() - t0:.0f} s)", flush=True)
    if workers > 1:
        pool.close()
        pool.join()
    results.sort(key=lambda rec: rec["dump"])
    ok = [rec for rec in results if "error" not in rec]
    failed = [rec["dump"] for rec in results if "error" in rec]

    # Channel statistics over the training dumps only (no validation or test data). A run limited
    # with --dumps writes them to a separate file so it never replaces the full-run statistics.
    train_ok = [rec for rec in ok if plan["split"][rec["dump"]] == "train"]
    stats_name = "channel_stats_partial.npz" if args.dumps else "channel_stats.npz"
    if train_ok:
        st = [rec["stats"] for rec in train_ok]
        count = sum(x["count"] for x in st)
        mean = sum(x["sum"] for x in st) / count
        std = np.sqrt(np.maximum(sum(x["sumsq"] for x in st) / count - mean ** 2, 0.0))
        lo = np.min([x["min"] for x in st], axis=0)
        hi = np.max([x["max"] for x in st], axis=0)
        np.savez_compressed(os.path.join(out_dir, stats_name), channel_names=np.array(CHANNEL_NAMES),
                            min=lo, max=hi, mean=mean, std=std, count=count, n_dumps=len(train_ok))
        print(f"{stats_name} over {len(train_ok)} training dumps:", flush=True)
        print("channel   min          max          mean         std", flush=True)
        for i, name in enumerate(CHANNEL_NAMES):
            print(f"  {name:8s} {lo[i]: .4e} {hi[i]: .4e} {mean[i]: .4e} {std[i]: .4e}")

    # Splits and dump times (merged with earlier runs into the same folder)
    splits_path = os.path.join(out_dir, "splits.json")
    times = {}
    if os.path.exists(splits_path):
        with open(splits_path) as f:
            times = json.load(f).get("t", {})
    times.update({str(rec["dump"]): rec["t"] for rec in ok})
    with open(splits_path, "w") as f:
        json.dump(dict(ranges=r, val_frac_for_training=plan["val_frac_for_training"],
                       train_val_dir=out_dir, test_dir=os.path.join(out_dir, "blind"),
                       t=dict(sorted(times.items(), key=lambda kv: int(kv[0])))), f, indent=1)

    # Validation summary
    processed = [rec for rec in ok if not rec["skipped"]]
    problems = {rec["dump"]: rec["problems"] for rec in processed if rec["problems"]}
    per_dump = {k: [rec["checks"][k] for rec in processed] for k in
                ("rho_min", "ug_min", "u0_min", "velocity_residual_p99", "velocity_residual_max")}
    summary = {"rho_min": min(per_dump["rho_min"], default=None), "ug_min": min(per_dump["ug_min"], default=None),
               "u0_min": min(per_dump["u0_min"], default=None),
               "velocity_residual_p99_worst": max(per_dump["velocity_residual_p99"], default=None),
               "velocity_residual_max_worst": max(per_dump["velocity_residual_max"], default=None)}
    if ok:
        first = np.load(ctx["paths"][ok[0]["dump"]])
        ratio = divb_ratio(first[5:8], geometry["gdet"], geometry["dx"], geo_checks["phi_full_circle"])
        res = geometry["resolved"]
        summary[f"divB_relative_median_dump{ok[0]['dump']}"] = {
            "full_resolution_cells": float(np.median(ratio[res])) if res.any() else None,
            "unrefined_cells": float(np.median(ratio[~res])) if (~res).any() else None}
    checks = dict(geometry=geo_checks, dumps=summary, processed=len(processed),
                  skipped_existing=len(ok) - len(processed), failed=failed, problems=problems)
    with open(os.path.join(out_dir, "checks.json"), "w") as f:
        json.dump(checks, f, indent=1)

    size_gb = np.prod(ctx["out_shape"]) * 4 / 1e9
    print(f"checks: rho_min {summary['rho_min']}, ug_min {summary['ug_min']}, u0_min {summary['u0_min']}, "
          f"velocity residual p99 worst {summary['velocity_residual_p99_worst']}")
    for key, val in summary.items():
        if key.startswith("divB"):
            print(f"  {key}: {val} (informational)")
    print(f"done in {time.time() - t0:.0f} s: {len(processed)} processed, {len(ok) - len(processed)} already "
          f"present, {len(failed)} failed. Output {ctx['out_shape']} float32, {size_gb:.3f} GB per dump, "
          f"{size_gb * len(plan['dumps']):.0f} GB for the full range.")
    print(f"In the training experiment set val_frac = {plan['val_frac_for_training']!r} "
          f"(train {r['train']['count']}, val {r['val']['count']}); test dumps are in {os.path.join(out_dir, 'blind')}.")
    if problems or failed:
        print(f"PROBLEMS: {len(problems)} dumps failed checks, {len(failed)} failed to process; see checks.json",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
