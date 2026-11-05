# Copyright (c) 2026 Kevin Fortune. All rights reserved, except for portions adapted from
# code by Sachin Venkatesh (the TRON model), which remain his. See LICENSE.
"""Geometry loader and data normalizer.

Copied from uno_simple_ds_v3.py (sachin branch, fall26/, commit ca0288555c99).
Changes from the original:
  - CONFIG is the shared PHYSICS settings dict (common/settings.py).
  - check_normalizer_constants takes the geometry path, logs through `logging`, and also warns
    when a B channel exceeds the saturation guard (b_mag_threshold).
  - The Lorentz-clamp counter accumulates on the device and lorentz_clamp_count() returns a
    tensor, so inverse() no longer waits on the GPU every call.
  - A normalizer describes its own channel layout (n_channels, vel, mag, spectral_groups) and
    saturation guard (b_excess), so the physics loss works with any of them.
DataNormalizer itself is unchanged, so its normalized tensors match TRON's exactly.

Normalizations (an experiment's "normalization" setting picks one; build_normalizer):
  - "arcsinh" (DataNormalizer), 8 channels: log10 rho, log10 ug, arcsinh of each ũ^i and B^i.
    TRON's normalization.
  - "magdir" (MagDirNormalizer), 10 channels: log10 rho, log10 ug, then for ũ and for B the log10
    of the vector's magnitude and its unit direction (3 components). Every value is relative, as
    log10 rho is, without arcsinh's cliffs where a component changes sign.
Both decode to the same 8 physical channels, so the physics loss and evaluation see the same data.
"""
import logging
import os

import numpy as np
import torch
import torch.nn as nn

from .settings import PHYSICS as CONFIG

logger = logging.getLogger(__name__)


def check_normalizer_constants(geometry_file):
    """Warn-only sanity check: whole-dataset channel stats from the
    preprocess channel_stats.npz vs. the normalizer's floor/ceiling constants.
    """
    stats_path = os.path.join(os.path.dirname(geometry_file), "channel_stats.npz")
    if not os.path.exists(stats_path):
        logger.warning("channel_stats.npz not found next to geometry_file; skipping normalizer bound check.")
        return
    st = np.load(stats_path)
    names = [str(n) for n in st['channel_names']]
    for name, lo_raw, hi_raw in zip(names, st['min'], st['max']):
        lo, hi = float(lo_raw), float(hi_raw)
        if name == 'rho':
            if lo < CONFIG["floor_rho"]:
                logger.warning(f"rho min {lo:.3e} < floor_rho {CONFIG['floor_rho']:.0e}: cells saturate at the floor in forward()")
            if hi > CONFIG["ceiling_rho"]:
                logger.warning(f"rho max {hi:.3e} > ceiling_rho {CONFIG['ceiling_rho']:.0e}: cells saturate at the ceiling in inverse()")
        elif name == 'ug':
            if lo < CONFIG["floor_ug"]:
                logger.warning(f"ug min {lo:.3e} < floor_ug {CONFIG['floor_ug']:.0e}: cells saturate at the floor in forward()")
            if hi > CONFIG["ceiling_ug"]:
                logger.warning(f"ug max {hi:.3e} > ceiling_ug {CONFIG['ceiling_ug']:.0e}: cells saturate at the ceiling in inverse()")
        elif name.startswith('B') and np.arcsinh(max(abs(lo), abs(hi))) > CONFIG["b_mag_threshold"]:
            logger.warning(f"{name} reaches |B| = {max(abs(lo), abs(hi)):.3e}, above the saturation guard "
                           f"(|arcsinh(B)| > b_mag_threshold {CONFIG['b_mag_threshold']}): "
                           "the loss would penalize the true values")
        else:
            logger.info(f"{name}: range [{lo:.3e}, {hi:.3e}]")


def load_geometry(path, device=None):
    """Load metric + run metadata into one canonical form.
    All metric buffers get leading (1,1,...) dims so they broadcast against
    (B, C, Nr, Ntheta, Nphi); used by both DataNormalizer and PhysicsLoss3D.
    """
    geo = np.load(path)
    def H(x):
        t = torch.tensor(x, dtype=torch.float32)
        if device is not None:
            t = t.to(device)
        return t.unsqueeze(0).unsqueeze(0)
    return dict(
        gdet=H(geo['gdet']),                     # (1,1,Nr,Ntheta,Nphi) via broadcast of (1,1,Nr,Ntheta,1)
        gcov=H(geo['gcov']),                     # (1,1,4,4,Nr,Ntheta,Nphi)
        gcon=H(geo['gcon']),                     # (1,1,4,4,Nr,Ntheta,Nphi)
        dx=geo['dx'],
        eos_gamma=float(geo['eos_gamma']) if 'eos_gamma' in geo else None,
        dt_dump=float(geo['dt_dump']) if 'dt_dump' in geo else None,
    )


class DataNormalizer(nn.Module):
    """Universal Scale-Invariant Normalizer with exact Lorentz clamp for relative velocities.

    Velocity channels (2:5) are HARM projected primitive velocities utilde^i = j^i_mu u^mu.
    In utilde space:
      - Spatial 3-metric gamma_ij = g_ij is positive-definite everywhere.
      - Lorentz factor gamma = sqrt(1 + gamma_ij utilde^i utilde^j) >= 1 is ALWAYS real
        and well-defined across all R^3 (including inside the ergosphere).
      - Coordinate 4-velocity components are recovered analytically:
          u^0 = gamma / alpha = gamma * sqrt(-g^{00})
          u^i = utilde^i - u^0 * beta^i = utilde^i + u^0 * (g^{0i} / g^{00})
      - Exact Lorentz clamp to max_gamma scales utilde by s = sqrt((max_gamma^2 - 1)/(gamma^2 - 1)) in (0, 1].

    Channel layout, read by the physics loss: channels 0, 1 are log10 rho and log10 ug in every
    normalization; `vel` and `mag` are the velocity and magnetic channels; `spectral_groups`
    lists the channels the spectral loss scores together, each with its denominator floor.
    """
    n_channels = 8
    vel, mag = slice(2, 5), slice(5, 8)
    spectral_groups = (((0, 1), 5.0), ((2, 3, 4), 0.08), ((5, 6, 7), 0.03))   # floors: PhysicsLoss3D.spectral_loss

    def __init__(self, gcov=None, gcon=None):
        super().__init__()
        if gcov is not None:
            assert gcov.shape[:3] == (1, 1, 4), f"gcov must be (1,1,4,4,Nr,Ntheta,Nphi), got {tuple(gcov.shape)}"
            self.register_buffer('gcov', gcov)
        else:
            self.gcov = None
        if gcon is not None:
            assert gcon.shape[:3] == (1, 1, 4), f"gcon must be (1,1,4,4,Nr,Ntheta,Nphi), got {tuple(gcon.shape)}"
            self.register_buffer('gcon', gcon)
        else:
            self.gcon = None
        self._lorentz_clamp_cells = 0

    def lorentz_clamp_count(self, reset=True):
        """Cells whose pre-clamp Lorentz factor exceeded max_gamma in inverse() since the last
        reset, as a tensor (so the caller can sum it across ranks before reading it)."""
        n = torch.as_tensor(self._lorentz_clamp_cells, dtype=torch.float64)
        if reset:
            self._lorentz_clamp_cells = 0
        return n

    def gamma_of_raw(self, utilde):
        """Exact Lorentz factor gamma = sqrt(1 + gamma_ij utilde^i utilde^j) from raw utilde."""
        u1, u2, u3 = (utilde[:, i:i + 1].double() for i in range(3))
        g = self.gcov[0, 0].double()
        qsq = (g[1, 1] * u1**2 + g[2, 2] * u2**2 + g[3, 3] * u3**2
               + 2.0 * (g[1, 2] * u1 * u2 + g[1, 3] * u1 * u3 + g[2, 3] * u2 * u3))
        return torch.sqrt(1.0 + torch.clamp(qsq, min=0.0))

    def u0_of(self, utilde):
        """Exact u^0 = gamma / alpha = gamma * sqrt(-g^{00}) from raw utilde."""
        gamma = self.gamma_of_raw(utilde)
        alpha = 1.0 / torch.sqrt(-self.gcon[0, 0, 0, 0].double() + 1e-300)
        return (gamma / alpha).clamp(max=3.0e38).float()

    def ucon_of(self, utilde):
        """Reconstruct full coordinate 4-velocity (u^0, u^1, u^2, u^3) from utilde."""
        u1, u2, u3 = (utilde[:, i:i + 1].double() for i in range(3))
        gamma = self.gamma_of_raw(utilde)
        alpha = 1.0 / torch.sqrt(-self.gcon[0, 0, 0, 0].double() + 1e-300)
        u0 = gamma / alpha
        gcon00 = self.gcon[0, 0, 0, 0].double()
        u_coord1 = (u1 + u0 * (self.gcon[0, 0, 0, 1].double() / gcon00)).float()
        u_coord2 = (u2 + u0 * (self.gcon[0, 0, 0, 2].double() / gcon00)).float()
        u_coord3 = (u3 + u0 * (self.gcon[0, 0, 0, 3].double() / gcon00)).float()
        return torch.cat([u0.float(), u_coord1, u_coord2, u_coord3], dim=1)

    def forward(self, raw):
        rho = torch.log10(torch.clamp(raw[:, 0], min=CONFIG["floor_rho"]))
        ug  = torch.log10(torch.clamp(raw[:, 1], min=CONFIG["floor_ug"]))
        v   = torch.arcsinh(raw[:, 2:5])
        b   = torch.arcsinh(raw[:, 5:8])
        return torch.cat([rho[:, None], ug[:, None], v, b], dim=1)

    def _decode_vectors(self, norm):
        """Raw ũ and B, each (B, 3, ...), from the normalized channels."""
        return torch.sinh(norm[:, 2:5].clamp(-80.0, 80.0)), torch.sinh(norm[:, 5:8].clamp(-80.0, 80.0))

    def b_excess(self, norm, threshold):
        """Saturation guard: how far each normalized B component is past |arcsinh(B)| = threshold."""
        return torch.relu(norm[:, self.mag].abs() - threshold)

    def _lorentz_scale_exact(self, utilde, gamma):
        """Closed-form exact Lorentz clamp scale in utilde space:
        gamma(s * utilde) = sqrt(1 + s^2 * q^2) = max_gamma => s = sqrt((max_gamma^2 - 1) / (gamma^2 - 1))
        """
        need = gamma.double() > CONFIG["max_gamma"]
        gamma_sq = gamma.double() ** 2
        max_gamma_sq = CONFIG["max_gamma"] ** 2
        denom = torch.clamp(gamma_sq - 1.0, min=1e-30)
        s_exact = torch.sqrt(torch.clamp((max_gamma_sq - 1.0) / denom, max=1.0))
        scale = torch.where(need, s_exact, torch.ones_like(gamma.double()))
        return scale, need

    def inverse(self, norm, return_gamma=False):
        """Decode normalized channels to physical units with closed-form exact Lorentz clamp."""
        safe_rho = torch.clamp(norm[:, 0], min=np.log10(CONFIG["floor_rho"]), max=np.log10(CONFIG["ceiling_rho"]))
        safe_ug  = torch.clamp(norm[:, 1], min=np.log10(CONFIG["floor_ug"]), max=np.log10(CONFIG["ceiling_ug"]))

        rho = torch.pow(10, safe_rho)
        ug  = torch.pow(10, safe_ug)
        v, b = self._decode_vectors(norm)

        gamma = None
        if self.gcov is not None:
            gamma = self.gamma_of_raw(v)
            scale, need = self._lorentz_scale_exact(v, gamma)
            v = (v.double() * scale).float()
            self._lorentz_clamp_cells = self._lorentz_clamp_cells + need.sum()   # stays on the GPU

        if CONFIG.get("detect_anomaly", False):
            assert torch.isfinite(rho).all() and torch.isfinite(ug).all(), \
                "detect_anomaly: non-finite rho/ug in inverse() (NaN contamination triage)"
            assert (rho >= 0).all() and (ug >= 0).all()

        out = torch.cat([rho[:, None], ug[:, None], v, b], dim=1)
        if return_gamma:
            return out, gamma
        return out


class MagDirNormalizer(DataNormalizer):
    """10 channels: log10 rho, log10 ug, then ũ and B each as the log10 of the vector's magnitude
    followed by its unit direction (3 components, in code coordinates).

    The magnitude, like log10 rho, makes every error relative: a factor of 2 costs log10(2) at any
    size, so the weak outer field counts as much as the strong inner field. A component changing
    sign is a smooth turn of the direction, not the jump arcsinh(x / s) makes for small s; the
    magnitude only vanishes where all three components do, at isolated nulls (floored at floor_vec).
    inverse() rescales the predicted direction to unit length.
    """
    n_channels = 10
    vel, mag = slice(2, 6), slice(6, 10)
    # Magnitudes are log values like rho and ug, so they share that group and its floor;
    # each direction is scored on its own.
    spectral_groups = (((0, 1, 2, 6), 5.0), ((3, 4, 5), 0.08), ((7, 8, 9), 0.08))

    @staticmethod
    def _magnitude_direction(x):
        """(B, 3, ...) vector -> (B, 4, ...): log10 of its magnitude, then its unit direction."""
        size = torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp(min=CONFIG["floor_vec"])
        return torch.cat([torch.log10(size), x / size], dim=1)

    @staticmethod
    def _vector(z):
        """(B, 4, ...) -> (B, 3, ...), the inverse of _magnitude_direction. The magnitude is capped
        at 1e34, as DataNormalizer's sinh(80) caps each component."""
        size = torch.pow(10, z[:, 0:1].clamp(min=np.log10(CONFIG["floor_vec"]), max=34.0))
        n = z[:, 1:4]
        return size * n / torch.linalg.vector_norm(n, dim=1, keepdim=True).clamp(min=1e-6)

    def forward(self, raw):
        rho = torch.log10(torch.clamp(raw[:, 0:1], min=CONFIG["floor_rho"]))
        ug  = torch.log10(torch.clamp(raw[:, 1:2], min=CONFIG["floor_ug"]))
        return torch.cat([rho, ug, self._magnitude_direction(raw[:, 2:5]),
                          self._magnitude_direction(raw[:, 5:8])], dim=1)

    def _decode_vectors(self, norm):
        return self._vector(norm[:, self.vel]), self._vector(norm[:, self.mag])

    def b_excess(self, norm, threshold):
        """Saturation guard: how far log10 |B| is past the |B| that arcsinh(|B|) = threshold allows."""
        return torch.relu(norm[:, self.mag.start:self.mag.start + 1] - np.log10(np.sinh(threshold)))


NORMALIZERS = {"arcsinh": DataNormalizer, "magdir": MagDirNormalizer}


def build_normalizer(name, gcov=None, gcon=None):
    """The normalizer an experiment's "normalization" setting names (NORMALIZERS)."""
    if name not in NORMALIZERS:
        raise KeyError(f"unknown normalization {name!r}; known: {', '.join(NORMALIZERS)}")
    return NORMALIZERS[name](gcov=gcov, gcon=gcon)
