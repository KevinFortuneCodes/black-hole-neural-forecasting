# Copyright (c) 2026 Kevin Fortune. All rights reserved, except for portions adapted from
# code by Sachin Venkatesh (the TRON model), which remain his. See LICENSE.
"""TRON's physics-informed training loss.

Copied from uno_simple_ds_v3.py (sachin branch, fall26/, commit ca0288555c99).
Changes from the original:
  - CONFIG is the shared PHYSICS settings dict (common/settings.py).
  - The geometry path is a constructor argument instead of CONFIG["geometry_file"].
  - The grid-mismatch warning goes through `logging`.
  - The div B term is the mean absolute difference from the truth's div B, not mean |div B| of
    the prediction. calc_divB uses central differences of cell-centred B, so even the true data
    is not divergence-free by it (the simulation's zero divergence holds for face-centred B with
    its own stencil); penalizing |div B| pulled predictions away from the data.
  - The mass-shell term weights every radial shell equally. TRON gated each shell by
    M_true(r) / max M_true, but shell mass grows roughly like r^3 (sqrt(-g) with x1 = ln r), so
    the gate gave the outermost shells nearly all the weight (rc300 at r ~ 1,050: 5e-3 at r = 100,
    half the weight beyond r ~ 780). Shells are compared by |log M_pred - log M_true|, a relative
    error at every mass (TRON's signed log1p is nearly absolute for M < 1).
  - The predicted u^0 in the mass shells comes from the Lorentz factor after the max_gamma clamp,
    matching the clamped velocities in the decoded prediction (TRON used the pre-clamp factor).
  - forward() returns the loss components as detached tensors rather than Python floats, so a
    training step does not wait on the GPU once per component; callers convert when they log.
  - The normalized channels are read through the normalizer's layout (vel, mag, spectral_groups,
    b_excess), so the loss works with either normalization in common/normalizer.py. With
    DataNormalizer every term is computed exactly as before.
The other loss terms, the weights and the curriculum are unchanged.
"""
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalizer import load_geometry
from .settings import PHYSICS as CONFIG

logger = logging.getLogger(__name__)


class PhysicsLoss3D(nn.Module):
    def __init__(self, device, geometry_file):
        super().__init__()
        geo = load_geometry(geometry_file, device)
        self.register_buffer('gdet', geo['gdet'])
        self.register_buffer('gcov', geo['gcov'])
        self.register_buffer('gcon', geo['gcon'])
        assert self.gcov.shape[:3] == (1, 1, 4), f"gcov must be (1,1,4,4,Nr,Ntheta,Nphi), got {tuple(self.gcov.shape)}"
        assert self.gcon.shape[:3] == (1, 1, 4), f"gcon must be (1,1,4,4,Nr,Ntheta,Nphi), got {tuple(self.gcon.shape)}"
        assert self.gdet.shape[:2] == (1, 1), f"gdet must be (1,1,Nr,Ntheta,Nphi), got {tuple(self.gdet.shape)}"
        self.dx, self.dy, self.dz = float(geo['dx'][0]), float(geo['dx'][1]), float(geo['dx'][2])

        self.eos_gamma = CONFIG.get("eos_gamma") or geo.get("eos_gamma")
        self.dt_dump = CONFIG.get("dt_dump") or geo.get("dt_dump")
        self._grid_warned = False

        # Hann window (r x theta) for the spectral loss — suppresses Fourier-Gibbs
        # artifacts on non-periodic r/theta axes (phi is periodic and is NOT windowed).
        nr, ntheta = self.gdet.shape[-3], self.gdet.shape[-2]
        win = torch.hann_window(nr, dtype=torch.float32, device=device).view(nr, 1, 1) \
              * torch.hann_window(ntheta, dtype=torch.float32, device=device).view(1, ntheta, 1)
        self.register_buffer('spec_win', win.unsqueeze(0).unsqueeze(0))  # (1,1,Nr,Ntheta,1)

        # Precomputed static geometric buffers (eliminates redundant pow/sqrt/tensor-casting on every forward pass)
        alpha_d = 1.0 / torch.sqrt(-self.gcon[0, 0, 0, 0].double() + 1e-300)
        self.register_buffer('alpha_d', alpha_d)
        self.register_buffer('gam_sp_d', self.gcov[0, 0, 1:, 1:].double())
        l_local_d = torch.pow(self.gdet.double() * self.dx * self.dy * self.dz, 1.0 / 3.0)
        self.register_buffer('l_local_d', l_local_d)

    @staticmethod
    def _safe_rho_ug(raw, floor, ceil):
        """Same clamp path as normalizer inverse: raw -> 10^(clamp(log10(raw), log10(floor), log10(ceil)))."""
        return torch.pow(10, torch.clamp(torch.log10(torch.clamp(raw, min=1e-300)),
                                         np.log10(floor), np.log10(ceil)))

    @staticmethod
    def _signed_log(x):
        """Signed-log transform sl(x) = sign(x)*log1p(|x|) for the shell terms."""
        x = x.double()
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def calc_divB(self, b1, b2, b3):
        f1, f2, f3 = self.gdet.double()*b1.double(), self.gdet.double()*b2.double(), self.gdet.double()*b3.double()

        # 1. Radial Derivative (Replicate)
        f1_pad = F.pad(f1, (0,0,0,0,1,1), mode='replicate')
        d1 = (f1_pad[:,:,2:] - f1_pad[:,:,:-2]) / (2*self.dx)

        # 2. Theta Derivative (Open Outflow Boundary: row-copy, no flip)
        f2_pad = torch.cat([f2[:, :, :, 0:1, :], f2, f2[:, :, :, -1:, :]], dim=3)
        d2 = (f2_pad[:, :, :, 2:] - f2_pad[:, :, :, :-2]) / (2*self.dy)

        # 3. Phi Derivative (Circular)
        f3_pad = F.pad(f3, (1,1,0,0,0,0), mode='circular')
        d3 = (f3_pad[:,:,:,:,2:] - f3_pad[:,:,:,:,:-2]) / (2*self.dz)

        return (d1 + d2 + d3) / (self.gdet + 1e-20)

    def calc_gradient_loss(self, pred, target):
        """Spatial cell-to-cell gradient loss on log-density to suppress Nyquist checkerboard modes."""
        p_dr = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        t_dr = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        p_dh = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        t_dh = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        p_dp = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        t_dp = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        return F.l1_loss(p_dr, t_dr) + F.l1_loss(p_dh, t_dh) + F.l1_loss(p_dp, t_dp)

    def spectral_loss(self, pred, target, groups):
        """Windowed relative 3D Fourier magnitude loss grouped by physical domain.
        `groups` is the normalizer's spectral_groups: (channels, denominator floor) pairs. For
        DataNormalizer:
        - Thermodynamics (rho, ug): channels 0:2
        - Velocity vector (u^r, u^theta, u^phi): channels 2:5
        - Magnetic vector (B^r, B^theta, B^phi): channels 5:8
        Each vector field shares a single denominator across its components, preserving
        vector isotropy and eliminating weak-component (B_theta) gradient spikes while
        ensuring B and v receive equal spectral weighting with thermodynamics.
        Floors audited across the entire 5,700-dump dataset (training + blind):
          th: 5.0  (observed steady-state 26.0–34.5; min overall 20.7)
          v : 0.08 (observed steady-state 0.40–0.75; guards early-onset transient min 0.043)
          b : 0.03 (observed steady-state 0.17–0.73; guards t=0 seed torus transient min 0.0004)
        """
        p_fft = torch.fft.rfftn((pred * self.spec_win).to(torch.float32), dim=[-3,-2,-1])
        t_fft = torch.fft.rfftn((target * self.spec_win).to(torch.float32), dim=[-3,-2,-1])
        diff = torch.abs(torch.abs(p_fft) - torch.abs(t_fft))
        t_abs = torch.abs(t_fft)

        # Each group shares one denominator across its channels; groups are weighted equally
        losses = [torch.mean(diff[:, list(ch)]) / torch.clamp(torch.mean(t_abs[:, list(ch)]), min=floor)
                  for ch, floor in groups]
        return sum(losses) / len(losses)

    def calc_saturation_penalty(self, pred_norm, normalizer):
        """Soft penalty for normalized rho/ug at inverse-clamp floors/ceilings,
        and B exceeding the saturation guard (the normalizer's b_excess)."""
        rho_viol = F.relu(np.log10(CONFIG["floor_rho"]) - pred_norm[:, 0]) \
                   + F.relu(pred_norm[:, 0] - np.log10(CONFIG["ceiling_rho"]))
        ug_viol = F.relu(np.log10(CONFIG["floor_ug"]) - pred_norm[:, 1]) \
                   + F.relu(pred_norm[:, 1] - np.log10(CONFIG["ceiling_ug"]))
        b_viol = normalizer.b_excess(pred_norm, CONFIG.get("b_mag_threshold", 5.0))
        return rho_viol.mean() + ug_viol.mean() + b_viol.mean()

    def _mass_shell(self, rho_raw, u0_d):
        """Fast angular-mean rest-mass M(r) = (1/N_ang) sum_{theta,phi} rho * u^0 * sqrt(-g) in float64."""
        rho = self._safe_rho_ug(rho_raw.double(), CONFIG["floor_rho"], CONFIG["ceiling_rho"])
        n_ang = float(rho.shape[3] * rho.shape[4])
        return torch.sum(rho * u0_d * self.gdet.double(), dim=(3, 4)) / n_ang

    def forward(self, pred_norm, target_raw, normalizer, epoch, return_items=False):
        if not self._grid_warned:
            ng = self.gcov.shape[-3:]
            nb = pred_norm.shape[2:]
            geom_ok = (ng[0], ng[1]) == (nb[0], nb[1]) and (ng[2] == 1 or ng[2] == nb[2])
            if not geom_ok:
                logger.warning(f"batch grid {tuple(nb)} vs geometry grid {tuple(ng)} "
                               "(geometry must match in Nr/Ntheta and have Nphi == 1 [axisym] "
                               "or Nphi == batch Nphi)")
            self._grid_warned = True

        target_norm = normalizer(target_raw)
        fluid, mag = slice(0, normalizer.mag.start), normalizer.mag

        # 1. Fluid Reconstruction (rho, ug and velocity channels: 0:5 for DataNormalizer)
        w_fluid = 1.0 + 2.0 * torch.sigmoid(target_norm[:,0:1])
        loss_fluid = torch.mean(w_fluid * torch.abs(pred_norm[:, fluid] - target_norm[:, fluid]))

        # 2. Magnetic Reconstruction (5:8 for DataNormalizer, Inverse Density Weighting to protect funnel)
        w_mag = 1.0 + 2.0 * (1.0 - torch.sigmoid(target_norm[:,0:1]))
        loss_mag = torch.mean(w_mag * torch.abs(pred_norm[:, mag] - target_norm[:, mag]))

        # 3. Spatial Gradient Stiffness (Split)
        # Smooth scalar thermodynamics (rho, ug) get proven 0.10 anchor
        loss_grad_smooth = self.calc_gradient_loss(pred_norm[:, 0:2], target_norm[:, 0:2])
        # Turbulent vector fields (u^i, B^i) get gentle 0.03 stiffness to close 2*dx null-space
        loss_grad_turb = self.calc_gradient_loss(pred_norm[:, 2:], target_norm[:, 2:])

        # 4. Grouped Vector Spectral Loss (all channels, balanced across the normalizer's groups)
        loss_spec = self.spectral_loss(pred_norm, target_norm, normalizer.spectral_groups)

        # 5. Decoded Physics & Exact Clamps (reusing gamma_rec for zero redundant contractions)
        rec, gamma_rec = normalizer.inverse(pred_norm, return_gamma=True)
        # gamma_rec is pre-clamp; rec's velocities are scaled to at most max_gamma
        u0_rec_d = torch.clamp(gamma_rec.double(), max=CONFIG["max_gamma"]) / self.alpha_d
        u0_tru_d = normalizer.u0_of(target_raw[:, 2:5]).double()

        # 6. Solenoidal consistency: match the truth's discrete div B (not zero; see module docstring)
        rec_b = rec[:, 5:8]  # (B,3,Nr,Nt,Np)
        divb = self.calc_divB(rec_b[:, 0:1], rec_b[:, 1:2], rec_b[:, 2:3])
        divb_true = self.calc_divB(target_raw[:, 5:6], target_raw[:, 6:7], target_raw[:, 7:8])
        loss_divb = torch.mean(torch.abs(divb - divb_true))

        # 7. Rest-mass shell profile, every shell weighted equally (no truth-mass gate; see module docstring)
        m_rec_shell = self._mass_shell(rec[:, 0:1], u0_rec_d)
        m_tru_shell = self._mass_shell(target_raw[:, 0:1], u0_tru_d)
        loss_mass = torch.mean(torch.abs(torch.log(torch.clamp(m_rec_shell, min=1e-300))
                                         - torch.log(torch.clamp(m_tru_shell, min=1e-300))))

        # 8. Guards (Saturation and Pre-Clamp Gamma Hinge)
        overshoot = F.relu(gamma_rec.double() / CONFIG["max_gamma"] - 1.0)
        loss_gamma = torch.mean(torch.log1p(overshoot))
        loss_sat = self.calc_saturation_penalty(pred_norm, normalizer)

        # Curriculum Schedule (Recon -> Ramp -> Stable)
        recon_only_epochs = CONFIG.get("recon_only_epochs", 50)
        ramp_epochs = CONFIG.get("curriculum_ramp_epochs", 100)
        if epoch < recon_only_epochs:
            w_phys = 0.0
        else:
            w_phys = min(1.0, (epoch - recon_only_epochs) / ramp_epochs)

        w_data = CONFIG.get("loss_weight_data", 1.0)
        w_grad_sm = CONFIG.get("loss_weight_grad_smooth", 0.10)
        w_grad_tb = CONFIG.get("loss_weight_grad_turb", 0.03)
        w_spec = CONFIG.get("loss_weight_spec", 0.01)
        w_divb = CONFIG.get("loss_weight_divb", 0.10)
        w_mass = CONFIG.get("loss_weight_mass", 0.03)
        w_guard = CONFIG.get("loss_weight_guards", 0.10)

        weighted = [
            (loss_fluid,       w_data),                 # 0  fluid reconstruction (normalized)
            (loss_mag,         w_data),                 # 1  mag reconstruction (normalized)
            (loss_grad_smooth, w_grad_sm),              # 2  smooth scalar spatial gradient stiffness (rho, ug)
            (loss_grad_turb,   w_grad_tb),              # 3  turbulent vector spatial gradient stiffness (u^i, B^i)
            (loss_spec,        w_spec),                 # 4  all-channel grouped spectral loss (unconditional)
            (loss_sat,         w_guard),                # 5  guard: saturation floors/ceilings
            (loss_divb,        w_divb * w_phys),        # 6  div B error vs the truth's div B (curriculum-gated)
            (loss_mass,        w_mass * w_phys),        # 7  rest-mass shell log profile (curriculum-gated)
            (loss_gamma,       w_guard),                # 8  guard: Gamma pre-clamp hinge
        ]
        total = sum((l.double() * w for l, w in weighted), torch.tensor(0.0, dtype=torch.float64, device=pred_norm.device))

        # Detached tensors, not floats: .item() here would stall every step on the GPU
        result = {
            "Fluid": loss_fluid.detach(), "Mag": loss_mag.detach(),
            "Grad_Sm": loss_grad_smooth.detach(), "Grad_Tb": loss_grad_turb.detach(),
            "Spec": loss_spec.detach(),
            "Saturation": loss_sat.detach(), "DivB": loss_divb.detach(),
            "Mass": loss_mass.detach(), "Gamma": loss_gamma.detach(),
            "Curriculum": w_phys
        }
        if return_items:
            return total, result, [l * w for l, w in weighted]
        return total, result
