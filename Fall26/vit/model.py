# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.
"""Patch-based vision transformer that forecasts the change in the GRMHD state.

Two ways to make tokens from 3D patches (token_mode):
  - "per_variable": one token per (variable, patch). Each of the 8 variables gets its own
    patch embedding and output layer, so attention relates variables explicitly; a
    192 x 96 x 96 grid with 8 x 8 x 4 patches gives 6,912 patches x 8 = 55,296 tokens.
  - "per_patch": one token per patch holding all 8 variables, as standard vision
    transformers treat colour channels. Joint quantities (rho v^2, B^2/rho) are formed in
    the embedding, and attention costs 64x less: 112 x 48 x 96 with 4 x 4 x 4 patches
    gives 8,064 tokens.
A fixed sinusoidal encoding of each patch's centre (log r, theta, phi) marks where a
token is (the 8 tokens of a patch share it in per_variable mode). A stack of pre-norm
transformer blocks mixes all tokens with full attention, and the output layers decode
every token back to its variables' values on its patch as the predicted change.

Precision: only the transformer blocks run under bf16 autocast. The embedding,
output layers and the residual stream between blocks stay fp32, so the small
predicted change is not rounded away.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from common.normalizer import NORMALIZERS


def _pad_amounts(grid, patch):
    """Cells to add at the end of each axis so it divides into whole patches."""
    return tuple((p - g % p) % p for g, p in zip(grid, patch))


def patch_centres(coord, patch):
    """Mean of a (Nr, Ntheta, Nphi) coordinate array over each patch -> (n_r, n_theta, n_phi)."""
    t = torch.as_tensor(np.asarray(coord, dtype=np.float64))[None, None]
    pr, pt, pp = _pad_amounts(t.shape[2:], patch)
    t = F.pad(t, (0, pp, 0, pt, 0, pr), mode="replicate")
    return F.avg_pool3d(t, kernel_size=patch, stride=patch)[0, 0]


def _sincos(x, freqs):
    """(n,) positions and (k,) angular frequencies -> (n, 2k) [sin | cos]."""
    angle = x[:, None] * freqs[None, :]
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)


def sinusoidal_encoding(r, theta, phi, d_model):
    """Fixed positional encoding from patch-centre coordinates, each (n_r, n_theta, n_phi).

    The d_model dimensions are split into groups of sin/cos pairs, one group per coordinate,
    so each dimension depends on one coordinate and the whole vector pins down all three.
      - phi uses whole-number harmonics 1 .. (n_phi - 1) // 2, each once, so the encoding is
        exactly periodic in phi. Harmonic n_phi / 2 is excluded: its cosine is zero at every
        evenly spaced patch centre. If there are more harmonics than a third of the pairs,
        a geometric subset is used (still each harmonic once).
      - log r and theta share the remaining pairs (r gets the odd one). They are rescaled to
        [0, 1] across the grid, and their frequencies run geometrically from pi (half a wave
        across the grid) to pi * n (one wave every two patches), n = patches along that axis.
    With 24 phi patches and d_model = 256 that is 11 phi, 59 r and 58 theta pairs.
    Returns (n_r * n_theta * n_phi, d_model), patches in row-major (r, theta, phi) order.
    """
    assert d_model % 2 == 0, "d_model must be even (sin/cos pairs)"
    pairs = d_model // 2
    n_r, n_t, n_p = r.shape

    def unit(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / torch.clamp(hi - lo, min=1e-12)

    def geometric(lo, hi, k):
        return torch.logspace(np.log10(lo), np.log10(max(hi, lo)), k, dtype=torch.float64)

    m_max = max(1, (n_p - 1) // 2)
    if m_max <= pairs // 3:
        m_p = torch.arange(1, m_max + 1, dtype=torch.float64)
    else:
        m_p = torch.unique(torch.round(geometric(1.0, m_max, pairs // 3)))
    k_r = -(-(pairs - len(m_p)) // 2)
    k_t = pairs - len(m_p) - k_r

    u_r = unit(torch.log(r)).reshape(-1)
    u_t = unit(theta).reshape(-1)
    f_r = np.pi * geometric(1.0, n_r, k_r)
    f_t = np.pi * geometric(1.0, n_t, k_t)
    enc = torch.cat([_sincos(u_r, f_r),
                     _sincos(u_t, f_t),
                     _sincos(phi.reshape(-1), m_p)], dim=1)
    return enc.float()


class Block(nn.Module):
    """Pre-norm transformer block: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, d_model, n_heads, mlp_ratio):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, mlp_ratio * d_model),
                                 nn.GELU(),
                                 nn.Linear(mlp_ratio * d_model, d_model))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = (self.qkv(self.norm1(x))
                   .view(B, T, 3, self.n_heads, D // self.n_heads)
                   .permute(2, 0, 3, 1, 4))
        attn = F.scaled_dot_product_attention(q, k, v)       # memory-efficient / flash kernel
        x = x + self.proj(attn.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.norm2(x))


class ViTForecaster(nn.Module):
    """Predicts the normalized change Δ for one step; the caller adds it to the input.

    coords: (r, theta, phi) arrays of shape (Nr, Ntheta, Nphi) from geometry.npz
            ('r', 'h', 'ph'); they fix the grid size and the positional encoding.
    token_mode: "per_variable" or "per_patch" (module docstring). Both are the same code with
            G token groups of n_vars // G variables each: G = n_vars or G = 1.
    """

    def __init__(self, coords, n_vars=8, patch=(8, 8, 8), d_model=256, depth=12,
                 n_heads=8, mlp_ratio=4, use_amp=True, grad_checkpoint=True, token_mode="per_variable"):
        super().__init__()
        assert token_mode in ("per_variable", "per_patch"), f"unknown token_mode {token_mode!r}"
        r, theta, phi = coords
        self.grid = tuple(np.shape(r))
        self.n_vars, self.patch, self.d_model = n_vars, tuple(patch), d_model
        self.token_mode = token_mode
        self.groups = n_vars if token_mode == "per_variable" else 1     # tokens per patch
        self.vars_per_token = n_vars // self.groups
        self.pad = _pad_amounts(self.grid, self.patch)
        self.use_amp, self.grad_checkpoint = use_amp, grad_checkpoint

        # One Conv3d(vars_per_token, d_model) per token group, run as a single grouped convolution
        self.embed = nn.Conv3d(n_vars, self.groups * d_model, kernel_size=self.patch,
                               stride=self.patch, groups=self.groups)
        pos = sinusoidal_encoding(patch_centres(r, self.patch), patch_centres(theta, self.patch),
                                  patch_centres(phi, self.patch), d_model)
        self.register_buffer("pos", pos, persistent=False)   # rebuilt from geometry, not saved

        self.blocks = nn.ModuleList(Block(d_model, n_heads, mlp_ratio) for _ in range(depth))
        self.norm = nn.LayerNorm(d_model)

        # One Linear(d_model, variables x patch volume) per token group; zero-initialized so Δ = 0 at start
        vol = self.vars_per_token * int(np.prod(self.patch))
        self.out_weight = nn.Parameter(torch.zeros(self.groups, d_model, vol))
        self.out_bias = nn.Parameter(torch.zeros(self.groups, vol))

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.embed.weight, std=0.02)
        nn.init.zeros_(self.embed.bias)

    @property
    def n_tokens(self):
        return self.pos.shape[0] * self.groups

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C = x.shape[:2]
        assert C == self.n_vars and tuple(x.shape[2:]) == self.grid, \
            f"expected (B, {self.n_vars}, {self.grid}), got {tuple(x.shape)}"
        pr, pt, pp = self.pad
        x = F.pad(x, (0, pp, 0, pt, 0, pr))

        G = self.groups
        h = self.embed(x)                                     # (B, G*d, n_r, n_t, n_p)
        n_r, n_t, n_p = h.shape[2:]
        n = n_r * n_t * n_p
        h = h.view(B, G, self.d_model, n).transpose(2, 3)     # (B, G, n, d)
        h = (h + self.pos).reshape(B, G * n, self.d_model)    # tokens: group-major

        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            for blk in self.blocks:
                if self.grad_checkpoint and self.training:
                    h = checkpoint(blk, h, use_reentrant=False)
                else:
                    h = blk(h)

        h = self.norm(h.float()).view(B, G, n, self.d_model)
        out = torch.einsum("bgnd,gdv->bgnv", h, self.out_weight) + self.out_bias[:, None]
        Pr, Pt, Pp = self.patch
        # (B, G, n, vars_per_token * vol) -> (B, C, n, vol): channel c = g * vars_per_token + v
        out = (out.view(B, G, n, self.vars_per_token, Pr * Pt * Pp).transpose(2, 3)
               .reshape(B, C, n, Pr * Pt * Pp))
        out = (out.view(B, C, n_r, n_t, n_p, Pr, Pt, Pp)
               .permute(0, 1, 2, 5, 3, 6, 4, 7)
               .reshape(B, C, n_r * Pr, n_t * Pt, n_p * Pp))
        Nr, Nt, Np = self.grid
        return out[:, :, :Nr, :Nt, :Np]


def build_model(cfg, coords):
    """The experiment's model. Its channel count is the normalization's: 8 for "arcsinh", 10 for "magdir"."""
    return ViTForecaster(coords, n_vars=NORMALIZERS[cfg["normalization"]].n_channels,
                         patch=cfg["patch"], d_model=cfg["d_model"],
                         depth=cfg["depth"], n_heads=cfg["n_heads"], mlp_ratio=cfg["mlp_ratio"],
                         use_amp=cfg["use_amp"], grad_checkpoint=cfg["grad_checkpoint"],
                         token_mode=cfg["token_mode"])
