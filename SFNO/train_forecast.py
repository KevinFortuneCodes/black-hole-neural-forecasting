# Copyright (c) 2026 Kevin Fortune. All rights reserved, except for portions adapted from
# code by Sachin Venkatesh (the Hybrid U-NO this model is built on), which remain his. See LICENSE.
#2+1D residual pred 4/15/26
import os
import glob
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from torch_harmonics import RealSHT, InverseRealSHT

# --- DDP IMPORTS ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# --- CONFIGURATION ---
CONFIG = {
    "data_dir": "/path/to/sfno_data/processed_forecast/",
    "geometry_file": "/path/to/sfno_data/processed_forecast/geometry.npz",
    "checkpoint_dir": "/path/to/sfno_data/processed_forecast/checkpoints",
    "log_dir": "/path/to/sfno_data/processed_forecast/logs",

    "batch_size":8,
    "learning_rate": 3e-4,
    "epochs": 200,
    "patience": 30,
    "grad_clip": 1.0,
    
    "unroll_steps": 2,
    "use_amp": False,

    # Input grid angular dimensions — fill in from your HAMR run (theta, phi cell counts)
    "grid_nlat": 48,
    "grid_nlon": 96,

    # Physics Constraints
    "max_gamma": 50.0,
    "floor_rho": 1e-20,
    "ceiling_rho":1e20,
    "floor_ug": 1e-22,
    "ceiling_ug":1e20,

    # ~~~ CHANGE 1: Loss tier weights and calibration settings ~~~
    "w_physics":     5.0,   # tier weight for divB, mass, gamma (most important)
    "w_primary":     1.0,   # tier weight for fluid, mag
    "w_regularizer": 0.1,   # tier weight for grad, spectral (least important)
    "calib_batches": 75,    # forward-only batches used to measure initial loss scales
    "norm_cap":      5.0,   # per-component normalized loss cap (prevents explosion)
    # ~~~ END CHANGE 1 ~~~
}

# --- SAFE LOGGING (Only Rank 0 creates folders and logs) ---
local_rank = int(os.environ.get("LOCAL_RANK", 0))

def log_print(msg):
    rank = dist.get_rank() if dist.is_initialized() else local_rank
    if rank == 0:
        print(msg)
        logging.info(msg)

# --- UNROLLED DATASET & NORMALIZER ---
class HAMRForecastDataset(Dataset):
    def __init__(self, files, unroll_steps=2):
        self.files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        self.unroll_steps = unroll_steps
        
    def __len__(self): 
        return len(self.files) - self.unroll_steps

    def __getitem__(self, i):
        sequence =[]
        for step in range(self.unroll_steps + 1):
            sequence.append(torch.tensor(np.load(self.files[i + step]), dtype=torch.float32))
        return sequence 

class DataNormalizer(nn.Module):
    """Universal Scale-Invariant Normalizer using SymLog (ArcSinh) for Vectors"""
    def __init__(self):
        super().__init__()
    def forward(self, raw):
        norm = torch.zeros_like(raw)
        norm[:,0] = torch.log10(torch.clamp(raw[:,0], min=CONFIG["floor_rho"]))
        norm[:,1] = torch.log10(torch.clamp(raw[:,1], min=CONFIG["floor_ug"]))
        norm[:,2:5] = torch.arcsinh(raw[:,2:5])
        norm[:,5:8] = torch.arcsinh(raw[:,5:8])
        return norm
        
    def inverse(self, norm):
        raw = torch.zeros_like(norm)
        safe_rho = torch.clamp(norm[:,0], min=np.log10(CONFIG["floor_rho"]), max=np.log10(CONFIG["ceiling_rho"]))
        safe_ug  = torch.clamp(norm[:,1], min=np.log10(CONFIG["floor_ug"]), max=np.log10(CONFIG["ceiling_ug"]))
        
        raw[:,0] = torch.pow(10, safe_rho)
        raw[:,1] = torch.pow(10, safe_ug)
        raw[:,2:5] = torch.sinh(norm[:,2:5])
        raw[:,5:8] = torch.sinh(norm[:,5:8])
        return raw

# --- PHYSICS LOSS ---
class PhysicsLoss3D(nn.Module):
    def __init__(self, device):
        super().__init__()
        geo = np.load(CONFIG["geometry_file"])
        self.register_buffer('gdet', torch.tensor(geo['gdet'], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0))
        self.register_buffer('gcov', torch.tensor(geo['gcov'], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0))
        self.dx, self.dy, self.dz = float(geo['dx'][0]), float(geo['dx'][1]), float(geo['dx'][2])

        # ~~~ CHANGE 2: Replace learned log_vars with static calibration buffers ~~~
        # (7 components in order: fluid, mag, grad, spec, divB, mass, gamma)
        self.register_buffer('loss_scales', torch.ones(7, dtype=torch.float32))
        self.register_buffer('scales_initialized', torch.tensor(False))
        # ~~~ END CHANGE 2 ~~~

    def calc_divB(self, b1, b2, b3):
        f1, f2, f3 = self.gdet*b1, self.gdet*b2, self.gdet*b3
        
        # 1. Radial Derivative (Replicate)
        f1_pad = F.pad(f1, (0,0,0,0,1,1), mode='replicate')
        d1 = (f1_pad[:,:,2:] - f1_pad[:,:,:-2]) / (2*self.dx)
        
        # 2. Theta Derivative (Polar Padding)
        shift = f2.shape[-1] // 2
        top = torch.flip(torch.roll(f2[:,:,:,0:1,:], shifts=shift, dims=-1), dims=[3])
        bot = torch.flip(torch.roll(f2[:,:,:,-1:,:], shifts=shift, dims=-1), dims=[3])
        # CRITICAL: -top and -bot because B^theta flips direction over the pole
        f2_pad = torch.cat([-top, f2, -bot], dim=3)
        d2 = (f2_pad[:,:,:,2:] - f2_pad[:,:,:,:-2]) / (2*self.dy)
        
        # 3. Phi Derivative (Circular)
        f3_pad = F.pad(f3, (1,1,0,0,0,0), mode='circular')
        d3 = (f3_pad[:,:,:,:,2:] - f3_pad[:,:,:,:,:-2]) / (2*self.dz)
        
        return (d1 + d2 + d3) / (self.gdet + 1e-20)

    def calc_lorentz_penalty(self, u1, u2, u3):
        g11, g22, g33 = self.gcov[:,:,1,1], self.gcov[:,:,2,2], self.gcov[:,:,3,3]
        g12, g13, g23 = self.gcov[:,:,1,2], self.gcov[:,:,1,3], self.gcov[:,:,2,3]
        u_sq = (g11*u1**2) + (g22*u2**2) + (g33*u3**2) + 2*(g12*u1*u2) + 2*(g13*u1*u3) + 2*(g23*u2*u3)
        gamma = torch.sqrt(1.0 + torch.clamp(u_sq, min=0.0))
        # softplus instead of relu: smooth gradient signal before the threshold is crossed.
        # The 1/(2*gamma) from the sqrt derivative provides natural self-limiting for large violations.
        return torch.mean(F.softplus(gamma - CONFIG["max_gamma"]))

    def spectral_loss(self, pred, target):
        p_fft = torch.fft.rfftn(pred.to(torch.float32), dim=[-3,-2,-1])
        t_fft = torch.fft.rfftn(target.to(torch.float32), dim=[-3,-2,-1])
        p_mag = (p_fft.real**2 + p_fft.imag**2).sqrt()
        t_mag = (t_fft.real**2 + t_fft.imag**2).sqrt()
        return F.l1_loss(p_mag, t_mag)

    def calc_gradient_loss(self, pred, target):
        p_dr = pred[:,:,1:,:,:] - pred[:,:,:-1,:,:]
        t_dr = target[:,:,1:,:,:] - target[:,:,:-1,:,:]
        p_dh = pred[:,:,:,1:,:] - pred[:,:,:,:-1,:]
        t_dh = target[:,:,:,1:,:] - target[:,:,:,:-1,:]
        p_dp = pred[:,:,:,:,1:] - pred[:,:,:,:,:-1]
        t_dp = target[:,:,:,:,1:] - target[:,:,:,:,:-1]
        return F.l1_loss(p_dr, t_dr) + F.l1_loss(p_dh, t_dh) + F.l1_loss(p_dp, t_dp)

    def forward(self, pred_norm, target_raw, normalizer, epoch, total_epochs, return_raw=False):
        target_norm = normalizer(target_raw)
        
        # 1. Fluid (Channels 0:5)
        w_fluid = 1.0 + 2.0 * torch.sigmoid(target_norm[:,0:1]) 
        loss_fluid = torch.mean(w_fluid * torch.abs(pred_norm[:,0:5] - target_norm[:,0:5]))
        
        # 2. Mag (Channels 5:8, Inverse Density Weighting to protect funnel)
        w_mag = 1.0 + 2.0 * (1.0 - torch.sigmoid(target_norm[:,0:1]))
        loss_mag = torch.mean(w_mag * torch.abs(pred_norm[:,5:8] - target_norm[:,5:8]))
        
        # 3. Gradient & Spectral (Computed on Log Density)
        loss_grad = self.calc_gradient_loss(pred_norm[:,0:1], target_norm[:,0:1])
        loss_spec = self.spectral_loss(pred_norm[:,0:1], target_norm[:,0:1])
        
        # 4. Physics
        rec = normalizer.inverse(pred_norm)
        loss_divb = torch.mean(torch.abs(self.calc_divB(rec[:,5:6], rec[:,6:7], rec[:,7:8])))
        
        m_rec = torch.sum(rec[:,0:1] * self.gdet, dim=(2,3,4))
        m_tru = torch.sum(target_raw[:,0:1] * self.gdet, dim=(2,3,4))
        loss_mass = torch.mean(torch.abs(m_rec - m_tru) / (m_tru + 1e-10))
        loss_gamma = self.calc_lorentz_penalty(rec[:,2:3], rec[:,3:4], rec[:,4:5])
        
        # ~~~ CHANGE 3: Calibrated tier-weighted loss (replaces homoscedastic) ~~~
        # Curriculum ramps physics from 0 → 1 over first THIRD of training (was first half)
        prog = epoch / total_epochs
        w_phys = min(1.0, prog * 3.0)

        raw_losses = torch.stack([
            loss_fluid, loss_mag, loss_grad, loss_spec,
            loss_divb, loss_mass, loss_gamma
        ])

        tier_weights = torch.tensor([
            CONFIG["w_primary"],     # fluid
            CONFIG["w_primary"],     # mag
            CONFIG["w_regularizer"], # grad
            CONFIG["w_regularizer"], # spec
            CONFIG["w_physics"],     # divB
            CONFIG["w_physics"],     # mass
            CONFIG["w_physics"],     # gamma
        ], device=raw_losses.device, dtype=raw_losses.dtype)

        curriculum = torch.tensor(
            [1., 1., 1., 1., w_phys, w_phys, w_phys],
            device=raw_losses.device, dtype=raw_losses.dtype
        )

        normalized = torch.clamp(raw_losses / self.loss_scales, max=CONFIG["norm_cap"])
        total = (tier_weights * curriculum * normalized).sum()
        # ~~~ END CHANGE 3 ~~~

        components = {
            "Fluid": loss_fluid.item(), "Mag": loss_mag.item(),
            "Grad": loss_grad.item(), "Spec": loss_spec.item(),
            "DivB": loss_divb.item(), "Mass": loss_mass.item(),
            "Gamma": loss_gamma.item(),
        }
        if return_raw:
            return total, components, raw_losses
        return total, components

# --- HYBRID (2+1)D U-NO ---
class Conv2Plus1D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        mid_c = (in_c * 3 * 3 * 3) // (3 * 3 + 3)
        # Internal convolutions use padding=0, padding is handled externally
        self.spatial = nn.Conv3d(in_c, mid_c, kernel_size=(1,3,3), stride=(1,stride,stride), padding=0)
        self.bn1 = nn.GroupNorm(8, mid_c)
        self.depth = nn.Conv3d(mid_c, out_c, kernel_size=(3,1,1), stride=(stride,1,1), padding=0)
        self.bn2 = nn.GroupNorm(8, out_c)
        self.act = nn.SiLU()

    def forward(self, x):
        # 1. Spatial Padding (Theta, Phi)
        x_spat = F.pad(x, (1, 1, 0, 0, 0, 0), mode='circular') # Phi
        shift = (x_spat.shape[-1] - 2) // 2 
        top = torch.flip(torch.roll(x_spat[:, :, :, 0:1, :], shifts=shift, dims=-1), dims=[3])
        bot = torch.flip(torch.roll(x_spat[:, :, :, -1:, :], shifts=shift, dims=-1), dims=[3])
        x_spat = torch.cat([top, x_spat, bot], dim=3) # Theta
        
        x_mid = self.act(self.bn1(self.spatial(x_spat)))
        
        # 2. Depth Padding (R)
        x_dep = F.pad(x_mid, (0, 0, 0, 0, 1, 1), mode='replicate')
        return self.act(self.bn2(self.depth(x_dep)))

class ResidualBlock2Plus1D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = Conv2Plus1D(c, c)
        self.c2 = Conv2Plus1D(c, c)
    def forward(self, x): return x + self.c2(self.c1(x))

class SphericalRadialConv(nn.Module):
    """
    Bottleneck spectral layer for a spherical (r, θ, φ) domain.

    Angular dims (θ, φ) are handled by RealSHT from torch-harmonics, giving exact
    spherical harmonic coefficients.  The radial dim (r) is handled by a full complex
    FFT applied to the (already complex) SHT coefficient arrays — rfft would be wrong
    here because SHT coefficients for m > 0 are complex-valued.

    Learned weights mix channels in the joint (k_r, l, m) spectral space via einsum.
    """
    def __init__(self, in_channels, out_channels, nlat_bot, nlon_bot,
                 lmax=8, mmax=8, radial_modes=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lmax = lmax
        self.mmax = mmax
        self.radial_modes = radial_modes

        self.sht  = RealSHT(nlat_bot, nlon_bot, lmax=lmax, mmax=mmax,
                             grid="equiangular", norm="ortho")
        self.isht = InverseRealSHT(nlat_bot, nlon_bot, lmax=lmax, mmax=mmax,
                                   grid="equiangular", norm="ortho")

        scale = 1.0 / (in_channels * out_channels)
        # Weights in (C_in, C_out, k_r, l, m) spectral space, stored as re/im pair
        self.weights_re = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, radial_modes, lmax, mmax))
        self.weights_im = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, radial_modes, lmax, mmax))

    def forward(self, x):
        B, C, D, H, W = x.shape

        # SHT over angular dims: (B·C·D, H, W) → (B·C·D, lmax, mmax) complex
        x_sht = self.sht(x.float().reshape(B * C * D, H, W))
        x_sht = x_sht.reshape(B, C, D, self.lmax, self.mmax)   # complex

        # Full FFT along radial dim (SHT coefficients are complex for m > 0)
        x_spec = torch.fft.fft(x_sht, dim=2)   # (B, C, D, lmax, mmax) complex

        # Truncated channel mixing in spectral space
        k_r = min(self.radial_modes, D)
        weights = torch.complex(self.weights_re[:, :, :k_r],
                                self.weights_im[:, :, :k_r])   # (C_in, C_out, k_r, lmax, mmax)
        out_spec = torch.zeros(B, self.out_channels, D, self.lmax, self.mmax,
                               dtype=torch.cfloat, device=x.device)
        out_spec[:, :, :k_r] = torch.einsum('biklm,ioklm->boklm',
                                             x_spec[:, :, :k_r], weights)

        # IFFT → (B, C_out, D, lmax, mmax) complex SHT coefficients
        x_r = torch.fft.ifft(out_spec, dim=2)

        # ISHT: (B·C_out·D, lmax, mmax) complex → (B·C_out·D, H, W) real
        out = self.isht(x_r.reshape(B * self.out_channels * D, self.lmax, self.mmax))
        return out.reshape(B, self.out_channels, D, H, W).to(dtype=x.dtype)

class SFNOBlock(nn.Module):
    """
    SFNO block per Figure 2 of Bonev et al. (2023).

    Two parallel paths are summed then normalized:
        u' = InstanceNorm( SphericalRadialConv(u) + MLP(u) )

    The MLP is point-wise (1×1×1 convs) so it only mixes channels, not spatial
    locations — matching the paper's spatial-domain channel-mixing branch.
    InstanceNorm is used (not GroupNorm) to preserve equivariance, per the paper.
    """
    def __init__(self, channels, nlat, nlon, lmax=8, mmax=8, radial_modes=8):
        super().__init__()
        self.spectral = SphericalRadialConv(channels, channels, nlat, nlon,
                                            lmax=lmax, mmax=mmax, radial_modes=radial_modes)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, channels, 1),
            nn.GELU(),
            nn.Conv3d(channels, channels, 1),
        )
        # Zero-init the output projection so the MLP starts as a no-op.
        # This lets the spectral path receive meaningful gradient signal from step 1.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.norm = nn.InstanceNorm3d(channels, affine=True)

    def forward(self, x):
        return self.norm(self.spectral(x) + self.mlp(x))


class Hybrid_UNO_3D(nn.Module):
    def __init__(self, in_c, nlat, nlon):
        super().__init__()
        self.inc = nn.Sequential(nn.Conv3d(in_c+3, 32, 1), ResidualBlock2Plus1D(32))
        self.down1 = nn.Sequential(Conv2Plus1D(32, 64, stride=2), ResidualBlock2Plus1D(64))
        self.down2 = nn.Sequential(Conv2Plus1D(64, 128, stride=2), ResidualBlock2Plus1D(128))
        self.down3 = nn.Sequential(Conv2Plus1D(128, 256, stride=2), ResidualBlock2Plus1D(256))

        # Bottleneck angular dims: input is padded to a multiple of 8, then halved 3×
        ph = (8 - nlat % 8) % 8
        pw = (8 - nlon % 8) % 8
        nlat_bot = (nlat + ph) // 8
        nlon_bot = (nlon + pw) // 8

        # Half-Nyquist truncation per the SFNO paper ("frequencies above half the
        # sampling frequency are dropped"). lmax Nyquist = nlat_bot, so half = nlat_bot//2.
        # mmax Nyquist = nlon_bot//2 (real SHT), so half = nlon_bot//4.
        lmax = nlat_bot // 2
        mmax = nlon_bot // 4

        self.sfno_blocks = nn.ModuleList([
            SFNOBlock(256, nlat_bot, nlon_bot, lmax=lmax, mmax=mmax, radial_modes=8)
            for _ in range(3)
        ])
        
        self.up1 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.conv1 = ResidualBlock2Plus1D(128)
        self.up2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.conv2 = ResidualBlock2Plus1D(64)
        self.up3 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.conv3 = ResidualBlock2Plus1D(32)
        self.outc = nn.Conv3d(32, in_c, 1)

        # --- RESIDUAL PREDICTION INIT ---
        # Initialize the final output layer to zero so that the initial prediction
        # is just the identity (i.e. model predicts zero change at start).
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

    def _get_grid(self, b, d, h, w, device, dtype):
        z, y, x = torch.meshgrid(torch.arange(d, device=device),
                                 torch.arange(h, device=device),
                                 torch.arange(w, device=device), indexing='ij')
        grid = torch.stack([x/max(w-1, 1), y/max(h-1, 1), z/max(d-1, 1)], dim=0).unsqueeze(0)
        return grid.to(dtype).expand(b, -1, -1, -1, -1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        coords = self._get_grid(b, d, h, w, x.device, x.dtype)
        x_in = torch.cat([x, coords], dim=1)
        
        pd, ph, pw = (8 - d % 8) % 8, (8 - h % 8) % 8, (8 - w % 8) % 8
        x_pad = F.pad(x_in, (0, pw, 0, ph, 0, pd))

        x1 = checkpoint(self.inc, x_pad, use_reentrant=False)
        x2 = checkpoint(self.down1, x1, use_reentrant=False)
        x3 = checkpoint(self.down2, x2, use_reentrant=False)
        x4 = checkpoint(self.down3, x3, use_reentrant=False)
        
        x4_sph = x4
        for blk in self.sfno_blocks:
            x4_sph = checkpoint(blk, x4_sph, use_reentrant=False)

        u = checkpoint(self.up1, x4_sph, use_reentrant=False)
        u = checkpoint(self.conv1, u + x3, use_reentrant=False)
        u = checkpoint(self.up2, u, use_reentrant=False)
        u = checkpoint(self.conv2, u + x2, use_reentrant=False)
        u = checkpoint(self.up3, u, use_reentrant=False)
        u = checkpoint(self.conv3, u + x1, use_reentrant=False)
        
        out = self.outc(u)
        return out[:, :, :d, :h, :w]


# --- DDP TRAINING LOOP ---
def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if dist.get_rank() == 0:
        os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
        os.makedirs(CONFIG["log_dir"], exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(CONFIG["log_dir"], "train_forecast.log"),
            level=logging.INFO
        )

    log_print(f"Training Hybrid U-NO on {device} via DDP. Unrolling={CONFIG['unroll_steps']}")
    
    files = sorted(glob.glob(os.path.join(CONFIG["data_dir"], "dump_*.npy")))
    n_train = int(0.9 * len(files))
    
    train_dataset = HAMRForecastDataset(files[:n_train], CONFIG["unroll_steps"])
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], sampler=train_sampler, num_workers=4)

    val_dataset = HAMRForecastDataset(files[n_train:], 1)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], sampler=val_sampler, num_workers=4)
    
    model = Hybrid_UNO_3D(in_c=8,
                          nlat=CONFIG["grid_nlat"],
                          nlon=CONFIG["grid_nlon"]).to(device)
    crit = PhysicsLoss3D(device).to(device)
    
    # ~~~ CHANGE 5: crit has no learnable parameters; optimize model only ~~~
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    # ~~~ END CHANGE 5 ~~~
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)
    norm = DataNormalizer().to(device)
    scaler = GradScaler('cuda', enabled=CONFIG["use_amp"])
    
    best_loss = float('inf')
    start_ep = 0
    patience_counter = 0
    ckpt_path = os.path.join(CONFIG["checkpoint_dir"], "latest_forecast.pth")
    
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False) 
        if 'crit' in ckpt: crit.load_state_dict(ckpt['crit'], strict=False)  # strict=False: tolerates old log_vars / missing loss_scales
        try:
            opt.load_state_dict(ckpt['opt'])
            start_ep = ckpt['epoch']
            best_loss = ckpt['best_loss']
            patience_counter = ckpt.get('patience_counter', 0)
            log_print(f"Resumed Epoch {start_ep} with best loss {best_loss:.4f}")
        except ValueError:
            log_print("Optimizer state incompatible. Starting fresh optimizer.")

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    loss_hist_path = os.path.join(CONFIG["log_dir"], "loss_history.npz")
    loss_history = {"train": {}, "val": {}}
    gnorm_history = {}
    if dist.get_rank() == 0 and os.path.exists(loss_hist_path):
        saved = np.load(loss_hist_path, allow_pickle=True)
        for key in saved.files:
            if key == "loss_scales":
                continue
            prefix, name = key.split("_", 1)
            if prefix in ("train", "val"):
                loss_history[prefix][name] = saved[key].tolist()
            elif prefix == "gnorm":
                gnorm_history[name] = saved[key].tolist()

    # ~~~ CHANGE 4: Scale calibration pass (forward-only, no gradient updates) ~~~
    # Skipped on resume if scales were already saved in the checkpoint.
    # Averages raw component losses over calib_batches, then all-reduces across ranks
    # so every GPU normalizes by the same values.
    if not crit.scales_initialized:
        log_print(f"Running {CONFIG['calib_batches']} calibration batches to measure loss scales...")
        model.eval()
        scale_acc = torch.zeros(7, device=device)
        n_acc = 0
        with torch.no_grad():
            calib_iter = iter(train_loader)
            for _ in range(CONFIG["calib_batches"]):
                try:
                    sequence = next(calib_iter)
                except StopIteration:
                    break
                sequence = [s.to(device) for s in sequence]
                curr_state = norm(sequence[0])
                pred_delta = model.module(curr_state)
                pred_norm = curr_state + pred_delta
                _, s = crit(pred_norm, sequence[1].to(device), norm, 0, CONFIG["epochs"])
                scale_acc += torch.tensor(
                    [s["Fluid"], s["Mag"], s["Grad"], s["Spec"],
                     s["DivB"], s["Mass"], s["Gamma"]], device=device
                )
                n_acc += 1
        avg_scales = (scale_acc / max(n_acc, 1)).clamp(min=1e-6)
        avg_scales[6] = 1.0  # gamma excluded: near-zero at init, fixed at 1.0 (Lorentz units)
        dist.all_reduce(avg_scales, op=dist.ReduceOp.SUM)
        avg_scales /= dist.get_world_size()
        crit.loss_scales.copy_(avg_scales)
        crit.scales_initialized.fill_(True)
        log_print(f"Calibrated loss scales (F/M/G/S/D/Ma/L): {crit.loss_scales.cpu().numpy()}")
    # ~~~ END CHANGE 4 ~~~

    warmup_epochs = max(1, int(CONFIG["epochs"] * 0.1)) # 10% of training is warmup
    init_lr = 1e-5
    peak_lr = CONFIG["learning_rate"]

    for ep in range(start_ep, CONFIG["epochs"]):
        train_sampler.set_epoch(ep)
        
        # --- LEARNING RATE WARMUP LOGIC ---
        if ep < warmup_epochs:
            current_lr = init_lr + (peak_lr - init_lr) * (ep / warmup_epochs)
            for param_group in opt.param_groups:
                param_group['lr'] = current_lr
            if dist.get_rank() == 0:
                log_print(f"Warmup Phase: Set LR to {current_lr:.2e}")

        model.train()
        stats = {}

        if dist.get_rank() == 0:
            loop = tqdm(train_loader, desc=f"Ep {ep}")
        else:
            loop = train_loader
            
        for sequence in loop:
            sequence =[s.to(device) for s in sequence]
            opt.zero_grad()
            total_loss = 0
            curr_state = norm(sequence[0])
            
            with autocast('cuda', enabled=CONFIG["use_amp"]):
                for step in range(1, CONFIG["unroll_steps"] + 1):
                    target_raw = sequence[step]
                    
                    # --- RESIDUAL PREDICTION ---
                    pred_delta = model(curr_state)
                    pred_norm = curr_state + pred_delta
                    
                    loss, s_dict = crit(pred_norm, target_raw, norm, ep, CONFIG["epochs"])
                    total_loss += loss
                    
                    for k,v in s_dict.items(): stats[k] = stats.get(k,0) + (v / CONFIG["unroll_steps"])
                    curr_state = pred_norm.detach() 

                total_loss = total_loss / CONFIG["unroll_steps"]
                stats["Total"] = stats.get("Total", 0) + total_loss.item()

            scaler.scale(total_loss).backward()
            scaler.unscale_(opt)
            
            # ~~~ CHANGE 6: Removed log_vars DDP all-reduce (log_vars no longer exists) ~~~
            # ~~~ END CHANGE 6 ~~~

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            stats["GradNorm"] = stats.get("GradNorm", 0) + grad_norm.item()
            
            valid_gradients = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if valid_gradients:
                scaler.step(opt)
            else:
                log_print("WARNING: NaN detected. Skipping batch.")
                opt.zero_grad()

            scaler.update()
            
            if dist.get_rank() == 0:
                loop.set_postfix(Loss=total_loss.item())
            
        model.eval()
        val_stats = {}
        with torch.no_grad():
            for sequence in val_loader:
                curr, target = sequence[0].to(device), sequence[1].to(device)
                with autocast('cuda', enabled=CONFIG["use_amp"]):
                    curr_state = norm(curr)
                    # --- RESIDUAL PREDICTION ---
                    pred_delta = model(curr_state)
                    pred_norm = curr_state + pred_delta

                    total_val_loss, s = crit(pred_norm, target, norm, ep, CONFIG["epochs"])
                    val_stats["Total"] = val_stats.get("Total", 0) + total_val_loss.item()
                    for k, v in s.items():
                        val_stats[k] = val_stats.get(k, 0) + v

        # Per-component gradient norm (rank 0, single val batch, model.module bypasses DDP sync)
        gnorm = {}
        if dist.get_rank() == 0:
            model.eval()
            try:
                _seq = next(iter(val_loader))
                _curr = norm(_seq[0].to(device))
                with torch.enable_grad():
                    _pred = model.module(_curr)
                    _pred_norm = _curr.detach() + _pred
                    _, _, _raw = crit(_pred_norm, _seq[1].to(device), norm,
                                      ep, CONFIG["epochs"], return_raw=True)
                    _names = ["Fluid", "Mag", "Grad", "Spec", "DivB", "Mass", "Gamma"]
                    for _i, _name in enumerate(_names):
                        _grads = torch.autograd.grad(
                            _raw[_i], model.module.parameters(),
                            retain_graph=(_i < len(_names) - 1), allow_unused=True
                        )
                        gnorm[_name] = torch.sqrt(
                            sum(g.detach().norm() ** 2 for g in _grads if g is not None)
                        ).item()
            except StopIteration:
                pass
            model.train()

        val_reduce_keys = sorted(val_stats.keys())
        val_tensor = torch.tensor([val_stats[k] for k in val_reduce_keys], device=device)
        dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
        n_val = len(val_loader) * dist.get_world_size()
        avg_val_stats = {k: val_tensor[i].item() / n_val for i, k in enumerate(val_reduce_keys)}
        avg_val = avg_val_stats["Fluid"]
        
        if ep >= warmup_epochs:
            sched.step(avg_val)
        
        should_stop = False
        if dist.get_rank() == 0:
            # ~~~ CHANGE 7: Log calibrated scales instead of dynamic weights ~~~
            scales = crit.loss_scales.detach().cpu().numpy()
            w_str = f"CalibScales(F/M/G/S/D/Ma/L): {np.array2string(scales, precision=3, separator=',')}"
            # ~~~ END CHANGE 7 ~~~

            log_str = f"Ep {ep} | ValF: {avg_val:.4f} | " + " | ".join([f"{k}: {v/len(train_loader):.4f}" for k,v in stats.items()])

            curr_lr = opt.param_groups[0]['lr']
            log_print(f"LR: {curr_lr:.2e} | " + log_str)
            log_print(w_str)

            # --- LOSS CURVE SAVING ---
            for k, v in stats.items():
                loss_history["train"].setdefault(k, []).append(v / len(train_loader))
            for k, v in avg_val_stats.items():
                loss_history["val"].setdefault(k, []).append(v)
            for k, v in gnorm.items():
                gnorm_history.setdefault(k, []).append(v)
            np.savez(
                loss_hist_path,
                **{f"train_{k}": np.array(v) for k, v in loss_history["train"].items()},
                **{f"val_{k}": np.array(v) for k, v in loss_history["val"].items()},
                **{f"gnorm_{k}": np.array(v) for k, v in gnorm_history.items()},
                loss_scales=crit.loss_scales.detach().cpu().numpy(),
            )

            state = {
                'epoch': ep+1,
                'model': model.module.state_dict(),
                'crit': crit.state_dict(),
                'opt': opt.state_dict(),
                'best_loss': best_loss,
                'patience_counter': patience_counter,
            }
            torch.save(state, ckpt_path)

            if avg_val < best_loss:
                best_loss = avg_val
                patience_counter = 0
                torch.save(state, os.path.join(CONFIG["checkpoint_dir"], "best_forecast.pth"))
                log_print("Saved Best Forecasting Model.")
            else:
                if ep >= warmup_epochs:
                    patience_counter += 1
                    log_print(f"No improvement for {patience_counter} epochs.")
                    if patience_counter >= CONFIG["patience"]:
                        log_print(f"Early stopping triggered after {patience_counter} epochs. Training halted.")
                        should_stop = True

        stop_tensor = torch.tensor(int(should_stop), device=device)
        dist.broadcast(stop_tensor, src=0)
        if stop_tensor.item():
            break

    dist.destroy_process_group()

if __name__ == "__main__": main()

#torchrun --nnodes=1 --nproc_per_node=gpu train_forecast.py
