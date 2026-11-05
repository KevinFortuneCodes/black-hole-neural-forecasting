# Written by Sachin Venkatesh (the TRON model); included with credit because code in this
# repository uses it. Kevin Fortune claims no ownership of this file. See LICENSE.
# Copied from evaluate_forecast_ds.py (sachin branch, fall26/, commit ca02885); only the data paths were
# replaced with placeholders. Run it through evaluate_vit.py.
"""
evaluate_forecast_ds.py 
Usage:
  python evaluate_forecast_ds.py --mode all --max_steps 50
  python evaluate_forecast_ds.py --mode ml        # ML section only
  python evaluate_forecast_ds.py --mode physics   # physics section only
  python evaluate_forecast_ds.py --mode single_step --start 0
  python evaluate_forecast_ds.py --mode rollout_curve --max_steps 100 --n_starts 10

All figures/saves go to --save_dir; all stochastic components are seeded
(--seed, default 42) — numbers are reproducible run-to-run.
"""

import os, sys, time, argparse, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Poloidal-plane grids (X = r sin theta, Y = r cos theta) are inherently
# non-monotonic in X across the poles, so pcolormesh's cell-center edge
# inference always warns. The rendering is correct (explicitly masked to the
# plotted radial range); silence the spurious warning only.
warnings.filterwarnings("ignore", message="The input coordinates to pcolormesh are interpreted as cell centers.*")

sys.path.append(os.getcwd())

os.environ.setdefault("LOCAL_RANK", "1")

from uno_simple_ds import TRON, Hybrid_UNO_3D, DataNormalizer, PhysicsLoss3D, load_geometry, CONFIG

data_dir       = "/path/to/5000run/processed_forecast/blind"
geometry_file  = "/path/to/5000run/processed_forecast/geometry.npz"
model_path     = "/path/to/5000run/processed_forecast/checkpoints_uno_simple/best_forecast.pth"

CHANNEL_NAMES = [r'$\rho$', r'$u_g$', r'$\tilde{u}^r$', r'$\tilde{u}^\theta$',
                 r'$\tilde{u}^\varphi$', r'$B^r$', r'$B^\theta$', r'$B^\varphi$']
CHANNEL_NAMES_PLAIN = ['rho', 'ug', 'utilde_r', 'utilde_th', 'utilde_ph', 'B_r', 'B_th', 'B_ph']

# Colorblind-safe, publication-consistent color mapping across all figures
CHANNEL_COLORS = [
    '#1f77b4',  # rho: steel blue
    '#ff7f0e',  # ug: amber orange
    '#2ca02c',  # utilde^r: emerald green
    '#d62728',  # utilde^theta: crimson
    '#9467bd',  # utilde^phi: royal purple
    '#8c564b',  # B^r: sienna brown
    '#e377c2',  # B^theta: orchid pink
    '#17becf',  # B^phi: dark cyan
]

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'axes.linewidth': 1.1,
    'grid.linewidth': 0.6,
    'grid.alpha': 0.35,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

R_DISK_MAX = 100.0  # Radial domain cap for disk-focused radial profiles (r in [r_H, R_DISK_MAX])

# ============================================================================
#  Data Loading & Normalizer Setup
# ============================================================================

def load_dump(data_dir, dump_id):
    path = os.path.join(data_dir, f"dump_{dump_id}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)

def dump_ids(data_dir):
    return sorted(
        int(f.split('_')[1].split('.')[0])
        for f in os.listdir(data_dir)
        if f.startswith("dump_") and f.endswith(".npy"))

def build_pipeline(geometry_file, model_path, device):
    """TRON model + normalizer + physics terms, all imported from uno_simple_ds.py.
    Returns (model, norm, norm_cpu, crit, geo, geo_raw)."""
    CONFIG["geometry_file"] = geometry_file
    geo_raw = np.load(geometry_file)
    geo = load_geometry(geometry_file, device)          # wrapped (1,1,...) buffers
    CONFIG["eos_gamma"] = float(geo['eos_gamma']) if geo.get('eos_gamma') is not None else None
    CONFIG["dt_dump"] = float(geo['dt_dump']) if geo.get('dt_dump') is not None else None

    model = TRON(in_c=8).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    sd = ckpt.get('model', ckpt)
    if any(k.startswith('module.') for k in sd):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"Loaded {sum(p.numel() for p in model.parameters())/1e6:.2f}M-param TRON "
          f"checkpoint (epoch {ckpt.get('epoch', '?')}, best_loss "
          f"{ckpt.get('best_loss', '?')})")

    norm = DataNormalizer(gcov=geo['gcov'], gcon=geo['gcon']).to(device)
    norm_cpu = DataNormalizer(gcov=geo['gcov'].cpu(), gcon=geo['gcon'].cpu())
    crit = PhysicsLoss3D(device)
    return model, norm, norm_cpu, crit, geo, geo_raw

def forecast_one_step(model, norm, curr_state, device):
    """Single autoregressive step: residual prediction in normalized space,
    decode with the exact Lorentz clamp in utilde space."""
    norm_in = norm(curr_state)
    pred_delta = model(norm_in)
    pred_norm = norm_in + pred_delta
    out = norm.inverse(pred_norm).float()
    out[:, 0] = torch.clamp(out[:, 0], min=CONFIG["floor_rho"], max=CONFIG["ceiling_rho"])
    out[:, 1] = torch.clamp(out[:, 1], min=CONFIG["floor_ug"], max=CONFIG["ceiling_ug"])
    return out

_ROLLOUT_CACHE = {}
_CLAMP_CACHE = {}

def rollout_predictions(model, norm, init_data, n_steps, device, dump_id=None):
    """Runs a single forward autoregressive rollout up to n_steps.
    Returns dict {step: state_numpy} for step in 1..n_steps.
    Caches predictions by dump_id so subsequent evaluation modes starting
    from the same dump reuse the rollout trajectory without re-running forward passes."""
    global _ROLLOUT_CACHE, _CLAMP_CACHE
    key = dump_id if dump_id is not None else id(init_data)
    if key not in _ROLLOUT_CACHE:
        _ROLLOUT_CACHE[key] = {}
        _CLAMP_CACHE[key] = {}
    cached = _ROLLOUT_CACHE[key]
    clamp_cached = _CLAMP_CACHE[key]

    curr_step = max(cached.keys()) if cached else 0
    if curr_step >= n_steps:
        return cached

    if curr_step == 0:
        curr = to_device(init_data, device)
    else:
        curr = to_device(cached[curr_step], device)

    with torch.no_grad():
        for step in range(curr_step + 1, n_steps + 1):
            curr = forecast_one_step(model, norm, curr, device)
            cached[step] = curr.cpu().numpy()[0]
            clamp_cached[step] = int(norm.lorentz_clamp_count(reset=True))
    return cached

def get_rollout_clamps(dump_id=None):
    global _CLAMP_CACHE
    return _CLAMP_CACHE.get(dump_id, {})

def to_device(raw, device):
    if torch.is_tensor(raw):
        t = raw.detach().to(device=device, dtype=torch.float32)
    elif isinstance(raw, np.ndarray):
        t = torch.from_numpy(raw).to(device=device, dtype=torch.float32)
    else:
        t = torch.tensor(raw, device=device, dtype=torch.float32)
    if t.ndim == 4:
        t = t.unsqueeze(0)
    return t

def to_tensor_cpu(raw):
    """Zero-copy or detached float32 CPU tensor with batch dimension."""
    if isinstance(raw, np.ndarray):
        t = torch.from_numpy(raw).float()
    elif torch.is_tensor(raw):
        t = raw.detach().cpu().float()
    else:
        t = torch.tensor(raw, dtype=torch.float32)
    if t.ndim == 4:
        t = t.unsqueeze(0)
    return t

def auto_horizons(max_steps):
    if max_steps <= 20:
        horizons = [1, max(1, max_steps // 4), max(1, max_steps // 2), max_steps]
    elif max_steps <= 50:
        horizons = [1, 10, 25, max_steps]
    elif max_steps <= 100:
        horizons = [1, 10, 25, 50, max_steps]
    else:
        horizons = [1, 10, 25, 50, 100, max_steps]
    return sorted(set([h for h in horizons if 1 <= h <= max_steps]))

def dynamic_xticks(max_val):
    if max_val <= 20:
        return list(range(1, max_val + 1))
    elif max_val <= 50:
        return list(range(1, max_val + 1, 5))
    else:
        return list(range(1, max_val + 1, 10))

def active_disk_mask(geo_raw, rmax=R_DISK_MAX):
    """Active disk boolean mask: (r <= rmax) & (|theta - pi/2| < 0.5)."""
    r = geo_raw['r']
    h = geo_raw['h']
    return (r <= rmax) & (np.abs(h - np.pi / 2.0) < 0.5)


# ============================================================================
#  ML Section Metrics
# ============================================================================

def nmae_per_channel(pred_raw, true_raw, norm_cpu):
    """Normalized-space mean absolute error per channel (the training metric)."""
    with torch.no_grad():
        p = norm_cpu(to_tensor_cpu(pred_raw))
        t = norm_cpu(to_tensor_cpu(true_raw))
        return (p - t).abs().mean(dim=(2, 3, 4)).squeeze(0).numpy()

def rel_l1_per_channel(pred_raw, true_raw):
    """Physical-space relative L1 per channel."""
    rel = np.zeros(8)
    for c in range(8):
        denom = np.abs(true_raw[c]).mean()
        rel[c] = np.nan if denom < 1e-30 else np.abs(pred_raw[c] - true_raw[c]).mean() / denom
    return rel


# ============================================================================
#  Physics Section Metrics (all reuse the training file's validated code)
# ============================================================================

def mass_shells(raw_t, crit, norm):
    """M(r) = sum_{theta,phi} rho u^0 sqrt(-g) [bare θφ-sum (no /n_ang, no dx); Fig 5/Table 2 normalize by n_ang at call sites; Table 5 intentionally bare on both sides]."""
    x = to_device(raw_t, crit.gdet.device)
    rho = crit._safe_rho_ug(x[:, 0:1], CONFIG["floor_rho"], CONFIG["ceiling_rho"])
    u0 = norm.u0_of(x[:, 2:5])
    return torch.sum(rho * u0 * crit.gdet, dim=(3, 4))   # (B,1,Nr)

def conserved_shells(raw_t, crit, norm):
    """Per-shell angular-mean conserved densities:
    E(r) = rest-mass-subtracted binding-energy density: (1/N_ang) sum_{theta,phi} sqrt(-g) (T^t_0 + rho u^0)
    L(r) = (1/N_ang) sum_{theta,phi} sqrt(-g) T^t_phi in float64.
    Self-contained out-of-sample physical transport diagnostic (not in TRON training loss)."""
    x = to_device(raw_t, crit.gdet.device).double()
    rho = crit._safe_rho_ug(x[:, 0:1], CONFIG["floor_rho"], CONFIG["ceiling_rho"])
    ug = crit._safe_rho_ug(x[:, 1:2], CONFIG["floor_ug"], CONFIG["ceiling_ug"])
    b1, b2, b3 = x[:, 5:6], x[:, 6:7], x[:, 7:8]
    gdet_d = crit.gdet.double()

    ucon = norm.ucon_of(x[:, 2:5]).double()
    u0 = ucon[:, 0:1]
    alpha = 1.0 / torch.sqrt(-crit.gcon[0, 0, 0, 0].double() + 1e-300)
    gamma = alpha * u0

    eos_gamma = crit.eos_gamma if crit.eos_gamma is not None else float(CONFIG.get("eos_gamma") or 5.0 / 3.0)
    p = (eos_gamma - 1.0) * ug
    w = rho + ug + p

    g = crit.gcov[0, 0].double()
    ucov = torch.einsum('bi...,ij...->bj...', ucon, g)

    Bcon = torch.cat([torch.zeros_like(u0), alpha * b1, alpha * b2, alpha * b3], dim=1)
    u_dot_B = torch.einsum('bi...,bi...->b...', ucov, Bcon).unsqueeze(1)
    bcon = (Bcon + ucon * u_dot_B) / gamma
    bcov = torch.einsum('bi...,ij...->bj...', bcon, g)
    n_dot_b = -alpha * bcon[:, 0:1]
    bsq = torch.einsum('bi...,bi...->b...', bcov, bcon).unsqueeze(1)

    U0 = (gamma * (w + bsq) * ucov[:, 0:1]
          - (p + bsq / 2.0) * (-alpha)
          + n_dot_b * bcov[:, 0:1]) / alpha + rho * u0
    U3 = (gamma * (w + bsq) * ucov[:, 3:4]
          + n_dot_b * bcov[:, 3:4]) / alpha
    n_ang = float(x.shape[3] * x.shape[4])
    return (torch.sum(U0 * gdet_d, dim=(3, 4)) / n_ang,
            torch.sum(U3 * gdet_d, dim=(3, 4)) / n_ang)

def signed_log_err(pred_shell, true_shell):
    """mean |sl(pred) - sl(true)| — shell error in signed-log space."""
    return torch.mean(torch.abs(crit_sl(pred_shell) - crit_sl(true_shell)))

def crit_sl(x):
    return PhysicsLoss3D._signed_log(x)

def physical_divb(raw_t, crit):
    """Volume-averaged physical solenoidal residual = mean(|div B|) [code units].
    Exact training loss formulation from uno_simple_ds.py (TRON) and its in-training monitor."""
    x = to_device(raw_t, crit.gdet.device)
    divb = crit.calc_divB(x[:, 5:6], x[:, 6:7], x[:, 7:8])
    return float(torch.mean(torch.abs(divb)).item())

def normalized_divb(raw_t, crit, geo_raw=None, return_global=False):
    """Normalized solenoidal residual = volume-mean(|divB| * Delta x_local / |B|_local).
    Delta x_local = (gdet * dx * dy * dz)^(1/3).
    Denominator is max(|B|_local, 1e-3 * median(|B|)) + 1e-12 (|B|_local is coordinate magnitude).
    If geo_raw is provided, evaluates over the active disk pool (r <= R_DISK_MAX, |theta - pi/2| < 0.5,
    rho > 100 * floor_rho) with boundary trimming 1:-1 in r and theta (same triple condition as flux)."""
    x = to_device(raw_t, crit.gdet.device)
    divb = crit.calc_divB(x[:, 5:6], x[:, 6:7], x[:, 7:8])
    dx_local = torch.pow(crit.gdet * crit.dx * crit.dy * crit.dz, 1.0 / 3.0)
    B_local = torch.sqrt(x[:, 5:6]**2 + x[:, 6:7]**2 + x[:, 7:8]**2)
    b_med = torch.median(B_local)
    denom = torch.maximum(B_local, 1e-3 * b_med) + 1e-12
    ndb_field = torch.abs(divb) * dx_local / denom
    ndb_global = float(torch.mean(ndb_field).item())

    if geo_raw is not None:
        ndb_int = ndb_field[0, 0, 1:-1, 1:-1, :].cpu().numpy()
        r_grid = geo_raw['r'][1:-1, 1:-1, :]
        h_grid = geo_raw['h'][1:-1, 1:-1, :]
        rho_safe = crit._safe_rho_ug(x[:, 0:1], CONFIG["floor_rho"], CONFIG["ceiling_rho"])
        rho_np = rho_safe[0, 0, 1:-1, 1:-1, :].cpu().numpy()
        floor_rho = float(CONFIG.get("floor_rho", 1e-4))
        geo_trim = {'r': r_grid, 'h': h_grid}
        disk_pool = active_disk_mask(geo_trim, rmax=R_DISK_MAX) & (rho_np > 100.0 * floor_rho)
        if disk_pool.sum() > 0:
            ndb_val = float(np.mean(ndb_int[disk_pool]))
        else:
            ndb_val = ndb_global
    else:
        ndb_val = ndb_global

    if return_global:
        return ndb_val, ndb_global
    return ndb_val

_FLUX_COUNT_PRINTED = False

def flux_balance_residual_local(x_k, x_k1, crit, norm, geo_raw, relative=True, n_cells=None, seed=0):
    """LOCAL, cellwise truth-free continuity residual between consecutive states:
        M_cell = rho u^0 sqrt(-g)
        F_cell^i = rho u^i sqrt(-g)  (using coordinate velocity u^i)
        R_cell = M_cell(k+1) - M_cell(k) + dt_dump * d_i F_cell^i
    Sampled from active disk pool (r <= 100, |theta - pi/2| < 0.5, rho > 100 * floor_rho)
    with boundary trimming 1:-1 in both r (axis 2) and theta (axis 3).
    Cellwise relative error: mean(|R| / (|M| + eps))."""
    dt = crit.dt_dump if crit.dt_dump is not None else float(CONFIG.get("dt_dump", 25.0))
    n_cells = n_cells if n_cells is not None else CONFIG.get("flux_local_cells", 10000)

    rho_k = crit._safe_rho_ug(x_k[:, 0:1], CONFIG["floor_rho"], CONFIG["ceiling_rho"]).double()
    rho_k1 = crit._safe_rho_ug(x_k1[:, 0:1], CONFIG["floor_rho"], CONFIG["ceiling_rho"]).double()
    ucon_k = norm.ucon_of(x_k[:, 2:5]).double()
    ucon_k1 = norm.ucon_of(x_k1[:, 2:5]).double()
    gdet_d = crit.gdet.double()

    M_k = rho_k * ucon_k[:, 0:1] * gdet_d
    M_k1 = rho_k1 * ucon_k1[:, 0:1] * gdet_d

    f_k = crit.calc_divB(rho_k * ucon_k[:, 1:2],
                         rho_k * ucon_k[:, 2:3],
                         rho_k * ucon_k[:, 3:4]) * gdet_d
    f_k1 = crit.calc_divB(rho_k1 * ucon_k1[:, 1:2],
                          rho_k1 * ucon_k1[:, 2:3],
                          rho_k1 * ucon_k1[:, 3:4]) * gdet_d
    R_cell = M_k1 - M_k + dt * (f_k + f_k1) / 2.0

    # Trim 1:-1 along r (axis 2) and theta (axis 3)
    R_int = R_cell[0, 0, 1:-1, 1:-1, :].cpu().numpy()
    M_int = M_k1[0, 0, 1:-1, 1:-1, :].cpu().numpy()

    r_grid = geo_raw['r'][1:-1, 1:-1, :]
    h_grid = geo_raw['h'][1:-1, 1:-1, :]
    rho_np = rho_k1[0, 0, 1:-1, 1:-1, :].cpu().numpy()

    floor_rho = float(CONFIG.get("floor_rho", 1e-4))
    geo_trim = {'r': r_grid, 'h': h_grid}
    disk_pool = active_disk_mask(geo_trim, rmax=R_DISK_MAX) & (rho_np > 100.0 * floor_rho)
    n_qual = int(disk_pool.sum())
    global _FLUX_COUNT_PRINTED
    if not _FLUX_COUNT_PRINTED:
        print(f"  [Flux Balance] Active disk qualifying cells: {n_qual}")
        _FLUX_COUNT_PRINTED = True

    if n_qual > 0:
        R_s = R_int[disk_pool]
        M_s = M_int[disk_pool]
    else:
        print("  Warning: No cells qualified in active disk pool; falling back to all interior cells.")
        R_s = R_int.ravel()
        M_s = M_int.ravel()

    if n_cells is not None and len(R_s) > n_cells:
        rng_f = np.random.default_rng(seed)
        idx = rng_f.choice(len(R_s), size=n_cells, replace=False)
        R_s = R_s[idx]
        M_s = M_s[idx]

    if relative:
        return float(np.mean(np.abs(R_s) / (np.abs(M_s) + 1e-10)))
    return float(np.mean(np.abs(R_s)))

def gamma_stats(raw_t, norm, device):
    """Pre-clamp Lorentz factor stats: gamma_of_raw(utilde^i)."""
    x = to_device(raw_t, device)
    g = norm.gamma_of_raw(x[:, 2:5]).double().cpu().numpy().ravel()
    return g

def radial_spectrum(raw_t, crit, channel):
    """Windowed (Hann r x theta, like the spectral loss) radial power spectrum:
    Mean-subtracted spatial field, rfftn over all three axes, average over (theta, phi) ->
    spectrum folded onto positive radial mode index |k| in [0, Nr//2]. Returns (k, power)."""
    x = to_device(raw_t, crit.gdet.device)
    field = x[:, channel:channel+1]
    field = field - torch.mean(field, dim=(-3, -2, -1), keepdim=True)
    ch = field * crit.spec_win
    fft = torch.fft.rfftn(ch.contiguous().to(torch.float32), dim=[-3, -2, -1])
    spec_full = (torch.abs(fft) ** 2).mean(dim=(-2, -1))[0, 0].cpu().numpy()
    Nr = len(spec_full)
    N_half = Nr // 2
    # Fold negative frequencies (indices Nr-1 down to N_half+1) onto |k|:
    spec_folded = np.zeros(N_half + 1, dtype=spec_full.dtype)
    spec_folded[0] = spec_full[0]
    spec_folded[1:N_half] = spec_full[1:N_half] + spec_full[Nr - 1:N_half:-1]
    spec_folded[N_half] = spec_full[N_half]
    k = np.arange(N_half + 1)
    return k, spec_folded

def spectral_slope(k, E):
    if len(k) < 10 or len(E) < 10:
        return 0.0, 0.0
    n = len(k)
    lo, hi = n // 4, 3 * n // 4
    if lo >= hi:
        return 0.0, 0.0
    lk = np.log10(k[lo:hi])
    lE = np.log10(np.maximum(E[lo:hi], 1e-30))
    poly = np.polyfit(lk, lE, 1)
    slope = float(poly[0])
    y_fit = poly[0] * lk + poly[1]
    ss_tot = np.sum((lE - np.mean(lE)) ** 2)
    ss_res = np.sum((lE - y_fit) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-30)) if ss_tot > 1e-30 else 0.0
    return slope, r2

def mdot_shells(raw_t, crit, norm, bare=True):
    """Mdot(r) = -sum_{theta,phi} rho u^r sqrt(-g) (accretion rate per shell;
    the F(r) flux of the continuity machinery; inflow positive). Reconstructs
    coordinate u^r from projected primitive utilde^i via norm.ucon_of.
    When bare=False, multiplies by dA = crit.dy * crit.dz for physical code units."""
    x = to_device(raw_t, crit.gdet.device)
    rho = crit._safe_rho_ug(x[:, 0:1], CONFIG["floor_rho"], CONFIG["ceiling_rho"])
    ucon = norm.ucon_of(x[:, 2:5])
    ur = ucon[:, 1:2]
    res = -torch.sum(rho * ur * crit.gdet, dim=(3, 4))
    if not bare:
        res = res * (crit.dy * crit.dz)
    return res

def magnetic_flux_shells(raw_t, crit, bare=True):
    """Phi(r) = sum_{theta,phi} |B^r| sqrt(-g) — the magnetic flux threading
    each spherical shell. The near-horizon value Phi(r_H) is the single most
    important BH-observable number (MAD/SANE ordering; jet power ~ Phi^2).
    When bare=False, multiplies by dA = crit.dy * crit.dz for physical code units."""
    x = to_device(raw_t, crit.gdet.device)
    res = torch.sum(torch.abs(x[:, 5:6]) * crit.gdet, dim=(3, 4))
    if not bare:
        res = res * (crit.dy * crit.dz)
    return res

def pdf_on_shared_bins(pred_c, true_c, n_bins=200, log_scale=False):
    """PDFs of pred/true on the SAME bin edges (percentiles from the combined
    sample) — the old version used per-distribution edges, making the JS
    divergence meaningless. Returns (bin_centers, pdf_pred, pdf_true, bin_widths)."""
    flat = np.concatenate([pred_c.ravel(), true_c.ravel()])
    if log_scale:
        abs_f = np.maximum(np.abs(flat), 1e-30)
        p1, p99 = np.percentile(abs_f, [1, 99])
        edges = np.logspace(np.log10(p1), np.log10(p99), n_bins + 1)
        hp, _ = np.histogram(np.clip(np.maximum(np.abs(pred_c), 1e-30), p1, p99), bins=edges, density=True)
        ht, _ = np.histogram(np.clip(np.maximum(np.abs(true_c), 1e-30), p1, p99), bins=edges, density=True)
    else:
        p1, p99 = np.percentile(flat, [0.5, 99.5])
        edges = np.linspace(p1, p99, n_bins + 1)
        hp, _ = np.histogram(np.clip(pred_c, p1, p99), bins=edges, density=True)
        ht, _ = np.histogram(np.clip(true_c, p1, p99), bins=edges, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:]) if log_scale else (edges[:-1] + edges[1:]) / 2
    widths = edges[1:] - edges[:-1]
    return centers, hp, ht, widths

def js_divergence(p, q, widths=None):
    if widths is not None:
        p = p * widths
        q = q * widths
    eps = 1e-12
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    with np.errstate(divide='ignore', invalid='ignore'):
        term_p = np.where(p > 0, p * np.log2(p / m), 0.0)
        term_q = np.where(q > 0, q * np.log2(q / m), 0.0)
    return 0.5 * (term_p.sum() + term_q.sum())



# ============================================================================
#  Table 1: Single-Step nMAE + Rel L1 + Persistence + Skill
# ============================================================================

def evaluate_single_step(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    ids = dump_ids(data_dir)
    test_ids = [d for d in ids if d >= args.start and (d + 1) in ids]
    if not test_ids:
        print(f"No consecutive test pairs found >= start {args.start}.")
        return None

    pair_nmae_m, pair_nmae_p, pair_rel_l1 = [], [], []
    for dump_id in test_ids:
        curr = load_dump(data_dir, dump_id)
        nxt = load_dump(data_dir, dump_id + 1)
        with torch.no_grad():
            pred = forecast_one_step(model, norm, to_device(curr, device), device).cpu().numpy()[0]
        nm_m = nmae_per_channel(pred, nxt, norm_cpu)
        nm_p = nmae_per_channel(curr, nxt, norm_cpu)
        pair_nmae_m.append(nm_m)
        pair_nmae_p.append(nm_p)
        pair_rel_l1.append(rel_l1_per_channel(pred, nxt))

    pair_nmae_m = np.array(pair_nmae_m)
    pair_nmae_p = np.array(pair_nmae_p)
    pair_rel_l1 = np.array(pair_rel_l1)

    nmae_model = np.mean(pair_nmae_m, axis=0)
    nmae_persist = np.mean(pair_nmae_p, axis=0)
    rel_l1 = np.nanmean(pair_rel_l1, axis=0)
    skill = np.where(nmae_persist < 1e-6, np.nan, 1.0 - nmae_model / (nmae_persist + 1e-30))

    # Paired bootstrap CI on the skill scores: stride-2 thinning of pairs
    # to account for temporal autocorrelation between consecutive dumps
    rng_b = np.random.default_rng(args.seed)
    B = 5000
    pair_m_thinned = pair_nmae_m[::2]
    pair_p_thinned = pair_nmae_p[::2]
    n_eff = len(pair_m_thinned)
    idx = rng_b.integers(0, n_eff, size=(B, n_eff))
    boot_m = pair_m_thinned[idx].mean(axis=1)     # (B, 8)
    boot_p = pair_p_thinned[idx].mean(axis=1)
    boot_skill = np.where(boot_p < 1e-6, np.nan, 1.0 - boot_m / (boot_p + 1e-30))
    lo = np.nanpercentile(boot_skill, 2.5, axis=0)
    hi = np.nanpercentile(boot_skill, 97.5, axis=0)

    def _fmt_s(v, w=9):
        if np.isnan(v):
            return f"{'n/a':>{w}}"
        val = 0.0 if abs(v) < 5e-4 else v
        return f"{val:>{w}.3f}"

    def _fmt_f(v, w=11, p=4):
        if np.isnan(v):
            return f"{'n/a':>{w}}"
        return f"{v:>{w}.{p}f}"

    print(f"\n{'='*76}")
    print(f" Table 1: Single-Step Accuracy on Blind Test Dumps ({len(test_ids)} pairs, n_eff={n_eff})")
    print(f"{'='*76}")
    print(f"{'Channel':<10} {'nMAE TRON':>11} {'Rel L1 TRON':>12} {'nMAE persist':>13} {'Skill SS':>9} {'95% CI':>16}")
    print("-" * 76)
    for c in range(8):
        ci_str = f"[{_fmt_s(lo[c], 6)},{_fmt_s(hi[c], 6)}]" if not (np.isnan(lo[c]) or np.isnan(hi[c])) else f"{'n/a':>16}"
        print(f"{CHANNEL_NAMES_PLAIN[c]:<10} {_fmt_f(nmae_model[c], 11, 4)} {_fmt_f(rel_l1[c], 12, 4)} "
              f"{_fmt_f(nmae_persist[c], 13, 4)} {_fmt_s(skill[c], 9)} {ci_str}")
    print("-" * 76)
    print(f"{'Mean':<10} {_fmt_f(np.nanmean(nmae_model), 11, 4)} {_fmt_f(np.nanmean(rel_l1), 12, 4)} "
          f"{_fmt_f(np.nanmean(nmae_persist), 13, 4)} {_fmt_s(np.nanmean(skill), 9)} {'—':>16}")
    print(f"\n  Fluid (0-4): nMAE {_fmt_f(np.nanmean(nmae_model[:5]), 6, 4).strip()}")
    print(f"  Magnetic (5-7): nMAE {_fmt_f(np.nanmean(nmae_model[5:]), 6, 4).strip()}")
    print(f"  95% bootstrap CI (paired, B={B}, seed {args.seed}, stride-2 thinned n_eff={n_eff} from {len(test_ids)} pairs) on channel skill scores.")
    print("  Mean row: unweighted channel mean (CI-free; channel metrics have disparate units/scales).")
    if n_eff < 10:
        print(f"  Warning: effective sample size n_eff={n_eff} < 10; bootstrap CIs may be unreliable due to small sample size.")

    np.savez(os.path.join(args.save_dir, "single_step_metrics.npz"),
             nmae_model=nmae_model, nmae_persist=nmae_persist,
             rel_l1_model=rel_l1, skill=skill, ci_lo=lo, ci_hi=hi,
             count=len(test_ids), n_eff=n_eff)
    return dict(nmae_model=nmae_model, nmae_persist=nmae_persist, rel_l1=rel_l1,
                skill=skill, ci_lo=lo, ci_hi=hi)


# ============================================================================
#  Fig 4: Rollout nMAE vs Steps (seeded starts, mean +/- std, persistence)
# ============================================================================

def channel_decorrelation_times(data_dir, norm_cpu, geo_raw=None, max_pairs=200):
    """Per-channel lag-1 autocorrelation decorrelation time from the TRUTH data:
    tau_c = -1/ln(r1_c), where r1_c = corr(x_t - <x>, x_{t+1} - <x>) over consecutive
    dump pairs, evaluated in the active disk (r <= 100 Rg, |theta-pi/2| < 0.5) with
    the time-mean background subtracted to isolate turbulent fluctuations.
    Returns (tau, is_saturated)."""
    ids = dump_ids(data_dir)[:max_pairs]
    if len(ids) < 2:
        return np.zeros(8), np.zeros(8, dtype=bool)

    disk_mask = None
    if geo_raw is not None:
        disk_mask = active_disk_mask(geo_raw, rmax=R_DISK_MAX)

    # Pass 1: compute time-mean field per channel
    sum_x = None
    valid_ids = []
    for did in ids:
        d = load_dump(data_dir, did)
        if d is None:
            continue
        nd = norm_cpu(to_tensor_cpu(d))[0].numpy()
        valid_ids.append(did)
        if sum_x is None:
            n_pts = nd[0][disk_mask].size if disk_mask is not None else nd[0].size
            sum_x = np.zeros((8, n_pts), dtype=np.float64)
        for c in range(8):
            fc = nd[c][disk_mask] if disk_mask is not None else nd[c].ravel()
            sum_x[c] += fc

    if len(valid_ids) < 2:
        return np.zeros(8), np.zeros(8, dtype=bool)

    mean_x = sum_x / len(valid_ids)

    # Pass 2: correlate fluctuations x - mean_x between consecutive dumps
    cov = np.zeros(8, dtype=np.float64)
    var_a = np.zeros(8, dtype=np.float64)
    var_b = np.zeros(8, dtype=np.float64)

    valid_id_set = set(valid_ids)
    pairs_counted = 0
    for did in valid_ids:
        if (did + 1) not in valid_id_set:
            continue
        a = load_dump(data_dir, did)
        b = load_dump(data_dir, did + 1)
        if a is None or b is None:
            continue
        na = norm_cpu(to_tensor_cpu(a))[0].numpy()
        nb = norm_cpu(to_tensor_cpu(b))[0].numpy()
        for c in range(8):
            fa = (na[c][disk_mask] if disk_mask is not None else na[c].ravel()) - mean_x[c]
            fb = (nb[c][disk_mask] if disk_mask is not None else nb[c].ravel()) - mean_x[c]
            cov[c] += np.dot(fa, fb)
            var_a[c] += np.dot(fa, fa)
            var_b[c] += np.dot(fb, fb)
        pairs_counted += 1

    if pairs_counted == 0:
        return np.zeros(8), np.zeros(8, dtype=bool)

    denom = np.sqrt(var_a * var_b) + 1e-30
    r1 = np.clip(cov / denom, -0.999, 0.999)
    is_saturated = r1 > 0.99

    tau = np.zeros(8)
    for c in range(8):
        if r1[c] <= 0:
            tau[c] = 0.0
        else:
            tau[c] = -1.0 / np.log(min(r1[c], 0.99))
    return tau, is_saturated


def evaluate_rollout_curve(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    all_ids = dump_ids(data_dir)
    all_ids_set = set(all_ids)
    candidates = [d for d in all_ids if d >= args.start]
    if not candidates:
        print(f"No dumps found >= args.start ({args.start})."); return None

    def contiguous_horizon(s):
        h = 0
        while (s + h + 1) in all_ids_set:
            h += 1
        return h

    max_avail_per_cand = {d: contiguous_horizon(d) for d in candidates}
    overall_max_avail = max(max_avail_per_cand.values()) if max_avail_per_cand else 0

    req_steps = args.max_steps
    if req_steps > overall_max_avail:
        print(f"\n  [Notice] Requested --max_steps {req_steps}, but the maximum contiguous "
              f"dump horizon from start >= {args.start} is {overall_max_avail} steps.")
        print(f"  Adjusting rollout horizon to {overall_max_avail} steps.")
        req_steps = overall_max_avail

    if req_steps < 1:
        print("  Error: No future dumps available for rollout."); return None

    valid = [d for d in candidates if max_avail_per_cand[d] >= req_steps]
    if not valid:
        print(f"No starts have {req_steps} contiguous future dumps."); return None

    rng = np.random.default_rng(args.seed)
    starts = valid if len(valid) <= args.n_starts else rng.choice(valid, args.n_starts, replace=False)
    print(f"  Rollout curve: evaluating {len(starts)} starts up to {req_steps} steps.")

    all_nmae, all_persist = [], []
    for s0 in starts:
        init = load_dump(data_dir, s0)
        curr = to_device(init, device)
        step_nmae, step_persist = [], []
        for step in range(1, req_steps + 1):
            true = load_dump(data_dir, s0 + step)
            if true is None:
                break
            with torch.no_grad():
                pred = forecast_one_step(model, norm, curr, device)
            curr = pred
            pred_np = pred.cpu().numpy()[0]
            step_nmae.append(nmae_per_channel(pred_np, true, norm_cpu))
            step_persist.append(nmae_per_channel(init, true, norm_cpu))
        if len(step_nmae) >= 1:
            all_nmae.append(np.array(step_nmae))
            all_persist.append(np.array(step_persist))

    if not all_nmae:
        print("No complete rollouts."); return None
    min_len = min(len(x) for x in all_nmae)
    A = np.array([x[:min_len] for x in all_nmae])          # (n_starts, steps, 8)
    P = np.array([x[:min_len] for x in all_persist])
    steps_arr = np.arange(1, min_len + 1)
    avg = A.mean(axis=-1)
    mean_nmae, std_nmae = avg.mean(axis=0), avg.std(axis=0)
    mean_persist = P.mean(axis=-1).mean(axis=0)
    skill_curve = 1.0 - mean_nmae / (mean_persist + 1e-30)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # (a) Mean nMAE vs Persistence
    ax = axes[0]
    ax.plot(steps_arr, mean_nmae, 'b-o', markersize=3, linewidth=1.5, label='TRON')
    ax.fill_between(steps_arr, mean_nmae - std_nmae, mean_nmae + std_nmae, alpha=0.2, color='blue')
    ax.plot(steps_arr, mean_persist, 'k--', linewidth=1.2, label='Persistence')
    ax.set_xlim(1, min_len)
    ax.set_xticks(dynamic_xticks(min_len))
    ax.set_xlabel("Forecast steps")
    ax.set_ylabel("Channel-averaged nMAE")
    ax.set_title(f"(a) Forecast Error")
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # (b) Per-channel growth indexed to step 1
    # Step-1 normalization illustrates the relative amplification of error over time;
    # guarded by a 1e-4 floor to fall back to absolute error if step-1 error is near-zero.
    ax2 = axes[1]
    fallback_used = False
    for c in range(8):
        c_step1 = A.mean(axis=0)[0, c]
        if c_step1 < 1e-4:
            fallback_used = True
            growth = A.mean(axis=0)[:, c]
            lbl = f"{CHANNEL_NAMES[c]} (abs)"
            print(f"  Warning: channel {CHANNEL_NAMES_PLAIN[c]} step-1 nMAE={c_step1:.2e} < 1e-4; falling back to absolute nMAE curve.")
        else:
            growth = A.mean(axis=0)[:, c] / c_step1
            lbl = CHANNEL_NAMES[c]
        ax2.plot(steps_arr, growth, linewidth=1.4, color=CHANNEL_COLORS[c], label=lbl)
    ax2.set_xlim(1, min_len)
    ax2.set_xticks(dynamic_xticks(min_len))
    ax2.set_xlabel("Forecast steps")
    ax2.set_ylabel("Error growth (relative to step 1)" if not fallback_used else "Error growth (rel or abs)")
    ax2.set_title("(b) Per-Channel Error Growth")
    ax2.legend(fontsize=8, ncol=2, frameon=True, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Autoregressive Forecast Accuracy ({len(starts)} seeded starts)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(args.save_dir, "fig4_rollout_analysis.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")

    # Attractor saturation: the long-rollout error floor vs the data's own
    # per-channel sigma. If the saturation nMAE stays BELOW sigma_c, the model
    # rollouts remain in the data's typical set instead of drifting to a
    # degenerate attractor (blow-up / collapse).
    ids_sat = all_ids[:50]
    sigma_c = np.zeros(8)
    n_val = 0
    for d in ids_sat:
        a = load_dump(data_dir, d)
        if a is not None:
            na = norm_cpu(to_tensor_cpu(a))[0].numpy()
            sigma_c += na.std(axis=(1, 2, 3))
            n_val += 1
    sigma_c /= max(1, n_val)
    sat = A.mean(axis=0)[-1]
    print(f"\n  Attractor saturation (t={min_len}): model nMAE vs data sigma per channel (diagnostic proxy)")
    for c in range(8):
        flag = "OK (typical set)" if sat[c] < sigma_c[c] else "EXCEEDS data sigma"
        print(f"    {CHANNEL_NAMES_PLAIN[c]:<10} sat {sat[c]:.4f}  sigma {sigma_c[c]:.4f}  {flag}")
    print("  Note: data sigma is normalized-space standard deviation; comparison serves as an attractor boundedness proxy.")

    np.savez(os.path.join(args.save_dir, "rollout_curve.npz"),
             steps=steps_arr, mean_nmae=mean_nmae, std_nmae=std_nmae,
             mean_persist=mean_persist,
             skill=skill_curve,
             per_channel=A.mean(axis=0), starts=starts)
    return dict(mean_nmae=mean_nmae, per_channel=A.mean(axis=0),
                skill=skill_curve)


# ============================================================================
#  Fig 5: Conserved Shell Profiles M(r), -E(r), L(r) + signed-log errors
# ============================================================================

def evaluate_shell_profiles(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    """Conserved shell radial profiles across forecast horizons:
    M(r) on log scale over full domain [r_H, r_max];
    binding energy -E(r) on log scale over active disk [r_H, R_DISK_MAX];
    angular momentum L(r) on log scale positive-only (negative shells counted
    via n_neg_L in stdout, not drawn) over [r_H, R_DISK_MAX].
    All rows display dashed ISCO markers at r_ISCO."""
    init = load_dump(data_dir, args.start)
    if init is None:
        print(f"Dump {args.start} not found in {data_dir}."); return
    r_mid = geo_raw['r'][:, geo_raw['r'].shape[1] // 2, 0]
    a_spin = float(geo_raw['spin']) if 'spin' in geo_raw else 0.0
    r_H, r_isco = kerr_horizon_isco(a_spin)
    r_max_M = float(r_mid.max())
    r_max_disk = R_DISK_MAX

    mask_M = (r_mid >= r_H) & (r_mid <= r_max_M)
    mask_EL = (r_mid >= r_H) & (r_mid <= r_max_disk)

    horizons = auto_horizons(args.max_steps)
    states = rollout_predictions(model, norm, init, max(horizons), device, dump_id=args.start)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=False)

    all_neg_E_t = []
    all_neg_E_p = []
    all_L_t = []
    all_L_p = []

    fc_colors = {1: '#ff7f0e', 10: '#d62728', 25: '#8c564b', 50: '#7f7f7f', 100: '#17becf'}
    err_colors = {1: 'purple', 10: '#ff7f0e', 25: '#2ca02c', 50: '#d62728', 100: '#8c564b'}
    def get_fc_color(h, idx):
        return fc_colors.get(h, plt.cm.tab10((idx + 1) % 10))
    def get_err_color(h, idx):
        if h == 1:
            return 'purple'
        return err_colors.get(h, plt.cm.tab10(idx % 10))

    for idx, n_steps in enumerate(horizons):
        target = load_dump(data_dir, args.start + n_steps)
        if target is None:
            continue
        pred_np = states[n_steps]

        with torch.no_grad():
            M_p = mass_shells(pred_np, crit, norm)[0, 0].cpu().numpy()
            M_t = mass_shells(target, crit, norm)[0, 0].cpu().numpy()
            (E_p, L_p) = conserved_shells(pred_np, crit, norm)
            (E_t, L_t) = conserved_shells(target, crit, norm)
            E_p, E_t = E_p[0, 0].cpu().numpy(), E_t[0, 0].cpu().numpy()
            L_p, L_t = L_p[0, 0].cpu().numpy(), L_t[0, 0].cpu().numpy()

        n_ang = float(crit.gdet.shape[-2] * crit.gdet.shape[-1])
        M_p = M_p / n_ang
        M_t = M_t / n_ang

        r0 = r_mid[mask_M]
        r1 = r_mid[mask_EL]
        neg_E_t = (-E_t)[mask_EL]
        neg_E_p = (-E_p)[mask_EL]
        L_t_plot = L_t[mask_EL]
        L_p_plot = L_p[mask_EL]

        # Truth line plotted once at the first horizon (dark blue, opacity=1, at top of legend)
        if n_steps == horizons[0]:
            axes[0, 0].plot(r0, M_t[mask_M], 'b-', linewidth=2.0, alpha=1.0, label='Truth')
            axes[1, 0].plot(r1, neg_E_t, 'b-', linewidth=2.0, alpha=1.0)
            axes[2, 0].plot(r1, L_t_plot, 'b-', linewidth=2.0, alpha=1.0)

        # Forecast curves
        col_fc = get_fc_color(n_steps, idx)
        axes[0, 0].plot(r0, M_p[mask_M], '--', color=col_fc, linewidth=1.5, label=f'TRON ({n_steps} steps)')
        axes[1, 0].plot(r1, neg_E_p, '--', color=col_fc, linewidth=1.5)
        axes[2, 0].plot(r1, L_p_plot, '--', color=col_fc, linewidth=1.5)

        # Signed-log errors (1 step is purple)
        col_err = get_err_color(n_steps, idx)
        sl_err_M = np.abs(np.sign(M_p) * np.log1p(np.abs(M_p)) - np.sign(M_t) * np.log1p(np.abs(M_t)))
        axes[0, 1].plot(r0, sl_err_M[mask_M], color=col_err, linewidth=1, label=f'{n_steps} steps')

        sl_err_E = np.abs(np.sign(-E_p) * np.log1p(np.abs(-E_p)) - np.sign(-E_t) * np.log1p(np.abs(-E_t)))
        axes[1, 1].plot(r1, sl_err_E[mask_EL], color=col_err, linewidth=1, label=f'{n_steps} steps')

        sl_err_L = np.abs(np.sign(L_p) * np.log1p(np.abs(L_p)) - np.sign(L_t) * np.log1p(np.abs(L_t)))
        axes[2, 1].plot(r1, sl_err_L[mask_EL], color=col_err, linewidth=1, label=f'{n_steps} steps')

        all_neg_E_t.extend(neg_E_t)
        all_neg_E_p.extend(neg_E_p)
        all_L_t.extend(L_t_plot)
        all_L_p.extend(L_p_plot)

    axes[0, 0].set_ylabel(r'$M(r)$')
    axes[0, 0].set_yscale('log')

    axes[1, 0].set_ylabel(r'$\mathcal{E}(r)$ (rest-mass-subtracted)')
    axes[1, 0].set_yscale('log')
    all_neg_E_arr = np.array(all_neg_E_t + all_neg_E_p) if (all_neg_E_t or all_neg_E_p) else np.array([1.0])
    n_neg_E = int((all_neg_E_arr < 0).sum())
    pos_E = all_neg_E_arr[all_neg_E_arr > 0]
    if len(pos_E) > 0:
        max_E = float(np.max(pos_E))
        p99_8_E = float(np.percentile(pos_E, 99.8))
        cap_E = E_YMAX_CAP
        y_max_E = min(cap_E, p99_8_E * 1.1)
        y_min_E = max(1e-5, float(np.percentile(pos_E, 0.2)) / 1.1)
        axes[1, 0].set_ylim(y_min_E, y_max_E)
        print(f"  Fig 5: -E(r) bounds: max={max_E:.3e}, p99.8={p99_8_E:.3e} -> adopted ylim=({y_min_E:.3e}, {y_max_E:.3e}), negative_shells={n_neg_E}")

    axes[2, 0].set_ylabel(r'$L(r)$')
    # L(r) plotted on log scale (positive-only); any counter-rotating negative shells counted via n_neg_L in stdout.
    axes[2, 0].set_yscale('log')
    all_L_arr = np.array(all_L_t + all_L_p) if (all_L_t or all_L_p) else np.array([1.0])
    n_neg_L = int((all_L_arr < 0).sum())
    pos_L = all_L_arr[all_L_arr > 0]
    if len(pos_L) > 0:
        max_L = float(np.max(pos_L))
        p99_8_L = float(np.percentile(pos_L, 99.8))
        cap_L = L_YMAX_CAP
        y_max_L = min(cap_L, p99_8_L * 1.1)
        y_min_L = max(1e-5, float(np.percentile(pos_L, 0.2)) / 1.1)
        axes[2, 0].set_ylim(y_min_L, y_max_L)
        print(f"  Fig 5: L(r) bounds: max={max_L:.3e}, p99.8={p99_8_L:.3e} -> adopted ylim=({y_min_L:.3e}, {y_max_L:.3e}), negative_shells={n_neg_L}")

    for row in range(3):
        axes[row, 0].set_xscale('log')
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 1].set_ylabel('Signed-log |err|')
        axes[row, 1].set_xscale('log')
        axes[row, 1].set_yscale('log')
        axes[row, 1].grid(True, alpha=0.3)

    # Row 0: M(r) spans computed domain [r_H, r_mid.max()]
    axes[0, 0].set_xlim(r_H, r_max_M)
    axes[0, 0].margins(x=0)
    axes[0, 1].set_xlim(r_H, r_max_M)
    axes[0, 1].margins(x=0)

    # Rows 1–2: -E(r) and L(r) active disk zoom r in [r_H, 100] Rg
    for r_idx in [1, 2]:
        axes[r_idx, 0].set_xlim(r_H, r_max_disk)
        axes[r_idx, 0].margins(x=0)
        axes[r_idx, 1].set_xlim(r_H, r_max_disk)
        axes[r_idx, 1].margins(x=0)

    # ISCO line on all 6 axes
    for row in range(3):
        for col in range(2):
            add_isco_line(axes[row, col], r_isco)

    axes[0, 0].legend(fontsize=8, ncol=1)
    axes[0, 1].legend(fontsize=8)
    axes[2, 0].set_xlabel(r'$r\;[R_g]$')
    axes[2, 1].set_xlabel(r'$r\;[R_g]$')
    fig.suptitle('Conserved Shell Profiles', fontsize=13)
    plt.tight_layout()
    path = os.path.join(args.save_dir, "fig5_conserved_shells.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================================
#  Table 2: Physics Consistency (shell errors, divB, continuity, gamma)
# ============================================================================

def evaluate_physics_table(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    init = load_dump(data_dir, args.start)
    if init is None:
        print(f"Dump {args.start} not found."); return
    d0 = physical_divb(init, crit)
    nd0, nd0_glob = normalized_divb(init, crit, geo_raw=geo_raw, return_global=True)
    horizons = auto_horizons(args.max_steps)
    max_h = max(horizons)
    states_np = rollout_predictions(model, norm, init, max_h, device, dump_id=args.start)
    states_torch = {0: to_device(init, device)}
    for h in horizons:
        if (h - 1) not in states_torch and (h - 1) in states_np:
            states_torch[h - 1] = to_device(states_np[h - 1], device)
        if h not in states_torch and h in states_np:
            states_torch[h] = to_device(states_np[h], device)

    print(f"\n{'='*144}")
    print(" Table 2: Physics Consistency (TRON vs solver truth across horizons)")
    print(f"{'='*144}")
    print(f"{'Steps':>5} {'sl-err M':>11} {'sl-err E':>11} {'sl-err L':>11} "
          f"{'|divB| pred':>14} {'|divB| truth':>14} {'norm-divB pred':>16} {'norm-divB truth':>17} "
          f"{'flux pred':>13} {'flux truth':>13} {'gmax pred':>11}")
    print("-" * 144)

    for h in horizons:
        target = load_dump(data_dir, args.start + h)
        if target is None:
            continue
        pred_np = states_np[h]
        prev_pred = states_torch[h - 1]
        curr_pred = states_torch[h]

        with torch.no_grad():
            n_ang = float(crit.gdet.shape[-2] * crit.gdet.shape[-1])
            M_p = mass_shells(pred_np, crit, norm)[0, 0] / n_ang
            M_t = mass_shells(target, crit, norm)[0, 0] / n_ang
            (E_p, L_p), (E_t, L_t) = conserved_shells(pred_np, crit, norm), conserved_shells(target, crit, norm)
            eM = float(signed_log_err(M_p, M_t).item())
            eE = float(signed_log_err(E_p[0, 0], E_t[0, 0]).item())
            eL = float(signed_log_err(L_p[0, 0], L_t[0, 0]).item())
            dp = physical_divb(pred_np, crit)
            dt = physical_divb(target, crit)
            ndp, ndp_glob = normalized_divb(pred_np, crit, geo_raw=geo_raw, return_global=True)
            ndt, ndt_glob = normalized_divb(target, crit, geo_raw=geo_raw, return_global=True)
            g_pred = gamma_stats(pred_np, norm, device)
            gmax = float(g_pred.max())
            # Truth-free continuity: solver-truth pair (start+h-1 -> start+h) vs
            # predicted pair (prev_pred -> curr_pred), both seeded identically.
            prev_true = load_dump(data_dir, args.start + h - 1)
            if prev_true is not None:
                flux_t = float(flux_balance_residual_local(
                    to_device(prev_true, device), to_device(target, device), crit, norm,
                    geo_raw, relative=True, seed=args.seed))
                flux_p = float(flux_balance_residual_local(
                    prev_pred, curr_pred, crit, norm,
                    geo_raw, relative=True, seed=args.seed))
            else:
                flux_t = flux_p = float('nan')
        print(f"{h:>5} {eM:>11.4f} {eE:>11.4f} {eL:>11.4f} {dp:>14.4e} {dt:>14.4e} "
              f"{ndp:>16.4e} {ndt:>17.4e} {flux_p:>13.4e} {flux_t:>13.4e} {gmax:>11.2f}")
        print(f"      [norm-divB transition] step {h:>2}: disk {ndp:.4e}/{ndt:.4e} | global {ndp_glob:.4e}/{ndt_glob:.4e}")

    print(f"\n  Init (t=0): |divB| = {d0:.4e} [code units], norm-divB disk = {nd0:.4e} (global = {nd0_glob:.4e})")
    print(f"  Note on training loss terms vs out-of-sample diagnostics:"
          f"\n    - sl-err M: Rest-mass shell profile error (ONLY conserved shell term in TRON training loss, w=0.03)."
          f"\n    - sl-err E, L: Conserved energy & angular momentum transport (OUT-OF-SAMPLE physical evaluation)."
          f"\n    - Note: E/L are out-of-sample with shell magnitudes ~1e3-1e5, so sl-err values are not comparable to M's scale."
          f"\n    - |divB|: Volume-averaged physical divergence mean(|div B|) [code units] (training loss term, w=0.10)."
          f"\n    - norm-divB: Normalized solenoidal residual mean(|div B| * Delta x_local / |B|_local) in active disk [dimensionless; global printed alongside as transition aid]."
          f"\n    - flux: Local cellwise continuity residual (sampled from active-disk cells, out-of-sample physical evaluation; absolute scale is snapshot-sampling error over dt=25 Rg/c; the model≈truth ratio is the result)."
          f"\n    - gmax: Max pre-clamp Lorentz factor gamma = alpha * u^0 (guards loss term, w=0.10).")


# ============================================================================
#  Computational Cost
# ============================================================================

def evaluate_cost(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*50}")
    print(" Computational Cost")
    print(f"{'='*50}")
    print(f"  Total parameters:     {n_params:,} ({n_params/1e6:.2f}M)")

    init = load_dump(data_dir, args.start)
    if init is None:
        init = load_dump(data_dir, dump_ids(data_dir)[0])
    if init is not None:
        curr = to_device(init, device)
        with torch.no_grad():
            for _ in range(3):
                curr = forecast_one_step(model, norm, curr, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        n_time = 20
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_time):
                curr = forecast_one_step(model, norm, curr, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n_time * 1000
        dt = float(geo['dt_dump']) if geo['dt_dump'] else 25.0
        print(f"  Inference time:       {ms:.1f} ms/step ({device.type})")
        print(f"  Physical time/step:   {dt:.1f} R_g/c (from geometry dt_dump)")
        print(f"  Forecast of {args.max_steps} steps: {ms*args.max_steps/1000:.1f} s")


# ============================================================================
#  Spectra (windowed, like the spectral loss)
# ============================================================================

def plot_spectrum(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    init = load_dump(data_dir, args.start)
    if init is None:
        print(f"Dump {args.start} not found."); return
    n_steps = args.max_steps
    while n_steps > 1 and load_dump(data_dir, args.start + n_steps) is None:
        n_steps -= 1
    target = load_dump(data_dir, args.start + n_steps)
    if target is None:
        print(f"dump {args.start + n_steps} missing"); return
    states = rollout_predictions(model, norm, init, n_steps, device, dump_id=args.start)
    pred = states[n_steps]
    pred_norm = norm_cpu(torch.from_numpy(pred).unsqueeze(0).float())
    target_norm = norm_cpu(torch.from_numpy(target).unsqueeze(0).float())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    kmax = None
    for ax, ch, name in [(axes[0], 0, r'$\rho$'), (axes[1], 5, r'$B^r$')]:
        kp, sp = radial_spectrum(pred_norm, crit, ch)
        kt, st = radial_spectrum(target_norm, crit, ch)
        sl_p, r2_p = spectral_slope(kp[1:], sp[1:])
        sl_t, r2_t = spectral_slope(kt[1:], st[1:])
        ax.loglog(kt[1:], st[1:], 'b-', label=f'Truth (slope {sl_t:.2f}, R²={r2_t:.2f})')
        ax.loglog(kp[1:], sp[1:], 'r--', label=f'TRON (slope {sl_p:.2f}, R²={r2_p:.2f})')
        kmax_ch = int(max(kt.max(), kp.max()))
        if kmax is None:
            kmax = kmax_ch
        ax.set_xlim(1.0, kmax_ch)
        ax.margins(x=0)
        ax.set_xlabel(r'Radial mode index $|k|$')
        ax.set_ylabel(r'Power $|\mathcal{F}[\widetilde{S}]|^2$ (Hann-windowed, norm-space)')
        ax.set_title(f'{name} radial spectrum (h={n_steps})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if ch == 0:                      # stash rho's spectra for the T(k) panel
            k0t, s0t, s0p = kt, st, sp
    # Transfer function T(k) = E_pred/E_true per mode: scale-resolved power ratio.
    # T(k) masked where truth power < 10th percentile (p10) to avoid noise amplification.
    ax = axes[2]
    p10_rho = np.percentile(s0t[1:], 10)
    m_rho = s0t[1:] >= p10_rho
    ratio_rho = s0p[1:][m_rho] / (s0t[1:][m_rho] + 1e-30)
    ax.plot(k0t[1:][m_rho], ratio_rho, '.-', color=CHANNEL_COLORS[0], label=r'$\rho$')

    kp2, sp2 = radial_spectrum(pred_norm, crit, 5)
    kt2, st2 = radial_spectrum(target_norm, crit, 5)
    p10_br = np.percentile(st2[1:], 10)
    m_br = st2[1:] >= p10_br
    ratio_br = sp2[1:][m_br] / (st2[1:][m_br] + 1e-30)
    ax.plot(kt2[1:][m_br], ratio_br, '.-', color=CHANNEL_COLORS[5], label=r'$B^r$')

    ax.axhline(1.0, color='k', linewidth=0.8, linestyle=':')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1.0, kmax if kmax is not None else int(kt2.max()))
    ax.margins(x=0)
    ax.set_xlabel(r'Radial mode index $|k|$')
    ax.set_ylabel(r'Power ratio $T(|k|) = E_{\rm pred}/E_{\rm true}$')
    ax.set_title('Scale-resolved power ratio')
    # Clipped symmetric log limits around unity (0.1 to 10 by default, expanded if needed)
    all_r = np.concatenate([ratio_rho, ratio_br])
    pos_r = all_r[all_r > 0]
    if len(pos_r):
        r_p1 = float(np.percentile(pos_r, 1))
        r_p99 = float(np.percentile(pos_r, 99))
        y_lo = max(1e-2, min(0.1, r_p1 * 0.8))
        y_hi = min(1e2, max(10.0, r_p99 * 1.2))
        ax.set_ylim(y_lo, y_hi)
    else:
        ax.set_ylim(0.1, 10.0)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    path = os.path.join(args.save_dir, "fig6_windowed_spectra.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    print(f"  Fig 6: Folded radial spectra (|k| in [1, {kmax}]):")
    print(f"    rho power ratio: mean={float(np.mean(ratio_rho)):.3f}, median={float(np.median(ratio_rho)):.3f}, range=[{float(np.min(ratio_rho)):.3f}, {float(np.max(ratio_rho)):.3f}]")
    print(f"    B^r power ratio: mean={float(np.mean(ratio_br)):.3f}, median={float(np.median(ratio_br)):.3f}, range=[{float(np.min(ratio_br)):.3f}, {float(np.max(ratio_br)):.3f}]")
    print("  Note: Single-snapshot radial spectra (unweighted cell-mean spectra, relative comparison only; T(k) masked at truth power < p10).")


# ============================================================================
#  Accretion Rate Mdot(r)
# ============================================================================

def plot_mdot(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    init = load_dump(data_dir, args.start)
    if init is None:
        print(f"Dump {args.start} not found."); return
    r_mid = geo_raw['r'][:, geo_raw['r'].shape[1] // 2, 0]
    a_spin = float(geo_raw['spin']) if 'spin' in geo_raw else 0.0
    r_H, r_isco = kerr_horizon_isco(a_spin)
    horizons = auto_horizons(args.max_steps)
    states = rollout_predictions(model, norm, init, max(horizons), device, dump_id=args.start)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    r_mask = (r_mid >= r_H) & (r_mid <= R_DISK_MAX)
    r_plot = r_mid[r_mask]
    all_M_vals = []

    cmap = plt.cm.tab10
    horizon_colors = {1: '#2ca02c', 10: '#ff7f0e', 25: '#d62728', 50: '#9467bd', 100: '#8c564b'}
    def get_h_color(h, idx):
        return horizon_colors.get(h, cmap(idx % 10))

    truth_lines, truth_labels = [], []
    fc_lines, fc_labels = [], []
    for idx, h in enumerate(horizons):
        target = load_dump(data_dir, args.start + h)
        if target is None:
            continue
        col = get_h_color(h, idx)
        M_p = mdot_shells(states[h], crit, norm=norm, bare=False)[0, 0].cpu().numpy()
        M_t = mdot_shells(target, crit, norm=norm, bare=False)[0, 0].cpu().numpy()
        line_t, = axes[0].plot(r_plot, M_t[r_mask], '-', color=col, linewidth=2.0, alpha=1.0)
        line_p, = axes[0].plot(r_plot, M_p[r_mask], '--', color=col, linewidth=1.5)
        truth_lines.append(line_t)
        truth_labels.append(f'Truth t+{h}')
        fc_lines.append(line_p)
        fc_labels.append(f'Forecast t+{h}')
        err = (np.abs(M_p - M_t) / (np.abs(M_t) + 1e-30))[r_mask]
        axes[1].plot(r_plot, np.maximum(err, 1e-3), color=col, linewidth=1, label=f'h={h}')
        all_M_vals.extend(M_t[r_mask])
        all_M_vals.extend(M_p[r_mask])

    all_M_arr = np.array(all_M_vals) if all_M_vals else np.array([1.0])
    abs_M = np.abs(all_M_arr)
    pos_M = abs_M[abs_M > 0]
    if len(pos_M) > 0:
        p5 = float(np.percentile(pos_M, 5))
        linthresh_exp = int(np.floor(np.log10(max(1e-4, p5))))
        linthresh = float(10 ** linthresh_exp)
    else:
        linthresh = 0.1
        linthresh_exp = -1

    max_M = float(np.max(abs_M)) if len(abs_M) > 0 else 10.0
    max_decade = int(np.ceil(np.log10(max(max_M, linthresh * 10))))
    min_decade = linthresh_exp

    pos_ticks = [float(10 ** d) for d in range(min_decade, max_decade + 1)]
    min_M = float(np.min(all_M_arr)) if len(all_M_arr) > 0 else 0.0
    neg_ticks = []
    if min_M < -linthresh:
        neg_min_dec = int(np.ceil(np.log10(abs(min_M))))
        neg_ticks = [-float(10 ** d) for d in range(min_decade, neg_min_dec + 1)]
    yticks = sorted(neg_ticks + [0.0] + pos_ticks)

    axes[0].set_ylabel(r'$\dot{M}(r) = -\int \rho u^r \sqrt{-g}\,d\theta d\varphi\;[\rm code\;units]$')
    axes[0].set_xscale('log')
    axes[0].set_yscale('symlog', linthresh=linthresh, linscale=1.0)
    axes[0].set_yticks(yticks)
    axes[0].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[0].set_xlim(r_H, R_DISK_MAX)
    axes[0].margins(x=0)
    add_isco_line(axes[0], r_isco)
    axes[0].set_title('Accretion Rate Profiles')
    axes[0].legend(truth_lines + fc_lines, truth_labels + fc_labels, fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)
    print(f"  Fig 7: Mdot y-symlog linthresh={linthresh}, yticks={yticks}")

    axes[1].set_ylabel('Rel. error')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlim(r_H, R_DISK_MAX)
    axes[1].margins(x=0)
    axes[1].set_xlabel(r'$r\;[R_g]$')
    add_isco_line(axes[1], r_isco)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(args.save_dir, "fig7_mdot_profile.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================================
#  Tables 8+9 + Fig 18: Accretion Diagnostics — magnetic flux Phi(r), horizon
#  flux, and the Mdot(t) light curve
# ============================================================================

def evaluate_accretion_diagnostics(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    """Accretion-physics diagnostics:
      (a) Magnetic flux threading Phi(r) = full-sphere int |B^r| sqrt(-g) dtheta dphi per shell.
      (b) Blandford-Znajek jet power P_jet(r) (funnel: sigma > 1, polar cut |theta-pi/2| > 1.0, outward flux).
      (c) Dimensionless horizon flux phi_BH = (1/2) Phi(r_H) / sqrt(|Mdot(r_H)|) using hemisphere flux (1/2) Phi(r_H).
      (d) Mdot(t) light curves at r_H and r = 10 R_g over the rollout, and Pearson correlation.
      Note: Phi is full-sphere integral; phi_BH uses hemisphere flux (1/2) Phi(r_H)."""
    init = load_dump(data_dir, args.start)
    if init is None:
        print(f"Dump {args.start} not found."); return
    r_mid = geo_raw['r'][:, geo_raw['r'].shape[1] // 2, 0]
    horizons = auto_horizons(args.max_steps)
    dt = float(geo_raw['dt_dump']) if 'dt_dump' in geo_raw else 25.0

    print(f"\n{'='*76}")
    print(" Table 8: Magnetic Flux Threading (pred vs solver truth; full-sphere Phi)")
    print(f"{'='*76}")
    print(f"{'Steps':>5} {'Phi(r_H) p/t':>20} {'Phi(10) p/t':>20} "
          f"{'rel err r_H':>14} {'rel err r=10':>14}")
    print("-" * 76)
    r10 = int(np.argmin(np.abs(r_mid - 10.0)))
    gamma = float(geo_raw['eos_gamma']) if 'eos_gamma' in geo_raw else 5.0 / 3.0
    a_spin = float(geo_raw['spin']) if 'spin' in geo_raw else 0.0
    r_H, r_isco = kerr_horizon_isco(a_spin)
    r_H_idx = int(np.argmin(np.abs(r_mid - r_H)))

    # Single forward rollout for all horizons and light curve steps
    states = rollout_predictions(model, norm, init, args.max_steps, device, dump_id=args.start)

    tab9 = []   # stashed jet rows
    saved_phi = {}     # h -> (Phi_p, Phi_t)
    saved_pjet = {}    # h -> (pjet_p, pjet_t)
    saved_phi_bh = {}  # h -> (phi_bh_p, phi_bh_t)
    for h in horizons:
        target = load_dump(data_dir, args.start + h)
        if target is None:
            continue
        pred_np = states[h]
        with torch.no_grad():
            Phi_p = magnetic_flux_shells(pred_np, crit, bare=False)[0, 0].cpu().numpy()
            Phi_t = magnetic_flux_shells(target, crit, bare=False)[0, 0].cpu().numpy()
        saved_phi[h] = (Phi_p, Phi_t)
        rel = np.abs(Phi_p - Phi_t) / (np.abs(Phi_t) + 1e-30)
        print(f"{h:>5} {Phi_p[r_H_idx]:>9.3e}/{Phi_t[r_H_idx]:<9.3e} "
              f"{Phi_p[r10]:>9.3e}/{Phi_t[r10]:<9.3e} "
              f"{rel[r_H_idx]:>14.3e} {rel[r10]:>14.3e}")

        # Blandford-Znajek jet power at r_H and the dimensionless horizon flux
        # phi_BH = (1/2) Phi(r_H) / sqrt(Mdot(r_H)) in physical code units (SANE ~ 15, MAD ~ 50).
        with torch.no_grad():
            Mp = mdot_shells(pred_np, crit, norm=norm, bare=False)[0, 0].cpu().numpy()
            Mt = mdot_shells(target, crit, norm=norm, bare=False)[0, 0].cpu().numpy()
        pjet_p, mask_p = jet_power_shells(pred_np, geo_raw, norm_cpu, gamma, bare=False, return_mask=True)
        pjet_t, mask_t = jet_power_shells(target, geo_raw, norm_cpu, gamma, bare=False, return_mask=True)
        saved_pjet[h] = (pjet_p, pjet_t)
        fill_p = float(mask_p[r_H_idx].mean())
        fill_t = float(mask_t[r_H_idx].mean())

        pj_p_val = pjet_p[r_H_idx]
        pj_t_val = pjet_t[r_H_idx]
        Mp_val = Mp[r_H_idx]
        Mt_val = Mt[r_H_idx]

        phi_bh_p = 0.5 * Phi_p[r_H_idx] / np.sqrt(np.abs(Mp_val) + 1e-30)
        phi_bh_t = 0.5 * Phi_t[r_H_idx] / np.sqrt(np.abs(Mt_val) + 1e-30)
        saved_phi_bh[h] = (phi_bh_p, phi_bh_t)
        rel_ph = np.abs(phi_bh_p - phi_bh_t) / (np.abs(phi_bh_t) + 1e-30)

        if np.abs(pj_t_val) < 1e-12 or np.abs(Mt_val) < 1e-12:
            eta_t_str = "n/a      "
            rel_pj_str = "           n/a"
        else:
            eta_t_val = pj_t_val / (np.abs(Mt_val) + 1e-30)
            eta_t_str = f"{eta_t_val:<9.3e}"
            rel_pj_val = np.abs(pj_p_val - pj_t_val) / (np.abs(pj_t_val) + 1e-30)
            rel_pj_str = f"{rel_pj_val:>14.3e}"

        if np.abs(Mt_val) < 1e-12:
            eta_p_str = "     n/a"
        elif np.abs(Mp_val) < 1e-12:
            eta_p_str = "0.000e+00"
        else:
            eta_p_val = pj_p_val / (np.abs(Mp_val) + 1e-30)
            eta_p_str = f"{eta_p_val:>8.3e}"

        tab9.append((h, fill_p, fill_t, pj_p_val, pj_t_val, eta_p_str, eta_t_str,
                     phi_bh_p, phi_bh_t, rel_pj_str, rel_ph))

    print(f"\n{'='*106}")
    print(f" Table 9: Jet Power & Horizon Flux (BZ funnel at r_H = {r_H:.3f} R_g; phi_BH uses 1/2 Phi)")
    print(f"{'='*106}")
    print(f"{'Steps':>5} {'fill p/t':>11} {'Pjet(r_H) p/t':>20} {'eta p/t':>18} "
          f"{'phi_BH(r_H) p/t':>18} {'rel err Pjet':>14} {'rel err phi':>14}")
    print("-" * 106)
    for (h, fill_p, fill_t, pj_p, pj_t, eta_p_str, eta_t_str, phi_p, phi_t, rel_pj_str, rel_ph) in tab9:
        fill_str = f"{fill_p:.3f}/{fill_t:.3f}"
        pj_str = f"{pj_p:>9.3e}/{pj_t:<9.3e}"
        eta_str = f"{eta_p_str}/{eta_t_str}"
        phi_str = f"{phi_p:>8.3e}/{phi_t:<9.3e}"
        print(f"{h:>5} {fill_str:>11} {pj_str:>20} {eta_str:>18} {phi_str:>18} "
              f"{rel_pj_str} {rel_ph:>14.3e}")
    print(f"\n  Note: fill p/t = funnel mask fill fraction (sigma>1 & |theta-pi/2|>1.0) at r_H; phi_BH uses hemisphere flux (1/2) Phi(r_H);"
          f"\n  Pjet is signed net outward energy flux (rel err & eta truth reported as n/a when |truth| < 1e-12).")

    # (d) Mdot(t) light curves over rollout using same cached states
    n_steps = args.max_steps
    mdot_p = {'r_H': [], 'r10': []}
    mdot_t = {'r_H': [], 'r10': []}
    steps = []
    for step in range(1, n_steps + 1):
        true = load_dump(data_dir, args.start + step)
        if true is None or step not in states:
            break
        pred_np = states[step]
        with torch.no_grad():
            mp = mdot_shells(pred_np, crit, norm=norm, bare=False)[0, 0].cpu().numpy()
            mt = mdot_shells(true, crit, norm=norm, bare=False)[0, 0].cpu().numpy()
        mdot_p['r_H'].append(mp[r_H_idx]); mdot_p['r10'].append(mp[r10])
        mdot_t['r_H'].append(mt[r_H_idx]); mdot_t['r10'].append(mt[r10])
        steps.append(step)
    if len(steps) < 3:
        print("  (Mdot light curve needs >= 3 steps)")
    else:
        steps = np.array(steps)
        for lab in ['r_H', 'r10']:
            a, b = np.array(mdot_p[lab]), np.array(mdot_t[lab])
            cc = np.corrcoef(a, b)[0, 1] if (np.std(a) != 0 and np.std(b) != 0) else float('nan')
            print(f"\n  Mdot(t) at {lab}: Pearson r(model, truth) = {cc:.3f} "
                  f"(n={len(steps)} steps, dt={dt:.0f} R_g/c)")
            print(f"    model: mean {np.mean(a):.4e} std {np.std(a):.4e} | "
                  f"truth: mean {np.mean(b):.4e} std {np.std(b):.4e}")

    # figure: 1x3 accretion & jet suite with multi-horizon forecasts
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    r_mask = (r_mid >= r_H) & (r_mid <= R_DISK_MAX)
    r_plot = r_mid[r_mask]
    cmap = plt.cm.tab10
    horizon_colors = {1: '#2ca02c', 10: '#ff7f0e', 25: '#d62728', 50: '#9467bd', 100: '#8c564b'}
    def get_h_color(h, idx):
        return horizon_colors.get(h, cmap(idx % 10))

    # (a) Left: Phi(r) profiles across horizons
    ax_a = axes[0]
    if horizons[0] in saved_phi:
        ax_a.loglog(r_plot, saved_phi[horizons[0]][1][r_mask], 'b-', linewidth=2.0, label='Truth')
    for idx, h in enumerate(horizons):
        if h in saved_phi:
            col = get_h_color(h, idx)
            ax_a.loglog(r_plot, saved_phi[h][0][r_mask], linestyle='--', color=col, linewidth=1.5,
                        label=f'TRON h={h}')
    ax_a.set_xlim(r_H, R_DISK_MAX)
    ax_a.margins(x=0)
    add_isco_line(ax_a, r_isco)
    ax_a.set_xlabel(r'$r\;[R_g]$')
    ax_a.set_ylabel(r'$\Phi(r) = \int |B^r| \sqrt{-g}\,d\theta d\varphi\;[\rm code\;units]$')
    ax_a.set_title('(a) Magnetic Flux Threading (cumulative shell flux)')
    ax_a.legend(fontsize=8)
    ax_a.grid(True, alpha=0.3)

    # (b) Middle: P_jet(r) profiles across horizons
    ax_b = axes[1]
    all_pj_vals = []
    if horizons[0] in saved_pjet:
        pj_t = saved_pjet[horizons[0]][1][r_mask]
        ax_b.plot(r_plot, pj_t, 'b-', linewidth=2.0, label='Truth')
        all_pj_vals.extend(pj_t)
    for idx, h in enumerate(horizons):
        if h in saved_pjet:
            col = get_h_color(h, idx)
            pj_p = saved_pjet[h][0][r_mask]
            ax_b.plot(r_plot, pj_p, linestyle='--', color=col, linewidth=1.5,
                      label=f'TRON h={h}')
            all_pj_vals.extend(pj_p)

    all_pj_arr = np.array(all_pj_vals) if all_pj_vals else np.array([1.0])
    abs_pj = np.abs(all_pj_arr)
    nz_pj = abs_pj[abs_pj != 0]
    if len(nz_pj):
        p5_pj = float(np.percentile(nz_pj, 5))
        linthresh_exp_pj = int(np.floor(np.log10(max(1e-4, p5_pj))))
        linthresh_pj = float(10 ** linthresh_exp_pj)
    else:
        linthresh_pj = 0.1
        linthresh_exp_pj = -1

    p99_8_pj = float(np.percentile(abs_pj, 99.8))
    y_hi_pj = p99_8_pj * 1.1
    max_decade_pj = int(np.ceil(np.log10(max(y_hi_pj, linthresh_pj * 10))))
    min_decade_pj = linthresh_exp_pj
    pos_ticks_pj = [float(10 ** d) for d in range(min_decade_pj, max_decade_pj + 1)]

    neg_pj = all_pj_arr[all_pj_arr < 0]
    n_neg_pj = int(len(neg_pj))
    neg_ticks_pj = []
    if n_neg_pj > 0:
        most_neg_pj = float(np.min(neg_pj))
        y_lo_pj = min(most_neg_pj * 1.1, -linthresh_pj * 1.1)
        neg_ticks_pj = [-float(10 ** d) for d in range(min_decade_pj, max_decade_pj + 1)]
    else:
        most_neg_pj = 0.0
        y_lo_pj = float(10 ** min_decade_pj)

    yticks_pj = [t for t in sorted(neg_ticks_pj + [0.0] + pos_ticks_pj) if y_lo_pj <= t <= y_hi_pj]

    ax_b.set_xscale('log')
    ax_b.set_yscale('symlog', linthresh=linthresh_pj, linscale=1.0)
    ax_b.set_yticks(yticks_pj)
    ax_b.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_b.set_ylim(y_lo_pj, y_hi_pj)
    ax_b.set_xlim(r_H, R_DISK_MAX)
    ax_b.margins(x=0)
    add_isco_line(ax_b, r_isco)
    ax_b.set_xlabel(r'$r\;[R_g]$')
    ax_b.set_ylabel(r'$P_{\rm jet}(r) = -\int_{\rm funnel} T^r{}_t \sqrt{-g}\,d\theta d\varphi\;[\rm code\;units]$')
    ax_b.set_title(r'(b) Blandford-Znajek Jet Power (signed net flux)')
    ax_b.legend(fontsize=8)
    ax_b.grid(True, alpha=0.3)

    max_pj = float(np.max(abs_pj))
    print(f"  Fig 18: P_jet bounds: max={max_pj:.3e}, p99.8={p99_8_pj:.3e}, most_negative={most_neg_pj:.3e} -> adopted ylim=({y_lo_pj:.3e}, {y_hi_pj:.3e}), linthresh={linthresh_pj:.1e}, negative_shells={n_neg_pj}")

    # (c) Right: phi_BH(r_H) vs forecast horizon
    ax_c = axes[2]
    h_list = [h for h in horizons if h in saved_phi_bh]
    if h_list:
        phi_t_vals = [saved_phi_bh[h][1] for h in h_list]
        phi_p_vals = [saved_phi_bh[h][0] for h in h_list]
        ax_c.plot(h_list, phi_t_vals, 'b-o', linewidth=1.5, markersize=5, label=r'Truth $\phi_{\rm BH}(r_H)$')
        ax_c.plot(h_list, phi_p_vals, 'r--s', linewidth=1.5, markersize=5, label=r'TRON $\phi_{\rm BH}(r_H)$')
        all_phi = phi_t_vals + phi_p_vals + [15.0]
        ax_c.set_ylim(0, max(all_phi) * 1.25)
        max_h = max(h_list) if h_list else args.max_steps
        h_margin = max(1.0, max_h * 0.02)
        ax_c.set_xlim(0, max_h + h_margin)
        ax_c.set_xlabel('Forecast Horizon $h$ [steps]')
        ax_c.set_ylabel(r'$\phi_{\rm BH} = \frac{1}{2}\Phi(r_H)/\sqrt{|\dot{M}(r_H)|}$')
        ax_c.set_title(r'(c) Dimensionless Horizon Flux $\phi_{\rm BH}$')
        ax_c.legend(fontsize=8)
        ax_c.grid(True, alpha=0.3)

    fig.suptitle('Accretion & Jet Diagnostics', fontsize=13)
    plt.tight_layout()
    path = os.path.join(args.save_dir, "fig18_accretion_diagnostics.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\nSaved: {path}")


# ============================================================================
#  Temporal Conservation: E, L_z, M totals (model drift vs truth drift)
# ============================================================================

def evaluate_temporal_consistency(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    """Total conservation drift over rollout: M, E, L totals pred vs truth.
    Measures the long-horizon stability of total conserved quantities."""
    init = load_dump(data_dir, args.start)
    if init is None:
        print(f"Dump {args.start} not found."); return
    target_steps = args.max_steps
    with torch.no_grad():
        M0 = float(torch.sum(mass_shells(init, crit, norm)).item())
        E0, L0 = conserved_shells(init, crit, norm)
        E0 = float(torch.sum(E0).item())
        L0 = float(torch.sum(L0).item())

        # Physical normalization scales:
        # Internal energy U0 = sum(ug * sqrt(-g)) bare at t=0
        x0 = to_device(init, crit.gdet.device)
        ug0 = crit._safe_rho_ug(x0[:, 1:2], CONFIG["floor_ug"], CONFIG["ceiling_ug"])
        U0 = float(torch.sum(ug0 * crit.gdet).item())

        # Characteristic angular momentum scale Lscale = M0 * r_ISCO
        a_spin = float(geo_raw['spin']) if 'spin' in geo_raw else 0.0
        r_H, r_isco = kerr_horizon_isco(a_spin)
        Lscale = abs(M0) * r_isco

    states = rollout_predictions(model, norm, init, target_steps, device, dump_id=args.start)
    steps, M_p, E_p, L_p, M_t, E_t, L_t = [], [], [], [], [], [], []
    for step in range(1, target_steps + 1):
        true = load_dump(data_dir, args.start + step)
        if true is None or step not in states:
            break
        pred = states[step]
        with torch.no_grad():
            M_p.append(float(torch.sum(mass_shells(pred, crit, norm)).item()))
            e, l = conserved_shells(pred, crit, norm)
            E_p.append(float(torch.sum(e).item()))
            L_p.append(float(torch.sum(l).item()))
            M_t.append(float(torch.sum(mass_shells(true, crit, norm)).item()))
            e, l = conserved_shells(true, crit, norm)
            E_t.append(float(torch.sum(e).item()))
            L_t.append(float(torch.sum(l).item()))
        steps.append(step)

    if not steps:
        return
    steps = np.array(steps)
    M_p, E_p, L_p = np.array(M_p), np.array(E_p), np.array(L_p)
    M_t, E_t, L_t = np.array(M_t), np.array(E_t), np.array(L_t)

    scale_M = abs(M0)
    scale_E = abs(E0)
    if scale_E < 1e-6 * U0:
        print(f"  [Drift Scale] |E0|={scale_E:.2e} < 1e-6 * U0 ({U0:.2e}); falling back to scale_E = U0")
        scale_E = U0
        lbl_E = "U_int,0"
    else:
        lbl_E = "|E0|"

    scale_L = abs(L0)
    if scale_L < 1e-6 * Lscale:
        print(f"  [Drift Scale] |L0|={scale_L:.2e} < 1e-6 * Lscale ({Lscale:.2e}); falling back to scale_L = Lscale")
        scale_L = Lscale
        lbl_L = "M0*r_ISCO"
    else:
        lbl_L = "|L0|"

    drift_M = lambda m: np.abs(m - M0) / (scale_M + 1e-30)
    drift_E = lambda e: np.abs(e - E0) / (scale_E + 1e-30)
    drift_L = lambda l: np.abs(l - L0) / (scale_L + 1e-30)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [('Mass M_total (scale |M0|)', M_p, M_t, drift_M),
              (f'Energy E_total (scale {lbl_E})', E_p, E_t, drift_E),
              (f'AngMom L_total (scale {lbl_L})', L_p, L_t, drift_L)]
    for ax, (name, p, t, d_fn) in zip([axes[0, 0], axes[0, 1], axes[1, 0]], panels):
        ax.plot(steps, d_fn(p), 'r-', linewidth=1.5, label='TRON drift')
        ax.plot(steps, d_fn(t), 'b--', linewidth=1.5, label='Solver truth drift')
        ax.set_yscale('log')
        ax.set_xlabel('Forecast step')
        ax.set_ylabel('Drift / scale')
        ax.set_title(f'{name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[1, 1].axis('off')
    txt = (f"Conservation drift after {len(steps)} steps (TRON vs solver):\n"
           f"Scales: M:|M0|={scale_M:.2e}, E:{lbl_E}={scale_E:.2e}, L:{lbl_L}={scale_L:.2e}\n\n"
           f"  Mass:    {drift_M(M_p)[-1]:.3e}  vs  {drift_M(M_t)[-1]:.3e}\n"
           f"  Energy:  {drift_E(E_p)[-1]:.3e}  vs  {drift_E(E_t)[-1]:.3e}\n"
           f"  AngMom:  {drift_L(L_p)[-1]:.3e}  vs  {drift_L(L_t)[-1]:.3e}\n\n"
           f"max TRON drift: M {drift_M(M_p).max():.3e} / E {drift_E(E_p).max():.3e} "
           f"/ L {drift_L(L_p).max():.3e}")
    axes[1, 1].text(0.05, 0.5, txt, fontsize=11, family='monospace',
                    verticalalignment='center', transform=axes[1, 1].transAxes)
    plt.tight_layout()
    path = os.path.join(args.save_dir, "temporal_consistency.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    print(f"  Final drift (model / solver truth):\n"
          f"    M (scale |M0|):       {drift_M(M_p)[-1]:.3e} / {drift_M(M_t)[-1]:.3e}\n"
          f"    E (scale {lbl_E}):   {drift_E(E_p)[-1]:.3e} / {drift_E(E_t)[-1]:.3e}\n"
          f"    L (scale {lbl_L}): {drift_L(L_p)[-1]:.3e} / {drift_L(L_t)[-1]:.3e}")
    np.savez(os.path.join(args.save_dir, "temporal_consistency.npz"),
             M_pred=M_p, E_pred=E_p, L_pred=L_p,
             M_true=M_t, E_true=E_t, L_true=L_t,
             M0=M0, E0=E0, L0=L0, U0=U0, Lscale=Lscale, steps=steps)


# ============================================================================
#  Lorentz Validity (gamma stats, clamp counts)
# ============================================================================

def evaluate_lorentz(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    init = load_dump(data_dir, args.start)
    if init is None:
        print(f"Dump {args.start} not found."); return
    g_t = gamma_stats(init, norm, device)
    print(f"\n{'='*60}")
    print(" Lorentz Validity (pre-clamp gamma = alpha * u^0)")
    print(f"{'='*60}")
    print(f"  Truth t=0: max {g_t.max():.3f} | p99.9 {np.percentile(g_t, 99.9):.3f} "
          f"| > 18: {(g_t > 18).sum():,} cells | > max_gamma(50): {(g_t > 50).sum():,}")
    n_clamp = 0
    gmax = 0.0
    gmax_step = 0
    p999_max = 0.0
    last_step = 0
    states = rollout_predictions(model, norm, init, args.max_steps, device, dump_id=args.start)
    clamps = get_rollout_clamps(args.start)
    for step in range(1, args.max_steps + 1):
        true = load_dump(data_dir, args.start + step)
        if true is None or step not in states:
            break
        g = gamma_stats(states[step], norm, device)
        curr_gmax = float(g.max())
        if curr_gmax > gmax:
            gmax = curr_gmax
            gmax_step = step
        p999_max = max(p999_max, float(np.percentile(g, 99.9)))
        n_clamp += clamps.get(step, 0)
        last_step = step
    print(f"  Rollout ({last_step} steps): max gamma {gmax:.3f} (step {gmax_step}) | max p99.9 {p999_max:.3f} | Lorentz clamp fires {n_clamp:,}")


# ============================================================================
#  Spatial Error: radial profiles + equatorial vs polar (fixed masks)
# ============================================================================

def compute_radial_error_profile(pred, true, geo_raw, theta_mask=None):
    r = geo_raw['r']
    Nr, Nt, Np = r.shape
    r_mid = r[:, Nt // 2, Np // 2]
    n_bins = max(8, min(24, Nr // 4))
    edges = np.geomspace(r_mid.min(), r_mid.max(), n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean bin center for log-spaced grid
    prof = np.full((n_bins, 8), np.nan)
    for i in range(n_bins):
        r_mask = (r_mid >= edges[i]) & (r_mid < edges[i + 1])
        if r_mask.sum() == 0:
            continue
        for c in range(8):
            if theta_mask is not None:
                pc = pred[c][r_mask][:, theta_mask, :]
                tc = true[c][r_mask][:, theta_mask, :]
            else:
                pc = pred[c][r_mask]
                tc = true[c][r_mask]
            if tc.size == 0:
                continue
            denom = float(np.abs(tc).mean())
            prof[i, c] = float(np.abs(pc - tc).mean() / denom) if denom > 1e-30 else np.nan
    return centers, prof

def evaluate_spatial_error(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    init = load_dump(data_dir, args.start)
    if init is None:
        print(f"Dump {args.start} not found."); return
    horizons = auto_horizons(args.max_steps)
    valid_h = [h for h in horizons if load_dump(data_dir, args.start + h) is not None]
    if not valid_h:
        return

    # Theta masks: disk core (|theta-pi/2| < 0.5) vs polar jet funnel (|theta-pi/2| > 1.0)
    theta_1d = geo_raw['h'][0, :, 0]
    mask_disk = np.abs(theta_1d - np.pi / 2) < 0.5
    mask_jet = np.abs(theta_1d - np.pi / 2) > 1.0

    preds = rollout_predictions(model, norm, init, max(valid_h), device, dump_id=args.start)

    # --- Figure 1: Fluid channels (0-4: rho, ug, u^r, u^theta, u^phi) ---
    fig_fl, axes_fl = plt.subplots(len(valid_h), 2, figsize=(14, 4.2 * len(valid_h)), sharex=True)
    if len(valid_h) == 1:
        axes_fl = axes_fl.reshape(1, 2)

    # --- Figure 2: Magnetic channels (5-7: B^r, B^theta, B^phi) ---
    fig_mg, axes_mg = plt.subplots(len(valid_h), 2, figsize=(14, 4.2 * len(valid_h)), sharex=True)
    if len(valid_h) == 1:
        axes_mg = axes_mg.reshape(1, 2)

    for idx, n_steps in enumerate(valid_h):
        target = load_dump(data_dir, args.start + n_steps)
        pred = preds[n_steps]
        centers, prof_disk = compute_radial_error_profile(pred, target, geo_raw, mask_disk)
        _, prof_jet = compute_radial_error_profile(pred, target, geo_raw, mask_jet)

        # Plot Fluid (channels 0..4)
        for c in range(5):
            axes_fl[idx, 0].plot(centers, prof_disk[:, c], linewidth=1.5, color=CHANNEL_COLORS[c], label=CHANNEL_NAMES[c])
            axes_fl[idx, 1].plot(centers, prof_jet[:, c], linewidth=1.5, color=CHANNEL_COLORS[c], label=CHANNEL_NAMES[c])
        for col_idx, reg_name in enumerate(['Disk Core', 'Jet Funnel']):
            ax = axes_fl[idx, col_idx]
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_ylabel('Rel. L1 Error')
            ax.set_title(f'Fluid Error — {reg_name} (t+{n_steps})')
            ax.legend(fontsize=8, ncol=2, frameon=True, framealpha=0.9)
            ax.grid(True, alpha=0.3, which='both')
        axes_fl[idx, 0].set_xlabel(r'$r\;[R_g]$')
        axes_fl[idx, 1].set_xlabel(r'$r\;[R_g]$')

        # Plot Magnetic (channels 5..7)
        for c in range(5, 8):
            axes_mg[idx, 0].plot(centers, prof_disk[:, c], linewidth=1.5, color=CHANNEL_COLORS[c], label=CHANNEL_NAMES[c])
            axes_mg[idx, 1].plot(centers, prof_jet[:, c], linewidth=1.5, color=CHANNEL_COLORS[c], label=CHANNEL_NAMES[c])
        for col_idx, reg_name in enumerate(['Disk Core', 'Jet Funnel']):
            ax = axes_mg[idx, col_idx]
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_ylabel('Rel. L1 Error')
            ax.set_title(f'Magnetic Error — {reg_name} (t+{n_steps})')
            ax.legend(fontsize=8, frameon=True, framealpha=0.9)
            ax.grid(True, alpha=0.3, which='both')
        axes_mg[idx, 0].set_xlabel(r'$r\;[R_g]$')
        axes_mg[idx, 1].set_xlabel(r'$r\;[R_g]$')

    fig_fl.suptitle('Spatial Error Profiles — Fluid Channels', fontsize=13)
    fig_fl.tight_layout()
    p_fl = os.path.join(args.save_dir, "spatial_error_fluid.png")
    fig_fl.savefig(p_fl)
    plt.close(fig_fl)
    print(f"Saved: {p_fl}")

    fig_mg.suptitle('Spatial Error Profiles — Magnetic Channels', fontsize=13)
    fig_mg.tight_layout()
    p_mg = os.path.join(args.save_dir, "spatial_error_mag.png")
    fig_mg.savefig(p_mg)
    plt.close(fig_mg)
    print(f"Saved: {p_mg}")


# ============================================================================
#  Fig 8: 2D Meridional (r,theta) Slices — the flagship visualization
# ============================================================================

def b2_magnitude(raw_t, geo_raw):
    """B^2 = gamma_ij B^i B^j using the spatial metric from the geometry.
    NOTE: raw gcov is (4,4,Nr,Nt,1) — the spatial block is the [1:,1:]
    slice of the FULL tensor. Do NOT index [0,0,...] on the raw array:
    that consumes both 4-indices and silently slices Nr and Nt instead
    (the wrapped (1,1,4,4,...) buffer used by the loss needs [0,0,1:,1:];
    the raw array does not)."""
    gam = geo_raw['gcov'][1:, 1:]                   # (3,3,Nr,Nt,1)
    B = raw_t[5:8]                                 # (3,Nr,Nt,Np)
    return np.einsum('iabc,ijabc,jabc->abc', B, gam, B)

def bsq_full_components(raw_t, geo_raw, norm_cpu):
    """Full covariant magnetic pressure b^2 = b^mu b_mu AND the fluid-frame
    field vectors (bcon = b^mu, bcov = b_mu, ucov = u_mu). IDENTICAL to
    analyse_forecast_ds.compute_bsq, so beta/sigma numbers agree between
    the plotting and evaluation scripts. b^mu is the fluid-frame field
    derived from the lab-frame B^i and u^mu (norm_cpu supplies the
    training-identical ucon_of on CPU buffers).
    The fluid-frame vectors are required for the Blandford-Znajek stress
    T^r_t = (w + b^2) u^r u_t - b^r b_t: substituting the LAB-frame B^r for
    b^r there is wrong (b^r = (B^r + b^0 u^r)/u^0), which is the classic
    mistake when evaluating the jet power from primitive output."""
    B1, B2, B3 = raw_t[5], raw_t[6], raw_t[7]
    g = geo_raw['gcov']                             # (4,4,Nr,Nt,1) raw

    x = to_device(raw_t, torch.device('cpu'))
    ucon_t = norm_cpu.ucon_of(x[:, 2:5]).numpy()[0]   # (4,Nr,Nt,Np) coordinate 4-velocity
    u0, u1, u2, u3 = ucon_t[0], ucon_t[1], ucon_t[2], ucon_t[3]

    ucov = np.zeros((4, *u0.shape))
    ucon = np.array([u0, u1, u2, u3])
    for mu in range(4):
        for nu in range(4):
            ucov[mu] += g[mu, nu] * ucon[nu]

    b0 = B1*ucov[1] + B2*ucov[2] + B3*ucov[3]
    u0_safe = np.where(np.abs(u0) > 1e-30, u0, 1e-30)
    bcon = np.array([b0, (B1 + b0*u1)/u0_safe, (B2 + b0*u2)/u0_safe, (B3 + b0*u3)/u0_safe])
    bcov = np.zeros_like(bcon)
    for mu in range(4):
        for nu in range(4):
            bcov[mu] += g[mu, nu] * bcon[nu]
    bsq = np.maximum(sum(bcon[mu]*bcov[mu] for mu in range(4)), 0.0)
    return bsq, bcon, bcov, ucov, ucon

def bsq_full(raw_t, geo_raw, norm_cpu):
    """Full covariant magnetic pressure b^2 = b^mu b_mu (2 p_mag)."""
    return bsq_full_components(raw_t, geo_raw, norm_cpu)[0]

def jet_power_shells(raw_t, geo_raw, norm_cpu, gamma, bare=True, return_mask=False):
    """Blandford-Znajek jet power per shell:
    P_jet(r) = sum_{theta,phi, sigma>1, funnel} (-T^r_t) sqrt(-g)   [code units]
    with -T^r_t = -(rho + gamma u_g + b^2) u^r u_t + b^r b_t using the
    FLUID-frame b^mu (see bsq_full_components). The funnel mask combines
    sigma = b^2/rho > 1 with the polar geometric funnel |theta - pi/2| > 1.0
    (consistent with plot_temporal_power_spectrum), isolating the outflowing jet
    from plunging coronal flow at the horizon. Reports signed net outward energy flux.
    When bare=False, multiplies by dA = dx_2 * dx_3 for physical code units."""
    bsq, bcon, bcov, ucov, ucon = bsq_full_components(raw_t, geo_raw, norm_cpu)
    w = raw_t[0] + gamma * raw_t[1]                 # enthalpy rho + u_g + p
    tr_t = (w + bsq) * ucon[1] * ucov[0] - bcon[1] * bcov[0]
    sigma = bsq / (raw_t[0] + 1e-30)
    h_angle = geo_raw['h']
    floor_rho = float(CONFIG.get("floor_rho", 1e-4))
    funnel = ((sigma > 1.0) &
              (np.abs(h_angle - np.pi / 2.0) > 1.0) &
              ((ucon[1] > 0) | (-tr_t > 0)) &
              (raw_t[0] > 100 * floor_rho))
    mask = funnel.astype(np.float32)
    gdet = geo_raw['gdet'][..., 0]                  # (Nr,Nt) -> broadcast (Nr,Nt,Np)
    res = -np.sum(tr_t * gdet[:, :, None] * mask, axis=(1, 2))
    if not bare:
        dA = float(geo_raw['dx'][1] * geo_raw['dx'][2])
        res = res * dA
    if return_mask:
        return res, mask
    return res

def plot_meridional_slice(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    """Truth / forecast / error maps of log10 rho and log10 B^2 in the
    (r sin theta, r cos theta) meridional plane at the mid phi slice —
    the standard GRMHD-paper flagship figure."""
    init = load_dump(data_dir, args.start)
    if init is None:
        print(f"Dump {args.start} not found."); return
    target_step = args.max_steps
    while target_step > 1 and load_dump(data_dir, args.start + target_step) is None:
        target_step -= 1
    target = load_dump(data_dir, args.start + target_step)
    if target is None:
        print(f"Dump {args.start + target_step} not found (rollout target)."); return
    if target_step < args.max_steps:
        print(f"[Notice] Requested {args.max_steps} steps for Fig 8, but dumps end at step {target_step}. Using t+{target_step}.")
    states = rollout_predictions(model, norm, init, target_step, device, dump_id=args.start)
    pred = states[target_step]

    r = geo_raw['r'][:, :, geo_raw['r'].shape[2] // 2]
    h = geo_raw['h'][:, :, geo_raw['h'].shape[2] // 2]
    X = r * np.sin(h)
    Y = r * np.cos(h)
    pm = pred[:, :, :, pred.shape[3] // 2]             # mid-phi meridional plane
    tm = target[:, :, :, target.shape[3] // 2]

    log_rho_p = np.log10(np.maximum(pm[0], 1e-30))
    log_rho_t = np.log10(np.maximum(tm[0], 1e-30))
    # Error in DEX: |log10 pred - log10 truth|. A raw (pred - truth) error
    # would (a) hide underprediction (clamped to the floor -> log10 = -30,
    # black) and (b) be dominated by the disk midplane where rho is largest
    # (jet errors invisible). dex error is scale-invariant and shows both
    # signs of misprediction equally.
    log_rho_e = np.abs(log_rho_p - log_rho_t)
    b2_p = np.log10(np.maximum(b2_magnitude(pred, geo_raw)[:, :, pred.shape[3] // 2], 1e-30))
    b2_t = np.log10(np.maximum(b2_magnitude(target, geo_raw)[:, :, target.shape[3] // 2], 1e-30))
    b2_e = np.abs(b2_p - b2_t)

    rho_all = np.concatenate([log_rho_t.ravel(), log_rho_p.ravel()])
    rho_lo, rho_hi = float(np.percentile(rho_all, 1)), float(np.percentile(rho_all, 99.5))
    b2_all = np.concatenate([b2_t.ravel(), b2_p.ravel()])
    b2_lo, b2_hi = float(np.percentile(b2_all, 1)), float(np.percentile(b2_all, 99.5))

    b2_ticks = np.arange(0.0, 3.6, 0.5)
    panels = [(r'$\log_{10}\rho$ Truth', log_rho_t, rho_lo, rho_hi, None),
              (r'$\log_{10}\rho$ Forecast', log_rho_p, rho_lo, rho_hi, None),
              (r'$|\Delta \log_{10}\rho|\ [\mathrm{dex}]$', log_rho_e, None, None, None),
              (r'$\log_{10} B^2$ Truth', b2_t, b2_lo, b2_hi, None),
              (r'$\log_{10} B^2$ Forecast', b2_p, b2_lo, b2_hi, None),
              (r'$|\Delta \log_{10} B^2|\ [\mathrm{dex}]$', b2_e, 0.0, 3.5, b2_ticks)]

    def _save_meridional(suffix, xlim=None, ylim=None):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True,
                                 gridspec_kw={'wspace': 0.12, 'hspace': 0.3})
        for ax, (title, Z, vlo, vhi, c_ticks) in zip(axes.ravel(), panels):
            cf = ax.pcolormesh(X, Y, Z, cmap='inferno', shading='auto', vmin=vlo, vmax=vhi)
            cb = fig.colorbar(cf, ax=ax, pad=0.01, fraction=0.046)
            if c_ticks is not None:
                cb.set_ticks(c_ticks)
            ax.set_title(title)
            ax.set_xlabel(r'$r\sin\theta\;[R_g]$')
            ax.set_ylabel(r'$r\cos\theta\;[R_g]$')
            ax.set_aspect('equal')
        if xlim is not None:
            axes[0, 0].set_xlim(*xlim)
        if ylim is not None:
            axes[0, 0].set_ylim(*ylim)
        zoom_str = f" ({suffix.strip('_')})" if suffix else ""
        fig.suptitle(f'Meridional Slices at t+{target_step} steps{zoom_str} (mid phi, seeded rollout)',
                     fontsize=13)
        plt.tight_layout()
        path = os.path.join(args.save_dir, f"fig8_meridional_slice{suffix}.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"Saved: {path}")

    _save_meridional("", xlim=None, ylim=None)
    _save_meridional("_r100", xlim=(0, 100), ylim=(-100, 100))
    _save_meridional("_r250", xlim=(0, 250), ylim=(-250, 250))


# ============================================================================
#  Statistical Properties: PDF (shared bins) + radial structure function
# ============================================================================

def evaluate_statistical_props(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    init = load_dump(data_dir, args.start)
    if init is None:
        print(f"Dump {args.start} not found."); return
    n_steps = max(1, args.max_steps // 2)
    while n_steps > 1 and load_dump(data_dir, args.start + n_steps) is None:
        n_steps -= 1
    target = load_dump(data_dir, args.start + n_steps)
    if target is None:
        print(f"Required dumps not found"); return
    states = rollout_predictions(model, norm, init, n_steps, device, dump_id=args.start)
    pred = states[n_steps]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    metrics = {}
    for c in range(8):
        row, col = c // 4, c % 4
        is_log = c in [0, 1]
        centers, pp, pt, widths = pdf_on_shared_bins(pred[c], target[c], log_scale=is_log)
        js = js_divergence(pp, pt, widths=widths)
        pred_mean = float(np.sum(centers * pp * widths) / (np.sum(pp * widths) + 1e-30))
        true_mean = float(np.sum(centers * pt * widths) / (np.sum(pt * widths) + 1e-30))
        metrics[c] = {'js': js, 'pred_mean': pred_mean, 'true_mean': true_mean}
        axes[row, col].plot(centers, pt, 'b-', label='Truth', linewidth=1.5)
        axes[row, col].plot(centers, pp, 'r--', label='TRON', linewidth=1.5)
        axes[row, col].set_xlabel(f'${CHANNEL_NAMES_PLAIN[c]}$')
        axes[row, col].set_ylabel('PDF (shared bins)')
        axes[row, col].set_title(f'{CHANNEL_NAMES_PLAIN[c]}: JS={js:.4f}')
        axes[row, col].legend(fontsize=7)
        axes[row, col].grid(True, alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(args.save_dir, "statistical_props_pdf.png")
    fig.savefig(p1)
    plt.close(fig)
    print(f"Saved: {p1}")

    print(f"\n{'='*60}")
    print(f" Statistical Properties ({n_steps} steps, shared-bin JS divergence)")
    print(f"{'='*60}")
    print(f"  {'Channel':<10} {'JS Div':>10} {'Pred Mean':>12} {'True Mean':>12}")
    for c in range(8):
        m = metrics[c]
        print(f"  {CHANNEL_NAMES_PLAIN[c]:<10} {m['js']:>10.4f} {m['pred_mean']:>12.4e} {m['true_mean']:>12.4e}")
    np.savez(os.path.join(args.save_dir, "statistical_metrics.npz"),
             js_divergence=np.array([metrics[c]['js'] for c in range(8)]),
             pred_mean=np.array([metrics[c]['pred_mean'] for c in range(8)]),
             true_mean=np.array([metrics[c]['true_mean'] for c in range(8)]))


# ============================================================================
#  Fig 19: Conditional error vs truth magnitude (where in the dynamic range
#  does the model fail?)
# ============================================================================

def evaluate_conditional_error(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    """nMAE binned by TRUTH magnitude quintiles (5 bins), per channel, in normalized
    space (so bins are comparable across channels). Answers 'where in the
    dynamic range is the model good?': floor-dominated cells (jets, ISCO
    vacuum), mid-range disk cells, or the extreme tails. Model vs persistence
    per bin shows whether the model beats the trivial baseline everywhere."""
    ids = dump_ids(data_dir)
    test_ids = [d for d in ids if d >= args.start and (d + 1) in ids]
    if not test_ids:
        print("No test pairs found."); return
    n_bins = 5
    # Compute global pooled quintile edges pre-loop across blind test dumps
    n_sample_dumps = min(50, len(test_ids))
    step_spacing = max(1, len(test_ids) // n_sample_dumps)
    sampled_ids = test_ids[::step_spacing][:n_sample_dumps]
    rng_pool = np.random.default_rng(args.seed)

    pool = [[] for _ in range(8)]
    for did in sampled_ids:
        dump_data = load_dump(data_dir, did)
        if dump_data is None:
            continue
        nd = norm_cpu(to_tensor_cpu(dump_data))[0].numpy()
        n_cells = nd[0].size
        n_sub = max(100, int(0.01 * n_cells))
        sub_idx = rng_pool.choice(n_cells, size=n_sub, replace=False)
        for c in range(8):
            pool[c].append(nd[c].ravel()[sub_idx].astype(np.float32))

    edges = np.zeros((8, n_bins + 1))
    for c in range(8):
        data_c = np.concatenate(pool[c]) if pool[c] else np.zeros(10, dtype=np.float32)
        mag = data_c if c in [0, 1] else np.abs(data_c)
        edges[c] = np.percentile(mag, np.linspace(0, 100, n_bins + 1))

    err_m_sum = np.zeros((n_bins, 8))
    err_p_sum = np.zeros((n_bins, 8))
    cnt = np.zeros((n_bins, 8))
    for dump_id in test_ids:
        curr = load_dump(data_dir, dump_id)
        nxt = load_dump(data_dir, dump_id + 1)
        with torch.no_grad():
            pred = forecast_one_step(model, norm, to_device(curr, device), device).cpu().numpy()[0]
        nt = norm_cpu(to_tensor_cpu(nxt))[0].numpy()       # (8,Nr,Nt,Np)
        np_ = norm_cpu(to_tensor_cpu(pred))[0].numpy()
        nc = norm_cpu(to_tensor_cpu(curr))[0].numpy()
        for c in range(8):
            mag = nt[c] if c in [0, 1] else np.abs(nt[c])
            for b in range(n_bins):
                if b < n_bins - 1:
                    m = (mag >= edges[c, b]) & (mag < edges[c, b + 1])
                else:
                    m = (mag >= edges[c, b]) & (mag <= edges[c, b + 1])
                n_in = int(m.sum())
                if n_in == 0:
                    continue
                err_m_sum[b, c] += np.abs(np_[c][m] - nt[c][m]).sum()
                err_p_sum[b, c] += np.abs(nc[c][m] - nt[c][m]).sum()
                cnt[b, c] += n_in

    err_m = np.full((n_bins, 8), np.nan)
    err_p = np.full((n_bins, 8), np.nan)
    valid_b = cnt > 0
    err_m[valid_b] = err_m_sum[valid_b] / cnt[valid_b]
    err_p[valid_b] = err_p_sum[valid_b] / cnt[valid_b]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    print(f"\n{'='*78}")
    print(" Conditional Error: nMAE by truth-magnitude quintile (normalized space, global edges)")
    print(f"{'='*78}")
    print(f"{'Channel':<10} " + "  ".join([f'b{b+1}(low->high)'.rjust(13) for b in range(n_bins)]))
    print("-" * 78)
    for c in range(8):
        row, col = c // 4, c % 4
        x = np.arange(n_bins)
        axes[row, col].plot(x, err_m[:, c], 'r-o', label='TRON')
        axes[row, col].plot(x, err_p[:, c], 'k--', label='Persistence')
        axes[row, col].set_xticks(x)
        axes[row, col].set_xticklabels([f'{b+1}' for b in range(n_bins)])
        xlabel = 'Truth Quantile (1=Floor -> 5=Dense)' if c in [0, 1] else 'Truth |x| Quantile (1=Weak -> 5=Strong)'
        axes[row, col].set_xlabel(xlabel)
        axes[row, col].set_ylabel('nMAE')
        axes[row, col].set_title(CHANNEL_NAMES[c])
        axes[row, col].legend(fontsize=7)
        axes[row, col].grid(True, alpha=0.3)
        row_s = "  ".join([f'{err_m[b, c]:.4f}'.rjust(13) if not np.isnan(err_m[b, c]) else 'n/a'.rjust(13) for b in range(n_bins)])
        print(f"{CHANNEL_NAMES_PLAIN[c]:<10} " + row_s)
    plt.tight_layout()
    path = os.path.join(args.save_dir, "fig19_conditional_error.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    np.savez(os.path.join(args.save_dir, "conditional_error.npz"),
             err_model=err_m, err_persistence=err_p, count=cnt, edges=edges)


# ============================================================================
#  Persistence-Error Correlation Matrix
# ============================================================================

def pairwise_masked_corrcoef(data):
    """Pairwise-NaN-masked Pearson correlation matrix for data (N_samples, N_channels).
    Masks out samples where either channel is NaN. If variance < 1e-12 or valid samples < 3,
    returns NaN (nan_to_num with zero is strictly forbidden)."""
    n_ch = data.shape[1]
    corr = np.full((n_ch, n_ch), np.nan)
    for i in range(n_ch):
        for j in range(n_ch):
            xi = data[:, i]
            xj = data[:, j]
            mask = ~(np.isnan(xi) | np.isnan(xj))
            if mask.sum() < 3:
                continue
            xi_m = xi[mask]
            xj_m = xj[mask]
            v_i = float(np.var(xi_m))
            v_j = float(np.var(xj_m))
            if v_i < 1e-12 or v_j < 1e-12:
                continue
            c = np.corrcoef(xi_m, xj_m)[0, 1]
            corr[i, j] = c
    return corr

def evaluate_error_correlation(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    ids = dump_ids(data_dir)
    test_ids = [d for d in ids if d >= args.start and (d + 1) in ids][:20]
    if len(test_ids) < 2:
        print("Not enough test samples"); return
    err_model = np.zeros((len(test_ids), 8))
    err_data = np.zeros((len(test_ids), 8))
    for i, d in enumerate(test_ids):
        curr, nxt = load_dump(data_dir, d), load_dump(data_dir, d + 1)
        with torch.no_grad():
            pred = forecast_one_step(model, norm, to_device(curr, device), device).cpu().numpy()[0]
        err_model[i] = rel_l1_per_channel(pred, nxt)
        err_data[i] = rel_l1_per_channel(curr, nxt)
    corr_model = pairwise_masked_corrcoef(err_model)
    corr_data = pairwise_masked_corrcoef(err_data)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, corr, title in [
        (axes[0], corr_model, 'Model Error Correlation'),
        (axes[1], corr_data, 'Persistence-Error Correlation'),
    ]:
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(8)); ax.set_yticks(range(8))
        ax.set_xticklabels(CHANNEL_NAMES_PLAIN, rotation=45, ha='right')
        ax.set_yticklabels(CHANNEL_NAMES_PLAIN)
        for i in range(8):
            for j in range(8):
                val = corr[i, j]
                txt = f"{val:.2f}" if not np.isnan(val) else "n/a"
                ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=8)
        ax.set_title(title + f' ({len(test_ids)} test pairs)')
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    path = os.path.join(args.save_dir, "error_correlation_matrix.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    np.savez(os.path.join(args.save_dir, "error_correlation.npz"),
             corr_model=corr_model, corr_data=corr_data,
             mean_err_model=np.nanmean(err_model, axis=0),
             mean_err_data=np.nanmean(err_data, axis=0))


# ============================================================================
#  Table 4: Predictability Horizon — data decorrelation vs model skill
# ============================================================================

def evaluate_correlation_times(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    """The headline ML table: how many correlation times (and how much physical
    time in R_g/c) is the model accurate for, per channel. tau_c comes from the
    TRUTH data (decorrelation of the physical process); skill crossing comes
    from the model rollout vs persistence. This frames every other number in
    the paper: 'the model tracks truth for N correlation times'."""
    tau_c, sat = channel_decorrelation_times(data_dir, norm_cpu, geo_raw=geo_raw)
    dt = float(geo['dt_dump']) if geo['dt_dump'] else 25.0
    dt = float(geo_raw['dt_dump']) if 'dt_dump' in geo_raw else dt

    all_ids = dump_ids(data_dir)
    all_ids_set = set(all_ids)
    req_steps = args.max_steps
    valid_candidates = [d for d in all_ids if d >= args.start]

    def contiguous_horizon(s):
        h = 0
        while (s + h + 1) in all_ids_set:
            h += 1
        return h

    horizons = {s: contiguous_horizon(s) for s in valid_candidates}
    max_avail = max(horizons.values()) if horizons else 0
    if max_avail == 0:
        print("No valid starts found with future dumps.")
        return None

    n_steps = min(req_steps, max_avail)
    if n_steps < req_steps:
        print(f"[Notice] Requested --max_steps {req_steps} for Table 4, but contiguous dumps allow {n_steps}. Evaluating up to step {n_steps}.")

    valid = [s for s in valid_candidates if horizons[s] >= n_steps]
    if not valid:
        print("No start has enough future dumps; use --start/--max_steps.")
        return None
    rng = np.random.default_rng(args.seed)
    starts = valid if len(valid) <= args.n_starts else rng.choice(valid, args.n_starts, replace=False)

    per_step_model = np.full((len(starts), n_steps, 8), np.nan)
    per_step_pers = np.full((len(starts), n_steps, 8), np.nan)
    for si, s0 in enumerate(starts):
        init = load_dump(data_dir, s0)
        curr = to_device(init, device)
        for step in range(1, n_steps + 1):
            true = load_dump(data_dir, s0 + step)
            if true is None:
                break
            with torch.no_grad():
                pred = forecast_one_step(model, norm, curr, device)
            curr = pred
            per_step_model[si, step - 1] = nmae_per_channel(
                pred.cpu().numpy()[0], true, norm_cpu)
            per_step_pers[si, step - 1] = nmae_per_channel(init, true, norm_cpu)

    min_starts = max(1, len(starts) // 2)
    valid_count_per_step = (~np.isnan(per_step_model[:, :, 0])).sum(axis=0)
    valid_step_indices = np.where(valid_count_per_step >= min_starts)[0]
    if not len(valid_step_indices):
        print("No valid rollout steps."); return None

    excluded_steps = n_steps - len(valid_step_indices)
    if excluded_steps > 0:
        print(f"  [Notice] Excluded {excluded_steps} steps with < {min_starts} valid starts (from {n_steps} total steps requested).")

    valid_steps = valid_step_indices + 1
    m_model = np.nanmean(per_step_model[:, valid_step_indices, :], axis=0)
    m_pers = np.nanmean(per_step_pers[:, valid_step_indices, :], axis=0)
    skill = 1.0 - m_model / (m_pers + 1e-30)

    crossing = np.zeros(8)
    never_crossed = np.zeros(8, dtype=bool)
    last_pos = np.zeros(8)
    frac_pos_after = np.full(8, np.nan)
    mean_s25_50 = np.full(8, np.nan)

    n_v = len(valid_steps)
    for c in range(8):
        s_c = skill[:, c]
        below = np.where(s_c < 0)[0]
        pos = np.where(s_c > 0)[0]

        if len(below) == 0:
            never_crossed[c] = True
            crossing[c] = valid_steps[-1]
            last_pos[c] = valid_steps[-1]
            frac_pos_after[c] = 1.0
        else:
            first_cross_idx = below[0]
            crossing[c] = valid_steps[first_cross_idx]
            last_pos[c] = valid_steps[pos[-1]] if len(pos) > 0 else 0
            remaining = s_c[first_cross_idx + 1:]
            if len(remaining) > 0:
                frac_pos_after[c] = np.mean(remaining > 0)
            else:
                frac_pos_after[c] = np.nan

        # mean skill over steps 25..50 (1-indexed: index 24..50)
        if n_v >= 25:
            end_idx = min(n_v, 50)
            mean_s25_50[c] = np.mean(s_c[24:end_idx])
        else:
            mean_s25_50[c] = np.nan

    print(f"\n{'='*78}")
    print(" Table 4: Predictability Horizon (TRON vs data decorrelation)")
    print(f"{'='*78}")
    print(f"{'Channel':<10} {'tau_c [dump / Rg/c]':>22} {'skill<0 at':>12} {'horizon/tau_c':>15} {'horizon [Rg/c]':>16}")
    print("-" * 78)
    for c in range(8):
        sat_mark = "*" if sat[c] else ""
        if tau_c[c] <= 0:
            tau_str = f"{'n/a':>22}"
        else:
            tau_str = f"{tau_c[c]:.2f}{sat_mark} / {tau_c[c]*dt:.1f}"

        if never_crossed[c]:
            cross_str = f">{int(valid_steps[-1])}"
            htau_str = "n/a" if tau_c[c] <= 0 else f">{valid_steps[-1]/tau_c[c]:.2f}"
            hrg_str = f">{valid_steps[-1]*dt:.1f}"
        else:
            cross_str = f"{int(crossing[c])}"
            htau_str = "n/a" if tau_c[c] <= 0 else f"{crossing[c]/tau_c[c]:.2f}"
            hrg_str = f"{crossing[c]*dt:.1f}"

        print(f"{CHANNEL_NAMES_PLAIN[c]:<10} {tau_str:>22} {cross_str:>12} {htau_str:>15} {hrg_str:>16}")

    recrossing = [c for c in range(8) if not never_crossed[c] and last_pos[c] > crossing[c]]
    if recrossing:
        recross_str = ", ".join([f"{CHANNEL_NAMES_PLAIN[c]} (cross={int(crossing[c])}, last_pos={int(last_pos[c])}, frac_pos={frac_pos_after[c]:.2f})" for c in recrossing])
        print(f"\n  Re-crossing channels (positive skill reappears after first crossing): {recross_str}")

    print(f"\n  tau_c = lag-1 decorrelation time of TRUTH fluctuations (active disk, * = saturated r1>0.99);\n"
          f"  'skill<0 at' = first step model stops beating persistence; horizon = crossing * dt_dump ({dt:.0f} Rg/c).\n"
          f"  Note: skill evaluated vs persistence baseline on normalized representations; tau_c is linear-correlation proxy;\n"
          f"  horizon/tau_c ratio is illustrative when tau_c ~ O(1 dump) or saturated-* (sub-unity ratios expected there).")

    np.savez(os.path.join(args.save_dir, "correlation_times.npz"),
             tau_c=tau_c, dt=dt, skill_crossing=crossing,
             never_crossed=never_crossed,
             last_pos=last_pos, frac_pos_after_first=frac_pos_after,
             mean_s25_50=mean_s25_50, skill=skill, steps=valid_steps,
             saturated=sat)
    return dict(tau_c=tau_c, crossing=crossing, last_pos=last_pos,
                frac_pos_after_first=frac_pos_after, mean_s25_50=mean_s25_50)


# ============================================================================
#  Evaluation Domain Constants & GR Helpers
# ============================================================================

E_YMAX_CAP = 1e5    # Fallback ceiling for -E(r) binding energy; adopted bound is data-driven with stdout provenance
L_YMAX_CAP = 1e6    # Fallback ceiling for L(r) angular momentum; adopted bound is data-driven with stdout provenance


def kerr_horizon_isco(a_spin):
    """Exact event horizon r_H and prograde ISCO radius r_ISCO (Bardeen 1972)
    for dimensionless Kerr spin parameter a.
    r_H = 1 + sqrt(max(0, 1 - a^2))
    Z_1 = 1 + (1 - a^2)^{1/3} [ (1+a)^{1/3} + (1-a)^{1/3} ]
    Z_2 = sqrt(3 a^2 + Z_1^2)
    r_ISCO = 3 + Z_2 - sqrt((3 - Z_1)(3 + Z_1 + 2 Z_2))
    """
    a = float(a_spin)
    abs_a = min(abs(a), 1.0)
    r_H = 1.0 + np.sqrt(max(0.0, 1.0 - a ** 2))
    z1 = 1.0 + (1.0 - abs_a ** 2) ** (1.0 / 3.0) * ((1.0 + abs_a) ** (1.0 / 3.0) + (1.0 - abs_a) ** (1.0 / 3.0))
    z2 = np.sqrt(3.0 * abs_a ** 2 + z1 ** 2)
    r_isco = 3.0 + z2 - np.sqrt(max(0.0, (3.0 - z1) * (3.0 + z1 + 2.0 * z2)))
    return float(r_H), float(r_isco)

def add_isco_line(ax, r_isco, label='ISCO', color='gray', linestyle='--', alpha=0.7, yloc=0.92):
    """Draw a vertical dashed marker at the ISCO radius with a text label."""
    ax.axvline(r_isco, color=color, linestyle=linestyle, alpha=alpha, linewidth=1.2)
    ax.text(r_isco * 1.04, yloc, label, transform=ax.get_xaxis_transform(),
            color=color, fontsize=8, alpha=alpha, va='top', ha='left')

def data_ylim(data_list, p_low=1.0, p_high=99.0, margin=1.1, symmetric=False, cap=None):
    """Compute robust data-driven y-limits from percentiles across multiple arrays."""
    clean = []
    for d in data_list:
        if d is not None:
            arr = np.asarray(d).ravel()
            finite = arr[np.isfinite(arr)]
            if len(finite) > 0:
                clean.append(finite)
    if not clean:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    merged = np.concatenate(clean)
    if symmetric:
        vmax = float(np.percentile(np.abs(merged), p_high)) * margin
        if cap is not None:
            vmax = min(float(cap), vmax)
        return -vmax, vmax
    else:
        vlo = float(np.percentile(merged, p_low))
        vhi = float(np.percentile(merged, p_high))
        span = vhi - vlo
        if span <= 0:
            span = max(abs(vhi), 1.0)
        low = vlo - (margin - 1.0) * span
        high = vhi + (margin - 1.0) * span
        if cap is not None:
            high = min(float(cap), high)
        return low, high

def kerr_keplerian_omega(r, a):
    """GR Keplerian orbital frequency in the equatorial plane of a Kerr BH:
    Omega_K = 1/(r^{3/2} + a) [c/Rg]. This is the analytic curve the disk's
    rotation should follow (frame dragging adds the +a term)."""
    return 1.0 / (r ** 1.5 + a)

def plot_mean_profiles(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    """Time+phi-averaged radial physics profiles, pred vs truth:
      (a) rotation curve Omega(r) = u^phi/u^0 in the equatorial band vs the
          Kerr Keplerian curve 1/(r^{3/2}+a) — does the model learn GR orbital
          mechanics? (b) vertical density structure rho(theta) at r=10/30/100 —
          the disk scale height, (c) plasma beta p_gas/p_mag in the equatorial
          band, (d) magnetization sigma = b^2/rho in the equatorial band."""
    init = load_dump(data_dir, args.start)
    if init is None:
        print(f"Dump {args.start} not found."); return
    r3 = geo_raw['r'][:, geo_raw['r'].shape[1] // 2, 0]
    # theta varies along axis 1 (h is constant along r for this grid),
    # so the theta profile must slice axis 1:
    h3 = geo_raw['h'][geo_raw['h'].shape[0] // 2, :, 0]
    Nr = r3.shape[0]
    a = float(geo_raw['spin']) if 'spin' in geo_raw else 0.0
    gamma = float(geo_raw['eos_gamma']) if 'eos_gamma' in geo_raw else 5.0 / 3.0

    req_steps = max(1, args.max_steps // 4)
    h = 0
    while load_dump(data_dir, args.start + h + 1) is not None and h < req_steps:
        h += 1
    n_steps = h
    if n_steps == 0:
        print(f"No future dumps found after {args.start}."); return
    r_dict = rollout_predictions(model, norm, init, n_steps, device, dump_id=args.start)
    states = [init] + [r_dict[s] for s in range(1, n_steps + 1)]
    n_t = len(states)
    pred = np.mean(np.stack(states), axis=0)          # (8, Nr, Nt, Np) averaged in time
    tr = np.mean(np.stack([load_dump(data_dir, args.start + i)
                           for i in range(n_t)]), axis=0)

    # NOTE: theta varies along field axis 1; the equatorial band mask must be
    # applied along that axis, NOT along r (h is constant along r here, so
    # r-sliced masks degenerate).
    band_theta = np.abs(h3 - np.pi / 2) < 0.1

    # rotation curve in the equatorial band: Omega = u^phi/u^0
    with torch.no_grad():
        ucon_p = norm.ucon_of(to_device(pred, device)[:, 2:5])[0].cpu().numpy()
        ucon_t = norm.ucon_of(to_device(tr, device)[:, 2:5])[0].cpu().numpy()
    omega_p = np.mean(ucon_p[3][:, band_theta, :] / ucon_p[0][:, band_theta, :], axis=(1, 2))
    omega_t = np.mean(ucon_t[3][:, band_theta, :] / ucon_t[0][:, band_theta, :], axis=(1, 2))
    omega_k = kerr_keplerian_omega(r3, a)

    # vertical structure at fixed radii
    r_sel = [10.0, 30.0, 100.0]
    r_idx = [int(np.argmin(np.abs(r3 - rr))) for rr in r_sel]
    rho_t_theta = np.abs(tr[0][r_idx, :, :]).mean(axis=-1)   # (3, Nt)
    rho_p_theta = np.abs(pred[0][r_idx, :, :]).mean(axis=-1)
    rho_t_norm = rho_t_theta / rho_t_theta.max(axis=1, keepdims=True)
    rho_p_norm = rho_p_theta / rho_p_theta.max(axis=1, keepdims=True)

    # plasma beta and magnetization per shell (theta-averaged, log-median).
    # Full covariant b^2 = b^mu b_mu (identical to analyse_forecast_ds),
    # NOT the gamma_ij B^i B^j proxy: the two scripts must agree on
    # beta = 2 p_gas / b^2 and sigma = b^2 / rho.
    def beta_sigma(x):
        p_gas = (gamma - 1.0) * x[1]
        b2 = bsq_full(x, geo_raw, norm_cpu)
        beta = p_gas / (0.5 * b2 + 1e-30)
        sigma = b2 / (x[0] + 1e-30)
        # clamp before log: b^2 can hit exactly 0 (vacuum cells) and a fully
        # undertrained model can produce null B -> log10(0) = -inf warning
        return (np.log10(np.maximum(beta[:, band_theta, :], 1e-30)),
                np.log10(np.maximum(sigma[:, band_theta, :], 1e-30)))
    lb_p, ls_p = beta_sigma(pred)
    lb_t, ls_t = beta_sigma(tr)
    beta_p = np.median(lb_p, axis=(1, 2))
    beta_t = np.median(lb_t, axis=(1, 2))
    sigma_p = np.median(ls_p, axis=(1, 2))
    sigma_t = np.median(ls_t, axis=(1, 2))

    r_H, r_isco = kerr_horizon_isco(a)
    r_mask_100 = (r3 >= r_H) & (r3 <= R_DISK_MAX)
    r3_plot = r3[r_mask_100]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax = axes[0, 0]
    ax.plot(r3_plot, omega_t[r_mask_100], 'b-', linewidth=1.5, label=f'Truth')
    ax.plot(r3_plot, omega_p[r_mask_100], 'r--', linewidth=1.5, label=f'TRON')
    ax.plot(r3_plot, omega_k[r_mask_100], 'k:', linewidth=1, label=f'Kerr Keplerian (a={a:.3f})')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(r_H, R_DISK_MAX)
    ax.margins(x=0)
    add_isco_line(ax, r_isco)
    ax.set_xlabel(r'$r\;[R_g]$'); ax.set_ylabel(r'$\Omega = u^\varphi/u^0\ [c/R_g]\ (u^0\equiv u^t\ \mathrm{coordinate})$')
    ax.set_title(f'Rotation Curve (equatorial band)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    all_omega = np.concatenate([omega_t[r_mask_100], omega_p[r_mask_100], omega_k[r_mask_100]])
    pos_omega = all_omega[all_omega > 0]
    if len(pos_omega) > 0:
        o_min = float(np.min(pos_omega))
        o_max = float(np.max(pos_omega))
        y_lo_omega = float(10 ** np.floor(np.log10(o_min)))
        y_hi_omega = float(10 ** np.ceil(np.log10(o_max)))
        ax.set_ylim(y_lo_omega, y_hi_omega)
        print(f"  Fig 10: Rotation curve adopted ylim=({y_lo_omega:.2e}, {y_hi_omega:.2e}) (data min={o_min:.2e}, max={o_max:.2e})")

    ax = axes[0, 1]
    theta_deg = np.degrees(h3)
    truth_lines, truth_labels = [], []
    fc_lines, fc_labels = [], []
    for i, rr in enumerate(r_sel):
        col = f'C{i}'
        lp, = ax.plot(theta_deg, rho_p_norm[i], '-', color=col, linewidth=1.5,
                      alpha=0.65, zorder=2)
        lt, = ax.plot(theta_deg, rho_t_norm[i], ':', color=col, linewidth=2.0,
                      alpha=1.0, zorder=3)
        truth_lines.append(lt)
        truth_labels.append(f'Truth r={rr:.0f}')
        fc_lines.append(lp)
        fc_labels.append(f'TRON r={rr:.0f}')
    ax.set_yscale('log')
    ax.set_xlim(0, 180)
    ax.margins(x=0)
    ax.set_xlabel(r'$\theta\;[\mathrm{degrees}]$')
    ax.set_ylabel(r'$\rho(\theta)/\rho_{max}$')
    ax.set_title(f'Vertical Disk Structure (phi-averaged)')
    ax.legend(truth_lines + fc_lines, truth_labels + fc_labels,
              loc='lower center', fontsize=7.5, ncol=2, framealpha=0.85)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(r3_plot, beta_t[r_mask_100], 'b-', linewidth=1.5, label=f'Truth')
    ax.plot(r3_plot, beta_p[r_mask_100], 'r--', linewidth=1.5, label=f'TRON')
    ax.set_xscale('log')
    ax.set_xlim(r_H, R_DISK_MAX)
    ax.margins(x=0)
    add_isco_line(ax, r_isco)
    ax.set_xlabel(r'$r\;[R_g]$'); ax.set_ylabel(r'$\log_{10}\beta = p_{gas}/p_{mag}$')
    ax.set_title(f'Plasma Beta Profile (equatorial band)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    b_lo, b_hi = data_ylim([beta_t[r_mask_100], beta_p[r_mask_100]], p_low=1.0, p_high=99.0, margin=1.15)
    b_lo = min(b_lo, -0.2)
    b_hi = max(b_hi, 0.2)
    ax.set_ylim(b_lo, b_hi)
    print(f"  Fig 10: Plasma beta adopted ylim=({b_lo:.2f}, {b_hi:.2f})")

    ax = axes[1, 1]
    ax.plot(r3_plot, sigma_t[r_mask_100], 'b-', linewidth=1.5, label=f'Truth')
    ax.plot(r3_plot, sigma_p[r_mask_100], 'r--', linewidth=1.5, label=f'TRON')
    ax.set_xscale('log')
    ax.set_xlim(r_H, R_DISK_MAX)
    ax.margins(x=0)
    add_isco_line(ax, r_isco)
    ax.set_xlabel(r'$r\;[R_g]$'); ax.set_ylabel(r'$\log_{10}\sigma = b^2/\rho$')
    ax.set_title(f'Magnetization Profile (equatorial band)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    s_lo, s_hi = data_ylim([sigma_t[r_mask_100], sigma_p[r_mask_100]], p_low=1.0, p_high=99.0, margin=1.15)
    ax.set_ylim(s_lo, s_hi)
    print(f"  Fig 10: Magnetization adopted ylim=({s_lo:.2f}, {s_hi:.2f})")

    fig.suptitle(f'Mean Radial Physics Profiles',
                 fontsize=13)
    plt.tight_layout()
    path10 = os.path.join(args.save_dir, "fig10_mean_profiles.png")
    fig.savefig(path10)
    plt.close(fig)
    print(f"Saved: {path10}")

    # Extra Fig 10: Full radial domain r in [r_H, 2000] (rotation curve, beta, sigma; no disk structure)
    r_mask_full = (r3 >= r_H) & (r3 <= 2000.0)
    r3_full = r3[r_mask_full]

    fig_full, axes_full = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Rotation curve full domain
    ax_om = axes_full[0]
    ax_om.plot(r3_full, omega_t[r_mask_full], 'b-', linewidth=1.5, label=f'Truth')
    ax_om.plot(r3_full, omega_p[r_mask_full], 'r--', linewidth=1.5, label=f'TRON')
    ax_om.plot(r3_full, omega_k[r_mask_full], 'k:', linewidth=1, label=f'Kerr Keplerian (a={a:.3f})')
    ax_om.set_xscale('log'); ax_om.set_yscale('log')
    ax_om.set_xlim(r_H, 2000.0)
    ax_om.margins(x=0)
    add_isco_line(ax_om, r_isco)
    ax_om.set_xlabel(r'$r\;[R_g]$'); ax_om.set_ylabel(r'$\Omega = u^\varphi/u^0\ [c/R_g]\ (u^0\equiv u^t\ \mathrm{coordinate})$')
    ax_om.set_title(f'Rotation Curve (equatorial band, full domain)')
    ax_om.legend(fontsize=8); ax_om.grid(True, alpha=0.3)
    pos_om_f = np.concatenate([omega_t[r_mask_full], omega_p[r_mask_full], omega_k[r_mask_full]])
    pos_om_f = pos_om_f[pos_om_f > 0]
    if len(pos_om_f) > 0:
        y_lo_om_f = float(10 ** np.floor(np.log10(np.min(pos_om_f))))
        y_hi_om_f = float(10 ** np.ceil(np.log10(np.max(pos_om_f))))
        ax_om.set_ylim(y_lo_om_f, y_hi_om_f)

    # 2. Plasma beta full domain
    ax_b = axes_full[1]
    ax_b.plot(r3_full, beta_t[r_mask_full], 'b-', linewidth=1.5, label=f'Truth')
    ax_b.plot(r3_full, beta_p[r_mask_full], 'r--', linewidth=1.5, label=f'TRON')
    ax_b.set_xscale('log')
    ax_b.set_xlim(r_H, 2000.0)
    ax_b.margins(x=0)
    add_isco_line(ax_b, r_isco)
    ax_b.set_xlabel(r'$r\;[R_g]$'); ax_b.set_ylabel(r'$\log_{10}\beta = p_{\rm gas}/p_{\rm mag}$')
    ax_b.set_title(f'Plasma Beta Profile (equatorial band, full domain)')
    ax_b.legend(fontsize=8); ax_b.grid(True, alpha=0.3)
    b_lo_f, b_hi_f = data_ylim([beta_t[r_mask_full], beta_p[r_mask_full]], p_low=1.0, p_high=99.0, margin=1.15)
    b_lo_f = min(b_lo_f, -0.2)
    b_hi_f = max(b_hi_f, 0.2)
    ax_b.set_ylim(b_lo_f, b_hi_f)

    # 3. Magnetization full domain
    ax_s = axes_full[2]
    ax_s.plot(r3_full, sigma_t[r_mask_full], 'b-', linewidth=1.5, label=f'Truth')
    ax_s.plot(r3_full, sigma_p[r_mask_full], 'r--', linewidth=1.5, label=f'TRON')
    ax_s.set_xscale('log')
    ax_s.set_xlim(r_H, 2000.0)
    ax_s.margins(x=0)
    add_isco_line(ax_s, r_isco)
    ax_s.set_xlabel(r'$r\;[R_g]$'); ax_s.set_ylabel(r'$\log_{10}\sigma = b^2/\rho$')
    ax_s.set_title(f'Magnetization Profile (equatorial band, full domain)')
    ax_s.legend(fontsize=8); ax_s.grid(True, alpha=0.3)
    s_lo_f, s_hi_f = data_ylim([sigma_t[r_mask_full], sigma_p[r_mask_full]], p_low=1.0, p_high=99.0, margin=1.15)
    ax_s.set_ylim(s_lo_f, s_hi_f)

    fig_full.suptitle(f'Mean Radial Physics Profiles — Full Domain (r <= 2000 Rg)',
                      fontsize=13)
    plt.tight_layout()
    path10_full = os.path.join(args.save_dir, "fig10_mean_profiles_full.png")
    fig_full.savefig(path10_full)
    plt.close(fig_full)
    print(f"Saved: {path10_full}")
    print(f"\n  Rotation curve check (Omega [c/Rg] at r = 10/30/100):")
    print(f"    truth:  {omega_t[r_idx[0]]:.4f} / {omega_t[r_idx[1]]:.4f} / {omega_t[r_idx[2]]:.4f}")
    print(f"    TRON:     {omega_p[r_idx[0]]:.4f} / {omega_p[r_idx[1]]:.4f} / {omega_p[r_idx[2]]:.4f}")
    print(f"    Keplerian: {omega_k[r_idx[0]]:.4f} / {omega_k[r_idx[1]]:.4f} / {omega_k[r_idx[2]]:.4f}")
    print(f"  log10 beta at r=10/30/100: truth {beta_t[r_idx[0]]:.2f}/{beta_t[r_idx[1]]:.2f}/"
          f"{beta_t[r_idx[2]]:.2f} | forecast {beta_p[r_idx[0]]:.2f}/{beta_p[r_idx[1]]:.2f}/"
          f"{beta_p[r_idx[2]]:.2f}")


# ============================================================================
#  Table 5: Mass Conservation Closure (dM/dt vs boundary fluxes)
# ============================================================================

def evaluate_conservation_closure(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    """The decisive physics test: TRON is NOT trained to conserve mass,
    yet if it has learned the continuity structure of the PDE, its rollouts
    should satisfy dM_enc/dt = Mdot(100) - Mdot(r_H) to within the solver's own
    tolerance. Evaluated in the restricted accretion domain r in [r_H, 100] Rg
    in full physical code units (dV = dx1*dx2*dx3 for mass, dA = dx2*dx3 for flux)."""
    init = load_dump(data_dir, args.start)
    if init is None:
        print(f"Dump {args.start} not found."); return
    dt = float(geo_raw['dt_dump']) if 'dt_dump' in geo_raw else 25.0
    n_steps = args.max_steps

    r_mid = geo_raw['r'][:, geo_raw['r'].shape[1] // 2, 0]
    a_spin = float(geo_raw['spin']) if 'spin' in geo_raw else 0.0
    r_H, r_isco = kerr_horizon_isco(a_spin)
    idx_rH = int(np.argmin(np.abs(r_mid - r_H)))
    idx_100 = int(np.argmin(np.abs(r_mid - R_DISK_MAX)))

    dV = float(crit.dx * crit.dy * crit.dz)
    dA = float(crit.dy * crit.dz)

    def mass_enc_bare(raw):
        m_sh = mass_shells(raw, crit, norm)[0, 0]
        return float(torch.sum(m_sh[idx_rH:idx_100 + 1]).item())

    def mdot_boundaries_bare(raw):
        mdot_sh = mdot_shells(raw, crit, norm=norm, bare=True)[0, 0].cpu().numpy()
        return float(mdot_sh[idx_rH]), float(mdot_sh[idx_100])

    M0_b = mass_enc_bare(init)
    rows_p_b, rows_t_b = [], []
    rows_p_ph, rows_t_ph = [], []
    M_prev_p_b, M_prev_t_b = M0_b, M0_b
    states = rollout_predictions(model, norm, init, n_steps, device, dump_id=args.start)
    for step in range(1, n_steps + 1):
        true = load_dump(data_dir, args.start + step)
        if true is None or step not in states:
            break
        pred = states[step]
        with torch.no_grad():
            Mp_b, Mt_b = mass_enc_bare(pred), mass_enc_bare(true)
            mrH_p_b, m100_p_b = mdot_boundaries_bare(pred)
            mrH_t_b, m100_t_b = mdot_boundaries_bare(true)

        dMdt_p_b = (Mp_b - M_prev_p_b) / dt
        dMdt_t_b = (Mt_b - M_prev_t_b) / dt
        rows_p_b.append((dMdt_p_b, mrH_p_b, m100_p_b))
        rows_t_b.append((dMdt_t_b, mrH_t_b, m100_t_b))

        dMdt_p_ph = dMdt_p_b * dV
        dMdt_t_ph = dMdt_t_b * dV
        mrH_p_ph, m100_p_ph = mrH_p_b * dA, m100_p_b * dA
        mrH_t_ph, m100_t_ph = mrH_t_b * dA, m100_t_b * dA
        rows_p_ph.append((dMdt_p_ph, mrH_p_ph, m100_p_ph))
        rows_t_ph.append((dMdt_t_ph, mrH_t_ph, m100_t_ph))

        M_prev_p_b, M_prev_t_b = Mp_b, Mt_b

    if not rows_p_ph:
        return
    R_b = np.array(rows_p_b); T_b = np.array(rows_t_b)
    R_ph = np.array(rows_p_ph); T_ph = np.array(rows_t_ph)

    # Bare imbalance: |dM/dt - (Mdot_100 - Mdot_rH)| / |Mdot_rH|
    imb_p_b = np.abs(R_b[:, 0] - (R_b[:, 2] - R_b[:, 1])) / (np.abs(R_b[:, 1]) + 1e-30)
    imb_t_b = np.abs(T_b[:, 0] - (T_b[:, 2] - T_b[:, 1])) / (np.abs(T_b[:, 1]) + 1e-30)

    # Physical imbalance: |dM/dt - (Mdot_100 - Mdot_rH)| / RMS(Mdot_rH, Mdot_100)
    rms_p = np.sqrt(0.5 * (R_ph[:, 1] ** 2 + R_ph[:, 2] ** 2)) + 1e-30
    rms_t = np.sqrt(0.5 * (T_ph[:, 1] ** 2 + T_ph[:, 2] ** 2)) + 1e-30
    imb_p_ph = np.abs(R_ph[:, 0] - (R_ph[:, 2] - R_ph[:, 1])) / rms_p
    imb_t_ph = np.abs(T_ph[:, 0] - (T_ph[:, 2] - T_ph[:, 1])) / rms_t

    print(f"\n{'='*88}")
    print(f" Table 5: Mass Conservation Closure dM_enc/dt vs Mdot(100) - Mdot(r_H) (r in [{r_H:.2f}, 100] Rg, physical code units)")
    print(f"{'='*88}")
    print(f"{'Quantity':<36} {'TRON':>14} {'solver truth':>14}")
    print("-" * 66)
    print(f"{'<|dM_enc/dt|> [code units]':<36} {np.abs(R_ph[:,0]).mean():>14.4e} {np.abs(T_ph[:,0]).mean():>14.4e}")
    print(f"{'<|Mdot_rH|> [code units]':<36} {np.abs(R_ph[:,1]).mean():>14.4e} {np.abs(T_ph[:,1]).mean():>14.4e}")
    print(f"{'<|Mdot_100|> [code units]':<36} {np.abs(R_ph[:,2]).mean():>14.4e} {np.abs(T_ph[:,2]).mean():>14.4e}")
    print(f"{'<imbalance> / RMS(Mdot)':<36} {imb_p_ph.mean():>14.4e} {imb_t_ph.mean():>14.4e}")
    print(f"{'max imbalance / RMS(Mdot)':<36} {imb_p_ph.max():>14.4e} {imb_t_ph.max():>14.4e}")
    print(f"\n  Note: Full physical units applied: M_enc * (dx1*dx2*dx3), Mdot * (dx2*dx3)."
          f"\n  imbalance = |dM_enc/dt - (Mdot_100 - Mdot_rH)| / RMS(Mdot_rH, Mdot_100)."
          f"\n  measured truth imbalance is {imb_t_ph.mean():.3e} (max {imb_t_ph.max():.3e}) on [r_H, 100] Rg."
          f"\n  Residual is the discrete snapshot sampling floor (dt={dt:.0f} Rg/c spans ~2.5 inner orbits),"
          f"\n  not a PDE violation. Model is NOT trained on this constraint; truth sets the baseline.")
    
    np.savez(os.path.join(args.save_dir, "conservation_closure.npz"),
             dMdt_p_bare=R_b[:, 0], mdot_rH_p_bare=R_b[:, 1], mdot_100_p_bare=R_b[:, 2],
             dMdt_t_bare=T_b[:, 0], mdot_rH_t_bare=T_b[:, 1], mdot_100_t_bare=T_b[:, 2],
             mdot_50_p_bare=R_b[:, 2], mdot_50_t_bare=T_b[:, 2],  # backwards-compatible alias
             imbalance_p_bare=imb_p_b, imbalance_t_bare=imb_t_b,
             dMdt_p_phys=R_ph[:, 0], mdot_rH_p_phys=R_ph[:, 1], mdot_100_p_phys=R_ph[:, 2],
             dMdt_t_phys=T_ph[:, 0], mdot_rH_t_phys=T_ph[:, 1], mdot_100_t_phys=T_ph[:, 2],
             mdot_50_p_phys=R_ph[:, 2], mdot_50_t_phys=T_ph[:, 2],  # backwards-compatible alias
             imbalance_p_phys=imb_p_ph, imbalance_t_phys=imb_t_ph,
             dMdt_p=R_ph[:, 0], mdot_rH_p=R_ph[:, 1], mdot_100_p=R_ph[:, 2],
             dMdt_t=T_ph[:, 0], mdot_rH_t=T_ph[:, 1], mdot_100_t=T_ph[:, 2],
             mdot_50_p=R_ph[:, 2], mdot_50_t=T_ph[:, 2],  # backwards-compatible alias
             imbalance_p=imb_p_ph, imbalance_t=imb_t_ph,
             dt=dt, r_H=r_H, dx1=crit.dx, dx2=crit.dy, dx3=crit.dz)


# ============================================================================
#  Fig 11: Turbulence — structure-function scaling exponents (zeta_2..4)
# ============================================================================

def structure_function_exponents(channel, max_order=4):
    """S_p(lag) = <|x(r+lag) - x(r)|^p>_r along r, averaged over (theta, phi);
    scaling exponent zeta_p from a log-log fit over intermediate lags.
    On a log-spaced grid, separation corresponds to lag = Delta ln r.
    Returns (zeta_p, r2_values, signed S3)."""
    nr = channel.shape[0]
    max_lag = max(4, nr // 8)
    lags = np.arange(1, max_lag)
    zetas = []
    r2s = []
    s3_sign = 0.0
    for p in range(2, max_order + 1):
        sp = np.zeros(len(lags))
        for i, lag in enumerate(lags):
            d = channel[lag:] - channel[:-lag]
            sp[i] = np.mean(np.abs(d) ** p)
        lo, hi = len(lags) // 4, 3 * len(lags) // 4
        if hi <= lo:
            zetas.append(0.0); r2s.append(0.0); continue
        lk = np.log10(lags[lo:hi])
        lE = np.log10(sp[lo:hi] + 1e-30)
        poly = np.polyfit(lk, lE, 1)
        slope = float(poly[0])
        y_fit = poly[0] * lk + poly[1]
        ss_tot = np.sum((lE - np.mean(lE)) ** 2)
        ss_res = np.sum((lE - y_fit) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-30)) if ss_tot > 1e-30 else 0.0
        zetas.append(slope)
        r2s.append(r2)
        if p == 3:
            d = channel[max_lag // 2:] - channel[:-max_lag // 2]
            s3_sign = float(np.mean(d ** 3))
    return np.array(zetas), np.array(r2s), s3_sign


def plot_turbulence_stats(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    """Structure-function scaling exponents zeta_2, zeta_3, zeta_4 of the normalized
    representation (log10 rho, arcsinh velocity/B) along r in the active disk (r in [2, 100] Rg),
    pred vs truth: whether the model reproduces the turbulence scaling of the solver."""
    init = load_dump(data_dir, args.start)
    if init is None:
        print(f"Dump {args.start} not found."); return
    n_steps = max(1, args.max_steps // 2)
    while n_steps > 1 and load_dump(data_dir, args.start + n_steps) is None:
        n_steps -= 1
    target = load_dump(data_dir, args.start + n_steps)
    if target is None:
        print(f"dump {args.start + n_steps} missing"); return
    states = rollout_predictions(model, norm, init, n_steps, device, dump_id=args.start)
    pred = states[n_steps]
    pred_n = norm_cpu(to_tensor_cpu(pred))[0].numpy()
    target_n = norm_cpu(to_tensor_cpu(target))[0].numpy()

    r_1d = geo_raw['r'][:, 0, 0]
    h_1d = geo_raw['h'][0, :, 0]
    r_disk = (r_1d >= 2.0) & (r_1d <= R_DISK_MAX)
    eq_disk = np.abs(h_1d - np.pi / 2.0) < 0.3

    channels = [0, 4, 5]           # rho, u^phi, B^r
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, c, name in zip(axes, channels, [r'$\rho$', r'$u^\varphi$', r'$B^r$']):
        xc_t = target_n[c][r_disk][:, eq_disk]
        xc_p = pred_n[c][r_disk][:, eq_disk]
        for lab, x, col in [('Truth', xc_t, 'b'), ('Forecast', xc_p, 'r')]:
            nr = x.shape[0]
            max_lag = max(4, nr // 8)
            lags = np.arange(1, max_lag)
            s2 = np.zeros(len(lags))
            for i, lag in enumerate(lags):
                d = x[lag:] - x[:-lag]
                s2[i] = np.mean(d ** 2)
            ax.loglog(lags, s2 / s2[0], '-', color=col, label=f'{lab} S2')
        ax.set_xlabel(r'Radial lag $\Delta\ln r$ (cells)')
        ax.set_ylabel(r'$S_2 / S_2(1)$ (normalized space)')
        ax.set_title(f'{name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(args.save_dir, "fig11_structure_functions.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")

    print(f"\n{'='*78}")
    print(f" Structure-Function Scaling Exponents (r-direction, lag = Delta ln r, active disk r in [2, 100] Rg)")
    print(f"{'='*78}")
    print(f"{'Channel':<10} {'zeta2 p (R2)':>14} {'zeta2 t (R2)':>14} {'zeta3 p':>8} {'zeta3 t':>8} "
          f"{'zeta4 p':>8} {'zeta4 t':>8} {'S3 (tentative)':>15}")
    print("-" * 78)
    for c, name in zip(channels, [r'$\rho$', r'$u^\varphi$', r'$B^r$']):
        xc_t = target_n[c][r_disk][:, eq_disk]
        xc_p = pred_n[c][r_disk][:, eq_disk]
        zp, r2p, s3p = structure_function_exponents(xc_p)
        zt, r2t, s3t = structure_function_exponents(xc_t)
        str_z2_p = f"{zp[0]:.3f} ({r2p[0]:.2f})"
        str_z2_t = f"{zt[0]:.3f} ({r2t[0]:.2f})"
        str_s3 = f"{s3p:+.2e}/{s3t:+.2e}"
        print(f"{name:<10} {str_z2_p:>14} {str_z2_t:>14} {zp[1]:>8.3f} {zt[1]:>8.3f} "
              f"{zp[2]:>8.3f} {zt[2]:>8.3f} {str_s3:>15}")
    print(f"\n  Note: Evaluated on normalized representation in the active disk (r in [2, 100] Rg, |theta-pi/2| < 0.3)."
          f"\n  Log-spaced radial grid implies cell lag represents lag = Delta ln r."
          f"\n  Turbulent scaling exponents characterize intermittency (Kolmogorov incompressible HD reference is zeta_3=1;"
          f"\n  compressible GRMHD disks exhibit distinct magnetic scaling with shallower slopes)."
          f"\n  S3 sign is reported tentatively due to anisotropic/non-homogeneous accretion geometry.")


# ============================================================================
#  Fig 12: Azimuthal m-mode spectra (non-axisymmetric structure across horizons)
# ============================================================================

def plot_azimuthal_spectra(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    """Azimuthal (m) power spectra in the equatorial band at mid radius:
    Fourier decomposition in phi across multi-horizon forecasts [10, 25, 50, max_steps].
    The disk's non-axisymmetric structure (spiral density waves, m=1..5 modes) tests
    whether the model reproduces azimuthal turbulent fluctuations without damping high-m modes."""
    init = load_dump(data_dir, args.start)
    if init is None:
        print(f"Dump {args.start} not found."); return

    cand_horizons = [10, 25, 50, args.max_steps]
    horizons = sorted(list({h for h in cand_horizons if 1 <= h <= args.max_steps}))
    horizons = [h for h in horizons if load_dump(data_dir, args.start + h) is not None]
    if not horizons:
        print("No dump files available for azimuthal spectra horizons."); return

    max_h = max(horizons)
    states = rollout_predictions(model, norm, init, max_h, device, dump_id=args.start)
    horizons = [h for h in horizons if h in states]
    if not horizons:
        print("No valid rollout states for azimuthal spectra."); return

    # h is separable (constant along r for this grid) — the equatorial band
    # must be a 1D mask over the theta axis, NOT a 2D (Nr,Nt) mask applied to
    # axis 1 (a 2D boolean there flattens the array and kills the spectrum).
    h3 = geo_raw['h'][:, :, 0]                        # (Nr, Nt)
    band = np.abs(h3[0] - np.pi / 2) < 0.15           # (Nt,) over theta
    nr_mid = slice(init.shape[1] // 4, 3 * init.shape[1] // 4)

    def m_spec(x, c):
        y = x[c][nr_mid][:, band]                     # (r_band, theta_band, Np)
        y = y - np.mean(y, axis=-1, keepdims=True)    # mean-subtracted along phi
        return np.abs(np.fft.rfft(y, axis=-1)) ** 2   # (r_band, theta_band, Np//2+1)

    channels = [0, 4, 5]
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']

    # Precompute spectra for all valid horizons
    truth_specs = {}
    pred_specs = {}
    for h in horizons:
        t_dump = load_dump(data_dir, args.start + h)
        p_dump = states[h]
        t_norm = norm_cpu(to_tensor_cpu(t_dump))[0].numpy()
        p_norm = norm_cpu(to_tensor_cpu(p_dump))[0].numpy()
        truth_specs[h] = {c: m_spec(t_norm, c).mean(axis=(0, 1)) for c in channels}
        pred_specs[h] = {c: m_spec(p_norm, c).mean(axis=(0, 1)) for c in channels}

    m_axis = np.arange(len(truth_specs[horizons[0]][channels[0]]))
    m_nyq = len(m_axis) - 1
    h_rep = 50 if 50 in horizons else horizons[-1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex='col')
    for col_idx, (c, name) in enumerate(zip(channels, [r'$\rho$', r'$u^\varphi$', r'$B^r$'])):
        # Row 0: Representative horizon spectrum (truth vs forecast)
        ax_top = axes[0, col_idx]
        ax_top.loglog(m_axis[1:], truth_specs[h_rep][c][1:], 'b-', linewidth=1.4, alpha=0.9,
                      label=f'Truth (h={h_rep})')
        ax_top.loglog(m_axis[1:], pred_specs[h_rep][c][1:], 'r--', linewidth=1.4, alpha=0.9,
                      label=f'TRON (h={h_rep})')
        ax_top.set_xlim(1, m_nyq)
        ax_top.margins(x=0)
        ax_top.set_ylabel(r'Power $|F_m|^2$ (equatorial band)')
        ax_top.set_title(f'{name} Spectrum (h={h_rep})')
        ax_top.legend(fontsize=8, loc='lower left')
        ax_top.grid(True, alpha=0.3)

        # Row 1: Power ratio strip (Pred / Truth) across horizons, masked at truth power < p10
        ax_bot = axes[1, col_idx]
        for idx, h in enumerate(horizons):
            col = colors[idx % len(colors)]
            t_sp = truth_specs[h][c][1:]
            p_sp = pred_specs[h][c][1:]
            p10 = np.percentile(t_sp, 10)
            m_valid = t_sp >= p10
            ratio = p_sp[m_valid] / (t_sp[m_valid] + 1e-30)
            ax_bot.semilogx(m_axis[1:][m_valid], ratio, '.-', color=col, linewidth=1.4, alpha=0.9,
                            label=f'h={h}')
        ax_bot.axhline(1.0, color='k', linestyle=':', linewidth=0.8)
        ax_bot.set_xlim(1, m_nyq)
        ax_bot.margins(x=0)
        ax_bot.set_xlabel('Azimuthal wavenumber $m$')
        ax_bot.set_ylabel('Power Ratio (Pred / Truth)')
        ax_bot.set_title(f'{name} Power Ratio')
        ax_bot.legend(fontsize=7.5, ncol=2, loc='upper left')
        ax_bot.grid(True, alpha=0.3)

    fig.suptitle(f'Azimuthal Mode Spectra and Multi-Horizon Drift', fontsize=13)
    plt.tight_layout()
    path = os.path.join(args.save_dir, "fig12_azimuthal_spectra.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")

    modes_to_report = list(range(1, 9))
    high_m_mask = m_axis > (m_nyq // 2)

    print(f"\n{'='*70}")
    print(f" Azimuthal Power Ratio (Pred / Truth) across Horizons (m = {modes_to_report}):")
    print(f"{'='*70}")
    for h in horizons:
        print(f"  --- Horizon h={h} (t+{h}) ---")
        for c, name in zip(channels, [r'$\rho$', r'$u^\varphi$', r'$B^r$']):
            t_sp = truth_specs[h][c]
            p_sp = pred_specs[h][c]
            p10 = np.percentile(t_sp[1:], 10)
            ratio = np.where(t_sp >= p10, p_sp / (t_sp + 1e-30), np.nan)
            parts = [f"m={m} {ratio[m]:.3f}" if not np.isnan(ratio[m]) else f"m={m} n/a" for m in modes_to_report if m < len(ratio)]
            print(f"    {name:<12}: " + " | ".join(parts))
            p_high = float(np.sum(pred_specs[h][c][high_m_mask])) / (float(np.sum(pred_specs[h][c])) + 1e-30)
            t_high = float(np.sum(truth_specs[h][c][high_m_mask])) / (float(np.sum(truth_specs[h][c])) + 1e-30)
            print(f"      High-m fraction (m > {m_nyq//2}): pred={p_high:.3f}, truth={t_high:.3f}")
    print("\n  Note: Azimuthal spectra in equatorial band (unweighted cell-mean spectra, relative comparison only; ratios masked at truth power < p10).")


# ============================================================================
#  Scale-Resolved Coherence C(k,t) (Fig 13)
# ============================================================================

def plot_coherence(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    """Scale-resolved coherence C(k,t) over the rollout — the predictability
    horizon per scale (the k-resolved analog of Table 4).
    C = |<F_p F_t*>(theta,phi)|^2 / (<|F_p|^2> <|F_t|^2>) per radial mode k
    and forecast step t. C ~ 1: the model keeps that scale's phase AND
    amplitude; C -> 0: the scale has decorrelated (unpredictable)."""
    init = load_dump(data_dir, args.start)
    if init is None:
        print(f"Dump {args.start} not found."); return

    # Fig 13: scale-resolved COHERENCE C(k,t) over the rollout
    n_steps = max(2, args.max_steps)
    states = rollout_predictions(model, norm, init, n_steps, device, dump_id=args.start)
    coh = {0: [], 5: []}
    tt = []
    for step in range(1, n_steps + 1):
        true = load_dump(data_dir, args.start + step)
        if true is None or step not in states:
            break
        pn = states[step]
        pn_n = norm_cpu(to_tensor_cpu(pn))[0].numpy()
        true_n = norm_cpu(to_tensor_cpu(true))[0].numpy()
        for c in [0, 5]:
            fp = np.fft.rfftn(pn_n[c])
            ft = np.fft.rfftn(true_n[c])
            num = np.abs(np.mean(fp * np.conj(ft), axis=(1, 2))) ** 2
            den = np.mean(np.abs(fp) ** 2, axis=(1, 2)) * np.mean(np.abs(ft) ** 2, axis=(1, 2))
            coh[c].append(num / (den + 1e-30))
        tt.append(step)
    if len(tt) >= 2:
        fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, c, name in zip(axes2, [0, 5], [r'$\rho$', r'$B^r$']):
            C = np.array(coh[c]).T
            im = ax.pcolormesh(np.array(tt), np.arange(C.shape[0]), C,
                               shading='auto', cmap='magma', vmin=0, vmax=1)
            fig2.colorbar(im, ax=ax)
            ax.set_xlabel('Forecast step t')
            ax.set_ylabel('Radial wavenumber k')
            ax.set_title(f'Coherence C(k,t) — {name}')
        plt.tight_layout()
        path2 = os.path.join(args.save_dir, "fig13_coherence.png")
        fig2.savefig(path2)
        plt.close(fig2)
        print(f"Saved: {path2}")

def plot_error_spectra(*args, **kwargs):
    warnings.warn("'plot_error_spectra' is deprecated, use 'plot_coherence' instead.", DeprecationWarning, stacklevel=2)
    return plot_coherence(*args, **kwargs)


# ============================================================================
#  Temporal Power Spectrum P(omega) — Turbulent Dynamics vs Freezing/Flicker
# ============================================================================

def plot_temporal_power_spectrum(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    """Temporal Fourier power spectrum P(f) = |rfft(x*w)|^2 / (fs * sum(w^2)) pred vs truth,
    evaluated separately in the turbulent disk core and the polar jet funnel.
    Diagnoses temporal dynamical fidelity: catches artificial laminar freezing
    (premature high-f decay) and numerical flicker/instability (high-f surge)."""
    init = load_dump(data_dir, args.start)
    if init is None:
        print(f"Dump {args.start} not found."); return
    n_steps = max(8, args.max_steps)

    # Rollout and truth stacks start at step 1 (evaluating forecast dynamics, not init condition)
    true_states = []
    for step in range(1, n_steps + 1):
        d = load_dump(data_dir, args.start + step)
        if d is None:
            break
        true_states.append(d)
    T_avail = len(true_states)
    if T_avail < 6:
        print(f"Not enough dumps ({T_avail}) for temporal power spectrum (need >= 6)."); return

    states = rollout_predictions(model, norm, init, T_avail, device, dump_id=args.start)
    pred_states = []
    for step in range(1, T_avail + 1):
        if step in states:
            pred_states.append(states[step])
        else:
            break
    T_avail = min(len(true_states), len(pred_states))
    if T_avail < 6:
        print(f"Not enough common dumps ({T_avail}) for temporal power spectrum."); return
    true_states = true_states[:T_avail]
    pred_states = pred_states[:T_avail]

    P_stack = np.stack(pred_states, axis=0)  # (T, 8, Nr, Nt, Np) starting from step 1
    T_stack = np.stack(true_states, axis=0)

    dt = float(geo_raw['dt_dump']) if 'dt_dump' in geo_raw else 25.0
    fs = 1.0 / dt
    freqs = np.fft.rfftfreq(T_avail, d=dt)

    # Deterministic masks: single bsq evaluation from time-mean-truth field
    mean_truth = np.mean(T_stack, axis=0)
    bsq_mean = bsq_full(mean_truth, geo_raw, norm_cpu)
    sigma_mean = bsq_mean / (mean_truth[0] + 1e-30)

    theta_3d = geo_raw['h']
    r_3d = geo_raw['r']
    mask_disk = (r_3d >= 5.0) & (r_3d <= R_DISK_MAX) & (np.abs(theta_3d - np.pi / 2.0) < 0.3)
    mask_jet = (r_3d >= 2.0) & (r_3d <= R_DISK_MAX) & (np.abs(theta_3d - np.pi / 2.0) > 1.0) & (sigma_mean > 1.0)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    print(f"\n{'='*75}")
    print(f" Temporal Power Spectrum P(f) across {T_avail} dumps (steps 1..{T_avail}, dt = {dt:.1f} Rg/c)")
    print(f"  Domains: disk r in [5.0, {R_DISK_MAX:.1f}] Rg (|theta-pi/2| < 0.3, {int(mask_disk.sum())} cells), "
          f"jet r in [2.0, {R_DISK_MAX:.1f}] Rg (|theta-pi/2| > 1.0, sigma > 1, {int(mask_jet.sum())} cells)")
    print(f"{'='*75}")

    w = np.hanning(T_avail)
    w_sum2 = float(np.sum(w ** 2))
    norm_factor = fs * w_sum2

    def compute_psd(data_stack, channel_idx, mask):
        cells = data_stack[:, channel_idx, mask]
        if cells.shape[1] == 0:
            return np.zeros_like(freqs)
        cells_fluct = cells - cells.mean(axis=0, keepdims=True)
        window = w[:, None]
        fft_vals = np.fft.rfft(cells_fluct * window, axis=0)
        psd = (np.abs(fft_vals) ** 2) / (norm_factor + 1e-30)
        return np.mean(psd, axis=1)

    psd_pred_disk = np.zeros((8, len(freqs)))
    psd_true_disk = np.zeros((8, len(freqs)))
    psd_pred_jet = np.zeros((8, len(freqs)))
    psd_true_jet = np.zeros((8, len(freqs)))

    for c in range(8):
        row, col = c // 4, c % 4
        ax = axes[row, col]

        psd_p_disk = compute_psd(P_stack, c, mask_disk)
        psd_t_disk = compute_psd(T_stack, c, mask_disk)
        psd_p_jet = compute_psd(P_stack, c, mask_jet)
        psd_t_jet = compute_psd(T_stack, c, mask_jet)

        psd_pred_disk[c] = psd_p_disk
        psd_true_disk[c] = psd_t_disk
        psd_pred_jet[c] = psd_p_jet
        psd_true_jet[c] = psd_t_jet

        valid_f = freqs > 0
        f_plot = freqs[valid_f]

        ax.loglog(f_plot, psd_t_disk[valid_f], '-', color='#1f77b4', linewidth=1.6, label='Truth Disk')
        ax.loglog(f_plot, psd_p_disk[valid_f], '--', color='#d62728', linewidth=1.5, label='TRON Disk')
        ax.loglog(f_plot, psd_t_jet[valid_f], ':', color='#17becf', linewidth=1.4, label='Truth Jet')
        ax.loglog(f_plot, psd_p_jet[valid_f], '-.', color='#ff7f0e', linewidth=1.3, label='TRON Jet')

        ax.set_title(f'{CHANNEL_NAMES[c]}')
        ax.set_xlabel(r'$f\;[1/R_g\text{-}c]$')
        ax.set_ylabel(r'PSD $P(f)$')
        ax.legend(fontsize=8, loc='lower left', frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, which='both')

    fig.suptitle(f'Temporal Power Spectra P(f) — Turbulent Dynamics ({T_avail} dumps, steps 1..{T_avail}, start {args.start})', fontsize=13)
    plt.tight_layout()
    path = os.path.join(args.save_dir, "temporal_power_spectrum.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    print("  Note: Single-rollout temporal PSD; spectral variance per frequency bin is order-unity (no ensemble Welch averaging).")
    np.savez(os.path.join(args.save_dir, "temporal_spectrum.npz"),
             freqs=freqs, channels=np.array(CHANNEL_NAMES_PLAIN),
             psd_pred_disk=psd_pred_disk, psd_true_disk=psd_true_disk,
             psd_pred_jet=psd_pred_jet, psd_true_jet=psd_true_jet)


# ============================================================================
#  Fig 14: Error Decomposition — Systematic Bias and Random RMSE Maps
# ============================================================================

def evaluate_bias_analysis(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    """Mean signed error (bias) and RMSE maps over multiple starts at h=1,
    in the meridional plane for 4 key physical channels (rho, u^phi, B^r, B^phi).
    Splits error into SYSTEMATIC (bias) vs random (scatter) without 16-panel overload."""
    all_ids = dump_ids(data_dir)
    all_ids_set = set(all_ids)
    valid = [d for d in all_ids if d >= args.start and (d + 1) in all_ids_set]
    if not valid:
        print("No test pairs for bias analysis."); return
    rng = np.random.default_rng(args.seed)
    starts = valid if len(valid) <= args.n_starts else rng.choice(valid, args.n_starts, replace=False)

    a0, b0 = load_dump(data_dir, starts[0]), load_dump(data_dir, starts[0] + 1)
    bias = np.zeros((8, a0.shape[1], a0.shape[2]))
    rmse = np.zeros_like(bias)
    n = 0
    for s0 in starts:
        a, b = load_dump(data_dir, s0), load_dump(data_dir, s0 + 1)
        with torch.no_grad():
            p = forecast_one_step(model, norm, to_device(a, device), device)
        if p is None:
            continue
        p = p.cpu().numpy()[0]
        d = p - b
        d_phi = d.mean(axis=3)  # (8, Nr, Nt) azimuthally averaged meridional error
        bias += d_phi
        rmse += d_phi ** 2
        n += 1
    if not n:
        print("no valid starts"); return
    bias /= n
    rmse = np.sqrt(rmse / n)

    # 4 core physical channels: rho (mass), u^phi (rotation), B^r (dynamo), B^phi (toroidal energy)
    sel_channels = [0, 4, 5, 7]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
    X = geo_raw['r'][:, :, 0] * np.sin(geo_raw['h'][:, :, 0])
    Y = geo_raw['r'][:, :, 0] * np.cos(geo_raw['h'][:, :, 0])
    mask_inner = active_disk_mask(geo_raw, rmax=R_DISK_MAX)[:, :, 0]

    for j, c in enumerate(sel_channels):
        # Top row: Systematic Bias (percentile on inner active domain r <= 100 Rg)
        vmax = np.percentile(np.abs(bias[c][mask_inner]), 99)
        if vmax <= 0: vmax = 1e-6
        im_b = axes[0, j].pcolormesh(X, Y, bias[c], cmap='RdBu_r',
                                     vmin=-vmax, vmax=vmax, shading='auto')
        fig.colorbar(im_b, ax=axes[0, j])
        axes[0, j].set_title(f'{CHANNEL_NAMES[c]} Bias (Signed Error)')
        axes[0, j].set_aspect('equal')
        axes[0, j].set_xlim(0, 100)
        axes[0, j].set_ylim(-100, 100)
        axes[0, j].set_ylabel(r'$r\cos\theta\;[R_g]$')

        # Bottom row: Random RMSE
        vmax_r = np.percentile(rmse[c][mask_inner], 99)
        if vmax_r <= 0: vmax_r = 1e-6
        im_r = axes[1, j].pcolormesh(X, Y, rmse[c], cmap='inferno',
                                     vmin=0, vmax=vmax_r, shading='auto')
        fig.colorbar(im_r, ax=axes[1, j])
        axes[1, j].set_title(f'{CHANNEL_NAMES[c]} RMSE')
        axes[1, j].set_aspect('equal')
        axes[1, j].set_xlim(0, 100)
        axes[1, j].set_ylim(-100, 100)
        axes[1, j].set_xlabel(r'$r\sin\theta\;[R_g]$')
        axes[1, j].set_ylabel(r'$r\cos\theta\;[R_g]$')

    fig.suptitle(f'Spatial Error Decomposition — Bias & RMSE at h=1 ({n} starts, phi-averaged, r <= {R_DISK_MAX:.0f} Rg)', fontsize=13)
    plt.tight_layout()
    path = os.path.join(args.save_dir, "fig14_error_decomposition.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    print(f"  Fig 14: Percentiles evaluated on active disk ({int(mask_inner.sum())} cells, r <= {R_DISK_MAX:.0f} Rg, |theta-pi/2| < 0.5), zoom=(0, 100) x (-100, 100)")

    print(f"\n  Mean |bias| / RMSE per channel (systematic vs total error):")
    for c in range(8):
        print(f"    {CHANNEL_NAMES_PLAIN[c]:<10} |bias| {np.abs(bias[c]).mean():.4e}  "
              f"RMSE {rmse[c].mean():.4e}  (bias fraction "
              f"{np.abs(bias[c]).mean()/(rmse[c].mean()+1e-30):.2f})")


# ============================================================================
#  Table 7: Stability — Empirical Lipschitz Estimate
# ============================================================================

def evaluate_stability(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    """Stability diagnostic:
      Single-step empirical Lipschitz estimate: L = ||f(x+eps) - f(x)||_2 / ||eps||_2
      over random relative perturbations, evaluated explicitly in normalized space.
      L <= 1.0 indicates discrete non-expansiveness; L in [1.0, 2.0] is typical
      for chaotic hyperbolic flows with advection CFL ~ 1."""
    init = load_dump(data_dir, args.start)
    if init is None:
        print(f"Dump {args.start} not found."); return
    curr = to_device(init, device)
    with torch.no_grad():
        base = forecast_one_step(model, norm, curr, device)
    rng = np.random.default_rng(args.seed)

    # (a) Lipschitz over 10 random perturbation draws
    ls = []
    skipped_draws = 0
    n_draws = 10
    norm_curr = norm(curr)[0]
    nb = norm(base)[0]
    for _ in range(n_draws):
        eps_raw = init * (1.0 + 1e-3 * rng.standard_normal(init.shape))
        with torch.no_grad():
            pert = forecast_one_step(model, norm, to_device(eps_raw, device), device)
        np_ = norm(pert)[0]
        ne = norm(to_device(eps_raw, device))[0]
        denom = (ne - norm_curr).norm().item()
        if denom < 1e-12:
            skipped_draws += 1
            continue
        l = float((np_ - nb).norm().item() / denom)
        ls.append(l)

    L_mean = float(np.mean(ls)) if ls else float('nan')
    L_std = float(np.std(ls)) if ls else float('nan')

    print(f"\n{'='*70}")
    print(" Table 7: Stability — Empirical Lipschitz Constant (normalized space)")
    print(f"{'='*70}")
    print(f"  Single-step Lipschitz L (1e-3 relative perturbation, {len(ls)}/{n_draws} draws): {L_mean:.3f} +/- {L_std:.3f}")
    if skipped_draws > 0:
        print(f"  Note: skipped {skipped_draws} draws with perturbation norm < 1e-12.")
    if not np.isnan(L_mean):
        if L_mean <= 1.0:
            print("  Status: Strictly non-expansive (L <= 1.0)")
        elif L_mean <= 2.0:
            print("  Status: Weakly expansive / chaotic advection (1.0 < L <= 2.0)")
        else:
            print(f"  Status: Expansive (L = {L_mean:.3f} > 2.0)")

    np.savez(os.path.join(args.save_dir, "stability.npz"),
             lipschitz=L_mean, lipschitz_std=L_std, lipschitz_samples=np.array(ls),
             skipped_draws=skipped_draws)


# ============================================================================
#  Fig 16: Meridional rollout evolution strip
# ============================================================================

def plot_rollout_evolution(args, model, norm, norm_cpu, crit, geo, geo_raw, device, data_dir):
    """Truth vs forecast log10 rho in the meridional plane at horizons
    [1, 10, 25, 50] (clipped to max_steps): the qualitative proof the model
    does not blow up and tracks the disk structure as it evolves."""
    init = load_dump(data_dir, args.start)
    if init is None:
        print(f"Dump {args.start} not found."); return
    horizons = auto_horizons(args.max_steps)
    horizons = [h for h in horizons if load_dump(data_dir, args.start + h) is not None]
    if not horizons:
        print("no horizons available"); return

    r = geo_raw['r'][:, :, geo_raw['r'].shape[2] // 2]
    h = geo_raw['h'][:, :, geo_raw['h'].shape[2] // 2]
    X = r * np.sin(h)
    Y = r * np.cos(h)

    states = rollout_predictions(model, norm, init, max(horizons), device, dump_id=args.start)
    states[0] = init

    fig, axes = plt.subplots(3, len(horizons), figsize=(4 * len(horizons), 11.5))
    if len(horizons) == 1:
        axes = axes.reshape(3, 1)
    mask_inner = active_disk_mask(geo_raw, rmax=R_DISK_MAX)[:, :, geo_raw['r'].shape[2] // 2]

    for j, hh in enumerate(horizons):
        t = load_dump(data_dir, args.start + hh)
        f = states[hh]
        phi_idx_t = t.shape[3] // 2
        phi_idx_f = f.shape[3] // 2
        Z_t = np.log10(np.maximum(t[0][:, :, phi_idx_t], 1e-30))
        Z_f = np.log10(np.maximum(f[0][:, :, phi_idx_f], 1e-30))
        Z_err = np.abs(Z_f - Z_t)

        vmin_rho = np.percentile(Z_t[mask_inner], 1)
        vmax_rho = np.percentile(Z_t[mask_inner], 99)
        vmax_err = np.percentile(Z_err[mask_inner], 99)
        if vmax_err <= 0: vmax_err = 0.5

        # Row 0: Truth
        im0 = axes[0, j].pcolormesh(X, Y, Z_t, cmap='inferno', vmin=vmin_rho, vmax=vmax_rho, shading='auto')
        fig.colorbar(im0, ax=axes[0, j])
        axes[0, j].set_title(f'Truth t+{hh}')
        axes[0, j].set_aspect('equal')
        axes[0, j].set_xlim(0, 100)
        axes[0, j].set_ylim(-100, 100)
        if j == 0:
            axes[0, j].set_ylabel(r'$r\cos\theta\;[R_g]$')

        # Row 1: TRON
        im1 = axes[1, j].pcolormesh(X, Y, Z_f, cmap='inferno', vmin=vmin_rho, vmax=vmax_rho, shading='auto')
        fig.colorbar(im1, ax=axes[1, j])
        axes[1, j].set_title(f'TRON t+{hh}')
        axes[1, j].set_aspect('equal')
        axes[1, j].set_xlim(0, 100)
        axes[1, j].set_ylim(-100, 100)
        if j == 0:
            axes[1, j].set_ylabel(r'$r\cos\theta\;[R_g]$')

        # Row 2: |Delta log10 rho| error (dex)
        im2 = axes[2, j].pcolormesh(X, Y, Z_err, cmap='viridis', vmin=0, vmax=vmax_err, shading='auto')
        fig.colorbar(im2, ax=axes[2, j])
        axes[2, j].set_title(f'|Error| t+{hh} (dex)')
        axes[2, j].set_aspect('equal')
        axes[2, j].set_xlim(0, 100)
        axes[2, j].set_ylim(-100, 100)
        axes[2, j].set_xlabel(r'$r\sin\theta\;[R_g]$')
        if j == 0:
            axes[2, j].set_ylabel(r'$r\cos\theta\;[R_g]$')

    fig.suptitle(f'Meridional log10 rho rollout evolution (start {args.start}, r <= {R_DISK_MAX:.0f} Rg)', fontsize=13)
    plt.tight_layout()
    path = os.path.join(args.save_dir, "fig16_rollout_evolution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    print(f"  Fig 16: Percentiles evaluated on active disk ({int(mask_inner.sum())} cells, r <= {R_DISK_MAX:.0f} Rg, |theta-pi/2| < 0.5), zoom=(0, 100) x (-100, 100)")


# ============================================================================
#  Evaluation card: machine-readable summary of everything
# ============================================================================

def write_evaluation_card(args, data_dir, geometry_file, model_path, device, geo_raw, log_file="out_res.txt"):
    """Reproducibility artifact: JSON summary of the run (args, paths, data and
    geometry metadata, checkpoint info) so the paper's supplementary can state
    exactly what produced every number."""
    import json
    card = {
        'args': vars(args),
        'data_dir': data_dir,
        'geometry_file': geometry_file,
        'model_path': model_path,
        'device': str(device),
        'dump_count': len(dump_ids(data_dir)),
        'grid_shape': [int(x) for x in geo_raw['r'].shape],
        'eos_gamma': float(geo_raw['eos_gamma']) if 'eos_gamma' in geo_raw else None,
        'dt_dump': float(geo_raw['dt_dump']) if 'dt_dump' in geo_raw else None,
        'spin': float(geo_raw['spin']) if 'spin' in geo_raw else None,
        'rin': float(geo_raw['rin']) if 'rin' in geo_raw else None,
        'rout': float(geo_raw['rout']) if 'rout' in geo_raw else None,
        'seed': args.seed,
        'log_file': log_file,
    }
    path = os.path.join(args.save_dir, "evaluation_card.json")
    with open(path, 'w') as f:
        json.dump(card, f, indent=2)
    print(f"Saved: {path}")


class TeeLogger:
    """Duplicates stdout writes and flushes simultaneously to the console and to a log file."""
    def __init__(self, filepath, stream):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TRON Paper Figure & Table Generator (ds)")
    parser.add_argument("--mode", default="all",
                        choices=["all", "ml", "physics", "single_step", "rollout_curve",
                                 "shell_profiles", "cost", "physics_table",
                                 "spectrum", "mdot", "temporal_consistency", "lorentz",
                                 "spatial_error", "statistical_props",
                                 "meridional_slice", "error_correlation",
                                 "correlation_times", "mean_profiles",
                                 "conservation_closure",
                                 "turbulence_stats", "azimuthal_spectra",
                                 "coherence", "error_spectra", "bias_analysis", "stability",
                                 "rollout_evolution", "temporal_spectrum",
                                 "accretion_diagnostics", "conditional_error"])
    parser.add_argument("--start", type=int, default=None, help="Start dump ID for rollouts (defaults to first available dump in data_dir)")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--n_starts", type=int, default=10, help="Rollout starts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", default="paper_eval_ds_simple")
    parser.add_argument("--data_dir", default=data_dir, help="Directory containing dataset dumps")
    parser.add_argument("--geometry_file", default=geometry_file, help="Path to geometry.npz")
    parser.add_argument("--model_path", default=model_path, help="Path to trained model checkpoint")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    out_res_path = os.path.join(args.save_dir, "out_res.txt") if args.mode == "all" else os.path.join(args.save_dir, f"out_res_{args.mode}.txt")
    tee = TeeLogger(out_res_path, sys.stdout)
    orig_stdout = sys.stdout
    sys.stdout = tee

    try:
        active_data_dir = args.data_dir
        active_geometry_file = args.geometry_file
        active_model_path = args.model_path

        avail = dump_ids(active_data_dir)
        if not avail:
            print(f"Error: No dump files found in {active_data_dir}"); return
        if args.start is None:
            args.start = avail[0]
            print(f"Auto-selected start dump: {args.start}")
        elif args.start not in avail:
            print(f"[Warning] Requested --start {args.start} not in available dumps ({avail[0]}..{avail[-1]}).")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        print(f"Automatic logging enabled -> {out_res_path}")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if args.mode == "all":
            print("Building shared pipeline...")
            pipe = build_pipeline(active_geometry_file, active_model_path, device)
            model, norm, norm_cpu, crit, geo, geo_raw = pipe
            for fn in [evaluate_single_step, evaluate_rollout_curve, evaluate_cost,
                       evaluate_physics_table, evaluate_shell_profiles,
                       plot_spectrum, plot_mdot, evaluate_temporal_consistency,
                       evaluate_lorentz, evaluate_spatial_error, evaluate_statistical_props,
                       plot_meridional_slice, evaluate_error_correlation,
                       evaluate_correlation_times, plot_mean_profiles,
                       evaluate_conservation_closure,
                       plot_turbulence_stats, plot_azimuthal_spectra, plot_coherence,
                       evaluate_bias_analysis, evaluate_stability,
                       plot_rollout_evolution, plot_temporal_power_spectrum,
                       evaluate_accretion_diagnostics, evaluate_conditional_error]:
                fn(args, model, norm, norm_cpu, crit, geo, geo_raw, device, active_data_dir)
            write_evaluation_card(args, active_data_dir, active_geometry_file, active_model_path, device, geo_raw, log_file=out_res_path)
            print("\nDone. All figures in:", args.save_dir)
            return

        model, norm, norm_cpu, crit, geo, geo_raw = build_pipeline(active_geometry_file, active_model_path, device)
        if args.mode == "ml":
            for fn in [evaluate_single_step, evaluate_rollout_curve,
                       evaluate_error_correlation, evaluate_cost, evaluate_correlation_times,
                       evaluate_stability]:
                fn(args, model, norm, norm_cpu, crit, geo, geo_raw, device, active_data_dir)
        elif args.mode == "physics":
            for fn in [evaluate_physics_table, evaluate_shell_profiles,
                       plot_spectrum, plot_mdot, evaluate_temporal_consistency,
                       evaluate_lorentz, evaluate_spatial_error, evaluate_statistical_props,
                       plot_meridional_slice, plot_mean_profiles,
                       evaluate_conservation_closure,
                       plot_turbulence_stats, plot_azimuthal_spectra, plot_coherence,
                       evaluate_bias_analysis, plot_rollout_evolution, plot_temporal_power_spectrum,
                       evaluate_accretion_diagnostics, evaluate_conditional_error]:
                fn(args, model, norm, norm_cpu, crit, geo, geo_raw, device, active_data_dir)
        else:
            FNS = {"single_step": evaluate_single_step, "rollout_curve": evaluate_rollout_curve,
                   "shell_profiles": evaluate_shell_profiles,
                   "cost": evaluate_cost, "physics_table": evaluate_physics_table,
                   "spectrum": plot_spectrum, "mdot": plot_mdot,
                   "temporal_consistency": evaluate_temporal_consistency,
                   "lorentz": evaluate_lorentz, "spatial_error": evaluate_spatial_error,
                   "statistical_props": evaluate_statistical_props,
                   "meridional_slice": plot_meridional_slice,
                   "error_correlation": evaluate_error_correlation,
                   "correlation_times": evaluate_correlation_times,
                   "mean_profiles": plot_mean_profiles,
                   "conservation_closure": evaluate_conservation_closure,
                   "turbulence_stats": plot_turbulence_stats,
                   "azimuthal_spectra": plot_azimuthal_spectra,
                   "coherence": plot_coherence,
                   "error_spectra": plot_error_spectra,
                   "bias_analysis": evaluate_bias_analysis,
                   "stability": evaluate_stability,
                   "rollout_evolution": plot_rollout_evolution,
                   "temporal_spectrum": plot_temporal_power_spectrum,
                   "accretion_diagnostics": evaluate_accretion_diagnostics,
                   "conditional_error": evaluate_conditional_error}
            if args.mode == "error_spectra":
                print("[Warning] '--mode error_spectra' is deprecated. Use '--mode coherence' instead.")
            FNS[args.mode](args, model, norm, norm_cpu, crit, geo, geo_raw, device, active_data_dir)
        print("\nDone.")
    finally:
        sys.stdout = orig_stdout
        tee.close()


if __name__ == "__main__":
    main()