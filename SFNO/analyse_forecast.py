# Written by Sachin Venkatesh (the Hybrid U-NO pipeline this SFNO was built on); included with
# credit. Kevin Fortune claims no ownership of this file. See LICENSE.
#pole deletion
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mticker
import sys

sys.path.append(os.getcwd())
from SFNO.train_forecast import Hybrid_UNO_3D, DataNormalizer, CONFIG

CONFIG["data_dir"]      = "/path/to/sfno_data/processed_forecast"
CONFIG["geometry_file"] = "/path/to/sfno_data/processed_forecast/geometry.npz"
CONFIG["model_path"]    = "/path/to/sfno_data/processed_forecast/checkpoints/best_forecast.pth"

# Raw channel indices in the .npy data: [rho, ug, u1, u2, u3, B1, B2, B3]
RAW_MAP = {'rho': 0, 'ug': 1, 'u1': 2, 'u2': 3, 'u3': 4, 'b1': 5, 'b2': 6, 'b3': 7}

FLOOR_RHO = 1e-20
FLOOR_UG = 1e-22
GAMMA_AD = 5.0 / 3.0   # Adiabatic index — match your H-AMR setup

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.linewidth': 1.5,
    'image.cmap': 'jet',
})

# ============================================================================
#  Geometry & Data Loading
# ============================================================================

def load_full_geometry():
    """Load geometry including metric for derived quantity computation."""
    geo = np.load(CONFIG["geometry_file"])
    return {
        'r': geo['r'], 'h': geo['h'], 'ph': geo['ph'],
        'gcov': geo['gcov'], 'gcon': geo['gcon'], 'gdet': geo['gdet'],
    }

def load_dump_file(dump_id):
    path = os.path.join(CONFIG["data_dir"], f"dump_{dump_id}.npy")
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return None
    return np.load(path)

# ============================================================================
#  Derived Quantity Computation (mirrors pp.py misc_calc + calc_RAD logic)
# ============================================================================

def compute_u0(data, gcov):
    """
    Compute u^0 from normalization g_μν u^μ u^ν = -1.
    data: (8, Nr, Nθ, Nφ)
    gcov: (4, 4, Nr, Nθ, Nφ)
    """
    u1, u2, u3 = data[2], data[3], data[4]
    g = gcov
    
    A = g[0, 0]
    B_coeff = 2.0 * (g[0, 1]*u1 + g[0, 2]*u2 + g[0, 3]*u3)
    C = (g[1, 1]*u1**2 + g[2, 2]*u2**2 + g[3, 3]*u3**2
         + 2*g[1, 2]*u1*u2 + 2*g[1, 3]*u1*u3 + 2*g[2, 3]*u2*u3 + 1.0)
    
    disc = np.maximum(B_coeff**2 - 4*A*C, 0.0)
    u0 = (-B_coeff + np.sqrt(disc)) / (2*A + 1e-30)
    return u0

def compute_bsq(data, gcov, u0):
    """
    Compute b² = b^μ b_μ  (magnetic pressure * 2).
    Mirrors pp.py misc_calc / pp_c.misc_calc.
    """
    u1, u2, u3 = data[2], data[3], data[4]
    B1, B2, B3 = data[5], data[6], data[7]
    g = gcov
    
    # u_i = g_{iμ} u^μ  (covariant spatial components)
    ucov = np.zeros((4, *u0.shape))
    ucon = np.array([u0, u1, u2, u3])
    for mu in range(4):
        for nu in range(4):
            ucov[mu] += g[mu, nu] * ucon[nu]
    
    # b^0 = B^k u_k  (sum over spatial k=1,2,3)
    b0 = B1*ucov[1] + B2*ucov[2] + B3*ucov[3]
    
    # b^i = (B^i + b^0 * u^i) / u^0
    u0_safe = np.where(np.abs(u0) > 1e-30, u0, 1e-30)
    bcon1 = (B1 + b0*u1) / u0_safe
    bcon2 = (B2 + b0*u2) / u0_safe
    bcon3 = (B3 + b0*u3) / u0_safe
    
    # b_μ = g_μν b^ν
    bcon = np.array([b0, bcon1, bcon2, bcon3])
    bcov = np.zeros_like(bcon)
    for mu in range(4):
        for nu in range(4):
            bcov[mu] += g[mu, nu] * bcon[nu]
    
    # b² = b^μ b_μ
    bsq = np.maximum(sum(bcon[mu]*bcov[mu] for mu in range(4)), 0.0)
    return bsq

def compute_derived(data, geo, var_name):
    """
    Compute a derived physical quantity from raw primitives + metric.
    Returns (3D field, vmin, vmax, cmap, label, use_log).
    
    Supported derived variables:
      lorentz  : Lorentz factor Γ = α u^0
      bsq      : Magnetic pressure b² 
      sigma    : Magnetization σ = b²/ρ
      beta     : Plasma β = 2P_gas / b²
      ug_rho   : Temperature proxy u_g/ρ
    """
    rho = data[0]
    ug  = data[1]
    gcov = geo['gcov']
    gcon = geo['gcon']
    
    u0 = compute_u0(data, gcov)
    alpha = 1.0 / np.sqrt(np.maximum(-gcon[0, 0], 1e-30))
    
    if var_name == 'lorentz':
        val = alpha * u0
        return val, 0, 2, 'inferno', r'$\log_{10}(\Gamma)$', True
    
    elif var_name == 'bsq':
        val = compute_bsq(data, gcov, u0)
        return val, -8, 2, 'inferno', r'$\log_{10}(b^2)$', True
    
    elif var_name == 'sigma':
        bsq = compute_bsq(data, gcov, u0)
        val = bsq / (rho + 1e-30)
        return val, -8, 2, 'jet', r'$\log_{10}(\sigma = b^2/\rho)$', True
    
    elif var_name == 'beta':
        bsq = compute_bsq(data, gcov, u0)
        pg = (GAMMA_AD - 1.0) * ug
        val = 2.0 * pg / (bsq + 1e-30)
        return val, -2, 4, 'RdBu_r', r'$\log_{10}(\beta)$', True
    
    elif var_name == 'ug_rho':
        val = ug / (rho + 1e-30)
        return val, -7, 2, 'hot', r'$\log_{10}(u_g/\rho)$', True
    
    else:
        raise ValueError(f"Unknown derived variable: {var_name}")


# Defines how each variable type should be visualized
VAR_CONFIG = {
    # Raw scalars — log scale, positive-definite
    'rho':     {'type': 'raw', 'idx': 0, 'log': True,  'vmin': -8, 'vmax': 1,  'cmap': 'jet',     'label': r'$\log_{10}(\rho)$'},
    'ug':      {'type': 'raw', 'idx': 1, 'log': True,  'vmin': -10,'vmax': 0,  'cmap': 'jet',     'label': r'$\log_{10}(u_g)$'},
    # Raw velocities — arcsinh scale, sign-preserving
    'u1':      {'type': 'raw', 'idx': 2, 'log': False, 'vmin': None, 'vmax': None, 'cmap': 'RdBu_r', 'label': r'$u^r$'},
    'u2':      {'type': 'raw', 'idx': 3, 'log': False, 'vmin': None, 'vmax': None, 'cmap': 'RdBu_r', 'label': r'$u^\theta$'},
    'u3':      {'type': 'raw', 'idx': 4, 'log': False, 'vmin': None, 'vmax': None, 'cmap': 'RdBu_r', 'label': r'$u^\phi$'},
    # Raw B-fields — arcsinh scale, sign-preserving
    'b1':      {'type': 'raw', 'idx': 5, 'log': False, 'vmin': None, 'vmax': None, 'cmap': 'RdBu_r', 'label': r'$B^r$'},
    'b2':      {'type': 'raw', 'idx': 6, 'log': False, 'vmin': None, 'vmax': None, 'cmap': 'RdBu_r', 'label': r'$B^\theta$'},
    'b3':      {'type': 'raw', 'idx': 7, 'log': False, 'vmin': None, 'vmax': None, 'cmap': 'RdBu_r', 'label': r'$B^\phi$'},
    # Derived quantities — computed from primitives + metric (like pp.py does)
    'lorentz': {'type': 'derived', 'label': r'$\log_{10}(\Gamma)$'},
    'bsq':     {'type': 'derived', 'label': r'$\log_{10}(b^2)$'},
    'sigma':   {'type': 'derived', 'label': r'$\log_{10}(\sigma)$'},
    'beta':    {'type': 'derived', 'label': r'$\log_{10}(\beta)$'},
    'ug_rho':  {'type': 'derived', 'label': r'$\log_{10}(u_g/\rho)$'},
}

# ============================================================================
#  Visualization Logic
# ============================================================================

def prepare_vis_data(data_3d, geo, var_name):
    """
    Convert raw 8-channel data to a visualization-ready 3D field.
    Returns (vis_field, vmin, vmax, cmap, label).
    """
    cfg = VAR_CONFIG[var_name]
    
    if cfg['type'] == 'derived':
        val, vmin, vmax, cmap, label, use_log = compute_derived(data_3d, geo, var_name)
        if use_log:
            vis = np.log10(np.maximum(np.abs(val), 1e-30))
        else:
            vis = val
        return vis, vmin, vmax, cmap, label
    
    # Raw variable
    idx = cfg['idx']
    raw = data_3d[idx]
    
    if cfg['log']:
        # Positive-definite scalars: use log10
        floor = FLOOR_RHO if var_name == 'rho' else FLOOR_UG
        vis = np.log10(np.maximum(raw, floor))
        vmin = cfg['vmin']
        vmax = cfg['vmax']
    else:
        # Sign-changing vectors: use arcsinh scaling for visualization
        # arcsinh(x) ≈ sign(x)*log(2|x|) for large |x|, ≈ x for small |x|
        vis = np.arcsinh(raw)
        if cfg['vmin'] is None:
            sym = np.percentile(np.abs(vis), 99)  # Use 99th percentile to avoid outliers
            vmin, vmax = -sym, sym
        else:
            vmin, vmax = cfg['vmin'], cfg['vmax']
    
    return vis, vmin, vmax, cfg['cmap'], cfg['label']


# ============================================================================
#  Model Forecast
# ============================================================================

def run_autoregressive_forecast(start_dump, steps, device):
    init_data = load_dump_file(start_dump)
    if init_data is None: 
        raise FileNotFoundError(f"Start dump {start_dump} missing.")
    
    model = Hybrid_UNO_3D(in_c=8, nlat=CONFIG["grid_nlat"], nlon=CONFIG["grid_nlon"]).to(device)
    ckpt = torch.load(CONFIG["model_path"], map_location=device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    
    norm = DataNormalizer().to(device)
    curr_state = torch.tensor(init_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    print(f"Forecasting {steps} steps (Start: {start_dump} -> Target: {start_dump + steps})...")
    
    with torch.no_grad():
        for i in range(steps):
            norm_in = norm(curr_state)
            pred_delta = model(norm_in)          
            pred_norm = norm_in + pred_delta      
            curr_state = norm.inverse(pred_norm)
            # Re-floor atmosphere to prevent inflation
            curr_state[:, 0] = torch.clamp(curr_state[:, 0], min=CONFIG["floor_rho"])
            curr_state[:, 1] = torch.clamp(curr_state[:, 1], min=CONFIG["floor_ug"])
            
    return curr_state.cpu().numpy()[0]


# ============================================================================
#  Slicing & Plotting (unchanged geometry logic)
# ============================================================================

def get_slice_data(r, h, ph, data_3d, plane, rmax):
    mid_j = r.shape[1] // 2
    r_prof = r[:, mid_j, 0]
    lim_idx = np.searchsorted(r_prof, rmax * 1.5)
    lim_idx = min(lim_idx + 5, r.shape[0])
    
    if plane == 'xz':
        nz = r.shape[2]
        iz_r = 0
        iz_l = nz // 2 
        
        Xr = r[:lim_idx, :, iz_r] * np.sin(h[:lim_idx, :, iz_r])
        Yr = r[:lim_idx, :, iz_r] * np.cos(h[:lim_idx, :, iz_r])
        Xl = -r[:lim_idx, :, iz_l] * np.sin(h[:lim_idx, :, iz_l])
        Yl = r[:lim_idx, :, iz_l] * np.cos(h[:lim_idx, :, iz_l])
        
        X_half_stitched = np.concatenate((Xl[:, ::-1], Xr), axis=1)
        Y_half_stitched = np.concatenate((Yl[:, ::-1], Yr), axis=1)
        data_half_stitched = np.concatenate((data_3d[:lim_idx, ::-1, iz_l], data_3d[:lim_idx, :, iz_r]), axis=1)
        
        X_full = np.concatenate((X_half_stitched, X_half_stitched[:, 0:1]), axis=1)
        Y_full = np.concatenate((Y_half_stitched, Y_half_stitched[:, 0:1]), axis=1)
        data_full = np.concatenate((data_half_stitched, data_half_stitched[:, 0:1]), axis=1)
        
        return X_full, Y_full, data_full
        
    elif plane == 'xy':
        mid_y = r.shape[1] // 2
        rs = r[:lim_idx, mid_y, :]
        phs = ph[:lim_idx, mid_y, :]
        
        X = rs * np.cos(phs)
        Y = rs * np.sin(phs)
        data_xy = data_3d[:lim_idx, mid_y, :]
        
        X_full = np.concatenate((X, X[:, 0:1]), axis=1)
        Y_full = np.concatenate((Y, Y[:, 0:1]), axis=1)
        data_full = np.concatenate((data_xy, data_xy[:, 0:1]), axis=1)
        
        return X_full, Y_full, data_full


def plot_forecast_comparison(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geo = load_full_geometry()
    r, h, ph = geo['r'], geo['h'], geo['ph']
    
    pred_3d = run_autoregressive_forecast(args.start, args.steps, device)
    
    target_dump_id = args.start + args.steps
    true_3d = load_dump_file(target_dump_id)
    
    if true_3d is None:
        print(f"Ground truth for dump {target_dump_id} not found. Cannot plot comparison.")
        return

    # Prepare visualization data using proper transforms
    T_vis, vmin, vmax, cmap, label = prepare_vis_data(true_3d, geo, args.var)
    P_vis, _, _, _, _ = prepare_vis_data(pred_3d, geo, args.var)

    # Compute error in the visualization space
    abs_diff = np.abs(T_vis - P_vis)
    # Use the visualization-space difference relative to the range
    data_range = np.percentile(np.abs(T_vis), 99) - np.percentile(np.abs(T_vis), 1)
    rel_err = abs_diff / (data_range + 1e-30)
    E_vis = np.log10(np.maximum(rel_err, 1e-22))

    planes = ['xz', 'xy'] if args.plane == 'both' else [args.plane]
    rows = len(planes)
    cols = 3 
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
    plt.subplots_adjust(wspace=0.4, hspace=0.25, left=0.07, right=0.91)
    
    if rows == 1:
        axes = np.expand_dims(axes, 0)
    
    for row_idx, plane_name in enumerate(planes):
        t_slice = get_slice_data(r, h, ph, T_vis, plane_name, args.rmax)
        p_slice = get_slice_data(r, h, ph, P_vis, plane_name, args.rmax)
        e_slice = get_slice_data(r, h, ph, E_vis, plane_name, args.rmax)
        
        slices_list = [t_slice, p_slice, e_slice]
        titles = [f"Ground Truth", f"Forecast ({args.steps} step)", r"$\log_{10}$ relative error"]
        cmaps = [cmap, cmap, 'inferno']
        
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            slice_dat = slices_list[col_idx]
            
            if col_idx < 2:
                curr_vmin = vmin
                curr_vmax = vmax
            else:
                curr_vmin = -3
                curr_vmax = 0

            levels = np.linspace(curr_vmin, curr_vmax, 500) 
            
            im = ax.contourf(
                slice_dat[0], slice_dat[1], slice_dat[2], 
                levels=levels, cmap=cmaps[col_idx], vmin=curr_vmin, vmax=curr_vmax, extend='both'
            )
                
            ax.set_aspect('equal')
            ax.set_xlim(-args.rmax, args.rmax)
            ax.set_ylim(-args.rmax, args.rmax)
            ax.set_title(f"{titles[col_idx]}")
            
            if row_idx == rows - 1:
                ax.set_xlabel(r"$x [R_g]$")
            if col_idx == 0:
                ax.set_ylabel(r"$z [R_g]$" if plane_name=='xz' else r"$y [R_g]$")
            
            cax = inset_axes(
                ax,
                width="8%",       
                height="100%",     
                loc="right",
                borderpad=-2
            )
            cbar = fig.colorbar(im, cax=cax)
            
            formatter = mticker.FormatStrFormatter('%.1f')
            cbar.formatter = formatter
            cbar.update_ticks()
            
            if col_idx < 2:
                cbar.set_label(label)
            else:
                cbar.set_label("Log Error")

    out_dir = "new_uno"
    os.makedirs(out_dir, exist_ok=True)
    out_name = os.path.join(out_dir, f"forecast_eval_{args.var}_start{args.start}+{args.steps}steps.png")
    plt.savefig(out_name, dpi=300)
    print(f"Comparison plot saved to: {out_name}")


if __name__ == "__main__":
    all_vars = list(VAR_CONFIG.keys())
    parser = argparse.ArgumentParser(
        description="Forecast Analysis",
        epilog=f"Available variables: {', '.join(all_vars)}"
    )
    parser.add_argument("--var", default="rho", choices=all_vars, help="Variable to plot")
    parser.add_argument("--plane", default="both", choices=["xz", "xy", "both"], help="Plot plane")
    parser.add_argument("--start", type=int, default=5000, help="Starting Dump ID")
    parser.add_argument("--steps", type=int, default=10, help="Steps to forecast")
    parser.add_argument("--rmax", type=float, default=50.0, help="Zoom radius")
    parser.add_argument("--gamma_ad", type=float, default=5.0/3.0, help="Adiabatic index")
    
    args = parser.parse_args()
    GAMMA_AD = args.gamma_ad
    plot_forecast_comparison(args)

    # Examples:
    # python analyse_forecast.py --var rho --plane both --start 500 --steps 1 --rmax 50
    # python analyse_forecast.py --var sigma --plane both --start 500 --steps 1 --rmax 50
    # python analyse_forecast.py --var beta --plane both --start 500 --steps 1 --rmax 50
    # python analyse_forecast.py --var lorentz --plane xz --start 500 --steps 1 --rmax 50
    # python analyse_forecast.py --var b1 --plane xz --start 500 --steps 5 --rmax 50
