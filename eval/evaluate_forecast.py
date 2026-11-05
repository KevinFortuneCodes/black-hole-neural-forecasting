# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.
"""
evaluate_forecast.py — Paper Figure & Table Generator for GRMHD Neural Operator
================================================================================
Generates quantitative evaluations:
  - Table 2: Single-step nMAE + Rel L1 per channel (+ persistence baseline)
  - Fig 4:   nMAE vs autoregressive forecast steps
  - Fig 5:   Shell-integrated mass profile M(r)
  - Table 3: DivB residual at various horizons
  - Computational cost: param count, inference timing

Pipeline: preprocess_forecast.py → uno_forecast_old.py → evaluate_forecast_old.py

Usage:
  python evaluate_forecast.py --mode all --start 5000 --max_steps 50
  python evaluate_forecast.py --mode single_step --start 5000
  python evaluate_forecast.py --mode rollout_curve --start 5000 --max_steps 100
  python evaluate_forecast.py --mode shell_profiles --start 5000
  python evaluate_forecast.py --mode divb_table --start 5000
  python evaluate_forecast.py --mode cost
  python evaluate_forecast.py --mode physics_table --start 5000 --max_steps 100
  python evaluate_forecast.py --mode spectrum --start 5000
  python evaluate_forecast.py --mode mass_drift --start 5000 --max_steps 100
"""

import os, sys, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SFNO.train_forecast import Hybrid_UNO_3D, DataNormalizer, CONFIG

# --- PATH DEFAULTS (overridable via --data_dir / --geometry / --checkpoint) ---
CONFIG["data_dir"]       = "/path/to/sfno_data/processed_forecast"
CONFIG["geometry_file"]  = "/path/to/sfno_data/processed_forecast/geometry.npz"
CONFIG["model_path"]     = "/path/to/sfno_data/processed_forecast/checkpoints/best_forecast.pth"

CHANNEL_NAMES = [r'$\rho$', r'$u_g$', r'$u^r$', r'$u^\theta$',
                 r'$u^\varphi$', r'$B^r$', r'$B^\theta$', r'$B^\varphi$']

# Publication-quality plot style
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.linewidth': 1.2, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight'
})


# ============================================================================
#  Model & Data Loading
# ============================================================================

def load_dump(dump_id):
    path = os.path.join(CONFIG["data_dir"], f"dump_{dump_id}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)

def load_geometry():
    return np.load(CONFIG["geometry_file"])

def build_inference_pipeline(device):
    model = Hybrid_UNO_3D(in_c=8, nlat=CONFIG["grid_nlat"], nlon=CONFIG["grid_nlon"]).to(device)
    ckpt = torch.load(CONFIG["model_path"], map_location=device, weights_only=False)

    state_dict = ckpt.get('model', ckpt)
    model.load_state_dict(state_dict, strict=True)
    print("Loaded model weights.")
        
    model.eval()

    geo = load_geometry()
    norm = DataNormalizer().to(device)

    return model, norm, geo

def forecast_one_step(model, norm, curr_state):
    """Single autoregressive step. Returns raw physical-space tensor."""
    norm_in = norm(curr_state)
    pred_delta = model(norm_in)
    pred_norm = norm_in + pred_delta
    out = norm.inverse(pred_norm)
    out[:, 0] = torch.clamp(out[:, 0], min=CONFIG["floor_rho"], max=CONFIG["ceiling_rho"])
    out[:, 1] = torch.clamp(out[:, 1], min=CONFIG["floor_ug"], max=CONFIG["ceiling_ug"])
    return out


# ============================================================================
#  Metrics
# ============================================================================

def compute_rel_l1_per_channel(pred_raw, true_raw):
    """Relative L1 in physical space, per channel."""
    rel = np.zeros(8)
    for c in range(8):
        denom = np.abs(true_raw[c]).mean()
        if denom < 1e-30:
            rel[c] = 0.0
        else:
            rel[c] = np.abs(pred_raw[c] - true_raw[c]).mean() / denom
    return rel


def compute_divb(raw_np, gdet_np, dx, dy, dz):
    """Volume-averaged |div(sqrt(g) B)| / sqrt(g) from numpy arrays."""
    b1, b2, b3 = raw_np[5:6], raw_np[6:7], raw_np[7:8]
    gdet = gdet_np[np.newaxis]
    f1, f2, f3 = gdet * b1, gdet * b2, gdet * b3

    # Radial
    f1p = np.pad(f1, ((0,0),(1,1),(0,0),(0,0)), mode='edge')
    d1 = (f1p[:,2:] - f1p[:,:-2]) / (2*dx)

    # Theta (polar reflection)
    shift = f2.shape[-1] // 2
    top = np.flip(np.roll(f2[:,:,0:1,:], shift=shift, axis=-1), axis=2)
    bot = np.flip(np.roll(f2[:,:,-1:,:], shift=shift, axis=-1), axis=2)
    f2p = np.concatenate([-top, f2, -bot], axis=2)
    d2 = (f2p[:,:,2:] - f2p[:,:,:-2]) / (2*dy)

    # Phi (circular)
    f3p = np.pad(f3, ((0,0),(0,0),(0,0),(1,1)), mode='wrap')
    d3 = (f3p[:,:,:,2:] - f3p[:,:,:,:-2]) / (2*dz)

    divb = (d1 + d2 + d3) / (gdet + 1e-20)
    return np.abs(divb).mean()


def compute_shell_mass(raw_np, gdet_np):
    """Shell-integrated mass profile M(r) = sum_{theta,phi} rho * sqrt(-g)."""
    rho = raw_np[0]
    return np.sum(rho * gdet_np, axis=(-2, -1))

    	
def compute_mass_error(pred_raw, true_raw, gdet):
    rho_p = pred_raw[0]
    rho_t = true_raw[0]
    m_pred = np.sum(rho_p * gdet)
    m_true = np.sum(rho_t * gdet)
    return np.abs(m_pred - m_true) / (np.abs(m_true) + 1e-10)
    
def compute_spectral_error(pred, true):
    p_fft = np.fft.rfftn(pred[0])
    t_fft = np.fft.rfftn(true[0])
    return np.mean(np.abs(np.abs(p_fft) - np.abs(t_fft)))
    
def compute_gradient_error(pred, true):
    def grad(x):
        dx = x[:,1:,:,:] - x[:,:-1,:,:]
        dy = x[:,:,1:,:] - x[:,:,:-1,:]
        dz = x[:,:,:,1:] - x[:,:,:,:-1]
        return dx, dy, dz
    gp = grad(pred[0:1])
    gt = grad(true[0:1])
    return sum(np.mean(np.abs(a - b))/(np.mean(np.abs(b)) + 1e-30) for a, b in zip(gp, gt))

# ============================================================================
#  Table 2: Single-Step nMAE + Rel L1
# ============================================================================

def evaluate_single_step(args, device):
    """Compute single-step metrics over the validation set."""
    model, norm, geo = build_inference_pipeline(device)

    # Find all dumps
    all_dumps = sorted([
        int(f.split('_')[1].split('.')[0])
        for f in os.listdir(CONFIG["data_dir"]) if f.startswith("dump_") and f.endswith(".npy")
    ])
    n_total = len(all_dumps)
    n_train = int(0.9 * n_total)
    val_dumps = all_dumps[n_train:]
    print(f"Validation set: {len(val_dumps)} dumps (indices {val_dumps[0]}–{val_dumps[-1]})")

    # CPU normalizer for nMAE metrics
    norm_cpu = DataNormalizer()

    nmae_model = np.zeros(8)
    nmae_persist = np.zeros(8)
    rel_l1_model = np.zeros(8)
    count = 0

    for dump_id in val_dumps[:-1]:
        curr_data = load_dump(dump_id)
        next_data = load_dump(dump_id + 1)
        if curr_data is None or next_data is None:
            continue

        curr_t = torch.tensor(curr_data, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast(device.type, enabled=CONFIG["use_amp"]):
            pred_t = forecast_one_step(model, norm, curr_t)
        pred_np = pred_t.cpu().numpy()[0]

        # nMAE in normalized space
        pred_n = norm_cpu(torch.tensor(pred_np).unsqueeze(0))
        true_n = norm_cpu(torch.tensor(next_data).unsqueeze(0))
        curr_n = norm_cpu(torch.tensor(curr_data).unsqueeze(0))
        nmae_model += (pred_n - true_n).abs().mean(dim=(2,3,4)).squeeze(0).numpy()
        nmae_persist += (curr_n - true_n).abs().mean(dim=(2,3,4)).squeeze(0).numpy()
        rel_l1_model += compute_rel_l1_per_channel(pred_np, next_data)
        count += 1

    nmae_model /= count
    nmae_persist /= count
    rel_l1_model /= count
    fluid_err = nmae_model[:5].mean()
    mag_err = nmae_model[5:].mean()

    # Print and save table
    lines = [
        f"\n{'='*70}",
        f" Table 2: Single-Step Prediction Accuracy ({count} val samples)",
        f"{'='*70}",
        f"{'Channel':<12} {'nMAE (ours)':>12} {'Rel L1 (ours)':>14} {'nMAE (persist)':>15}",
        "-"*55,
    ]
    for c in range(8):
        lines.append(f"{CHANNEL_NAMES[c]:<12} {nmae_model[c]:>12.4f} {rel_l1_model[c]:>14.4f} {nmae_persist[c]:>15.4f}")
    lines += [
        "-"*55,
        f"{'Average':<12} {nmae_model.mean():>12.4f} {rel_l1_model.mean():>14.4f} {nmae_persist.mean():>15.4f}",
        "\nChannel Groups:",
        f"  Fluid (0–4): {fluid_err:.4f}",
        f"  Magnetic (5–7): {mag_err:.4f}",
    ]
    for l in lines:
        print(l)

    txt_path = os.path.join(args.save_dir, "table2_single_step.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {txt_path}")

    np.savez(os.path.join(args.save_dir, "single_step_metrics.npz"),
             nmae_model=nmae_model, nmae_persist=nmae_persist,
             rel_l1_model=rel_l1_model, count=count)
    print(f"Saved: {args.save_dir}/single_step_metrics.npz")
    return nmae_model, nmae_persist, rel_l1_model


# ============================================================================
#  Fig 4: nMAE vs Forecast Steps
# ============================================================================

def evaluate_rollout_curve(args, device):
    """nMAE as a function of autoregressive steps from multiple starting points."""
    model, norm, geo = build_inference_pipeline(device)
    norm_cpu = DataNormalizer()

    all_dumps = sorted([
        int(f.split('_')[1].split('.')[0])
        for f in os.listdir(CONFIG["data_dir"])
        if f.startswith("dump_")
    ])
    
    # Keep only dumps ≥ start AND with enough future steps
    valid_starts = [
        d for d in all_dumps
        if d >= args.start and load_dump(d + args.max_steps) is not None
    ]
    
    # Randomly sample (or take all if few)
    np.random.seed(0)  # reproducibility
    n_samples = min(10, len(valid_starts))  # configurable
    starts = np.random.choice(valid_starts, size=n_samples, replace=False)
    
    print(f"Selected {len(starts)} random start points from {len(valid_starts)} valid dumps")
    print(f"Rollout curve from {len(starts)} starting points, up to {args.max_steps} steps each")

    all_nmae = []
    all_persist = []

    for s_idx, s0 in enumerate(starts):
        print(f"  Start {s_idx+1}/{len(starts)}: dump {s0}")
        init_data = load_dump(s0)
        curr = torch.tensor(init_data, dtype=torch.float32).unsqueeze(0).to(device)

        step_nmae = []
        step_persist = []
        with torch.no_grad():
            for step in range(1, args.max_steps + 1):
                curr = forecast_one_step(model, norm, curr)
                true_data = load_dump(s0 + step)
                if true_data is None:
                    break
                pred_np = curr.cpu().numpy()[0]
                pred_n = norm_cpu(torch.tensor(pred_np).unsqueeze(0))
                true_n = norm_cpu(torch.tensor(true_data).unsqueeze(0))
                init_n = norm_cpu(torch.tensor(init_data).unsqueeze(0))
                step_nmae.append((pred_n - true_n).abs().mean(dim=(2,3,4)).squeeze(0).numpy())
                step_persist.append((init_n - true_n).abs().mean(dim=(2,3,4)).squeeze(0).numpy())

        all_nmae.append(np.array(step_nmae))
        all_persist.append(np.array(step_persist))

    min_len = min(len(x) for x in all_nmae)
    all_nmae = np.array([x[:min_len] for x in all_nmae])
    all_persist = np.array([x[:min_len] for x in all_persist])

    avg_nmae = all_nmae.mean(axis=-1)
    avg_persist = all_persist.mean(axis=-1)

    mean_nmae = avg_nmae.mean(axis=0)
    std_nmae = avg_nmae.std(axis=0)
    mean_persist = avg_persist.mean(axis=0)

    steps_arr = np.arange(1, min_len + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps_arr, mean_nmae, 'b-o', markersize=3, linewidth=1.5, label='Hybrid U-NO (ours)')
    ax.fill_between(steps_arr, mean_nmae - std_nmae, mean_nmae + std_nmae, alpha=0.2, color='blue')
    plt.xlim(1,args.max_steps)
    plt.xticks([1, 20, 40, 60, 80, args.max_steps])
    ax.plot(steps_arr, mean_persist, 'k--', linewidth=1, label='Persistence')
    ax.set_xlabel("Forecast steps")
    ax.set_ylabel("Channel-averaged nMAE")
    ax.set_title("Rollout Error vs. Forecast Horizon")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(args.save_dir, "fig4_rollout_error_vs_steps.png")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close()

    np.savez(os.path.join(args.save_dir, "rollout_curve.npz"),
             steps=steps_arr, mean_nmae=mean_nmae, std_nmae=std_nmae,
             mean_persist=mean_persist, all_nmae=all_nmae)
    return mean_nmae, std_nmae


# ============================================================================
#  Fig 5: Shell-Integrated Mass Profile
# ============================================================================

def evaluate_shell_profiles(args, device):
    """Compare shell-integrated mass M(r) at multiple rollout horizons."""
    model, norm, geo = build_inference_pipeline(device)
    gdet_np = geo['gdet']
    r_mid = geo['r'][:, geo['r'].shape[1]//2, 0]

    horizons = [1, 10, 50, 100]
    init_data = load_dump(args.start)
    if init_data is None:
        print(f"Dump {args.start} not found."); return

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    for n_steps in horizons:
        target = load_dump(args.start + n_steps)
        if target is None:
            print(f"  Skipping horizon {n_steps}: dump {args.start + n_steps} missing"); continue

        curr = torch.tensor(init_data, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            for _ in range(n_steps):
                curr = forecast_one_step(model, norm, curr)
        pred_np = curr.cpu().numpy()[0]

        M_true = compute_shell_mass(target, gdet_np)
        M_pred = compute_shell_mass(pred_np, gdet_np)
        rel_err = np.abs(M_pred - M_true) / (np.abs(M_true) + 1e-30)

        axes[0].plot(r_mid, M_true, '-', linewidth=1.5, alpha=0.5, label=f'Truth ({n_steps} steps)')
        axes[0].plot(r_mid, M_pred, '--', linewidth=1.5, label=f'Forecast ({n_steps} steps)')
        axes[1].plot(r_mid, rel_err, linewidth=1, label=f'{n_steps} steps')

    axes[0].set_ylabel(r'$M(r) = \sum_{\theta,\varphi} \rho \sqrt{-g}$')
    axes[0].set_yscale('log')
    axes[0].set_title('Shell-Integrated Mass Profile')
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)
    xmin, xmax = r_mid.min(), r_mid.max()
    

    axes[1].set_xlabel(r'$r\;[R_g]$')
    axes[1].set_ylabel('Relative Error')
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(xmin, xmax)
    
    #     # set limits first
    # for ax in axes:
    #     ax.set_xlim(xmin, xmax)
    
    # # major ticks: explicitly include endpoints
    # major_ticks = [xmin, xmax]
    
    # # minor ticks: standard log subdivisions (2–9)
    # minor_locator = mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
    
    # for ax in axes:
    #     ax.xaxis.set_major_locator(mticker.FixedLocator(major_ticks))
    #     ax.xaxis.set_minor_locator(minor_locator)
    #     ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    path = os.path.join(args.save_dir, "fig5_shell_mass_profile.png")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close()
    print(r_mid.min(), r_mid.max())


# ============================================================================
#  Table 3: DivB Residual
# ============================================================================

def evaluate_divb_table(args, device):
    """DivB residual at various forecast horizons."""
    model, norm, geo = build_inference_pipeline(device)
    gdet_np = geo['gdet']
    dx, dy, dz = float(geo['dx'][0]), float(geo['dx'][1]), float(geo['dx'][2])

    horizons = [1, 10, 50, 100]
    init_data = load_dump(args.start)
    if init_data is None:
        print(f"Dump {args.start} not found."); return

    rows, divb_vals, step_labels = [], [], []

    for n_steps in horizons:
        curr = torch.tensor(init_data, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            for _ in range(n_steps):
                curr = forecast_one_step(model, norm, curr)
        divb = compute_divb(curr.cpu().numpy()[0], gdet_np, dx, dy, dz)
        rows.append(f"{n_steps:>8} {divb:>16.6e}")
        divb_vals.append(divb)
        step_labels.append(n_steps)

    gt = load_dump(args.start)
    if gt is not None:
        divb_gt = compute_divb(gt, gdet_np, dx, dy, dz)
        rows.append(f"{'GT(t=0)':>8} {divb_gt:>16.6e}")
        divb_vals.append(divb_gt)
        step_labels.append(0)

    lines = [
        f"\n{'='*45}",
        f" Table 3: Volume-Averaged |div B| Residual",
        f"{'='*45}",
        f"{'Steps':>8} {'|div B|':>16}",
        "-"*26,
    ] + rows
    for l in lines:
        print(l)

    txt_path = os.path.join(args.save_dir, "table3_divb.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {txt_path}")

    np.savez(os.path.join(args.save_dir, "table3_divb.npz"),
             steps=np.array(step_labels), divb=np.array(divb_vals))
    print(f"Saved: {args.save_dir}/table3_divb.npz")


# ============================================================================
#  Computational Cost
# ============================================================================

def evaluate_cost(args, device):
    """Parameter count and inference timing."""
    model, norm, geo = build_inference_pipeline(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    est_3d = n_params * (27.0 / 12.0)
    cost_data = {
        "total_params": n_params,
        "trainable_params": n_trainable,
        "est_3d_params": est_3d,
    }

    dummy = load_dump(args.start)
    if dummy is not None:
        curr = torch.tensor(dummy, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            for _ in range(3):
                curr = forecast_one_step(model, norm, curr)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        n_time = 20
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_time):
                curr = forecast_one_step(model, norm, curr)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        ms_per_step = (t1 - t0) / n_time * 1000
        cost_data["ms_per_step"] = ms_per_step
        timing_line = f"  Inference time:       {ms_per_step:.1f} ms/step"
    else:
        timing_line = "  (Could not time — start dump not found)"

    lines = [
        f"\n{'='*50}",
        f" Computational Cost",
        f"{'='*50}",
        f"  Total parameters:     {n_params:,} ({n_params/1e6:.2f}M)",
        f"  Trainable parameters: {n_trainable:,} ({n_trainable/1e6:.2f}M)",
        f"  Est. 3D U-NO params:  {est_3d/1e6:.2f}M (2.25x factor)",
        timing_line,
        f"  Physical time/step:   25 R_g/c",
    ]
    for l in lines:
        print(l)

    txt_path = os.path.join(args.save_dir, "cost_summary.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {txt_path}")

    np.savez(os.path.join(args.save_dir, "cost_summary.npz"), **cost_data)
    print(f"Saved: {args.save_dir}/cost_summary.npz")

def evaluate_physics_table(args, device):
    model, norm, geo = build_inference_pipeline(device)
    gdet = geo['gdet']
    dx, dy, dz = float(geo['dx'][0]), float(geo['dx'][1]), float(geo['dx'][2])
    
    horizons = [1, 10, 50, 100]
    
    init = load_dump(args.start)
    rows = []
    metrics = {"steps": [], "mass_err": [], "divb": [], "spec_err": [], "grad_err": []}

    for h in horizons:
        curr = torch.tensor(init, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            for _ in range(h):
                curr = forecast_one_step(model, norm, curr)
        pred = curr.cpu().numpy()[0]
        true = load_dump(args.start + h)

        mass = compute_mass_error(pred, true, gdet)
        divb = compute_divb(pred, gdet, dx, dy, dz)
        spec = compute_spectral_error(pred, true)
        grad = compute_gradient_error(pred, true)

        rows.append(f"{h:>6} {mass:>12.4e} {divb:>12.4e} {spec:>12.4e} {grad:>12.4e}")
        metrics["steps"].append(h)
        metrics["mass_err"].append(mass)
        metrics["divb"].append(divb)
        metrics["spec_err"].append(spec)
        metrics["grad_err"].append(grad)

    lines = [
        f"\n{'='*65}",
        " Table 4: Physics Consistency Metrics",
        f"{'='*65}",
        f"{'Steps':>6} {'Mass Err':>12} {'DivB':>12} {'Spec Err':>12} {'Grad Err':>12}",
        "-"*65,
    ] + rows
    for l in lines:
        print(l)

    txt_path = os.path.join(args.save_dir, "table4_physics.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {txt_path}")

    np.savez(os.path.join(args.save_dir, "table4_physics.npz"),
             **{k: np.array(v) for k, v in metrics.items()})
    print(f"Saved: {args.save_dir}/table4_physics.npz")
        
def plot_spectrum(args, device):
    model, norm, geo = build_inference_pipeline(device)
    init = load_dump(args.start)
    target = load_dump(args.start + 10)
    curr = torch.tensor(init, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(10):
            curr = forecast_one_step(model, norm, curr)
            
    pred = curr.cpu().numpy()[0][0]
    true = target[0]
    p_fft = np.abs(np.fft.rfftn(pred))
    t_fft = np.abs(np.fft.rfftn(true))
    p_spec = p_fft.mean(axis=(1,2))
    t_spec = t_fft.mean(axis=(1,2))
    k = np.arange(len(p_spec))
    
    plt.figure()
    plt.loglog(k, t_spec, label='Truth')
    plt.loglog(k, p_spec, '--', label='Prediction')
    plt.xlabel('Wavenumber k')
    plt.ylabel('Energy')
    plt.legend()
    plt.title('Spectral Energy Comparison')
    path = os.path.join(args.save_dir, "fig6_spectrum.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")
        
def plot_mass_metrics(args, device):
    model, norm, geo = build_inference_pipeline(device)
    gdet = geo['gdet']
    init = load_dump(args.start)
    curr = torch.tensor(init, dtype=torch.float32).unsqueeze(0).to(device)
    mass_pred = []
    mass_true = []
    steps = []
    with torch.no_grad():
        for step in range(1, args.max_steps + 1):
            curr = forecast_one_step(model, norm, curr)
            pred = curr.cpu().numpy()[0]
            true = load_dump(args.start + step)
            if true is None:
                break
            M_pred = np.sum(pred[0] * gdet)
            M_true = np.sum(true[0] * gdet)
            mass_pred.append(M_pred)
            mass_true.append(M_true)
            steps.append(step)
    mass_pred = np.array(mass_pred)
    mass_true = np.array(mass_true)
    # Mass drift (conservation)
    drift = np.abs(mass_pred - mass_pred[0]) / (np.abs(mass_pred[0]) + 1e-10)
    # Mass error (accuracy)
    error = np.abs(mass_pred - mass_true) / (np.abs(mass_true) + 1e-10)
    plt.figure()
    plt.plot(steps, drift, label="Mass Drift (Conservation)")
    plt.plot(steps, error, label="Mass Error (vs Truth)")
    plt.xlim(1,args.max_steps)
    plt.xticks([1, 20, 40, 60, 80, args.max_steps])
    plt.xlabel("Steps")
    plt.ylabel("Relative Error")
    plt.yscale("log")
    plt.legend()
    plt.title("Mass Conservation vs Accuracy")
    path = os.path.join(args.save_dir, "mass_metrics.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")
    
# ============================================================================
#  Field Comparison: Predicted vs. Ground Truth
# ============================================================================

# ============================================================================
#  Polar Field Comparison: Meridional log-r Cartesian Projection
# ============================================================================

# Each entry: (channel_spec, row_label, scale, cmap)
# channel_spec is an int (raw channel index) or 'u' / 'B' for computed magnitudes.
# scale: 'log' for log10, 'symlog' for signed fields, 'linear' for magnitudes.
_POLAR_ROWS = [
    (0,   r'$\rho$  (density)',          'log',    'inferno'),
    (1,   r'$u_g$  (energy)',            'log',    'plasma'),
    ('u', r'$|u^i|$ (fluid speed)',      'linear', 'viridis'),
    (5,   r'$B^r$',                      'symlog', 'RdBu_r'),
    (6,   r'$B^\theta$',                 'symlog', 'RdBu_r'),
    (7,   r'$B^\varphi$',               'symlog', 'RdBu_r'),
]

POLAR_STEPS = [1, 25, 50, 100]


def _polar_coords(geo):
    """
    Build meridional (X, Z) mesh in log10(r) × (sinθ, cosθ) coordinates.
    Using log10(r) compresses the large radial dynamic range so the inner
    funnel and outer disk are both visible at once.
    Returns X, Z each of shape (Nr, Ntheta).
    """
    r = geo['r'][:, :, 0]   # (Nr, Ntheta) — actual radii in R_g
    h = geo['h'][:, :, 0]   # (Nr, Ntheta) — polar angle θ in radians
    log_r = np.log10(np.maximum(r, 1.0))
    return log_r * np.sin(h), log_r * np.cos(h)


def _extract_polar_slice(data, ch_spec):
    """
    Extract (Nr, Ntheta) slice for the requested channel.
    data is either (8, Nr, Ntheta, Nphi) from a live inference array
    or (8, Nr, Ntheta) when loaded from a saved snapshots .npz.
    """
    d = data if data.ndim == 3 else data[:, :, :, 0]
    if ch_spec == 'u':
        return np.sqrt(d[2]**2 + d[3]**2 + d[4]**2)
    elif ch_spec == 'B':
        return np.sqrt(d[5]**2 + d[6]**2 + d[7]**2)
    else:
        return d[ch_spec]




# ============================================================================
#  Polar Evolution: Single-Page Time Series (step on x axis)
# ============================================================================

# Four composite quantities to keep the figure on one page.
# Each entry: (channel_spec, row_label, scale, cmap)
_EVOL_ROWS = [
    (0,   'Fluid Density\n' r'$\rho$',              'log',    'inferno'),
    (1,   'Fluid Internal Energy\n' r'$u_g$',            'log',    'plasma'),
    ('u', 'Fluid Flow Speed\n' r'$|u^i|$',           'linear', 'viridis'),
    ('B', 'Magnetic Field Strength\n' r'$|B^i|$',  'linear', 'cividis'),
]

# Sub-row labels
_SUB_ROWS = ['Model', 'Simulation', 'Rel. Error']

# Physical time per step (gravitational units)
_T_PER_STEP = 25   # R_g / c


def _apply_scale(arr, scale):
    if scale == 'log':
        return np.log10(np.clip(arr, 1e-30, None))
    return arr


def _collect_snapshots(args, device):
    """Run autoregressive inference and return snapshots dict {step: (pred, truth)}."""
    model, norm, geo = build_inference_pipeline(device)
    init_data = load_dump(args.start)
    if init_data is None:
        print(f"Dump {args.start} not found."); return None, None

    snapshots = {}
    curr = torch.tensor(init_data, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        for step in range(1, max(POLAR_STEPS) + 1):
            curr = forecast_one_step(model, norm, curr)
            if step in POLAR_STEPS:
                true_data = load_dump(args.start + step)
                if true_data is None:
                    print(f"  Skipping step {step}: dump {args.start + step} missing")
                    continue
                snapshots[step] = (curr.cpu().numpy()[0], true_data)

    # Save phi=0 meridional slice only — full 3D arrays would exceed GitHub's 100 MB limit
    npz_path = os.path.join(args.save_dir, "polar_evolution_snapshots.npz")
    np.savez_compressed(npz_path,
                        steps=np.array(list(snapshots.keys())),
                        **{f"pred_{s}": snapshots[s][0][:, :, :, 0] for s in snapshots},
                        **{f"true_{s}": snapshots[s][1][:, :, :, 0] for s in snapshots})
    print(f"Saved snapshots: {npz_path}")
    return snapshots, geo


def _load_snapshots(path):
    """Reload snapshots saved by _collect_snapshots."""
    d = np.load(path)
    steps = d["steps"].tolist()
    return {s: (d[f"pred_{s}"], d[f"true_{s}"]) for s in steps}


def _render_polar_evolution(snapshots, geo, args):
    """Render polar_evolution.png from a snapshots dict. No model needed."""
    X, Z = _polar_coords(geo)

    n_ch  = len(_EVOL_ROWS)
    n_sub = len(_SUB_ROWS)
    n_cols = len(POLAR_STEPS)
    n_rows = n_ch * n_sub

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 2.8 * n_rows),
        gridspec_kw={'hspace': 0.04, 'wspace': 0.04},
    )

    # Compute per-channel colour limits across all steps so each row is comparable
    clims = {}   # ch_spec -> (vmin, vmax)
    # Colour limits fixed across all steps for comparability
    clims = {}
    for ch_spec, _, scale, _ in _EVOL_ROWS:
        vals = []
        for step, (pred_np, true_np) in snapshots.items():
            vals.append(_apply_scale(_extract_polar_slice(pred_np, ch_spec), scale))
            vals.append(_apply_scale(_extract_polar_slice(true_np, ch_spec), scale))
        all_vals = np.concatenate([v.ravel() for v in vals])
        clims[ch_spec] = (np.percentile(all_vals, 1), np.percentile(all_vals, 99))

    err_clims = {}
    for ch_spec, _, _, _ in _EVOL_ROWS:
        errs = []
        for step, (pred_np, true_np) in snapshots.items():
            p = _extract_polar_slice(pred_np, ch_spec)
            t = _extract_polar_slice(true_np, ch_spec)
            errs.append(np.abs(p - t) / (np.abs(t) + 1e-30))
        err_clims[ch_spec] = np.percentile(np.concatenate([e.ravel() for e in errs]), 99)

    for ci, (ch_spec, ch_label, scale, cmap) in enumerate(_EVOL_ROWS):
        vmin, vmax = clims[ch_spec]
        err_vmax   = err_clims[ch_spec]

        for si, step in enumerate(POLAR_STEPS):
            if step not in snapshots:
                for sub in range(n_sub):
                    axes[ci * n_sub + sub, si].axis('off')
                continue

            pred_np, true_np = snapshots[step]
            p_raw = _extract_polar_slice(pred_np, ch_spec)
            t_raw = _extract_polar_slice(true_np, ch_spec)
            p_plot = _apply_scale(p_raw, scale)
            t_plot = _apply_scale(t_raw, scale)
            rel_err = np.abs(p_raw - t_raw) / (np.abs(t_raw) + 1e-30)

            data_panels = [p_plot, t_plot, rel_err]
            cmaps_      = [cmap, cmap, 'Reds']
            vmins_      = [vmin, vmin, 0]
            vmaxs_      = [vmax, vmax, err_vmax]

            for sub, (dat, cm, vlo, vhi) in enumerate(zip(data_panels, cmaps_, vmins_, vmaxs_)):
                row = ci * n_sub + sub
                ax  = axes[row, si]
                ax.pcolormesh(X, Z, dat, cmap=cm, vmin=vlo, vmax=vhi, shading='gouraud')
                ax.axhline(0, color='white', lw=0.4, alpha=0.35, ls='--')
                ax.axvline(0, color='white', lw=0.4, alpha=0.35, ls='--')
                ax.set_aspect('equal')
                ax.set_xticks([]); ax.set_yticks([])
                if si == 0:
                    ax.set_ylabel(f'{ch_label}\n{_SUB_ROWS[sub]}', fontsize=9, labelpad=4)

        for sub in range(n_sub):
            row = ci * n_sub + sub
            last_step = max(s for s in POLAR_STEPS if s in snapshots)
            last_si   = POLAR_STEPS.index(last_step)
            pred_np, true_np = snapshots[last_step]
            p_raw = _extract_polar_slice(pred_np, ch_spec)
            t_raw = _extract_polar_slice(true_np, ch_spec)
            rel_err = np.abs(p_raw - t_raw) / (np.abs(t_raw) + 1e-30)
            data_panels = [_apply_scale(p_raw, scale), _apply_scale(t_raw, scale), rel_err]
            cm  = [cmap, cmap, 'Reds'][sub]
            vlo = [vmin, vmin, 0][sub]
            vhi = [vmax, vmax, err_vmax][sub]
            sm  = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vlo, vhi))
            sm.set_array([])
            plt.colorbar(sm, ax=axes[row, last_si], fraction=0.046, pad=0.02)

    for si, step in enumerate(POLAR_STEPS):
        t_phys = step * _T_PER_STEP
        axes[0, si].set_title(
            f'Step {step}\n' r'$t = $' f'${t_phys}' r'\,R_g/c$',
            fontsize=10, fontweight='bold', pad=6
        )

    fig.suptitle(
        'Black Hole Accretion Disk — Field Evolution\n'
        r'Meridional cross-section ($\varphi = 0$) · '
        r'Radial axis: $\log_{10}(r / R_g)$ · '
        r'Fluid density \& energy shown on log scale',
        fontsize=11, y=1.005,
    )

    path = os.path.join(args.save_dir, 'polar_evolution.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_polar_evolution(args, device):
    """
    If --snapshots points to a saved .npz, skip inference and re-render directly.
    Otherwise run inference, save snapshots, then render.
    """
    geo = load_geometry()

    if args.snapshots:
        print(f"Loading snapshots from {args.snapshots}")
        snapshots = _load_snapshots(args.snapshots)
    else:
        snapshots, geo = _collect_snapshots(args, device)
        if snapshots is None:
            return

    _render_polar_evolution(snapshots, geo, args)




# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Paper Figure & Table Generator")
    _MODES = ["all", "single_step", "rollout_curve", "shell_profiles",
              "divb_table", "cost", "physics_table", "spectrum",
              "mass_drift", "polar_evolution"]
    parser.add_argument("--mode", nargs="+", default=["all"], choices=_MODES,
                        metavar="MODE",
                        help=f"One or more modes to run: {', '.join(_MODES)}")
    parser.add_argument("--start", type=int, default=5000, help="Starting dump ID")
    parser.add_argument("--max_steps", type=int, default=50, help="Max rollout steps")
    parser.add_argument("--save_dir", default="eval_results", help="Output directory")
    parser.add_argument("--data_dir",   default=None, help="Override path to processed dump directory")
    parser.add_argument("--geometry",   default=None, help="Override path to geometry.npz")
    parser.add_argument("--checkpoint", default=None, help="Override path to model checkpoint .pth")
    parser.add_argument("--snapshots",  default=None, help="Path to polar_evolution_snapshots.npz — skips inference for polar_evolution")
    args = parser.parse_args()

    if args.data_dir:   CONFIG["data_dir"]      = args.data_dir
    if args.geometry:   CONFIG["geometry_file"] = args.geometry
    if args.checkpoint: CONFIG["model_path"]    = args.checkpoint

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    modes = set(args.mode)
    run_all = "all" in modes

    if run_all or "single_step"   in modes: evaluate_single_step(args, device)
    if run_all or "rollout_curve" in modes: evaluate_rollout_curve(args, device)
    if run_all or "shell_profiles"in modes: evaluate_shell_profiles(args, device)
    if run_all or "divb_table"    in modes: evaluate_divb_table(args, device)
    if run_all or "cost"          in modes: evaluate_cost(args, device)
    if run_all or "physics_table" in modes: evaluate_physics_table(args, device)
    if run_all or "spectrum"      in modes: plot_spectrum(args, device)
    if run_all or "mass_drift"    in modes: plot_mass_metrics(args, device)
    if run_all or "polar_evolution" in modes: plot_polar_evolution(args, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
