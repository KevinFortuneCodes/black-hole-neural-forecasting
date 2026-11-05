# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.
"""Training and validation curves of one run, from its metrics.csv and config.json.

Saves two figures and returns a short text summary:
  fig0_training_total.png   total loss, train and validation, with the learning rate on a second
                            axis; below it, the training total split into each term's weighted
                            share, so the plot shows which terms are active in each phase.
  fig0_loss_components.png  one line chart per loss term (unweighted, as logged), plus the
                            physics weight and early-stopping patience.
Vertical lines mark the epochs where terms switch on: ∇·B and mass-shell terms enter at the
first epoch with a nonzero curriculum weight and reach full weight at stable_epoch, where
best.pth tracking starts. Every other term is on from epoch 0 (common/physics_loss.py).
evaluate_vit.py runs this into the evaluation folder. On its own, from the repo root:
    python Fall26/eval/plot_training.py --experiment vit_rc300_v1
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make Fall26/ importable

from common.settings import PHYSICS
from vit.config import eval_dir, run_dir

# (logged key, label, weight key in PHYSICS, gated by the curriculum, color)
TERMS = [
    ("Fluid", "fluid reconstruction", "loss_weight_data", False, "#1f77b4"),
    ("Mag", "magnetic reconstruction", "loss_weight_data", False, "#aec7e8"),
    ("Grad_Sm", "gradient, ρ and u_g", "loss_weight_grad_smooth", False, "#2ca02c"),
    ("Grad_Tb", "gradient, ũ and B", "loss_weight_grad_turb", False, "#98df8a"),
    ("Spec", "spectrum", "loss_weight_spec", False, "#17becf"),
    ("DivB", "∇·B vs truth", "loss_weight_divb", True, "#d62728"),
    ("Mass", "mass shells", "loss_weight_mass", True, "#ff7f0e"),
    ("Saturation", "guard: saturation", "loss_weight_guards", False, "#7f7f7f"),
    ("Gamma", "guard: Lorentz γ", "loss_weight_guards", False, "#bcbd22"),
]


def read_metrics(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def physics_weight(epochs, recon_only, ramp):
    """The curriculum weight on the gated terms, as PhysicsLoss3D.forward computes it."""
    return np.where(epochs < recon_only, 0.0, np.minimum(1.0, (epochs - recon_only) / ramp))


def plot_training(run, out_dir):
    """Write both figures for the run in `run` to `out_dir`; return the summary lines."""
    m = read_metrics(run / "metrics.csv")
    cfg = json.loads((run / "config.json").read_text())
    ep = m["epoch"]
    w_phys = physics_weight(ep, cfg["recon_only_epochs"], cfg["curriculum_ramp_epochs"])
    gated_on = int(ep[np.argmax(w_phys > 0)]) if (w_phys > 0).any() else None
    stable = cfg["stable_epoch"]
    train_recon = m["train_Fluid"] + m["train_Mag"]
    val_recon = m["val_recon"]
    # best.pth: lowest validation recon from the stable epoch on (vit/train.py)
    eligible = ep >= stable
    best = int(ep[eligible][np.argmin(val_recon[eligible])]) if eligible.any() else None

    plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.linewidth": 1.1,
                         "savefig.dpi": 200, "savefig.bbox": "tight"})
    events = ([gated_on] if gated_on is not None else []) + [stable]
    # The phases between events, named by what is active; written above the top panel
    phase_names = [f"all terms at full weight\n(best.pth tracked from epoch {stable})"]
    if gated_on is not None:
        phase_names = [f"∇·B, mass off\n(epochs 0–{gated_on - 1})",
                       f"∇·B, mass ramping in\n(weight {w_phys[ep == gated_on][0]:.2f} → 1)"] + phase_names
    phase_edges = [ep[0]] + events + [ep[-1]]

    def mark(ax, labels=False):
        for x in events:
            ax.axvline(x, color="0.25", lw=1.1)
        if labels:
            for lo, hi, name in zip(phase_edges[:-1], phase_edges[1:], phase_names):
                ax.text((lo + hi) / 2, 1.02, name, transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=8.5, color="0.2")
        if best is not None:
            ax.axvline(best, color="0.45", ls=":", lw=1.2)
        ax.grid(True, alpha=0.3)

    # Figure 1: total loss with learning rate, and the weighted terms that make it up
    fig, (ax, ax_s) = plt.subplots(2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]})
    mark(ax, labels=True)
    ax.plot(ep, m["train_loss"], color="#1f77b4", lw=1.8, label="Train total")
    ax.plot(ep, m["val_total"], color="#d62728", lw=1.8, label="Validation total")
    ax.set_ylabel("Total loss")
    ax.set_title("Total loss and learning rate (solid lines: terms switch on; dotted: best.pth)", pad=34)
    lr_ax = ax.twinx()
    lr_ax.plot(ep, m["lr"], color="0.35", ls="--", lw=1.3, label="Learning rate")
    lr_ax.set_yscale("log")
    lr_ax.set_ylabel("Learning rate", color="0.35")
    lr_ax.tick_params(axis="y", colors="0.35")
    handles = ax.get_legend_handles_labels()[0] + lr_ax.get_legend_handles_labels()[0]
    ax.legend(handles, [h.get_label() for h in handles], loc="upper left", bbox_to_anchor=(1.09, 1),
              fontsize=8.5, frameon=False)

    mark(ax_s)
    shares, labels, colors = [], [], []
    for key, label, wkey, gated, color in TERMS:
        share = PHYSICS[wkey] * m[f"train_{key}"] * (w_phys if gated else 1.0)
        if share.max() > 0:
            shares.append(share)
            labels.append(f"{label} ×{PHYSICS[wkey]:g}" + (" × curriculum" if gated else ""))
            colors.append(color)
    ax_s.stackplot(ep, shares, labels=labels, colors=colors, alpha=0.9)
    ax_s.plot(ep, m["train_loss"], color="black", lw=1.0, label="logged train total")
    ax_s.set_ylabel("Weighted share of train total")
    ax_s.set_xlabel("Epoch")
    ax_s.set_title("What the training total is made of (terms that stay at zero are left out)")
    ax_s.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8.5, frameon=False)
    fig.suptitle(f"{cfg['experiment']}, seed {cfg['seed']}", fontsize=13)
    fig.tight_layout()
    total_path = out_dir / "fig0_training_total.png"
    fig.savefig(total_path)
    plt.close(fig)

    # Figure 2: one line chart per term
    fig, axes = plt.subplots(3, 4, figsize=(16, 10.5), sharex=True)
    panels = [("recon", "reconstruction (fluid + magnetic)", None, False)] + \
             [(key, label, wkey, gated) for key, label, wkey, gated, _ in TERMS]
    for a, (key, label, wkey, gated) in zip(axes.flat, panels):
        mark(a)
        if key == "recon":
            a.plot(ep, train_recon, color="#1f77b4", lw=1.5, label="train")
            a.plot(ep, val_recon, color="#d62728", lw=1.5, label="validation")
            if best is not None:
                a.plot(best, val_recon[ep == best][0], "o", color="#d62728", ms=5)
            a.legend(fontsize=8.5)
            a.set_title(label, fontsize=11)
        else:
            y = m[f"train_{key}"]
            a.plot(ep, y, color="#1f77b4", lw=1.5, label="train")
            if key == "Spec":
                a.plot(ep, m["val_spec"], color="#d62728", lw=1.5, label="validation")
                a.legend(fontsize=8.5)
            weight = f"weight {PHYSICS[wkey]:g}" + (" × curriculum" if gated else "")
            a.set_title(f"{label}\n({weight})", fontsize=10.5)
            if y.max() == 0:
                a.text(0.5, 0.75, "0 every epoch", transform=a.transAxes, ha="center", color="0.4")
        positive = key == "recon" or m[f"train_{key}"].min() > 0
        if positive:
            a.set_yscale("log")
    a = axes.flat[len(panels)]
    mark(a)
    a.plot(ep, w_phys, color="#d62728", lw=1.5)
    a.set_title("curriculum weight on ∇·B and mass", fontsize=10.5)
    a.set_ylim(-0.05, 1.05)
    a = axes.flat[len(panels) + 1]
    mark(a)
    a.step(ep, m["patience"], where="mid", color="#ff7f0e", lw=1.5)
    a.axhline(cfg["early_stop_patience"], color="#ff7f0e", ls="--", lw=0.9)
    a.set_title(f"early-stopping patience (stops at {cfg['early_stop_patience']})", fontsize=10.5)
    for a in axes[-1]:
        a.set_xlabel("Epoch")
    fig.suptitle(f"Loss components, {cfg['experiment']} seed {cfg['seed']} (unweighted, as logged; "
                 f"solid lines: terms switching on; dotted: best.pth)", fontsize=12.5)
    fig.tight_layout()
    comp_path = out_dir / "fig0_loss_components.png"
    fig.savefig(comp_path)
    plt.close(fig)

    lines = [f"Saved: {total_path}", f"Saved: {comp_path}", "",
             "Training curves (reconstruction = fluid + magnetic, density-weighted MAE in normalized space)",
             f"  {len(ep)} epochs (0..{int(ep[-1])}). On from epoch 0: reconstruction, gradients, spectrum, guards.",
             f"  ∇·B and mass shells: off before epoch {gated_on}, ramping to full weight at epoch {stable}."]
    for label, i in [("first epoch", 0), (f"epoch {stable} (stable)", int(np.argmax(ep >= stable))),
                     (f"best.pth (epoch {best})", int(np.flatnonzero(ep == best)[0]) if best is not None else None),
                     ("last epoch", len(ep) - 1)]:
        if i is not None:
            lines.append(f"  {label:<24} recon train {train_recon[i]:.4f}  val {val_recon[i]:.4f}  "
                         f"val/train {val_recon[i] / train_recon[i]:.2f}")
    lines.append("  val/train near 1: no overfitting. Rising while validation stays flat: the model keeps "
                 "fitting the training dumps without forecasting new ones better.")
    # Validation logs only total, recon and spectrum, so its other terms appear only as total - recon
    s, e = int(np.argmax(ep >= stable)), len(ep) - 1
    lines.append(f"  total - recon (gradient, spectrum, ∇·B, mass, guards, weighted), epoch {stable} -> last: "
                 f"train {m['train_loss'][s] - train_recon[s]:.4f} -> {m['train_loss'][e] - train_recon[e]:.4f}, "
                 f"validation {m['val_total'][s] - val_recon[s]:.4f} -> {m['val_total'][e] - val_recon[e]:.4f}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--out_dir", default=None, help="default: the run's eval/best_blind/")
    args = parser.parse_args()
    out = Path(args.out_dir) if args.out_dir else eval_dir(args.experiment, args.train_seed)
    out.mkdir(parents=True, exist_ok=True)
    print("\n".join(plot_training(run_dir(args.experiment, args.train_seed), out)))


if __name__ == "__main__":
    main()
