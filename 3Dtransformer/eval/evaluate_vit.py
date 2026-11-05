# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.
"""Score a ViT checkpoint with TRON's evaluation suite, so both models get the same metrics.

evaluate_forecast_ds.py is TRON's evaluator, copied unchanged. Only two of its functions touch
the model: build_pipeline (loads TRON) and forecast_one_step (one residual step). This script
swaps in ViT versions of those two and runs the rest as is:
  - its `from uno_simple_ds import ...` gets our common/ copies (same normalizer and physics),
  - "TRON" in its table headers, legends and notes reads "ViT".
The forecast step matches training: pred = x + model(x) in the run's own normalization ("arcsinh"
or "magdir", from its config.json), then TRON's decode (normalizer inverse with the Lorentz clamp,
rho and ug clamped to the floors and ceilings). The suite's "normalized space" metrics (nMAE, skill,
conditional error, Lipschitz) always use TRON's arcsinh normalization, so runs with different
normalizations get comparable numbers.

Defaults come from the experiment: the blind test set in <data_dir>/blind, the run's best.pth,
and results in 3Dtransformer/outputs/<experiment>/seed<N>/eval/<checkpoint>_blind/. Before TRON's suite it
plots the run's training and validation curves there too (plot_training.py; training_curves.txt).
From the repo root:
    python 3Dtransformer/eval/evaluate_vit.py --experiment vit_rc300_v1 --mode all --max_steps 50
Every other argument (--mode, --max_steps, --n_starts, --start, ...) goes to TRON's evaluator.
"""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # make 3Dtransformer/ importable

from common.normalizer import DataNormalizer, build_normalizer, load_geometry
from common.physics_loss import PhysicsLoss3D
from common.settings import PHYSICS
from vit.config import ADDED_SETTINGS, eval_dir, run_dir
from vit.model import build_model
from plot_training import plot_training


def load_tron_evaluator():
    """evaluate_forecast_ds.py as a module, importing our common/ in place of uno_simple_ds."""
    shim = types.ModuleType("uno_simple_ds")
    shim.DataNormalizer, shim.PhysicsLoss3D, shim.load_geometry = DataNormalizer, PhysicsLoss3D, load_geometry
    shim.CONFIG = PHYSICS
    shim.ViT = shim.Hybrid_UNO_3D = None   # model classes; build_pipeline is replaced below
    sys.modules["uno_simple_ds"] = shim

    path = HERE / "evaluate_forecast_ds.py"
    module = types.ModuleType("evaluate_forecast_ds")
    module.__file__ = str(path)
    source = path.read_text(encoding="utf-8").replace("TRON", "ViT")
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def make_build_pipeline(run_config):
    def build_pipeline(geometry_file, model_path, device):
        """ViT + normalizer + physics terms. Returns (model, norm, norm_cpu, crit, geo, geo_raw).
        norm and norm_cpu are TRON's arcsinh normalizer, the space the suite's metrics use; the
        model's own normalizer goes on the model as model.normalizer, for forecast_one_step."""
        PHYSICS["geometry_file"] = geometry_file
        geo_raw = np.load(geometry_file)
        geo = load_geometry(geometry_file, device)
        model = build_model(run_config, (geo_raw["r"], geo_raw["h"], geo_raw["ph"])).to(device)
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"Loaded {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M-param ViT "
              f"checkpoint {model_path} (epoch {ckpt.get('epoch', '?')}, best val recon "
              f"{ckpt.get('best_loss', '?')}), {model.n_vars} channels, "
              f"{run_config['normalization']} normalization")
        model.normalizer = build_normalizer(run_config["normalization"],
                                            gcov=geo["gcov"], gcon=geo["gcon"]).to(device)
        norm = DataNormalizer(gcov=geo["gcov"], gcon=geo["gcon"]).to(device)
        norm_cpu = DataNormalizer(gcov=geo["gcov"].cpu(), gcon=geo["gcon"].cpu())
        crit = PhysicsLoss3D(device, geometry_file).to(device)
        return model, norm, norm_cpu, crit, geo, geo_raw
    return build_pipeline


def forecast_one_step(model, norm, curr_state, device):
    """One step as in training (pred = x + model(x)) in the model's own normalization, decoded the
    way TRON's evaluator decodes. The suite reads Lorentz-clamp counts from `norm` (its metric
    normalizer), so the model normalizer's count is passed on to it."""
    rep = model.normalizer
    norm_in = rep(curr_state)
    pred_norm = norm_in + model(norm_in)
    out = rep.inverse(pred_norm).float()
    norm._lorentz_clamp_cells = norm._lorentz_clamp_cells + rep.lorentz_clamp_count(reset=True)
    out[:, 0] = torch.clamp(out[:, 0], min=PHYSICS["floor_rho"], max=PHYSICS["ceiling_rho"])
    out[:, 1] = torch.clamp(out[:, 1], min=PHYSICS["floor_ug"], max=PHYSICS["ceiling_ug"])
    return out


def _last_value(argv, option):
    """The value TRON's parser will use for `option`: its last occurrence wins."""
    values = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == option]
    return Path(values[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0], add_help=False)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--train_seed", type=int, default=0, help="seed of the run to score (--seed is the evaluator's)")
    parser.add_argument("--checkpoint", default="best", choices=["best", "latest"])
    ours, rest = parser.parse_known_args()

    run = run_dir(ours.experiment, ours.train_seed)
    run_config = {**ADDED_SETTINGS, **json.loads((run / "config.json").read_text())}   # the run's settings
    defaults = ["--data_dir", str(Path(run_config["data_dir"]) / "blind"),
                "--geometry_file", run_config["geometry_file"],
                "--model_path", str(run / f"{ours.checkpoint}.pth"),
                "--save_dir", str(eval_dir(ours.experiment, ours.train_seed, ours.checkpoint, "blind"))]

    save_dir = _last_value(defaults + rest, "--save_dir")
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        lines = plot_training(run, save_dir)
        (save_dir / "training_curves.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
    except Exception as e:   # the curves are a convenience; never let them stop the evaluation
        print(f"[Warning] training curves skipped: {e!r}")

    ev = load_tron_evaluator()
    ev.build_pipeline = make_build_pipeline(run_config)
    ev.forecast_one_step = forecast_one_step
    # TRON's parser keeps the last value of a repeated option, so arguments given here win.
    sys.argv = [sys.argv[0]] + defaults + rest
    ev.main()


if __name__ == "__main__":
    main()
