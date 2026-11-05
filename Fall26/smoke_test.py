# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.
"""Smoke tests for the ViT baseline.

Synthetic (default, runs on CPU in about a minute):
    python Fall26/smoke_test.py
  Builds a tiny fake dataset and checks the model (shapes, parameter count, Δ = 0 at
  start, padding, positional encoding, bf16 path, per-step backward) and the training
  loop (checkpoints, metrics, resume, early stopping).

Real data on one Delta GPU (see scripts/smoke_delta.sbatch):
    python Fall26/smoke_test.py --real [--experiment vit_rc300_v1] [--batch-size 8] [--steps 5]
  Runs full-size training steps on real dumps and reports time per step, peak GPU
  memory and the estimated epoch time on a 4-GPU node.
"""
import argparse
import csv
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))   # make Fall26/ importable

from vit.config import CONFIG, DEFAULT_EXPERIMENT, EXPERIMENTS, eval_dir, experiment_config, run_dir
from vit.model import ViTForecaster, build_model, patch_centres, sinusoidal_encoding

FULL_GRID = (192, 96, 96)
PASSED = []


def check(name, condition, detail=""):
    if not condition:
        raise AssertionError(f"FAILED: {name} {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"  ({detail})" if detail else ""))


def synthetic_coords(grid, r_in=1.24, r_out=1960.0):
    """Log-spaced r, uniform theta and phi, matching the real grid's layout."""
    nr, nt, npp = grid
    r = np.exp(np.linspace(np.log(r_in), np.log(r_out), nr))
    th = np.linspace(0.05, np.pi - 0.05, nt)
    ph = (np.arange(npp) + 0.5) * 2 * np.pi / npp
    R, T, P = np.meshgrid(r, th, ph, indexing="ij")
    return R, T, P


def write_synthetic_dataset(root, grid=(24, 16, 16), n_dumps=40):
    """Flat-metric geometry plus smooth, slowly rotating fields."""
    nr, nt, npp = grid
    R, T, P = synthetic_coords(grid, r_out=100.0)
    eye = np.diag([-1.0, 1.0, 1.0, 1.0])[:, :, None, None, None] * np.ones((1, 1, nr, nt, 1))
    np.savez(root / "geometry.npz", r=R, h=T, ph=P,
             gdet=np.ones((nr, nt, 1), np.float32), gcov=eye.astype(np.float32),
             gcon=eye.astype(np.float32),
             dx=np.array([np.log(R[1, 0, 0] / R[0, 0, 0]), T[0, 1, 0] - T[0, 0, 0], P[0, 0, 1] - P[0, 0, 0]]),
             eos_gamma=np.array(13 / 9), dt_dump=np.array(25.0))
    for t in range(n_dumps):
        wave = np.sin(P - 0.2 * t) * np.sin(T) / np.sqrt(R)
        rho = 10 ** (-2 + 0.5 * wave)
        dump = np.stack([rho, 0.1 * rho, 0.05 * wave, 0.02 * wave, 0.3 / np.sqrt(R),
                         0.01 * np.cos(T) / R, 0.01 * wave, 0.02 * wave]).astype(np.float32)
        np.save(root / f"dump_{t}.npy", dump)


def small_model(coords, **kw):
    args = dict(patch=(8, 8, 8), d_model=32, depth=2, n_heads=4, mlp_ratio=4,
                use_amp=False, grad_checkpoint=False)
    args.update(kw)
    return ViTForecaster(coords, **args)


def run_synthetic():
    torch.manual_seed(0)
    print("Experiment checks")
    keys = set(EXPERIMENTS[DEFAULT_EXPERIMENT])
    check("every experiment spells out the same full set of hyperparameters",
          all(set(e) == keys for e in EXPERIMENTS.values()), f"{len(EXPERIMENTS)} experiment(s), {len(keys)} keys")

    print("Model checks")
    patch, d, depth, mlp = CONFIG["patch"], CONFIG["d_model"], CONFIG["depth"], CONFIG["mlp_ratio"]
    vol = int(np.prod(patch))
    n_patches = int(np.prod([g // p for g, p in zip(FULL_GRID, patch)]))
    # Independent count: per block LN 2x2d, qkv 3d^2+3d, proj d^2+d, MLP 2*mlp*d^2 + mlp*d + d;
    # per variable an embedding (vol*d + d) and an output layer (d*vol + vol); final LN 2d.
    expected = (depth * ((4 + 2 * mlp) * d * d + (9 + mlp) * d)
                + 8 * (vol * d + d) + 8 * (d * vol + vol) + 2 * d)
    full = build_model(CONFIG, synthetic_coords(FULL_GRID))
    n_params = sum(p.numel() for p in full.parameters())
    check("full-size parameter count", n_params == expected, f"{n_params:,}")
    check("full-size token count", full.n_tokens == 8 * n_patches,
          f"{full.pos.shape[0]:,} patches x {full.groups} tokens per patch = {full.n_tokens:,}")
    check("one embedding per variable", tuple(full.embed.weight.shape) == (8 * d, 1, *patch)
          and full.embed.groups == 8)
    del full

    pe = sinusoidal_encoding(*(patch_centres(c, patch) for c in synthetic_coords(FULL_GRID)), d)
    check("positional encoding: every dimension varies with position", bool((pe.std(0) > 1e-6).all()))
    n_distinct = torch.unique(torch.round(pe.T * 1e4), dim=0).shape[0]
    check("positional encoding: no two dimensions are identical", n_distinct == d, f"{n_distinct} of {d} distinct")
    r_c, t_c, p_c = (patch_centres(c, patch) for c in synthetic_coords(FULL_GRID))
    pe_wrapped = sinusoidal_encoding(r_c, t_c, p_c + 2 * np.pi, d)
    check("positional encoding: periodic in phi", torch.allclose(pe, pe_wrapped, atol=1e-4))

    coords = synthetic_coords((24, 16, 16))
    x = torch.randn(2, 8, 24, 16, 16)
    model = small_model(coords)
    out = model(x)
    check("output shape equals input shape", out.shape == x.shape, str(tuple(out.shape)))
    check("predicts zero change at start (persistence)", torch.count_nonzero(out).item() == 0)

    print("Token mode checks")
    pp = small_model(coords, token_mode="per_patch")
    n_patch = (24 // 8) * (16 // 8) * (16 // 8)
    check("per_patch: one token per patch, all variables in one embedding",
          pp.n_tokens == n_patch and tuple(pp.embed.weight.shape) == (32, 8, 8, 8, 8),
          f"{pp.n_tokens} tokens for {n_patch} patches")
    check("per_patch: output shape and zero change at start",
          pp(x).shape == x.shape and torch.count_nonzero(pp(x)).item() == 0)
    for m in (model, pp):                 # each output channel and in-patch cell lands where it should
        vol = int(np.prod(m.patch))
        with torch.no_grad():
            m.out_weight.zero_()
            chan = torch.arange(m.n_vars, dtype=torch.float32).view(m.groups, m.vars_per_token, 1)
            cell = torch.arange(vol, dtype=torch.float32).view(1, 1, vol) / vol
            m.out_bias.copy_((chan + cell).reshape(m.groups, -1))
        o = m(x)[0]
        Pr, Pt, Pp = m.patch
        expect = (torch.arange(m.n_vars).view(-1, 1, 1, 1)
                  + (torch.arange(vol, dtype=torch.float32) / vol).view(1, Pr, Pt, Pp))
        ok = all(torch.allclose(o[:, i * Pr:(i + 1) * Pr, j * Pt:(j + 1) * Pt, k * Pp:(k + 1) * Pp], expect)
                 for i in range(o.shape[1] // Pr) for j in range(o.shape[2] // Pt) for k in range(o.shape[3] // Pp))
        check(f"{m.token_mode}: each variable and patch cell decodes to its place in the grid", ok)
        with torch.no_grad():
            m.out_bias.zero_()
    torch.nn.init.normal_(pp.out_weight, std=0.02)
    pp_ckpt = small_model(coords, token_mode="per_patch", grad_checkpoint=True)
    pp_ckpt.load_state_dict(pp.state_dict())
    pp.train(); pp_ckpt.train()
    pp(x).sum().backward(); pp_ckpt(x).sum().backward()
    check("per_patch: gradient checkpointing gives the same gradients",
          all(torch.allclose(a.grad, b.grad, atol=1e-5) for a, b in zip(pp.parameters(), pp_ckpt.parameters())))

    odd = small_model(synthetic_coords((30, 14, 18)))
    check("pads and crops grids that patches don't divide",
          odd(torch.randn(1, 8, 30, 14, 18)).shape == (1, 8, 30, 14, 18))

    tiny_cfg = dict(CONFIG, d_model=32, depth=2, n_heads=4, use_amp=False, grad_checkpoint=False)
    for name, n in (("arcsinh", 8), ("magdir", 10)):
        m = build_model(dict(tiny_cfg, normalization=name), coords)
        x_n = torch.randn(1, n, 24, 16, 16)
        check(f"{name} normalization builds a {n}-channel model", m.n_vars == n and m(x_n).shape == x_n.shape)

    torch.nn.init.normal_(model.out_weight, std=0.02)   # make the output non-trivial for the next checks
    ref = model(x)
    model.use_amp = True
    amp = model(x)
    model.use_amp = False
    rel = ((amp - ref).norm() / ref.norm()).item()
    check("bf16 blocks keep an fp32 output close to full precision", amp.dtype == torch.float32 and rel < 0.05,
          f"relative difference {rel:.4f}")

    # One backward per unrolled step must match one backward of the summed loss. Training now
    # predicts a single step (unroll_steps = 1); this keeps unroll_steps > 1 correct if re-enabled.
    seq = [torch.randn(2, 8, 24, 16, 16) for _ in range(3)]
    def losses():
        curr, out = seq[0], []
        for t in (1, 2):
            pred = curr + model(curr)
            out.append(((pred - seq[t]) ** 2).mean())
            curr = pred.detach()
        return out
    model.zero_grad()
    sum(losses()).backward()
    g_sum = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    curr = seq[0]
    for t in (1, 2):
        pred = curr + model(curr)
        ((pred - seq[t]) ** 2).mean().backward()
        curr = pred.detach()
    same = all(torch.allclose(a, p.grad, atol=1e-6) for a, p in zip(g_sum, model.parameters()))
    check("per-step backward gives the same gradients as the summed loss (unroll_steps > 1)", same)

    ckpt_model = small_model(coords, grad_checkpoint=True)
    ckpt_model.load_state_dict(model.state_dict())
    ckpt_model.train()
    model.train()
    model.zero_grad()
    model(x).sum().backward()
    g_plain = [p.grad.clone() for p in model.parameters()]
    ckpt_model(x).sum().backward()
    same = all(torch.allclose(a, p.grad, atol=1e-5) for a, p in zip(g_plain, ckpt_model.parameters()))
    check("gradient checkpointing gives the same gradients", same)

    print("Schedule checks")
    from vit import train
    from common.settings import PHYSICS
    sched = {"peak": 5e-4, "min_lr": 1e-6, "warmup_steps": 100, "total_steps": 1000}
    lrs = [train.lr_at(s, sched) for s in range(1000)]
    check("learning rate warms up linearly from near 0 to the peak",
          lrs[0] < 1e-5 and abs(lrs[99] - 5e-4) < 1e-12 and all(a < b for a, b in zip(lrs[:99], lrs[1:100])),
          f"step 0: {lrs[0]:.1e}, step 99: {lrs[99]:.1e}")
    check("then decays by cosine to min_lr at the last step",
          all(a >= b for a, b in zip(lrs[100:], lrs[101:])) and abs(lrs[-1] - 1e-6) < 1e-8
          and abs(lrs[550] - (1e-6 + (5e-4 - 1e-6) / 2)) < 1e-6,
          f"midpoint {lrs[550]:.2e}, last {lrs[-1]:.1e}")
    groups = train.param_groups(model, 0.05)
    no_decay = {id(p) for p in groups[1]["params"]}
    names_no_decay = sorted({n.split(".")[-1] for n, p in model.named_parameters() if id(p) in no_decay})
    check("weight decay skips biases and LayerNorm weights",
          groups[0]["weight_decay"] == 0.05 and groups[1]["weight_decay"] == 0.0
          and id(model.out_weight) not in no_decay and id(model.out_bias) in no_decay
          and id(model.blocks[0].norm1.weight) in no_decay and id(model.blocks[0].qkv.weight) not in no_decay,
          f"no decay on: {', '.join(names_no_decay)}")

    tiny = {"d_model": 32, "depth": 2, "n_heads": 4, "use_amp": False, "grad_checkpoint": False,
            "batch_size": 2, "num_workers": 0, "log_every": 1000}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data = tmp / "data"
        data.mkdir()
        write_synthetic_dataset(data)
        common = ["--data-dir", str(data), "--geometry-file", str(data / "geometry.npz"),
                  "--out-dir", str(tmp / "out")]

        print("Normalization checks")
        from common.normalizer import DataNormalizer, build_normalizer, load_geometry
        from common.physics_loss import PhysicsLoss3D
        geo = load_geometry(data / "geometry.npz")
        crit = PhysicsLoss3D(torch.device("cpu"), data / "geometry.npz")
        raw = torch.from_numpy(np.stack([np.load(data / f"dump_{t}.npy") for t in (0, 1)]))
        for name in ("arcsinh", "magdir"):
            norm = build_normalizer(name, gcov=geo["gcov"], gcon=geo["gcon"])
            z = norm(raw)
            back = norm.inverse(z)
            check(f"{name}: {norm.n_channels} channels that decode back to the 8 physical ones",
                  z.shape[1] == norm.n_channels and back.shape == raw.shape
                  and torch.allclose(back, raw, rtol=1e-4, atol=1e-7))
            total, s = crit(z + 0.01 * torch.randn_like(z), raw, norm, 1000)
            check(f"{name}: physics loss finite with every term on",
                  bool(torch.isfinite(total)) and all(bool(torch.isfinite(torch.as_tensor(v))) for v in s.values()),
                  f"total {float(total):.3f}")
        unit = torch.linalg.vector_norm(z[:, 3:6], dim=1), torch.linalg.vector_norm(z[:, 7:10], dim=1)
        check("magdir: velocity and field directions are unit vectors",
              all(torch.allclose(u, torch.ones_like(u), atol=1e-5) for u in unit))
        a, b = torch.randn(2, 8, 24, 16, 16), torch.randn(2, 8, 24, 16, 16)
        pf, tf = (torch.fft.rfftn(t * crit.spec_win, dim=[-3, -2, -1]) for t in (a, b))
        diff, t_abs = (pf.abs() - tf.abs()).abs(), tf.abs()
        tron = sum(diff[:, s].mean() / t_abs[:, s].mean().clamp(min=f)
                   for s, f in ((slice(0, 2), 5.0), (slice(2, 5), 0.08), (slice(5, 8), 0.03))) / 3
        check("arcsinh: spectral loss equals TRON's fixed channel groups",
              torch.allclose(crit.spectral_loss(a, b, DataNormalizer.spectral_groups), tron))

        print("Training loop checks")

        base = dict(tiny, stable_epoch=0, epochs=3)
        # Job 1 stops after 2 of 3 epochs, as if it hit Delta's 48-hour limit.
        res = train.main(common + ["--stop-after", "2"], overrides=base)
        run = res["run_dir"]
        check("outputs are named after the experiment and seed",
              run == run_dir(DEFAULT_EXPERIMENT, 0, tmp / "out"), str(run.relative_to(tmp / "out")))
        saved = json.loads((run / "config.json").read_text())
        check("config.json records every hyperparameter of the experiment",
              all(k in saved for k in EXPERIMENTS[DEFAULT_EXPERIMENT])
              and saved["experiment"] == DEFAULT_EXPERIMENT and saved["seed"] == 0,
              f"{len(saved)} settings saved")
        check("evaluation outputs are named after the experiment, seed, checkpoint and split",
              eval_dir(DEFAULT_EXPERIMENT, 0, "best", "blind", tmp / "out") == run / "eval" / "best_blind",
              str(eval_dir(DEFAULT_EXPERIMENT, 0, "best", "blind", tmp / "out").relative_to(tmp / "out")))
        rows = (run / "metrics.csv").read_text().strip().splitlines()
        check("trains 2 epochs and logs metrics", len(rows) == 3, f"{len(rows) - 1} epochs logged")
        check("writes latest and best checkpoints", (run / "latest.pth").exists() and (run / "best.pth").exists())
        ckpt = torch.load(run / "latest.pth", weights_only=False)
        fresh = small_model(synthetic_coords((24, 16, 16)), patch=CONFIG["patch"],
                            mlp_ratio=CONFIG["mlp_ratio"])
        fresh.load_state_dict(ckpt["model"], strict=True)
        check("checkpoint 'model' key loads strictly (what evaluation_new.py reads)", True)
        check("physics loss uses our phase-in timing, not TRON's",
              (PHYSICS["recon_only_epochs"], PHYSICS["curriculum_ramp_epochs"])
              == (CONFIG["recon_only_epochs"], CONFIG["curriculum_ramp_epochs"]),
              f"reconstruction only to epoch {PHYSICS['recon_only_epochs']}, "
              f"full by epoch {PHYSICS['recon_only_epochs'] + PHYSICS['curriculum_ramp_epochs']}")
        with open(run / "metrics.csv") as f:
            last = list(csv.DictReader(f))[-1]
        check("training and validation losses are finite and non-zero",
              all(np.isfinite(float(last[k])) and float(last[k]) > 0
                  for k in ("train_loss", "val_total", "val_recon")),
              f"train {float(last['train_loss']):.3f}, val recon {float(last['val_recon']):.3f}")

        # Runs from before the "normalization" setting have no such key in config.json; they must still resume.
        saved.pop("normalization")
        (run / "config.json").write_text(json.dumps(saved, indent=2))
        res = train.main(common, overrides=base)   # job 2 resumes and finishes
        rows = (run / "metrics.csv").read_text().strip().splitlines()
        check("resumes from latest.pth and continues, also from a config.json older than a new setting",
              len(rows) == 4 and rows[-1].startswith("2,"), "epoch 2 appended after restart")
        try:
            train.main(common, overrides=dict(base, epochs=4))
            refused = False
        except ValueError:
            refused = True
        check("refuses to resume a run whose experiment settings changed", refused)

        # Learning rate 0: validation never improves, so patience 1 must stop at the second epoch.
        stuck = dict(tiny, stable_epoch=0, early_stop_patience=1, learning_rate=0.0, min_lr=0.0, epochs=10)
        res = train.main(common + ["--seed", "1"], overrides=stuck)
        check("early stopping fires when validation stops improving",
              res["stopped_early"] and res["last_epoch"] == 1, f"stopped after epoch {res['last_epoch']}")
        res = train.main(common + ["--seed", "1"], overrides=stuck)
        rows = (res["run_dir"] / "metrics.csv").read_text().strip().splitlines()
        check("a resumed job does not continue a run that already stopped early",
              res["stopped_early"] and len(rows) == 3, f"{len(rows) - 1} epochs logged")

        res = train.main(common + ["--seed", "2"], overrides=dict(tiny, stable_epoch=0, epochs=1, normalization="magdir"))
        with open(res["run_dir"] / "metrics.csv") as f:
            last = list(csv.DictReader(f))[-1]
        check("trains with the magdir normalization (10 channels)",
              json.loads((res["run_dir"] / "config.json").read_text())["normalization"] == "magdir"
              and all(np.isfinite(float(last[k])) for k in ("train_loss", "val_total", "val_recon")),
              f"train {float(last['train_loss']):.3f}, val recon {float(last['val_recon']):.3f}")

    print(f"\nAll {len(PASSED)} synthetic checks passed.")


def run_real(args):
    from torch.utils.data import DataLoader
    from common.data import HAMRForecastDataset, list_dumps, split_train_val
    from common.normalizer import NORMALIZERS, build_normalizer, load_geometry
    from common.physics_loss import PhysicsLoss3D
    from vit.train import train_one_epoch, validate

    assert torch.cuda.is_available(), "--real needs a GPU"
    device = torch.device("cuda")
    cfg = dict(experiment_config(args.experiment), batch_size=args.batch_size, num_workers=args.num_workers,
               max_batches=args.steps + 1, log_every=1)
    print(f"torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}")

    files = list_dumps(cfg["data_dir"])
    train_files, val_files = split_train_val(files, cfg["val_frac"])
    print(f"{len(files)} dumps: {len(train_files)} train, {len(val_files)} validation")
    geo_npz = np.load(cfg["geometry_file"])
    coords = (geo_npz["r"], geo_npz["h"], geo_npz["ph"])
    print(f"grid {coords[0].shape}")

    from torch.nn.attention import SDPBackend, sdpa_kernel
    per_patch = NORMALIZERS[cfg["normalization"]].n_channels if cfg["token_mode"] == "per_variable" else 1
    n_tokens = per_patch * int(np.prod([-(-g // p) for g, p in zip(coords[0].shape, cfg["patch"])]))
    head_dim = cfg["d_model"] // cfg["n_heads"]
    q = torch.randn(1, cfg["n_heads"], n_tokens, head_dim, device=device, dtype=torch.bfloat16)
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            torch.nn.functional.scaled_dot_product_attention(q, q, q)
        print(f"flash attention kernel: available for bf16, {n_tokens:,} tokens, head dim {head_dim}")
    except RuntimeError as e:
        print(f"flash attention kernel: NOT available ({e}); PyTorch will fall back to another kernel")

    model = build_model(cfg, coords).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"{model.n_tokens:,} tokens")
    geo = load_geometry(cfg["geometry_file"], device)
    norm = build_normalizer(cfg["normalization"], gcov=geo["gcov"], gcon=geo["gcon"]).to(device)
    crit = PhysicsLoss3D(device, cfg["geometry_file"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    train_ds = HAMRForecastDataset(train_files, cfg["unroll_steps"])
    val_ds = HAMRForecastDataset(val_files, cfg["unroll_steps"])
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)

    t0 = time.time()
    batch = next(iter(loader))
    print(f"first batch loaded in {time.time() - t0:.1f} s; sample shape {tuple(batch[0].shape)}")
    with torch.no_grad():
        delta = model(norm(batch[0].to(device)))
    print(f"predicted change at start: max |Δ| = {delta.abs().max().item():.2e} (expect 0)")

    torch.cuda.reset_peak_memory_stats()
    # Epoch 150: all physics terms on, so the timing includes the full loss.
    schedule = {"peak": cfg["learning_rate"], "min_lr": cfg["min_lr"], "warmup_steps": 1, "total_steps": 100}
    cfg_one = dict(cfg, max_batches=1)
    train_one_epoch(model, crit, norm, loader, opt, cfg_one, 150, device, False, schedule, 0)   # warm-up step
    torch.cuda.synchronize()
    t0 = time.time()
    stats = train_one_epoch(model, crit, norm, loader, opt, dict(cfg, max_batches=args.steps),
                            150, device, False, schedule, 1)
    torch.cuda.synchronize()
    sec_train = (time.time() - t0) / stats["batches"]
    peak = torch.cuda.max_memory_allocated() / 2**30

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True)
    t0 = time.time()
    va = validate(model, crit, norm, val_loader, dict(cfg, max_batches=2), 150, device, False)
    torch.cuda.synchronize()
    sec_val = (time.time() - t0) / 2

    gpus = 4
    train_batches = math.ceil(len(train_ds) / (args.batch_size * gpus))
    val_batches = math.ceil(len(val_ds) / (args.batch_size * gpus))
    epoch_min = (train_batches * sec_train + val_batches * sec_val) / 60
    print(f"\ntraining loss {stats['loss']:.4f}, validation recon {va['recon']:.4f} (finite: "
          f"{np.isfinite(stats['loss']) and np.isfinite(va['recon'])})")
    print(f"time per training step (batch {args.batch_size}, {cfg['unroll_steps']} prediction step(s), "
          f"data loading included): {sec_train:.2f} s")
    print(f"time per validation batch: {sec_val:.2f} s")
    print(f"peak GPU memory: {peak:.1f} GiB of {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
    print(f"estimated epoch on {gpus} GPUs: {epoch_min:.1f} min "
          f"({train_batches} train + {val_batches} validation batches per GPU); "
          f"about {48 * 60 / epoch_min:.0f} epochs per 48-hour job")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", action="store_true", help="run on real data on a GPU")
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="experiment whose data and model to time (--real)")
    p.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    p.add_argument("--steps", type=int, default=5, help="timed training steps (--real)")
    p.add_argument("--num-workers", type=int, default=CONFIG["num_workers"])
    a = p.parse_args()
    run_real(a) if a.real else run_synthetic()
