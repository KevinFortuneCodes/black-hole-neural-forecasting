# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.
"""Train one seed of one experiment (see EXPERIMENTS in vit/config.py).

On Delta (see scripts/train_delta.sbatch), one node with 4 GPUs:
    torchrun --standalone --nproc_per_node=4 Fall26/vit/train.py --experiment vit_baseline_v1 --seed 0
Without torchrun it runs as a single process on one GPU or the CPU (the smoke test does this).
Outputs go to Fall26/outputs/<experiment>/seed<N>/; rerunning the same command resumes there.
"""
import argparse
import contextlib
import csv
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make Fall26/ importable

from common.data import HAMRForecastDataset, list_dumps, split_train_val
from common.normalizer import build_normalizer, check_normalizer_constants, load_geometry
from common.physics_loss import PhysicsLoss3D
from common.settings import PHYSICS
from vit.config import (ADDED_SETTINGS, DEFAULT_EXPERIMENT, RUNTIME, experiment_config,
                        run_dir as experiment_run_dir)
from vit.model import build_model

logger = logging.getLogger("vit.train")

STAT_KEYS = ["Fluid", "Mag", "Grad_Sm", "Grad_Tb", "Spec", "Saturation", "DivB", "Mass", "Gamma"]


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="name in EXPERIMENTS (vit/config.py)")
    p.add_argument("--seed", type=int, default=0)
    # Hyperparameters and data come from the experiment. --data-dir/--geometry-file replace the
    # experiment's data (tests); a run refuses to resume with different data.
    p.add_argument("--data-dir")
    p.add_argument("--geometry-file")
    # Runtime settings only.
    p.add_argument("--out-dir")
    p.add_argument("--num-workers", type=int)
    p.add_argument("--max-batches", type=int, help="limit batches per epoch (testing only)")
    p.add_argument("--stop-after", type=int, help="stop this job after N epochs (testing only)")
    return p.parse_args(argv)


def build_config(args, overrides):
    """The experiment's hyperparameters plus runtime settings. `overrides` is for tests only."""
    cfg = experiment_config(args.experiment)
    for key in ("data_dir", "geometry_file", "out_dir", "num_workers", "max_batches", "stop_after"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    cfg.update(overrides or {})
    cfg["seed"] = args.seed
    return cfg


def run_settings(cfg, world):
    """What defines a run: every hyperparameter and the data, but not paths to outputs or loader
    settings. JSON-normalized so a saved config.json compares equal."""
    skip = {"out_dir", "num_workers", "log_every", "max_batches", "stop_after", "description"}
    settings = {k: v for k, v in cfg.items() if k not in skip}
    settings["world_size"] = world
    return json.loads(json.dumps(settings))


def record_config(path, cfg, world, rank, distributed):
    """Write config.json for a new run; for an existing run, refuse if its settings changed."""
    current = run_settings(cfg, world)
    if rank == 0 and not path.exists():
        path.write_text(json.dumps({**current, "description": cfg.get("description", "")}, indent=2))
    if distributed:
        dist.barrier()
    saved = {**ADDED_SETTINGS, **json.loads(path.read_text())}
    saved.pop("description", None)
    changed = sorted(k for k in set(saved) | set(current) if saved.get(k) != current.get(k))
    if changed:
        detail = ", ".join(f"{k}: {saved.get(k)!r} -> {current.get(k)!r}" for k in changed)
        raise ValueError(f"{path} was written by a run with different settings ({detail}). "
                         "Add a new experiment in vit/config.py instead of changing this one.")


def setup_distributed():
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank, local_rank = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    else:
        rank = local_rank = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world, device


def setup_logging(run_dir, rank):
    root = logging.getLogger()
    root.handlers.clear()
    if rank != 0:
        root.setLevel(logging.ERROR)
        return
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "train.log")):
        handler.setFormatter(fmt)
        root.addHandler(handler)


def shutdown(distributed):
    if distributed:
        dist.destroy_process_group()
    for handler in logging.getLogger().handlers:
        handler.close()
    logging.getLogger().handlers.clear()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=Path(__file__).parent).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def lr_at(step, schedule):
    """Linear warmup from 0 to the peak, then cosine decay to min_lr at the last step."""
    peak, low = schedule["peak"], schedule["min_lr"]
    warmup, total = schedule["warmup_steps"], schedule["total_steps"]
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return low + (peak - low) * 0.5 * (1.0 + math.cos(math.pi * progress))


def param_groups(model, weight_decay):
    """AdamW groups: weight decay on weight matrices and kernels, none on biases or LayerNorm."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if name.endswith("bias") or p.ndim == 1 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def make_loader(files, cfg, shuffle, distributed, seed):
    """Training (shuffle): DistributedSampler, which pads so every rank gets equal batches.
    Validation: each rank takes every world-th sample, so every sample is scored exactly once."""
    ds = HAMRForecastDataset(files, cfg["unroll_steps"])
    sampler, data = None, ds
    if distributed and shuffle:
        sampler = DistributedSampler(ds, shuffle=True, seed=seed)
    elif distributed:
        data = Subset(ds, range(dist.get_rank(), len(ds), dist.get_world_size()))
    loader = DataLoader(data, batch_size=cfg["batch_size"], sampler=sampler,
                        shuffle=shuffle and sampler is None, num_workers=cfg["num_workers"],
                        pin_memory=torch.cuda.is_available(),
                        persistent_workers=cfg["num_workers"] > 0)
    return ds, sampler, loader


def train_one_epoch(model, crit, norm, loader, opt, cfg, ep, device, distributed, schedule, step0):
    """One pass over the training data; step0 is the global optimizer step at the epoch's start."""
    model.train()
    unroll = cfg["unroll_steps"]
    params = [p for p in model.parameters() if p.requires_grad]
    # Running sums stay on the GPU; they are read once per log line and at the end of the epoch.
    sums = dict.fromkeys(STAT_KEYS, 0.0)
    loss_sum, n_batches, skipped = 0.0, 0, 0
    for b, seq in enumerate(loader):
        if cfg["max_batches"] and b >= cfg["max_batches"]:
            break
        seq = [s.to(device, non_blocking=True) for s in seq]
        for group in opt.param_groups:
            group["lr"] = lr_at(step0 + b, schedule)
        opt.zero_grad(set_to_none=True)
        curr = norm(seq[0])
        batch_loss = 0.0
        for step in range(1, unroll + 1):
            # The unrolled steps are detached from each other, so a backward per step gives the
            # same gradients as one backward of the summed loss while holding only one step's
            # activations. DDP all-reduces gradients on the last step only.
            last = step == unroll
            sync = model.no_sync() if distributed and not last else contextlib.nullcontext()
            with sync:
                pred = curr + model(curr)
                loss, s = crit(pred, seq[step], norm, ep)
                (loss / unroll).backward()
            curr = pred.detach()
            batch_loss += loss.detach() / unroll
            for k in STAT_KEYS:
                sums[k] += s[k] / unroll
        # Gradients are identical on every rank after the all-reduce, so ranks agree on skipping.
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        if torch.isfinite(grad_norm):
            opt.step()
        else:
            skipped += 1
            opt.zero_grad(set_to_none=True)
            logger.warning(f"Ep {ep} batch {b}: non-finite gradient norm, batch skipped.")
        loss_sum += batch_loss
        n_batches += 1
        if b % cfg["log_every"] == 0:
            logger.info(f"Ep {ep} batch {b}/{len(loader)} loss {float(batch_loss):.4f} "
                        f"grad_norm {float(grad_norm):.3f}")
    n = max(n_batches, 1)
    return {"loss": float(loss_sum) / n, **{k: float(v) / n for k, v in sums.items()},
            "skipped": skipped, "batches": n_batches}


@torch.no_grad()
def validate(net, crit, norm, loader, cfg, ep, device, distributed):
    """Per-sample mean total, reconstruction (Fluid + Mag) and spectral loss over all unrolled
    steps. Every validation sample counts once, whatever the batch sizes and number of ranks."""
    net.eval()
    acc = torch.zeros(4, dtype=torch.float64, device=device)   # total, recon, spec (sample-weighted), samples
    for b, seq in enumerate(loader):
        if cfg["max_batches"] and b >= cfg["max_batches"]:
            break
        curr = norm(seq[0].to(device))
        n = seq[0].shape[0]
        for step in range(1, cfg["unroll_steps"] + 1):
            pred = curr + net(curr)
            total, s = crit(pred, seq[step].to(device), norm, ep)
            acc += torch.stack([total.double() * n, (s["Fluid"] + s["Mag"]).double() * n,
                                s["Spec"].double() * n, torch.tensor(float(n), dtype=torch.float64, device=device)])
            curr = pred
    if distributed:
        dist.all_reduce(acc)
    samples = acc[3].item()
    if samples == 0:
        raise RuntimeError("validation produced no samples: need more than unroll_steps validation dumps")
    return {"total": acc[0].item() / samples, "recon": acc[1].item() / samples, "spec": acc[2].item() / samples}


def save_checkpoint(path, state):
    """Write to a temporary file first so a job killed mid-save never corrupts the checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def append_metrics(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def main(argv=None, overrides=None):
    args = parse_args(argv)
    cfg = build_config(args, overrides)
    rank, world, device = setup_distributed()
    distributed = world > 1

    run_dir = experiment_run_dir(cfg["experiment"], cfg["seed"], cfg["out_dir"])
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    record_config(run_dir / "config.json", cfg, world, rank, distributed)
    setup_logging(run_dir, rank)

    random.seed(cfg["seed"] + rank)
    np.random.seed(cfg["seed"] + rank)
    torch.manual_seed(cfg["seed"])   # same initial weights on every rank

    logger.info(f"Run {cfg['experiment']}/seed{cfg['seed']} | commit {git_commit()} | "
                f"{world} process(es) on {device}")
    logger.info(f"Experiment: {cfg['description']}")
    logger.info(f"Config: {cfg}")

    # A job chained after a finished or early-stopped run exits before loading anything.
    latest = run_dir / "latest.pth"
    ckpt = torch.load(latest, map_location="cpu", weights_only=False) if latest.exists() else None
    if ckpt is not None:
        finished = ckpt["epoch"] >= cfg["epochs"]
        stopped = ckpt["epoch"] > cfg["stable_epoch"] and ckpt["patience"] >= cfg["early_stop_patience"]
        if finished or stopped:
            logger.info("This run already stopped early; nothing to do." if stopped
                        else "This run already finished all its epochs; nothing to do.")
            shutdown(distributed)
            return {"run_dir": run_dir, "last_epoch": ckpt["epoch"] - 1, "stopped_early": stopped}

    files = list_dumps(cfg["data_dir"])
    train_files, val_files = split_train_val(files, cfg["val_frac"])
    train_ds, train_sampler, train_loader = make_loader(train_files, cfg, True, distributed, cfg["seed"])
    val_ds, _, val_loader = make_loader(val_files, cfg, False, distributed, cfg["seed"])
    logger.info(f"Dumps: {len(train_files)} train ({Path(train_files[0]).name}..{Path(train_files[-1]).name}), "
                f"{len(val_files)} validation ({Path(val_files[0]).name}..{Path(val_files[-1]).name})")

    geo_npz = np.load(cfg["geometry_file"])
    coords = (geo_npz["r"], geo_npz["h"], geo_npz["ph"])
    sample_grid = tuple(train_ds[0][0].shape[1:])
    assert sample_grid == tuple(coords[0].shape), \
        f"dumps are {sample_grid} but geometry.npz is {tuple(coords[0].shape)}"

    # Our physics phase-in timing replaces TRON's in the shared settings PhysicsLoss3D reads.
    PHYSICS.update(recon_only_epochs=cfg["recon_only_epochs"],
                   curriculum_ramp_epochs=cfg["curriculum_ramp_epochs"],
                   curriculum_stable_epoch=cfg["stable_epoch"])

    geo = load_geometry(cfg["geometry_file"], device)
    norm = build_normalizer(cfg["normalization"], gcov=geo["gcov"], gcon=geo["gcon"]).to(device)
    crit = PhysicsLoss3D(device, cfg["geometry_file"]).to(device)
    if rank == 0:
        check_normalizer_constants(cfg["geometry_file"])

    model = build_model(cfg, coords).to(device)
    n_tokens = model.n_tokens
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters, "
                f"{model.n_vars} channels ({cfg['normalization']} normalization), "
                f"grid {model.grid}, patch {model.patch}, {n_tokens:,} tokens")

    opt = torch.optim.AdamW(param_groups(model, cfg["weight_decay"]), lr=cfg["learning_rate"])
    steps_per_epoch = len(train_loader)
    if cfg["max_batches"]:
        steps_per_epoch = min(steps_per_epoch, cfg["max_batches"])
    schedule = {"peak": cfg["learning_rate"], "min_lr": cfg["min_lr"],
                "warmup_steps": cfg["warmup_epochs"] * steps_per_epoch,
                "total_steps": cfg["epochs"] * steps_per_epoch}
    logger.info(f"Schedule: {steps_per_epoch} steps per epoch, warmup {schedule['warmup_steps']:,} "
                f"then cosine to {cfg['min_lr']:.0e} at step {schedule['total_steps']:,}")

    start_ep, best_loss, patience = 0, float("inf"), 0
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])          # copies onto the model's device
        opt.load_state_dict(ckpt["opt"])              # moves optimizer state to the parameters' device
        start_ep, best_loss, patience = ckpt["epoch"], ckpt["best_loss"], ckpt["patience"]
        logger.info(f"Resumed from {latest} at epoch {start_ep} (best val recon {best_loss:.4f}, "
                    f"patience {patience})")
        del ckpt

    net = model
    if distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    stop = False
    ep = start_ep
    for ep in range(start_ep, cfg["epochs"]):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)

        t0 = time.time()
        tr = train_one_epoch(model, crit, norm, train_loader, opt, cfg, ep, device, distributed,
                             schedule, ep * steps_per_epoch)
        clamp_train = norm.lorentz_clamp_count(reset=True).to(device)
        va = validate(net, crit, norm, val_loader, cfg, ep, device, distributed)
        clamps = torch.stack([clamp_train, norm.lorentz_clamp_count(reset=True).to(device)])
        if distributed:
            dist.all_reduce(clamps)        # cells clamped on all ranks: [train, validation]
        minutes = (time.time() - t0) / 60

        stop_flag = torch.zeros(1, device=device)
        if rank == 0:
            # Best checkpoint and early stopping start once the physics curriculum is fully on.
            if ep >= cfg["stable_epoch"]:
                if va["recon"] < best_loss:
                    best_loss, patience = va["recon"], 0
                    improved = True
                else:
                    patience += 1
                    improved = False
                if patience >= cfg["early_stop_patience"]:
                    stop_flag[0] = 1.0
            else:
                improved = False
            state = {"epoch": ep + 1, "model": net.state_dict(), "opt": opt.state_dict(),
                     "best_loss": best_loss, "patience": patience,
                     "val_recon": va["recon"], "config": cfg}
            save_checkpoint(latest, state)
            if improved:
                save_checkpoint(run_dir / "best.pth", state)

            append_metrics(run_dir / "metrics.csv", {
                "epoch": ep, "lr": opt.param_groups[0]["lr"], "minutes": round(minutes, 2),
                "train_loss": tr["loss"], **{f"train_{k}": tr[k] for k in STAT_KEYS},
                "val_total": va["total"], "val_recon": va["recon"], "val_spec": va["spec"],
                "best_val_recon": best_loss, "patience": patience,
                "skipped_batches": tr["skipped"],
                "train_lorentz_clamped_cells": int(clamps[0]), "val_lorentz_clamped_cells": int(clamps[1])})
            logger.info(f"Ep {ep} | {minutes:.1f} min | LR {opt.param_groups[0]['lr']:.2e} | "
                        f"train {tr['loss']:.4f} | val total {va['total']:.4f} recon {va['recon']:.4f} | "
                        f"best {best_loss:.4f} | patience {patience}/{cfg['early_stop_patience']}"
                        + (" | saved best" if improved else ""))
            if stop_flag.item():
                logger.info(f"Early stopping: no improvement in validation recon for {patience} epochs.")
        if distributed:
            dist.broadcast(stop_flag, src=0)
        if stop_flag.item():
            stop = True
            break
        if cfg["stop_after"] and ep + 1 - start_ep >= cfg["stop_after"]:
            logger.info(f"Stopping this job after {cfg['stop_after']} epoch(s) (stop_after).")
            break

    shutdown(distributed)
    return {"run_dir": run_dir, "last_epoch": ep, "stopped_early": stop}


if __name__ == "__main__":
    main()
