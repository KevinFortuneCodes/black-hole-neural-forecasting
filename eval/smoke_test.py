# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.
"""Local smoke test — no DDP, no real data, CPU only.
Validates: PhysicsLoss3D returns all 8 components, loss_history.npz saves and reloads correctly.
"""
import sys, os, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SFNO"))
import train_forecast as tf

# ── Synthetic geometry ────────────────────────────────────────────────────────
tmpdir = tempfile.mkdtemp()
log_dir = os.path.join(tmpdir, "logs")
os.makedirs(log_dir)
geo_path = os.path.join(tmpdir, "geometry.npz")

NRAD, NLAT, NLON = 4, 8, 16
np.savez(
    geo_path,
    gdet=np.ones((NRAD, NLAT, NLON), np.float32),
    gcov=np.eye(4, dtype=np.float32),
    dx=np.array([0.1, 0.1, 0.1], np.float32),
)

tf.CONFIG["geometry_file"] = geo_path
tf.CONFIG["log_dir"] = log_dir

# ── Instantiate loss + normalizer ─────────────────────────────────────────────
device = torch.device("cpu")
norm = tf.DataNormalizer()
crit = tf.PhysicsLoss3D(device)

# ── Synthetic batch (B=2, C=8, D=4, H=8, W=16) ───────────────────────────────
B, C, D, H, W = 2, 8, NRAD, NLAT, NLON
pred_norm = torch.randn(B, C, D, H, W)
target_raw = torch.abs(torch.randn(B, C, D, H, W)) + 1e-5
target_raw[:, 0] = target_raw[:, 0].clamp(min=1e-20)  # rho
target_raw[:, 1] = target_raw[:, 1].clamp(min=1e-22)  # ug

# ── Forward pass (standard) ───────────────────────────────────────────────────
total, components = crit(pred_norm, target_raw, norm, epoch=1, total_epochs=10)
print(f"Total loss : {total.item():.6f}")
for k, v in components.items():
    print(f"  {k:12s}: {v:.6f}")

EXPECTED_KEYS = {"Fluid", "Mag", "Grad", "Spec", "DivB", "Mass", "Gamma"}
missing = EXPECTED_KEYS - set(components.keys())
assert not missing, f"Missing component keys: {missing}"
extra = set(components.keys()) - EXPECTED_KEYS
assert not extra, f"Unexpected component keys: {extra}"
print("Component keys OK")

# ── return_raw=True ────────────────────────────────────────────────────────────
pred_norm_grad = pred_norm.detach().requires_grad_(True)
total_r, components_r, raw_losses = crit(pred_norm_grad, target_raw, norm, epoch=1, total_epochs=10, return_raw=True)
assert raw_losses.shape == (7,), f"Expected raw_losses shape (7,), got {raw_losses.shape}"
assert raw_losses.requires_grad, "raw_losses must retain grad for per-component gnorm pass"
# Verify each component is differentiable w.r.t. pred_norm
for i in range(7):
    g = torch.autograd.grad(raw_losses[i], pred_norm_grad, retain_graph=True)[0]
    assert g is not None and g.shape == pred_norm_grad.shape, f"Component {i} grad failed"
print("return_raw OK")

# ── NPZ save / reload ─────────────────────────────────────────────────────────
loss_hist_path = os.path.join(log_dir, "loss_history.npz")
loss_history = {"train": {}, "val": {}}
gnorm_history = {}
for k, v in components.items():
    loss_history["train"].setdefault(k, []).append(float(v))
    loss_history["val"].setdefault(k, []).append(float(v) * 0.9)
# Simulate gnorm entry (real values come from torch.autograd.grad in training)
fake_gnorm = {k: float(i) for i, k in enumerate(EXPECTED_KEYS)}
for k, v in fake_gnorm.items():
    gnorm_history.setdefault(k, []).append(v)

np.savez(
    loss_hist_path,
    **{f"train_{k}": np.array(v) for k, v in loss_history["train"].items()},
    **{f"val_{k}": np.array(v) for k, v in loss_history["val"].items()},
    **{f"gnorm_{k}": np.array(v) for k, v in gnorm_history.items()},
    loss_scales=crit.loss_scales.detach().numpy(),
)

saved = np.load(loss_hist_path)
print(f"NPZ keys: {sorted(saved.files)}")
for key in EXPECTED_KEYS:
    assert f"train_{key}" in saved.files, f"Missing train_{key} in NPZ"
    assert f"val_{key}" in saved.files,   f"Missing val_{key} in NPZ"
    assert f"gnorm_{key}" in saved.files, f"Missing gnorm_{key} in NPZ"
assert "train_Curriculum" not in saved.files, "Stale Curriculum key found in NPZ"
assert "loss_scales" in saved.files, "Missing loss_scales in NPZ"
assert saved["loss_scales"].shape == (7,), f"Expected loss_scales shape (7,), got {saved['loss_scales'].shape}"
print(f"loss_scales: {saved['loss_scales']}")
print("NPZ save/load OK")

# ── Second epoch — verify accumulation appends and loading handles all key types ─
saved2 = np.load(loss_hist_path, allow_pickle=True)
gnorm_history2 = {}
for key in saved2.files:
    if key == "loss_scales":
        continue
    prefix, name = key.split("_", 1)
    if prefix in ("train", "val"):
        loss_history[prefix][name] = saved2[key].tolist()
    elif prefix == "gnorm":
        gnorm_history2[name] = saved2[key].tolist()
assert set(gnorm_history2.keys()) == EXPECTED_KEYS, f"gnorm reload mismatch: {gnorm_history2.keys()}"

total2, components2 = crit(pred_norm, target_raw, norm, epoch=2, total_epochs=10)
for k, v in components2.items():
    loss_history["train"].setdefault(k, []).append(float(v))
    loss_history["val"].setdefault(k, []).append(float(v) * 0.9)
for k, v in fake_gnorm.items():
    gnorm_history2.setdefault(k, []).append(v)

np.savez(
    loss_hist_path,
    **{f"train_{k}": np.array(v) for k, v in loss_history["train"].items()},
    **{f"val_{k}": np.array(v) for k, v in loss_history["val"].items()},
    **{f"gnorm_{k}": np.array(v) for k, v in gnorm_history2.items()},
    loss_scales=crit.loss_scales.detach().numpy(),
)

saved3 = np.load(loss_hist_path)
assert saved3["train_Fluid"].shape == (2,), f"Expected 2 epochs, got {saved3['train_Fluid'].shape}"
print("Epoch accumulation OK (2 entries per key)")

print("\nSMOKE TEST PASSED")
