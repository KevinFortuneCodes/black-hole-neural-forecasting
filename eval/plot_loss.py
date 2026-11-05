# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.
import os
import numpy as np
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = np.load(os.path.join(_ROOT, "eval_results", "loss_history.npz"), allow_pickle=True)
epochs = np.arange(1, len(d["train_Total"]) + 1)

components = ["Total", "Fluid", "Mag", "Grad", "Spec", "DivB", "Mass", "Gamma"]

fig, axes = plt.subplots(4, 2, figsize=(14, 16))
axes = axes.flatten()

for ax, name in zip(axes, components):
    train = d[f"train_{name}"]
    val   = d[f"val_{name}"]
    ax.plot(epochs, train, label="Train")
    ax.plot(epochs, val,   label="Val")
    ax.set_title(name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(_ROOT, "eval_results", "loss_curves.png")
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
