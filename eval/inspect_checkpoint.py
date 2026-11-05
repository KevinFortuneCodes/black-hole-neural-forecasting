# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.
# Usage: python eval/inspect_checkpoint.py [path/to/best_forecast.pth]
import sys

import torch
import numpy as np

ckpt = torch.load(
    sys.argv[1] if len(sys.argv) > 1 else "eval_results/best_forecast.pth",
    map_location="cpu", weights_only=False
)

print(f"Keys in checkpoint: {list(ckpt.keys())}")
print(f"Epoch: {ckpt.get('epoch', 'N/A')}")
print(f"Best loss: {ckpt.get('best_loss', 'N/A'):.4f}")

if 'crit' in ckpt:
    log_vars = ckpt['crit']['log_vars']
    weights = torch.exp(-log_vars).numpy()
    names = ["Fluid", "Mag", "Grad", "Spec", "DivB", "Mass", "Gamma"]
    print(f"\n{'Component':<12} {'log_var':>10} {'weight (exp(-lv))':>18}")
    print("-" * 42)
    for name, lv, w in zip(names, log_vars.numpy(), weights):
        print(f"{name:<12} {lv:>10.4f} {w:>18.4f}")
else:
    print("No 'crit' key found in checkpoint.")
