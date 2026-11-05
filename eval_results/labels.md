Total — weighted sum of all components. Train drops from 16.7 → 5.7, but val only goes from 16.6 → 14.0. That gap is a sign of overfitting.

Fluid — L1 error on ρ, u_g, and velocity channels. Barely moves (0.083 → 0.091 train), consistent with what you saw in single-step metrics — the model isn't improving much on fluid prediction.

Mag — L1 error on B^r, B^θ, B^φ. Gets worse over training (0.062 → 0.127 train, 0.063 → 0.154 val). The model is degrading on magnetic fields, which matches your single-step results showing it can't beat persistence on B channels.

Grad — spatial gradient error on ρ, penalizing incorrect structure. Roughly flat, slightly worse on val — the model isn't learning sharp spatial features, consistent with the spectral smoothing you saw in fig 6.

Spec — spectral energy error. Barely changes (~33 throughout) — the model never learns to match the energy spectrum, which directly explains the fig 6 result.

DivB — divergence of B, should be zero (∇·B = 0 is a fundamental constraint). Improves on train (1.33 → 0.82) but gets worse on val (1.30 → 1.62) — the model enforces ∇·B on training data but doesn't generalize it.

Mass — relative mass conservation error. Gets significantly worse over training (0.002 → 0.042 train, 0.002 → 0.033 val) — the model is learning to trade away mass conservation as it optimizes other terms.

Gamma — Lorentz factor penalty, keeps the fluid velocities physically valid (Γ > 1). Val explodes from 0 → 17.9 while train only goes to 3.4 — a major red flag. The model is producing unphysical velocities on unseen data, meaning it's violating special relativity on the validation set.

The most concerning findings are Mag getting worse, Gamma exploding on val, and the large train/val gap on Total. The model has learned to fit the training distribution but isn't generalizing well to unseen timesteps.