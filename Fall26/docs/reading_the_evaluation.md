# Reading the evaluation report

`Fall26/eval/evaluate_vit.py` runs TRON's evaluation suite (`evaluate_forecast_ds.py`, copied from the
`sachin` branch) on a ViT checkpoint. With `--mode all` it prints about 15 tables to `out_res.txt` and
saves 20-odd figures, all in `Fall26/outputs/<experiment>/seed<N>/eval/<checkpoint>_blind/`.

This guide explains what each table measures and how to read it, in the order worth reading them.
The worked examples are TRON's own report (`fall26/eval_results_v3/out_res.txt` on the `sachin` branch:
the 5000run, 192 × 96 × 96, Δt = 25 R_g/c, 50-step rollouts). **Those numbers illustrate how to read a
table; they are not a benchmark for the ViT.** rc300 is a different simulation with Δt = 100 R_g/c, so
no number in the two reports is directly comparable (see "rc300 caveats" at the end).

## Five ideas that the whole report rests on

**1. Normalized space.** Most ML metrics are computed after the normalizer, not in physical units:
log10 for density ρ and internal energy u_g, arcsinh for the three velocities ũ and the three magnetic
field components B. So an error of 0.05 in ρ means about 0.05 dex, i.e. ~12% off, and errors in
different channels are on roughly comparable scales. Physical-unit errors appear separately (Rel L1, the
physics tables).

**2. nMAE** is the mean absolute error in normalized space, per channel, averaged over every grid cell
with equal weight. It is the quantity the model was trained on (up to the physics terms).

**3. Persistence** is the trivial forecast "nothing changes": predict that the next dump equals the
current one (for a rollout, that every future dump equals the starting one). It is the baseline every
learned model has to beat. Any channel that barely evolves in one step is easy for persistence.

**4. Skill score** = 1 − nMAE_model / nMAE_persistence.
- 1 is perfect, 0 is no better than persistence, negative is worse than persistence.
- 0.3 means the model removes 30% of the error that persistence makes.

**5. Single-step vs rollout.** Single-step feeds the true dump in and scores one step ahead, over every
test pair. A rollout feeds the model its own output for many steps, so errors compound. Single-step
measures accuracy; rollouts measure whether the forecast stays useful and physical.

**One caveat to keep in mind throughout:** Tables 1 and 4, the rollout curve, the bias maps and the
conditional error average over many test pairs or 10 rollout starts. **Most of the physics tables
(2, 5, 8, 9, drift, Lorentz, the spectra and slices) follow one rollout from one start dump**
(`--start`, the first blind dump by default). Treat them as a case study, not a statistic: a single
trajectory can look unusually good or bad.

## Read in this order

### 0. Training curves (`fig0_training_total.png`, `fig0_loss_components.png`, `training_curves.txt`)

`evaluate_vit.py` draws these first, from the run's `metrics.csv` (`plot_training.py`). They show
how the scored checkpoint was reached.

- **The loss is phased in.** Reconstruction, both gradient terms, the spectrum and the guards are on
  from epoch 0. ∇·B and mass shells are off until `recon_only_epochs`, then ramp up to full weight at
  `stable_epoch` (11 and 30 for `vit_rc300_v1`). Vertical lines mark those epochs. Each new term raises
  the total, so compare total loss only within one phase. The stacked panel shows each term's weighted
  share of the training total.
- **The fair train-vs-validation comparison is reconstruction** (fluid + magnetic), the first panel of
  the component grid. Validation logs only total, reconstruction and spectrum.
- Good: validation follows training down and validation/train stays near 1. Warning sign: the ratio
  grows while validation stays flat. `vit_rc300_v1` goes from 1.16 at epoch 30 to 1.76 at epoch 91,
  and its validation reconstruction is flat from about epoch 40.

### 1. Table 1: Single-step accuracy (the headline)

```
Channel      nMAE TRON  Rel L1 TRON  nMAE persist  Skill SS           95% CI
rho             0.0458       0.3615        0.0833     0.451 [ 0.433, 0.471]
utilde_th       0.0024       0.6217        0.0024     0.006 [ 0.006, 0.007]
B_th            0.0019       0.4867        0.0019     0.000 [ 0.000, 0.000]
Mean            0.0200       0.3081        0.0320     0.249                —
```

- **nMAE model** vs **nMAE persist**: the model should be lower in every channel.
- **Skill SS** is the column to read first. TRON removes 45% of persistence's error on ρ and 25% on
  average, but **≈ 0 on ũ^θ and B^θ**: for those channels it effectively copies its input. A skill of
  zero with a tiny nMAE is not "accurate"; it means the field barely changes in one step and the model
  learned nothing beyond that.
- **95% CI** is a bootstrap confidence interval on the skill. Consecutive dumps are correlated, so the
  script keeps every second pair (`n_eff` = half the pairs). An interval that excludes 0 means the
  model reliably beats persistence on that channel.
- **Rel L1** is the physical-space relative error, mean|pred − true| / mean|true|. It is large for
  channels whose true mean is near zero (ũ^θ, B^θ average to ~0 over the sphere), so read it only for
  ρ, u_g, ũ^r, ũ^φ, B^r, B^φ.
- The **Fluid / Magnetic** lines are channel means of nMAE (ρ, u_g, ũ, then B).

### 2. Fig 4 and attractor saturation: how errors grow over a rollout

`fig4_rollout_analysis.png`, panel (a): mean nMAE vs step for the model (10 starts, ± 1 std) and for
persistence. Panel (b): each channel's error divided by its step-1 error, i.e. how fast each grows.

- Good: the model curve stays below persistence for many steps, and the spread across starts stays
  narrow.
- Warning: the curves cross early (the model is worse than doing nothing), or the error grows
  exponentially (instability).

**Attractor saturation** compares the error at the last step with the data's own spread (σ per
channel, normalized). `OK (typical set)` means the forecast is still in the range of plausible
states; `EXCEEDS data sigma` means it has drifted somewhere the simulation never goes (blow-up or
collapse). TRON is OK on every channel at 50 steps.

### 3. Table 4: Predictability horizon (how long the forecast is useful)

```
Channel       tau_c [dump / Rg/c]   skill<0 at   horizon/tau_c   horizon [Rg/c]
rho                   1.72 / 43.0           43           25.02           1075.0
B_th                  0.54 / 13.4            1            1.86             25.0
```

- **τ_c** is from the *truth* only: how many dumps the turbulence takes to decorrelate (lag-1
  autocorrelation of the fluctuations in the disk). It is the natural clock of the flow.
- **skill<0 at** is the first rollout step at which the model stops beating persistence. Multiply by
  Δt to get the **horizon** in R_g/c. `>50` means it never crossed within the rollout.
- **horizon/τ_c** says how many decorrelation times the forecast stays useful. TRON's ρ: 25 τ_c;
  B^θ: crosses at step 1 (consistent with its zero skill in Table 1).
- "Re-crossing channels": skill went negative and later positive again. That is noise around zero,
  not recovery; trust the first crossing.

### 4. Table 7 (stability) and Lorentz validity: is the model numerically safe?

**Lipschitz L**: perturb the input by 0.1% at random and measure how much the output moves, as a ratio.
- L ≤ 1: perturbations shrink (non-expansive; TRON: 0.90). Rollouts won't blow up from small errors.
- 1 < L ≤ 2: weakly expansive, normal for chaotic flow.
- L > 2: small errors amplify quickly; expect unstable rollouts.

**Lorentz validity**: γ is the Lorentz factor of the predicted velocity before the clamp at γ = 50.
Compare the rollout's max and 99.9th percentile with the truth's. `Lorentz clamp fires 0` is good; a
nonzero count means the model predicted superluminal-ish velocities that had to be clipped.

### 5. Table 2: Physics consistency over the rollout

One rollout, scored at steps 1, 10, 25, 50. **Always compare a "pred" column with its "truth" column,
never with zero**, because the truth itself has nonzero residuals on this grid.

| Column | What it is | Read it as |
|---|---|---|
| sl-err M | Error of the mass profile M(r) (mass per radial shell), in signed-log space | Small and slowly growing is good. It is in the training loss, so it should be the best of the three |
| sl-err E, L | The same for energy and angular momentum per shell | Not trained on, so an honest test; scales are larger than M's, compare across steps, not with M |
| \|divB\| pred / truth | Mean \|∇·B\| (magnetic field should be divergence-free) | pred ≈ truth is the goal. TRON: 1.4e-2 vs 1.1e-2 at step 1, 2.9e-2 vs 1.3e-2 at 50 (drifts to 2×) |
| norm-divB | \|∇·B\| relative to local \|B\| and cell size, in the disk | Same reading; TRON is 4.5× the truth at step 1 |
| flux pred / truth | How well mass changes match mass flowing in and out, cell by cell | Even the truth is not zero (dumps are snapshots Δt apart); the ratio pred/truth is the result |
| gmax pred | Max Lorentz factor in the prediction | Should stay well below 50 |

### 6. Conservation: drift and Table 5

**Final drift** (printed after `temporal_consistency.png`): total mass, energy and angular momentum on
the grid, change since t = 0, relative to the starting value. The truth drifts a little (matter flows
through the boundaries). Read the **model / truth ratio**:

```
M:  7.630e-01 / 9.518e-03      model drifts 80× more than the simulation
E:  1.147e+07 / 2.131e-01      enormous: the model's energy is not physical by step 50
```

Numbers like TRON's E drift are the most important thing to catch in a report: the per-cell errors
look small (Table 1) while a global physical quantity is badly off.

**Table 5** checks mass bookkeeping inside r < 100: the change in enclosed mass should equal inflow at
r = 100 minus outflow into the black hole. Compare the model's imbalance with the truth's (TRON 1.30
vs 0.56). Watch for single quantities wildly off: TRON's ⟨|Ṁ(100)|⟩ is 104 against 1.9 in the truth.

### 7. Tables 8 and 9: black-hole and jet quantities (the astrophysics)

- **Φ (magnetic flux through the horizon and at r = 10)**: relative error should stay small; it sets
  how magnetically arrested the disk is. TRON: 0.08% at step 1, 30% at step 50.
- **P_jet, η (jet power and efficiency), φ_BH (normalized horizon flux)**: these are what an
  astrophysicist would ask the forecast for. P_jet errors grow fastest (TRON: 5% → 184%).
- **Ṁ(t) Pearson r**: correlation over the 50 steps between model and true accretion rate. TRON 0.72 at
  the horizon but 0.16 at r = 10: it tracks the time variability in one place and not the other.

### 8. Distributions, structure and error maps (secondary)

- **Statistical properties (JS divergence)**: whether the *distribution* of values after 25 steps
  matches the truth, ignoring where they are. Near 0 is good (TRON ≤ 0.03). A forecast can have the
  right distribution with everything in the wrong place, so this complements nMAE, not replaces it.
- **Spectra (Fig 6, azimuthal power ratios, coherence, temporal PSD)**: power at each spatial or
  temporal scale, prediction / truth. A ratio of 1 is right. Below 1 at small scales (high k or m) means
  the forecast is too smooth (blurring, typical of models trained on MAE); above 1 means spurious
  small-scale noise. "High-m fraction" compares how much power sits at m > 24.
- **Structure-function exponents ζ**: how turbulent fluctuations scale with separation; pred (p) vs
  truth (t) should agree. Advanced; skip at first.
- **Bias analysis**: splits the one-step error into a systematic part (the same error every time:
  `|bias|`) and the rest. A bias fraction near 1 means a consistent offset that could be corrected; near 0
  means random scatter. Maps in `fig14_error_decomposition.png`.
- **Conditional error**: nMAE split into five bins by the true value, low to high. TRON's error is
  concentrated in the high bins (dense disk, strong field), not in the floor-dominated jet and vacuum.
- **Figures to look at, not just the tables**: `fig8_meridional_slice*.png` (prediction next to truth
  in a slice through the axis), `fig16_rollout_evolution.png` (the same over time),
  `spatial_error_*.png` (where errors live in r and θ).

### 9. Cost

Parameters and inference time per step; "forecast of 50 steps" vs the simulated time it covers.

## rc300 caveats when you read the ViT's report

- **Δt is 100 R_g/c, four times the 5000run's 25.** One rc300 step is a much bigger change, so
  persistence is weaker, per-step skill reads higher and errors grow more per step. Compare horizons in
  R_g/c or in τ_c, never in steps, and never put rc300 and 5000run numbers side by side.
- **∇·B of the rc300 truth is far from zero**: normalized ∇·B in the disk has median ~2.2 (the reduced
  run keeps every 4th cell; `rc300_preprocessing.md`, section on ∇·B). Only pred vs truth means
  anything, and our training loss penalizes exactly that difference.
- **r < 6.11 holds copied, not resolved, values** (the unrefined ring; `resolved` in `geometry.npz` is
  False there). Horizon quantities in Tables 8 and 9 (Φ(r_H), P_jet, φ_BH, Ṁ at r_H) are computed in
  that ring, so they test the model on duplicated data. The evaluator does not mask it.
- **Test set**: blind dumps 11672–12635 (964 dumps, 963 single-step pairs, n_eff = 482). The physics
  tables start at dump 11672 unless `--start` says otherwise.
- **Training-loss terms**: the ViT is trained on reconstruction plus the same physics terms as TRON
  (mass shells, ∇·B, gradients, spectrum, guards). The notes in Table 2 that say "TRON training loss"
  (printed as "ViT" by our wrapper) apply to it too.

## Glossary

| Term | Meaning |
|---|---|
| R_g, R_g/c | Gravitational radius GM/c² and the light-crossing time over it: the units of length and time |
| r_H | Event horizon radius, 1 + √(1 − a²): 1.34 R_g at rc300's spin a = 0.94 |
| ISCO | Innermost stable circular orbit; inside it, gas plunges into the hole |
| Ṁ | Mass accretion rate through a sphere of radius r |
| Φ, φ_BH | Magnetic flux through a sphere; φ_BH normalizes the horizon flux by √Ṁ (large values mean a magnetically arrested disk) |
| P_jet, η | Jet power (outgoing energy flux in the funnel) and efficiency P_jet / Ṁc² |
| σ (magnetization) | Magnetic over rest-mass energy density; σ > 1 marks the jet funnel |
| β (plasma beta) | Gas pressure over magnetic pressure |
| ũ^i | HARM's velocity primitives relative to the normal observer (always physical) |
| sl-err | Error after the signed log sl(x) = sign(x)·log(1 + \|x\|), which compares numbers across decades |
| JS divergence | Distance between two distributions, 0 = identical |
| Lipschitz constant | Max ratio of output change to input change |
