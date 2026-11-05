# Notes for later: training on multiple systems

Not planned work. The current goal is to show the ViT can model one system (the rc300 reduced run).
These are things to think about if we later train on several simulations.

- **Gravity of the gas is not in the simulations.** The black hole's spacetime is fixed and the gas
  has no self-gravity, so gas outside a crop only matters through contact (pressure, inflow, magnetic
  connection), limited by signal speed. Real quasar disks can self-gravitate far out; models trained
  on these simulations won't learn that.
- **Spin is not a model input.** The positional encoding uses only (log r, θ, φ). Spin changes the
  spacetime (horizon, frame dragging, innermost stable orbit), so across systems the model would have
  to infer it. Add spin, or the geometry, as an input.
- **Each system has its own outer environment** (ambient density and pressure, initial disk, field).
  A model that sees only the crop can't tell environments apart; consider giving it a few descriptors.
- **Crop extent can differ between systems.** Attention handles any number of tokens. But the
  positional encoding is *not* yet comparable across grids: `sinusoidal_encoding` rescales log r and
  θ to [0, 1] across each grid and sets its frequencies from the number of patches, so the same r
  gets a different encoding under a different crop or resolution. Before mixing grids, encode in
  physical units (e.g. ln r and θ with fixed frequencies). Choose each system's `r_max` where its flow stops evolving (the per-shell change check
  in `rc300_preprocessing.md`, section 6), or use one shared radius, the largest any system needs, so
  that every sample has the same shape.
- **Grid spacing matters more than crop extent.** Patches are a fixed number of cells, so systems at
  different resolutions would give patches of different physical size. Resample all systems onto a
  common spacing in (log r, θ, φ) during preprocessing.
- **Code that assumes one grid:** the model builds its positional encoding once at startup,
  `3Dtransformer/vit/train.py` checks every dump against one geometry, and `PhysicsLoss3D` loads one geometry
  file. Variable shapes also need batching by system, or padding with attention masks.
