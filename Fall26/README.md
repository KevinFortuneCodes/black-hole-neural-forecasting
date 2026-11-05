# Fall26

The transformer forecaster: the current model (the repository's top-level README has the overview).

- The SFNO it replaced is in `../SFNO/`.
- TRON, the comparison model by Sachin Venkatesh, is not included. The parts of it the ViT shares
  (data loading, normalization, physics loss, evaluation suite) are in `common/` and
  `eval/evaluate_forecast_ds.py`, credited in each file.

## ViT baseline

A patch-based vision transformer that forecasts the next GRMHD frame, as a baseline for how
transformers do on this problem. Design decisions are in the design doc, "Transformer
Baseline for GRMHD Forecasting".

```text
common/          copied from TRON's uno_simple_ds_v3.py (source commit in each file)
  data.py          dump listing, 90/10 split, HAMRForecastDataset
  normalizer.py    load_geometry, DataNormalizer
  physics_loss.py  PhysicsLoss3D
  settings.py      TRON's floors and loss weights
vit/
  config.py        EXPERIMENTS registry, paths, output naming
  model.py         ViTForecaster
  train.py         training loop: DDP, bf16 blocks, warmup + cosine, checkpoint and resume
data/
  preprocess_rc300.py  raw H-AMR rc300 run -> dump_<N>.npy + geometry.npz (runs on Perlmutter)
  check_against_pp.py  cross-checks it against H-AMR's own reader on one dump
eval/
  evaluate_forecast_ds.py  TRON's evaluation suite, copied unchanged (source commit in the file)
  evaluate_vit.py      runs that suite on a ViT checkpoint, on the blind test set
  plot_training.py     training and validation curves from a run's metrics.csv
docs/            rc300_preprocessing.md (data handoff), multi_system_notes.md,
                 reading_the_evaluation.md (what each table and figure of the evaluation means)
scripts/         Delta job scripts; preprocess_rc300_perlmutter.sbatch (Perlmutter)
smoke_test.py    synthetic checks (CPU) and --real (one GPU, real dumps)
outputs/         link to /path/to/work/fall26_outputs (not in git)
```

## Experiments

Every hyperparameter lives in a named entry of `EXPERIMENTS` in `vit/config.py`. The first is
`vit_baseline_v1`:
- tokens: one per variable and 8 × 8 × 4 patch (55,296);
- model: 12 blocks, d = 256, MLP ratio 8, 16.8M parameters;
- training: one-step prediction, 5-epoch warmup, then cosine decay over 100 epochs.

Each experiment names its data (`data_dir`, `geometry_file`). `vit_baseline_v1` uses the 5000run;
`vit_rc300_v1` uses the same blocks and schedule on the rc300 reduced run (112 × 48 × 96 grid, 4 × 4 × 4
patches since rc300 cells are near-cubic; see `docs/rc300_preprocessing.md`) with `token_mode =
"per_patch"`: one token per patch holding all 8 variables, so 8,064 tokens and 64× cheaper attention
than one token per variable (`"per_variable"`, used by `vit_baseline_v1`). Its width is 512 (8 heads of 64),
matching the 512 values per token, for ~63M parameters, with `val_frac` set so validation is 10% of all dumps.

To try something new, add a new entry; never edit one that already has runs. Each run saves
its full configuration to `config.json`. Training refuses to resume a run whose experiment has
changed since.

Outputs are named after the experiment and seed:

| Path | Contents |
|---|---|
| `outputs/<experiment>/seed<N>/` | `config.json`, `train.log`, `metrics.csv` (one row per epoch), `latest.pth`, `best.pth` |
| `outputs/<experiment>/seed<N>/eval/<checkpoint>_<split>/` | evaluation of one checkpoint, e.g. `eval/best_blind/` |
| `outputs/slurm/<experiment>_s<N>_<jobid>.log` | job logs |

`best.pth` is the best validation reconstruction loss once the physics terms are fully on
(from epoch 30). `vit.config.run_dir()` and `vit.config.eval_dir()` build these paths.

## On Delta

Run everything from the repo root.

```bash
mkdir -p Fall26/outputs/slurm

# Smoke test: synthetic checks, then timing and memory on real dumps (one GPU, a few minutes)
sbatch Fall26/scripts/smoke_delta.sbatch

# Train: experiment, seed, number of chained 48-hour jobs (each resumes from the last)
Fall26/scripts/submit_chain.sh vit_baseline_v1 0 2

# Evaluate a run's best.pth on the blind test set (one GPU; results in outputs/<experiment>/seed<N>/eval/best_blind/)
sbatch --job-name=eval_vit_rc300_v1_s0 --export=ALL,EXPERIMENT=vit_rc300_v1,SEED=0 Fall26/scripts/eval_delta.sbatch
```

Locally, `python Fall26/smoke_test.py` runs the synthetic checks on CPU in about 10 seconds.
