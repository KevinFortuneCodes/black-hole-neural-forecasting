# Preprocessing the rc300 H-AMR run: handoff

**Purpose.** This document holds everything needed to write (and now maintain) the preprocessing script that turns the
raw H-AMR simulation `ml_data_rc300` (on NERSC Perlmutter) into training data for the models in
`Fall26/`. It is self-contained: it records what was learned by inspecting the files, which tools
read them, what the training code expects, and what is still open.

**Status (2026-10-17).**
- The raw format is fully decoded and checked against the physics (see "Raw format").
- **The preprocessed data exists** (`Fall26/data/preprocess_rc300.py`, section 5): Perlmutter job
  59960339 (2026-10-18) wrote all 9,636 dumps to `/path/to/scratch/rc300_reduced` (149 GB) in
  98 s with no failed dumps and no failed checks. The script also agrees exactly with H-AMR's own reader.
  Next: copy it to Delta (section 9).
- **Decided:** the reduced run, from dump 3000, at the finest grid with no coarsening, with a
  chronological 80/10/10 train/validation/test split. See "Decisions and script settings" below.
- **Cropping (decided 2026-10-18):** r up to ≈ 1,050 (112 radial cells), all θ and φ. That keeps 98–99%
  of the one-step density change, makes the disk (r ≤ 100) two-thirds of the grid (Sachin's 5000run:
  60%), and keeps the outflow beyond, which is 1–2% of the change, out of the loss. Section 6 has the
  alternatives and the measurements behind them.

Each fact below is marked **[verified]** (checked on the real files), **[from code]** (read from
H-AMR's own reader source, not yet exercised), or **[open]** (unknown; must be checked).

---

## 1. Goal and context

- **Long-term goal:** an ML emulator for the **inner part of the simulation** (roughly the inner third
  of the radial grid), fast enough to stand in for part of an H-AMR run. The science interest is
  **turbulence in quasar accretion disks**.
- **Near-term goal:** a proof of concept that a model can predict *some* fine-scale behaviour. It
  will be used to justify the compute cost of new full-resolution dumps at a much finer time
  cadence, which don't exist yet.
- **The dataset to prepare is `ml_data_rc300`.** It is one simulation, stored at two resolutions and
  cadences (section 2). **The reduced run was chosen** (2026-10-17). The reasoning:
  - **Reduced** (coarse, a dump every ~100): chosen for the proof of concept. Its time step is a
    small fraction of an orbit over most of the inner region, it has ~10× more samples, and it is
    cheap to store and train on. It can't resolve turbulence at the scale of the disk's thickness.
  - **Full** (fine, a dump every ~1,000): resolves the turbulence spatially. But inside r ≈ 30 the
    flow orbits one or more times between dumps, so individual structures can't be followed from
    step to step there. It is also very large (2.6 TB raw).
- **Delta's `5000run` dataset is out of scope.** It is a *different* simulation (different grid,
  radial range and cadence), and the team has ruled it out.

## Decisions and script settings

**The script is written:** `Fall26/data/preprocess_rc300.py` (section 5 has how to run it). Every
decision is a setting in its `PREPROCESS` block, which is the single source of truth; override any
with `--set KEY=VALUE` (JSON values, e.g. `--set r_max=null`, `--set theta_deg=[30,150]`). The
settings used are saved to `out_dir/preprocess_config.json`, and a rerun into the same folder with
different settings is refused unless `--overwrite` is given.

| Setting | Value | Status |
|---|---|---|
| `run_dir` | `/path/to/ml_data_rc300/reduced` | decided |
| `first_dump`, `last_dump` | 3000, last available (12635) | decided (skip the early unstable phase) |
| `lowres` | `[1, 1, 1]` (finest grid 224 × 48 × 96); k > 1 averages k cells per direction | decided |
| `fill_coarse` | `"copy"`; or `"interpolate"` (the script's own linear interpolation) for r < 6.11 | decided: copy. The duplicated ring is accepted and noted (section 6); `resolved` in `geometry.npz` marks it |
| `r_min`, `r_max` | inner boundary, 1,040 (radial cell 112, edge r = 1,048) | decided |
| `theta_deg`, `phi_deg` | all | default |
| `align_cells` | `[8, 8, 4]` (the ViT patch) | default |
| `split_fracs` | train 0.8, val 0.1, test 0.1, chronological | decided |
| `out_dir` | `$SCRATCH/rc300_reduced` (Perlmutter staging) | decided |
| `delta_dir` | `/path/to/work/rc300_reduced` (where to copy it) | decided |

With these settings the reduced run gives 9,636 dumps (3000–12635):

| Split | Dumps | Count |
|---|---|---|
| Train | 3000–10707 | 7,708 |
| Validation | 10708–11671 | 964 |
| Test | 11672–12635 | 964 |

**Fitting the splits to the training code.** `Fall26/common/data.py` takes one folder and splits it
chronologically with the experiment's `val_frac`; it does not look in subfolders. So:
- write train and validation dumps to `out_dir/`, and test dumps to `out_dir/blind/`;
- set the experiment's `val_frac` to val / (train + val) = **1/9**, not 0.1. The script prints the
  exact value to use and writes `splits.json` with each split's dump range and each dump's time.

## 2. Where the data is

**Machine: NERSC Perlmutter**, accessed through the group's NERSC project. Group-read access comes from ACLs.

| Path | What | Readable |
|---|---|---|
| `/path/to/ml_data_rc300/dumps0` … `dumps1290` | **Full** run, raw dumps | yes |
| `/path/to/ml_data_rc300/dumps` | probably grid files for the full run [open] | yes |
| `/path/to/ml_data_rc300/reduced/dumps0` … `dumps12635` | **Reduced** run, raw dumps | yes |
| `.../reduced/gdumps/` | `grid` (block table) [open: confirm present] + ~9,000 `gdump<N>` per-block geometry files | yes |
| `.../reduced/binary_files/` | ~89,900 per-dump diagnostics: `Mdot_in_D=<n>.npy`, `Mdot_out_…`, `Edot_…`, `Edot_jet_…`, `radius_…`, `theta_x_444_…` | yes |
| `.../reduced/radial_scalings_time_aver_rho_weighted_<n>.npz` | time-averaged radial profiles | yes |
| `.../reduced/post_process*.txt`, `backup/` (owner `mtie`), `global/` (owner `emilfrom`) | notes and other products | **no** |

- `/pscratch` is **scratch storage that NERSC purges**. These files date from April 2025. Copy anything
  you depend on.
- `binary_files/` isn't training input, but it is useful for **evaluation**. Comparing predicted and
  true accretion rate (`Mdot`) and jet power (`Edot_jet`) over time is a strong physical check.

**H-AMR's own reader (verified to exist and to match these files)** is in
`/path/to/h-amr-tools/`:
- `pp.py` (4,784 lines): the Python layer;
- `pp_c.pyx`: a Cython wrapper;
- `functions.c`: the C kernels that read the bytes.

Local copies of all three are on Kevin's laptop at `C:\Users\fortu\Dropbox\PC (3)\Downloads\rc300_notes\`.
They are third-party code; don't commit them to this repo.

**Training machine: NCSA Delta.** Repo clone at `~/black-hole-neural-forecasting`; outputs
at `/path/to/work/fall26_outputs`, linked as `Fall26/outputs`. Delta's `/work` is shared
with DeltaAI. The default `/work/hdd` quota is about 1 TB, shared by the whole allocation.

## 3. The two datasets [verified unless marked]

Both come from the same simulation: spin a = +0.94, radial range r = 1.1 to 10⁶, adiabatic index
from `parameters`. The reduced run is exactly 4× coarser in every direction at the base level: it keeps
**every 4th cell** of the full run rather than averaging (section 4.5).

| | Full | Reduced |
|---|---|---|
| Dumps | 1,291 (`dumps0`–`dumps1290`, contiguous) | 12,636 (`dumps0`–`dumps12635`, contiguous) |
| Time span | t = 0 to 1,237,031 | t = 0 to 1,263,531 |
| Interval between dumps | ≈ 1,000 for the first ~10 dumps, then **≈ 960**: dump 100 at t = 96,402.4, dump 600 at 573,814.8, dump 1200 at 1,149,628.6 | **100.0**: dump 1000 at t = 100,002.7, dump 6000 at 600,015.8, dump 12000 at 1,200,030.2 |
| Base layout | 8 × 2 × 4 blocks of 56 × 48 × 48 cells → **448 × 96 × 192** | 8 × 2 × 4 blocks of 14 × 12 × 12 → **112 × 24 × 48** |
| Active blocks (AMR) | 456 in `dumps100` | 456 in `dumps1000` |
| Files per dump | `new_dump0`–`new_dump7` (253 MB each) + `parameters` | `new_dump0`–`new_dump7` (4 MB each) + `parameters` |
| Raw size | ~2 GB per dump, **~2.6 TB total** | ~33 MB per dump, **~420 GB total** |

- **Time alignment: full *k* ↔ reduced 10*k* does NOT hold** [verified]. Full `dumps10` and reduced
  `dumps100` are both at t = 10,000.3, but full `dumps100` is at t = 96,402.4 while reduced `dumps1000`
  is at 100,002.7. **Pair by time** (read `t` from each `parameters`), not by dump number. The reduced
  run's interval is close to constant (≈ 100.0027), so full dump 100 should pair with reduced dump
  ≈ 964 (t ≈ 96,402.6). **[open]** Whether full-run times always land within a step of a reduced dump.
- **Two refinement levels** [verified at reduced dumps 0, 1000, 6000, 12000 and full dumps 0, 100,
  600, 1200]: levels {0, 1} in every direction, 456 active blocks at every dump checked, so the
  **finest uniform grid is base × 2**: **896 × 192 × 384 (full)** and **224 × 48 × 96 (reduced)**.
- **The refinement is static and nearly uniform** [verified, same in both runs]:
  - the set of active blocks is identical at every dump checked, so the mesh never changes;
  - **everything is at level 1 except the innermost radial ring, r < 6.11**: one base block in r × 2 in
    θ × 4 in φ = 8 blocks stay at level 0. The other 7 × 2 × 4 base blocks are each split into 8,
    giving 448 + 8 = 456.
  - On the finest grid, **r < 6.11 therefore holds no extra detail**: each level-0 cell fills 2 × 2 × 2
    output cells (copied, or interpolated with the script's `fill_coarse = "interpolate"`). That is the innermost 28 of the
    reduced run's 224 radial cells.

## 4. Raw format

### 4.1 `dumps<N>/parameters` [verified; field order from `pp.py` `rpar_new`, lines 95–203]

Little-endian binary, read in this order:

```
float64  t
int32    n_active, n_active_total, nstep
float64  Dtd, Dtl, Dtr
int32    dump_cnt, rdump_cnt
float64  dt
int32    failed
int32    bs1, bs2, bs3, nmax, nb1, nb2, nb3     # block size; nmax is a capacity (≈2.4M), not a count
float64  startx1, startx2, startx3
float64  _dx1, _dx2, _dx3
float64  tf, a (spin), gam, cour, Rin, Rout, R0, density_scale
int32    13 values (unused here)
int32    flags   # decimal digits: +1000 P_NUM, +100 TWO_T, +10 RESISTIVE, +1 RAD_M1
int32    1 value (unused)
# then, if filesize >= 66*4 + 12*n_active_total:
int32    n_active_total × (n_ord, TIMELEVEL, NODE)
# (an older variant has only (n_ord, TIMELEVEL) pairs: filesize >= 66*4 + 8*n_active_total)
```

- **flags = 0** in both runs [verified]: no radiation, resistive, two-temperature or photon-number variables.
- `n_ord[n]` is the global block ID of the n-th active block. `NODE[n]` is the parallel task (file) that wrote it.
- `TIMELEVEL` is H-AMR's local time-stepping level (1, 2, 4, …, 32). It doesn't affect reading.

### 4.2 `dumps<N>/new_dump<K>`: the field data [verified]

- **One file per parallel task K = 0…7.** File K holds the blocks with `NODE == K`, about 57 each.
- **Read them in order, `new_dump0`, `new_dump1`, …:** together they hold the active blocks in the order
  n = 0 … n_active_total − 1. Block n's ID is `n_ord[n]`. Blocks in a file = filesize / (9 · bs1·bs2·bs3 · 4).
- **The block order changes between dumps; always use that dump's own `n_ord`.** The set of active
  blocks never changes (section 4.3), but blocks are redistributed across tasks, so position n holds
  a different block in different dumps. Using one dump's `n_ord` for another silently scrambles the
  data: a check that did this showed adjacent dumps as uncorrelated (median relative change in ρ of
  ~1.5–2.0 per step). To compare or stack dumps, reorder each by block ID
  (`blocks[np.argsort(n_ord)]`). `pp` handles this when `rpar_new(dump)` is called for every dump.
- **No headers or padding:** exactly 9 float32 values per cell [verified in every file of both runs].
- **Within a block**, the value k of cell (i, j, z) is at
  `((i·bs2 + j)·bs3 + z)·9 + k`. That is: i indexes x1 (r) and is slowest; z indexes x3 (φ) and is fastest.
- **The 9 values per cell** (from `functions.c` `kernel_rdump_new`, lines 546–631; checked against the data):

| k | Quantity | Check on real data |
|---|---|---|
| 0 | ρ, rest-mass density | 0% negative |
| 1 | u_g, internal energy density | 0% negative |
| 2 | u⁰, time component of the 4-velocity (Lorentz factor γ = α·u⁰) | min exactly 1.000 |
| 3–5 | coordinate 4-velocity u¹, u², u³ (see 4.5) | signed; radial mostly negative (inflow), φ ~97% positive (rotation) |
| 6–8 | B¹, B², B³, magnetic field in code coordinates | signed |

### 4.3 Block table: `gdumps/grid` [from code: `pp.py` `rblock_new`, lines 450–565]

- `rblock_new` uses `dumps<N>/grid` if it exists, otherwise `gdumps/grid`. The raw dump folders here
  contain no `grid` file. **Both runs use a single `gdumps/grid`** [verified]:
  `ml_data_rc300/gdumps/grid` and `ml_data_rc300/reduced/gdumps/grid`, shared by every dump.
- **Layout:** int32 `nmax`, then `NV·nmax` int32, where `NV = (filesize − 1) // nmax // 4`. Reshape as
  `(NV, nmax)` in Fortran order, then transpose, so that `block[id, field]` is the entry for block ID `id`.
- **Fields used:**

| Field | Index |
|---|---|
| `AMR_ACTIVE` | 0 |
| `AMR_LEVEL` | 1 |
| `AMR_COORD1`–`3` (block position at its own level) | 3–5 |
| `AMR_NODE` | 21 |
| `AMR_TIMELEVEL` | 36 |
| `AMR_LEVEL1`–`3` (per-direction levels) | 110–112, if NV ≥ 170; otherwise copy `AMR_LEVEL` to all three |

- **The table covers every block ID** (up to `nmax`). Index it with the active IDs `n_ord` from each
  dump's `parameters`.
- **Refinement over time:** none [verified]. The set of active block IDs is identical at reduced dumps
  0, 1000, 6000, 12000 and full dumps 0, 100, 600, 1200. Only r < 6.11 is at level 0 (section 3).
  One block table, read once, serves every dump.

### 4.4 Geometry: `gdumps/gdump<blockID>` [from code: `pp.py` 685–695, `functions.c` `kernel_rgdump_*`]

- **One file per block ID.** float64; size `(9·bs1·bs2·bs3 + 49·bs1·bs2)·8` bytes for an axisymmetric
  metric (or `58·bs1·bs2·bs3·8`).
- The reduced run's 207 KB files match the axisymmetric formula, so **`axisym = 1`** (Kerr metric,
  independent of φ).
- They hold cell coordinates and the metric. `pp`'s `rgdump_new` / `rgdump_griddata` read them.
- The spacetime is fixed. The ~9,000 files exist because the adaptive mesh creates and retires blocks.

### 4.5 Coordinates and velocity convention

- **Code coordinates** [from code, `pp.py` 166–169 and 639–646]:
  - x1 = ln r, uniform, with `_dx1 = (ln Rout − ln Rin)/(bs1·nb1)` at the base level;
  - x2 ∈ [−1, 1], uniform, with θ = π(x2 + 1)/2;
  - x3 = φ ∈ [0, 2π).

  Cell centres are at `startx + (i + 0.5)·dx`. Each refinement level halves dx.
- **Two traps in the grid spacing** [verified from `parameters`; compare with the gdump coordinates]:
  - **The reduced run is the full run subsampled, not averaged** [verified on the gdump coordinates].
    It was written by pp's `downscale()`, which reads with `rgdump_new`/`rdump_new` at `lowres = 4`:
    those keep cell `i*4` of each group (`functions.c` 70, 121, 586) and `rpar_write` keeps the old
    `_dx`. So each reduced value is a point sample taken **1/8 of the way into its cell** (the first
    of 4 sub-cells), not a cell average: 0.375 cells off-centre in refined blocks, 0.75 finest cells
    in the unrefined ring. Fine structure is aliased rather than smoothed. Metric and fields are
    sampled at the same points, so they stay consistent. The script places coordinates at
    `startx + (i + 0.5/D)·dx` (D = 4 here, 1 for the full run) and checks this against every gdump.
  - **The reduced run's `parameters` carry the full run's `_dx1.._dx3`,** 4× too small for its cells.
    With them the reduced domain comes out as r 1.1–34, θ 1.9–45.9°, φ 0–90°, a quarter of the full
    run's in each direction. Multiplying by 4 gives exactly the full run's domain.
  - **θ excludes a polar cap of 1.875° at each pole.** In both runs `startx2` = −1 + 1/48 ≈ −0.979 and
    the file's spacing spans θ = 1.875° to 178.125°. `pp.rpar_new` (`pp.py` 167–169) ignores the file
    and sets `_dx2 = 2/(bs2·nb2)`, which assumes the full [−1, 1]. That is about 2% too wide; summed over the
    grid it puts the last θ edge at 181.9° instead of 178.1°. (`_dx1` from Rin and Rout and `_dx3 = 2π/(bs3·nb3)` agree
    with the file.)
  - **So:** take coordinates from the gdump files, and set the output `dx` from the true code-coordinate
    spacing, i.e. the file's `_dx` × (Rin/Rout check factor: 4 for reduced, 1 for full) ÷ 2 for
    level 1. Check it against neighbouring cell differences of the gdump x1, x2, x3. Don't trust pp's
    `_dx2` or its `do_box` θ indices without that check.
- **∇·B of the stored data is not zero by the training/eval stencil** [verified]. The simulation keeps
  ∇·B = 0 for face-centred B with its own stencil (constrained transport); the dumps hold cell-centred B,
  and `PhysicsLoss3D.calc_divB` takes central differences of it. Normalized ∇·B (|∇·B| Δx / |B|) in the
  active disk: full run median 0.19 (mean 0.61); reduced run median 2.2 (mean 6.0), because keeping every
  4th cell leaves B poorly resolved. So ∇·B must be compared with the truth's, never with zero:
  `Fall26/common/physics_loss.py` now penalizes |∇·B(pred) − ∇·B(truth)|, and the eval should report the
  same error (to do when the eval is ported).
- **Velocity: values 2–5 are the coordinate 4-velocity (u⁰, u¹, u², u³)** [verified on both runs].
  - Test: with u = (value 2, values 3–5), g_μν u^μ u^ν = −1 holds to ~1e-7 in every cell (99th
    percentile 3e-7). Read as ũ instead, the identity √(1 + γ_ij ũ^i ũ^j) = α u⁰ fails in ~14% of cells
    (99th percentile 0.12), all in the active region r ≲ 150.
  - **Trap:** far from the hole the shift β is tiny, so both readings fit there and a *median* test
    wrongly says "ũ". The script decides on the 99th percentile.
  - Our training channels are ũ, so the script converts: ũ^i = u^i + u⁰β^i with β^i = −g^{0i}/g^{00}
    (HARM's convention; the inverse of `DataNormalizer.ucon_of`), then re-checks the identity on every
    dump it writes (99th percentile ~6e-8 on dumps 3000, 3001, 12635).
  - `pp`'s uniform reader does the same conversion when `interpolate_var = 1`:
    `uu[i] = γβ^i/α + value[2+i]` = u^i + u⁰β^i with β^i = α²g^{0i} (`functions.c` 753–756, 795–798).

## 5. Running the preprocessing, and `pp`

**The script (`Fall26/data/preprocess_rc300.py`) reads the raw files directly with numpy** (sections
4.1–4.4), so it needs only `module load python`, not `pp`. Per dump it reads the 8 `new_dump` files,
sorts blocks by ID, checks positivity and the velocity identity, places every block on the finest
grid (copying or interpolating unrefined blocks), coarsens if `lowres > 1`, crops, and writes float32.
Geometry is built once from the gdump files, after checking that their coordinates match the uniform
grid and the r, θ, φ mapping. Reruns skip dumps already written. It was tested on a synthetic H-AMR run
in the same format (AMR blocks, shuffled block order, the reduced run's wrong `_dx`), including
loading the output with the training code.

**1. Get the repo on Perlmutter** (once):

```bash
cd $HOME && git clone https://github.com/KevinFortuneCodes/black-hole-neural-forecasting.git
cd black-hole-neural-forecasting && mkdir -p Fall26/outputs/slurm
```

**2. Test on three dumps on a login node** (one train, one adjacent, one test dump):

```bash
module load python
python Fall26/data/preprocess_rc300.py --set out_dir=$SCRATCH/rc300_test --dumps 3000 3001 12635 --workers 3
cat $SCRATCH/rc300_test/checks.json
```

It prints the output grid and crop, the velocity convention it detected, per-channel ranges, checks,
and the `val_frac` to use; it exits with status 1 if any dump fails a check.

**3. Cross-check against `pp`** on one dump (needs the `pp` build below): fields should match exactly,
geometry to float32 precision.

```bash
conda activate pp311
python Fall26/data/check_against_pp.py --pp-dir $SCRATCH/pp_rc300 --dump 3000
```

**4. Full run** on one CPU node (about 570 GB of reading and writing; reruns skip finished dumps):

```bash
sbatch Fall26/scripts/preprocess_rc300_perlmutter.sbatch
```

Then copy `$SCRATCH/rc300_reduced` to `delta_dir` with Globus (section 9) and point the training
experiment's `data_dir` and `geometry_file` at it, with `val_frac` as printed.

### Building `pp` (only for the cross-check)

**Build** it on Perlmutter, in your own copy:

```bash
mkdir -p $SCRATCH/pp_rc300 && cd $SCRATCH/pp_rc300
cp /path/to/h-amr-tools/pp.py /path/to/h-amr-tools/pp_c.pyx /path/to/h-amr-tools/functions.c .
module load python                      # Perlmutter's default Python is 3.13: no distutils
conda create -y -n pp311 python=3.11 "numpy<2" "cython<3" scipy sympy matplotlib
conda activate pp311
sed -i '38s/^/# /' pp.py                # see gotchas: matplotlib rejects this line
cat > build_pp_c.py <<'EOF'
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize
setup(ext_modules=cythonize([Extension(
    "pp_c", ["pp_c.pyx", "functions.c"],
    include_dirs=[np.get_include()],
    # functions.c calls access() with its unistd.h include commented out; gcc 14 rejects that
    extra_compile_args=["-O2", "-fopenmp", "-include", "unistd.h", "-fpermissive"],
    extra_link_args=["-fopenmp"])],
    compiler_directives={"language_level": 2}))
EOF
python build_pp_c.py build_ext --inplace
python -c "import pp, pp_c; print('pp ok')"   # verified on Perlmutter, 2026-10-17
```

**Known gotchas:**
- **`pp.py` imports `distutils`** (lines 14, 17, 18, 35), which Python 3.12 removed. Perlmutter's
  `module load python` gives 3.13 (verified: `ModuleNotFoundError: No module named 'distutils'`), hence
  the 3.11 environment. It also imports Cython, sympy, matplotlib and scipy at module level.
- **Don't build with `python pp.py build_ext --inplace`.** `pp.py` only calls `setup()` inside
  `set_mpi()`. Under `__main__` that runs as `set_mpi(1)`, which needs `mpi4py`, and the script then
  calls `post_process` on a hard-coded Windows path. Use the separate `build_pp_c.py` above, which
  compiles the same sources with the same flags, plus two fixes.
- **`functions.c` doesn't compile as-is with Perlmutter's gcc** [verified]. It calls `access()`, but
  its `#include <unistd.h>` (line 10) is commented out. gcc 14 treats the implicit declaration as an
  error, so the build forces that header in with `-include unistd.h` and adds `-fpermissive`.
- **`pp.py` line 38 fails on import with current matplotlib** [verified]. It sets
  `text.latex.preamble` to a list, which newer matplotlib rejects. It only matters for LaTeX text
  rendering, which line 42 turns off, so the local copy comments it out.
- **Line 9 appends an unrelated `/gpfs/alpine/...` path.** It's harmless.
- **`pp` keeps everything in module globals and reads files by relative path** (`dumps%d/parameters`,
  `gdumps/grid`). `os.chdir(run_dir)` first and pass `dir=run_dir`. Calls are stateful, so don't run two
  reads in one process concurrently.
- **Sachin's existing pipeline doesn't apply as-is.** `fall26/preprocess_forecast_ds.py` on the `sachin`
  branch was written for a single-grid run:
  - it calls `pp.rgdump_direct()`, which **does not exist in the `JET1` copy**;
  - it refuses multi-block (AMR) runs;
  - it treats `pp.uu` as coordinate velocities and adds u⁰β^i to get ũ. That is right with
    `interpolate_var = 0` (raw values are coordinate u^i), but with `= 1` pp has already converted
    them and the shift would be added twice.

  Reuse his overall structure and output format (MPI or multiprocessing over dumps, a per-dump `.npy`,
  `geometry.npz`, `channel_stats.npz`), but not these calls. His settings line also carries the comment
  `lowres = 1 # change to 4 for Aris data`.

**Relevant `pp` entry points:**

| Call | What it does |
|---|---|
| `pp.set_mpi(0)` | Disable MPI inside `pp` |
| `pp.rpar_new(dump)` | Read `parameters`: sets `n_active_total`, `n_ord`, `bs*`, `nb*`, `_dx*`, `a`, `gam`, flags |
| `pp.rblock_new(dump)` | Read the block table into `pp.block` |
| `pp.rgdump_griddata(dir)` | Geometry on a **uniform** grid: `pp.r`, `h`, `ph`, `gcov`, `gcon`, `gdet`, `dxdxp`; sets `gridsizex1..3` and `_dx*` |
| `pp.rdump_griddata(dir, dump)` | Fields on that uniform grid: `pp.rho`, `ug`, `uu`, `B[1..3]`. With `interpolate_var = 0`, `uu[0..3]` are the raw values 2–5; with `1`, `uu[1..3]` are converted (`functions.c` 794–802) |
| `pp.rgdump_new(dir)` / `pp.rdump_new(dir, dump)` | **Per-block** raw arrays `(nb, bs1/lowres1, …)`, with no AMR assembly and values unconverted (`uu[0..3]` = values 2–5) |
| `pp_c.griddata3D(...)` (`pp.py` 885) | Places any per-block array onto the uniform grid, using the block levels and coordinates |

Module settings to set before reading:

```python
pp.lowres1 = pp.lowres2 = pp.lowres3 = 1   # coarsening factors (section 6)
pp.do_box = 0                              # cropping (section 6)
pp.axisym = 1
pp.interpolate_var = 0                     # in rdump_griddata: 0 = raw velocities, 1 = converted ones
pp.check_files = 0; pp.export_raytracing_RAZIEH = 0
pp.RAD_M1 = pp.RESISTIVE = pp.TWO_T = pp.P_NUM = 0
pp.DISK_THICKNESS = 0.03                   # only used for an unrelated heating diagnostic (Rdot)
```

**Two workable approaches:**
1. **Use `rgdump_griddata` + `rdump_griddata` directly**, then recover ũ^i from the converted `uu[1..3]`
   (needs γ, which `rdump_griddata` doesn't return).
2. **Read per block with `rdump_new`,** which keeps values 2–5 unconverted, then place each channel onto
   the uniform grid with `pp_c.griddata3D`. This keeps the stored ũ^i exactly and avoids the conversion
   question.

Either way, validate with section 7. (The script takes approach 2's route without `pp`.)

**Cross-check result** (dump 3000, reduced) [verified]: all 9 stored values agree **exactly** with pp's
uniform reader in refined and unrefined cells; r, θ, φ, gdet and the metric agree to float32 precision
(relative ~5e-8). pp's `_dx2` is 0.041667 against the true 0.040799 (section 4.5).

## 6. Options: resolution and cropping (decided: see "Decisions and script settings")

`pp` supports both natively.

**Coarsening (`lowres1..3`).** Per direction, the uniform grid size is
`gridsize_d = nb_d · 2^(max level) · bs_d / lowres_d` (`pp.py` 631–633). With `lowres > 1`,
`rdump_griddata` **volume-averages** ρ, u_g, B and the converted velocity over each coarsening volume
(`functions.c`), which is the physically sensible way to coarsen. Each factor must divide the block
size at the finest level.

**Cropping (`do_box = 1`).** Set `pp.r_min`, `pp.r_max`, `pp.theta_min`, `pp.theta_max`, `pp.phi_min`
and `pp.phi_max` (θ and φ in radians). `rgdump_griddata` turns these into index ranges on the uniform
grid (`pp.py` 639–650), and both readers return only that box. The geometry must be cropped
identically; `rgdump_griddata` does this when called with the same settings.

**Considerations for choosing:**
- **The inner third of the radial grid.** The grid is log-spaced, about 75 cells per decade at the full
  run's base level, so the inner third of the radial cells is roughly r ≲ 115.
- **Which radii evolve** [verified at dump 3000, reduced run]. The intent is to keep **every part that
  evolves**. Relative change in ρ per radial shell, with blocks matched by ID:

| r | ρ change per step (median / top 1%) | ρ change, 3000 → 12635 (median) |
|---|---|---|
| 1.1–148 | 4–21% / 150–196% | 35–57% |
| 148–1,050 | 7%–0.8% / 155%–85% | 20–30% |
| 1,050–7,450 | 0.4%–0.05% / 42%–7% | 7–22% |
| 7,450–12,200 | 0.007% / 2% | 8% |
| 12,200–52,900 | ~0.0004% / ~0.002% | 3–6%, a steady drift (per-step change × 9,635 steps) |
| > 52,900 | ~0.0001% / ~0.001% | 0.1–0.2% |

  - **The active flow ends sharply at r ≈ 12,200** (the edge of radial cell 152): the top-1% change per
    step drops by three orders of magnitude across that edge.
  - **Beyond it, 12,200–52,900 drifts smoothly** by a few percent over the run, about 10⁻⁶ per step
    (the atmosphere slowly falling in; the free-fall time there is comparable to the run). Beyond 52,900
    nothing measurably changes.
  - **By the end of the run the active edge has moved out** [verified, dumps 12534 → 12535 → 12634 →
    12635]. Top-1% change per step: 0.9% at 12,200–19,800 and 0.3% at 19,800–32,400 (both ~0.002% at
    dump 3000); 0.01% at 32,400–52,900 and still rising over 100 steps; ~0.0005% beyond 52,900. An
    outflow front travels from r ≈ 12,000 to ≈ 30,000–50,000 over the training range.
  - **So `r_max` ≈ 52,900** (the edge of radial cell 176) would keep everything that evolves at any time
    in the data. The region 12,200–52,900 is quiet in early training dumps but active in the validation
    and test dumps (the last 20%), so expect the model to do relatively worse there.
- **Size of a processed dump** = 8 channels × N_r × N_θ × N_φ × 4 bytes. At the base level:

| Grid | Per dump | Whole run |
|---|---|---|
| 896 × 192 × 384 (full, finest, `lowres = 1`) | 2.1 GB | 2.7 TB |
| 448 × 96 × 192 (full, base, `lowres = 2`) | 264 MB | 340 GB |
| 224 × 48 × 96 (reduced, finest, `lowres = 1`) | 33 MB | 420 GB |
| 112 × 24 × 48 (reduced, base, `lowres = 2`) | 4.1 MB | 52 GB |

  Cropping reduces these in proportion to the cells kept.
- **Model compatibility.** The ViT tokenizes 3D patches (`patch` in `Fall26/vit/config.py`), so each
  grid dimension should divide by the patch size; otherwise the model pads. Attention cost grows with
  the square of the number of patches.
- **Crop options for the chosen setup** (reduced run, finest grid 224 × 48 × 96, dumps 3000–12635).
  Radial cell n has its outer edge at r ≈ 1.1·e^(0.06125·n), assuming x1 = ln r is uniform from 1.1
  to 10⁶. Sizes are for the full θ and φ ranges:

| `r_max` | Radial cells | Per dump | 9,636 dumps |
|---|---|---|---|
| ≈ 91 | 72 | 10.6 MB | 102 GB |
| ≈ 148 | 80 | 11.8 MB | 114 GB |
| **≈ 1,050 (chosen)** | 112 | 16.5 MB | 159 GB |
| ≈ 12,150 (active flow at dump 3000 only) | 152 | 22.4 MB | 216 GB |
| ≈ 52,900 (active flow through the end of the run) | 176 | 26.0 MB | 250 GB |
| none (10⁶) | 224 | 33.0 MB | 318 GB |

  - **θ:** keeping all of it (48 cells) is the default. Cropping to 30°–150° (32 cells) removes the
    polar jet funnels, where density is usually held at numerical floors rather than evolved
    physically. It also turns the θ edges into boundaries the model can't see across.
  - **φ:** keep all of it (96 cells). φ is periodic, and the ViT's positional encoding uses whole
    harmonics of φ, which assume the full 2π.
  - **r_min:** keep the inner boundary (r = 1.1, just inside the horizon at r ≈ 1.34). Matter and
    magnetic flux crossing the horizon matter physically. The cost: r < 6.11 is unrefined (level 0),
    so its 28 cells on the finest grid are duplicated or interpolated, not real detail. Setting
    `r_min ≈ 6.11` would keep only true level-1 data, but drops the horizon and the innermost stable
    orbit (r ≈ 2.0 at spin 0.94).
  - **How to fill r < 6.11 (`fill_coarse`):** `"copy"` repeats each coarse value over 2 × 2 × 2 cells
    (as pp does) and is honest about the resolution, but blocky. `"interpolate"` is the script's own
    linear interpolation from the coarser level, smooth but inventing detail. Either way, discrete
    gradients and ∇·B there are not exact; `checks.json` reports ∇·B separately for those cells.
- **Why not r ≈ 52,900: what that crop does to the training loss** [verified on dumps 3000/3001 and 12634/12635;
  figures in `Fall26/outputs/figures/`, not in git]. Inside r ≈ 150 are 45% of the cells but 87–95% of
  the one-step density change; beyond r ≈ 1,050 are 36% of the cells and 1–2% of the change. That
  matters differently per loss term:
  - **Cell-averaged terms** (reconstruction, gradient): quiet cells add almost nothing once "no change"
    is learned, so they mostly cost compute and model capacity rather than biasing the fit.
  - **Spectral loss:** its radial Hann window is centred on the middle radial cell, so with 176 cells it
    peaks at r ≈ 240 and gives r < 6 (horizon, ISCO) under a quarter of the weight. The crop decides
    which radii this term watches.
  - **Mass-shell loss:** its per-shell gate is M(r)/max M, and M(r) grows roughly like r³ρ (gdet is
    1.6e3 at r = 10 and 2e14 at r = 5e4). At this crop the gate is 6e-8 at r = 100 and half of its weight
    lies beyond r ≈ 42,000, so this term effectively ignores the disk. At a crop of r ≈ 150 the same
    gate puts half its weight beyond r ≈ 100, and at the chosen r ≈ 1,050 beyond r ≈ 780. **Fixed
    (2026-10-18):** `Fall26/common/physics_loss.py` now weights every shell equally.
- **Time pairing.** If both resolutions are processed, keep native dump numbers and record each dump's
  `t`, and pair by time (section 3): full *k* ↔ reduced 10*k* only holds for the first ~10 dumps.

## 7. What the training code expects (output contract)

Read by `Fall26/common/data.py`, `Fall26/common/normalizer.py`, `Fall26/common/physics_loss.py` and
`Fall26/vit/train.py`.

**One folder per dataset** (point `data_dir` in a `Fall26/vit/config.py` experiment at it) containing:

**`dump_<N>.npy`**, one per dump:
- `<N>` is the native H-AMR dump number. Files are sorted numerically, and the **chronological** 90/10
  train/validation split is taken over all of them, so keep a separate test set out of this folder
  (e.g. a `blind/` subfolder for the last part of the run);
- float32, shape **(8, N_r, N_θ, N_φ)**;
- channels in this order: **ρ, u_g, ũ¹, ũ², ũ³, B¹, B², B³**, with ũ^i the relative 4-velocity
  (HARM primitive) and B^i in code coordinates;
- the loader accepts 8 channels, or 10 (keeping the first 8).

**`geometry.npz`**, matching the cropped and coarsened grid exactly:

| Key | Shape | Used by |
|---|---|---|
| `r`, `h`, `ph` | (N_r, N_θ, N_φ) | ViT positional encoding (`h` = θ, `ph` = φ); the grid check against the dumps. Each output cell's sample position, `startx + (i + sample_frac)·dx` (not the cell centre for the reduced run) |
| `gcov`, `gcon` | (4, 4, N_r, N_θ, 1) for axisym, or (…, N_φ) | normalizer (Lorentz factor, ũ↔u), physics loss |
| `gdet` | (N_r, N_θ, 1) or (…, N_φ) | physics loss (divB, mass shells) |
| `dx` | (3,) | physics loss: central differences in **code coordinates** on the output grid (finest-level spacing × `lowres`; for the reduced run 0.06125, 0.04080, 0.06545; not pp's `_dx2`) |
| `eos_gamma`, `dt_dump` | scalars | optional metadata (dt_dump ≈ 100 reduced, ≈ 1,000 full) |
| `spin`, `rin`, `rout`, `t_start`, `density_scale`, … | scalars | recommended metadata (Sachin's script saves these) |
| `resolved` | (N_r, N_θ, N_φ) bool | written by the script: False where values come from an unrefined block (r < 6.11), i.e. copied or interpolated rather than resolved. Use it to mask or separate those cells in evaluation |
| `sample_frac`, `format_version`, `crop_index`, `finest_grid`, `lowres`, `dt_dump_exact` | scalars / small arrays | written by the script: where values sit in their cells (0.125 for the reduced run), the output format, and how the grid was cut |

**Coordinates and metric in the unrefined ring (r < 6.11).** With `fill_coarse = "copy"`, each value
there and the metric are copies of one coarse sample, while `r`, `h`, `ph` give the nominal fine-grid
position of each output cell. They differ by up to 7/8 of a cell inside that ring. Nothing in training
combines the two (the ViT uses only coordinates; the normalizer and physics loss only the metric), and
`resolved` marks the cells.

**`channel_stats.npz`** (optional): `channel_names`, `min`, `max`, `mean`, `std`, `count`, `n_dumps`, over
the **training dumps only**, used for a warning-only check of the normalizer's floors (ρ ≥ 1e-20,
u_g ≥ 1e-22, ceilings 1e5). A run limited with `--dumps` writes `channel_stats_partial.npz` instead.

## 8. Validation the script must pass

Run these on the first few dumps before processing everything:

1. **Positivity:** ρ > 0, u_g > 0, and u⁰ ≥ 1 everywhere (these already hold in the raw files).
2. **Velocity convention:** √(1 + γ_ij ũ^i ũ^j) = α·u⁰ to float32 precision (section 4.5).
3. **Divergence-free field:** |∇·B| on the output grid near zero relative to |B|/Δx, computed as in
   `PhysicsLoss3D.calc_divB`. A large value means B, `gdet` or `dx` don't match.
4. **AMR assembly:** no holes or blocks left at zero on the uniform grid. Each cell is filled exactly
   once, or averaged when `lowres > 1`.
5. **Cross-resolution check (if both runs are processed):** a full dump subsampled `[::4, ::4, ::4]` (not averaged) should
   match the reduced dump at the same `t` (section 3; not reduced 10*k* in general).
6. **Grid consistency:** every dump's array shape equals `geometry.npz`'s `r.shape`, and `Fall26/vit/train.py`'s
   startup assertion passes.

Raw-file self-check (verified on both runs; it reproduces the table in 4.2):

```python
import numpy as np
x = np.fromfile(".../reduced/dumps1000/new_dump3", np.float32).reshape(-1, 9)
for k, n in enumerate(["rho", "ug", "u0", "v1", "v2", "v3", "B1", "B2", "B3"]):
    print(n, (x[:, k] < 0).mean(), x[:, k].min(), x[:, k].max())
```

Refinement levels and the finest uniform grid (**run on both runs; results in section 3**):

```python
import os, numpy as np
R, d = "/path/to/ml_data_rc300/reduced", 1000
gfile = next(p for p in (f"{R}/gdumps/grid", f"{R}/dumps/grid", f"{R}/dumps{d}/grid") if os.path.isfile(p))
size = os.path.getsize(gfile)
with open(gfile, "rb") as f:
    nmax = int(np.fromfile(f, np.int32, 1)[0]); NV = (size - 1) // nmax // 4
    block = np.fromfile(f, np.int32, NV * nmax).reshape((NV, nmax), order="F").T
with open(f"{R}/dumps{d}/parameters", "rb") as f:
    f64 = lambda n=1: np.fromfile(f, np.float64, n); i32 = lambda n=1: np.fromfile(f, np.int32, n)
    f64(); _, nat, _ = i32(3); f64(3); i32(2); f64(); i32()
    bs1, bs2, bs3, nmx, nb1, nb2, nb3 = i32(7); f64(6); f64(8); i32(13); i32(); i32()
    n_ord = np.fromfile(f, np.int32, 3 * nat).reshape(-1, 3)[:, 0]
lev = block[n_ord, 110:113] if NV >= 170 else np.repeat(block[n_ord, 1:2], 3, axis=1)
L = lev.max(axis=0)
print("levels per axis:", [dict(zip(*np.unique(lev[:, k], return_counts=True))) for k in range(3)])
print("finest uniform grid:", nb1 * 2**L[0] * bs1, nb2 * 2**L[1] * bs2, nb3 * 2**L[2] * bs3)
```

Where the refinement sits, and whether it moves between dumps (**run on both runs; results in sections
3, 4.3 and 4.5**). It uses `_dx` from `parameters`, so the reduced run's edges and ranges print 4× too
small (section 4.5):

```python
import os, numpy as np
def grid(R):
    g = f"{R}/gdumps/grid"; size = os.path.getsize(g)
    with open(g, "rb") as f:
        nmax = int(np.fromfile(f, np.int32, 1)[0]); NV = (size - 1) // nmax // 4
        return np.fromfile(f, np.int32, NV * nmax).reshape((NV, nmax), order="F").T
def par(R, d):
    with open(f"{R}/dumps{d}/parameters", "rb") as f:
        f64 = lambda n=1: np.fromfile(f, np.float64, n); i32 = lambda n=1: np.fromfile(f, np.int32, n)
        t = f64()[0]; _, nat, _ = i32(3); f64(3); i32(2); f64(); i32()
        b = i32(7); sx = f64(3); dx = f64(3); p = f64(8); i32(13); i32(); i32()
        n_ord = np.fromfile(f, np.int32, 3 * nat).reshape(-1, 3)[:, 0]
    return n_ord, b[:3], b[4:7], sx, dx, p[4], p[5]
def merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out
B = "/path/to/ml_data_rc300"
for R, ds in ((f"{B}/reduced", (0, 1000, 6000, 12000)), (B, (0, 100, 600, 1200))):
    block = grid(R); sets = []
    for d in ds:
        n_ord, bs, nb, sx, dx, Rin, Rout = par(R, d)
        sets.append(set(n_ord.tolist()))
        if d != ds[0]: continue
        lev = block[n_ord, 110:113]; crd = block[n_ord, 3:6]
        lo = crd * bs / 2.0**lev; hi = (crd + 1) * bs / 2.0**lev   # in base-level cells
        N = nb * bs
        conv = (lambda i: np.exp(sx[0] + i * dx[0]),               # r
                lambda j: 90 * (sx[1] + j * dx[1] + 1),             # theta, degrees
                lambda k: np.degrees(sx[2] + k * dx[2]))            # phi, degrees
        print(R.split("/")[-1], "edges: r %.3g-%.3g (Rin %.3g, Rout %.3g), theta %.1f-%.1f, phi %.1f-%.1f"
              % (conv[0](0), conv[0](N[0]), Rin, Rout, conv[1](0), conv[1](N[1]), conv[2](0), conv[2](N[2])))
        combos, counts = np.unique(lev, axis=0, return_counts=True)
        print("  blocks per (r, theta, phi) level:", dict(zip(map(tuple, combos.tolist()), counts.tolist())))
        for k, name in enumerate(("r", "theta", "phi")):
            m = lev[:, k] == 1
            print(f"  refined in {name}:", ", ".join("%.4g-%.4g" % (conv[k](a), conv[k](b))
                                                     for a, b in merge(zip(lo[m, k], hi[m, k]))))
    print("  active blocks differing from the first dump checked:", [len(s ^ sets[0]) for s in sets])
```

## 9. Where to run it, and storage

- **Preprocess on Perlmutter,** where the raw data lives. The full run's raw data (2.6 TB) is too large
  to move. The NERSC project has CPU time; the C reader uses OpenMP.
- **Transfer only the processed, cropped output to Delta.** Use Globus (NERSC and NCSA both have
  endpoints) for anything over a few GB.
- **Delta location:** `/path/to/fast_storage/rc300_reduced` (copied there 2026-10-19 and verified:
  same 9,641 files and sizes, checksum sample identical). Use `/work/nvme`, not `/work/hdd`: some GPU nodes
  (gpua016) returned I/O errors on 14 of 150 rc300 files on `/work/hdd` that read fine elsewhere, and
  `/work/nvme` loads a training batch in about 0.1 s against 3.4 s. The `/work/hdd` copy is now redundant.
- Keep the scripts in `Fall26/` (for example `Fall26/data/preprocess_rc300.py` plus a Perlmutter job
  script), following the existing style in `Fall26/common/` and `Fall26/scripts/`.

## 10. Open questions

1. The θ grid (polar cap at 1.875°) and the true dx: the script checks both against the gdump
   coordinates and stops if they disagree (`checks.json`, `geometry`).
2. ~~The velocity convention~~: answered, coordinate u^i (section 4.5); the script converts to ũ.
3. How exactly full-run times line up with reduced dumps (section 3). Full *k* ↔ reduced 10*k* is
   known to fail after the first ~10 dumps.
4. The disk's thickness (H/r) near the target radii, which decides how many cells resolve turbulent
   eddies at each resolution. Measure it from ρ once the data is readable.
5. ~~The full run~~: done (job 59960339, every check passes). Remaining: the Globus copy to Delta.

## 11. References

- `Fall26/common/` (data, normalizer, physics loss) and `Fall26/vit/` (model, training, config):
  consumers of the output.
- `sachin` branch, `fall26/preprocess_forecast_ds.py`: an earlier preprocessing script for a single-grid
  run. A useful template for the output format and multiprocessing; see section 5 for what not to reuse.
- `/path/to/h-amr-tools/{pp.py, pp_c.pyx, functions.c}`, with line references above:
  - `rpar_new` 95–203, `rblock_new` 450–565, `rgdump_griddata` 613–695, `rdump_new` 707–756,
    `rdump_griddata` 758–823, build 3259–3274 (all `pp.py`);
  - `kernel_rdump_new` 546–631, `kernel_rdump_griddata` from 647 (`functions.c`).
- H-AMR method paper: https://arxiv.org/abs/1912.10192. A possibly related run setup (quasar disks from
  cosmological initial conditions; unconfirmed link to this run): https://arxiv.org/abs/2410.01877
