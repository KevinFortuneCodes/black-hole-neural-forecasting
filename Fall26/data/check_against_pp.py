# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.
"""Cross-check preprocess_rc300's block assembly against H-AMR's own reader (pp) on one dump.

Both put every block on the uniform finest grid, copying unrefined blocks' values (pp with
interpolate_var = 0, lowres = 1, no box). Fields should agree exactly and geometry to float32
precision. Run where pp is built (doc section 5):

    module load python && conda activate pp311
    python Fall26/data/check_against_pp.py --pp-dir $SCRATCH/pp_rc300 --dump 3000
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocess_rc300 as P  # noqa: E402

PP_SETTINGS = dict(lowres1=1, lowres2=1, lowres3=1, do_box=0, axisym=1, interpolate_var=0, check_files=0,
                   export_raytracing_RAZIEH=0, export_raytracing_GRTRANS=0, DISK_THICKNESS=0.03)


def compare(name, ours, theirs, unrefined):
    ours, theirs = np.asarray(ours, np.float64), np.asarray(theirs, np.float64)
    if ours.shape != theirs.shape:
        print(f"{name:7s} SHAPE MISMATCH ours {ours.shape} pp {theirs.shape}")
        return
    diff = np.abs(ours - theirs)
    rel = diff / (np.abs(theirs) + 1e-30)
    mask = np.broadcast_to(unrefined, diff.shape)
    parts = []
    for label, m in (("refined", ~mask), ("unrefined", mask)):
        if m.any():
            parts.append(f"{label}: max |diff| {diff[m].max():.3e}, max rel {rel[m].max():.3e}")
    print(f"{name:7s} " + "; ".join(parts))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pp-dir", required=True, help="folder with the built pp.py and pp_c")
    p.add_argument("--run-dir", default=P.PREPROCESS["run_dir"])
    p.add_argument("--dump", type=int, default=P.PREPROCESS["first_dump"])
    args = p.parse_args()
    run = os.path.abspath(args.run_dir)

    par = P.read_parameters(run, args.dump)
    layout = P.Layout(par, P.read_block_table(run))
    ours = layout.place(P.read_blocks(run, args.dump, par, layout.bs))           # (N1, N2, N3, 9)
    coords, metric = P.read_gdumps(run, layout.ids, layout.bs)
    ours_xyz = layout.place(coords)
    ours_met = layout.place(metric[:, :, :, None, :], axisym=True)
    axes = P.grid_axes(par, layout, [1, 1, 1])
    unrefined = layout.coarse_mask
    print(f"grid {layout.shape}, {unrefined.mean():.1%} of cells from unrefined blocks")

    sys.path.insert(0, os.path.abspath(args.pp_dir))
    import pp
    pp.set_mpi(0)
    for key, value in PP_SETTINGS.items():
        setattr(pp, key, value)
    os.chdir(run)                                  # pp reads dumps<N>/ and gdumps/ by relative path
    pp.rblock_new(args.dump)                       # before rpar_new, which fills n_ord
    pp.rpar_new(args.dump)
    pp.rgdump_griddata(run)
    pp.rdump_griddata(run, args.dump)

    print("fields (pp's uniform reader with interpolate_var = 0 returns values 2-5 unconverted):")
    for k, name in enumerate(["rho", "ug"]):
        compare(name, ours[..., k], (pp.rho, pp.ug)[k][0], unrefined)
    for i in range(4):
        compare(f"value{2 + i}", ours[..., 2 + i], pp.uu[i, 0], unrefined)
    for i in range(3):
        compare(f"B{i + 1}", ours[..., 6 + i], pp.B[i + 1, 0], unrefined)

    print("geometry (ours: gdump values copied onto the grid):")
    for k, name in ((6, "r"), (7, "h"), (8, "ph")):
        compare(name, ours_xyz[..., k], getattr(pp, name)[0], unrefined)
    compare("gdet", ours_met[..., 32], pp.gdet[0], unrefined[..., :1])
    for (m, n) in ((0, 0), (1, 1), (2, 2), (3, 3), (0, 3), (1, 3)):
        compare(f"gcov{m}{n}", ours_met[..., 4 * m + n], pp.gcov[m, n, 0], unrefined[..., :1])
    compare("gcon00", ours_met[..., 16], pp.gcon[0, 0, 0], unrefined[..., :1])

    print("grid spacing: ours", axes["dx_fine"].tolist(), " pp", [float(pp._dx1), float(pp._dx2), float(pp._dx3)])
    print("theta of first/last cell (deg): ours", np.degrees(P.to_theta(axes["fine_samples"][1][[0, -1]])).tolist(),
          " pp", np.degrees(pp.h[0, 0, [0, -1], 0]).tolist())


if __name__ == "__main__":
    main()
