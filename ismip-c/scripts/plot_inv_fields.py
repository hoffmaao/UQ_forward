#!/usr/bin/env python3
import argparse, pathlib, numpy as np
import firedrake as fd
import matplotlib.pyplot as plt
import icepack.plot

def to_regular_grid(fun, nx=200, ny=200):
    """Sample a Firedrake Function on a regular (x,y) grid for imshow."""
    mesh = fun.function_space().mesh()
    xy = mesh.coordinates.dat.data_ro
    xmin, ymin = xy.min(axis=0); xmax, ymax = xy.max(axis=0)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    vals = np.array([fun.at(p) for p in pts])
    if fun.ufl_element().value_size() == 1:
        Z = vals.reshape(ny, nx)
        return xs, ys, Z
    else:
        Z = vals.reshape(ny, nx, -1)
        return xs, ys, Z



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Checkpoint *.h5 chosen from the L-curve")
    ap.add_argument("--out", default="plots", help="Output directory")
    ap.add_argument("--grid", type=int, default=200, help="Grid for imshow sampling")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    with fd.CheckpointFile(args.ckpt, "r") as chk:
        mesh  = chk.load_mesh(name="ismip-c")
        u     = chk.load_function(mesh, name="velocity")
        C     = chk.load_function(mesh, name="friction")       # friction coefficient
        theta = chk.load_function(mesh, name="log_friction")   # control field (log C/beta)

    fig, axes = icepack.plot.subplots()
    colors = firedrake.tripcolor(u, axes=axes)
    fig.colorbar(colors, ax=axes, fraction=0.012, pad=0.04);
    fig.savefig("velocity_optimized.png")

    fig, axes = icepack.plot.subplots()
    colors = firedrake.tripcolor(C, axes=axes)
    fig.colorbar(colors, ax=axes, fraction=0.012, pad=0.04);
    fig.savefig("friction_optimized.png")

if __name__ == "__main__":
    main()
