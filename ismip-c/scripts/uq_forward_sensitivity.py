import argparse, pathlib
import firedrake as fd
import numpy as np
import csv

def l2_inner(mesh, f, g):
    return float(fd.assemble(fd.inner(f, g) * fd.dx(mesh)))

def analytic_mode_contributions(mesh, gQ, lambdas, phis):
    """Compute contributions of eigenmodes to QoI variance at one timestep."""
    contribs = []
    for lam, phi in zip(lambdas, phis):
        a = l2_inner(mesh, gQ, phi)
        contribs.append((a*a)/lam)
    return contribs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_file", default="ismip-c.h5",
                    help="orignal mesh checkpoint file with mesh, inversion, forward, eigendec results")
    ap.add_argument("--eigvals_file", default="outputs/eigenvalues.txt",
                    help="Text file with eigenvalues (one per line)")
    ap.add_argument("--timesteps", type=int, nargs="+", required=True,
                    help="List of forward timesteps (indices) to process")
    ap.add_argument("--outdir", default="outputs/sensitivity",
                    help="Directory to write results")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load eigenvalues
    eigvals = []
    times=[]
    Qval = []
    sigmaQ = []

    with open(args.eigvals_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            eigvals.append(float(parts[1]))
    eigvals = np.array(eigvals)
    print(eigvals)

    with fd.CheckpointFile(args.mesh_file, "r") as chk:
        mesh = chk.load_mesh("ismip-c")

        # Load eigenmodes phi_####
        phis = []
        i = 0
        for i in range(0,40):
            name = f"phi_{i:04d}"
            phis.append(chk.load_function(mesh, name))
            print(i, fd.norm(phis[i]))
            i += 1
        print(f"Loaded {len(phis)} eigenmodes.")


        # For each requested timestep, load dQ/dtheta and compute contributions
        for k in args.timesteps:
            gQ = chk.load_function(mesh, "dQ_dtheta_timeseries", idx=k)
            h = chk.load_function(mesh, "thickness_timeseries", idx=k)
            print("gQ min/max:", gQ.dat.data_ro.min(), gQ.dat.data_ro.max())
            print("gQ norm:", fd.norm(gQ))
            contribs = analytic_mode_contributions(mesh, gQ, eigvals, phis)
            total_var = sum(contribs)

            print(f"Timestep {k}: variance contributions = {total_var:.6e}")

            # Save contributions to file
            with open(outdir / f"contribs_t{k:04d}.txt", "w") as f:
                for j,(lam,c) in enumerate(zip(eigvals, contribs)):
                    f.write(f"{j} {lam:.6e} {c:.6e}\n")
            times.append(k*30./120.)
            Qval.append(float(fd.assemble(h*h*fd.dx(mesh))))
            varQ = sum(((l2_inner(mesh, gQ, phi))**2)/lam for lam,phi in zip(eigvals, phis))
            sigmaQ.append(np.sqrt(varQ))

    with open(outdir / "uq_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "QoI", "sigma"])
        for t, Q, s in zip(times, Qval, sigmaQ):
            writer.writerow([t, Q, s])

    print(f"Wrote results to {outdir}/uq_results.csv")


if __name__ == "__main__":
    main()
