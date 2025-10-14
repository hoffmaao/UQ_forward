#!/usr/bin/env python3
import argparse, pathlib, glob, re, numpy as np
import firedrake as fd
from firedrake_adjoint import *
import icepack

# ---------- friction: control is theta = log(beta) ----------
def friction_stress(u, C, m):

    r"""Compute the shear stress for a given sliding velocity"""
    return -C * fd.sqrt(fd.inner(u, u)) ** (1 / m - 1) * u


def friction(**kwargs):
    r"""Return the bed friction part of the ice stream action functional

    The frictional part of the ice stream action functional is

    .. math::
       E(u) = -\frac{m}{m + 1}\int_\Omega\tau(u, C)\cdot u\; dx

    where :math:`\\tau(u, C)` is the basal shear stress

    .. math::
       \tau(u, C) = -C|u|^{1/m - 1}u
    """
    u = kwargs["velocity"]
    theta = kwargs["log_friction"]
    m = kwargs["sliding_exponent"]
    beta = kwargs["friction"]
    C = beta*fd.exp(theta)

    τ = friction_stress(u, C, m)
    return -m / (m + 1) * fd.inner(τ, u)


# ---------- I/O helpers ----------
def load_mesh_and_fields(inv_ckpt,gamma,tag="ismipc_stats"):
    with fd.CheckpointFile(str(inv_ckpt), "r") as chk:   # <-- cast to str
        mesh = chk.load_mesh(name="ismip-c")
        h0    = chk.load_function(mesh, name=f"{tag}_gamma{gamma:.2e}_thickness")
        s0    = chk.load_function(mesh, name=f"{tag}_gamma{gamma:.2e}_surface")
        b    = chk.load_function(mesh, name=f"{tag}_gamma{gamma:.2e}_bed")
        u    = chk.load_function(mesh, name=f"{tag}_gamma{gamma:.2e}_velocity")
        C    = chk.load_function(mesh, name=f"{tag}_gamma{gamma:.2e}_friction")
        theta= chk.load_function(mesh, name=f"{tag}_gamma{gamma:.2e}_log_friction")
    V = u.function_space()
    Q = h0.function_space()
    U = s0.function_space()
    return mesh, V, Q, U, h0, s0, b, u, theta, C

def find_best_gamma_ckpt(inv_dir, tag="ismipc_stats"):
    # pick the smallest misfit+reg from L-curve CSV if it exists; otherwise most recent file
    csv = pathlib.Path(inv_dir)/f"{tag}_lcurve.csv"
    if csv.exists():
        data = []
        for line in csv.read_text().splitlines()[1:]:
            g,m,r = line.split(",")
            data.append((float(g), float(m), float(r)))
        # "elbow" is subjective; as a simple heuristic, choose minimal (m + r)
        best = min(data, key=lambda t: t[1]+t[2])
        g = best[0]
    return g

def load_theta_from_inv(inv_ckpt, mesh, Q):
    print(inv_ckpt)
    # Your inverse saves either θ or β (named 'beta'); convert β→θ robustly:contentReference[oaicite:2]{index=2}
    with fd.CheckpointFile(str(inv_ckpt), "r") as chk:
        try:
            theta = chk.load_function(mesh, name="theta")
        except Exception:
            beta  = chk.load_function(mesh, name="beta")
            arr   = np.maximum(beta.dat.data_ro, 1e-12)
            theta = fd.Function(Q, name="theta"); theta.dat.data[:] = np.log(arr)
    return theta

# ---------- Icepack solver ----------
def build_solver(mesh, A_MPa_yr):
    A = fd.Constant(A_MPa_yr, domain=mesh)
    model = icepack.models.IceStream(
        friction=friction,
    )
    # use actual exterior facet markers as side walls (matches run_inv):contentReference[oaicite:3]{index=3}
    side_ids = list(mesh.exterior_facets.unique_markers)
    opts = {
        "side_wall_ids": side_ids,
        "diagnostic_solver_type": "petsc",
        "diagnostic_solver_parameters": {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_max_it": 100,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        "prognostic_solver_parameters": {
            "ksp_type": "gmres",
            "pc_type": "ilu",
        },
    }
    return icepack.solvers.FlowSolver(model, **opts), A

# ---------- QoI ----------
def qoi_form(h, mesh):
    # ISMIP‑C QoI: ∫ h^2 dx (the guide’s recommended choice for the periodic box)
    return (h*h) * fd.dx(mesh)

# ---------- forward integration ----------
def run_forward(mesh, V, Q, U, h0, s0, b, theta, A, T_years, nsteps,
                n_sens=1, mesh_file = "ismip-c.h5", outdir="outputs/forward", seed_speed=100.0):
    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    dt = fd.Constant(T_years/float(nsteps), domain=mesh)
    accum = fd.Function(U, name="accum"); accum.assign(0.0)  # ISMIP‑C extension: no SMB

    h = h0.copy(deepcopy=True)
    s = s0.copy(deepcopy=True)

    beta = fd.Constant(1000.0/1000000,domain=mesh)

    A = fd.Constant(A, domain=mesh)
    model = icepack.models.IceStream(
        friction=friction,
    )
    side_ids = list(mesh.exterior_facets.unique_markers)
    opts = {
        "side_wall_ids": side_ids,
        "diagnostic_solver_type": "petsc",
        "diagnostic_solver_parameters": {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_max_it": 100,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        "prognostic_solver_parameters": {
            # The prognostic step is linear, so no nonlinear iterations
            "snes_type": "ksponly",
            # One linear solve with a direct factorization
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    }

    solver=icepack.solvers.FlowSolver(model, **opts)

    u_seed = fd.Function(V); u_seed.interpolate(fd.as_vector([seed_speed, 0.0]))

    # choose sensitivity output times (like -s in fenics_ice forward):contentReference[oaicite:5]{index=5}
    if n_sens <= 1:
        sens_idxs = [nsteps]
    else:
        sens_idxs = np.linspace(0, nsteps, n_sens, dtype=int).tolist()

    results = []

    # Start with a clean tape and explicitly enable annotation.
    get_working_tape().clear_tape()
    continue_annotation()

    # Create the Control once; reuse it each time we form a ReducedFunctional
    control_theta = Control(theta)


    with fd.CheckpointFile(mesh_file, "a") as chk:
        for k in range(0, nsteps):
            # diagnostic velocity for current state
            u = solver.diagnostic_solve(
                velocity=u_seed, thickness=h, surface=s,
                log_friction=theta, friction=beta, fluidity=fd.Constant(A,domain=mesh), sliding_exponent=fd.Constant(1.0,domain=mesh))
            # prognostic thickness step
            h = solver.prognostic_solve(thickness=h, velocity=u, accumulation=accum, dt=dt)
            # simple hydrostatic update of surface for this toy ISMIP‑C forward
            sfd .interpolate(b + h)

            if k in sens_idxs:
                Jk_form = qoi_form(h, mesh)
                Jk = fd.assemble(Jk_form)           # Overloaded Functional (annotated)
                Jk_val = float(Jk)                  # numeric value for logging

                rf = ReducedFunctional(Jk, control_theta)
                dJ_dtheta = rf.derivative()         # gradient in same space as theta
                g = fd.Function(Q); g.interpolate(dJ_dtheta)
                
                chk.save_function(h, name="thickness_timeseries", idx=k)                     # "h"
                chk.save_function(u, name="velocity_timeseries", idx=k)                     # "u"
                chk.save_function(g, name="dQ_dtheta_timeseries", idx=k)
                print(k)

                results.append((k, float(Jk_val)))

    with open(outdir/"qoi_timeseries.csv", "w") as f:
        f.write("step,Q\n")
        for k, Qk in results: f.write(f"{k},{Qk}\n")
    return results

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_chk", default="ismip-c.h5",
                    help="checkpoint from build_mesh.py (mesh + h,s,b)")
    ap.add_argument("--inv_dir", default="outputs/inversion_stats",
                    help="directory with inversion outputs (per-γ files)")
    ap.add_argument("--tag", default="ismipc_stats",
                    help="filename tag used by inversion")
    ap.add_argument("--A", type=float, default=100.0,
                    help="fluidity (MPa^-3 yr^-1)")
    ap.add_argument("--T", type=float, default=30.0,
                    help="total years")
    ap.add_argument("--n", type=int, default=120,
                    help="timesteps")
    ap.add_argument("--s", type=int, default=1,
                    help="number of sensitivity outputs (1=last only)")
    ap.add_argument("--outdir", default="outputs/forward",
                    help="output directory")
    args = ap.parse_args()

    # load base mesh + geometry
    # pick the γ-run to use and load MAP control (θ or β -> θ)
    gamma = find_best_gamma_ckpt(args.inv_dir, tag=args.tag)
    print(gamma)
    mesh, V, Q, U, h0, s0, b, u, theta, C = load_mesh_and_fields(args.mesh_chk, gamma= gamma)

    A = fd.Constant(100.0)

    # run prognostic forward and write QoI & sensitivities
    run_forward(mesh, V, Q, U, h0, s0, b, theta,
                A=A, T_years=args.T, nsteps=args.n,
                n_sens=args.s, mesh_file = args.mesh_chk, outdir=args.outdir)

if __name__ == "__main__":
    main()
