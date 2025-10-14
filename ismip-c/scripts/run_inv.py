import argparse, pathlib, math
import numpy as np
import firedrake as fd
from firedrake_adjoint import *
import icepack
from icepack.statistics import StatisticsProblem, MaximumProbabilityEstimator
import matplotlib.pyplot as plt



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


def load_from_checkpoint(chk_path):
    with fd.CheckpointFile(chk_path, "r") as chk:
        mesh = chk.load_mesh(name="ismip-c")
        h    = chk.load_function(mesh, name="thickness")
        s    = chk.load_function(mesh, name="surface")
        b    = chk.load_function(mesh, name="bed")
        # optional names used earlier:
        try:
            u_obs = chk.load_function(mesh, name="u_obs")
        except Exception:
            u_obs = chk.load_function(mesh, name="velocity_obs")
        try:
            beta0 = chk.load_function(mesh, name="beta")
        except Exception:
            beta0 = chk.load_function(mesh, name="friction")
    V = u_obs.function_space()
    Q = beta0.function_space()
    U = s.function_space()
    return mesh, V, Q, U, h, s, b, u_obs, beta0





def lcurve(mesh, V, Q, h, b, s, u_obs, beta_init,
           A=100.0,  # ≈ 1e-16 Pa^-3 yr^-1 in Icepack units
           sigma_obs=1.0,   # m/yr
           gammas=(1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,1e1,3e1,1e2,3e2,1e3,3e3,1e4,4e4),
           L =7.5e3,       # length‑scale (m)
           theta_scale =1.0, # nondimensional scale for θ
           outdir=pathlib.Path("outputs/inversion_stats")):

    # (Re)build solver on the current mesh
    model = icepack.models.IceStream(
        friction=friction
    )
    # On a periodic-x mesh, only y-walls are exterior facets:
    opts = {
        'side_wall_ids': [1, 2]
    }

    solver=icepack.solvers.FlowSolver(model, **opts)

    # domain area for normalisation
    area = fd.Constant(fd.assemble(fd.Constant(1.0) * fd.dx(mesh)),domain=mesh)
    theta = fd.Function(Q, name="theta")
    beta = fd.Constant(1000.0/1000000,domain=mesh)


    def simulation(theta):
        return solver.diagnostic_solve(
                velocity=u_obs, thickness=h, surface=s,
                fluidity=A, log_friction=theta, friction=beta,
                sliding_exponent=fd.Constant(1,domain=mesh)
            )

    def loss_function(u):
        du = u - u_obs
        return 0.5 / area * ((du[0]/sigma_obs)**2 + (du[1]/sigma_obs)**2) * fd.dx

    def regularization(theta):
        return 0.5 / area  * gamma * (L / theta_scale)**2 * fd.inner(fd.grad(theta), fd.grad(theta)) * fd.dx


    #u = simulation(theta)





    points = []
    for g in gammas:
        gamma=fd.Constant(g,domain=mesh)
        problem = StatisticsProblem(
            simulation=simulation,
            loss_functional=loss_function,
            regularization=regularization,
            controls=theta,
        )
        estimator = MaximumProbabilityEstimator(
            problem,
            gradient_tolerance=1e-8,
            step_tolerance=1e-8,
            max_iterations=400,
        )
        theta = estimator.solve()
        u = simulation(theta)
        # evaluate terms for L‑curve
        mis = fd.assemble(loss_function(u))
        reg = fd.assemble(regularization(theta))
        # save result for this γ
        with fd.CheckpointFile(str(outdir), "a") as chk:
            chk.save_mesh(mesh)             # for viewing; periodic IDs aren’t preserved
            chk.save_function(theta, name=f"gamma{g:.2e}_log_friction")
            C = fd.Function(Q, name=f"gamma{g:.2e}_beta"); C.interpolate(beta*fd.exp(theta))
            chk.save_function(C, name=f"gamma{g:.2e}_friction")
            chk.save_function(u, name=f"gamma{g:.2e}_velocity")
            chk.save_function(s, name=f"gamma{g:.2e}_surface")
            chk.save_function(h, name=f"gamma{g:.2e}_thickness")
            chk.save_function(b, name=f"gamma{g:.2e}_bed")
            chk.save_function(u_obs, name="u_obs")

        points.append((g, mis, reg))
        print(f"[γ={g:.2e}] misfit={mis:.6e}, regularizer={reg:.6e}")

    # L‑curve plot (misfit vs regularizer) on log–log axes
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    plt.figure(figsize=(6,5))
    plt.loglog(xs, ys, marker="o")
    for (g,x,y) in zip(gammas, xs, ys):
        plt.annotate(f"{g:.0e}", (x,y), textcoords="offset points", xytext=(4,4), fontsize=8)
    plt.xlabel("velocity misfit term")
    plt.ylabel("regularization term")
    plt.title("L‑curve (β inversion, Icepack statistics)")
    plt.tight_layout()
    plt.savefig(outdir "/" f"lcurve.png", dpi=200)
    with open(outdir "/" f"lcurve.csv", "w") as f:
        f.write("gamma,misfit,regularization\n")
        for g,m,r in points:
            f.write(f"{g},{m},{r}\n")
    return points

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chk", default="ismip-c.h5", help="checkpoint written by build_mesh.py")
    ap.add_argument("--A", type=float, default=100.0, help="fluidity in MPa^-3 yr^-1")
    ap.add_argument("--sigma", type=float, default=1.0, help="velocity noise (m/yr)")
    ap.add_argument("--gammas", default="1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,1e1,3e1,1e2,3e2,1e3,3e3,1e4")
    ap.add_argument("--L", type=float, default=7.5e3, help="H1 length scale (m)")
    ap.add_argument("--theta_scale", type=float, default=1.0)
    args = ap.parse_args()

    mesh, V, Q, U, h, s, b, u_obs, beta0 = load_from_checkpoint(args.chk)

    # If you need *periodic-x*, rebuild the periodic mesh here and
    # reinterpolate the fields; otherwise proceed with the loaded mesh.

    gammas = tuple(float(v) for v in args.gammas.split(","))
    lcurve(mesh, V, Q, h, b, s, u_obs, beta0,
           A=fd.Constant(args.A,domain=mesh), sigma_obs=fd.Constant(args.sigma,domain=mesh),
           gammas=gammas, L=fd.Constant(args.L,domain=mesh), theta_scale=fd.Constant(args.theta_scale,domain=mesh),
           outdir=args.chk)

if __name__ == "__main__":
    main()
