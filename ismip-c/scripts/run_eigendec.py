import argparse, pathlib, numpy as np
import firedrake as fd
from firedrake.adjoint import *                
import icepack

# optional TLM/adjoint path
# optional TLM/adjoint path (Firedrake backend)
try:
    import tlm_adjoint.firedrake as tla  # <-- note the .firedrake backend
    from tlm_adjoint.firedrake import (
        Functional, CachedHessian, reset_manager, start_manager, stop_manager
    )
    HAVE_TLM = True
except Exception:
    HAVE_TLM = False
    tla = None


# optional eigensolver path
try:
    from scipy.sparse.linalg import LinearOperator, eigsh
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ---------------- friction (θ = log β) ----------------
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


# ---------------- I/O ----------------
def load_mesh_and_fields(mesh_chk, gamma, tag="ismipc_stats"):
    with fd.CheckpointFile(str(mesh_chk), "r") as chk:
        mesh = chk.load_mesh(name="ismip-c")
        h    = chk.load_function(mesh, name=f"{tag}_gamma{gamma:.2e}_thickness")
        s    = chk.load_function(mesh, name=f"{tag}_gamma{gamma:.2e}_surface")
        b    = chk.load_function(mesh, name=f"{tag}_gamma{gamma:.2e}_bed")
        u    = chk.load_function(mesh, name=f"{tag}_gamma{gamma:.2e}_velocity")
        u_obs= chk.load_function(mesh, name="u_obs")
        C    = chk.load_function(mesh, name=f"{tag}_gamma{gamma:.2e}_friction")
        theta= chk.load_function(mesh, name=f"{tag}_gamma{gamma:.2e}_log_friction")
    V = u.function_space()
    Q = h.function_space()
    U = s.function_space()
    return mesh, V, Q, U, h, s, b, u, u_obs, theta, C

def load_theta(inv_ckpt, mesh, Q):
    with fd.CheckpointFile(str(inv_ckpt), "r") as chk:
        if "theta" in chk._index:
            return chk.load_function(mesh, name="theta")
        beta = chk.load_function(mesh, name="beta")
    th = fd.Function(Q, name="theta")
    th.dat.data[:] = np.log(np.maximum(beta.dat.data_ro, 1e-12))
    return th

# ---------------- reduced functional pieces ----------------
def build_reduced_pieces(mesh, V, Q, h, s, u_obs,
                         A_MPa_yr, sigma_obs, gamma, ell, theta_scale):
    A       = fd.Constant(A_MPa_yr)
    sigma   = fd.Constant(sigma_obs)
    gamma_c = fd.Constant(gamma)
    ell_c   = fd.Constant(ell)
    ths_c   = fd.Constant(theta_scale)

    # Area normalization
    area_val = float(fd.assemble(fd.Constant(1.0) * fd.dx(mesh)))
    invA     = fd.Constant(1.0 / area_val)
    beta = fd.Constant(1000.0/1000000,domain=mesh)

    model = icepack.models.IceStream(friction=friction)
    side_ids = list(mesh.exterior_facets.unique_markers)
    solver = icepack.solvers.FlowSolver(
        model,
        side_wall_ids=side_ids,
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_max_it": 100,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )

    def simulation(theta):
        return solver.diagnostic_solve(
                velocity=u_obs, thickness=h, surface=s,
                fluidity=A, log_friction=theta, friction=beta,
                sliding_exponent=fd.Constant(1,domain=mesh)
            )

    def misfit(u):
        du = u - u_obs
        # (1/2σ^2) ∫ |du|^2 dx / area
        return (fd.Constant(0.5) / sigma**2) * fd.inner(du, du) * invA * fd.dx(mesh)

    def prior(theta):
        # (γ/2) || (ℓ/θs) ∇θ ||^2 / area
        return (fd.Constant(0.5) * gamma_c * (ell_c/ths_c)**2
                * fd.inner(fd.grad(theta), fd.grad(theta)) * invA * fd.dx(mesh))

    params = {"sigma": sigma, "invA": invA}
    return simulation, misfit, prior, params


# ---------------- GN Hessian·v with tlm_adjoint ----------------
def make_Hv_tlm(mesh, V, Q, simulation, misfit, prior, theta_map, params, eps=1e-3):
    if not HAVE_TLM:
        raise RuntimeError("tlm_adjoint is not available")

    # Record the forward computation at the MAP parameter
    tla.reset_manager()
    # If you run this repeatedly in one Python session, also do:
    # from tlm_adjoint.firedrake import clear_caches; clear_caches()

    def forward(theta):
        # 1) forward PDE solve
        u = simulation(theta)
        # 2) define the cost as a tlm_adjoint Functional (annotated)
        J = Functional(name="J")
        # misfit(u) and prior(theta) are 0-forms; assign the sum
        J.assign(misfit(u) + prior(theta))
        return J

    start_manager()
    J_map = forward(theta_map)   # records the forward DAG at the MAP point
    stop_manager()

    # Build a CachedHessian object so we can reuse forward/adjoint state
    H = CachedHessian(J_map)     # See tlm_adjoint example notebook

    def Hv(v):
        # Returns (J(theta), dJ/dtheta[v], H[v]) where H[v] is a dual (cofunction)
        _, _, H_dual = H.action(theta_map, v)
        # Convert dual V* to a primal Function in Q with a Riesz map (L2 is fine)
        H_fun = H_dual.riesz_representation("L2")
        return H_fun

    return Hv


# ---------------- fallback FD Hessian·v (if no tlm) ----------------
def make_Hv_fd(Q, simulate, misfit, prior, theta_map, eps=1e-3):
    def grad(theta):
        u  = simulate(theta)
        J  = assemble(misfit(u) + prior(theta))         # <-- NOTE: assemble(...) not fd.assemble(...)
        rf = ReducedFunctional(J, Control(theta))
        return rf.derivative()
    def Hv(v):
        g0 = grad(theta_map)
        thp = theta_map.copy(deepcopy=True); thp.assign(theta_map + eps*v)
        g1 = grad(thp)
        out = g1.copy(deepcopy=True); out.assign((g1 - g0)*(1.0/eps))
        return out
    return Hv

# ---------------- eigensolve (ARPACK via SciPy) ----------------
def eigs_linear_operator(Q, Hv, k=40, maxiter=None):
    if not HAVE_SCIPY:
        raise RuntimeError("SciPy is required for eigsh in the no‑slepc path")
    n = Q.dof_dset.size
    def matvec(x):
        v = fd.Function(Q); v.dat.data[:] = x
        Hv_v = Hv(v)
        return Hv_v.dat.data_ro.copy()
    Aop = LinearOperator((n, n), matvec=matvec, dtype=float)
    vals, vecs = eigsh(Aop, k=min(k, n-2), which="LM", maxiter=maxiter)
    # sort descending
    idx = np.argsort(-vals); vals = vals[idx]; vecs = vecs[:, idx]
    phis = []
    for i in range(vals.shape[0]):
        f = fd.Function(Q); f.dat.data[:] = vecs[:, i]
        phis.append(f)
    return vals, phis


# --- prior K^{-1} solver (whitened) -----------------------------------------
def make_prior_solve(mesh, Q, *,
                     gamma,         # fd.Constant, >0
                     ell,           # fd.Constant, >0
                     theta_scale,   # fd.Constant, >0
                     sigma,         # fd.Constant, >0  (controls mass-term weight)
                     invA=None,     # optional area normalizer (fd.Constant(1/area))
                     use_lu=False,  # True -> direct solve (MUMPS)
                     hypre_strong_thres=0.7):
    """
    Returns a callable Kinv_apply(v) that solves:  (delta*M + gamma*(ell/theta_scale)^2*K) y = v
    where M ~ L2 mass, K ~ Laplacian stiffness, all optionally scaled by invA.
    """

    # Trial/test and holders
    y = fd.Function(Q, name="Kinv_sol")   # solution
    z = fd.Function(Q, name="rhs_func")   # RHS function; we'll assign 'v' into this
    v = fd.TrialFunction(Q)
    w = fd.TestFunction(Q)

    # Area normalization (optional)
    if invA is None:
        invA = fd.Constant(1.0, domain=mesh)

    # Make the operator strictly SPD by adding a mass term.
    # A practical, robust choice is delta = 1/sigma^2.
    delta = fd.Constant(1.0, domain=mesh) / (sigma * sigma)
    scale_grad = (ell / theta_scale) * (ell / theta_scale)

    a_form = ( delta * invA * fd.inner(v, w) * fd.dx
             + gamma * scale_grad * invA * fd.inner(fd.grad(v), fd.grad(w)) * fd.dx )

    # The RHS is the Riesz map of 'z' in the same inner product we used above
    # (here we keep it simple and use L2 with the same invA scaling):
    L_form = invA * fd.inner(z, w) * fd.dx

    # Linear variational problem
    problem = fd.LinearVariationalProblem(a_form, L_form, y, bcs=None)

    # Robust solver defaults: CG + BoomerAMG (or LU/MUMPS on small meshes)
    if use_lu:
        solver_parameters = {
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    else:
        solver_parameters = {
            "mat_type": "aij",
            "ksp_type": "cg",
            "ksp_rtol": 1e-10,
            "ksp_atol": 0.0,
            "ksp_max_it": 500,
            "ksp_norm_type": "preconditioned",
            "ksp_converged_reason": None,     # print reason on exit
            # AMG preconditioner
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "pc_hypre_boomeramg_strong_threshold": hypre_strong_thres,
            "pc_hypre_boomeramg_max_iter": 1,
            "pc_hypre_boomeramg_cycle_type": "V",
            "pc_hypre_boomeramg_relax_type_all": "symmetric-SOR/Jacobi",
        }

    solver = fd.LinearVariationalSolver(problem, solver_parameters=solver_parameters)

    # Build the apply() closure. We pause tlm_adjoint annotation for this pure linear solve.
    def Kinv_apply(rhs_fun):
        # rhs_fun is an fd.Function on Q
        z.assign(rhs_fun)
        # Pause tlm_adjoint (if present) so it doesn't wrap this linear solve
        try:
            import tlm_adjoint as _tla
            _tla.stop_manager()
        except Exception:
            pass
        try:
            solver.solve()
        finally:
            try:
                _tla.start_manager()
            except Exception:
                pass
        return y.copy(deepcopy=True)

    # quick self-check hook (optional; set to True to print residuals once)
    Kinv_apply._check = lambda: None
    return Kinv_apply
# ---------------------------------------------------------------------------


# --- replace eigs_linear_operator(...) call with whitened version ---

def eigs_whitened(Q, Hv, Kinv_apply, k=40, maxiter=None):
    """ARPACK eigensolve for K^{-1} H (prior‑whitened)."""
    from scipy.sparse.linalg import LinearOperator, eigsh
    n = Q.dof_dset.size
    def matvec(x):
        v = fd.Function(Q); v.dat.data[:] = x
        Hv_v = Hv(v)                 # H v
        y = Kinv_apply(Hv_v)         # K^{-1} H v
        return y.dat.data_ro.copy()
    Aop = LinearOperator((n, n), matvec=matvec, dtype=float)
    vals, vecs = eigsh(Aop, k=min(k, n-2), which="LM", maxiter=maxiter)
    # sort by descending eigenvalue (largest = most data‑informed)
    idx = np.argsort(-vals); vals = vals[idx]; vecs = vecs[:, idx]
    phis = []
    for i in range(vals.shape[0]):
        f = fd.Function(Q); f.dat.data[:] = vecs[:, i]
        phis.append(f)
    return vals, phis


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inv_ckpt", required=True)
    ap.add_argument("--A", type=float, default=100.0)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--ell", type=float, default=7.5e3)
    ap.add_argument("--theta_scale", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--outdir", default="outputs/eigendec_tlm")
    args = ap.parse_args()

    mesh, V, Q, U, h, s, b, u, u_obs, theta_map, C = load_mesh_and_fields(args.inv_ckpt,gamma=args.gamma)
    #theta_map = load_theta(args.inv_ckpt, mesh, Q)


    simulate, misfit, prior, params = build_reduced_pieces(
        mesh, V, Q, h, s, u_obs,
        A_MPa_yr=args.A, sigma_obs=args.sigma,
        gamma=args.gamma, ell=args.ell, theta_scale=args.theta_scale
    )

    # Build H·v
    if HAVE_TLM:
        Hv = make_Hv_tlm(mesh, V, Q, simulate, misfit, prior, theta_map, params, eps=args.eps)
        print("Using tlm_adjoint for GN Hessian actions.")
    else:
        Hv = make_Hv_fd(Q, simulate, misfit, prior, theta_map, eps=args.eps, params=params)
        print("tlm_adjoint not found; falling back to FD Hessian actions.")

    Kinv = make_prior_solve(
        mesh, Q,
        gamma=fd.Constant(args.gamma),
        ell=fd.Constant(args.ell),
        theta_scale=fd.Constant(args.theta_scale),
        sigma=fd.Constant(params["sigma"]),
        invA=params.get("invA", fd.Constant(1.0, domain=mesh)),
        use_lu=False  # flip to True on tiny meshes or for a quick sanity test
    )

    vals, phis = eigs_whitened(Q, Hv, Kinv, k=args.k)

    # Eigensolve (SciPy ARPACK)
    #vals, phis = eigs_linear_operator(Q, Hv, k=args.k)

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    with fd.CheckpointFile(str(args.inv_ckpt), "a") as chk:
        for i, phi in enumerate(phis):
            phi.rename(f"phi_{i:04d}")
            chk.save_function(phi)
    with open(outdir/"eigenvalues.txt", "w") as f:
        for i, lam in enumerate(vals):
            f.write(f"{i} {lam:.16e}\n")
    print(f"Converged {len(vals)} modes; wrote {outdir}/gn_eigs.h5 and eigenvalues.txt")

if __name__ == "__main__":
    main()
