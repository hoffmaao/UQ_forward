import os, pathlib, numpy as np
import firedrake as fd
import icepack

# --- tlm_adjoint (Firedrake backend) ---
try:
    from tlm_adjoint.firedrake import (
        Functional, CachedHessian, reset_manager, start_manager, stop_manager
    )
except Exception as exc:
    raise RuntimeError(f"tlm_adjoint not available: {exc}")

# --- SciPy / ARPACK ---
try:
    from scipy.sparse.linalg import LinearOperator, eigsh
except Exception as exc:
    raise RuntimeError(f"SciPy eigsh required but not available: {exc}")

# Optional: mitigate locking quirks on shared filesystems
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# ---------------------------- Physics blocks ---------------------------------
def friction_stress(u, C, m, u_reg=fd.Constant(1e-6)):
    speed = fd.sqrt(fd.inner(u, u) + u_reg**2)
    return -C * speed ** (1.0 / m - 1.0) * u

def friction(**kwargs):
    u        = kwargs["velocity"]
    theta    = kwargs["log_friction"]
    mexp     = kwargs.get("sliding_exponent", fd.Constant(1.0))
    C0       = kwargs["friction"]
    grounded = kwargs.get("grounded", None)
    C = C0 * fd.exp(theta)
    if grounded is not None:
        C = C * grounded
    tau = friction_stress(u, C, mexp)
    return -mexp / (mexp + 1.0) * fd.inner(tau, u)

def viscosity(**kwargs):
    A0  = kwargs['fluidity']
    u   = kwargs['velocity']
    h   = kwargs['thickness']
    phi = kwargs['log_fluidity']
    A   = A0 * fd.exp(phi)
    return icepack.models.viscosity.viscosity_depth_averaged(
        velocity=u, thickness=h, fluidity=A
    )

# ---------------------------- Flotation & masks -------------------------------
def flotation_surface_from_bed(bed, rho_i, rho_w):
    return bed + (rho_w / rho_i) * fd.max_value(-bed, 0.0)

def _smooth(q0, alpha=2e3):
    q = q0.copy(deepcopy=True)
    J = 0.5 * ((q - q0)**2 + alpha**2 * fd.inner(fd.grad(q), fd.grad(q))) * fd.dx
    F = fd.derivative(J, q)
    fd.solve(F == 0, q)
    return q

def grounded_mask(surface, s_float, Q):
    z = _smooth(fd.interpolate(surface - s_float, Q), alpha=100.0)
    floating = fd.interpolate(fd.conditional(z < 0, 1.0, 0.0), Q)
    grounded = fd.interpolate(1.0 - floating, Q)
    return floating, grounded

# ---------------------------- I/O ---------------------------------------------
def load_from_checkpoint(path, gamma):
    """Names match your venable run_inv.py/forward."""
    with fd.CheckpointFile(str(path), "r") as chk:
        mesh  = chk.load_mesh(name="venable")
        h     = chk.load_function(mesh, name="thickness")
        s     = chk.load_function(mesh, name="surface")
        b     = chk.load_function(mesh, name="bed")
        uobs  = chk.load_function(mesh, name="velocity_obs")
        sig   = chk.load_function(mesh, name="sigma_obs")      # (σx, σy)
        C0    = chk.load_function(mesh, name="friction")
        A0    = chk.load_function(mesh, name="fluidity")
        theta = chk.load_function(mesh, name=f"gamma{gamma:.2e}_log_friction")
        phi   = chk.load_function(mesh, name=f"gamma{gamma:.2e}_log_fluidity")
    V = uobs.function_space()
    Q = C0.function_space()
    return mesh, V, Q, h, s, b, uobs, sig, C0, A0, theta, phi

# ------------------------ Forward (Dirichlet via solve) -----------------------
def make_forward_dirichlet_solve(
    *, model, mesh, V, h, s, C0, A0, grounded,
    dirichlet_ids, g_value, newton_sp=None
):
    """
    forward(theta, phi) -> u using firedrake.solve with hard Dirichlet BCs.
    We *reference* the BC Function in the residual with a zero-weight term so
    tlm_adjoint records it on the tape (avoids 'Invalid dependency').
    """
    if newton_sp is None:
        newton_sp = {
            "snes_type": "newtonls", "snes_linesearch_type": "bt",
            "snes_rtol": 1e-9, "snes_atol": 1e-10, "snes_max_it": 60,
            "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
        }

    bc_val = g_value
    if isinstance(g_value, fd.Function) and (g_value.function_space() != V):
        bc_val = fd.project(g_value, V)
    if isinstance(bc_val, (int, float)):
        bc_val = fd.Constant((float(bc_val), 0.0))

    bcs = [fd.DirichletBC(V, bc_val, int(bid)) for bid in dirichlet_ids]

    def forward(theta, phi):
        u = fd.Function(V, name="u")
        v = fd.TestFunction(V)
        # seed to avoid |u|=0 corner in friction
        if isinstance(bc_val, fd.Function):
            u.assign(bc_val)
        else:
            u.interpolate(fd.as_vector([fd.Constant(1e-6), fd.Constant(0.0)]))

        E = model.action(
            velocity=u, thickness=h, surface=s,
            log_friction=theta, friction=C0,
            log_fluidity=phi,  fluidity=A0,
            grounded=grounded
        )
        F = fd.derivative(E, u, v)

        # Register BC Function with the tape (no physics change)
        if isinstance(bc_val, fd.Function):
            zero = fd.Constant(0.0, domain=mesh)
            F += zero * fd.inner(bc_val, v) * fd.dx(mesh)

        fd.solve(F == 0, u, bcs=bcs, solver_parameters=newton_sp)
        return u

    return forward

# ---------------------- GN Hessian action via zero-residual -------------------
def make_Hv_gn(forward_u, *, mesh, invA, sigma_vec, theta_map, phi_map):
    """
    Exact GN reduced Hessian action using the zero-residual trick:
      tape J(u) = 0.5 ∫ invA * ((u - u_MAP)/σ)^2 dx
    """
    reset_manager()
    u_MAP = forward_u(theta_map, phi_map).copy(deepcopy=True)

    def misfit_GN(u):
        du = u - u_MAP
        return 0.5 * invA * ((du[0]/sigma_vec[0])**2 + (du[1]/sigma_vec[1])**2) * fd.dx(mesh)

    def record(theta, phi):
        th = fd.Function(theta.function_space()); th.assign(theta)
        ph = fd.Function(phi.function_space());   ph.assign(phi)
        u  = forward_u(th, ph)
        J  = Functional(name="J"); J.assign(misfit_GN(u))
        return J

    start_manager(); J_map = record(theta_map, phi_map); stop_manager()
    H = CachedHessian(J_map)

    def Hv_pair(vθ, vφ):
        _, _, ddJ = H.action([theta_map, phi_map], [vθ, vφ])
        return ddJ[0].riesz_representation("L2"), ddJ[1].riesz_representation("L2")

    return Hv_pair

# -------------------------- Prior K^{-1} applies ------------------------------
def make_prior_solves(mesh, Q, invA,
                      gamma_theta, ell_theta, theta_scale,
                      gamma_phi,   ell_phi,   phi_scale):
    def one(gamma, ell, scale):
        y = fd.Function(Q); z = fd.Function(Q)
        v, w = fd.TrialFunction(Q), fd.TestFunction(Q)
        a = ( invA*fd.inner(v, w)
              + gamma*(ell/scale)**2 * invA * fd.inner(fd.grad(v), fd.grad(w)) ) * fd.dx(mesh)
        L = invA * fd.inner(z, w) * fd.dx(mesh)
        pr = fd.LinearVariationalProblem(a, L, y)
        sp = {"ksp_type":"cg","ksp_rtol":1e-10,"pc_type":"hypre","pc_hypre_type":"boomeramg"}
        so = fd.LinearVariationalSolver(pr, solver_parameters=sp)
        def apply(rhs):
            z.assign(rhs); so.solve(); return y.copy(deepcopy=True)
        return apply
    Kθ_inv = one(fd.Constant(gamma_theta, domain=mesh),
                 fd.Constant(ell_theta,   domain=mesh),
                 fd.Constant(theta_scale, domain=mesh))
    Kφ_inv = one(fd.Constant(gamma_phi,   domain=mesh),
                 fd.Constant(ell_phi,     domain=mesh),
                 fd.Constant(phi_scale,   domain=mesh))
    return Kθ_inv, Kφ_inv

# ---------------------------- ARPACK eigensolve -------------------------------
def eigs_whitened_pair(Q, Hv_pair, Kθ_inv, Kφ_inv, k, tol=1e-6, maxiter=None):
    """
    ARPACK on the (similar-to-symmetric) operator A = K^{-1} H (GN).
    Returns eigenvalues (desc) and MixedFunction modes [Q,Q].
    """
    n = Q.dof_dset.size

    def _vec_to_pair(x):
        vθ = fd.Function(Q); vθ.dat.data[:] = x[:n]
        vφ = fd.Function(Q); vφ.dat.data[:] = x[n:]
        return vθ, vφ

    def _pair_to_vec(aθ, aφ):
        return np.concatenate([aθ.dat.data_ro.copy(), aφ.dat.data_ro.copy()])

    def matvec(x):
        vθ, vφ = _vec_to_pair(x)
        hθ, hφ = Hv_pair(vθ, vφ)  # H v
        yθ = Kθ_inv(hθ)           # K^{-1} H v
        yφ = Kφ_inv(hφ)
        return _pair_to_vec(yθ, yφ)

    A = LinearOperator((2*n, 2*n), matvec=matvec, dtype=float)
    k_eff = min(k, 2*n - 2)  # ARPACK requirement
    vals, vecs = eigsh(A, k=k_eff, which="LM", tol=tol, maxiter=maxiter)

    # sort descending
    idx = np.argsort(-vals); vals = vals[idx]; vecs = vecs[:, idx]

    QQ = fd.MixedFunctionSpace([Q, Q])
    modes = []
    for i in range(vals.size):
        z = fd.Function(QQ)
        z.sub(0).dat.data[:] = vecs[:n, i]
        z.sub(1).dat.data[:] = vecs[n:, i]
        modes.append(z)
    return vals, modes

# ------------------------ Robust, chunked HDF5 saving -------------------------
def save_modes_chunked(CKPT, mesh, modes, chunk=200, same_file=True, modes_path=None):
    """
    Save MixedFunction modes in chunks to avoid HDF5 viewer close errors.
    If same_file=False, write to a separate file (recommended for thousands of modes).
    """
    target = CKPT if same_file else (modes_path or CKPT.replace(".h5", "_modes.h5"))
    # ensure the modes file has the mesh
    if not same_file:
        with fd.CheckpointFile(target, "w", comm=fd.COMM_SELF) as chk:
            chk.save_mesh(mesh, name="venable")
    for base in range(0, len(modes), chunk):
        hi = min(base + chunk, len(modes))
        with fd.CheckpointFile(target, "a", comm=fd.COMM_SELF) as chk:
            for i in range(base, hi):
                modes[i].rename(f"mode_{i:04d}")
                chk.save_function(modes[i])
    return target

# ============================== Configuration =================================
CKPT            = "../mesh/venable.h5"
GAMMA           = 3.0          # which MAP set to read (gamma^{misfit} you selected)
DIRICHLET_IDS   = [1, 2, 4]    # Venable: clamp walls; calving front = natural
K_LEADING       = 200          # request this many modes (cap enforced internally)

# Prior hyperparameters (match your inversion)
GAMMA_THETA = 3.0;  ELL_THETA = 7.5e3; THETA_SCALE = 1.0
GAMMA_PHI   = 1.0;  ELL_PHI   = 7.5e3; PHI_SCALE   = 1.0

# Save options
SAVE_SAME_FILE   = False   # write modes to venable_modes.h5 (safer for big runs)
SAVE_CHUNK       = 200

# ================================ Pipeline ====================================
mesh, V, Q, h, s, b, uobs, sigma_vec, C0, A0, theta_map, phi_map = load_from_checkpoint(CKPT, GAMMA)  # venable names

# Model & masks
model  = icepack.models.IceStream(friction=friction, viscosity=viscosity)
rho_i  = icepack.constants.ice_density
rho_w  = icepack.constants.water_density
s_float = flotation_surface_from_bed(b, rho_i, rho_w)
_, grounded = grounded_mask(s, s_float, Q)

# Forward with hard Dirichlet BCs (BC Function referenced on tape)
forward_u = make_forward_dirichlet_solve(
    model=model, mesh=mesh, V=V, h=h, s=s, C0=C0, A0=A0, grounded=grounded,
    dirichlet_ids=DIRICHLET_IDS, g_value=uobs
)

# Area normalization for GN misfit
A = float(fd.assemble(fd.Constant(1.0) * fd.dx(mesh)))
invA = fd.Constant(1.0 / A, domain=mesh)

# GN Hessian action at the MAP
Hv_pair = make_Hv_gn(forward_u, mesh=mesh, invA=invA, sigma_vec=sigma_vec,
                     theta_map=theta_map, phi_map=phi_map)  # mirrors prior ARPACK scripts

# Prior K^{-1} applies (scalar blocks)
Kθ_inv, Kφ_inv = make_prior_solves(
    mesh, Q, invA,
    gamma_theta=GAMMA_THETA, ell_theta=ELL_THETA, theta_scale=THETA_SCALE,
    gamma_phi=GAMMA_PHI,     ell_phi=ELL_PHI,     phi_scale=PHI_SCALE
)

# Eigensolve (ARPACK) on K^{-1} H_GN
print("[info] building ARPACK operator and requesting k =", K_LEADING)
vals, modes = eigs_whitened_pair(Q, Hv_pair, Kθ_inv, Kφ_inv, k=K_LEADING)

# Save eigenvalues
out_eigs = pathlib.Path("../results/eigenvalues.txt")
out_eigs.parent.mkdir(parents=True, exist_ok=True)
with open(out_eigs, "w") as f:
    for i, lam in enumerate(vals):
        f.write(f"{i} {lam:.16e}\n")

# Save eigenmodes (chunked)
target_modes = save_modes_chunked(
    CKPT, mesh, modes, chunk=SAVE_CHUNK,
    same_file=SAVE_SAME_FILE,
    modes_path=CKPT.replace(".h5", "_modes.h5")
)

print(f"[OK] wrote {out_eigs.resolve()}")
print(f"[OK] saved {len(modes)} modes to {target_modes}")
