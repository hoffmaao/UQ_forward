import pathlib
import numpy as np
import firedrake as fd
import icepack

# --- tlm_adjoint (Firedrake backend) ---
try:
    from tlm_adjoint.firedrake import Functional, CachedHessian, reset_manager, start_manager, stop_manager
    # We don't need deps/BC-unpacking because we avoid DirichletBC entirely.
    HAVE_TLM = True
except Exception as exc:
    HAVE_TLM = False
    _TLM_ERR = exc

# --- SciPy (ARPACK) ---
try:
    from scipy.sparse.linalg import LinearOperator, eigsh
    HAVE_SCIPY = True
except Exception as exc:
    HAVE_SCIPY = False
    _SCIPY_ERR = exc


# ============================ Physics blocks =================================
def friction_stress(u, C, m, u_reg=fd.Constant(1e-6)):
    """Regularized Weertman stress to avoid 0^0 when |u|≈0."""
    speed = fd.sqrt(fd.inner(u, u) + u_reg**2)
    return -C * speed ** (1.0 / m - 1.0) * u

def friction(**kwargs):
    """Weertman basal friction with theta = log(C/C0); zeroed on shelf via 'grounded'."""
    u       = kwargs["velocity"]
    theta   = kwargs["log_friction"]
    mexp    = kwargs.get("sliding_exponent", fd.Constant(1.0))
    C0      = kwargs["friction"]
    grounded= kwargs.get("grounded", None)
    C = C0 * fd.exp(theta)
    if grounded is not None:
        C = C * grounded
    tau = friction_stress(u, C, mexp)
    return -mexp / (mexp + 1.0) * fd.inner(tau, u)

def viscosity(**kwargs):
    A0  = kwargs["fluidity"]
    u   = kwargs["velocity"]
    h   = kwargs["thickness"]
    phi = kwargs["log_fluidity"]
    A   = A0 * fd.exp(phi)
    return icepack.models.viscosity.viscosity_depth_averaged(
        velocity=u, thickness=h, fluidity=A
    )


# ============================ I/O & masks ====================================
def load_from_checkpoint(path, gamma):
    """Load Venable mesh + fields written by your run_inv.py."""
    with fd.CheckpointFile(str(path), "r") as chk:
        mesh  = chk.load_mesh(name="venable")
        h     = chk.load_function(mesh, name="thickness")
        s     = chk.load_function(mesh, name="surface")
        b     = chk.load_function(mesh, name="bed")
        uobs  = chk.load_function(mesh, name="velocity_obs")
        sigma = chk.load_function(mesh, name="sigma_obs")
        C0    = chk.load_function(mesh, name="friction")
        A0    = chk.load_function(mesh, name="fluidity")
        # MAP fields from inversion
        u_map = chk.load_function(mesh, name=f"gamma{gamma:.2e}_velocity")
        th_map= chk.load_function(mesh, name=f"gamma{gamma:.2e}_log_friction")
        ph_map= chk.load_function(mesh, name=f"gamma{gamma:.2e}_log_fluidity")
    V = uobs.function_space()
    Q = C0.function_space()
    return mesh, V, Q, h, s, b, uobs, sigma, C0, A0, th_map, ph_map

def flotation_surface(bed, rho_i, rho_w):
    # s_float = b + h_float, with h_float = (rho_w/rho_i) * max(0, -b)
    return bed + (rho_w/rho_i) * fd.max_value(-bed, 0.0)

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


# ===================== Forward: soft‑Dirichlet sidewalls only =================
def make_forward_soft_dirichlet_venable(
    *, model, mesh, V, h, s, C0, A0, grounded,
    dirichlet_ids, g_value,
    penalty=1e3,                       # increase if boundary mismatch is too loose
    terminus_ids=None, include_ocean_pressure=False,
    newton_sp=None
):
    """
    forward(theta, phi) -> u
    Adds a boundary penalty 0.5*γ_D * ∫_{Γ_D} |u - g_value|^2 ds on sidewalls (Γ_D).
    No DirichletBC objects → tlm_adjoint never unpacks BCs (avoids 'Invalid dependency').
    Calving front facets (terminus_ids) remain natural; optional simple hydrostatic load.
    """
    if newton_sp is None:
        newton_sp = {
            "snes_type": "newtonls", "snes_linesearch_type": "bt",
            "snes_rtol": 1e-8, "snes_atol": 1e-8, "snes_max_it": 60,
            "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
        }

    # Ensure boundary value lives on V
    bc_val = g_value
    if isinstance(g_value, fd.Function) and (g_value.function_space() != V):
        bc_val = fd.project(g_value, V)

    γD  = fd.Constant(float(penalty), domain=mesh)
    n   = fd.FacetNormal(mesh)
    rho_w = icepack.constants.water_density
    ggrav = icepack.constants.gravity

    def forward(theta, phi):
        u = fd.Function(V, name="u")
        v = fd.TestFunction(V)


        u.assign(bc_val)

        # Core Icepack action
        E_core = model.action(
            velocity=u, thickness=h, surface=s,
            log_friction=theta, friction=C0,
            log_fluidity=phi,  fluidity=A0,
            grounded=grounded
        )

        # Soft Dirichlet penalty on sidewalls only
        E_bc = 0
        for bid in dirichlet_ids:
            E_bc += 0.5 * γD * fd.inner(u - bc_val, u - bc_val) * fd.ds(int(bid), domain=mesh)

        # Optional simple hydrostatic ocean pressure at the calving front
        E_front = 0
        if include_ocean_pressure and terminus_ids:
            # depth‑integrated traction model: t ≈ (1/2) ρ_w g h^2 n
            p_front = 0.5 * rho_w * ggrav * (h**2)
            for bid in terminus_ids:
                # external work:  -∫ t·u ds
                E_front -= p_front * fd.inner(u, n) * fd.ds(int(bid), domain=mesh)

        E = E_core + E_bc + E_front
        F = fd.derivative(E, u, v)

        # Solve (no DirichletBC objects involved)
        fd.solve(F == 0, u, solver_parameters=newton_sp)
        return u

    return forward


# =========================== GN Hessian–vector (H·v) ==========================
def make_Hv_gn(forward_u, *, mesh, invA, sigma_vec, theta_map, phi_map):
    """Gauss–Newton H·v via zero‑residual trick: exact Jᵀ W J."""
    if not HAVE_TLM:
        raise RuntimeError(f"tlm_adjoint not available: {_TLM_ERR}")

    reset_manager()  # off-tape MAP solve
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

    start_manager()
    J_map = record(theta_map, phi_map)
    stop_manager()

    H = CachedHessian(J_map)

    def Hv_pair(vθ, vφ):
        _, _, ddJ = H.action([theta_map, phi_map], [vθ, vφ])
        return ddJ[0].riesz_representation("L2"), ddJ[1].riesz_representation("L2")

    return Hv_pair


# =============================== Prior K^{-1} =================================
def make_prior_solve_components(mesh, Q, invA,
                                gamma_theta, ell_theta, theta_scale,
                                gamma_phi,   ell_phi,   phi_scale):
    def one_solver(gamma, ell, scale):
        y = fd.Function(Q); z = fd.Function(Q)
        v, w = fd.TrialFunction(Q), fd.TestFunction(Q)
        a = ( invA*fd.inner(v, w)
              + gamma*(ell/scale)**2 * invA * fd.inner(fd.grad(v), fd.grad(w)) ) * fd.dx(mesh)
        L = invA * fd.inner(z, w) * fd.dx(mesh)
        pr = fd.LinearVariationalProblem(a, L, y)
        sp = {"ksp_type":"cg","ksp_rtol":1e-12,
      "pc_type":"hypre","pc_hypre_type":"boomeramg"}

        so = fd.LinearVariationalSolver(pr, solver_parameters=sp)
        def apply(rhs):
            z.assign(rhs); so.solve(); return y.copy(deepcopy=True)
        return apply
    Kθ_inv = one_solver(fd.Constant(gamma_theta, domain=mesh),
                        fd.Constant(ell_theta,   domain=mesh),
                        fd.Constant(theta_scale, domain=mesh))
    Kφ_inv = one_solver(fd.Constant(gamma_phi,   domain=mesh),
                        fd.Constant(ell_phi,     domain=mesh),
                        fd.Constant(phi_scale,   domain=mesh))
    return Kθ_inv, Kφ_inv


# ===================== ARPACK eigensolve for K^{-1}H =========================
def eigs_whitened_pair(Q, Hv_pair, Kθ_inv, Kφ_inv, k=40):
    if not HAVE_SCIPY:
        raise RuntimeError(f"SciPy eigsh required: {_SCIPY_ERR}")
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

    A = LinearOperator((2*n, 2*n), matvec=matvec, rmatvec=matvec, dtype=float)
    vals, vecs = eigsh(A, k=min(k, 2*n-2), which="LA", tol=1e-12, maxiter=10000)

    # sort by descending value
    idx = np.argsort(-vals); vals = vals[idx]; vecs = vecs[:, idx]

    # pack modes as MixedFunctions on Q×Q
    QQ = fd.MixedFunctionSpace([Q, Q])
    modes = []
    for i in range(vals.size):
        z = fd.Function(QQ)
        z.sub(0).dat.data[:] = vecs[:n, i]
        z.sub(1).dat.data[:] = vecs[n:, i]
        modes.append(z)
    return vals, modes


# --------------------------------- Config ------------------------------------
CKPT      = "../mesh/venable.h5"
GAMMA     = 3.0
K_LEADING = 400

# Prior hyperparameters (match your inversion scales)
GAMMA_THETA = 3.0
ELL_THETA   = 7.5e3
THETA_SCALE = 1.0
GAMMA_PHI   = 1.0
ELL_PHI     = 7.5e3
PHI_SCALE   = 1.0


# -------------------------------- Pipeline -----------------------------------
# Load Venable checkpoint & MAP fields (names as in your run_inv.py).
mesh, V, Q, h, s, b, uobs, sigma_vec, C0, A0, theta_map, phi_map = load_from_checkpoint(CKPT, GAMMA) # noqa
# (This mirrors your inversion outputs and naming.) :contentReference[oaicite:1]{index=1}

# Model + grounding
model = icepack.models.IceStream(friction=friction, viscosity=viscosity)
rho_i = icepack.constants.ice_density
rho_w = icepack.constants.water_density
s_float = flotation_surface(b, rho_i, rho_w)
_, grounded = grounded_mask(s, s_float, Q)

# Boundary partition: sidewalls clamped (as in your Venable inversion); front is complement.
DIRICHLET_IDS = [1, 2, 4]  # <- same as run_inv.py for Venable. :contentReference[oaicite:2]{index=2}
ALL_MARKERS   = list(mesh.exterior_facets.unique_markers)
TERMINUS_IDS  = [bid for bid in ALL_MARKERS if bid not in DIRICHLET_IDS]

# Build forward: soft Dirichlet on sidewalls; calving front is natural (optionally add ocean load).
forward_u = make_forward_soft_dirichlet_venable(
    model=model, mesh=mesh, V=V, h=h, s=s, C0=C0, A0=A0, grounded=grounded,
    dirichlet_ids=DIRICHLET_IDS, g_value=uobs,
    terminus_ids=TERMINUS_IDS, include_ocean_pressure=False,
    penalty=1e3
)

# Area normalization for GN misfit
area = float(fd.assemble(fd.Constant(1.0) * fd.dx(mesh)))
invA  = fd.Constant(1.0 / area, domain=mesh)

# GN Hessian action at MAP
Hv_pair = make_Hv_gn(forward_u, mesh=mesh, invA=invA, sigma_vec=sigma_vec,
                     theta_map=theta_map, phi_map=phi_map)

# Prior K^{-1} for both controls
Kθ_inv, Kφ_inv = make_prior_solve_components(
    mesh, Q, invA,
    gamma_theta=GAMMA_THETA, ell_theta=ELL_THETA, theta_scale=THETA_SCALE,
    gamma_phi=GAMMA_PHI,     ell_phi=ELL_PHI,     phi_scale=PHI_SCALE
)

# Eigensolve on K^{-1} H_GN
vals, modes = eigs_whitened_pair(Q, Hv_pair, Kθ_inv, Kφ_inv, k=K_LEADING)

# Save eigenmodes & eigenvalues
with fd.CheckpointFile(CKPT, "a") as chk:
    for i, mode in enumerate(modes):
        mode.rename(f"mode_{i:04d}")
        chk.save_function(mode)

out = pathlib.Path("../results/eigenvalues.txt")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    for i, lam in enumerate(vals):
        f.write(f"{i} {lam:.16e}\n")

print(f"[OK] Saved {len(modes)} modes to {CKPT}")
print(f"[OK] Wrote eigenvalues to {out.resolve()}")
