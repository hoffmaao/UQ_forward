import pathlib
import numpy as np
import firedrake as fd
from firedrake_adjoint import *    # Control, ReducedFunctional, tape utils
import icepack

# ---------------------------- Physics blocks ---------------------------------
def friction_stress(u, C, m):
    """Regularized Weertman stress: -C (|u|_reg)^(1/m - 1) u"""
    speed = fd.sqrt(fd.inner(u, u))
    return -C * speed ** (1.0 / m - 1.0) * u

def friction(**kwargs):
    """Weertman friction with θ=log(C/C0); masked by 'grounded' on shelf."""
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
    """Return surface elevation at flotation: s_float = b + (rho_w/rho_i)*max(0, -b)."""
    return bed + (rho_w / rho_i) * fd.max_value(-bed, 0.0)

def grounded_mask(surface, s_float, Q):
    """Smooth a bit, then build floating/grounded masks."""
    def _smooth(q0, alpha=2e3):
        q = q0.copy(deepcopy=True)
        J = 0.5 * ((q - q0)**2 + alpha**2 * fd.inner(fd.grad(q), fd.grad(q))) * fd.dx
        F = fd.derivative(J, q)
        fd.solve(F == 0, q)
        return q
    z = _smooth(fd.interpolate(surface - s_float, Q), alpha=100.0)
    floating = fd.interpolate(fd.conditional(z < 0, 1.0, 0.0), Q)
    grounded = fd.interpolate(1.0 - floating, Q)
    return floating, grounded

# ---------------------------- I/O ---------------------------------------------
def load_from_checkpoint(path, gamma):
    """Load Venable mesh & fields saved by your inversion (names follow run_inv.py)."""
    with fd.CheckpointFile(str(path), "r") as chk:
        mesh  = chk.load_mesh(name="venable")
        h0    = chk.load_function(mesh, name="thickness")
        s0    = chk.load_function(mesh, name="surface")
        b     = chk.load_function(mesh, name="bed")
        uobs  = chk.load_function(mesh, name="velocity_obs")
        sig   = chk.load_function(mesh, name="sigma_obs")      # (σx, σy) if needed later
        C0    = chk.load_function(mesh, name="friction")
        A0    = chk.load_function(mesh, name="fluidity")
        # MAP controls from inversion (for sensitivities)
        theta_map = chk.load_function(mesh, name=f"gamma{gamma:.2e}_log_friction")
        phi_map   = chk.load_function(mesh, name=f"gamma{gamma:.2e}_log_fluidity")
        # Optional: MAP velocity (not strictly needed for forward)
        try:
            u_map = chk.load_function(mesh, name=f"gamma{gamma:.2e}_velocity")
        except Exception:
            u_map = None
    V = uobs.function_space()
    Q = C0.function_space()
    return mesh, V, Q, h0, s0, b, uobs, sig, C0, A0, theta_map, phi_map, u_map

# ---------------------------- Solver ------------------------------------------
def build_solver_for_venable(mesh):
    """Icepack FlowSolver with Venable boundary partition: sidewalls clamped, front free."""
    model = icepack.models.IceStream(friction=friction, viscosity=viscosity)
    opts = {
        "dirichlet_ids": [1, 2, 4],                         # <-- clamp only these faces
        "diagnostic_solver_type": "petsc",
        "diagnostic_solver_parameters": {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-9,
            "snes_atol": 1e-10,
            "snes_max_it": 60,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        "prognostic_solver_parameters": {
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    }
    return icepack.solvers.FlowSolver(model, **opts)

# --------------------- QoI: Volume Above Flotation ----------------------------
def qoi_vaf_form(s, b, mesh, rho_i, rho_w):
    """Return UFL form for the total volume of ice above flotation."""
    s_float = flotation_surface_from_bed(b, rho_i, rho_w)
    haf = fd.max_value(s - s_float, 0.0)      # height above flotation (m)
    return haf * fd.dx(mesh)                   # integral → volume (m^3)

# --------------------------- Forward loop -------------------------------------
def run_forward_venable(ckpt_path="../mesh/venable.h5",
                        gamma=3.0,
                        T_years=20.0, nsteps=80,
                        outdir="../results/forward",
                        accum_rate=0.0):
    """
    Time‑step the thickness with the Icepack prognostic solver and record VAF.
    If `accum_control=True`, include a scalar α that scales the accumulation rate.
    """
    # I/O & constants
    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    mesh, V, Q, h0, s0, b, uobs, sig, C0, A0, theta_map, phi_map, u_map = load_from_checkpoint(ckpt_path, gamma)  # :contentReference[oaicite:3]{index=3}
    rho_i = icepack.constants.ice_density
    rho_w = icepack.constants.water_density

    # Time step
    dt_years = float(T_years) / float(nsteps)
    dt = fd.Constant(dt_years, domain=mesh)

    # Grounding mask
    s_float0 = flotation_surface_from_bed(b, rho_i, rho_w)
    floating, grounded = grounded_mask(s0, s_float0, Q)

    # Build solver (Venable BC partition)
    solver = build_solver_for_venable(mesh)  # :contentReference[oaicite:4]{index=4}

    # State & accumulation
    h = h0.copy(deepcopy=True)
    s = s0.copy(deepcopy=True)
    u = fd.Function(V, name="u0")
    # Seed diagnostic solve close to observed speeds to help Newton (optional)

    u.interpolate(uobs)


    # Accumulation: either a Constant or a spatial field; we scale by α if requested
    if isinstance(accum_rate, (int, float)):
        accum_template = fd.Constant(float(accum_rate), domain=mesh)
    elif isinstance(accum_rate, fd.Constant):
        accum_template = accum_rate
    elif isinstance(accum_rate, fd.Function):
        accum_template = accum_rate
    else:
        accum_template = fd.Constant(0.0, domain=mesh)

    accum = fd.Function(Q, name="accum")
    accum.interpolate(accum_rate)

    records = []

    # Prepare adjoint tape
    get_working_tape().clear_tape()
    continue_annotation()

    ctrl_theta = Control(theta_map)
    ctrl_phi   = Control(phi_map)

    # Forward march
    with fd.CheckpointFile(str(ckpt_path), "a") as chk:
        for k in range(1, nsteps + 1):
            # accumulation for this step

            # 1) diagnostic SSA
            u = solver.diagnostic_solve(
                velocity=u,
                thickness=h,
                surface=s,
                log_friction=theta_map,
                friction=C0,
                log_fluidity=phi_map,
                fluidity=A0,
                grounded=grounded,
                sliding_exponent=fd.Constant(1.0, domain=mesh),
            )

            # 2) prognostic thickness update
            h = solver.prognostic_solve(thickness=h, velocity=u, accumulation=accum, dt=dt)

            # 3) update surface s = b + h
            s = icepack.compute_surface(thickness=h, bed=b)

            #get_working_tape().clear_tape()
            #continue_annotation()
            with stop_annotating():
                floating, grounded = grounded_mask(s, s_float0, Q) 
            continue_annotation()

            print(f"we have stepped to {k}")

            J_form = qoi_vaf_form(s, b, mesh, rho_i, rho_w)
            J = fd.assemble(J_form)   # OverloadedFloat tracked by firedrake_adjoint

            # scalar VAF
            VAFk = float(J)

            # sensitivities wrt θ and φ
            rf_theta = ReducedFunctional(J, ctrl_theta)
            dJ_dtheta = rf_theta.derivative()
            gθ = fd.Function(Q, name=f"dVAF_dtheta_t{k:04d}"); gθ.interpolate(dJ_dtheta)

            rf_phi = ReducedFunctional(J, ctrl_phi)
            dJ_dphi = rf_phi.derivative()
            gφ = fd.Function(Q, name=f"dVAF_dphi_t{k:04d}"); gφ.interpolate(dJ_dphi)


            # save fields for this index
            chk.save_function(h, name="thickness_timeseries", idx=k)
            chk.save_function(u, name="velocity_timeseries",  idx=k)
            chk.save_function(gθ, name="dVAF_dtheta_timeseries", idx=k)
            chk.save_function(gφ, name="dVAF_dphi_timeseries",   idx=k)
            records.append((k, VAFk))
            #get_working_tape().clear_tape()
            #continue_annotation()

    # Write CSV
    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    ts = outdir / "vaf_timeseries.csv"
    with open(ts, "w") as f:
        f.write("step,VAF_m3\n")
        for k, VAFk in records:
            f.write(f"{k},{VAFk}\n")

    print(f"[OK] Wrote VAF time series to {ts}")
    return records


# ------------------------------ Defaults / run --------------------------------

# Checkpoint and control choice
CKPT  = "../mesh/venable.h5"
GAMMA = 3.0

# Time grid & output
T_YEARS = 20.0
NSTEPS  = 80
N_SENS  = 10
OUTDIR  = "../results/forward"

# Accumulation (set a scalar rate in m/yr, a Constant, or a Q-field)
ACCUM_RATE   = 0.0

# Run forward
run_forward_venable(ckpt_path=CKPT,
                    gamma=GAMMA,
                    T_years=T_YEARS, nsteps=NSTEPS,
                    outdir=OUTDIR,
                    accum_rate=ACCUM_RATE)
