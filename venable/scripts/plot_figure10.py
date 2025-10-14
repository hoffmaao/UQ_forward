#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure-10-like subpanels for the Venable experiment:
(A) posterior SD map of alpha (Weertman law in your setup),
(B) prior vs posterior 2σ(VAF) through time,
(C) marginal change δσ at the final time vs # modes, with an exponential tail fit,
(D) VAF trajectory Q_T - Q_0.

Inputs expected (as written by your existing scripts):
  - ../mesh/venable.h5             (mesh + eigenmodes + sensitivities)
  - ../results/eigenvalues.txt     (whitened eigenvalues for K^{-1}H)
  - ../results/forward/vaf_timeseries.csv

This script uses the same prior and inner products as your eigendec & forward pipeline.
"""

import pathlib, math, numpy as np
import matplotlib.pyplot as plt
import firedrake as fd

# ----------------------------- Paths & knobs ---------------------------------
CKPT     = "../mesh/venable.h5"
EIG_TXT  = "../results/eigenvalues.txt"
VAF_CSV  = "../results/forward/vaf_timeseries.csv"
FIG_PNG  = "../figures/fig10_like_subpanels.png"

# Leading modes to include (None = all in EIG_TXT)
N_MODES = None

# Prior hyperparameters (must match your eigendec build!)
GAMMA_THETA = 3.0     # friction prior strength
ELL_THETA   = 7.5e3   # m
THETA_SCALE = 1.0

GAMMA_PHI   = 1.0     # fluidity prior strength
ELL_PHI     = 7.5e3   # m
PHI_SCALE   = 1.0

# Panel-A map cost control: compute SD on every 'stride'-th DOF (then smooth).
STRIDE = 5    # increase to speed up, decrease for higher fidelity
SMOOTH_ALPHA = 3e3  # m; smoothing length for filling dof gaps in the SD map

# -------------------------- Load mesh & spaces -------------------------------
with fd.CheckpointFile(str(CKPT), "r") as chk:
    mesh  = chk.load_mesh(name="venable")
    C0    = chk.load_function(mesh, name="friction")
    A0    = chk.load_function(mesh, name="fluidity")
    theta_map = chk.load_function(mesh, name="gamma3.00e+00_log_friction")
    phi_map   = chk.load_function(mesh, name="gamma3.00e+00_log_fluidity")
Q = C0.function_space()
QQ = fd.MixedFunctionSpace([Q, Q])

# Domain area normaliser used consistently (as in eigendec)
A = float(fd.assemble(fd.Constant(1.0) * fd.dx(mesh)))
invA = fd.Constant(1.0 / A, domain=mesh)

# -------------------------- Read eigenpairs ---------------------------------
vals_all = np.loadtxt(EIG_TXT)
lam = np.atleast_1d(vals_all[:,1]) if vals_all.ndim > 1 else np.array([float(vals_all[1])])
if N_MODES is not None:
    lam = lam[:N_MODES]
r = len(lam)

modes = []
with fd.CheckpointFile(str(CKPT), "r") as chk:
    for i in range(r):
        modes.append(chk.load_function(mesh, name=f"mode_{i:04d}"))  # MixedFunction [Q,Q]

# -------------------------- K^{-1} solvers ----------------------------------
def make_Kinv(gamma, ell, scale):
    y = fd.Function(Q); z = fd.Function(Q)
    v, w = fd.TrialFunction(Q), fd.TestFunction(Q)
    a = ( invA*fd.inner(v,w) + gamma*(ell/scale)**2*invA*fd.inner(fd.grad(v), fd.grad(w)) )*fd.dx(mesh)
    L = invA*fd.inner(z, w)*fd.dx(mesh)
    pr = fd.LinearVariationalProblem(a, L, y)
    sp = {"ksp_type":"cg", "ksp_rtol":1e-10, "pc_type":"hypre", "pc_hypre_type":"boomeramg"}
    so = fd.LinearVariationalSolver(pr, solver_parameters=sp)
    def apply(rhs):
        z.assign(rhs); so.solve(); return y.copy(deepcopy=True)
    return apply

Kθ_inv = make_Kinv(fd.Constant(GAMMA_THETA, domain=mesh),
                   fd.Constant(ELL_THETA,   domain=mesh),
                   fd.Constant(THETA_SCALE, domain=mesh))
Kφ_inv = make_Kinv(fd.Constant(GAMMA_PHI,   domain=mesh),
                   fd.Constant(ELL_PHI,     domain=mesh),
                   fd.Constant(PHI_SCALE,   domain=mesh))

def inner(a, b):  # L2 inner product
    return float(fd.assemble(a*b*fd.dx(mesh)))

# ------------------------- Panel (B) & (C): VAF σ(t) and δσ ------------------
# Read time series (step,VAF)
steps, VAF = [], []
for line in pathlib.Path(VAF_CSV).read_text().splitlines()[1:]:
    k, v = line.split(","); steps.append(int(k)); VAF.append(float(v))
steps = np.array(steps, dtype=int)
VAF   = np.array(VAF, dtype=float)
t_years = np.linspace(0.0, 20.0, steps.max())[:steps.size]  # adjust if your forward used a different grid

# Pre-load y·mode inner products for each step (cheap once K^{-1}g is computed)
sigma_prior_t, sigma_post_t = [], []
coeffs_by_step = []   # for panel (C) at final time
sigmas = []

with fd.CheckpointFile(str(CKPT), "r") as chk:
    for k in steps:
        gθ = chk.load_function(mesh, name="dVAF_dtheta_timeseries", idx=int(k))
        gφ = chk.load_function(mesh, name="dVAF_dphi_timeseries",   idx=int(k))

        yθ = Kθ_inv(gθ)
        yφ = Kφ_inv(gφ)

        prior_var = inner(gθ, yθ) + inner(gφ, yφ)    # g^T K^{-1} g
        # whitened coefficients c_i = <K^{-1}g, mode_i>
        ci = np.array([ inner(yθ, z.sub(0)) + inner(yφ, z.sub(1)) for z in modes ])
        # posterior variance using exact identity:
        #   var_post = var_prior - sum_i [ c_i^2 * lambda_i/(1+lambda_i) ]
        red = np.sum( (ci**2) * (lam/(1.0+lam)) )
        post_var = max(prior_var - red, 0.0)

        sigma_prior_t.append(math.sqrt(max(prior_var,0.0)))
        sigma_post_t.append(math.sqrt(post_var))


        if k == steps.max():
            coeffs_by_step = ci  # store for panel (C)

        # Projections in the K^{-1}-weighted inner product
        coeffs = []
        for i, z in enumerate(modes):
            ci = fd.assemble( (yθ * z.sub(0) + yφ * z.sub(1)) * fd.dx(mesh) )
            coeffs.append(float(ci))
        coeffs = np.asarray(coeffs)
        var_k = np.sum( (coeffs**2) / (1.0 + lam) )
        sigmas.append(np.sqrt(max(var_k, 0.0)))

sigmas = np.array(sigmas)
sigma_prior_t = np.array(sigma_prior_t)
sigma_post_t  = np.array(sigma_post_t)

# Build δσ curve at final time (use positive magnitude, like Fig. 10f)
sigma0 = float(sigma_prior_t[-1])
red_increments = (coeffs_by_step**2) * (lam/(1.0+lam))   # contributions to variance reduction
var_r = sigma0**2 - np.cumsum(red_increments)            # posterior variance after r modes
sigma_r = np.sqrt(np.maximum(var_r, 0.0))
delta_sigma = np.empty_like(sigma_r)
delta_sigma[0] = sigma_r[0] - sigma0
delta_sigma[1:] = sigma_r[1:] - sigma_r[:-1]
delta_sigma = np.abs(delta_sigma)  # show positive “rate of change”

# Exponential tail fit (log-linear) for δσ_r ≈ a exp(br)
mask = delta_sigma > 0
x = np.arange(1, delta_sigma.size+1)[mask]
y = delta_sigma[mask]
if y.size >= 3:
    p = np.polyfit(x, np.log(y), 1)  # log y ~ p[0]*x + p[1]
    b, a_ln = p[0], p[1]
    y_fit = np.exp(a_ln + b*x)
    # crude σ∞ estimate (geometric tail of δσ):
    ratio = math.exp(b)
    tail = y[-1] * (ratio / (1.0 - ratio)) if ratio < 1.0 else 0.0
    sigma_inf = sigma_r[-1] + tail
    r2 = 1.0 - np.sum((np.log(y) - (a_ln + b*x))**2)/np.sum((np.log(y)-np.mean(np.log(y)))**2)
else:
    y_fit = None; sigma_inf = np.nan; r2 = np.nan

# ------------------------- Panel (A): posterior SD map of alpha ---------------
# We approximate pointwise SD via “local average” patches on P1 DOFs,
# using the same low-rank identity as above:
#   var_post(〈θ〉_patch) = var_prior - sum_i (c_i^2 * λ_i/(1+λ_i)),
# where c_i = <K^{-1} g_patch, v_i^θ> and g_patch = φ_j / ∫φ_j (P1 nodal patch).
alpha_map = fd.Function(Q); alpha_map.interpolate(C0*fd.exp(theta_map))

sd_alpha = fd.Function(Q, name="std_alpha")
sd_alpha.assign(0.0)

dofs = np.arange(Q.dof_dset.size)
sel  = dofs[::max(1, STRIDE)]
phi = fd.Function(Q)
for j in sel:
    # nodal basis function φ_j (set coefficient 1 at dof j)
    phi.dat.data[:] = 0.0
    phi.dat.data[j] = 1.0
    mj = float(fd.assemble(phi * fd.dx(mesh)))
    if mj == 0.0:  # safety
        continue
    gpatch = fd.Function(Q); gpatch.assign(phi / mj)

    yj = Kθ_inv(gpatch)
    prior_var = inner(gpatch, yj)

    cj = np.array([ inner(yj, z.sub(0)) for z in modes ])
    red = np.sum( (cj**2) * (lam/(1.0+lam)) )
    post_var = max(prior_var - red, 0.0)

    # std of theta-average on patch; propagate to alpha via linearization
    theta_sd = math.sqrt(post_var)
    sd_alpha.dat.data[j] = float(alpha_map.dat.data_ro[j]) * theta_sd

# Smooth-fill uncomputed DOFs for a clean map
def smooth_fill(q0, alpha=SMOOTH_ALPHA):
    q = q0.copy(deepcopy=True)
    J = 0.5*((q-q0)**2 + alpha**2*fd.inner(fd.grad(q), fd.grad(q)))*fd.dx
    F = fd.derivative(J, q)
    fd.solve(F == 0, q)
    return q
sd_alpha_full = smooth_fill(sd_alpha, alpha=SMOOTH_ALPHA)

# ----------------------------- Plot (2×2) ------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(11, 7.2))
# (A) SD(alpha) map
axA = ax[0,0]
tpc = fd.tripcolor(sd_alpha_full, axes=axA, shading='gouraud', cmap='magma')
cA = fig.colorbar(tpc, ax=axA, orientation='vertical', fraction=0.046, pad=0.04)
cA.set_label('posterior SD of α')
axA.set_title(r'SD of log$_{10} \theta$ sliding parameter')
axA.set_aspect('equal'); axA.set_xticks([]); axA.set_yticks([])

# (B) prior vs posterior 2σ (time)
axB = ax[0,1]
axB.plot(t_years[1:], 2*sigma_prior_t[1:], '--', lw=1.8, label='prior 2σ')
axB.plot(t_years[1:], 2*sigma_post_t[1:],  '-', lw=2.2,  label='posterior 2σ')
axB.set_xlabel('time (years)'); axB.set_ylabel('2σ(VAF) [m³]')
axB.grid(alpha=0.2); axB.legend(frameon=False)
axB.set_title('Prior vs posterior 2σ of VAF')

# (C) δσ vs # modes (final time)
axC = ax[1,0]
axC.plot(np.arange(1, delta_sigma.size+1), delta_sigma, lw=1.6)
if y_fit is not None:
    axC.plot(x, y_fit, ':', lw=2.0, label='exp. fit')
    axC.legend(frameon=False)
txt = f"σ_T={sigma_r[-1]:.3e}  σ_∞≈{sigma_inf:.3e}  r²={r2:.2f}"
axC.text(0.02, 0.95, txt, transform=axC.transAxes, va='top')
axC.set_xlabel('eigenvalue index r'); axC.set_ylabel('Δσ_T (per added mode)')
axC.set_yscale('log'); axC.grid(alpha=0.2, which='both')
axC.set_title('Marginal change in σ(VAF) at T')

# (D) VAF trajectory
axD = ax[1,1]
axD.plot(t_years[1:], VAF[1:], lw=2.0, color="k", label="VAF (MAP)")
axD.fill_between(t_years[1:], VAF[1:] - sigmas[1:], VAF[1:] + sigmas[1:], color="tab:blue", alpha=0.25, label="±1σ")
axD.fill_between(t_years[1:], VAF[1:] - 2*sigmas[1:], VAF[1:] + 2*sigmas[1:], color="tab:blue", alpha=0.12, label="±2σ")

axD.set_xlabel('time (years)'); axD.set_ylabel(r'$Q_T - Q_0$ [m$^3$]')
axD.grid(alpha=0.2)
axD.set_title('VAF trajectory (context)')
fig.tight_layout()
pathlib.Path(FIG_PNG).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_PNG, dpi=300);
print(f"[OK] wrote {FIG_PNG}")
