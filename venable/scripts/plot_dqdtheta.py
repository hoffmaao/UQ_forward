import pathlib
import numpy as np
import firedrake as fd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
CKPT = "../mesh/venable.h5"
TITLE = "Venable: sensitivity of VAF"
STEPS = [20, 40, 60, 80]    # time-step indices (1-based)
T_YEARS = 20.0              # total simulated years (for labels)
N_STEPS = 80                # total number of steps
CMAP = "RdBu_r"
ROBUST_Q = 0.995            # robust symmetric limits per row
FIGOUT = "../figures/fig_A3_like_horizontal_rightcbar.png"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_mesh_and_Q(path):
    with fd.CheckpointFile(str(path), "r") as chk:
        mesh = chk.load_mesh(name="venable")
        C0   = chk.load_function(mesh, name="friction")  # to get Q
    return mesh, C0.function_space()

def load_field(mesh, name, k):
    with fd.CheckpointFile(CKPT, "r") as chk:
        return chk.load_function(mesh, name=name, idx=int(k))

def robust_sym_limits(funcs, q=ROBUST_Q):
    data = np.concatenate([f.dat.data_ro for f in funcs]) if funcs else np.array([1.0])
    data = data[np.isfinite(data)]
    vmax = np.quantile(np.abs(data), q) if data.size else 1.0
    return -vmax, vmax

# ---------------------------------------------------------------------
# Load data & determine per-row color scales
# ---------------------------------------------------------------------
mesh, Q = load_mesh_and_Q(CKPT)

gtheta = [load_field(mesh, "dVAF_dtheta_timeseries", k) for k in STEPS]
gphi   = [load_field(mesh, "dVAF_dphi_timeseries",   k) for k in STEPS]

vmin_th, vmax_th = robust_sym_limits(gtheta)
vmin_ph, vmax_ph = robust_sym_limits(gphi)
norm_th = mcolors.TwoSlopeNorm(vmin=vmin_th, vcenter=0.0, vmax=vmax_th)
norm_ph = mcolors.TwoSlopeNorm(vmin=vmin_ph, vcenter=0.0, vmax=vmax_ph)

# ---------------------------------------------------------------------
# Figure: 2 rows × 4 columns, vertical colorbars to the right
# ---------------------------------------------------------------------
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6.2), constrained_layout=False)
fig.suptitle(TITLE, y=0.995, fontsize=12)

# Top row: ∂VAF/∂θ
for j, k in enumerate(STEPS):
    ax = axes[0, j]
    m = fd.tripcolor(gtheta[j], axes=ax, cmap=CMAP, norm=norm_th)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    t_years = (k / float(N_STEPS)) * float(T_YEARS)
    ax.set_title(f"t = {t_years:.1f} yr — ∂VAF/∂θ", fontsize=10)

# Bottom row: ∂VAF/∂φ
for j, k in enumerate(STEPS):
    ax = axes[1, j]
    m = fd.tripcolor(gphi[j], axes=ax, cmap=CMAP, norm=norm_ph)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    t_years = (k / float(N_STEPS)) * float(T_YEARS)
    ax.set_title(f"t = {t_years:.1f} yr — ∂VAF/∂φ", fontsize=10)

# --- Vertical colorbars to the RIGHT of the right-most panels ---
# Use ScalarMappable so both rows share the same per-row norm
sm_th = plt.cm.ScalarMappable(norm=norm_th, cmap=CMAP)
sm_ph = plt.cm.ScalarMappable(norm=norm_ph, cmap=CMAP)

# Top-row colorbar (to the right of axes[0, -1])
div0 = make_axes_locatable(axes[0, -1])
cax0 = div0.append_axes("right", size="3%", pad=0.05)  # width & gap
cb0  = fig.colorbar(sm_th, cax=cax0, orientation="vertical")
cb0.set_label(r"$\partial \mathrm{VAF}/\partial\theta$  (m$^3$ per unit $\log C$)", fontsize=9)

# Bottom-row colorbar (to the right of axes[1, -1])
div1 = make_axes_locatable(axes[1, -1])
cax1 = div1.append_axes("right", size="3%", pad=0.05)
cb1  = fig.colorbar(sm_ph, cax=cax1, orientation="vertical")
cb1.set_label(r"$\partial \mathrm{VAF}/\partial\phi$  (m$^3$ per unit $\log A$)", fontsize=9)

# Tidy subplot spacing (gives room for the right-side colorbars)
plt.subplots_adjust(wspace=0.05, hspace=0.25, right=0.92)

# Save
pathlib.Path(FIGOUT).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGOUT, dpi=300)
print(f"[OK] wrote {FIGOUT}")
