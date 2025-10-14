import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import firedrake as fd

EIG_DIR = pathlib.Path("../results/eigendec_tlm")     # <-- adjust if needed
H5_FILE = EIG_DIR / "gn_eigs.h5"
EVALS   = EIG_DIR / "eigenvalues.txt"
PLOTS   = "../figures/"
PLOTS.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1) Eigenvalue spectrum (semilogy)
# -----------------------------
# eigenvalues.txt format: "<index> <value>"
idx, lam = np.loadtxt(EVALS, unpack=True)
# In Gauss–Newton + prior we expect nonnegative lambdas; tiny negatives can appear from numerical error.
pos = lam > 0
neg = lam < 0

plt.figure(figsize=(7,4.5))
plt.semilogy(np.nonzero(pos)[0], lam[pos], ".", label="positive")
if np.any(neg):
    plt.semilogy(np.nonzero(neg)[0], -lam[neg], ".", label="|negative|")
plt.xlabel("mode index")
plt.ylabel("eigenvalue magnitude")
plt.title("GN Hessian eigenspectrum")
plt.grid(True, which="both", lw=0.5, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS / "eigenvalues_semilogy.png", dpi=220)
plt.close()

# -----------------------------
# 2) Load eigenfunctions from HDF5 checkpoint
# -----------------------------
with fd.CheckpointFile(str(H5_FILE), "r") as chk:
    # If only one mesh was saved, load_mesh() with no name is fine.
    mesh = chk.load_mesh(name="ismip-c")
    phis = []
    i = 0
    while True:
        name = f"phi_{i:04d}"
        try:
            f = chk.load_function(mesh, name=name)
            phis.append(f)
            i += 1
        except Exception:
            break

print(f"Loaded {len(phis)} eigenfunctions from {H5_FILE}")

# -----------------------------
# 3) Grid plot of the leading modes
# -----------------------------
from firedrake.pyplot import tripcolor

n_show = min(9, len(phis))           # show first 9 by default
cols = 3
rows = int(np.ceil(n_show/cols))
fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows), constrained_layout=True)
axes = np.asarray(axes).reshape(rows, cols)

# Empty panels off
for ax in axes.flat[n_show:]:
    ax.axis("off")

for k in range(n_show):
    ax = axes.flat[k]
    c = tripcolor(phis[k], axes=ax)  # Firedrake helper for triangulated plots
    ax.set_aspect("equal")
    ax.set_title(f"mode {k}")
    fig.colorbar(c, ax=ax, shrink=0.8)

fig.suptitle("Leading eigenfunctions (Gauss–Newton Hessian modes)", y=1.02)
fig.savefig(PLOTS / "leading_modes_grid.png", dpi=220)
plt.close(fig)

# -----------------------------
# 4) ParaView-friendly time series (scroll through modes)
# -----------------------------
pvd = fd.File(str(EIG_DIR / "eigenmodes.pvd"))
for k, f in enumerate(phis[:min(40, len(phis))]):   # write first 40 as a quick-look series
    f.rename("phi")
    pvd.write(f, time=float(k))

print(f"✓ Wrote plots to: {PLOTS}")
print(f"✓ Wrote ParaView series: {EIG_DIR / 'eigenmodes.pvd'} (use time slider to browse modes)")
