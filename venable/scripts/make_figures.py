import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import geojson
import rasterio
import pygmsh
import firedrake
from firedrake import Constant, sqrt, inner, grad, dx
import icepack
from icepack.constants import (
    ice_density as ρ_I, water_density as ρ_W, gravity as g, weertman_sliding_law as m
)


def load_from_checkpoint(chk_path,gamma=3.00e-01):
    with firedrake.CheckpointFile(chk_path, "r") as chk:
        mesh = chk.load_mesh(name="venable")
        h    = chk.load_function(mesh, name="thickness")
        s    = chk.load_function(mesh, name="surface")
        b    = chk.load_function(mesh, name="bed")
        u_obs = chk.load_function(mesh, name="velocity_obs")
        σ_obs = chk.load_function(mesh, name="sigma_obs")
        C0 = chk.load_function(mesh, name="friction")
        A0 = chk.load_function(mesh, name="fluidity")
        θ = chk.load_function(mesh, name=f"gamma{g:.2e}_log_friction")
        φ = chk.load_function(mesh, name=f"gamma{g:.2e}_log_fluidity")

    V = u_obs.function_space()
    Q = C0.function_space()
    C = firedrake.Function(Q, name=f"gamma{g:.2e}_friction"); C.interpolate(C0*firedrake.exp(θ))
    A = firedrake.Function(Q, name=f"gamma{g:.2e}_fluidity"); A.interpolate(A0*firedrake.exp(φ))
    return mesh, V, Q, h, s, b, u_obs, σ_obs, C0, A0, θ, φ, A, C

def subplots(*args, **kwargs):
    fig, axes = plt.subplots()
    axes.set_aspect("equal")
    xmin, ymin, xmax, ymax = rasterio.windows.bounds(window, transform)
    axes.imshow(
        image,
        cmap="Greys_r",
        vmin=12e3,
        vmax=19.38e3,
        extent=(xmin, xmax, ymin, ymax),
    )
    axes.tick_params(labelrotation=25)

    return fig, axes

g = 3.00e01

mesh, V, Q, h, s, b, u_obs, σ_obs, C0, A0, θ, φ, A, C = load_from_checkpoint("../mesh/venable.h5", gamma = g)

outline_filename = "../mesh/venable.geojson"
with open(outline_filename, "r") as outline_file:
    outline = geojson.load(outline_file)

coords = np.array(list(geojson.utils.coords(outline)))
delta = 30e3
xmin, xmax = coords[:, 0].min() - delta, coords[:, 0].max() + delta
ymin, ymax = coords[:, 1].min() - delta, coords[:, 1].max() + delta


image_filename = icepack.datasets.fetch_mosaic_of_antarctica()
with rasterio.open(image_filename, "r") as image_file:
    height, width = image_file.height, image_file.width
    transform = image_file.transform
    window = rasterio.windows.from_bounds(
        left=xmin,
        bottom=ymin,
        right=xmax,
        top=ymax,
        transform=transform,
    )
    image = image_file.read(indexes=1, window=window, masked=True)



fig, axes = subplots()
colors = firedrake.tripcolor(C, axes=axes,alpha=.5)
fig.colorbar(colors);
fig.savefig(f'../figures/friction_gamma{g:.2e}.png')

fig, axes = subplots()
colors = firedrake.tripcolor(A, axes=axes,alpha=.5)
fig.colorbar(colors);
fig.savefig(f'../figures/fluidity_gamma{g:.2e}.png')