import argparse, pathlib, math
import numpy as np
import firedrake
from firedrake import assemble, Constant, inner, grad, dx
from firedrake_adjoint import *
import icepack
from icepack.statistics import StatisticsProblem, MaximumProbabilityEstimator
import matplotlib.pyplot as plt




def load_from_checkpoint(chk_path):
    # NOTE: Checkpointed meshes may not retain periodic identifications;
    # if you rely on periodic-x, rebuild the mesh separately and load fields onto it.
    with firedrake.CheckpointFile(chk_path, "r") as chk:
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

mesh, V, Q, U, h, s, b, u_obs, beta0 = load_from_checkpoint("ismip-c.h5")
beta = Constant(1000.0/1000000,domain=mesh)
A = Constant(100.0,domain=mesh)
gamma = Constant(1.0,domain=mesh)
m=1.0


def friction_stress(u, C):
    r"""Compute the shear stress for a given sliding velocity"""
    return -C * firedrake.sqrt(firedrake.inner(u, u)) ** (1 / m - 1) * u


def bed_friction(**kwargs):
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
    C = beta*firedrake.exp(theta)

    τ = friction_stress(u, C)
    return -m / (m + 1) * firedrake.inner(τ, u)




# (Re)build solver on the current mesh
model = icepack.models.IceStream(
    friction=bed_friction
)
# On a periodic-x mesh, only y-walls are exterior facets:
opts = {
    'side_wall_ids': [1, 2],}


solver=icepack.solvers.FlowSolver(model, **opts)


def simulation(theta):
    return solver.diagnostic_solve(
            velocity=u_obs, thickness=h, surface=s,
            fluidity=A, log_friction=theta
        )

# domain area for normalisation
area = Constant(assemble(Constant(1.0) * dx(mesh)),domain=mesh)
theta = firedrake.Function(Q)

def loss_function(u):
    du = u - u_obs
    return 0.5 / area * ((du[0])**2 + (du[1])**2) * dx

def regularization(theta):
	L = Constant(5e3,domain=mesh)
	theta_scale=Constant(1.0,domain=mesh)
	return 0.5 / area  * (L / theta_scale)**2 * inner(grad(theta), grad(theta)) * dx

problem = StatisticsProblem(
            simulation=simulation,
            loss_functional=loss_function,
            regularization=regularization,
            controls=theta,
        )
estimator = MaximumProbabilityEstimator(
    problem,
    gradient_tolerance=1e-4,
    step_tolerance=1e-1,
    max_iterations=50,
)
theta = estimator.solve()