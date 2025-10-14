import firedrake
import icepack
import matplotlib.pyplot as plt
import numpy as np
import math

def generate_surface(s,b,sigma):
    X=firedrake.interpolate(mesh.coordinates, V).dat.data_ro
    error = np.random.normal(0,sigma, size=s.dat.data.shape)
    s_obs = s.copy(deepcopy=True)
    s_obs.dat.data[:]=error+s.dat.data[:]
    s_obs.dat.data[s_obs.dat.data<b.dat.data]=s.dat.data[s_obs.dat.data<b.dat.data]
    
    return s_obs 
    
def generate_velocity(u,sigma):
    X=firedrake.interpolate(mesh.coordinates, V).dat.data_ro
    error = np.random.normal(0,sigma, size=u.dat.data.shape)
    u_obs = u.copy(deepcopy=True)
    u_obs.dat.data[:]=error+u.dat.data[:]
    
    return u_obs

def friction_stress(u, C, m):
    r"""Compute the shear stress for a given sliding velocity"""
    return -C * firedrake.sqrt(firedrake.inner(u, u)) ** (1 / m - 1) * u


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
    C = kwargs["friction"]
    m = kwargs["sliding_exponent"]

    τ = friction_stress(u, C, m)
    return -m / (m + 1) * firedrake.inner(τ, u)



L=40000

nx, ny = 40, 40  # resolution
mesh = firedrake.PeriodicRectangleMesh(nx, ny, L, L, direction='x',name="ismip-c")

fig, axes = plt.subplots()
axes.set_aspect("equal")
firedrake.triplot(mesh, axes=axes)
axes.legend(loc="upper right");
fig.savefig('mesh.png')

V = firedrake.VectorFunctionSpace(mesh, 'CG', 2)
Q = firedrake.FunctionSpace(mesh, 'CG', 2)
U = firedrake.FunctionSpace(mesh, 'DG', 2)

# Extract coordinates
x, y = firedrake.SpatialCoordinate(mesh)

# Define surface elevation
s = firedrake.Function(U, name="surface")
s.interpolate( - x * firedrake.tan(math.radians(.1)))

# Define ice thickness as s - b
h = firedrake.Function(Q, name="thickness")
h.interpolate(firedrake.Constant(1000))

# Define bedrock elevation
bed = firedrake.Function(Q, name="bed")
bed.interpolate(s - h)


A = firedrake.Constant(100)
m=firedrake.Constant(1.0)
omega = 2*np.pi/L
beta = firedrake.Function(Q, name = "beta")
beta_expr = (1000. + 1000.*firedrake.sin(omega * x) * firedrake.sin(omega * y))/(1000000) 
beta.interpolate(beta_expr)

# Initial guess and parameters
#alpha = firedrake.Function(Q, name="alpha")
#alpha_expr = 100 + 50 * firedrake.exp(-((x - L/2)**2 + (y - L/2)**2) / (2 * (L/4)**2))
#alpha.interpolate(alpha_expr)

# Observations (synthetic example)
u_expr = firedrake.as_vector([0.1, 0.0])
u0 = firedrake.Function(V, name="u_exact")
u0.interpolate(u_expr)


opts = {
    'side_wall_ids': [1, 2],
    "diagnostic_solver_type": "petsc",
    "diagnostic_solver_parameters": {
        "snes_type": "newtontr",
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
    "prognostic_solver_parameters": {
        "ksp_type": "gmres",
        "pc_type": "ilu",
    },
}

#opts = {
#    'side_wall_ids': [1, 2],
#}

model = icepack.models.IceStream(friction=friction)
flow_solver = icepack.solvers.FlowSolver(model,**opts)
u = flow_solver.diagnostic_solve(
    thickness=h,
    velocity=u0,
    surface=s,
    friction=beta,
    sliding_exponent=m,
    fluidity=A
)

sigma = 1.0
u_obs = firedrake.Function(V, name="u_obs")
u_obs = generate_velocity(u,sigma)

fig, (ax1,ax2) = plt.subplots(2,1)
colors_u = firedrake.tripcolor(u, axes=ax1)
fig.colorbar(colors_u, ax=ax1, fraction=0.012, pad=0.04);

colors_u_obs = firedrake.tripcolor(u_obs, axes=ax2)
fig.colorbar(colors_u_obs, ax=ax2, fraction=0.012, pad=0.04);
fig.savefig('velocity.png')

fig, ax = plt.subplots()
colors_beta = firedrake.tripcolor(beta, axes=ax)
fig.colorbar(colors_beta, ax=ax, fraction=0.012, pad=0.04);
fig.savefig('beta.png')



with firedrake.CheckpointFile("../mesh/ismip-c.h5", "w") as checkpoint:
    checkpoint.save_mesh(mesh)
    checkpoint.save_function(beta, name="friction")
    checkpoint.save_function(u, name="velocity")
    checkpoint.save_function(u_obs, name="velocity_obs")
    checkpoint.save_function(bed, name="bed")
    checkpoint.save_function(s, name="surface")
    checkpoint.save_function(h, name="thickness")