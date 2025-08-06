import firedrake
import icepack
import matplotlib.pyplot as plt
import numpy as np
import math
import sympy
from sympy import legendre as sympy_legendre

def generate_surface(s,b,sigma):
    error = np.random.normal(0,sigma, size=s.dat.data.shape)
    s_obs = s.copy(deepcopy=True)
    s_obs.dat.data[:]=error+s.dat.data[:]
    s_obs.dat.data[s_obs.dat.data<b.dat.data]=s.dat.data[s_obs.dat.data<b.dat.data]
    
    return s_obs 
    
def compute_surface_velocity(q):
	r"""Return the weighted depth average of a function on an extruded mesh"""
	def weight(n, ζ):
		"""Compute the normalization factor for Legendre polynomials."""
		norm = sympy.integrate(sympy_legendre(n, ζ) ** 2, (ζ, 0, 1))  # L2 norm squared
		return sympy_legendre(n, ζ) / norm  # Return symbolic expression

	def legendre(n, ζ):
		"""Map Legendre polynomial to the domain [0,1]."""
		return sympy_legendre(n, 2 * ζ - 1)

	Q = q.function_space()
	mesh = Q.mesh()
	ζ = firedrake.SpatialCoordinate(mesh)[mesh.geometric_dimension() - 1]

	xdegree_q, zdegree_q = q.ufl_element().degree()

	ζsym = sympy.symbols("ζsym", positive=True)

	full_weight_symbolic = sum(
		[
	    	legendre(k, 1) * weight(k, ζsym)  # Symbolic expression
	    	for k in range(zdegree_q)
		]
	).doit() 

	weight_function_numeric = sympy.lambdify(ζsym, full_weight_symbolic, "numpy")

	# Compute the weighted depth average
	q_surface = icepack.utilities.depth_average(q, weight=weight_function_numeric(ζ))

	return q_surface

def generate_velocity(u,sigma):
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


L=500000

nx, ny = 25, 25  # resolution
mesh2d = firedrake.PeriodicRectangleMesh(nx, ny, L, L, direction='x')
mesh = firedrake.ExtrudedMesh(mesh2d, layers=1)


fig, axes = plt.subplots()
axes.set_aspect("equal")
firedrake.triplot(mesh2d, axes=axes)
axes.legend(loc="upper right");
fig.savefig('mesh.png')

V = firedrake.VectorFunctionSpace(
    mesh, "CG", 2, dim=2, vfamily="GL", vdegree=2
)

Q = firedrake.FunctionSpace(
    mesh, "CG", 2, vfamily="R", vdegree=0
)

# Extract coordinates
x, y, ζ = firedrake.SpatialCoordinate(mesh)


# Define surface elevation
s = firedrake.Function(Q, name="surface")
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

model = icepack.models.HybridModel(friction=friction)
flow_solver = icepack.solvers.FlowSolver(model,**opts)
u = flow_solver.diagnostic_solve(
    thickness=h,
    velocity=u0,
    surface=s,
    friction=beta,
    sliding_exponent=m,
    fluidity=A
)


u_s=compute_surface_velocity(u)
V_s=u_s.function_space()
sigma = 1.0
u_obs = firedrake.Function(V_s, name="u_obs")

u_obs = generate_velocity(u_s,sigma)

fig, (ax1,ax2) = plt.subplots(2,1)
colors_u = firedrake.tripcolor(u_s, axes=ax1)
fig.colorbar(colors_u, ax=ax1, fraction=0.012, pad=0.04);

colors_u_obs = firedrake.tripcolor(u_obs, axes=ax2)
fig.colorbar(colors_u_obs, ax=ax2, fraction=0.012, pad=0.04);
fig.savefig('velocity.png')

fig, ax = plt.subplots()
colors_beta = firedrake.tripcolor(icepack.utilities.depth_average(beta), axes=ax)
fig.colorbar(colors_beta, ax=ax, fraction=0.012, pad=0.04);
fig.savefig('beta.png')



with firedrake.CheckpointFile("ismip-c.h5", "w") as checkpoint:
    checkpoint.save_mesh(mesh)
    checkpoint.save_function(beta)
    checkpoint.save_function(u)
    checkpoint.save_function(u_obs)
    checkpoint.save_function(bed)
    checkpoint.save_function(s)
    checkpoint.save_function(h)