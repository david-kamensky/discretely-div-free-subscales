## Author: Sajje <Sajje@COMA-PC>
## Created: 2021-11-10

from dolfin import *
import math
import ufl
from ufl import Max, Min
import sys
import argparse

mpirank = MPI.rank(MPI.comm_world)

# Use TSFC representation, due to complicated forms:
parameters["form_compiler"]["representation"] = "tsfc"
sys.setrecursionlimit(10000)

# Get initial input data from argparse in command line:
parser = argparse.ArgumentParser()
parser.add_argument('--Nel',dest='Nel',default=10,help='Number of elements in each direction.')
parser.add_argument('--Re',dest='Re',default=100.0,help='Reynolds number.')

# Initial Input Data
args = parser.parse_args()
Nel = int(args.Nel)
Re = float(args.Re)

####### Preprocessing #######

if(mpirank==0):
    print("Generating the regular Taylor-Hood element...")

# The definition of the Taylor-Hood element.
# Quadratic velocity.
degs_velocity = 2
# Linear pressure.
degs_pressure = 1

# Mesh and derivation of mesh-size metric:
mesh = UnitSquareMesh(Nel,Nel)
x = SpatialCoordinate(mesh)

# Definition of the Taylor-Hood element, augmented with the fine-scale pressure:
cell = mesh.ufl_cell()
u_el = VectorElement("Lagrange",cell,degs_velocity)
p_el = FiniteElement("Lagrange",cell,degs_pressure)
pP_el = p_el
X_el = MixedElement([u_el,p_el,pP_el])
X = FunctionSpace(mesh,X_el)

# Solution and test function:
w = Function(X)
uh,ph,pP = split(w)
dw = TestFunction(X)
vh,qh,qP = split(dw)

####### Exact Solution Data #######

if(mpirank==0):
    print("Generating exact solution and quadrature rules...")

# Overkill quadrature to ensure optimal convergence:
Quadrature_Degree = 2*((degs_velocity)+1)
dx = dx(metadata={"quadrature_degree":Quadrature_Degree})

# Get the exact solution for regularized lid-driven cavity:
# Exact x-component velocity:
x_velocity_exact = "8.0*(pow(x[0],4) - 2.0*pow(x[0],3) + pow(x[0],2))*(4.0*pow(x[1],3) - 2.0*x[1])"
# Exact y-component velocity:
y_velocity_exact = "-8.0*(4.0*pow(x[0],3) - 6.0*pow(x[0],2) + 2.0*x[0])*(pow(x[1],4) - pow(x[1],2))"

# Vector expression representing exact velocity solution:
u_expr = Expression((x_velocity_exact,y_velocity_exact),degree=Quadrature_Degree,cell=cell)

# Obtain vector form of exact velocity:
u_IC = as_vector((eval(x_velocity_exact),eval(y_velocity_exact)))

# The exact solution for the pressure.
p_IC = sin(pi*x[0])*sin(pi*x[1])

print("Applying boundary conditions...")
# Apply strong BCs to normal velocity component and pin down one pressure DoF:
corner_str = "near(x[0],0.0) && near(x[1],0.0)"
bcs = [DirichletBC(X.sub(0),u_expr,"on_boundary"),
       DirichletBC(X.sub(1),Constant(0.0),corner_str,"pointwise"),
       DirichletBC(X.sub(2),Constant(0.0),corner_str,"pointwise")]

# Set advection velocity $\textbf{a}$ to be the exact velocity solution:
a = u_IC

####### Operator and Form Definitions #######

## Operator definitions that appear in the formulation:

# This definition of kinematic viscosity appears in the derivation of non-dimensionalized incompressible N-S equations...
nu = Constant(1.0/Re)

# Definition of global cell diameter h:
h = Constant(1/Nel)

#Definition of l2 norm of a vector: "verti"
#Norm has to be computed this way because norm(u,'l2') doesn't work for some reason.
def verti(u):
    return sqrt(inner(u,u))

# Definition of "non-dimensionalized" cauchy stress sigma without pressure term and negative sign:
def nd_sigma(u):
    return 2.0*nu*sym(grad(u))

# The defintion above comes from "non-dimensionalizing" unsteady incompressible NS equations and separating the "pressure" and "kinematic viscosity" parts.
# If the assumption that density $\rho = 1$ is made, then you can just divide the entire incopressible NS equations by $\rho$ to obtain the first equation
# that appears in problem S of section 2.1.

# Form definitions that appear in the formulation:

# Define: Convective Form (Section 2)
def c(v1,v2,v3):
    return dot(dot(grad(v2),v1),v3)*dx

# Define: Diffusive Form (Section 2)
def k(v1,v2):
    return inner(nd_sigma(v1),sym(grad(v2)))*dx

# Define: Constraint Form (Section 2)
def b(v,q):
    return (div(v))*q*dx

####### Initialize velocity/pressure functions #######

if(mpirank==0):
    print("Starting analysis...")

####### Problem Formulation #######
# Define stabilization parameters:
C_I = Constant(60.0) # The choice of C_I = 60.0 above is "arbitrary."

# Definition of stabilization parameter $\tau_M$. There is no difference between dynamic and quasi-static subscale definition of $\tau_M$ since the problem is steady.
# Definition of stabilization parameters are found in section 3.
# Define stabilization parameter $\tau_M$.
print("Generating stabilization parameter tau_M.")
tau_M_1 = h / (2*verti(a))
tau_M_2 = (h*h) / (C_I*nu)
tau_M = Min(tau_M_1,tau_M_2)

# Define $\tau_C$:
print("Generating stabilization parameter tau_C.")
tau_C_1 = h*verti(a)
tau_C_2 = nu
tau_C = Max(tau_C_1,tau_C_2)

# Here, f is not assumed to be zero. So first, manufacture the source term using the exact solution data:
f = dot(grad(u_IC),u_IC) + grad(p_IC) - div(nd_sigma(u_IC))

# Definition of fine-scale velocity u'.
print("Generating fine-scale velocity u'.")
uP = tau_M*(f - dot(grad(uh),a) + div(nd_sigma(uh)) - grad(pP))

# Define reduced formulation A_red:
print("Generating reduced formulation.")
A_red = c(a,uh,vh) + k(uh,vh) - b(vh,ph) + b(uh,qh) \
            + inner(dot(grad(uh),a) - div(nd_sigma(uh)) + grad(pP),tau_M*(dot(grad(vh),a) + grad(qP)))*dx \
            + (tau_C)*(div(uh))*(div(vh))*dx

# Define nonlinear residual by summing reduced formulation and source term. Then define its Jacobian:
residual_SUM = A_red - inner(f,vh)*dx - inner(f,tau_M*(dot(grad(vh),a) + grad(qP)))*dx
residual_SUM_jacobian = derivative(residual_SUM,w)

####### Start solving the problem #######

#Compute up_h
solve(residual_SUM==0,w,J=residual_SUM_jacobian,bcs=bcs)

####### Postprocessing #######

# Check velocity error in $H^1$ and pressure error in $L^2$:
# Compute and print error:
err_u_H1 = math.sqrt(assemble(inner(grad(uh - u_IC),grad(uh - u_IC))*dx))
err_p_L2 = math.sqrt(assemble(inner(ph - p_IC,ph -p_IC)*dx))

if(mpirank==0):
    print("======= Final Results =======")
    print("log(h) = "+str(math.log(1.0/Nel)))
    print("log(H^1 velocity error) = "+str(math.log(err_u_H1)))
    print("log(L^2 pressure error) = "+str(math.log(err_p_L2)))

#Output the required files to be read and processed.
output_file = open('copypasta-ldc-Oseen-regular.txt','w')
output_file.write('Re = '+str(Re))
output_file.write(', Nel = '+str(Nel))
output_file.write(', h = '+str(1.0/Nel))
output_file.write(', H^1 velocity error = '+str(err_u_H1))
output_file.write(', L^2 pressure error = '+str(err_p_L2))
output_file.close()
