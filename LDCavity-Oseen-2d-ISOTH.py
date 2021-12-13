## Author: Sajje <Sajje@COMA-PC>
## Created: 2021-11-10

from tIGAr import *
from tIGAr.compatibleSplines import *
from tIGAr.timeIntegration import *
from tIGAr.BSplines import *
import math
import ufl
from ufl import Max, Min
import sys
import argparse

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
    print("Generating extraction data...")

# The definition of the isogeometric Taylor-Hood element.
# Quadratic velocity.
degs_velocity = [3,3]
# Linear pressure.
degs_pressure = [2,2]

# Knot vectors for defining the control mesh.
kvecs_velocity = [uniformKnots(degs_velocity[0],0.0,1.0,Nel,False,continuityDrop=1), uniformKnots(degs_velocity[1],0.0,1.0,Nel,False,continuityDrop=1)]
kvecs_pressure = [uniformKnots(degs_pressure[0],0.0,1.0,Nel,False), uniformKnots(degs_pressure[1],0.0,1.0,Nel,False)]

# Define a trivial mapping from parametric to physical space, via explicit
# B-spline.  
controlMesh = ExplicitBSplineControlMesh(degs_velocity,kvecs_velocity)

# Initialize field list to be blank, then add four fields...
fieldList = []
# Field containing x-component velocity.
fieldList += [BSpline(degs_velocity,kvecs_velocity),]
# Field containing y-component velocity.
fieldList += [BSpline(degs_velocity,kvecs_velocity),]
# Field containing coarse-scale pressure.
fieldList += [BSpline(degs_pressure,kvecs_pressure),]
# Field containing fine-scale pressure.
fieldList += [BSpline(degs_pressure,kvecs_pressure),]

# Generate spline:
splineGenerator = FieldListSpline(controlMesh,fieldList)

# Apply strong BCs to normal velocity component and pin down one pressure DoF:
for field in range(0,2):
    scalarSpline = splineGenerator.getFieldSpline(field)
    for side in range(0,2):
        for direction in range(0,2):
            sideDofs = scalarSpline.getSideDofs(direction,side)
            splineGenerator.addZeroDofs(field,sideDofs)
# Set the coarse-scale pressure to be zero in the corner.
splineGenerator.addZeroDofs(2,[0,])
# Set the fine-scale pressure to be zero in the corner.
splineGenerator.addZeroDofs(3,[0,])

####### Exact Solution Data #######

if(mpirank==0):
    print("Setting up extracted spline...")

# Overkill quadrature to ensure optimal convergence:
Quadrature_Degree = 2*(max(degs_velocity)+1)
spline = ExtractedSpline(splineGenerator,Quadrature_Degree)

# Get the exact solution for regularized lid-driven cavity:
x = spline.spatialCoordinates()
# Exact x-component velocity:
x_velocity_exact = (8.0)*(pow(x[0],4) - 2.0*pow(x[0],3) + pow(x[0],2))*(4.0*pow(x[1],3) - 2.0*x[1])
# Exact y-component velocity:
y_velocity_exact = (-8.0)*(4.0*pow(x[0],3) - 6.0*pow(x[0],2) + 2.0*x[0])*(pow(x[1],4) - pow(x[1],2))

# Put the velocity solution together. Components 3 and 4 are the coarse-scale and fine-scale pressure fields respectively.
# They need to be added here to avoid shape-error when this solution is projected into the initial velocity field later.
u_IC = as_vector((x_velocity_exact,y_velocity_exact,0,0))

# Exact solution of velocity u without the pressure fields added in components 3,4. This is used when checking the error later.
u_IC_nopressure = as_vector((x_velocity_exact,y_velocity_exact))

# The exact solution for the pressure.
p_IC = sin(pi*x[0])*sin(pi*x[1])

# Set advection velocity $\textbf{a}$ to be the exact velocity solution:
a = u_IC_nopressure

####### Operator and Form Definitions #######

## Operator definitions that appear in the formulation:

# Kinematic viscosity:
nu = Constant(1.0/Re)

# Definition of global cell diameter h:
h = Constant(1/Nel)

# Definition of l2 norm of a vector:
def verti(u):
    return sqrt(inner(u,u))

# Definition of symmetrized gradient of u, $\nabla_x^s u$:
# This is also the rate of strain tensor $\epsilon$.
def sym_grad(u):
    return 0.5*(spline.grad(u) + spline.grad(u).T)

# Non-dimensionalized viscous contribution to Cauchy stress:
def nd_sigma(u):
    return 2.0*nu*sym_grad(u)

# Form definitions that appear in the formulation:
def c(v1,v2,v3):
    return dot(dot(spline.grad(v2),v1),v3)*spline.dx
def k(v1,v2):
    return inner(nd_sigma(v1),sym_grad(v2))*spline.dx
def b(v,q):
    return (spline.div(v))*q*spline.dx

####### Initialize velocity/pressure functions #######

if(mpirank==0):
    print("Starting analysis...")

# "Un-pack" a velocity--pressure pair from spline.V, which is just a mixed
# space with four scalar fields.
def unpack(up):
    d = spline.mesh.geometry().dim()
    unpacked_u = as_vector([up[i] for i in range(0,d)])
    unpacked_p = up[d]
    unpacked_pP = up[-1]
    return unpacked_u, unpacked_p, unpacked_pP

# The velocity and fine-scale pressure:
up_h = Function(spline.V)
uh, ph, pP = unpack(up_h)

# The physical test functions:
vq_h = TestFunction(spline.V)
vh, qh, qP = unpack(vq_h)

####### Problem Formulation #######

# Define stabilization parameters:
C_I = Constant(60.0)
tau_M_1 = h / (2*verti(a))
tau_M_2 = (h*h) / (C_I*nu)
tau_M = Min(tau_M_1,tau_M_2)
tau_C_1 = h*verti(a)
tau_C_2 = nu
tau_C = Max(tau_C_1,tau_C_2)

# Source term to manufacture exact solution:
f = dot(spline.grad(u_IC_nopressure),u_IC_nopressure) + spline.grad(p_IC) - spline.div(nd_sigma(u_IC_nopressure))

# Definition of fine-scale velocity u':
uP = tau_M*(f - dot(spline.grad(uh),a) + spline.div(nd_sigma(uh)) - spline.grad(pP))

# Define reduced formulation A_red:
A_red = c(a,uh,vh) + k(uh,vh) - b(vh,ph) + b(uh,qh) \
            + inner(dot(spline.grad(uh),a) - spline.div(nd_sigma(uh)) + spline.grad(pP),tau_M*(dot(spline.grad(vh),a) + spline.grad(qP)))*spline.dx \
            + (tau_C)*(spline.div(uh))*(spline.div(vh))*spline.dx

# Define nonlinear residual by summing reduced formulation and source term. Then define its Jacobian:
residual_SUM = A_red - inner(f,vh)*spline.dx - inner(f,tau_M*(dot(spline.grad(vh),a) + spline.grad(qP)))*spline.dx
residual_SUM_jacobian = derivative(residual_SUM,up_h)

####### Start solving the problem #######

# Initialize nonlinear solver with $L^2$ projection of exact solution:
up_h.assign(spline.project(u_IC,applyBCs=False,rationalize=False,lumpMass=False))

#Compute up_h
spline.linearSolver = PETScLUSolver("mumps")
spline.solveNonlinearVariationalProblem(residual_SUM,residual_SUM_jacobian,up_h)

####### Postprocessing #######

# Check velocity error in $H^1$ and pressure error in $L^2$:
# Compute and print error:
err_u_H1 = math.sqrt(assemble(inner(spline.grad(uh - u_IC_nopressure),spline.grad(uh - u_IC_nopressure))*spline.dx))
err_p_L2 = math.sqrt(assemble(inner(ph - p_IC,ph -p_IC)*spline.dx))

if(mpirank==0):
    print("======= Final Results =======")
    print("log(h) = "+str(math.log(1.0/Nel)))
    print("log(H^1 velocity error) = "+str(math.log(err_u_H1)))
    print("log(L^2 pressure error) = "+str(math.log(err_p_L2)))

# Output the required files to be read and processed.
output_file = open('data-ldcavity-Oseen-Iso.txt','a')
output_file.write('Re = '+str(Re))
output_file.write(', Nel = '+str(Nel))
output_file.write(', h = '+str(1.0/Nel))
output_file.write(', H^1 velocity error = '+str(err_u_H1))
output_file.write(', L^2 pressure error = '+str(err_p_L2))
output_file.write('\r\n')
output_file.close()
