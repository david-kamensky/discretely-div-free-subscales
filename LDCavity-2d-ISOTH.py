## Author: Sajje <Sajje@COMA-PC>
## Created: 2021-06-28

from tIGAr import *
from tIGAr.compatibleSplines import *
from tIGAr.timeIntegration import *
from tIGAr.BSplines import *
import math
import ufl
import sys
import argparse

# Use TSFC representation, due to complicated forms:
parameters["form_compiler"]["representation"] = "tsfc"
sys.setrecursionlimit(10000)

# Get initial input data from argparse in command line:
parser = argparse.ArgumentParser()
parser.add_argument('--Nel',dest='Nel',default=10,help='Number of elements in each direction.')
parser.add_argument('--Re',dest='Re',default=100.0,help='Reynolds number.')
#parser.add_argument('--T',dest='T',default=1.0,help='Length of time interval to consider.') #This line is not needed since the problem is assumed to be steady.
#parser.add_argument('--Dyn',dest='Dynamic_Subscales',default='Yes',help='"Yes" or "No" answer determining whether or not you want to use dynamic subscales.') #This line is not needed since the problem is assumed to be steady.

# The commented out lines were kept in the case that we need to use this code for unsteady lid-driven cavity problems...

# Initial Input Data
args = parser.parse_args()
Nel = int(args.Nel)
Re = float(args.Re)
#T = float(args.T) #This line is not needed since the problem is assumed to be steady.
#Dynamic_Subscales = str(args.Dynamic_Subscales) #This line is not needed since the problem is assumed to be steady.

# Print output to tell user if dynamic subscales are on or off.
#if Dynamic_Subscales.lower() == 'yes':
#    print("Dynamic subscales are ON.")
#else:
#    print("Dynamic subscales are OFF. Formulation is now using quasi-staic subscales.")

# None of the above is needed since the problem is assumed to be steady.
print("Problem is steady. Defaulting to quasi-static subscale definitions.")

#Nel = int(10.0)
#Re = Constant(100.0)
#T = float(1.0)
#Dynamic_Subscales = "No"

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
kvecs_pressure = [uniformKnots(degs_pressure[0],0.0,1.0,Nel,False,continuityDrop=1), uniformKnots(degs_pressure[1],0.0,1.0,Nel,False,continuityDrop=1)]

# Define a trivial mapping from parametric to physical space, via explicit
# B-spline.  Extraction is done to triangular elements, to permit the use
# of Quadrature-type elements for the dynamic subgrid scales.
controlMesh = ExplicitBSplineControlMesh(degs_velocity,kvecs_velocity,useRect=False)

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

# Put the velocity solution together. Component 3 and 4 are the coarse-scale and fine-scale pressure fields respectively.
# They need to be added here to avoid shape-error when this solution is projected into the initial velocity field later.
u_IC = as_vector((x_velocity_exact,y_velocity_exact,0,0))

# Exact solution of velocity u without the pressure fields added in component 3,4. This is used when checking the error later.
u_IC_nopressure = as_vector((x_velocity_exact,y_velocity_exact))

# The exact solution for the pressure.
p_IC = sin(pi*x[0])*sin(pi*x[1])

####### Operator and Form Definitions #######

## Operator definitions that appear in the formulation:

# This definition of kinematic viscosity appears in the derivation of non-dimensionalized incompressible N-S equations...
nu = 1.0/Re

#Definition of symmetrized gradient of u, $\nabla_x^s u$:
#This is also the rate of strain tensor $\epsilon$.
def sym_grad(u):
    return 0.5*(spline.grad(u) + spline.grad(u).T)

# spline.grad(u).T is the transpose of grad(u).

# Definition of "non-dimensionalized" cauchy stress sigma without pressure term and negative sign:
def nd_sigma(u):
    return 2.0*nu*sym_grad(u)

# The defintion above comes from "non-dimensionalizing" unsteady incompressible NS equations and separating the "pressure" and "kinematic viscosity" parts.
# If the assumption that density $\rho = 1$ is made, then you can just divide the entire incopressible NS equations by $\rho$ to obtain the first equation
# that appears in problem S of section 2.1.

# Form definitions that appear in the formulation:

# 2nd equation that appears in problem (V) in section 2.1:
def c(v1,v2,v3):
    return dot(dot(spline.grad(v2),v1),v3)*spline.dx

# 3rd equation that appears in problem (V) in section 2.1:
def k(v1,v2):
    return inner(nd_sigma(v1),sym_grad(v2))*spline.dx

# 4th equation that appears in problem (V) in section 2.1:
def b(v,q):
    return (spline.div(v))*q*spline.dx

# Equation (14) that appears in section 2.3:
def c_cons(a,v1,v2):
    return (-1.0)*dot(v1,dot(spline.grad(v2),a))*spline.dx

# Equation (15) that appears in section 2.3:
def c_skew(a,v1,v2):
    return (0.5)*(c(a,v1,v2) + c_cons(a,v1,v2))

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
# Define mesh size metric:
dx_dxiHat = 0.5*ufl.Jacobian(spline.mesh)
G = inv(dx_dxiHat.T*dx_dxiHat)

# Define stabilization parameters:
C_I = Constant(60.0)

# The choice of C_I = 60.0 above is "arbitrary."

# Definition of stabilization parameter $\tau_M$. There is no difference between dynamic and quasi-static subscale definition of $\tau_M$ since the problem is steady.
print("Generating stabilization parameter tau_M.")
tau_M = 1.0/(sqrt(dot(uh,G*uh) + (C_I**2)*(nu**2)*inner(G,G)))

# Define tau_C, not sure where this definition is from but yolo...
tau_C = 1.0/(tau_M*tr(G))

# Define residual of the momentum balance equation in problem (S); given by equation (12):
# Here, f is not assumed to be zero. So first, manufacture the source term using the exact solution data:
f = dot(spline.grad(u_IC_nopressure),u_IC_nopressure) + spline.grad(p_IC) - spline.div(nd_sigma(u_IC_nopressure))

# Then compute residual:
r_M = dot(spline.grad(uh),uh) - spline.div(nd_sigma(uh)) + spline.grad(ph) - f

# Pick which definition of fine-scale velocity u' to use based on whether or not we are using dynamic or quasi-static subscales.
print("Generating fine-scale velocity u'.")
uP = (-1.0*tau_M)*(spline.grad(pP)) + (-1.0*tau_M)*(r_M)

# Define coarse-scale subproblem found in problem $V^h$ in section 2.5 of the paper.
print("Generating coarse-scale residual.")
residual_coarse = c_skew(uh,uh,vh) + k(uh,vh) - b(vh,ph) + b(uh,qh) \
                    + c_cons(uh,uP,vh) + c_skew(uP,uh,vh) + c_cons(uP,uP,vh) \
                    + (tau_C)*(spline.div(uh))*(spline.div(vh))*spline.dx - inner(f,vh)*spline.dx

# Define fine-scale subproblem found in problem $V^h$ found in section 2.5 of the paper:
residual_fine = (-1.0)*inner(spline.grad(qP),uP)*spline.dx

#The other terms are killed off because of static condensation on the $v^'$.

# Define nonlinear residual by summing coarse and fine scale problems. Then define its Jacobian:
residual_SUM = residual_coarse + residual_fine
residual_SUM_jacobian = derivative(residual_SUM,up_h)

####### Start solving the problem #######

# First project the initial condition into B-Spline space
up_h.assign(spline.project(u_IC,applyBCs=False,rationalize=False,lumpMass=False))

#Compute up_h
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

#Output the required files to be read and processed.
output_file = open('copypasta-ldcavity-Iso.txt','w')
output_file.write('Re = '+str(Re))
output_file.write(', Nel = '+str(Nel))
output_file.write(', h = '+str(1.0/Nel))
output_file.write(', H^1 velocity error = '+str(err_u_H1))
output_file.write(', L^2 pressure error = '+str(err_p_L2))
output_file.close()

# Definition of Quasi-Static subscale definition of $u^'$ comes from equation (44) in paper.
# Formulation comes from definition of problem $V^h$ on page 7 on the paper.
# Static-Condensation of fine scale velocity u' is found in section 2.7 of paper. Basically, all v' terms in fine-scale equation die and time derivative of u' die.



