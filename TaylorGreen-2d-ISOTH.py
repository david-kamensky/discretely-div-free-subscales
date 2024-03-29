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
import numpy as np
import scipy.io as sio

# Use TSFC representation, due to complicated forms:
parameters["form_compiler"]["representation"] = "tsfc"
sys.setrecursionlimit(10000)

# Get initial input data from argparse in command line:
parser = argparse.ArgumentParser()
parser.add_argument('--Nel',dest='Nel',default=10,help='Number of elements in each direction.')
parser.add_argument('--Re',dest='Re',default=100.0,help='Reynolds number.')
parser.add_argument('--T',dest='T',default=1.0,help='Length of time interval to consider.')
parser.add_argument('--Dyn',dest='Dynamic_Subscales',default='Yes',help='"Yes" or "No" answer determining whether or not you want to use dynamic subscales.')
parser.add_argument('--KE',dest='Kinetic_Energy',default='No',help='"Yes" or "No" answer determining whether or not you want to track the decay of kinetic energy over time.')

# Initial Input Data
args = parser.parse_args()
Nel = int(args.Nel)
Re = float(args.Re)
T = float(args.T)
Dynamic_Subscales = str(args.Dynamic_Subscales)
Kinetic_Energy = str(args.Kinetic_Energy)

# Print output to tell user if dynamic subscales are on or off.
if Dynamic_Subscales.lower() == 'yes':
    print("Dynamic subscales are ON.")
else:
    print("Dynamic subscales are OFF. Formulation is now using quasi-static subscales.")

# Print output to tell user if kinetic energy is being tracked.
if Kinetic_Energy.lower() == 'yes':
    print("Kinetic energy will be computed and tracked over time.")
    # Generate kinetic energy storage matrix that will be returned once the entire script has finished running:
    KE_Storage = np.empty([Nel,1])
else:
    print("Kinetic energy will not be computed.")

####### Preprocessing #######

if(mpirank==0):
    print("Generating extraction data...")

# The definition of the isogeometric Taylor-Hood element.
# Quadratic velocity.
degs_velocity = [3,3]
# Linear pressure.
degs_pressure = [2,2]

# Knot vectors for defining the control mesh.
kvecs_velocity = [uniformKnots(degs_velocity[0],-math.pi,math.pi,Nel,False,continuityDrop=1), uniformKnots(degs_velocity[1],-math.pi,math.pi,Nel,False,continuityDrop=1)]
kvecs_pressure = [uniformKnots(degs_pressure[0],-math.pi,math.pi,Nel,False), uniformKnots(degs_pressure[1],-math.pi,math.pi,Nel,False)]

# Define a trivial mapping from parametric to physical space, via explicit
# B-spline.  Extraction is done to triangular elements, to permit the use
# of Quadrature-type elements for the dynamic subgrid scales.
useRect=False
controlMesh = ExplicitBSplineControlMesh(degs_velocity,kvecs_velocity,
                                         useRect=useRect)

# Initialize field list to be blank, then add four fields...
fieldList = []
# Field containing x-component velocity.
fieldList += [BSpline(degs_velocity,kvecs_velocity,useRect=useRect),]
# Field containing y-component velocity.
fieldList += [BSpline(degs_velocity,kvecs_velocity,useRect=useRect),]
# Field containing coarse-scale pressure.
fieldList += [BSpline(degs_pressure,kvecs_pressure,useRect=useRect),]
# Field containing fine-scale pressure.
fieldList += [BSpline(degs_pressure,kvecs_pressure,useRect=useRect),]

splineGenerator = FieldListSpline(controlMesh,fieldList)

# Apply strong BCs to normal velocity component and pin down one pressure DoF:
for field in range(0,2):
    scalarSpline = splineGenerator.getFieldSpline(field)
    for side in range(0,2):
        sideDofs = scalarSpline.getSideDofs(field,side)
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

# Initial condition for the Taylor--Green vortex
x = spline.spatialCoordinates()

# Put the velocity solution together. Components 3 and 4 are the coarse-scale and fine-scale pressure fields respectively.
# They need to be added here to avoid shape-error when this solution is projected into the initial velocity field later.
u_IC = as_vector((sin(x[0])*cos(x[1]),-cos(x[0])*sin(x[1]),0,0))

#Initial conidtion of velocity u without the pressure fields added in components 3 and 4. This is used when checking the error later.
u_IC_nopressure = as_vector((sin(x[0])*cos(x[1]),-cos(x[0])*sin(x[1])))

# The pressure is made zero at corners by the non-standard `- 2.0`; the
# purpose of this is for easy comparison with discrete solutions which
# are constrained to be zero at a corner for uniqueness.
# (This relies on knowing that the 0-th DoF for the pressure will be at a
# corner, which is true, but not strictly specified by the tIGAr API.)
p_IC = 0.25*(cos(2.0*x[0]) + cos(2.0*x[1]) - 2.0)

# Time dependence of exact solution:
solnt = Expression("exp(-2.0*nu*t)",nu=1.0/float(Re),t=0.0,degree=1)

####### Operator and Form Definitions #######

## Operator definitions that appear in the formulation:

# Kinematic viscosity:
nu = Constant(1.0/Re)

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
def c_cons(a,v1,v2):
    return (-1.0)*dot(v1,dot(spline.grad(v2),a))*spline.dx
def c_skew(a,v1,v2):
    return (0.5)*(c(a,v1,v2) + c_cons(a,v1,v2))

# Kinetic energy of the coarse + fine solutions:
def KEnergy(v1,v2):
    return 0.5*((math.sqrt(assemble(inner((v1 + v2),(v1 + v2))*spline.dx)))**2)

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
up_old_h = Function(spline.V)
uh_old, ph_old, pP_old = unpack(up_old_h)


# If dynamic subscales are being used, a separate function space must be made for the fine-scale velocity. If no dynamic subscales are being used, then
# the code can just proceed as normal.
if Dynamic_Subscales.lower() == 'yes':
    print("Generating separate function space for fine-scale velocity.")
    # Generate a separate function space for the fine-scale velocity.
    uP_VectorElement = VectorElement("Quadrature",spline.mesh.ufl_cell(),Quadrature_Degree,quad_scheme="default")
    # The space $V'$ where $u'$ lives is generated from the vector element above.
    VP = FunctionSpace(spline.mesh,uP_VectorElement)
    uP_old = Function(VP)
else:
    print("No separate function space for the fine-scale velocity is needed.")


# Define time derivative:
N_STEPS = Nel
Dt = Constant(T/N_STEPS)
uh_t = (uh - uh_old)/Dt
uh_mid = 0.5*(uh + uh_old)

# The physical test functions:
vq_h = TestFunction(spline.V)
vh, qh, qP = unpack(vq_h)

####### Problem Formulation #######

# Define mesh size metric:
dx_dxiHat = 0.5*ufl.Jacobian(spline.mesh)
G = inv(dx_dxiHat*dx_dxiHat.T)

# Define stabilization parameters:
C_I = Constant(60.0)

# Pick which definition of stabilization parameter $\tau_M$ to use based on whether or not we are using dynamic or quasi-static subscales.
if Dynamic_Subscales.lower() == 'yes':
    # Using dynamic subscale definition of parameter $\tau_M$:
    print("Using dynamic subscales definition of stabilization parameter tau_M.")
    tau_M = 1.0/(sqrt(dot(uh_mid,G*uh_mid) + (C_I**2)*(nu**2)*inner(G,G)))
else:
    # Using quasi-static subscale definition of parameter $\tau_M$:
    print("Using quasi-static subscale definition of stabilization parameter tau_M.")
    tau_M = 1.0/(sqrt((4.0/(Dt**2)) + dot(uh_mid,G*uh_mid) + (C_I**2)*(nu**2)*inner(G,G)))

# Define tau_C in terms of tau_M:
tau_C = 1.0/(tau_M*tr(G))

# Strong residual of momentum balance, with zero source term:
r_M = uh_t + dot(spline.grad(uh_mid),uh_mid) - spline.div(nd_sigma(uh_mid)) + spline.grad(ph)

# Pick which definition of fine-scale velocity u' to use based on whether or not we are using dynamic or quasi-static subscales.
if Dynamic_Subscales.lower() == 'yes':
    print("Using dynamic subscales definition of u'.")
    # Definition of dynamic subscale u' is found in equation (42) of the paper.
    I = Identity(spline.mesh.geometry().dim())
    LHS = (1.0/Dt + 0.5/tau_M)*I + 0.5*spline.grad(uh_mid)
    RHS = -r_M - spline.grad(pP) + (1.0/Dt)*uP_old - (0.5/tau_M)*uP_old - 0.5*spline.grad(uh_mid)*uP_old
    uP = inv(LHS)*RHS

    # Get midpoint u' velocity:
    uP_mid = 0.5*(uP + uP_old)

    # Get time-derivative of u' velocity:
    uP_t = (uP - uP_old)/Dt
else:
    print("Using quasi-static subscale definition of u'.")
    uP_mid = (-1.0*tau_M)*(spline.grad(pP)) + (-1.0*tau_M)*(r_M)


# Define coarse-scale subproblem:
if Dynamic_Subscales.lower() == 'yes':
    print("Using dynamic subscales definition of the coarse-scale residual.")
    # Definition of coarse-scale residual for dynamic subscale definition.
    residual_coarse = inner(uh_t,vh)*spline.dx + c_skew(uh_mid,uh_mid,vh) + k(uh_mid,vh) - b(vh,ph) + b(uh_mid,qh) \
                    + c_cons(uh_mid,uP_mid,vh) + c_skew(uP_mid,uh_mid,vh) + c_cons(uP_mid,uP_mid,vh) + dot(uP_t,vh)*spline.dx \
                    + (tau_C)*(spline.div(uh_mid))*(spline.div(vh))*spline.dx
else:
    print("Using quasi-static subscale definition of the coarse-scale residual.")
    # The time-derivative term of $u^'$ in problem $V^h$ is not included in the residual below. In the quasi-static subscale case, this term is zero.
    residual_coarse = inner(uh_t,vh)*spline.dx + c_skew(uh_mid,uh_mid,vh) + k(uh_mid,vh) - b(vh,ph) + b(uh_mid,qh) \
                    + c_cons(uh_mid,uP_mid,vh) + c_skew(uP_mid,uh_mid,vh) + c_cons(uP_mid,uP_mid,vh) \
                    + (tau_C)*(spline.div(uh_mid))*(spline.div(vh))*spline.dx


# Define fine-scale subproblem found in problem $V^h$ found in section 2.5 of the paper:
residual_fine = (-1.0)*inner(spline.grad(qP),uP_mid)*spline.dx

# The other terms are killed off because of static condensation on the $v^'$.

# Define nonlinear residual by summing coarse and fine scale problems. 
residual_SUM = residual_coarse + residual_fine
residual_SUM_jacobian = derivative(residual_SUM,up_h)

####### Start solving the problem #######

# First project the initial condition into B-Spline space
up_old_h.assign(spline.project(u_IC,applyBCs=False,rationalize=False,lumpMass=False))

# Then start the time stepping loop to get solution:
for step in range(0,N_STEPS):
    if(mpirank==0):
        print("======= Time step "+str(step+1)+"/"+str(N_STEPS)+" =======")
        print("Current time: "+str(float(Dt*(step+1)))+" seconds.")

    #Compute up_h
    spline.solveNonlinearVariationalProblem(residual_SUM,residual_SUM_jacobian,up_h)
    
    # Track the decay of kinetic energy, if desired:
    if Kinetic_Energy.lower() == 'yes':
        print("Current KE: "+str(KEnergy(uh_mid,uP_mid))+".")
        KE_Storage[step,0] = float(KEnergy(uh_mid,uP_mid))

    # Update old coarse-scale variables:
    up_old_h.assign(up_h)


####### Postprocessing #######

# Check velocity error in $H^1$ and pressure error in $L^2$:
grad_u_exact = solnt*spline.grad(u_IC_nopressure)
p_exact = (solnt**2)*p_IC
solnt.t = T
grad_e_u = spline.grad(uh) - grad_u_exact
e_p = ph - p_exact
err_u_H1 = math.sqrt(assemble(inner(grad_e_u,grad_e_u)*spline.dx))
err_p_L2 = math.sqrt(assemble(inner(e_p,e_p)*spline.dx))
if(mpirank==0):
    print("======= Final Results =======")
    print("log(h) = "+str(math.log(1.0/Nel)))
    print("log(H^1 velocity error) = "+str(math.log(err_u_H1)))
    print("log(L^2 pressure error) = "+str(math.log(err_p_L2)))

# Output the required files to be read and processed.
output_file = open('data-tg-Iso.txt','a')
output_file.write('Using dynamic subscales? '+str(Dynamic_Subscales))
output_file.write(', Re = '+str(Re))
output_file.write(', Time Length = '+str(T))
output_file.write(', Nel = '+str(Nel))
output_file.write(', h = '+str(1.0/Nel))
output_file.write(', H^1 velocity error = '+str(err_u_H1))
output_file.write(', L^2 pressure error = '+str(err_p_L2))
output_file.write('\r\n')
output_file.close()

# Print output to tell user if kinetic energy is being tracked.
if Kinetic_Energy.lower() == 'yes':
    #Verify that Kinetic Energy decreased over time:

    # Define some "boolean" marker determining whether or not KE decreased. Default will be set to "Yes" as the theory expects.
    bool_KE = "Yes"

    # Simple for loop will check whether or not KE decreases over time. KE at time-step t+1 must be less than KE at time-step t.
    for t in range(0,Nel-1):
        bool_KE_Check = KE_Storage[t+1,0] - KE_Storage[t,0]
        # If KE at time-step t+1 is larger than KE at time-step t, then set the boolean marker to "No".
        if (bool_KE_Check > 0):
            bool_KE = "NO"

    # Print the answer to the question: "Did KE decrease?"
    # Yes/No answer printed directed to the command line so that you don't have to look through the output .mat file manually.
    # If the answer output here is "No", then there is probably something wrong with the formulation.
    print("Did Kinetic Energy decrease over time: "+str(bool_KE))

    # Output a MATLAB matrix containing the tracked kinetic energy:
    sio.savemat('Kinetic Energy Isogeometric.mat', {'KE_Storage': KE_Storage})
