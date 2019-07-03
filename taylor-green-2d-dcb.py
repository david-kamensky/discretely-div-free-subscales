"""
2D Taylor--Green vortex, using div-conforming B-splines.  Subscales are quasi-static.
"""

from tIGAr import *
from tIGAr.compatibleSplines import *
from tIGAr.BSplines import *
import math
import ufl

# Suppress warnings about Krylov solver non-convergence:
set_log_level(40)

# Use TSFC representation, due to complicated forms:
parameters["form_compiler"]["representation"] = "tsfc"
import sys
sys.setrecursionlimit(10000)

####### Parameters #######

# Arguments are parsed from the command line, with some hard-coded defaults
# here.  See help messages for descriptions of parameters.

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--NEL',dest='NEL',default=16,
                    help='Number of elements in each direction.')
parser.add_argument('--kPrime',dest='kPrime',default=1,
                    help='Degree up to which velocity is complete.')
parser.add_argument('--Re',dest='Re',default=100.0,
                    help='Reynolds number.')
parser.add_argument('--T',dest='T',default=1.0,
                    help='Length of time interval to consider.')
parser.add_argument('--VIZ',dest='VIZ',action='store_true',
                    help='Include option to output visualization files.')
parser.add_argument('--MAX_KSP_IT',dest='MAX_KSP_IT',default=500,
                    help='Maximum number of Krylov iterations.')
parser.add_argument('--LINEAR_TOL',dest='LINEAR_TOL',default=1e-2,
                    help='Relative tolerance for Krylov solves.')
parser.add_argument('--NONLIN_TOL',dest='NONLIN_TOL',default=1e-4,
                    help='Relative tolerance for nonlinear solves.')
parser.add_argument('--penalty',dest='penalty',default=1e4,
                    help='Dimensionless penalty for iterated penalty solver.')

args = parser.parse_args()
NEL = int(args.NEL)
kPrime = int(args.kPrime)
Re = Constant(float(args.Re))
T = float(args.T)
VIZ = bool(args.VIZ)
MAX_KSP_IT = int(args.MAX_KSP_IT)
LINEAR_TOL = float(args.LINEAR_TOL)
NONLIN_TOL = float(args.NONLIN_TOL)
penalty = float(args.penalty)

####### Preprocessing #######

if(mpirank==0):
    print("Generating extraction data...")

# Polynomial degree (in the sense of $k'$, i.e., the degree up to which
# velocity approximation is polynomially-complete) in each
# parametric direction.  (Note that degree=1 still implies some
# $C^1$ quadratic B-splines for the div-conforming unknown fields.)
degs = 2*[kPrime,]

# Knot vectors for defining the control mesh.
kvecs = [uniformKnots(degs[0],-math.pi,math.pi,NEL,False),
         uniformKnots(degs[1],-math.pi,math.pi,NEL,False)]

# Define a trivial mapping from parametric to physical space, via explicit
# B-spline.
controlMesh = ExplicitBSplineControlMesh(degs,kvecs)

# Define the spaces for RT-type compatible splines on this geometry.
fieldList = generateFieldsCompat(controlMesh,"RT",degs)
# Include an extra scalar field for the fine-scale pressure.
fieldList += [BSpline(degs,kvecs),]
splineGenerator = FieldListSpline(controlMesh,fieldList)

# Apply strong BCs to normal velocity component:
for field in range(0,2):
    scalarSpline = splineGenerator.getFieldSpline(field)
    for side in range(0,2):
        sideDofs = scalarSpline.getSideDofs(field,side)
        splineGenerator.addZeroDofs(field,sideDofs)

####### Analysis #######

if(mpirank==0):
    print("Setting up extracted spline...")

# Overkill quadrature to ensure optimal convergence:
QUAD_DEG = 2*(max(degs)+1)
spline = ExtractedSpline(splineGenerator,QUAD_DEG)

# Initial condition for the Taylor--Green vortex
x = spline.spatialCoordinates()
u_IC = as_vector((sin(x[0])*cos(x[1]),-cos(x[0])*sin(x[1])))

# Time dependence of exact solution:
solnt = Expression("exp(-2.0*nu*t)",nu=1.0/float(Re),t=0.0,degree=1)

# Useful operators for defining the formulation:
nu = 1.0/Re
def Ladv(u,v,q):
    return spline.grad(v)*u + spline.grad(q)
def eps(u):
    return 0.5*(spline.grad(u) + spline.grad(u).T)
def sigmaVisc(u):
    return 2.0*nu*eps(u)
def Lvisc(u):
    return -spline.div(sigmaVisc(u))

if(mpirank==0):
    print("Starting analysis...")

# "Un-pack" a velocity--pressure pair from spline.V, which is just a mixed
# space with four scalar fields.
def unpack(up):
    d = spline.mesh.geometry().dim()
    u_hat = as_vector([up[i] for i in range(0,d)])
    p_hat = up[-1]
    return u_hat, p_hat

# Note on variable naming:  The "fine-scale pressure" here is in fact treated
# in the formulation as $p^h + p'$ when it is used in the stabilization terms,
# in which sense the name `p` makes sense (rather than, say, `pPrime`).  The
# coarse scale pressure is not included in the formulation, as the coarse-scale
# velocity is constrained to be solenoidal by the use of an iterated penalty
# solver.

# The unknown parametric velocity and fine-scale pressure:
up_hat = Function(spline.V)
u_hat, p_hat = unpack(up_hat)
up_hat_old = Function(spline.V)
u_hat_old, p_hat_old = unpack(up_hat_old)

# The physical velocity and its temporal partial derivative:
u = cartesianPushforwardRT(u_hat,spline.F)
u_old = cartesianPushforwardRT(u_hat_old,spline.F)
p = cartesianPushforwardW(p_hat,spline.F)

# Partial time derivative:
N_STEPS = NEL
Dt = Constant(T/N_STEPS)
u_t = (u - u_old)/Dt
u_mid = 0.5*(u + u_old)

# The parametric and physical test functions:
vq_hat = TestFunction(spline.V)
v_hat, q_hat = unpack(vq_hat)
v = cartesianPushforwardRT(v_hat,spline.F)
q = cartesianPushforwardW(q_hat,spline.F)

# Contribution to the weak problem for given test function; plugging u into
# this as the test function will be considered the "resolved dissipation".
def resVisc(v):
    return inner(sigmaVisc(u_mid),eps(v))*spline.dx

# The material time derivative of the velocity:
Du_Dt = u_t + spline.grad(u_mid)*u_mid

# The problem is posed on a solenoidal subspace, as enforced by the iterative
# penalty solver; no pressure terms are necessary in the weak form.
resGalerkin = inner(Du_Dt,v)*spline.dx + resVisc(v)

# Extra term associated with stabilization.  This technically leaves
# an un-determined hydrostatic mode in the fine-scale
# pressure, but we can let the iterative solver choose it for us with no
# effect on the velocity solution.
resStrong = Du_Dt + Lvisc(u_mid) + spline.grad(p)
dxi_dxiHat = 0.5*ufl.Jacobian(spline.mesh)
dx_dxi = spline.parametricGrad(spline.F)
dx_dxiHat = dx_dxi*dxi_dxiHat
Ginv = dx_dxiHat.T*dx_dxiHat
G = inv(Ginv)
C_I = Constant(3.0*max(degs)**2)
tau_M = 1.0/sqrt(dot(u_mid,G*u_mid) + (2.0/Dt)**2
                 + (C_I**2)*(nu**2)*inner(G,G))
uPrime = -tau_M*resStrong
resStab = inner(uPrime,-Ladv(u_mid,v,q))*spline.dx \
          + inner(v,spline.grad(u_mid)*uPrime)*spline.dx \
          - inner(spline.grad(v),outer(uPrime,uPrime))*spline.dx

# Define nonlinear residual and Jacobian:
res = resGalerkin + resStab
Dres = derivative(res,up_hat)

# Divergence of the velocity field, given a function in the mixed space.
# It is weighted to make the penalty dimensionless and
# indepenent of position, as needed for the update step of the iterated
# penalty solver.
tau_C = 1.0/(tau_M*tr(G))
divOp = lambda up : sqrt(tau_C)*spline.div(cartesianPushforwardRT
                                           (unpack(up)[0],spline.F))

# Use iterative solver.  
spline.linearSolver = PETScKrylovSolver("gmres","jacobi")
spline.linearSolver.parameters["relative_tolerance"] = LINEAR_TOL
# Linear solver sometimes fails to converge, but convergence of nonlinear
# iteration is still enforced to within spline.relativeTolerance.
spline.linearSolver.parameters["error_on_nonconvergence"] = False
spline.linearSolver.parameters["maximum_iterations"] = MAX_KSP_IT
spline.linearSolver.ksp().setGMRESRestart(MAX_KSP_IT)

# Tolerance may need to be tighter to observe optimal convergence
# for high kPrime and/or NEL.
spline.relativeTolerance = NONLIN_TOL

if(mpirank==0):
    print("Projecting initial condition...")

# Project the initial condition:
up_hat_old.assign(divFreeProject(u_IC,spline,
                                 getVelocity=lambda up : unpack(up)[0],
                                 applyBCs=False))

# Time stepping loop:
for step in range(0,N_STEPS):
    if(mpirank==0):
        print("======= Time step "+str(step+1)+"/"+str(N_STEPS)+" =======")

    # Solve for the coarse-scale velocity and fine-scale pressure:
    iteratedDivFreeSolve(res,up_hat,vq_hat,spline,
                         divOp=divOp,
                         penalty=Constant(penalty),
                         J=Dres,reuseLHS=True)
        
    up_hat_old.assign(up_hat)

# Check error in $H^1$:
grad_u_exact = solnt*spline.grad(u_IC)
solnt.t = T
grad_e_u = spline.grad(u) - grad_u_exact
err_u_H1 = math.sqrt(assemble(inner(grad_e_u,grad_e_u)*spline.dx))
if(mpirank==0):
    print("log(h) = "+str(math.log(1.0/NEL)))
    print("log(H^1 velocity error) = "+str(math.log(err_u_H1)))

# Output ParaView files as a sanity check, if desired.
if(VIZ):
    # Take advantage of explicit B-spline geometry to simplify visualization:
    ux, uy, p = up_hat.split()
    ux.rename("u","u")
    uy.rename("v","v")
    File("results-dcb/ux.pvd") << ux
    File("results-dcb/uy.pvd") << uy
