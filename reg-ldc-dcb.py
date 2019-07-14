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
parser.add_argument('--Nel',dest='Nel',default=16,
                    help='Number of elements in each direction.')
parser.add_argument('--kPrime',dest='kPrime',default=1,
                    help='Degree up to which velocity is complete.')
parser.add_argument('--Re',dest='Re',default=100.0,
                    help='Reynolds number.')
parser.add_argument('--VIZ',dest='VIZ',action='store_true',
                    help='Include option to output visualization files.')
parser.add_argument('--NONLIN_TOL',dest='NONLIN_TOL',default=1e-5,
                    help='Relative tolerance for nonlinear solves.')
parser.add_argument('--penalty',dest='penalty',default=1e3,
                    help='Dimensionless penalty for iterated penalty solver.')

args = parser.parse_args()
Nel = int(args.Nel)
kPrime = int(args.kPrime)
Re = Constant(float(args.Re))
VIZ = bool(args.VIZ)
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
kvecs = [uniformKnots(degs[0],0.0,1.0,Nel,False),
         uniformKnots(degs[1],0.0,1.0,Nel,False)]

# Define a trivial mapping from parametric to physical space, via explicit
# B-spline.
controlMesh = ExplicitBSplineControlMesh(degs,kvecs)

# Define the spaces for RT-type compatible splines on this geometry.
fieldList = generateFieldsCompat(controlMesh,"RT",degs)
# Include an extra scalar field for the fine-scale pressure.
fieldList += [BSpline(degs,kvecs),]
splineGenerator = FieldListSpline(controlMesh,fieldList)

# Apply strong BCs to both components of velocity on all boundaries.
for field in range(0,2):
    scalarSpline = splineGenerator.getFieldSpline(field)
    for side in range(0,2):
        for direction in range(0,2):
            sideDofs = scalarSpline.getSideDofs(direction,side)
            splineGenerator.addZeroDofs(field,sideDofs)

####### Analysis #######

if(mpirank==0):
    print("Setting up extracted spline...")

# Overkill quadrature to ensure optimal convergence:
QUAD_DEG = 2*(max(degs)+1)
spline = ExtractedSpline(splineGenerator,QUAD_DEG)

# Define the exact solution:
x = spline.spatialCoordinates()
u0_exact = 8.0*(pow(x[0],4) - 2.0*pow(x[0],3) + pow(x[0],2))\
           *(4.0*pow(x[1],3) - 2.0*x[1])
u1_exact = -8.0*(4.0*pow(x[0],3) - 6.0*pow(x[0],2) + 2.0*x[0])\
           *(pow(x[1],4) - pow(x[1],2))
u_exact = as_vector((u0_exact,u1_exact))
p_exact = sin(pi*x[0])*sin(pi*x[1])

# Manufacture a source term, using the strong form of the PDE system:
nu = 1.0/Re
def Ladv(u,v,q):
    return spline.grad(v)*u + spline.grad(q)
def eps(u):
    return 0.5*(spline.grad(u) + spline.grad(u).T)
def sigmaVisc(u):
    return 2.0*nu*eps(u)
def Lvisc(u):
    return -spline.div(sigmaVisc(u))

f = Ladv(u_exact,u_exact,p_exact) + Lvisc(u_exact)

if(mpirank==0):
    print("Starting analysis...")

# "Un-pack" a velocity--pressure pair from spline.V, which is just a mixed
# space with $d+1$ scalar fields.
def unpack(up):
    d = spline.mesh.geometry().dim()
    u_hat = as_vector([up[i] for i in range(0,d)])
    p_hat = up[-1]
    return u_hat, p_hat

# Note on variable naming:  The "fine-scale pressure" here is in fact treated
# in the formulation as $p^h + p'$ when it is used in the stabilization terms,
# in which sense the name `p` makes sense (rather than, say, `pPrime`).  The
# coarse scale pressure is not included in the formulation, as the coarse-scale
# velocity is constrained to be solenoidal by the use of an iterated penaly
# solver.

# The unknown parametric velocity and fine-scale pressure:
up_hat = Function(spline.V)
u_hat, p_hat = unpack(up_hat)

# The physical velocity and its temporal partial derivative:
u = cartesianPushforwardRT(u_hat,spline.F)
p = cartesianPushforwardW(p_hat,spline.F)

# The parametric and physical test functions:
vq_hat = TestFunction(spline.V)
v_hat, q_hat = unpack(vq_hat)
v = cartesianPushforwardRT(v_hat,spline.F)
q = cartesianPushforwardW(q_hat,spline.F)

# Contribution to the weak problem for given test function; plugging u into
# this as the test function will be considered the "resolved dissipation".
def resVisc(v):
    return inner(sigmaVisc(u),eps(v))*spline.dx

# The (steady) material time derivative of the velocity:
Du_Dt = spline.grad(u)*u

# The problem is posed on a solenoidal subspace, as enforced by the iterative
# penalty solver; no pressure terms are necessary in the weak form.
resGalerkin = inner(Du_Dt,v)*spline.dx + resVisc(v) - inner(f,v)*spline.dx

# Extra term associated with stabilization.  This technically leaves
# an un-determined hydrostatic mode in the fine-scale
# pressure, but we can let the iterative solver choose it for us with no
# effect on the velocity solution.
resStrong = Du_Dt + Lvisc(u) + spline.grad(p) - f
dxi_dxiHat = 0.5*ufl.Jacobian(spline.mesh)
dx_dxi = spline.parametricGrad(spline.F)
dx_dxiHat = dx_dxi*dxi_dxiHat
Ginv = dx_dxiHat.T*dx_dxiHat
G = inv(Ginv)
C_I = Constant(3.0*max(degs)**2)
tau_M = 1.0/sqrt(dot(u,G*u) + (C_I**2)*(nu**2)*inner(G,G))
uPrime = -tau_M*resStrong
resStab = inner(uPrime,-Ladv(u,v,q))*spline.dx \
          + inner(v,spline.grad(u)*uPrime)*spline.dx \
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

# Higher-than-default tolerance may need to be even tighter to observe
# optimal convergence for kPrime > 2 and/or Nel > 64.  
spline.relativeTolerance = NONLIN_TOL

if(mpirank==0):
    print("Projecting exact solution to set boundary conditions...")

# Assign the exact solution as an initial guess, to impose boundary
# conditions.  (This is not "cheating", since the discretization is not
# equivalent to $L^2$ projection of the exact solution, so there will
# be some nontrivial residual.)
up_hat.assign(divFreeProject(u_exact,spline,
                             getVelocity=lambda up : unpack(up)[0],
                             getOtherFields=lambda up : unpack(up)[1],
                             applyBCs=False))

if(mpirank==0):
    print("Solving system...")

# Solve for the coarse-scale velocity and fine-scale pressure:
iteratedDivFreeSolve(res,up_hat,vq_hat,spline,
                     divOp=divOp,
                     penalty=Constant(penalty),
                     J=Dres,reuseLHS=False)

# Compute and print error:
def H1err(u,u_exact,w=Constant(1.0)):
    e_u = u - u_exact
    return math.sqrt(assemble(w*inner(spline.grad(e_u),
                                      spline.grad(e_u))*spline.dx))
err_u_H1 = H1err(u,u_exact)
err_p_H1 = H1err(p,p_exact,w=tau_M)

if(mpirank==0):
    print("log(h) = "+str(math.log(1.0/Nel)))
    print("log(H^1 velocity error) = " + str(math.log(err_u_H1)))
    print("log(tau-weighted H^1 pressure error) = " + str(math.log(err_p_H1)))

if(VIZ):
    # Take advantage of explicit B-spline geometry to simplify visualization:
    ux, uy, p = up_hat.split()
    ux.rename("u","u")
    uy.rename("v","v")
    File("results-dcb/ux.pvd") << ux
    File("results-dcb/uy.pvd") << uy
