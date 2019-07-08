"""
Div-conforming B-spline discretization of 3D Taylor--Green flow, using the
method of subgrid vortices.
"""

from tIGAr import *
from tIGAr.compatibleSplines import *
from tIGAr.BSplines import *
from tIGAr.timeIntegration import *
import math
import ufl

# Re-ordering of DoFs causes FunctionSpace creation to slow down dramatically
# in larger problems.  These parameters partially alleviate the issue.
parameters['reorder_dofs_serial'] = False
parameters['dof_ordering_library'] = 'random'

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
parser.add_argument('--Nel',dest='Nel',default=32,
                    help='Number of elements in each direction.')
parser.add_argument('--N_STEPS_over_Nel',
                    dest='N_STEPS_over_Nel',default=8,
                    help='Ratio of number of time steps to Nel.')
parser.add_argument('--kPrime',dest='kPrime',default=1,
                    help='Degree up to which velocity is complete.')
parser.add_argument('--Re',dest='Re',default=1600.0,
                    help='Reynolds number.')
parser.add_argument('--T',dest='T',default=10.0,
                    help='Length of time interval to consider.')
parser.add_argument('--QUAD_REDUCE',dest='QUAD_REDUCE',default=0,
                    help='Degree by which to reduce quadrature accuracy.')
parser.add_argument('--VIZ',dest='VIZ',action='store_true',
                    help='Include option to output visualization files.')
parser.add_argument('--GALERKIN',dest='GALERKIN',action='store_true',
                    help='Include option to output visualization files.')
parser.add_argument('--MAX_KSP_IT',dest='MAX_KSP_IT',default=1000,
                    help='Maximum number of Krylov iterations.')
parser.add_argument('--LINEAR_TOL',dest='LINEAR_TOL',default=1e-3,
                    help='Relative tolerance for Krylov solves.')
parser.add_argument('--NONLIN_TOL',dest='NONLIN_TOL',default=1e-3,
                    help='Relative tolerance for nonlinear solves.')
parser.add_argument('--penalty',dest='penalty',default=1e4,
                    help='Dimensionless penalty for iterated penalty solver.')
parser.add_argument('--RHO_INF',dest='RHO_INF',default=0.5,
                    help='Spectral radius of generalized-alpha integrator.')
parser.add_argument('--OUT_SKIP',dest='OUT_SKIP',default=10,
                    help='Number of steps to skip between writing files.')

args = parser.parse_args()
Nel = int(args.Nel)
kPrime = int(args.kPrime)
Re = Constant(float(args.Re))
TIME_INTERVAL = float(args.T)
VIZ = bool(args.VIZ)
GALERKIN = bool(args.GALERKIN)
MAX_KSP_IT = int(args.MAX_KSP_IT)
LINEAR_TOL = float(args.LINEAR_TOL)
NONLIN_TOL = float(args.NONLIN_TOL)
penalty = float(args.penalty)
N_STEPS_over_Nel = int(args.N_STEPS_over_Nel)
RHO_INF = float(args.RHO_INF)
OUT_SKIP = int(args.OUT_SKIP)
QUAD_REDUCE = int(args.QUAD_REDUCE)

####### Preprocessing #######

if(mpirank==0):
    print("Generating extraction data...")
    sys.stdout.flush()

# Polynomial degree in each direction.
degs = 3*[kPrime,]

# Knot vectors for defining the control mesh.
kvecs = [uniformKnots(degs[i],0.0,math.pi,Nel,False) for i in range(0,3)]

# Define a trivial mapping from parametric to physical space, via explicit
# B-spline.
controlMesh = ExplicitBSplineControlMesh(degs,kvecs)

# Define the spaces for RT-type compatible splines on this geometry.
fieldList = generateFieldsCompat(controlMesh,"RT",degs)
# Include an extra scalar field for the fine-scale pressure.
fieldList += [BSpline(degs,kvecs),]
splineGenerator = FieldListSpline(controlMesh,fieldList)

# Apply strong BCs in parametric normal directions. 
for field in range(0,3):
    scalarSpline = splineGenerator.getFieldSpline(field)
    for side in range(0,2):
        sideDofs = scalarSpline.getSideDofs(field,side)
        splineGenerator.addZeroDofs(field,sideDofs)
        
####### Analysis #######

if(mpirank==0):
    print("Setting up extracted spline...")

# Important to use sufficient quadrature:
QUAD_DEG = 2*(max(degs)+1)-QUAD_REDUCE
spline = ExtractedSpline(splineGenerator,QUAD_DEG)

if(mpirank==0):
    print("Starting analysis...")

# Parameters of the time discretization:
N_STEPS = N_STEPS_over_Nel*Nel
DELTA_T = Constant(TIME_INTERVAL/float(N_STEPS))

# Define the viscosity based on the desired Reynolds number.
nu = Constant(1.0/Re)

# The initial condition for the flow:
x = spline.spatialCoordinates()
soln0 = sin(x[0])*cos(x[1])*cos(x[2])
soln1 = -cos(x[0])*sin(x[1])*cos(x[2])
soln = as_vector([soln0,soln1,Constant(0.0)])

# For 3D computations, use an iterative solver.
spline.linearSolver = PETScKrylovSolver("gmres","jacobi")
spline.linearSolver.parameters["relative_tolerance"] = LINEAR_TOL
# Linear solver sometimes fails to converge, but convergence of nonlinear
# iteration is still enforced to within spline.relativeTolerance.
spline.linearSolver.parameters["error_on_nonconvergence"] = False
spline.linearSolver.parameters["maximum_iterations"] = MAX_KSP_IT
spline.linearSolver.ksp().setGMRESRestart(MAX_KSP_IT)
spline.relativeTolerance = NONLIN_TOL

# "Un-pack" a velocity--pressure pair from spline.V, which is just a mixed
# space with four scalar fields.
def unpack(up):
    d = spline.mesh.geometry().dim()
    u_hat = as_vector([up[i] for i in range(0,d)])
    p_hat = up[-1]
    return u_hat, p_hat

# The unknown parametric velocity and fine-scale pressure:
up_hat = Function(spline.V)
u_hat, p_hat = unpack(up_hat)

# Parametric velocity at the old time level:
up_old_hat = Function(spline.V)

# Parametric $\partial_t u$ at the old time level.  (Note that the suffix "dot"
# is a convention coming from the origins of generalized-$\alpha$ in structural
# problems, and does not refer to a material time derivative here.)
updot_old_hat = Function(spline.V)

# Create a generalized-alpha time integrator.
timeInt = GeneralizedAlphaIntegrator(RHO_INF,DELTA_T,up_hat,
                                     (up_old_hat, updot_old_hat))

# The alpha-level parametric velocity and its partial derivative w.r.t. time:
up_hat_alpha = timeInt.x_alpha()
updot_hat_alpha = timeInt.xdot_alpha()

# A helper function to take the symmetric gradient:
def eps(u):
    return 0.5*(spline.grad(u) + spline.grad(u).T)

# The physical velocity and its temporal partial derivative:
u = cartesianPushforwardRT(unpack(up_hat_alpha)[0],spline.F)
udot = cartesianPushforwardRT(unpack(updot_hat_alpha)[0],spline.F)
p = cartesianPushforwardW(p_hat,spline.F)

# The parametric and physical test functions:
vq_hat = TestFunction(spline.V)
v_hat, q_hat = unpack(vq_hat)
v = cartesianPushforwardRT(v_hat,spline.F)
q = cartesianPushforwardW(q_hat,spline.F)

# The material time derivative of the velocity:
Du_Dt = udot + spline.grad(u)*u

# The viscous part of the Cauchy stress:
sigmaVisc = 2.0*nu*eps(u)

# Contribution to the weak problem for given test function; plugging u into
# this as the test function will be considered the "resolved dissipation".
def resVisc(v):
    return inner(sigmaVisc,eps(v))*spline.dx

# The problem is posed on a solenoidal subspace, as enforced by the iterative
# penalty solver; no pressure terms are necessary in the weak form.
resGalerkin = inner(Du_Dt,v)*spline.dx + resVisc(v)

# Extra term associated with SUPG stabilization.  This technically leaves
# an un-determined hydrostatic mode in the fine-scale
# pressure, but we can let the iterative solver choose it for us with no
# effect on the velocity solution.
resStrong = Du_Dt - spline.div(sigmaVisc) + spline.grad(p)
def Ladv(v,q):
    return spline.grad(v)*u + spline.grad(q)

# Mesh size information:
dxi_dxiHat = 0.5*ufl.Jacobian(spline.mesh)
dx_dxi = spline.parametricGrad(spline.F)
dx_dxiHat = dx_dxi*dxi_dxiHat
Ginv = dx_dxiHat.T*dx_dxiHat
G = inv(Ginv)

# Defining the stabilization constants:
C_I = Constant(3.0*max(degs)**2)
tau_M = 1.0/sqrt(dot(u,G*u) + 4.0/DELTA_T**2 + ((C_I*nu)**2)*inner(G,G))
tau_C = 1.0/(tau_M*tr(G))

# Fine scale velocity using the quasi-static subscale model:
uPrime = -tau_M*resStrong

# We need to be able to plug in the velocity solution as a test function
# later, to compute model dissipation.  
def resSUPG(v,q,uPrime):
    return inner(uPrime,-Ladv(v,q))*spline.dx 
def resVMS(v,q,uPrime):
    return inner(v,spline.grad(u)*uPrime)*spline.dx \
        - inner(spline.grad(v),outer(uPrime,uPrime))*spline.dx
def resModel(v,q,uPrime):
    return resSUPG(v,q,uPrime) + resVMS(v,q,uPrime)

# Residual of the full formulation and consistent linearization:
res = resGalerkin
if(not GALERKIN):
    res += resModel(v,q,uPrime)
else:
    res += inner(p,q)*spline.dx # (Pin down redundant field.)
Dres = derivative(res, up_hat)

# Divergence of the velocity field, given a function in the mixed space.
# It is weighted to make the penalty dimensionless and indepenent of position,
# as needed for the update step of the iterated
# penalty solver.
divOp = lambda up : sqrt(tau_C)*spline.div(cartesianPushforwardRT
                                           (unpack(up)[0],spline.F))

# Auxiliary Function to re-use during the iterated penalty solves:
w = Function(spline.V)

# Projection of initial condition; need to specify how to get the (parametric)
# velocity vector field from a function in the mixed space.
if(mpirank==0):
    print("Projecting velocity IC...")
up_old_hat.assign(divFreeProject(soln,spline,
                                 getVelocity=lambda up : unpack(up)[0]))
up_hat.assign(up_old_hat)

# Files for optional ParaView output:
if(VIZ):
    uFile = File("results/ux.pvd")
    vFile = File("results/uy.pvd")
    wFile = File("results/uz.pvd")

# Time stepping loop:
for i in range(0,N_STEPS):

    if(mpirank == 0):
        print("\n------- Time step "+str(i+1)+"/"+str(N_STEPS)
              +" , t = "+str(timeInt.t)+" -------\n")
        sys.stdout.flush()

    # Output ParaView files if desired.
    if(VIZ and (i%OUT_SKIP==0)):
        # Take advantage of explicit B-spline geometry to simplify
        # visualization.
        ux, uy, uz, _ = up_hat_old.split()
        ux.rename("u","u")
        uy.rename("v","v")
        uz.rename("w","w")
        uFile << ux
        vFile << ux
        wFile << uz
        
    # Solve for velocity in a solenoidal subspace of the RT-type
    # B-spline space, where divergence acts on the velocity unknowns in the
    # parametric domain.
    iteratedDivFreeSolve(res,up_hat,vq_hat,spline,
                         divOp=divOp,
                         penalty=Constant(penalty),
                         J=Dres,w=w)

    # Assemble the dissipation rates, and append them to a file that can be
    # straightforwardly plotted as a function of time using gnuplot.
    dissipationScale = (1.0/pi**3)
    resolvedDissipationRate = assemble(dissipationScale*resVisc(u))
    if(GALERKIN):
        modelDissipationRate = 0.0
    else:
        modelDissipationRate = assemble(dissipationScale
                                        *resModel(u,Constant(0.0)*p,uPrime))
    dissipationRate = resolvedDissipationRate + modelDissipationRate

    # Because the algebraic problem is solved only approximately, there is a
    # nonzero divergence to the velocity field.  If the tolerances are set
    # small enough and/or penalty set high enough, this can be driven down
    # to machine precision.  
    divError = math.sqrt(assemble((spline.div(u)**2)*spline.dx))
    
    if(mpirank==0):
        print("Divergence error ($L^2$): "+str(divError))
        mode = "a"
        if(i==0):
            mode = "w"
        outFile = open("dissipationRate.dat",mode)
        outFile.write(str(timeInt.t)+" "+str(dissipationRate)+" "
                      +str(modelDissipationRate)+"\n")
        outFile.close()

    # Move to the next time step.
    timeInt.advance()
