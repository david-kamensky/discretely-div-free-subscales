"""
3D Taylor--Green vortex, using the Taylor--Hood element.  Subscales can be
quasi-static or dynamic.  (See command line parameters.)  In 3D, an
"iterated perturbed-Lagrangian" solver is used, based on the iterated
penalty solver applied in the conforming setting.  
"""

from dolfin import *
import ufl
import math
mpirank = MPI.rank(MPI.comm_world)

# Re-ordering of DoFs causes FunctionSpace creation to slow down dramatically
# in larger problems.  These parameters partially alleviate the issue.
parameters['reorder_dofs_serial'] = False
parameters['dof_ordering_library'] = 'random'

# Suppress warnings about Krylov solver non-convergence:
set_log_level(40)

# Use TSFC representation, to handle quadrature elements.  (The expression
# that must be projected to quadrature points is too complicated for the
# deprecated quadrature representation.)
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
parser.add_argument('--N_STEPS_over_Nel',dest='N_STEPS_over_Nel',default=8,
                    help='Number of elements in each direction.')
parser.add_argument('--MAX_IT',dest='MAX_IT',default=50,
                    help='Maximum iterations for the nonlinear solver.')
parser.add_argument('--MAX_KSP_IT',dest='MAX_KSP_IT',default=500,
                    help='Maximum iterations for the Krylov solver.')
parser.add_argument('--Re',dest='Re',default=1600.0,
                    help='Reynolds number.')
parser.add_argument('--T',dest='T',default=10.0,
                    help='Length of time interval to consider.')
parser.add_argument('--penalty',dest='penalty',default=1.0e3,
                    help='Dimensionless penalty for iterated solver.')
parser.add_argument('--NONLIN_TOL',dest='NONLIN_TOL',default=1.0e-3,
                    help='Relative tolerance for iterated solver.')
parser.add_argument('--LINEAR_TOL',dest='LINEAR_TOL',default=1.0e-2,
                    help='Relative tolerance for Krylov solver.')
parser.add_argument('--QS_SUBSCALES',
                    dest='QS_SUBSCALES',action='store_true',
                    help='Include option to use quasi-static subscales.')
parser.add_argument('--VIZ',dest='VIZ',action='store_true',
                    help='Include option to output visualization files.')

args = parser.parse_args()
Nel = int(args.Nel)
N_STEPS_over_Nel = int(args.N_STEPS_over_Nel)
MAX_IT = int(args.MAX_IT)
MAX_KSP_IT = int(args.MAX_KSP_IT)
Re = Constant(float(args.Re))
T = float(args.T)
penalty = Constant(float(args.penalty))
NONLIN_TOL = float(args.NONLIN_TOL)
LINEAR_TOL = float(args.LINEAR_TOL)
VIZ = bool(args.VIZ)
DYN_SUBSCALES = (not bool(args.QS_SUBSCALES))

##########################

# Quadrature degree to use in all integrals:
QUAD_DEG = 6

# Use fixed quadrature rule.  (Automatic degree estimation skyrockets
# due to the residual-based stabilization terms.)
dx = dx(metadata={"quadrature_degree":QUAD_DEG})

# Mesh and derivation of mesh-size metric:
mesh = BoxMesh(Point(0.0,0.0,0.0),
               Point(math.pi,math.pi,math.pi),
               Nel,Nel,Nel)
dx_dxiHat = 0.5*ufl.Jacobian(mesh)
G = inv(dx_dxiHat.T*dx_dxiHat)

# Definition of the Taylor--Hood element, augmented with the fine-scale
# pressure:
cell = mesh.ufl_cell()
u_el = VectorElement("Lagrange",cell,2,
                     quad_scheme="default")
p_el = FiniteElement("Lagrange",cell,1,
                     quad_scheme="default")
pPrime_el = p_el

# Unfortunately, a mixed element including the quadrature representation of
# uPrime does not work, as requesting derivatives of any of the other
# components leads to the form compiler attempting to differentiate all
# components, including the quadrature space for which derivatives are
# ill-defined.
X_el = MixedElement([u_el,p_el,pPrime_el])
X = FunctionSpace(mesh,X_el)

# Separate space to store old fine-scale velocity.  (Current fine-scale
# velocity can be expressed symbolically in terms of this and other current
# quantities, in a static condensation.)
uPrime_el = VectorElement("Quadrature",cell,QUAD_DEG,
                          quad_scheme="default")
VPrime = FunctionSpace(mesh,uPrime_el)

# Solution and test function:
w = Function(X)
uh,ph_penalty,pPrime = split(w)
w_accum = Function(X)
_,ph_accum,_ = split(w_accum)
ph = ph_accum + ph_penalty
w_old = Function(X)
uh_old, ph_old, pPrime_old = split(w_old)
uPrime_old = Function(VPrime)
dw = TestFunction(X)
v,q,qPrime = split(dw)

# Initial condition for the Taylor--Green vortex
x = SpatialCoordinate(mesh)
u_IC0 = "sin(x[0])*cos(x[1])*cos(x[2])"
u_IC1 = "-cos(x[0])*sin(x[1])*cos(x[2])"
u_IC = Expression((u_IC0,u_IC1,"0","0","0"),degree=3)

# Viscous stress and advection operator:
nu = 1.0/Re
def sigmaVisc(u):
    return 2.0*nu*sym(grad(u))
def Ladv(u,v):
    return dot(grad(v),u)

# Time derivatives:
N_STEPS = N_STEPS_over_Nel*Nel
Dt = Constant(T/N_STEPS)
uh_t = (uh - uh_old)/Dt

# Midpoint coarse-scale velocity:
uh_mid = 0.5*(uh + uh_old)

# Define Forms for use in the formulation:
def c(v1,v2,v3):
    return dot(dot(grad(v2),v1),v3)*dx
def k(v1,v2):
    return inner(sigmaVisc(v1),sym(grad(v2)))*dx
def b(v,q):
    return div(v)*q*dx
def c_cons(a,v1,v2):
    return -dot(v1,dot(grad(v2),a))*dx
def c_skew(a,v1,v2):
    return 0.5*(c(a,v1,v2) + c_cons(a,v1,v2))

# Definition of stabilization parameters:
C_I = Constant(36.0)
tau_M_denom2 = dot(uh_mid,G*uh_mid) + (C_I**2)*(nu**2)*inner(G,G)
if(not DYN_SUBSCALES):
    tau_M_denom2 += 4.0/(Dt**2)
tau_M = 1.0/sqrt(tau_M_denom2)
tau_C = 1.0/(tau_M*tr(G))

# Residual of the strong problem
r_M = uh_t + Ladv(uh_mid,uh_mid) - div(sigmaVisc(uh_mid)) + grad(ph)

if(DYN_SUBSCALES):
    # Static condensation of fine-scale velocity at current time level, i.e.,
    # symbolically inverting the (block-diagonal) $v'$ set of equations:
    I = Identity(mesh.geometry().dim())
    uPrimeLHS = (1.0/Dt + 0.5/tau_M)*I + 0.5*grad(uh_mid)
    uPrimeRHS = -r_M - grad(pPrime) + (1.0/Dt)*uPrime_old \
                - (0.5/tau_M)*uPrime_old - 0.5*grad(uh_mid)*uPrime_old
    uPrime = inv(uPrimeLHS)*uPrimeRHS
    
    # Midpoint fine-scale velocity:
    uPrime_mid = 0.5*(uPrime + uPrime_old)
else:
    uPrime_mid = -tau_M*(grad(pPrime) + r_M)
    

# Model residual with interchangeable test function:
def resModel(v):
    return c_cons(uh_mid,uPrime_mid,v) + c_skew(uPrime_mid,uh_mid,v) \
        + c_cons(uPrime_mid,uPrime_mid,v) \
        + tau_C*div(uh_mid)*div(v)*dx

# Coarse-scale subproblem:
res_coarse = inner(uh_t,v)*dx \
             + c_skew(uh_mid,uh_mid,v) + k(uh_mid,v) \
             - b(v,ph) + b(uh_mid,q) \
             + resModel(v)

# Remaining fine-scale subproblem, after statically condensing-out
# the $v'$ equations:
res_fine = -inner(grad(qPrime),uPrime_mid)*dx

# Pressure penalty term for iterated solver:
res_penalty = (1.0/(penalty*tau_C))*ph_penalty*q*dx

# Nonlinear formulation in residual form:
res = res_coarse + res_fine + res_penalty

# Consistent tangent:
Dres = derivative(res,w)

# BCs: Use Dirichlet for normal components of the coarse-scale velocity,
# and pin down fine scale pressure in one corner to remove hydrostatic modes.
# (Coarse scale pressure is selected by the iterative solver.)
#
# Note: In this example the BCs must be homogeneous.  The reason is that they
# are applied directly to the increment in a nonlinear solver algorithm.
# (Inhomogeneous BCs can still be applied in such a setting by initializing the
# iterative algorithm to a lifting of the desired boundary data.)
bcs = [DirichletBC(X.sub(0).sub(i),Constant(0.0),
                   "near(x["+str(i)+"],0.0) || near(x["+str(i)+"],pi)")
       for i in range(0,3)]
corner_str = "near(x[0],0.0) && near(x[1],0.0) && near(x[2],0.0)"
bcs += [DirichletBC(X.sub(2),Constant(0.0),corner_str,"pointwise"),]

# Lumped mass for efficient projections to quadrature spaces:
def lumpedProject(f,V):
    v = TestFunction(V)
    s = ufl.shape(v)
    if(len(s)>0):
        lhsTrial = as_vector(s[0]*(Constant(1.0),))
    else:
        lhsTrial = Constant(1.0)
    lhs = assemble(inner(lhsTrial,v)*dx)
    rhs = assemble(inner(f,v)*dx)
    u = Function(V)
    as_backend_type(u.vector())\
        .vec().pointwiseDivide(as_backend_type(rhs).vec(),
                               as_backend_type(lhs).vec())
    return u

# Project the initial condition:
w_old.interpolate(u_IC)
uPrime_old.assign(lumpedProject(as_vector((u_IC[0],u_IC[1],u_IC[2]))
                                - uh_old,VPrime))
w.assign(w_old)

# Linear solver for iterative approach:
linearSolver = PETScKrylovSolver("gmres","jacobi")
linearSolver.ksp().setGMRESRestart(MAX_KSP_IT)
linearSolver.parameters['relative_tolerance'] = LINEAR_TOL
linearSolver.parameters['maximum_iterations'] = MAX_KSP_IT
linearSolver.parameters['error_on_nonconvergence'] = False

# Time stepping loop:
t = 0.0
for step in range(0,N_STEPS):
    t += float(Dt)
    if(mpirank==0):
        print("======= Time step "+str(step+1)+"/"+str(N_STEPS)+" =======")

    # Iterated solve:
    for i in range(0,MAX_IT):
        B = assemble(res)
        A = assemble(Dres)
        for bc in bcs:
            bc.apply(A,B)
        currentNorm = norm(B)
        if(i==0):
            initialNorm = currentNorm
        relativeNorm = currentNorm/initialNorm
        if(mpirank == 0):
            print("Solver iteration: "+str(i)+" , Relative norm: "
                  + str(relativeNorm))
            sys.stdout.flush()
        if(currentNorm/initialNorm < NONLIN_TOL):
            converged = True
            break
        if(i==MAX_IT-1):
            print("ERROR: Iterated penalty solver failed to converge")
            exit()
        dw = Function(X)
        linearSolver.solve(A,dw.vector(),B)
        w.assign(w-dw)
        w_accum.assign(w_accum + w)

    # Evaluate and output the dissipation rate:
    dissipationScale = (1.0/pi**3)
    modelDissipationRate = dissipationScale*assemble(resModel(uh_mid)
                                                + k(uh_mid,uh_mid))
    resolvedDissipationRate = dissipationScale*assemble(k(uh_mid,uh_mid))
    dissipationRate = resolvedDissipationRate + modelDissipationRate    

    # Check the discrete continuity residual:
    discreteDivErr = norm(assemble(b(uh_mid,q)))
    if(mpirank==0):
        print("Discrete continuity residual norm = "+str(discreteDivErr))
        mode = "a"
        if(step==0):
            mode = "w"
        outFile = open("dissipationRate.dat",mode)
        outFile.write(str(t)+" "+str(dissipationRate)+" "
                      +str(modelDissipationRate)+"\n")
        outFile.close()
    
    if(DYN_SUBSCALES):
        uPrime_old.assign(lumpedProject(uPrime,VPrime))

    # Move coarse scale unknowns to next time step:
    w_old.assign(w)
