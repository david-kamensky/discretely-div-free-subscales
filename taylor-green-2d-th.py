"""
2D Taylor--Green vortex, using the Taylor--Hood element.  Subscales can be
quasi-static or dynamic.  (See command line parameters.)
"""

from dolfin import *
import ufl
import math
import numpy as np
import scipy.io as sio

mpirank = MPI.rank(MPI.comm_world)

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
parser.add_argument('--Nel',dest='Nel',default=16,
                    help='Number of elements in each direction.')
parser.add_argument('--Re',dest='Re',default=100.0,
                    help='Reynolds number.')
parser.add_argument('--T',dest='T',default=1.0,
                    help='Length of time interval to consider.')
parser.add_argument('--QS_SUBSCALES',
                    dest='QS_SUBSCALES',action='store_true',
                    help='Include option to use quasi-static subscales.')
parser.add_argument('--VIZ',dest='VIZ',action='store_true',
                    help='Include option to output visualization files.')
parser.add_argument('--KE',dest='KE',action='store_true',
                    help='Include option to track Kinetic Energy.')

args = parser.parse_args()
Nel = int(args.Nel)
Re = Constant(float(args.Re))
T = float(args.T)
VIZ = bool(args.VIZ)
KE = bool(args.KE)
DYN_SUBSCALES = (not bool(args.QS_SUBSCALES))


# Print output to tell user if dynamic subscales are on or off.
if(DYN_SUBSCALES):
    print("Dynamic subscales are ON.")
else:
    print("Dynamic subscales are OFF. Formulation is now using quasi-static subscales.")

# Print output to tell user if kinetic energy is being tracked.
if(KE):
    print("Kinetic energy will be computed and tracked over time.")
    # Generate kinetic energy storage matrix that will be returned once the entire script has finished running:
    KE_Storage = np.empty([Nel,1])
else:
    print("Kinetic energy will not be computed.")



##########################

# Quadrature degree to use in all integrals:
QUAD_DEG = 6

# Use fixed quadrature rule.  (Automatic degree estimation skyrockets
# due to the residual-based stabilization terms.)
dx = dx(metadata={"quadrature_degree":QUAD_DEG})

# Mesh and derivation of mesh-size metric:
mesh = RectangleMesh(Point(-math.pi,-math.pi),
                     Point(math.pi,math.pi),
                     Nel,Nel)
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
uh,ph,pPrime = split(w)
w_old = Function(X)
uh_old, ph_old, pPrime_old = split(w_old)
uPrime_old = Function(VPrime)
dw = TestFunction(X)
v,q,qPrime = split(dw)

# Initial condition for the Taylor--Green vortex
x = SpatialCoordinate(mesh)
u_IC = as_vector((sin(x[0])*cos(x[1]),-cos(x[0])*sin(x[1])))

p_IC = 0.25*(cos(2.0*x[0]) + cos(2.0*x[1]) - 2.0)

# Time dependence of exact solution:
solnt = Expression("exp(-2.0*nu*t)",nu=1.0/float(Re),t=0.0,degree=1)

# Viscous stress and advection operator:
nu = 1.0/Re
def sigmaVisc(u):
    return 2.0*nu*sym(grad(u))
def Ladv(u,v):
    return dot(grad(v),u)

# Time derivatives:
N_STEPS = Nel
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

# Equation (27) that appears in section 2.6:
def KEnergy(v1,v2):
    return 0.5*((math.sqrt(assemble(inner((v1 + v2),(v1 + v2))*dx)))**2)


# Definition of stabilization parameters:
C_I = Constant(60.0)
tau_M_denom2 = dot(uh_mid,G*uh_mid) + (C_I**2)*(nu**2)*inner(G,G)
if(not DYN_SUBSCALES):
    tau_M_denom2 += 4.0/(Dt**2)
tau_M = 1.0/sqrt(tau_M_denom2)
tau_C = 1.0/(tau_M*tr(G))

# Residual of the strong problem
r_M = uh_t + Ladv(uh_mid,uh_mid) - div(sigmaVisc(uh_mid)) + grad(ph)

# Choose which branch of coarse-scale residual to use, based on whether we are using dynamic or quasi-static subscales.
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

    # Define time derivative of uPrime:
    uPrime_t = (uPrime - uPrime_old)/Dt

    # Coarse-scale subproblem for dynamic subscales:
    res_coarse = inner(uh_t,v)*dx \
             + c_skew(uh_mid,uh_mid,v) + k(uh_mid,v) - b(v,ph) + b(uh_mid,q) \
             + c_cons(uh_mid,uPrime_mid,v) + c_skew(uPrime_mid,uh_mid,v) \
             + c_cons(uPrime_mid,uPrime_mid,v) + dot(uPrime_t,v)*dx \
             + tau_C*div(uh_mid)*div(v)*dx
else:
    # Quasi-static subscale definition of $u'$ found in Eq. (44):
    uPrime_mid = -tau_M*(grad(pPrime) + r_M)

    # Coarse-scale subproblem for quasi-static subscales:
    # $u'$ in this branch does not depend on time, so the "dot(uPrime_t,v)*dx" term is removed.
    res_coarse = inner(uh_t,v)*dx \
             + c_skew(uh_mid,uh_mid,v) + k(uh_mid,v) - b(v,ph) + b(uh_mid,q) \
             + c_cons(uh_mid,uPrime_mid,v) + c_skew(uPrime_mid,uh_mid,v) \
             + c_cons(uPrime_mid,uPrime_mid,v) \
             + tau_C*div(uh_mid)*div(v)*dx


# Remaining fine-scale subproblem, after statically condensing-out
# the $v'$ equations:
res_fine = -inner(grad(qPrime),uPrime_mid)*dx

# Nonlinear formulation in residual form:
res = res_coarse + res_fine

# Consistent tangent:
Dres = derivative(res,w)

# BCs: Use Dirichlet for normal components of the coarse-scale velocity,
# and pin down pressures in one corner to remove hydrostatic modes.
corner_str = "near(x[0],-pi,1e-10) && near(x[1],-pi,1e-10)"
bcs = [DirichletBC(X.sub(0).sub(0),Constant(0.0),
                   "near(x[0],-pi,1e-10) || near(x[0],pi,1e-10)"),
       DirichletBC(X.sub(0).sub(1),Constant(0.0),
                   "near(x[1],-pi,1e-10) || near(x[1],pi,1e-10)"),
       DirichletBC(X.sub(1),Constant(0.0),corner_str,"pointwise"),
       DirichletBC(X.sub(2),Constant(0.0),corner_str,"pointwise")]

# Project the initial condition:
w_old.assign(project(as_vector((u_IC[0],u_IC[1],
                                Constant(0.0),Constant(0.0))),X))
uPrime_old.assign(project(u_IC - uh_old,VPrime,
                          form_compiler_parameters
                          ={"quadrature_degree":QUAD_DEG}))

# Time stepping loop:
for step in range(0,N_STEPS):
    if(mpirank==0):
        print("======= Time step "+str(step+1)+"/"+str(N_STEPS)+" =======")

    # Implicit solve:
    solve(res==0,w,J=Dres,bcs=bcs)

    if(DYN_SUBSCALES):
        # Note that projection to quadrature points can be optimized, as shown
        # in Jeremy Bleyer's tutorial here:
        #
        #  https://comet-fenics.readthedocs.io/en/latest/tips_and_tricks.html#efficient-projection-on-dg-or-quadrature-spaces
        #
        # However, the following is most concise.  
        uPrime_old.assign(project(uPrime,VPrime,form_compiler_parameters
                                  ={"quadrature_degree":QUAD_DEG}))

    if(KE):
        # Compute kinetic energy:
        print("Current KE: "+str(KEnergy(uh_mid,uPrime_mid))+".")
        KE_Storage[step,0] = float(KEnergy(uh_mid,uPrime_mid))

    # Move coarse scale unknowns to next time step:
    w_old.assign(w)

# Check error in $H^1$:
grad_u_exact = solnt*grad(u_IC)
p_exact = (solnt**2)*p_IC
solnt.t = T
grad_e_u = grad(uh) - grad_u_exact
err_u_H1 = math.sqrt(assemble(inner(grad_e_u,grad_e_u)*dx))
err_p_L2 = math.sqrt(assemble(inner(ph-p_exact,ph-p_exact)*dx))
if(mpirank==0):
    print("======= Final Results =======")
    print("log(h) = "+str(math.log(1.0/Nel)))
    print("log(H^1 velocity error) = "+str(math.log(err_u_H1)))
    print("log(L^2 pressure error) = "+str(math.log(err_p_L2)))

#Output the required files to be read and processed.
output_file = open('copypasta-tg.txt','w')
output_file.write('Nel = '+str(Nel))
output_file.write(', h = '+str(1.0/Nel))
output_file.write(', H^1 velocity error = '+str(err_u_H1))
output_file.write(', L^2 pressure error = '+str(err_p_L2))
output_file.close()

# Print output to tell user if kinetic energy is being tracked.
if(KE):
    #Verify that Kinetic Energy decreased over time:

    #Define some "boolean" marker determining whether or not KE decreased. Default will be set to "Yes" as the theory expects.
    bool_KE = "Yes"

    #Simple for loop will check whether or not KE decreases over time. KE at time-step t+1 must be less than KE at time-step t.
    for t in range(0,Nel-1):
        bool_KE_Check = KE_Storage[t+1,0] - KE_Storage[t,0]
        #If KE at time-step t+1 is larger than KE at time-step t, then set the boolean marker to "No".
        if (bool_KE_Check > 0):
            bool_KE = "NO"

    #Print the answer to the question: "Did KE decrease or what"
    #Yes/No answer printed directed to the command line so that you don't have to look through the output .mat file manually.
    #If the answer output here is "No", then there is probably something wrong with the formulation.
    print("Did Kinetic Energy decrease over time: "+str(bool_KE))

    #Output a MATLAB matrix containing the tracked kinetic energy:
    sio.savemat('Kinetic Energy Regular.mat', {'KE_Storage': KE_Storage})


# Output ParaView files as a sanity check, if desired.
if(VIZ):
   uh,ph,pPrime = w.split()
   uh.rename("u","u")
   File("results/u.pvd") << uh
   ph.rename("p","p")
   File("results/p.pvd") << ph
