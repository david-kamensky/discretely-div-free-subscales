"""
Regularized lid-driven cavity, using the Taylor--Hood element.
"""

from dolfin import *
import ufl

####### Parameters #######

# Arguments are parsed from the command line, with some hard-coded defaults
# here.  See help messages for descriptions of parameters.

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--Nel',dest='Nel',default=16,
                    help='Number of elements in each direction.')
parser.add_argument('--Re',dest='Re',default=100.0,
                    help='Reynolds number.')
parser.add_argument('--VIZ',dest='VIZ',action='store_true',
                    help='Include option to output visualization files.')

args = parser.parse_args()
Nel = int(args.Nel)
Re = Constant(float(args.Re))
VIZ = bool(args.VIZ)

##########################

# Quadrature degree to use in all integrals:
QUAD_DEG = 6

# Use fixed quadrature rule.  (Automatic degree estimation skyrockets
# due to the residual-based stabilization terms.)
dx = dx(metadata={"quadrature_degree":QUAD_DEG})

# Mesh and derivation of mesh-size metric:
mesh = UnitSquareMesh(Nel,Nel)
dx_dxiHat = 0.5*ufl.Jacobian(mesh)
G = inv(dx_dxiHat.T*dx_dxiHat)

# Definition of the Taylor--Hood element, augmented with the fine-scale
# pressure:
cell = mesh.ufl_cell()
u_el = VectorElement("Lagrange",cell,2)
p_el = FiniteElement("Lagrange",cell,1)
pPrime_el = p_el
X_el = MixedElement([u_el,p_el,pPrime_el])
X = FunctionSpace(mesh,X_el)

# Solution and test function:
w = Function(X)
uh,ph,pPrime = split(w)
dw = TestFunction(X)
v,q,qPrime = split(dw)

# Define the exact solution; we need it as an Expression for applying BCs,
# but it is more convenient as UFL for manufacturing solutions, hence the
# initial definition as a string, and subsequent use of eval() to obtain
# a UFL definition.
u0_str = "8.0*(pow(x[0],4) - 2.0*pow(x[0],3) + pow(x[0],2))"\
          +"*(4.0*pow(x[1],3) - 2.0*x[1])"
u1_str = "-8.0*(4.0*pow(x[0],3) - 6.0*pow(x[0],2) + 2.0*x[0])"\
         +"*(pow(x[1],4) - pow(x[1],2))"
# (Degree 8 interpolation captures all monomials of the exact solution.)
u_expr = Expression((u0_str,u1_str),degree=8,cell=cell)
x = SpatialCoordinate(mesh)
u = as_vector((eval(u0_str),eval(u1_str)))
p = sin(pi*x[0])*sin(pi*x[1])

# Manufacture the corresponding source term:
nu = 1.0/Re
def sigmaVisc(u):
    return 2.0*nu*sym(grad(u))
def Ladv(u,v):
    return dot(grad(v),u)
f = Ladv(u,u) - div(sigmaVisc(u)) + grad(p)

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
def F(v):
    return inner(f,v)*dx

# Definition of fine-scale velocity:
C_I = Constant(60.0)
tau_M = 1.0/sqrt(dot(uh,G*uh) + (C_I**2)*(nu**2)*inner(G,G))
tau_C = 1.0/(tau_M*tr(G))
r_M = Ladv(uh,uh) - div(sigmaVisc(uh)) + grad(ph) - f
uPrime = -tau_M*(grad(pPrime) + r_M)

# Nonlinear formulation in residual form:
res = c_skew(uh,uh,v) + k(uh,v) - b(v,ph) + b(uh,q) \
      + c_cons(uh,uPrime,v) + c_skew(uPrime,uh,v) \
      + c_cons(uPrime,uPrime,v) \
      + tau_C*div(uh)*div(v)*dx - F(v) \
      - inner(grad(qPrime),uPrime)*dx

# BCs: Use Dirichlet everywhere on velocity and pin down pressures in
# one corner to remove hydrostatic modes.
corner_str = "near(x[0],0.0) && near(x[1],0.0)"
bcs = [DirichletBC(X.sub(0),u_expr,"on_boundary"),
       DirichletBC(X.sub(1),Constant(0.0),corner_str,"pointwise"),
       DirichletBC(X.sub(2),Constant(0.0),corner_str,"pointwise")]

# Nonlinear solve:
solve(res==0,w,J=derivative(res,w),bcs=bcs)

# Check error in $H^1$:
import math
e_u = u - uh
err_u_H1 = math.sqrt(assemble(inner(grad(e_u),grad(e_u))*dx))
err_p_L2 = math.sqrt(assemble(inner(ph-p,ph-p)*dx))
if(MPI.rank(MPI.comm_world)==0):
    print("======= Final Results =======")
    print("log(h) = "+str(math.log(1.0/Nel)))
    print("log(H^1 velocity error) = "+str(math.log(err_u_H1)))
    # Edit the code to print h and H1 velocity error instead of computing ln(...):
    #print("h = "+str(1.0/Nel))
    #print("H^1 velocity error = "+str(err_u_H1))
    print("log(L^2 pressure error) = "+str(math.log(err_p_L2)))

#Output the required files to be read and processed.
output_file = open('copypasta-ldc.txt','w')
output_file.write('Nel = '+str(Nel))
output_file.write(', h = '+str(1.0/Nel))
output_file.write(', H^1 velocity error = '+str(err_u_H1))
output_file.write(', L^2 pressure error = '+str(err_p_L2))
output_file.close()

# Output ParaView files as a sanity check, if desired.
if(VIZ):
   uh,ph,pPrime = w.split()
   uh.rename("u","u")
   File("results/u.pvd") << uh
   ph.rename("p","p")
   File("results/p.pvd") << ph
