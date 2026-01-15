'''
this is Heat Equation demo from
firedrakeproject.org/Irksome/demos/demo_heat.py.html
Equation:
u_t - Delta_u = f
u = 0 on boundary
'''

from firedrake import *
from irksome import GaussLegendre, Dt, MeshConstant, TimeStepper

butcher_tableau = GaussLegendre(1)   # creates Butcher tableau for lowest-order Runge-Kutta
ns = butcher_tableau.numstages

# Define Mesh
N = 100
x0 = 0.0
x1 = 10.0
y0 = 0.0
y1 = 10.0

mesh = RectangleMesh(N,N,x1,y1)
V = FunctionSpace(mesh,'CG',1)

# time-step variables
MC = MeshConstant(mesh)
dt = MC.Constant(10.0/N)  # time-step
t = MC.Constant(0.0)  # current time value

# define right-side by method of manufactured solutions
x,y = SpatialCoordinate(mesh)
S = Constant(2.0)
C = Constant(1000.0)
B = (x-Constant(x0))*(x-Constant(x1))*(y-Constant(y0))*(y-Constant(y1))/C
R = (x*x + y*y)**0.5
uexact = B * atan(t)*(pi/2.0 - atan(S*(R-T)))
rhs = Dt(uexact) - div(grad(uexact))

# define initial conditions
u = Function(V)
u.interpolate(uexact)

# define the semi-discrete variational problem
v = TestFunction(V)
F = inner(Dt(u),v)*dx + inner(grad(u),grad(v))*dx - inner(rhs,v)*dx
bc = DirichletBC(V,0,"on_boundary")

luparams = {"mat_type":"aij",
            "ksp_type":"preonly",
            "pc_type":"lu"}
# TimeStepper transforms the semidiscrete F into a fully discrete form and sets up variational problem
stepper = TimeStepper(F,butcher_tableau,t,dt,u,bcs=bc,solver_parameters=luparams)
# uses TimeStepper's 'advance' method, which solves the variational problem
while (float(t)<1.0):
    if (float(t)+float(dt)>1.0):
        dt.assign(1.0-float(t))
    stepper.advance()
    print(float(t))
    t.assign(float(t)+float(dt))

# print relative L^2 error
print()
print(norm(u-uexact)/norm(uexact))