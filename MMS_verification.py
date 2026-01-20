from firedrake import *
import numpy as np

size = 16
mesh = UnitSquareMesh(size,size)
V = FunctionSpace(mesh,"CG",1)
W = V*V
U = Function(W)
U_n = Function(W)

# Define trial and test functions
(u,v) = split(U)
(u_n,v_n) = split(U_n)
(psi,phi) = TestFunctions(W)

# Define Constants
D_u = Constant(1.0) # diffusivity of u
D_v = Constant(1.0) # diffusivity of v
alpha = Constant(1.0) # predation interaction coeff of prey
beta = Constant(1.0) # predation interaction coeff of pred
gamma = Constant(1.0) # death rate of pred
dt = Constant(0.1) # timestep 

# Define Advection Fields
V_u = as_vector((1.0,0.5))
V_v = as_vector((-0.5,1.0))

# Manufactured Solutions
x, y = SpatialCoordinate(mesh)
t = Constant(0.0)
u_exact = sin(pi*x)*sin(pi*y)*exp(-t)
v_exact = cos(pi*x)*cos(pi*y)*exp(-2*t)

f_u = (diff(u_exact,t) - D_u*div(grad(u_exact)) + dot(V_u,grad(u_exact)) - (u_exact*(1-u_exact)-alpha*u_exact*v_exact))
f_v = (diff(v_exact,t) - D_u*div(grad(v_exact)) + dot(V_v,grad(v_exact)) - (beta*u_exact*v_exact-gamma*v_exact))

# weak form
F_u = ((u-u_n)/dt * psi *dx) + D_u*dot(grad(u),grad(psi)) * dx + dot(V_u,grad(u)) * psi * dx - (u*(1-u)-alpha*u*v) * psi * dx - f_u*psi*dx
F_v = ((v-v_n)/dt * phi *dx) + D_v*dot(grad(v),grad(phi)) * dx + dot(V_v,grad(v)) * phi * dx - (beta*u*v-gamma*v) * phi * dx - f_v*phi*dy
F = F_u + F_v

# boundary conditions
bc_u = DirichletBC(W.sub(0),Constant(0.0),"on_boundary")
bc_v = DirichletBC(W.sub(1),COnstant(0.0),"on_boundary")
U.assign(U_n)

# Time loop
t_final = 1.0
t_init = 0.0
while t_init < t_final:
    t.assign(t_init+ float(dt))
    solve(F==0, U, bcs=bcs,solver_parameters = {
        "snes_type":"newtonls",
        "ksp_type":"gmres",
        "pc_type":"ilu"
    })
    U_n.assign(U)
    t_init += float(dt)

u_h,v_h = U.split()
u_e = Function(V).interpolate(u_exact)
v_e = Function(V).interpolate(v_exact)
error_u = errornorm(u_e,u_h,norm_type="L2")
error_v = errornorm(v_e,v_h,norm_type="L2")

print("Error in u: "+error_u)
print("Error in v: "+error_v)

