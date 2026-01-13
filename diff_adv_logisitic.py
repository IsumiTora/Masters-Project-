from firedrake import *
import numpy as np

size = 20
mesh = UnitSquareMesh(size,size)
V = FunctionSpace(mesh,"CG",1)    # galerkin or other choice?
W = V*V

U = Function(W)
U_n = Function(W)

(u,v) = split(U)
(u_n,v_n) = split(U_n)
(psi,phi) = TestFunctions(W)

D_u = 1      # diffusion constant for prey
D_v = 1      # diffusion constant for predator
alpha = 1    # predation interaction coeff of prey
beta = 1     # predation interaction coeff for pred
gamma = 1    # pred death coeff
dt = 0.1     # timestep length

x,y = SpatialCoordinate(mesh)
V_u = as_vector((1.0,0.0))   # simple motion, possibility to add nonlinearity here
V_v = as_vector((0.5,0.0))  

F_u = ((u-u_n)/dt * psi *dx) + D_u*dot(grad(u),grad(psi)) * dx + dot(V_u,grad(u)) * psi * dx - (u(1-u)-alpha*u*v) * psi * dx
F_v = ((v-v_n)/dt * phi *dx) + D_v*dot(grad(v),grad(phi)) * dx + dot(V_v,grad(v)) * phi * dx - (beta*u*v-gamma*v) * phi * dx
F = F_u + F_v
