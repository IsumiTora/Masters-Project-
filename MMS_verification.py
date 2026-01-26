from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

mesh_sizes = [8, 16, 32, 64]
h_values = [1.0 / size for size in mesh_sizes]
errors_u = []
errors_v = []

# timestep values
t_final = 0.1

for size in mesh_sizes:

    mesh = UnitSquareMesh(size,size)
    V = FunctionSpace(mesh,"CG",1)
    W = V*V

    U = Function(W)
    U_n = Function(W)

    (u,v) = split(U)
    (u_n,v_n) = split(U_n)
    (psi,phi) = TestFunctions(W)

    D_u = Constant(1.0) # diffusivity
    D_v = Constant(1.0) 
    alpha = Constant(1.0)
    beta = Constant(1.0)
    gamma = Constant(1.0)
    dt = Constant(0.01)
    V_u = as_vector((1.0, 0.5)) # advective terms
    V_v = as_vector((-0.5, 1.0))

    x, y = SpatialCoordinate(mesh)
    t = Constant(0.0)

    u_exact = sin(pi*x)*sin(pi*y)*exp(-t)
    v_exact = sin(2*pi*x)*sin(2*pi*y)*exp(-2*t)

    u_exact_n = sin(pi*x)*sin(pi*y)*exp(-(t - dt))
    v_exact_n = sin(2*pi*x)*sin(2*pi*y)*exp(-2*(t - dt))

    f_u = ((u_exact - u_exact_n)/dt - D_u*div(grad(u_exact)) + dot(V_u, grad(u_exact)) - (u_exact*(1 - u_exact) - alpha*u_exact*v_exact))
    f_v = ((v_exact - v_exact_n)/dt - D_v*div(grad(v_exact)) + dot(V_v, grad(v_exact)) - (beta*u_exact*v_exact - gamma*v_exact))

    # weak forms
    F_u = ((u - u_n)/dt * psi * dx + D_u*dot(grad(u), grad(psi)) * dx + dot(V_u, grad(u)) * psi * dx - (u*(1 - u) - alpha*u*v) * psi * dx - f_u * psi * dx)
    F_v = ((v - v_n)/dt * phi * dx + D_v*dot(grad(v), grad(phi)) * dx + dot(V_v, grad(v)) * phi * dx - (beta*u*v - gamma*v) * phi * dx - f_v * phi * dx)
    F = F_u + F_v

    bcs = [DirichletBC(W.sub(0), 0.0, "on_boundary"),DirichletBC(W.sub(1), 0.0, "on_boundary")]

    # initial conditions
    U_n.sub(0).interpolate(u_exact)
    U_n.sub(1).interpolate(v_exact)
    U.assign(U_n)

    t_curr = 0.0
    while t_curr < t_final - 1e-12:
        t.assign(t_curr + dt_value)
        solve(F == 0,U,bcs=bcs,solver_parameters={
                "snes_type": "newtonls",
                "ksp_type": "gmres",
                "pc_type": "ilu",
            }
        )

        U_n.assign(U)
        t_curr += dt_value

    u_h, v_h = U.subfunctions
    u_e = Function(V).interpolate(u_exact)
    v_e = Function(V).interpolate(v_exact)

    error_u = errornorm(u_e, u_h, norm_type="L2")
    error_v = errornorm(v_e, v_h, norm_type="L2")

    errors_u.append(error_u)
    errors_v.append(error_v)

plt.figure()
plt.loglog(h_values, errors_u, "o-", label="u error")
plt.loglog(h_values, errors_v, "s-", label="v error")
plt.xlabel("h")
plt.ylabel("Error")
plt.legend()
plt.grid(True, which="both")
plt.show()
