from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

# final time
t_final = 0.1

# method for computing MMS source ... FIXME
#   I recommend removing the "discrete" way of computing f_u, f_v, and instead
#   only use the symbolic route.  Note that the discrete way introduces another
#   discretization error.
discrete = False

# mesh refinement in space *and time*
mesh_sizes = [8, 16, 32, 64, 128]
h_values = [1.0 / size for size in mesh_sizes]
dt_values = [t_final / size for size in mesh_sizes]

errors_u = []
errors_v = []

for size, h, dt in zip(mesh_sizes, h_values, dt_values):
    print(f"solving with {size}x{size} mesh (h={h:.4f}) and dt={dt:.4f} ...")

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
    V_u = as_vector((1.0, 0.5)) # advective terms
    V_v = as_vector((-0.5, 1.0))

    x, y = SpatialCoordinate(mesh)
    t = Constant(0.0)

    u_exact = sin(pi*x)*sin(pi*y)*exp(-t)
    v_exact = sin(2*pi*x)*sin(2*pi*y)*exp(-2*t)

    if discrete:
        u_exact_n = sin(pi*x)*sin(pi*y)*exp(-(t - dt))
        v_exact_n = sin(2*pi*x)*sin(2*pi*y)*exp(-2*(t - dt))

        f_u = (u_exact - u_exact_n)/dt - D_u*div(grad(u_exact)) + dot(V_u, grad(u_exact)) - (u_exact*(1 - u_exact) - alpha*u_exact*v_exact)
        f_v = (v_exact - v_exact_n)/dt - D_v*div(grad(v_exact)) + dot(V_v, grad(v_exact)) - (beta*u_exact*v_exact - gamma*v_exact)
    else:
        t = variable(t)
        f_u = diff(u_exact, t) - D_u*div(grad(u_exact)) + dot(V_u, grad(u_exact)) - (u_exact*(1 - u_exact) - alpha*u_exact*v_exact)
        f_v = diff(v_exact, t) - D_v*div(grad(v_exact)) + dot(V_v, grad(v_exact)) - (beta*u_exact*v_exact - gamma*v_exact)
        t = Constant(t)

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
        t.assign(t_curr + dt)
        solve(F == 0,U,bcs=bcs,solver_parameters={
                "snes_type": "newtonls",
                "snes_converged_reason":None,
                "ksp_type": "gmres",
                "pc_type": "ilu",
            }, options_prefix=f"t={t_curr + dt:.4f}"
        )

        U_n.assign(U)
        t_curr += dt

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
