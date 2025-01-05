"""
********************************************************************************
2D Poisson equation: -div(grad(u)) = f
solve with finite difference method (FDM) and Jacobi iteration
********************************************************************************
"""

import numpy as np
import matplotlib.pyplot as plt

def Jacobi(u, f, g_d, g_n, dx, dy, tol=1e-8, maxit=int(1e6)):
    """
    Jacobi iteration

    args:
        u: 2D array, solution
        f: 2D array, source term
        g_d: constant, Dirichlet boundary condition
        g_n: constant, Neumann boundary condition
        dx: float, grid spacing in x
        dy: float, grid spacing in y

    returns:
        u: 2D array, solution
    """

    res = 9999.
    it = 0
    while res > tol:
        # copy
        u_old = u.copy()

        # update
        u[1:-1, 1:-1] \
            = (
                (u[1:-1, 2:] + u[1:-1, :-2]) * dy**2 \
                + (u[2:, 1:-1] + u[:-2, 1:-1]) * dx**2 \
                - f[1:-1, 1:-1] * dx**2 * dy**2
            ) / (2. * (dx**2 + dy**2))

        # boundary conditions
        u[0,     :] = u[1,     :] + g_n * dx   # bottom
        u[-1,    :] = g_d   # top
        u[1:-1, -1] = 0.   # right
        u[1:-1,  0] = 0.   # left

        # # boundary conditions
        # u[0,     :] = g_d   # bottom
        # u[-1,    :] = u[-2,    :] + g_n * dx   # top
        # u[1:-1, -1] = u[1:-1, -2] + g_n * dy   # right
        # u[1:-1,  0] = u[1:-1,  1] + g_n * dy   # left

        # residual
        res = np.sqrt(np.sum((u - u_old)**2) / np.sum(u_old**2))

        # iteration count
        it += 1
        if it % 200 == 0:
            print(f"it: {it:d}, res: {res:.6e}")
        if it > maxit:
            print(f"it: {it:d}, res: {res:.6e}")
            raise RuntimeError("Jacobi iteration not converged")

    return u



def main():
    # domain
    xmin, xmax = 0., 1.
    ymin, ymax = 0., 1.

    nx, ny = 201, 201

    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    # mesh
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    # initial guess
    u = np.zeros((nx, ny)) + 1e-3

    # source term
    k = 1.
    f = k * np.sin(2. * np.pi * (X + Y))

    # boundary conditions
    g_d = np.sin(1. * np.pi * (x))
    g_n = 0.

    # g_d = 0.
    # g_n = 0.

    # solve
    u = Jacobi(u, f, g_d, g_n, dx, dy)

    # plot
    cmap = plt.get_cmap("turbo")
    plt.figure(figsize=(5, 4))
    levels = np.linspace(np.min(u), np.max(u), 64)
    ticks = np.linspace(np.min(u), np.max(u), 5)
    plt.contourf(X, Y, u, cmap=cmap, levels=levels, extend="both")
    plt.colorbar(ticks=ticks, label="$u$")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Poisson equation, $-\\nabla^2 u = f$")
    plt.xticks(np.linspace(xmin, xmax, 5))
    plt.yticks(np.linspace(ymin, ymax, 5))
    plt.tight_layout()
    plt.savefig("./poisson_fdm.png")
    plt.close()

    # 3D plot
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, u, cmap=cmap)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$u$")
    plt.tight_layout()
    plt.savefig("./poisson_fdm_3d.png")
    plt.close()

    # save
    np.save("./poisson_fdm_x.npy", x)
    np.save("./poisson_fdm_y.npy", y)
    np.save("./poisson_fdm_X.npy", X)
    np.save("./poisson_fdm_Y.npy", Y)
    np.save("./poisson_fdm_u.npy", u)


def plot_setting():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["legend.framealpha"] = 1.
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["savefig.dpi"] = 300


if __name__ == "__main__":
    plot_setting()
    main()
