'''
********************************************************************************
load results.npz and visualize
********************************************************************************
'''

import numpy as np
import matplotlib.pyplot as plt

import references

def main():
    # load data
    data = np.load("results.npz")

    print(data.files)
    # print(data['x'])
    # print(data['y'])
    # print(data['X'])
    # print(data['Y'])
    # print(data['u'])
    # print(data['v'])
    # print(data['p'])
    # print(data['p_bar'])

    x = data['x']
    y = data['y']
    X = data['X']
    Y = data['Y']
    u = data['u']
    v = data['v']
    p = data['p']
    p_bar = data['p_bar']

    nx = x.shape[0]
    ny = y.shape[0]

    print(f"x.shape = {x.shape}")
    print(f"y.shape = {y.shape}")
    print(f"X.shape = {X.shape}")
    print(f"Y.shape = {Y.shape}")
    print(f"u.shape = {u.shape}")
    print(f"v.shape = {v.shape}")
    print(f"p.shape = {p.shape}")
    print(f"p_bar.shape = {p_bar.shape}")

    # velocity norm
    vel_norm = np.sqrt(u**2 + v**2)

    # interpolate p and p_bar to uv node
    p_uv = (p[1:, 1:] + p[:-1, 1:] + p[1:, :-1] + p[:-1, :-1]) / 4.
    p_bar_uv = (p_bar[1:, 1:] + p_bar[:-1, 1:] + p_bar[1:, :-1] + p_bar[:-1, :-1]) / 4.
    print(f"p_uv.shape = {p_uv.shape}")
    print(f"p_bar_uv.shape = {p_bar_uv.shape}")

    # plot
    plt.figure(figsize=(6, 5))
    levels = np.linspace(0., 1., 32)
    ticks = np.linspace(0., 1., 5)
    plt.contourf(X, Y, vel_norm, levels=levels, cmap='turbo', extend='both')
    plt.colorbar(ticks=ticks)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.tight_layout()
    plt.savefig("vel_norm.svg")
    plt.close()

    plt.figure(figsize=(6, 5))
    levels = np.linspace(-.1, .1, 32)
    ticks = np.linspace(-.1, .1, 5)
    plt.contourf(X[1:-1, 1:-1], Y[1:-1, 1:-1], p_uv, levels=levels, cmap='turbo', extend='both')
    plt.colorbar(ticks=ticks)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.tight_layout()
    plt.savefig("pressure.svg")
    plt.close()

    plt.figure(figsize=(6, 5))
    levels = np.linspace(-.1, .1, 32)
    ticks = np.linspace(-.1, .1, 5)
    plt.contourf(X[1:-1, 1:-1], Y[1:-1, 1:-1], p_bar_uv, levels=levels, cmap='turbo', extend='both')
    plt.colorbar(ticks=ticks)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.tight_layout()
    plt.savefig("pressure_bar.svg")
    plt.close()

    # comparison against reference solutions
    ref_Ghia = references.Ghia(Re=1000.)
    ref_Erturk = references.Erturk(Re=1000.)

    plt.figure(figsize=(5, 5))
    plt.scatter(ref_Ghia["u"], ref_Ghia["y"],     marker="+", label="Ghia et al. 1982")
    plt.scatter(ref_Erturk["u"], ref_Erturk["y"], marker="x", label="Erturk et al. 2005")
    plt.plot(u[1:-1, ny//2], y[1:-1], c='k', ls='--', label="FDM")
    plt.legend(loc='lower right')
    plt.xlabel(r"$u$")
    plt.ylabel(r"$y$")
    # plt.xticks(np.arange(-2., 2., .2))
    # plt.yticks(np.arange(-.2, 1.2, .2))
    plt.xlim([-.5, 1.1])
    plt.ylim([0., 1.])
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig("comparison_u.svg")
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(ref_Ghia["x"], ref_Ghia["v"],     marker='+', label="Ghia et al. 1982")
    plt.scatter(ref_Erturk["x"], ref_Erturk["v"], marker='x', label="Erturk et al. 2005")
    plt.plot(x[1:-1], v[nx//2, 1:-1], c='k', ls='--', label="FDM")
    plt.legend(loc='lower left')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$v$")
    plt.xticks(np.arange(-.2, 1.2, .2))
    plt.yticks(np.arange(-2., 2., .2))
    plt.xlim([0., 1.])
    plt.ylim([-.6, .4])
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig("comparison_v.svg")
    plt.close()



def plot_setting():
    plt.style.use("default")
    # plt.style.use("seaborn-deep")
    plt.style.use("seaborn-poster")   # paper / notebook / talk / poster
    # plt.rcParams["font.size"] = 12
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["figure.autolayout"] = True
    # plt.rcParams["axes.grid"] = True
    # plt.rcParams["grid.alpha"] = .3
    plt.rcParams["legend.framealpha"] = 1.
    plt.rcParams["legend.facecolor"] = "w"
    plt.rcParams["savefig.dpi"] = 300


if __name__ == "__main__":
    plot_setting()
    main()
