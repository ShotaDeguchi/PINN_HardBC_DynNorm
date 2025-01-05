"""
********************************************************************************
plotters, utils for visualization
********************************************************************************
"""

import os
import pathlib

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


def plot_setting():
    # plt.style.use("seaborn-talk")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["legend.framealpha"] = 1.
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["savefig.dpi"] = 300


def plot_comparison(
    x, y, u_ref, u_inf, u_err,
    umin, umax, vmin, vmax,
    xmin, xmax, xlabel,
    ymin, ymax, ylabel,
    epoch,
    save_path,
    file_name,
    file_types=[".png", ".svg", ".pdf"]
):
    xticks = (xmax - xmin) / 4.
    yticks = (ymax - ymin) / 4.

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    levels = np.linspace(umin, umax+1e-6, 64)
    ticks = np.linspace(umin, umax+1e-6, 5)
    plt.contourf(
        tf.reshape(x, shape=(201, 201)),
        tf.reshape(y, shape=(201, 201)),
        tf.reshape(u_ref, shape=(201, 201)),
        levels=levels, extend="both", cmap="turbo"
    )
    plt.colorbar(ticks=ticks, label=r"$u$")
    plt.xticks(np.arange(xmin, xmax+1e-6, xticks))
    plt.yticks(np.arange(ymin, ymax+1e-6, yticks))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Reference")

    plt.subplot(1, 3, 2)
    plt.contourf(
        tf.reshape(x, shape=(201, 201)),
        tf.reshape(y, shape=(201, 201)),
        tf.reshape(u_inf, shape=(201, 201)),
        levels=levels, extend="both", cmap="turbo"
    )
    plt.colorbar(ticks=ticks, label=r"$\hat{u}$")
    plt.xticks(np.arange(xmin, xmax+1e-6, xticks))
    plt.yticks(np.arange(ymin, ymax+1e-6, yticks))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Inference")

    plt.subplot(1, 3, 3)
    levels = np.linspace(vmin, vmax+1e-6, 64)
    ticks = np.linspace(vmin, vmax+1e-6, 5)
    plt.contourf(
        tf.reshape(x, shape=(201, 201)),
        tf.reshape(y, shape=(201, 201)),
        tf.reshape(np.abs(u_err), shape=(201, 201)),
        levels=levels, extend="both", cmap="turbo"
    )
    plt.colorbar(ticks=ticks, label=r"$| \hat{u} - u |$")
    plt.xticks(np.arange(xmin, xmax+1e-6, xticks))
    plt.yticks(np.arange(ymin, ymax+1e-6, yticks))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    u_err = tf.norm(u_err, ord=2) / tf.norm(u_ref, ord=2)
    plt.title(f"Abs. err. (rel. L2 err.: {u_err:.3e})")

    plt.tight_layout()
    file_name = "comparison_" + file_name + "_" + str(epoch)
    for file_type in file_types:
        plt.savefig(os.path.join(save_path, file_name + file_type), dpi=300)
    plt.close()


def plot_3images(
    x, xmin, xmax, xlabel,
    y, ymin, ymax, ylabel,
    u1, u1min, u1max, u1label, u1cmap,
    u2, u2min, u2max, u2label, u2cmap,
    u3, u3min, u3max, u3label, u3cmap,
    epoch,
    save_path,
    file_name,
    file_types=[".png", ".svg", ".pdf"]
):
    xticks = (xmax - xmin) / 4.
    yticks = (ymax - ymin) / 4.

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    levels = np.linspace(u1min, u1max+1e-6, 64)
    ticks = np.linspace(u1min, u1max+1e-6, 5)
    plt.contourf(
        tf.reshape(x, shape=(201, 201)),
        tf.reshape(y, shape=(201, 201)),
        tf.reshape(u1, shape=(201, 201)),
        levels=levels, extend="both", cmap=u1cmap
    )
    plt.colorbar(ticks=ticks, label=u1label)
    plt.xticks(np.arange(xmin, xmax+1e-6, xticks))
    plt.yticks(np.arange(ymin, ymax+1e-6, yticks))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.subplot(1, 3, 2)
    levels = np.linspace(u2min, u2max+1e-6, 64)
    ticks = np.linspace(u2min, u2max+1e-6, 5)
    plt.contourf(
        tf.reshape(x, shape=(201, 201)),
        tf.reshape(y, shape=(201, 201)),
        tf.reshape(u2, shape=(201, 201)),
        levels=levels, extend="both", cmap=u2cmap
    )
    plt.colorbar(ticks=ticks, label=u2label)
    plt.xticks(np.arange(xmin, xmax+1e-6, xticks))
    plt.yticks(np.arange(ymin, ymax+1e-6, yticks))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.subplot(1, 3, 3)
    levels = np.linspace(u3min, u3max+1e-6, 64)
    ticks = np.linspace(u3min, u3max+1e-6, 5)
    plt.contourf(
        tf.reshape(x, shape=(201, 201)),
        tf.reshape(y, shape=(201, 201)),
        tf.reshape(u3, shape=(201, 201)),
        levels=levels, extend="both", cmap=u3cmap
    )
    plt.colorbar(ticks=ticks, label=u3label)
    plt.xticks(np.arange(xmin, xmax+1e-6, xticks))
    plt.yticks(np.arange(ymin, ymax+1e-6, yticks))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()
    file_name = "3images_" + file_name + "_" + str(epoch)
    for file_type in file_types:
        plt.savefig(os.path.join(save_path, file_name + file_type), dpi=300)
    plt.close()


def plot_4images(
    x, xmin, xmax, xlabel,
    y, ymin, ymax, ylabel,
    u1, u1min, u1max, u1label, u1cmap,
    u2, u2min, u2max, u2label, u2cmap,
    u3, u3min, u3max, u3label, u3cmap,
    u4, u4min, u4max, u4label, u4cmap,
    epoch,
    save_path,
    file_name,
    file_types=[".png", ".svg", ".pdf"]
):
    xticks = (xmax - xmin) / 4.
    yticks = (ymax - ymin) / 4.

    plt.figure(figsize=(20, 4))

    plt.subplot(1, 4, 1)
    levels = np.linspace(u1min, u1max+1e-6, 64)
    ticks = np.linspace(u1min, u1max+1e-6, 5)
    plt.contourf(
        tf.reshape(x, shape=(201, 201)),
        tf.reshape(y, shape=(201, 201)),
        tf.reshape(u1, shape=(201, 201)),
        levels=levels, extend="both", cmap=u1cmap
    )
    plt.colorbar(ticks=ticks, label=u1label)
    plt.xticks(np.arange(xmin, xmax+1e-6, xticks))
    plt.yticks(np.arange(ymin, ymax+1e-6, yticks))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.subplot(1, 4, 2)
    levels = np.linspace(u2min, u2max+1e-6, 64)
    ticks = np.linspace(u2min, u2max+1e-6, 5)
    plt.contourf(
        tf.reshape(x, shape=(201, 201)),
        tf.reshape(y, shape=(201, 201)),
        tf.reshape(u2, shape=(201, 201)),
        levels=levels, extend="both", cmap=u2cmap
    )
    plt.colorbar(ticks=ticks, label=u2label)
    plt.xticks(np.arange(xmin, xmax+1e-6, xticks))
    plt.yticks(np.arange(ymin, ymax+1e-6, yticks))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.subplot(1, 4, 3)
    levels = np.linspace(u3min, u3max+1e-6, 64)
    ticks = np.linspace(u3min, u3max+1e-6, 5)
    plt.contourf(
        tf.reshape(x, shape=(201, 201)),
        tf.reshape(y, shape=(201, 201)),
        tf.reshape(u3, shape=(201, 201)),
        levels=levels, extend="both", cmap=u3cmap
    )
    plt.colorbar(ticks=ticks, label=u3label)
    plt.xticks(np.arange(xmin, xmax+1e-6, xticks))
    plt.yticks(np.arange(ymin, ymax+1e-6, yticks))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.subplot(1, 4, 4)
    levels = np.linspace(u4min, u4max+1e-6, 64)
    ticks = np.linspace(u4min, u4max+1e-6, 5)
    plt.contourf(
        tf.reshape(x, shape=(201, 201)),
        tf.reshape(y, shape=(201, 201)),
        tf.reshape(u4, shape=(201, 201)),
        levels=levels, extend="both", cmap=u4cmap
    )
    plt.colorbar(ticks=ticks, label=u4label)
    plt.xticks(np.arange(xmin, xmax+1e-6, xticks))
    plt.yticks(np.arange(ymin, ymax+1e-6, yticks))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()
    file_name = "4images_" + file_name + "_" + str(epoch)
    for file_type in file_types:
        plt.savefig(os.path.join(save_path, file_name + file_type), dpi=300)
    plt.close()


def plot_comparison_3D(
    x, y, u_ref, u_inf, u_err,
    umin, umax,
    vmin, vmax,
    xmin, xmax, xlabel,
    ymin, ymax, ylabel,
    epoch,
    save_path,
    file_types=[".png", ".svg", ".pdf"]
):
    fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.plot_surface(
        tf.reshape(x, shape=(201, 201))[1:-1, 1:-1],
        tf.reshape(y, shape=(201, 201))[1:-1, 1:-1],
        tf.reshape(u_ref, shape=(201, 201))[1:-1, 1:-1],
        cmap="turbo", vmin=umin, vmax=umax, antialiased=False
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(r"$u$")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(umin, umax)
    ax.set_title("Reference")

    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.plot_surface(
        tf.reshape(x, shape=(201, 201))[1:-1, 1:-1],
        tf.reshape(y, shape=(201, 201))[1:-1, 1:-1],
        tf.reshape(u_inf, shape=(201, 201))[1:-1, 1:-1],
        cmap="turbo", vmin=umin, vmax=umax, antialiased=False
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(r"$\hat{u}$")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(umin, umax)
    ax.set_title("Inference")

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    ax.plot_surface(
        tf.reshape(x, shape=(201, 201))[1:-1, 1:-1],
        tf.reshape(y, shape=(201, 201))[1:-1, 1:-1],
        np.abs(u_err),
        # tf.reshape(np.abs(u_err), shape=(201, 201)),
        cmap="turbo", vmin=vmin, vmax=vmax, antialiased=False
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(r"$| \hat{u} - u| $")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(vmin, vmax)
    u_err = tf.norm(u_err, ord=2) / tf.norm(u_ref, ord=2)
    ax.set_title(f"Abs. err. (rel. L2 err.: {u_err:.3e})")

    plt.tight_layout()
    file_name = "3D_comparison_" + str(epoch)
    for file_type in file_types:
        plt.savefig(os.path.join(save_path, file_name + file_type), dpi=300)
    plt.close()



def plot_loss_curve(
    epoch_log,
    loss_glb_log,
    loss_pde_log,
    loss_bc_log,
    epoch,
    save_path,
    file_types=[".png", ".svg", ".pdf"]
):
    plt.plot(epoch_log, loss_glb_log, ls="-",  alpha=1., c="tab:gray",  label=r"$\mathcal{L}$")
    plt.plot(epoch_log, loss_pde_log, ls="--", alpha=.3, c="tab:blue",  label=r"$\mathcal{L}_{\mathrm{PDE}}$")
    plt.plot(epoch_log, loss_bc_log,  ls="--", alpha=.3, c="tab:green", label=r"$\mathcal{L}_{\mathrm{BC}}$")
    plt.legend(loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xscale("linear")
    plt.yscale("log")
    plt.grid(alpha=.5)
    plt.tight_layout()
    file_name = "loss_curve_" + str(epoch)
    for file_type in file_types:
        plt.savefig(os.path.join(save_path, file_name + file_type), dpi=300)
    plt.close()



def plot_grad_curve(
    epoch_log,
    grad_glb_log,
    grad_pde_log,
    grad_bc_log,
    epoch,
    save_path,
    file_types=[".png", ".svg", ".pdf"]
):
    plt.plot(epoch_log, grad_glb_log, ls="-",  alpha=1., c="tab:gray",  label=r"$\| \nabla_{\theta} \mathcal{L} \|_{2}$")
    plt.plot(epoch_log, grad_pde_log, ls="--", alpha=.3, c="tab:blue",  label=r"$\| \nabla_{\theta} \mathcal{L}_{\mathrm{PDE}} \|_{2}$")
    plt.plot(epoch_log, grad_bc_log,  ls="--", alpha=.3, c="tab:green", label=r"$\| \nabla_{\theta} \mathcal{L}_{\mathrm{BC}} \|_{2}$")
    plt.legend(loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient norm")
    plt.xscale("linear")
    plt.yscale("log")
    plt.grid(alpha=.5)
    plt.tight_layout()
    file_name = "grad_curve_" + str(epoch)
    for file_type in file_types:
        plt.savefig(os.path.join(save_path, file_name + file_type), dpi=300)
    plt.close()



def plot_error_curve(
    epoch_log,
    u_l2_log, u_mse_log, u_sem_log,
    g_l2_log, g_mse_log, g_sem_log,
    epoch,
    save_path,
    file_types=[".png", ".svg", ".pdf"]
):
    plt.plot(epoch_log, u_l2_log,  ls="-",  lw=1., alpha=.7, c="tab:blue", label=r"$\| \hat{u} - u \|_{2} / \| u \|_{2}$")
    plt.plot(epoch_log, u_mse_log, ls="--", lw=1., alpha=.7, c="tab:cyan", label=r"$\mathrm{MSE}(\hat{u}, u)$")
    plt.fill_between(
        epoch_log, 
        np.array(u_mse_log) + np.array(u_sem_log), 
        np.array(u_mse_log) - np.array(u_sem_log), 
        alpha=.3, color="tab:cyan", label=r"$\mathrm{SE}(\hat{u}, u)$"
    )
    plt.plot(epoch_log, g_l2_log,  ls="-",  lw=1., alpha=.7, c="tab:red", label=r"$\| \hat{g} \|_{2}$")
    plt.plot(epoch_log, g_mse_log, ls="--", lw=1., alpha=.7, c="tab:pink", label=r"$\mathrm{MSE}(\hat{g}, 0)$")
    plt.fill_between(
        epoch_log, 
        np.array(g_mse_log) + np.array(g_sem_log), 
        np.array(g_mse_log) - np.array(g_sem_log), 
        alpha=.3, color="tab:pink", label=r"$\mathrm{SE}(\hat{g}, 0)$"
    )
    plt.legend(loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.yscale("log")
    plt.grid(alpha=.5)
    plt.tight_layout()
    file_name = "error_curve_" + str(epoch)
    for file_type in file_types:
        plt.savefig(os.path.join(save_path, file_name + file_type), dpi=300)
    plt.close()



def plot_gamma_curve(
    epoch_log,
    gamma_bc_log,
    gamma_bc_hat_log,
    epoch,
    save_path,
    file_types=[".png", ".svg", ".pdf"]
):
    plt.plot(epoch_log, gamma_bc_log,     ls="--", lw=1., alpha=.7, c="tab:green", label=r"$\gamma_{\mathrm{BC}}$")
    plt.plot(epoch_log, gamma_bc_hat_log, ls="-",  lw=1., alpha=.7, c="limegreen", label=r"$\hat{\gamma}_{\mathrm{BC}}$")
    plt.legend(loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel(r"$\gamma$")
    plt.grid(alpha=.5)
    plt.tight_layout()
    file_name = "gamma_curve_" + str(epoch)
    for file_type in file_types:
        plt.savefig(os.path.join(save_path, file_name + file_type), dpi=300)
    plt.close()



def plot_grad_dist(
    model: tf.keras.Model,
    depth: int,
    x_pde, y_pde,
    x_nth, y_nth, u_nth,
    x_sth, y_sth, u_sth,
    x_est, y_est, u_est,
    x_wst, y_wst, u_wst,
    epoch,
    save_path,
    file_types=[".png", ".svg", ".pdf"]
):
    gamma_bc = model.gamma_bc.numpy()
    gamma_bc_hat = model.gamma_bc_hat.numpy()

    _grad_glb = tf.zeros(shape=(0))
    _grad_pde = tf.zeros(shape=(0))
    _grad_bc  = tf.zeros(shape=(0))

    for l in range(depth):
        with tf.GradientTape(persistent=True) as tp:
            loss_glb, loss_pde, loss_bc = model.loss_glb(
                x_pde, y_pde,
                x_nth, y_nth, u_nth,
                x_sth, y_sth, u_sth,
                x_est, y_est, u_est,
                x_wst, y_wst, u_wst
            )

        # grad to weights
        grad_glb = tp.gradient(loss_glb, model._weights[l])
        grad_pde = tp.gradient(loss_pde, model._weights[l])
        grad_bc  = tp.gradient(loss_bc,  model._weights[l])
        grad_glb = tf.cast(grad_glb, dtype=tf.float32)
        grad_pde = tf.cast(grad_pde, dtype=tf.float32)
        grad_bc  = tf.cast(grad_bc,  dtype=tf.float32)
        _grad_glb = tf.concat([_grad_glb, tf.reshape(grad_glb, [-1])], axis=0)
        _grad_pde = tf.concat([_grad_pde, tf.reshape(grad_pde, [-1])], axis=0)
        _grad_bc  = tf.concat([_grad_bc,  tf.reshape(grad_bc,  [-1])], axis=0)

        # grad to biases (last bias may not be tracked due to the problem setup)
        try:
            grad_glb = tp.gradient(loss_glb, model._biases[l])
            grad_pde = tp.gradient(loss_pde, model._biases[l])
            grad_bc  = tp.gradient(loss_bc,  model._biases[l])
            grad_glb = tf.cast(grad_glb, dtype=tf.float32)
            grad_pde = tf.cast(grad_pde, dtype=tf.float32)
            grad_bc  = tf.cast(grad_bc,  dtype=tf.float32)
            _grad_glb = tf.concat([_grad_glb, tf.reshape(grad_glb, [-1])], axis=0)
            _grad_pde = tf.concat([_grad_pde, tf.reshape(grad_pde, [-1])], axis=0)
            _grad_bc  = tf.concat([_grad_bc,  tf.reshape(grad_bc,  [-1])], axis=0)
        except:
            pass
        del tp

    # cummulated histogram
    sns.histplot(_grad_glb,               stat="density", element="step", fill=False, ls="-",  alpha=.7, color="tab:gray",  label=r"$\nabla_{\theta} \mathcal{L}$")
    sns.histplot(_grad_pde,               stat="density", element="step", fill=False, ls="-",  alpha=.7, color="tab:blue",  label=r"$\nabla_{\theta} \mathcal{L}_{\mathrm{PDE}}$")
    sns.histplot(gamma_bc * _grad_bc,     stat="density", element="step", fill=False, ls="--", alpha=.7, color="tab:green", label=r"$\gamma_{\mathrm{BC}} \nabla_{\theta} \mathcal{L}_{\mathrm{BC}}$")
    sns.histplot(gamma_bc_hat * _grad_bc, stat="density", element="step", fill=False, ls="-",  alpha=.7, color="limegreen", label=r"$\hat{\gamma}_{\mathrm{BC}} \nabla_{\theta} \mathcal{L}_{\mathrm{BC}}$")
    plt.xlabel("Gradient")
    plt.ylabel("Density")
    plt.xscale("linear")
    plt.yscale("symlog")
    # plt.xlim(-5., 5.)
    # plt.ylim(0., 10 ** 1)
    plt.legend(loc="upper right")
    grad_norm_pde = tf.norm(_grad_pde, ord=2)
    grad_norm_bc  = tf.norm(_grad_bc,  ord=2)
    plt.title(
        r"$\| \nabla_{\theta} \mathcal{L}_{\mathrm{PDE}} \|_2$: " + f"{grad_norm_pde:.3e}, " \
        + r"$\| \nabla_{\theta} \mathcal{L}_{\mathrm{BC}} \|_2$: " + f"{grad_norm_bc:.3e}, " \
        + "\n" + r" $\gamma_{\mathrm{BC}}$: " + f"{gamma_bc:.3e}, " \
        + r"$\hat{\gamma}_{\mathrm{BC}}$: " + f"{gamma_bc_hat:.3e}"
    )
    plt.grid(alpha=.5)
    plt.tight_layout()
    file_name = "grad_dist_" + str(epoch)
    for file_type in file_types:
        plt.savefig(os.path.join(save_path, file_name + file_type), dpi=300)
    plt.close()



