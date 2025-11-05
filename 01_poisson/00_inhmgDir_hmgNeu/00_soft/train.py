"""
********************************************************************************
training
********************************************************************************
"""

import os
import time
import yaml
import pathlib
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from config_device import *
from pinn import *
from loggers import *
from plotters import *


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", type=int, default=0, help="seed")
parser.add_argument("-e", "--epochs", type=int, default=1000, help="number of epochs")
parser.add_argument("-b", "--batch_size", type=int, default=-1, help="batch size (-1 for full-batch)")
parser.add_argument("-p", "--patience", type=int, default=100, help="early stopping patience")
parser.add_argument("-d", "--device", type=int, default=0, help="device id")
args = parser.parse_args()


def main(args):
    # seed
    seed = args.seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # read 
    with open("./settings.yaml", mode="r") as f:
        settings = yaml.safe_load(f)

    # path
    path_seed = pathlib.Path(f"./seed{seed}")
    if not path_seed.exists():
        path_seed.mkdir(exist_ok=True)

    path_fig = path_seed / "fig"
    path_weights = path_seed / "weights"
    path_best_weights = path_seed / "best_weights"
    if not path_fig.exists():
        path_fig.mkdir(exist_ok=True)
    if not path_weights.exists():
        path_weights.mkdir(exist_ok=True)
    if not path_best_weights.exists():
        path_best_weights.mkdir(exist_ok=True)

    # logger
    path_train_log = path_seed / "train_log.txt"
    path_train_log = make_logger(path_train_log)

    # reference
    # x_ref = np.load("../99_reference/poisson_fdm_x.npy")
    # y_ref = np.load("../99_reference/poisson_fdm_y.npy")
    X_ref = np.load("../99_reference/poisson_fdm_X.npy")
    Y_ref = np.load("../99_reference/poisson_fdm_Y.npy")
    u_ref = np.load("../99_reference/poisson_fdm_u.npy")
    # print(x_ref.shape, y_ref.shape, u_ref.shape)
    print(X_ref.shape, Y_ref.shape, u_ref.shape)

    x_reshape = X_ref.reshape(-1, 1)
    y_reshape = Y_ref.reshape(-1, 1)
    u_reshape = u_ref.reshape(-1, 1)
    print(x_reshape.shape, y_reshape.shape, u_reshape.shape)

    x_reshape2 = X_ref[1:-1, 1:-1].reshape(-1, 1)
    y_reshape2 = Y_ref[1:-1, 1:-1].reshape(-1, 1)
    u_reshape2 = u_ref[1:-1, 1:-1].reshape(-1, 1)
    print(x_reshape2.shape, y_reshape2.shape, u_reshape2.shape)

    plt.figure(figsize=(5, 4))
    vmin, vmax = 0., 1.
    levels = np.linspace(vmin, vmax, 32)
    ticks = np.linspace(vmin, vmax, 5)
    plt.contourf(X_ref, Y_ref, u_ref, cmap="turbo", levels=levels, extend="both")
    plt.colorbar(ticks=ticks, label="$u$")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.xticks(np.linspace(0., 1., 5))
    plt.yticks(np.linspace(0., 1., 5))
    plt.tight_layout()
    plt.savefig(path_fig / "poisson_reference.png")
    plt.close()


    # collocation points
    N_pde = int(2**12)
    N_bc  = int(2**8)

    eps = 1e-2
    xmin, xmax = 0. + eps, 1. - eps
    ymin, ymax = 0. + eps, 1. - eps

    # pde residual
    X_pde = tf.random.uniform((N_pde, 2), xmin, xmax, dtype=tf.float64, seed=42)
    x_pde = X_pde[:, 0:1]
    y_pde = X_pde[:, 1:2]
    # north
    x_nth = tf.random.uniform((N_bc, 1), xmin, xmax, dtype=tf.float64, seed=42)
    y_nth = tf.ones_like(x_nth) * ymax
    u_nth = tf.sin(np.pi * x_nth)
    # south
    x_sth = tf.random.uniform((N_bc, 1), xmin, xmax, dtype=tf.float64, seed=42)
    y_sth = tf.ones_like(x_sth) * ymin
    u_sth = tf.zeros_like(x_sth)
    # east
    y_est = tf.random.uniform((N_bc, 1), ymin, ymax, dtype=tf.float64, seed=42)
    x_est = tf.ones_like(y_est) * xmax
    u_est = tf.zeros_like(y_est)
    # west
    y_wst = tf.random.uniform((N_bc, 1), ymin, ymax, dtype=tf.float64, seed=42)
    x_wst = tf.ones_like(y_wst) * xmin
    u_wst = tf.zeros_like(y_wst)

    plt.figure(figsize=(4, 4))
    plt.scatter(x_pde, y_pde, alpha=.7, label="pde")
    plt.scatter(x_nth, y_nth, alpha=.7, label="north")
    plt.scatter(x_sth, y_sth, alpha=.7, label="south")
    plt.scatter(x_est, y_est, alpha=.7, label="east")
    plt.scatter(x_wst, y_wst, alpha=.7, label="west")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_fig / "collocation_points.png")
    plt.close()

    # bounds (for feature scaling)
    f_in_lb = tf.constant([xmin, ymin], dtype=tf.float64)
    f_in_ub = tf.constant([xmax, ymax], dtype=tf.float64)
    f_in_mean = tf.constant([np.mean(x_pde), np.mean(y_pde)], dtype=tf.float64)
    f_in_std = tf.constant([np.std(x_pde, ddof=1), np.std(y_pde, ddof=1)], dtype=tf.float64)

    # test the feature scaling
    x_random = tf.random.uniform((256, 1), xmin, xmax, dtype=tf.float64, seed=42)
    y_random = tf.random.uniform((256, 1), ymin, ymax, dtype=tf.float64, seed=42)

    x_scaled1 = (x_random - f_in_lb[0]) / (f_in_ub[0] - f_in_lb[0]) * 2. - 1.
    y_scaled1 = (y_random - f_in_lb[1]) / (f_in_ub[1] - f_in_lb[1]) * 2. - 1.

    x_scaled2 = (x_random - f_in_mean[0]) / f_in_std[0]
    y_scaled2 = (y_random - f_in_mean[1]) / f_in_std[1]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.scatter(x_random, y_random)
    plt.xlim(-2., 2.); plt.ylim(-2., 2.)
    plt.grid(alpha=.3)
    plt.title("random")

    plt.subplot(1, 3, 2)
    plt.scatter(x_scaled1, y_scaled1)
    plt.xlim(-2., 2.); plt.ylim(-2., 2.)
    plt.grid(alpha=.3)
    plt.title("normalization")

    plt.subplot(1, 3, 3)
    plt.scatter(x_scaled2, y_scaled2)
    plt.xlim(-2., 2.); plt.ylim(-2., 2.)
    plt.grid(alpha=.3)
    plt.title("standardization")

    plt.tight_layout()
    plt.savefig(path_fig / "feature_scaling.png")
    plt.close()

    # define a model
    f_in = settings["PARAM_MODEL"]["f_in"]
    f_out = settings["PARAM_MODEL"]["f_out"]
    f_hid = settings["PARAM_MODEL"]["f_hid"]
    depth = settings["PARAM_MODEL"]["depth"]
    act = settings["PARAM_MODEL"]["act"]

    w_init = settings["PARAM_MODEL"]["w_init"]
    b_init = settings["PARAM_MODEL"]["b_init"]
    inv = settings["PARAM_MODEL"]["inv"]

    optim = settings["PARAM_OPTIM"]["optim"]
    lr = settings["PARAM_OPTIM"]["lr"]

    dn_flag = settings["PARAM_DN"]["flag"]
    tau = settings["PARAM_DN"]["tau"]
    beta = settings["PARAM_DN"]["beta"]

    # instantiate
    model = PINN(
        f_in, f_out, f_hid, depth, act,
        w_init=w_init, b_init=b_init,
        optim=optim, lr=lr,
        inv=inv, a0=1., a1=1.,
        dtype=tf.float64, seed=seed,
        f_scale="std", f_in_lb=f_in_lb, f_in_ub=f_in_ub, f_in_mean=f_in_mean, f_in_std=f_in_std,
        l_laaf=False, g_enhc=False, d_norm=dn_flag,
    )

    # history
    hist_dict = {
        "epoch": [],
        "loss_glb": [],
        "loss_pde": [],
        "loss_bc": [],
        "err_u": [],
        "err_r": [],
    }

    # training
    wait = 0
    loss_best = 9999.   # counts wait
    loss_save = 9999.   # saves weights
    t0 = time.perf_counter()
    for epoch in range(0, args.epochs+1):
        if dn_flag and epoch % tau == 0:
            # update gamma
            model.update_gamma(
                tf.cast(epoch, dtype=tf.float64),
                tf.cast(tau, dtype=tf.float64),
                tf.cast(beta, dtype=tf.float64),
                x_pde, y_pde, 
                x_nth, y_nth, u_nth,
                x_sth, y_sth, u_sth,
                x_est, y_est, u_est,
                x_wst, y_wst, u_wst,
            )

        # gradient descent
        loss_glb, loss_pde, loss_bc = model.train_step(
            x_pde, y_pde, 
            x_nth, y_nth, u_nth,
            x_sth, y_sth, u_sth,
            x_est, y_est, u_est,
            x_wst, y_wst, u_wst,
        )

        # log gamma
        gamma_bc = model.gamma_bc.numpy()
        gamma_bc_hat = model.gamma_bc_hat.numpy()

        # inference
        # x_reshape = tf.cast(x_reshape, dtype=tf.float64)
        # y_reshape = tf.cast(y_reshape, dtype=tf.float64)
        # u_, u_x_, u_y_, u_xx_, u_yy_, r_ = model.infer(x_reshape, y_reshape)
        # u_err = u_ - u_reshape

        x_reshape2 = tf.cast(x_reshape2, dtype=tf.float64)
        y_reshape2 = tf.cast(y_reshape2, dtype=tf.float64)
        u_, u_x_, u_y_, u_xx_, u_yy_, r_ = model.infer(x_reshape2, y_reshape2)
        u_err = u_ - u_reshape2
        r_err = r_
        u_err_l2 = np.linalg.norm(u_err, ord=2) / np.linalg.norm(u_reshape, ord=2)
        r_err_l2 = np.linalg.norm(r_err, ord=2)

        # print
        t1 = time.perf_counter()
        elps = t1 - t0
        train_log = \
            f"epoch: {epoch:d}, " \
            f"loss_glb: {loss_glb:.6e}, " \
            f"loss_pde: {loss_pde:.6e}, " \
            f"loss_bc: {loss_bc:.6e}, " \
            f"gamma_bc: {gamma_bc:.6e}, " \
            f"gamma_bc_hat: {gamma_bc_hat:.6e}, " \
            f"u_err_l2: {u_err_l2:.6e}, " \
            f"r_err_l2: {r_err_l2:.6e}, " \
            f"loss_best: {loss_best:.6e}, " \
            f"wait: {wait:d}, " \
            f"elps: {elps:.3f}"
        print(train_log)
        write_logger(path_train_log, train_log)

        # save weights
        if epoch % 200 == 0:
            model.save_weights(path_weights / f"weights_{epoch:05d}.h5")

        # early stopping
        # if loss_glb < loss_best:
            # loss_best = loss_glb
        if u_err_l2 < loss_best:
            loss_best = u_err_l2
            wait = 0
            if epoch % 200 == 0:
                model.save_weights(path_best_weights / "weights.h5")
        else:
            wait += 1
            if wait > args.patience:
                print(">>>>> early stopping")
                break

        # plot
        if epoch % 200 == 0:
            plot_3images(
                x=x_reshape2, xmin=0., xmax=1., xlabel=r"$x$",
                y=y_reshape2, ymin=0., ymax=1., ylabel=r"$y$",
                u1=u_, u1min=0., u1max=1.,   u1label=r"$\hat{u}$", u1cmap="turbo",
                u2=np.abs(u_err), u2min=0., u2max=5e-2, u2label=r"$\left| \hat{u} - u \right|$", u2cmap="turbo",
                u3=u_y_,  u3min=-1., u3max=1., u3label=r"$\partial_y \hat{u}$", u3cmap="seismic",
                epoch=epoch,
                save_path=path_fig,
                file_name="inf_err_grady",
                file_types=[".svg"]
            )

            plot_4images(
                x=x_reshape2, xmin=0., xmax=1., xlabel=r"$x$",
                y=y_reshape2, ymin=0., ymax=1., ylabel=r"$y$",
                u1=u_reshape2, u1min=0., u1max=1.,   u1label=r"$u$", u1cmap="turbo",
                u2=u_, u2min=0., u2max=1.,   u2label=r"$\hat{u}$", u2cmap="turbo",
                u3=np.abs(u_err), u3min=0., u3max=5e-2, u3label=r"$\left| \hat{u} - u \right|$", u3cmap="turbo",
                u4=u_y_,  u4min=-1., u4max=1., u4label=r"$\partial_y \hat{u}$", u4cmap="seismic",
                epoch=epoch,
                save_path=path_fig,
                file_name="ref_inf_err_grady",
                file_types=[".svg"]
            )



        #     plot_comparison_3D(
        #         x=x_ref, y=y_ref, u_ref=u_ref, u_inf=u_, u_err=u_err,
        #         umin=0., umax=1.,
        #         vmin=0., vmax=5e-2,
        #         xmin=xmin, xmax=xmax, xlabel=r"$x$",
        #         ymin=ymin, ymax=ymax, ylabel=r"$y$",
        #         epoch=epoch,
        #         save_path=path_figures,
        #         file_types=["png"]
        #     )
        #     plot_loss_curve(
        #         epoch_log,
        #         loss_glb_log, loss_pde_log, loss_bc_log,
        #         epoch=epoch,
        #         save_path=path_figures,
        #         file_types=["png"]
        #     )
        #     plot_grad_curve(
        #         epoch_log,
        #         grad_glb_log, grad_pde_log, grad_bc_log,
        #         epoch=epoch,
        #         save_path=path_figures,
        #         file_types=["png"]
        #     )
        #     plot_error_curve(
        #         epoch_log,
        #         u_l2_log, u_mse_log, u_sem_log,
        #         g_l2_log, g_mse_log, g_sem_log,
        #         epoch=epoch,
        #         save_path=path_figures,
        #         file_types=["png"]
        #     )
        #     plot_gamma_curve(
        #         epoch_log,
        #         gamma_bc_log, gamma_bc_hat_log,
        #         epoch=epoch,
        #         save_path=path_figures,
        #         file_types=["png"]
        #     )
        #     if epoch > 0:
        #         plot_grad_dist(
        #             model,
        #             depth,
        #             x_pde, y_pde,
        #             x_nth, y_nth, u_nth,
        #             x_sth, y_sth, u_sth,
        #             x_est, y_est, u_est,
        #             x_wst, y_wst, u_wst,
        #             epoch=epoch,
        #             save_path=path_figures,
        #             file_types=["png"]
        #         )


if __name__ == "__main__":
    plot_setting()
    config_device(args.device)
    main(args)

