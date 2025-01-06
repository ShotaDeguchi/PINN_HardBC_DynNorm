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
import pyDOE
import sobol

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
    ref_data = np.load("../99_reference/results.npz")
    # x_ref = ref_data['x']   # 1D
    # y_ref = ref_data['y']
    X_ref = ref_data['X']   # 2D (meshgrid)
    Y_ref = ref_data['Y']
    u_ref = ref_data['u']   # 2D (meshgrid)
    v_ref = ref_data['v']
    p_ref = ref_data['p']   # raw pressure
    p_bar_ref = ref_data['p_bar']   # pressure with zero mean
    w_ref = np.sqrt(u_ref**2 + v_ref**2)
    p_ref = (p_ref[1:, 1:] + p_ref[:-1, 1:] + p_ref[1:, :-1] + p_ref[:-1, :-1]) / 4.
    p_bar_ref = (p_bar_ref[1:, 1:] + p_bar_ref[:-1, 1:] + p_bar_ref[1:, :-1] + p_bar_ref[:-1, :-1]) / 4.

    # # reshape
    # x_reshaped = X_ref.reshape(-1, 1); print(f'min: {np.min(x_reshaped)}, max: {np.max(x_reshaped)}, shape: {x_reshaped.shape}')
    # y_reshaped = Y_ref.reshape(-1, 1)
    # u_reshaped = u_ref.reshape(-1, 1)
    # v_reshaped = v_ref.reshape(-1, 1)
    # vel_norm_reshaped = w_ref.reshape(-1, 1)
    # prs_reshaped = pressure_ref.reshape(-1, 1)
    # prs_bar_reshaped = pressure_bar_ref.reshape(-1, 1)

    # remove edges
    x_removed = X_ref[2:-2,2:-2]
    y_removed = Y_ref[2:-2,2:-2]
    u_removed = u_ref[2:-2,2:-2]
    v_removed = v_ref[2:-2,2:-2]
    w_removed = w_ref[2:-2,2:-2]
    p_removed = p_ref[1:-1,1:-1]
    p_bar_removed = p_bar_ref[1:-1,1:-1]

    # reshape
    x_ref = x_removed.reshape(-1, 1)
    y_ref = y_removed.reshape(-1, 1)
    u_ref = u_removed.reshape(-1, 1)
    v_ref = v_removed.reshape(-1, 1)
    w_ref = w_removed.reshape(-1, 1)   # w = sqrt(u**2 + v**2)
    p_ref = p_removed.reshape(-1, 1)   # raw pressure
    p_bar_ref = p_bar_removed.reshape(-1, 1)   # pressure with zero mean

    # collocation points
    # 2**6 = 64, 2**8 = 256, 2**10 = 1024, 2**12 = 4096
    N_pde = int(2**12)
    N_bc  = int(2**8)
    N_dat = int(2**8)

    eps = 0.
    xmin, xmax = 0. + eps, 1. - eps
    ymin, ymax = 0. + eps, 1. - eps

    # pde residual
    M = 2   # spatial dimension
    X_pde = tf.random.uniform(shape=(N_pde, M), minval=0., maxval=1., dtype=tf.float64, seed=seed)
    x_pde = tf.constant(X_pde[:, 0:1], dtype=tf.float64)
    y_pde = tf.constant(X_pde[:, 1:2], dtype=tf.float64)
    plt.figure(figsize=(4, 4))
    plt.scatter(x_pde, y_pde, marker='.')
    plt.title(
        f'N = {x_pde.shape[0]:d}, '
        f'xmin = {np.min(x_pde.numpy()):.3e}, '
        f'xmax = {np.max(x_pde.numpy()):.3e}'
    )
    plt.savefig(path_fig / "mc.png")
    plt.close()

    # for LHS
    M = 2
    X_lhs = pyDOE.lhs(M, N_pde)
    x_lhs = tf.constant(X_lhs[:, 0:1], dtype=tf.float64)
    y_lhs = tf.constant(X_lhs[:, 1:2], dtype=tf.float64)
    plt.figure(figsize=(4, 4))
    plt.scatter(x_lhs, y_lhs, marker='.')
    plt.title(
        f'N = {x_lhs.shape[0]:d}, '
        f'xmin = {np.min(x_lhs.numpy()):.3e}, '
        f'xmax = {np.max(x_lhs.numpy()):.3e}'
    )
    plt.savefig(path_fig / "lhs.png")
    plt.close()

    # for Sobol sequence
    X_sbl = sobol.sample(M, n_points=N_pde)
    x_sbl = tf.constant(X_sbl[:, 0:1], dtype=tf.float64)
    y_sbl = tf.constant(X_sbl[:, 1:2], dtype=tf.float64)
    plt.figure(figsize=(4, 4))
    plt.scatter(x_sbl, y_sbl, marker='.')
    plt.title(
        f'N = {x_sbl.shape[0]:d}, '
        f'xmin = {np.min(x_sbl.numpy()):.3e}, '
        f'xmax = {np.max(x_sbl.numpy()):.3e}'
    )
    plt.savefig(path_fig / "sbl.png")
    plt.close()

    # north
    x_nth = tf.random.uniform((N_bc, 1), xmin, xmax, dtype=tf.float64, seed=seed)
    y_nth = tf.ones_like(x_nth) * ymax
    u_nth = tf.ones_like(x_nth)
    v_nth = tf.zeros_like(x_nth)
    # south
    x_sth = tf.random.uniform((N_bc, 1), xmin, xmax, dtype=tf.float64, seed=seed)
    y_sth = tf.ones_like(x_sth) * ymin
    u_sth = tf.zeros_like(x_sth)
    v_sth = tf.zeros_like(x_sth)
    # east
    y_est = tf.random.uniform((N_bc, 1), ymin, ymax, dtype=tf.float64, seed=seed)
    x_est = tf.ones_like(y_est) * xmax
    u_est = tf.zeros_like(y_est)
    v_est = tf.zeros_like(y_est)
    # west
    y_wst = tf.random.uniform((N_bc, 1), ymin, ymax, dtype=tf.float64, seed=seed)
    x_wst = tf.ones_like(y_wst) * xmin
    u_wst = tf.zeros_like(y_wst)
    v_wst = tf.zeros_like(y_wst)

    # data
    idx_dat = np.random.choice(x_ref.shape[0], N_dat, replace=False)
    x_dat = x_ref[idx_dat]
    y_dat = y_ref[idx_dat]
    u_dat = u_ref[idx_dat]
    v_dat = v_ref[idx_dat]
    w_dat = w_ref[idx_dat]
    p_dat = p_ref[idx_dat]

    # plot
    plt.figure(figsize=(4, 4))
    plt.scatter(x_pde, y_pde, alpha=.7, marker='.', label=r"$\mathcal{R}$" + f" ({x_pde.shape[0]:d})")
    plt.scatter(x_nth, y_nth, alpha=.7, marker='.', label=r"$\mathcal{N}$" + f" ({x_nth.shape[0]:d})")
    plt.scatter(x_sth, y_sth, alpha=.7, marker='.', label=r"$\mathcal{S}$" + f" ({x_sth.shape[0]:d})")
    plt.scatter(x_est, y_est, alpha=.7, marker='.', label=r"$\mathcal{E}$" + f" ({x_est.shape[0]:d})")
    plt.scatter(x_wst, y_wst, alpha=.7, marker='.', label=r"$\mathcal{W}$" + f" ({x_wst.shape[0]:d})")
    plt.scatter(x_dat, y_dat, alpha=.7, marker='.', label=r"$\mathcal{D}$" + f" ({x_dat.shape[0]:d})")
    plt.legend(loc="center")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.tight_layout()
    plt.savefig(path_fig / "collocation_points.png")
    plt.close()

    # bounds (for feature scaling)
    f_in_lb = tf.constant([xmin, ymin], dtype=tf.float64)
    f_in_ub = tf.constant([xmax, ymax], dtype=tf.float64)
    f_in_mean = tf.constant([np.mean(x_pde), np.mean(y_pde)], dtype=tf.float64)
    f_in_std = tf.constant([np.std(x_pde, ddof=1), np.std(y_pde, ddof=1)], dtype=tf.float64)

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

    # learning rate decay
    # lr = tf.keras.optimizers.schedules.ExponentialDecay(
    #     lr, decay_steps=1000, decay_rate=.9, staircase=False
    # )
    # lr = tf.keras.optimizers.schedules.ExponentialDecay(
    #     lr, decay_steps=2000, decay_rate=.9, staircase=False
    # )
    # lr = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate=lr,
    #     decay_steps=args.epochs,
    #     alpha=1e-2
    # )

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
                x_nth, y_nth, u_nth, v_nth,
                x_sth, y_sth, u_sth, v_sth,
                x_est, y_est, u_est, v_est,
                x_wst, y_wst, u_wst, v_wst,
                x_dat, y_dat, u_dat, v_dat,
            )

        # gradient descent
        loss_glb, loss_pde, loss_bc, loss_dat = model.train_step(
            x_pde, y_pde, 
            x_nth, y_nth, u_nth, v_nth,
            x_sth, y_sth, u_sth, v_sth,
            x_est, y_est, u_est, v_est,
            x_wst, y_wst, u_wst, v_wst,
            x_dat, y_dat, u_dat, v_dat,
        )

        # log gamma
        gamma_bc  = model.gamma_bc.numpy()
        gamma_dat = model.gamma_dat.numpy()
        gamma_div = model.gamma_div.numpy()
        gamma_bc_hat  = model.gamma_bc_hat.numpy()
        gamma_dat_hat = model.gamma_dat_hat.numpy()
        gamma_div_hat = model.gamma_div_hat.numpy()

        # log parameter estimate
        Re_hat = model.Re.numpy()
        Re_tilde = tf.math.exp(Re_hat)

        # inference
        u_hat, v_hat, p_hat, r0_hat, r1_hat, r2_hat = model.infer(x_ref, y_ref)
        w_hat = tf.sqrt(u_hat**2 + v_hat**2)
        p_bar_hat = p_hat - tf.reduce_mean(p_hat)
        u_err = u_hat - u_ref
        v_err = v_hat - v_ref
        w_err = w_hat - w_ref
        p_err = p_hat - p_ref
        p_bar_err = p_bar_hat - p_bar_ref

        u_err_l2 = np.linalg.norm(u_err, ord=2) / np.linalg.norm(u_ref, ord=2)
        v_err_l2 = np.linalg.norm(v_err, ord=2) / np.linalg.norm(v_ref, ord=2)
        w_err_l2 = np.linalg.norm(w_err, ord=2) / np.linalg.norm(w_ref, ord=2)
        p_err_l2 = np.linalg.norm(p_err, ord=2) / np.linalg.norm(p_ref, ord=2)
        p_bar_err_l2 = np.linalg.norm(p_bar_err, ord=2) / np.linalg.norm(p_bar_ref, ord=2)
        r0_err_l2 = np.linalg.norm(r0_hat, ord=2)
        r1_err_l2 = np.linalg.norm(r1_hat, ord=2)
        r2_err_l2 = np.linalg.norm(r2_hat, ord=2)

        # print
        t1 = time.perf_counter()
        elps = t1 - t0
        train_log = \
            f"epoch: {epoch:d}, " \
            f"loss_glb: {loss_glb:.6e}, " \
            f"loss_pde: {loss_pde:.6e}, " \
            f"loss_bc: {loss_bc:.6e}, " \
            f"loss_dat: {loss_dat:.6e}, " \
            f"gamma_bc: {gamma_bc:.6e}, " \
            f"gamma_bc_hat: {gamma_bc_hat:.6e}, " \
            f"gamma_dat: {gamma_dat:.6e}, " \
            f"gamma_dat_hat: {gamma_dat_hat:.6e}, " \
            f"gamma_div: {gamma_div:.6e}, " \
            f"gamma_div_hat: {gamma_div_hat:.6e}, " \
            f"Re_hat: {Re_hat:.6e}, " \
            f"Re_tilde: {Re_tilde:.6e}, " \
            f"u_err_l2: {u_err_l2:.6e}, " \
            f"v_err_l2: {v_err_l2:.6e}, " \
            f"w_err_l2: {w_err_l2:.6e}, " \
            f"p_err_l2: {p_err_l2:.6e}, " \
            f"p_bar_err_l2: {p_bar_err_l2:.6e}, " \
            f"r0_err_l2: {r0_err_l2:.6e}, " \
            f"r1_err_l2: {r1_err_l2:.6e}, " \
            f"r2_err_l2: {r2_err_l2:.6e}, " \
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

        # monitor
        if epoch % 200 == 0:
            plot_comparison(
                x=x_ref, y=y_ref, u_ref=w_ref, u_inf=w_hat, u_err=w_err,
                umin=0., umax=1., 
                vmin=0., vmax=5e-2,
                xmin=0., xmax=1., xlabel=r"$x$",
                ymin=0., ymax=1., ylabel=r"$y$",
                epoch=epoch,
                save_path=path_fig,
                file_name="velocity_norm",
                file_types=[".svg"]
            )
            plot_comparison(
                x=x_ref, y=y_ref, u_ref=p_ref, u_inf=p_hat, u_err=p_err,
                umin=-.1, umax=.1, 
                vmin=0., vmax=5e-2,
                xmin=0., xmax=1., xlabel=r"$x$",
                ymin=0., ymax=1., ylabel=r"$y$",
                epoch=epoch,
                save_path=path_fig,
                file_name="pressure",
                file_types=[".svg"]
            )
            plot_comparison(
                x=x_ref, y=y_ref, u_ref=p_bar_ref, u_inf=p_bar_hat, u_err=p_bar_err,
                umin=-.1, umax=.1, 
                vmin=0., vmax=5e-2,
                xmin=0., xmax=1., xlabel=r"$x$",
                ymin=0., ymax=1., ylabel=r"$y$",
                epoch=epoch,
                save_path=path_fig,
                file_name="pressure_bar",
                file_types=[".svg"]
            )

if __name__ == "__main__":
    plot_setting()
    config_device(args.device)
    main(args)

