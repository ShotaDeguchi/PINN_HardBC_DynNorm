"""
********************************************************************************
inference
********************************************************************************
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import yaml
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
parser.add_argument("-d", "--device", type=int, default=-1, help="device id")
args = parser.parse_args()


def main(args):
    # read 
    with open("./settings.yaml", mode="r") as f:
        settings = yaml.safe_load(f)

    # seed
    seed = args.seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # prepare directory
    dir_path = "seed" + str(seed)
    path_figures, path_saved_weights, path_best_weights = check_dirs(dir_path)

    # prepare logger
    logger_path = make_logger(
        save_path=dir_path,
        file_name="infer_log",
        file_type=".csv",
        seed=seed
    )

    # define a domain
    xmin, xmax, nx = 0., 1., 201
    ymin, ymax, ny = 0., 1., 201
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    x, y = np.meshgrid(x, y)

    x_ref, y_ref = x.reshape(-1, 1), y.reshape(-1, 1)
    x_ref, y_ref = tf.cast(x_ref, dtype=tf.float64), tf.cast(y_ref, dtype=tf.float64)

    # reference (fdm solution)
    x_fdm = np.load("../99_fdm/poisson_fdm_x.npy")
    y_fdm = np.load("../99_fdm/poisson_fdm_y.npy")
    u_fdm = np.load("../99_fdm/poisson_fdm_u.npy")
    u_ref = u_fdm.reshape(-1, 1)
    u_ref = tf.cast(u_ref, dtype=tf.float64)

    # bounds
    in_lb = tf.constant([xmin, ymin], dtype=tf.float64)
    in_ub = tf.constant([xmax, ymax], dtype=tf.float64)
    in_mean = tf.reduce_mean([in_lb, in_ub], axis=0)

    # define a model
    f_in  = settings["NET_ARCH"]["f_in"]
    f_out = settings["NET_ARCH"]["f_out"]
    f_hid = settings["NET_ARCH"]["f_hid"]
    depth = settings["NET_ARCH"]["depth"]

    inv   = settings["PARAM_TRAIN"]["inv"]
    lr    = settings["PARAM_TRAIN"]["lr"]
    beta  = settings["PARAM_TRAIN"]["beta"]
    tau   = settings["PARAM_TRAIN"]["tau"]

    model = PINN(
        f_in, f_out, f_hid, depth, 
        in_lb, in_ub, in_mean, inv, 
        w_init="Glorot", b_init="zeros", act="repu", lr=lr, seed=seed
    )
    model.load_weights(os.path.join(path_best_weights, "best_weights"))

    # inference
    u_, u_x_, u_y_, u_xx_, u_yy_, g_ = model.infer(x_ref, y_ref)
    # u_err = u_ - u_ref
    u_err = tf.reshape(u_, shape=[ny, nx])[1:-1, 1:-1] - tf.reshape(u_ref, shape=[ny, nx])[1:-1, 1:-1]
    u_l2 = np.linalg.norm(u_err, ord=2) / np.linalg.norm(u_ref, ord=2)
    u_mse = np.mean(np.square(u_err))
    u_sem = np.std (np.square(u_err), ddof=1) / np.sqrt(u_err.shape[0])
    # g_err = g_
    g_err = tf.reshape(g_, shape=[ny, nx])[1:-1, 1:-1]
    g_l2 = np.linalg.norm(g_err, ord=2)
    g_mse = np.mean(np.square(g_err))
    g_sem = np.std (np.square(g_err), ddof=1) / np.sqrt(g_err.shape[0])

    logger_data = \
        f"u_l2: {u_l2:.6e}, " \
        f"u_mse: {u_mse:.6e}, " \
        f"u_sem: {u_sem:.6e}, " \
        f"g_l2: {g_l2:.6e}, " \
        f"g_mse: {g_mse:.6e}, " \
        f"g_sem: {g_sem:.6e}"
    print(logger_data)
    write_logger(logger_path, logger_data)

    plot_comparison(
        x=x_ref, y=y_ref, u_ref=u_ref, u_inf=u_, u_err=u_err,
        umin=0., umax=1.,
        vmin=0., vmax=.03,
        xmin=xmin, xmax=xmax, xlabel=r"$x$",
        ymin=ymin, ymax=ymax, ylabel=r"$y$",
        epoch="inference",
        save_path=path_figures,
        file_types=[".png", ".svg", ".pdf"]
    )
    plot_comparison_3D(
        x=x_ref, y=y_ref, u_ref=u_ref, u_inf=u_, u_err=u_err,
        umin=0., umax=1.,
        vmin=0., vmax=.03,
        xmin=xmin, xmax=xmax, xlabel=r"$x$",
        ymin=ymin, ymax=ymax, ylabel=r"$y$",
        epoch="inference",
        save_path=path_figures,
        file_types=[".png", ".svg", ".pdf"]
    )



if __name__ == "__main__":
    plot_setting()
    config_device(args.device)
    main(args)

