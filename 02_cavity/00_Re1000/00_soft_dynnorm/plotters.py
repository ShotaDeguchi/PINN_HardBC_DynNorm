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
    # plt.style.use("seaborn-deep")
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
    levels = np.linspace(umin, umax+1e-6, 32)
    ticks = np.linspace(umin, umax+1e-6, 5)
    plt.contourf(
        tf.reshape(x, shape=(199, 199)),
        tf.reshape(y, shape=(199, 199)),
        tf.reshape(u_ref, shape=(199, 199)),
        levels=levels, extend="both", cmap="turbo"
    )
    plt.colorbar(ticks=ticks)
    plt.xticks(np.arange(xmin, xmax+1e-6, xticks))
    plt.yticks(np.arange(ymin, ymax+1e-6, yticks))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Reference")

    plt.subplot(1, 3, 2)
    plt.contourf(
        tf.reshape(x, shape=(199, 199)),
        tf.reshape(y, shape=(199, 199)),
        tf.reshape(u_inf, shape=(199, 199)),
        levels=levels, extend="both", cmap="turbo"
    )
    plt.colorbar(ticks=ticks)
    plt.xticks(np.arange(xmin, xmax+1e-6, xticks))
    plt.yticks(np.arange(ymin, ymax+1e-6, yticks))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Inference")

    plt.subplot(1, 3, 3)
    levels = np.linspace(vmin, vmax+1e-6, 32)
    ticks = np.linspace(vmin, vmax+1e-6, 5)
    plt.contourf(
        tf.reshape(x, shape=(199, 199)),
        tf.reshape(y, shape=(199, 199)),
        tf.reshape(np.abs(u_err), shape=(199, 199)),
        levels=levels, extend="both", cmap="turbo"
    )
    plt.colorbar(ticks=ticks)
    plt.xticks(np.arange(xmin, xmax+1e-6, xticks))
    plt.yticks(np.arange(ymin, ymax+1e-6, yticks))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    u_err = tf.norm(u_err, ord=2) / tf.norm(u_ref, ord=2)
    plt.title(rf"Abs. err. (rel. $\ell^2$ err.: {u_err:.3e})")

    plt.tight_layout()
    file_name = "comparison_" + file_name + "_" + str(epoch)
    for file_type in file_types:
        plt.savefig(os.path.join(save_path, file_name + file_type), dpi=300)
    plt.close()
