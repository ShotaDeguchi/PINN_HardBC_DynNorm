"""
********************************************************************************
Author: Shota DEGUCHI
        Structural Analysis Laboratory, Kyushu University (Aug. 25th, 2023)

compute approximate distance functions based on Rvachev's method
********************************************************************************
"""

import numpy as np
import tensorflow as tf


def adf_line_segment(X, Y, P1, P2, dtype=tf.float64):
    """
    compute ADF to a line segment

    args:
        X, Y: coordinates of points to compute ADF
        P1 (tuple): coordinates of the 1st point of the line segment
        P2 (tuple): coordinates of the 2nd point of the line segment

    returns:
        phi: approximate distance function
    """

    # position (start and end of the line segment)
    x1, y1 = P1
    x2, y2 = P2
    L = tf.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    L = tf.cast(L, dtype=dtype)
    xc = (x1 + x2) / 2.
    yc = (y1 + y2) / 2.

    # tangential and normal
    tx = (x2 - x1) / L
    ty = (y2 - y1) / L
    nx = -ty
    ny =  tx

    # signed distance function
    f = ((X - x1) * (y2 - y1) - (Y - y1) * (x2 - x1)) / L

    # trimming function
    g = ((L / 2.)**2 - ((X - xc)**2 + (Y - yc)**2)) / L

    # approximate distance function
    phi = (f**2 + (((f**4 + g**2)**.5 - g) / 2.)**2)**.5

    return phi

    # with tf.GradientTape(persistent=True) as tp:
    #     tp.watch([X, Y])

    #     # signed distance function
    #     # implicit representation of an infinite line passing through (x1, y1) and (x2, y2)
    #     f = ((X - x1) * (y2 - y1) - (Y - y1) * (x2 - x1)) / L

    #     # trimming function
    #     # implicit representation of a disk with radius L/2 and center (xc, yc)
    #     g = ((L / 2.)**2 - ((X - xc)**2 + (Y - yc)**2)) / L

    #     # approximate distance function
    #     phi = (f**2 + (((f**4 + g**2)**.5 - g) / 2.)**2)**.5

    # # compute gradients
    # grad_phi = tp.gradient(phi, [X, Y])
    # del tp
    # return f, g, phi, grad_phi


def adf_circular_arc(X, Y, P, R, A, B, dtype=tf.float64):
    # cast
    X = tf.cast(X, dtype=dtype)
    Y = tf.cast(Y, dtype=dtype)
    P = tf.cast(P, dtype=dtype)
    R = tf.cast(R, dtype=dtype)
    A = tf.cast(A, dtype=dtype)
    B = tf.cast(B, dtype=dtype)

    # position
    x0, y0 = P
    x1 = x0 + R * tf.cos(A)   # P1 = (x1, y1)
    y1 = y0 + R * tf.sin(A)
    x2 = x0 + R * tf.cos(B)   # P2 = (x2, y2)
    y2 = y0 + R * tf.sin(B)
    L = tf.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    xc = (x1 + x2) / 2.
    yc = (y1 + y2) / 2.

    tx = (x2 - x1) / L
    ty = (y2 - y1) / L
    nx = -ty
    ny =  tx

    with tf.GradientTape(persistent=True) as tp:
        tp.watch([X, Y])
        f = (R**2 - ((X - x0)**2 + (Y - y0)**2)) / (2. * R)
        g = ((X - x1) * (y2 - y1) - (Y - y1) * (x2 - x1)) / L
        phi = (f**2 + (((f**4 + g**2)**.5 - g) / 2.)**2)**.5
    grad_phi = tp.gradient(phi, [X, Y])
    del tp
    return f, g, phi, grad_phi


def adf_elliptic_arc(X, Y, P, a, b, A, B, dtype=tf.float64):
    # cast
    X = tf.cast(X, dtype=dtype)
    Y = tf.cast(Y, dtype=dtype)
    P = tf.cast(P, dtype=dtype)
    A = tf.cast(A, dtype=dtype)
    B = tf.cast(B, dtype=dtype)
    a = tf.cast(a, dtype=dtype)
    b = tf.cast(b, dtype=dtype)

    # position
    x0, y0 = P
    x1 = x0 + a * tf.cos(A)   # P1 = (x1, y1)
    y1 = y0 + b * tf.sin(A)
    x2 = x0 + a * tf.cos(B)   # P2 = (x2, y2)
    y2 = y0 + b * tf.sin(B)
    L = tf.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    xc = (x1 + x2) / 2.
    yc = (y1 + y2) / 2.

    tx = (x2 - x1) / L
    ty = (y2 - y1) / L
    nx = -ty
    ny =  tx

    with tf.GradientTape(persistent=True) as tp:
        tp.watch([X, Y])
        f = (a**2 * b**2 - a**2 * (Y - y0)**2 - b**2 * (X - x0)**2) / (a * b)
        g = ((X - x1) * (y2 - y1) - (Y - y1) * (x2 - x1)) / L
        phi = (f**2 + (((f**4 + g**2)**.5 - g) / 2.)**2)**.5
    grad_phi = tp.gradient(phi, [X, Y])
    del tp
    return f, g, phi, grad_phi


def adf_circle(X, Y, P, R, dtype=tf.float64):
    # cast
    X = tf.cast(X, dtype=dtype)
    Y = tf.cast(Y, dtype=dtype)
    P = tf.cast(P, dtype=dtype)
    R = tf.cast(R, dtype=dtype)

    # position
    x0, y0 = P

    with tf.GradientTape(persistent=True) as tp:
        tp.watch([X, Y])
        phi = (R**2 - ((X - x0)**2 + (Y - y0)**2)) / (2. * R)
        phi = tf.sqrt(phi**2)
    grad_phi = tp.gradient(phi, [X, Y])
    del tp
    return phi, grad_phi


def adf_ellipse(X, Y, P, a, b, dtype=tf.float64):
    # cast
    X = tf.cast(X, dtype=dtype)
    Y = tf.cast(Y, dtype=dtype)
    P = tf.cast(P, dtype=dtype)
    a = tf.cast(a, dtype=dtype)
    b = tf.cast(b, dtype=dtype)

    # position
    x0, y0 = P

    with tf.GradientTape(persistent=True) as tp:
        tp.watch([X, Y])
        phi = (a**2 * b**2 - a**2 * (Y - y0)**2 - b**2 * (X - x0)**2) / (a * b)
        phi = tf.sqrt(phi**2)
    grad_phi = tp.gradient(phi, [X, Y])
    del tp
    return phi, grad_phi


def join_two_adfs(phi_1, phi_2, m=1):
    phi = phi_1 * phi_2 / (phi_1**m + phi_2**m)**(1. / m)
    return phi


def join_multiple_adfs(phis, m=1):
    phi = phis[0]
    for i in range(1, len(phis)):
        phi = join_two_adfs(phi, phis[i], m=m)
    return phi


def transfinite_interpolation(gs, phis, mus, method=2):
    if method == 1:
        # method 1 (1st equation, numerically unstable)
        denom = 0.
        for j in range(len(phis)):
            denom += phis[j]**(-mus[j])

        g = 0.
        for i in range(len(phis)):
            w = phis[i]**(- mus[i]) / denom
            g += w * gs[i]

    elif method == 2:
        # method 2 (2nd equation, numerically stable)
        ws = []
        for i in range(len(phis)):
            w = 1.
            for j in range(len(phis)):
                if i != j:
                    w *= phis[j]**mus[j]
            ws.append(w)

        denom = 0.
        for w in ws:
            denom += w

        g = 0.
        for i in range(len(phis)):
            g += ws[i] / denom * gs[i]

    return g
