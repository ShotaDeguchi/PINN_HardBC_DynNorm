"""
********************************************************************************
compute approximate distance functions based on R-functions
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

    # # cast
    # X = tf.cast(X, dtype=dtype)
    # Y = tf.cast(Y, dtype=dtype)
    # P1 = tf.cast(P1, dtype=dtype)
    # P2 = tf.cast(P2, dtype=dtype)

    # position (start and end of the line segment)
    x1, y1 = P1
    x2, y2 = P2

    # length of the line segment
    L = tf.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    L = tf.cast(L, dtype=dtype)

    # center of the line segment
    xc = (x1 + x2) / 2.
    yc = (y1 + y2) / 2.

    # unit vector targetial to the line segment
    tx = (x2 - x1) / L
    ty = (y2 - y1) / L

    # unit vector normal to the line segment
    nx = -ty
    ny =  tx

    # signed distance function
    # implicit representation of an infinite line passing through (x1, y1) and (x2, y2)
    f = ((X - x1) * (y2 - y1) - (Y - y1) * (x2 - x1)) / L

    # trimming function
    # implicit representation of a disk with radius L/2 and center (xc, yc)
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
    """
    compute ADF to an arc, defined by the center P, the radius R, and the angle A and B
    starting from the point P1 and ending at the point P2, which are computed from P, R, A, and B
    draw a circle in the counter-clockwise direction from P1 to P2 with radius R

    f: distance function to a circle (implicit representation of a circle with radius R and center (xc, yc))
    g: trimming function (implicit representation of a line that passes through P1=(x1, y1) and P2=(x2, y2))

    args:
        X, Y: coordinates of points to compute ADF
        P (tuple): coordinates of the center of the arc
        R (float): radius of the arc
        A (float): angle to the 1st point of the arc, starting from horizontal axis
        B (float): angle of the 2nd point of the arc, starting from horizontal axis (B > A)

    returns:
        phi: approximate distance function
    """

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

    # length of the line segment between P1 and P2
    L = tf.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # center of the line segment between P1 and P2
    xc = (x1 + x2) / 2.
    yc = (y1 + y2) / 2.

    # unit vector targetial to the line segment
    tx = (x2 - x1) / L
    ty = (y2 - y1) / L

    # unit vector normal to the line segment
    nx = -ty
    ny =  tx

    with tf.GradientTape(persistent=True) as tp:
        tp.watch([X, Y])

        # f: distance function to a circle (implicit representation of a circle with radius R and center P)
        f = (R**2 - ((X - x0)**2 + (Y - y0)**2)) / (2. * R)

        # g: trimming function (implicit representation of a line that passes through (x1, y1) and (x2, y2))
        g = ((X - x1) * (y2 - y1) - (Y - y1) * (x2 - x1)) / L

        # approximate distance function
        phi = (f**2 + (((f**4 + g**2)**.5 - g) / 2.)**2)**.5

    # compute gradients
    grad_phi = tp.gradient(phi, [X, Y])
    del tp
    return f, g, phi, grad_phi


def adf_elliptic_arc(X, Y, P, a, b, A, B, dtype=tf.float64):
    """
    compute ADF to an elliptic arc, defined by the center P, the angle A and B, and the semi-major and semi-minor axes a and b
    starting from the point P1 and ending at the point P2, which are computed from P, A, B, a, and b
    draw an elliptic arc in the counter-clockwise direction from P1 to P2 with semi-major axis a and semi-minor axis b

    f: distance function to an ellipse (implicit representation of an ellipse with semi-major axis a, semi-minor axis b, and center (xc, yc))
    g: trimming function (implicit representation of a line that passes through P1=(x1, y1) and P2=(x2, y2))

    args:
        X, Y: coordinates of points to compute ADF
        P (tuple): coordinates of the center of the arc
        a (float): semi-major axis of the ellipse
        b (float): semi-minor axis of the ellipse
        A (float): angle to the 1st point of the arc, starting from horizontal axis
        B (float): angle of the 2nd point of the arc, starting from horizontal axis (B > A)

    returns:
        phi: approximate distance function
    """

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

    # length of the line segment between P1 and P2
    L = tf.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # center of the line segment between P1 and P2
    xc = (x1 + x2) / 2.
    yc = (y1 + y2) / 2.

    # unit tangent vector
    tx = (x2 - x1) / L
    ty = (y2 - y1) / L

    # unit normal vector
    nx = -ty
    ny =  tx

    with tf.GradientTape(persistent=True) as tp:
        tp.watch([X, Y])

        # f: distance function to an ellipse (implicit representation of an ellipse with semi-major axis a, semi-minor axis b, and center P)
        f = (a**2 * b**2 - a**2 * (Y - y0)**2 - b**2 * (X - x0)**2) / (a * b)

        # g: trimming function (implicit representation of a line that passes through (x1, y1) and (x2, y2))
        g = ((X - x1) * (y2 - y1) - (Y - y1) * (x2 - x1)) / L

        # approximate distance function
        phi = (f**2 + (((f**4 + g**2)**.5 - g) / 2.)**2)**.5

    # compute gradients
    grad_phi = tp.gradient(phi, [X, Y])
    del tp
    return f, g, phi, grad_phi


def adf_circle(X, Y, P, R, dtype=tf.float64):
    """
    compute ADF to a circle

    args:
        X, Y: coordinates of points to compute ADF
        P (tuple): coordinates of the center of the circle
        R (float): radius of the circle

    returns:
        phi: approximate distance function
    """

    # cast
    X = tf.cast(X, dtype=dtype)
    Y = tf.cast(Y, dtype=dtype)
    P = tf.cast(P, dtype=dtype)
    R = tf.cast(R, dtype=dtype)

    # position
    x0, y0 = P

    with tf.GradientTape(persistent=True) as tp:
        tp.watch([X, Y])

        # adf to a circle is directly computed
        phi = (R**2 - ((X - x0)**2 + (Y - y0)**2)) / (2. * R)

        # zero on the boundary, positive elsewhere
        phi = tf.sqrt(phi**2)

    # compute gradients
    grad_phi = tp.gradient(phi, [X, Y])
    del tp
    return phi, grad_phi


def adf_ellipse(X, Y, P, a, b, dtype=tf.float64):
    """
    compute ADF to an ellipse

    args:
        X, Y: coordinates of points to compute ADF
        P (tuple): coordinates of the center of the ellipse
        a (float): semi-major axis of the ellipse
        b (float): semi-minor axis of the ellipse

    returns:
        phi: approximate distance function
    """

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

        # adf to an ellipse is directly computed
        phi = (a**2 * b**2 - a**2 * (Y - y0)**2 - b**2 * (X - x0)**2) / (a * b)

        # zero on the boundary, positive elsewhere
        phi = tf.sqrt(phi**2)

    # compute gradients
    grad_phi = tp.gradient(phi, [X, Y])
    del tp
    return phi, grad_phi


def join_two_adfs(phi_1, phi_2, m=1):
    """
    join two ADFs with normalization
    normalization order m >= 1

    args:
        phi_1, phi_2: approximate distance functions to line segments
        m: normalization parameter (m >= 1)

    returns:
        phi: joined approximate distance function
    """

    phi = phi_1 * phi_2 / (phi_1**m + phi_2**m)**(1. / m)
    return phi


def join_multiple_adfs(phis, m=1):
    """
    join multiple ADFs

    args:
        phis: list of approximate distance functions to line segments
        m: normalization parameter (m >= 1)

    returns:
        phi: joined approximate distance function
    """

    # use join_two_adfs() recursively
    phi = phis[0]
    for i in range(1, len(phis)):
        phi = join_two_adfs(phi, phis[i], m=m)
    return phi


def transfinite_interpolation(gs, phis, mus, method=2):
    """
    transfinite interpolation

    args:
        gs:   g's;   list of Dirichlet boundary values
        phis: phi's; list of approximate distance functions
        mus:  mu's;  list of interpolation parameters (interpolant is (mu-1) times differentiable)
        method: method to compute the weights (1 or 2, default: 2)

    returns:
        g: interpolant
    g = sum_i (w_i * g_i),
    where w_i = phi_i^(-mu_i) / sum_j (phi_j^(-mu_j))
              = prod_{j != i} (phi_j^(mu_j)) / sum_k (prod_{j != k} (phi_j^(mu_j)))
    """

    if method == 1:
        # method 1 (1st equation, said to be numerically unstable)
        denom = 0.
        for j in range(len(phis)):
            denom += phis[j]**(-mus[j])

        g = 0.
        for i in range(len(phis)):
            w = phis[i]**(- mus[i]) / denom
            g += w * gs[i]

    elif method == 2:
        # method 2 (2nd equation, said to be numerically stable)
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
