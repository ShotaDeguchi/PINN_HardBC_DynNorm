"""
********************************************************************************
Author: Shota DEGUCHI
        Yosuke SHIBATA
        Structural Analysis Laboratory, Kyushu University (Jul. 19th, 2021)
Physics-Informed Neural Networks (Raissi+2019)
********************************************************************************
"""

import os
import time

import numpy as np
import tensorflow as tf

import rvachev

class PINN(tf.keras.Model):
    def __init__(
        self,
        f_in, f_out, f_hid, depth, act,
        w_init="Glorot", b_init="zeros",
        optim="Adam", lr=1e-3,
        inv=False, a0=1., a1=1.,
        dtype=tf.float64, seed=42,
        f_scale=None, f_in_lb=None, f_in_ub=None, f_in_mean=None, f_in_std=None,
        l_laaf=False, g_enhc=False, d_norm=False,
    ):
        super().__init__()

        # dtype and seed
        self.d_type = dtype
        self.r_seed = seed
        os.environ["PYTHONHASHSEED"] = str(self.r_seed)
        np.random.seed(self.r_seed)
        tf.random.set_seed(self.r_seed)

        # arch
        self.f_in   = f_in
        self.f_out  = f_out
        self.f_hid  = f_hid
        self.depth  = depth
        self.act    = act

        # enhancements
        self.l_laaf = l_laaf   # L-LAAF (Jagtap+2020)
        self.g_enhc = g_enhc   # Gradient Enhancement (Yu+2022)
        self.d_norm = d_norm   # Dynamic Normalization, w/ bias corr (Deguchi+2023)

        # neural network
        self._layers = [self.f_in] + (self.depth - 1) * [self.f_hid] + [self.f_out]
        self._weights, self._biases, self._alphas, self._params \
            = self.dnn_initializer(self._layers, w_init, b_init)
        self._n_params  = self.flatten(self._params) .shape[0]
        self._n_weights = self.flatten(self._weights).shape[0]
        self._n_biases  = self.flatten(self._biases) .shape[0]
        print(f">>> self._n_params: {self._n_params}")
        print(f">>> self._n_weights: {self._n_weights}")
        print(f">>> self._n_biases: {self._n_biases}")

        # optimizer
        print(f">>> optimizer: {optim}")
        self.optimizer = self.get_optimizer(optim, lr)

        # inverse analysis?
        self.inv = inv
        self.Re  = a0
        # self.a1  = a1
        if self.inv:
            print(">>> inverse analysis, adding trainable variables...")
            self.Re = tf.Variable(a0, dtype=self.d_type)
            # self.a1 = tf.Variable(a1, dtype=self.d_type)
            self._params.append(self.Re)
            # self._params.append(self.a1)
        else:
            print(">>> forward analysis, using constant values...")
            self.Re = tf.constant(a0, dtype=self.d_type)
            self.a1 = tf.constant(a1, dtype=self.d_type)

        # feature scaling
        self.f_scale = f_scale   # "linear" / "minmax" / "mean" / "std"
        self.f_in_lb = f_in_lb
        self.f_in_ub = f_in_ub
        self.f_in_mean = f_in_mean
        self.f_in_std = f_in_std

        # dynamic normalization (Deguchi+2023)
        # biased estimator
        self.gamma_bc = tf.Variable(0., dtype=self.d_type)
        self.gamma_dat = tf.Variable(0., dtype=self.d_type)
        self.gamma_div = tf.Variable(0., dtype=self.d_type)
        # unbiased estimator
        self.gamma_bc_hat = tf.Variable(0., dtype=self.d_type)
        self.gamma_dat_hat = tf.Variable(0., dtype=self.d_type)
        self.gamma_div_hat = tf.Variable(0., dtype=self.d_type)

        # hello
        print("***************************************************************")
        print("************************* DELLO WORLD *************************")
        print("***************************************************************")
        time.sleep(3)

################################################################################

    def dnn_initializer(self, layers, w_init, b_init):
        weights = []
        biases  = []
        alphas  = []
        params  = []
        for l in range(0, self.depth):
            w = self.weight_initializer(
                init=w_init, shape=[layers[l], layers[l+1]], depth=l
            )
            b = self.bias_initializer(
                init=b_init, shape=[1, layers[l+1]], depth=l
            )
            weights.append(w)
            biases.append(b)
            params.append(w)
            params.append(b)
            if self.l_laaf and l < self.depth - 1:
                a = tf.Variable(1., dtype=self.d_type, name="a"+str(l))
                alphas.append(a)
                params.append(a)
            elif not self.l_laaf and l < self.depth - 1:
                a = tf.constant(1., dtype=self.d_type, name="a"+str(l))
                alphas.append(a)
        return weights, biases, alphas, params

    def weight_initializer(self, init, shape, depth):
        in_dim  = shape[0]
        out_dim = shape[1]
        if init == "Glorot":
            std = np.sqrt(2 / (in_dim + out_dim))
        elif init == "He":
            std = np.sqrt(2 / in_dim)
        elif init == "LeCun":
            std = np.sqrt(1 / in_dim)
        else:
            raise NotImplementedError(">>>>> weight_initializer")
        weight = tf.Variable(
            tf.random.truncated_normal(
                shape=[in_dim, out_dim], mean=0., stddev=std, dtype=self.d_type
            ), 
            dtype=self.d_type, 
            name="w"+str(depth))
        return weight

    def bias_initializer(self, init, shape, depth):
        in_dim  = shape[0]
        out_dim = shape[1]
        if init == "zeros":
            bias = tf.Variable(
                tf.zeros(shape=[in_dim, out_dim], dtype=self.d_type), 
                dtype=self.d_type, 
                name="b"+str(depth)
                )
        elif init == "ones":
            bias = tf.Variable(
                tf.ones(shape=[in_dim, out_dim], dtype=self.d_type), \
                dtype=self.d_type, 
                name="b"+str(depth)
                )
        else:
            raise NotImplementedError(">>>>> bias_initializer")
        return bias

    def get_optimizer(self, optim, lr):
        if optim == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optim == "Adamax":
            optimizer = tf.keras.optimizers.Adamax(learning_rate=lr)
        elif optim == "Nadam":
            optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
        else:
            raise NotImplementedError(">>>>> get_optimizer")
        return optimizer

################################################################################

    def forward_pass(self, x):
        # feature scaling
        if self.f_scale == None or self.f_scale == "linear":
            z = tf.identity(x)
        elif self.f_scale == "minmax":
            a, b = -1., 1   # lower and upper bounds
            z = (b - a) * (x - self.f_in_lb) / (self.f_in_ub - self.f_in_lb) + a
        elif self.f_scale == "std":
            z = (x - self.f_in_mean) / self.f_in_std
        else:
            raise NotImplementedError(">>>>> forward_pass (f_scale)")

        # forward pass
        for l in range(0, self.depth - 1):
            w = self._weights[l]
            b = self._biases [l]
            a = self._alphas [l]
            u = tf.math.add(tf.linalg.matmul(z, w), b)
            u = tf.multiply(a, u)
            if self.act == "tanh":
                z = tf.nn.tanh(u)
            elif self.act == "softplus":
                z = tf.nn.softplus(u)
            elif self.act == "relu":
                z = tf.nn.relu(u)
            elif self.act == "repu":   # rectified power unit (choose power)
                z = tf.nn.relu(u)**3
            elif self.act == "silu":
                z = tf.nn.silu(u)
                # z = tf.multiply(u, tf.math.sigmoid(u))   # this also works fine
            elif self.act == "gelu":
                z = tf.nn.gelu(u)
                # z = tf.multiply(u, tf.math.sigmoid(1.702 * u))   # approximation
            elif self.act == "mish":
                z = tf.multiply(u, tf.nn.tanh(tf.nn.softplus(u)))
            elif self.act == "gauss":
                z = tf.math.exp(- u**2)
            else:
                raise NotImplementedError(">>>>> forward_pass (act)")
        w = self._weights[-1]
        b = self._biases [-1]
        u = tf.math.add(tf.linalg.matmul(z, w), b)
        z = tf.identity(u)
        y = tf.identity(z)
        return y

################################################################################

    def compute_pde(self, x, y):
        """
        soft BC imposition
        """
        with tf.GradientTape(persistent=True) as tp1:
            tp1.watch([x, y])
            with tf.GradientTape(persistent=True) as tp2:
                tp2.watch([x, y])
                uvp = self.forward_pass(tf.concat([x, y], axis=1))
                u = uvp[:, 0:1]
                v = uvp[:, 1:2]
                p = uvp[:, 2:3]
            u_x = tp2.gradient(u, x)
            u_y = tp2.gradient(u, y)
            v_x = tp2.gradient(v, x)
            v_y = tp2.gradient(v, y)
            p_x = tp2.gradient(p, x)
            p_y = tp2.gradient(p, y)
            del tp2
        u_xx = tp1.gradient(u_x, x)
        u_yy = tp1.gradient(u_y, y)
        v_xx = tp1.gradient(v_x, x)
        v_yy = tp1.gradient(v_y, y)
        del tp1

        # operators
        div = u_x + v_y
        adv_u = u * u_x + v * u_y
        adv_v = u * v_x + v * v_y
        lap_u = u_xx + u_yy
        lap_v = v_xx + v_yy

        # positivity enforcement
        Re_tilde = tf.math.exp(self.Re)

        # residual
        r0 = div
        r1 = adv_u + p_x - 1. / Re_tilde * lap_u
        r2 = adv_v + p_y - 1. / Re_tilde * lap_v

        return u, v, p, r0, r1, r2


    # def compute_pde(self, x, y):
    #     with tf.GradientTape(persistent=True) as tp1:
    #         tp1.watch([x, y])
    #         with tf.GradientTape(persistent=True) as tp2:
    #             tp2.watch([x, y])
    #             # 3rd tape
    #             with tf.GradientTape(persistent=True) as tp3:
    #                 tp3.watch(x)
    #                 tp3.watch(y)
    #                 u = self.forward_pass(tf.concat([x, y], axis=1))
    #             u_x = tp3.gradient(u, x)
    #             u_y = tp3.gradient(u, y)
    #             del tp3
    #             # phi: approx distance function (Rvachev+2001, Sukumar+2022)
    #             eps = 1e-4
    #             phi_nth = self.adf_line_segment(x, y, P1=(0.+eps, 1.+eps), P2=(1.+eps, 1.+eps))
    #             phi_sth = self.adf_line_segment(x, y, P1=(0.+eps, 0.+eps), P2=(1.+eps, 0.+eps))
    #             phi_est = self.adf_line_segment(x, y, P1=(1.+eps, 0.+eps), P2=(1.+eps, 1.+eps))
    #             phi_wst = self.adf_line_segment(x, y, P1=(0.+eps, 0.+eps), P2=(0.+eps, 1.+eps))
    #             # normalization parameter for merging ADFs
    #             m = 2
    #             # phi = self.merge_multiple_adfs(
    #             #     phi_list=[phi_nth, phi_sth, phi_est, phi_wst],
    #             #     m=m
    #             # )
    #             # phi_Dir: approx distance function to Dirichlet Boundary
    #             phi_Dir = self.merge_multiple_adfs(
    #                 phi_list=[phi_nth, phi_est, phi_wst],
    #                 m=m
    #             )
    #             # phi_N: approx distance function to Neumann Boundary
    #             phi_Neu = phi_sth

    #             # check if phi_Dir and phi_Neu include nan values
    #             nan_phi_Dir = tf.math.count_nonzero(tf.math.is_nan(phi_Dir))
    #             nan_phi_Neu = tf.math.count_nonzero(tf.math.is_nan(phi_Neu))
    #             # print(f">>> phi_Dir: include nan values? {tf.math.is_nan(phi_Dir)}, nan_phi_Dir: {nan_phi_Dir}")
    #             print(f">>> nan in phi_Dir: {nan_phi_Dir} | elements in phi_Dir: {phi_Dir.shape[0]}")
    #             print(f">>> nan in phi_Neu: {nan_phi_Neu} | elements in phi_Neu: {phi_Neu.shape[0]}")

    #             # g_Dir: transfinite interpolation of Dirichlet BC (Rvachev+2001, Sukumar+2022)
    #             g_Dir = self.transfinite_interpolation(
    #                 x, y,
    #                 psi_list=[tf.sin(np.pi * x), 0., 0.],
    #                 phi_list=[phi_nth, phi_est, phi_wst],
    #                 mu_list=[1., 1., 1.],
    #             )
    #             # solution structure for Dirichlet BC
    #             u_Dir = g_Dir

    #             nan_g_Dir = tf.math.count_nonzero(tf.math.is_nan(g_Dir))
    #             print(f">>> nan in g_Dir: {nan_g_Dir} | elements in g_Dir: {g_Dir.shape[0]}")

    #             # g_Neu: Neumann BC (transfinite interpolation not needed)
    #             g_Neu = 0.   # homogeneous Neumann BC

    #             # gradint of u
    #             # u_x = tp2.gradient(u, x)
    #             # u_y = tp2.gradient(u, y)
    #             grad_u = tf.concat([u_x, u_y], axis=1)
    #             nan_grad_u = tf.math.count_nonzero(tf.math.is_nan(grad_u))
    #             print(f">>> nan in grad_u: {nan_grad_u} | elements in grad_u: {grad_u.shape[0], grad_u.shape[1]}")

    #             # gradint of phi_Neu
    #             phi_Neu_x = tp2.gradient(phi_Neu, x)
    #             phi_Neu_y = tp2.gradient(phi_Neu, y)
    #             grad_phi_Neu = tf.concat([phi_Neu_x, phi_Neu_y], axis=1)
    #             nan_grad_phi_Neu = tf.math.count_nonzero(tf.math.is_nan(grad_phi_Neu))
    #             print(f">>> nan in grad_phi_Neu: {nan_grad_phi_Neu} | elements in grad_phi_Neu: {grad_phi_Neu.shape[0], grad_phi_Neu.shape[1]}")
    #             print(f">>> x: {tf.squeeze(x)}, y: {tf.squeeze(y)}, grad_phi_Neu: {grad_phi_Neu}")

    #             # count nan values
    #             nan_phi_Neu_x = tf.math.count_nonzero(tf.math.is_nan(phi_Neu_x))
    #             nan_phi_Neu_y = tf.math.count_nonzero(tf.math.is_nan(phi_Neu_y))
    #             print(f">>> nan in phi_Neu_x: {nan_phi_Neu_x} | elements in phi_Neu_x: {phi_Neu_x.shape[0]}")
    #             print(f">>> nan in phi_Neu_y: {nan_phi_Neu_y} | elements in phi_Neu_y: {phi_Neu_y.shape[0]}")

    #             # solution structure for Neumann BC

    #             # inner product of grad_u and grad_phi_Neu
    #             dot = tf.multiply(grad_u, grad_phi_Neu)
    #             nan_grad_u = tf.math.count_nonzero(tf.math.is_nan(grad_u))
    #             nan_grad_phi_Neu = tf.math.count_nonzero(tf.math.is_nan(grad_phi_Neu))
    #             print(f">>> nan in grad_u: {nan_grad_u} | elements in grad_u: {grad_u.shape[0], grad_u.shape[1]}")
    #             print(f">>> nan in grad_phi_Neu: {nan_grad_phi_Neu} | elements in grad_phi_Neu: {grad_phi_Neu.shape[0], grad_phi_Neu.shape[1]}")
    #             nan_dot = tf.math.count_nonzero(tf.math.is_nan(dot))
    #             print(f">>> nan in dot: {nan_dot} | elements in dot: {dot.shape[0], dot.shape[1]}")
    #             u_n = tf.reduce_sum(tf.multiply(grad_u, grad_phi_Neu), axis=1, keepdims=True)

    #             # u_n = - tf.add(tf.multiply(u_x, phi_Neu_x), tf.multiply(u_y, phi_Neu_y))
    #             nan_u_n = tf.math.count_nonzero(tf.math.is_nan(u_n))
    #             print(f">>> nan in u_n: {nan_u_n} | elements in u_n: {u_n.shape[0]}")

    #             # # u_n is zero everywhere except on the Neumann boundary (phi_Neu = 0)
    #             # u_n = tf.zeros_like(u)
    #             # # get index of Neumann boundary
    #             # idx_Neu = tf.squeeze(tf.where(tf.equal(phi_Neu, 0.)))
    #             # # set u_n to the correct value on the Neumann boundary (i.e. u_n = u_x * phi_Neu_x + u_y * phi_Neu_y)
    #             # u_n = tf.tensor_scatter_nd_update(
    #             #     u_n,
    #             #     indices=idx_Neu,
    #             #     updates=tf.gather_nd(grad_u, indices=idx_Neu) * tf.gather_nd(grad_phi_Neu, indices=idx_Neu)
    #             # )
    #             u_Neu = u + phi_Neu * u_n - phi_Neu * g_Neu
    #             # nan values in u_Neu?
    #             nan_u_Neu = tf.math.count_nonzero(tf.math.is_nan(u_Neu))
    #             print(f">>> nan in u_Neu: {nan_u_Neu} | elements in u_Neu: {u_Neu.shape[0]}")

    #             # solution structure for Dirichlet and Neumann BC
    #             mu_Dir, mu_Neu = 1, 2
    #             w_Dir = phi_Dir**(-mu_Dir) / (phi_Dir**(-mu_Dir) + phi_Neu**(-mu_Neu))
    #             w_Neu = phi_Neu**(-mu_Neu) / (phi_Dir**(-mu_Dir) + phi_Neu**(-mu_Neu))
    #             # nan values in w_Dir and w_Neu?
    #             nan_w_Dir = tf.math.count_nonzero(tf.math.is_nan(w_Dir))
    #             nan_w_Neu = tf.math.count_nonzero(tf.math.is_nan(w_Neu))
    #             print(f">>> nan in w_Dir: {nan_w_Dir} | elements in w_Dir: {w_Dir.shape[0]}")
    #             print(f">>> nan in w_Neu: {nan_w_Neu} | elements in w_Neu: {w_Neu.shape[0]}")

    #             # remainder
    #             remainder = phi_Dir * phi_Neu**2 * u

    #             # solution structure for mixed BC
    #             u = w_Dir * u_Dir + w_Neu * u_Neu + remainder
    #             # # u = psi + u * phi   # for Dirichlet BC
    #             # nan values in u?
    #             nan_u = tf.math.count_nonzero(tf.math.is_nan(u))
    #             print(f">>> nan in u_tilde: {nan_u} | elements in u_tilde: {u.shape[0]}")
    #         u_x = tp2.gradient(u, x)
    #         u_y = tp2.gradient(u, y)
    #         del tp2
    #     u_xx = tp1.gradient(u_x, x)
    #     u_yy = tp1.gradient(u_y, y)
    #     del tp1
    #     div_u = u_x + u_y
    #     lap_u = u_xx + u_yy
    #     f = tf.sin(2. * np.pi * (x + y))
    #     r = lap_u - f
    #     return u, u_x, u_y, u_xx, u_yy, r

    @tf.function
    def loss_pde(self, x, y):
        # *_, r0_, r1_, r2_ = self.compute_pde(x, y)
        # loss = tf.reduce_mean(tf.square(r0_)) \
        #         + tf.reduce_mean(tf.square(r1_)) \
        #         + tf.reduce_mean(tf.square(r2_))
        # return loss
        *_, r0_, r1_, r2_ = self.compute_pde(x, y)
        loss_div = tf.reduce_mean(tf.square(r0_))
        loss_mmt = tf.reduce_mean(tf.square(r1_)) \
                    + tf.reduce_mean(tf.square(r2_))
        return loss_mmt, loss_div

    @tf.function
    def loss_dat(self, x, y, u, v):
        u_, v_, *_ = self.compute_pde(x, y)
        loss = tf.reduce_mean(tf.square(u_ - u)) \
                + tf.reduce_mean(tf.square(v_ - v))
        return loss

    # @tf.function
    # def loss_bc_Dir(self, x, y, u):
    #     u_, *_ = self.compute_pde(x, y)
    #     loss = tf.reduce_mean(tf.square(u_ - u))
    #     return loss

    # @tf.function
    # def loss_bc_Neu_x(self, x, y, u_x):
    #     u_, u_x_, u_y_, u_xx_, u_yy_, r_ = self.compute_pde(x, y)
    #     loss = tf.reduce_mean(tf.square(u_x_ - u_x))
    #     return loss

    # @tf.function
    # def loss_bc_Neu_y(self, x, y, u_y):
    #     u_, u_x_, u_y_, u_xx_, u_yy_, r_ = self.compute_pde(x, y)
    #     loss = tf.reduce_mean(tf.square(u_y_ - u_y))
    #     return loss

    @tf.function
    def loss_sr(self, alphas, weight):
        """
        slope recovery for L-LAAF (Jagtap+2020)
        weight = 1., 10., 20., etc.
        """
        loss = 1. / tf.reduce_mean(tf.exp(alphas))
        loss *= weight
        return loss

    @tf.function
    def loss_ge(self, x, y, weight):
        """
        gradient enhancement (Yu+2022)
        weight \in [1e-5, 1e-2]
        """
        with tf.GradientTape(persistent=True) as tp:
            tp.watch([x, y])
            *_ , r0_, r1_, r2_ = self.compute_pde(x, y)
        r0_x_ = tp.gradient(r0_, x)
        r0_y_ = tp.gradient(r0_, y)
        r1_x_ = tp.gradient(r1_, x)
        r1_y_ = tp.gradient(r1_, y)
        r2_x_ = tp.gradient(r2_, x)
        r2_y_ = tp.gradient(r2_, y)
        del tp
        loss = tf.reduce_mean(tf.square(r0_x_)) \
                + tf.reduce_mean(tf.square(r0_y_)) \
                + tf.reduce_mean(tf.square(r1_x_)) \
                + tf.reduce_mean(tf.square(r1_y_)) \
                + tf.reduce_mean(tf.square(r2_x_)) \
                + tf.reduce_mean(tf.square(r2_y_))
        loss *= weight
        return loss

    @tf.function
    def loss_glb(
        self, 
        x_pde, y_pde, 
        x_nth, y_nth, u_nth, v_nth,
        x_sth, y_sth, u_sth, v_sth,
        x_est, y_est, u_est, v_est,
        x_wst, y_wst, u_wst, v_wst,
        x_dat, y_dat, u_dat, v_dat,
    ):
        # loss_pde = self.loss_pde(x_pde, y_pde)
        loss_mmt, loss_div = self.loss_pde(x_pde, y_pde)
        loss_nth = self.loss_dat(x_nth, y_nth, u_nth, v_nth)
        loss_sth = self.loss_dat(x_sth, y_sth, u_sth, v_sth)
        loss_est = self.loss_dat(x_est, y_est, u_est, v_est)
        loss_wst = self.loss_dat(x_wst, y_wst, u_wst, v_wst)
        loss_bc  = (loss_nth + loss_sth + loss_est + loss_wst) / 4.
        loss_dat = self.loss_dat(x_dat, y_dat, u_dat, v_dat)
        # loss_glb = loss_pde + loss_bc + loss_dat
        loss_glb = loss_mmt + loss_div + loss_bc + loss_dat

        if self.d_norm:
            # loss_glb = loss_pde \
            #             + self.gamma_bc_hat * loss_bc \
            #             + self.gamma_dat_hat * loss_dat
            loss_glb = loss_mmt \
                        + self.gamma_div_hat * loss_div \
                        + self.gamma_bc_hat * loss_bc \
                        + self.gamma_dat_hat * loss_dat
        if self.l_laaf:
            loss_glb += self.loss_sr(self._alphas, weight=1.)
        if self.g_enhc:
            loss_glb += self.loss_ge(x_pde, y_pde, weight=1e-4)
        # return loss_glb, loss_pde, loss_bc, loss_dat
        return loss_glb, loss_mmt, loss_div, loss_bc, loss_dat

    @tf.function
    def train_step(
        self, 
        x_pde, y_pde, 
        x_nth, y_nth, u_nth, v_nth,
        x_sth, y_sth, u_sth, v_sth,
        x_est, y_est, u_est, v_est,
        x_wst, y_wst, u_wst, v_wst,
        x_dat, y_dat, u_dat, v_dat,
    ):
        with tf.GradientTape(persistent=True) as tp:
            # loss_glb, loss_pde, loss_bc, loss_dat = self.loss_glb(
            #     x_pde, y_pde, 
            #     x_nth, y_nth, u_nth, v_nth,
            #     x_sth, y_sth, u_sth, v_sth,
            #     x_est, y_est, u_est, v_est,
            #     x_wst, y_wst, u_wst, v_wst,
            #     x_dat, y_dat, u_dat, v_dat,
            # )
            loss_glb, loss_mmt, loss_div, loss_bc, loss_dat = self.loss_glb(
                x_pde, y_pde, 
                x_nth, y_nth, u_nth, v_nth,
                x_sth, y_sth, u_sth, v_sth,
                x_est, y_est, u_est, v_est,
                x_wst, y_wst, u_wst, v_wst,
                x_dat, y_dat, u_dat, v_dat,
            )
        grad = tp.gradient(loss_glb, self._params)
        del tp
        self.optimizer.apply_gradients(zip(grad, self._params))
        # return loss_glb, loss_pde, loss_bc, loss_dat
        return loss_glb, loss_mmt + loss_div, loss_bc, loss_dat

    @tf.function
    def infer(self, x, y):
        return self.compute_pde(x, y)


################################################################################
# dynamic normalization (or, one of the following)
# reference: Wang+2021 (https://doi.org/10.1137/20M1318043)
#            Maddu+2022 (https://dx.doi.org/10.1088/2632-2153/ac3712)
#            Deguchi+2023
################################################################################

    @tf.function
    def update_gamma(
        self,
        epoch, tau, beta,
        x_pde, y_pde, 
        x_nth, y_nth, u_nth, v_nth,
        x_sth, y_sth, u_sth, v_sth,
        x_est, y_est, u_est, v_est,
        x_wst, y_wst, u_wst, v_wst,
        x_dat, y_dat, u_dat, v_dat,
    ):
        """
        Update gamma's (weights of loss components) using gradient norms
        """

        # _grad_glb = tf.zeros(shape=(0))
        # _grad_pde = tf.zeros(shape=(0))
        # _grad_bc  = tf.zeros(shape=(0))
        # _grad_dat = tf.zeros(shape=(0))
        _grad_glb = tf.zeros(shape=(0))
        _grad_mmt = tf.zeros(shape=(0))
        _grad_div = tf.zeros(shape=(0))
        _grad_bc  = tf.zeros(shape=(0))
        _grad_dat = tf.zeros(shape=(0))
        for l in range(self.depth):
            with tf.GradientTape(persistent=True) as tp:
                # loss_glb, loss_pde, loss_bc, loss_dat = self.loss_glb(
                #     x_pde, y_pde, 
                #     x_nth, y_nth, u_nth, v_nth,
                #     x_sth, y_sth, u_sth, v_sth,
                #     x_est, y_est, u_est, v_est,
                #     x_wst, y_wst, u_wst, v_wst,
                #     x_dat, y_dat, u_dat, v_dat,
                # )
                loss_glb, loss_mmt, loss_div, loss_bc, loss_dat = self.loss_glb(
                    x_pde, y_pde, 
                    x_nth, y_nth, u_nth, v_nth,
                    x_sth, y_sth, u_sth, v_sth,
                    x_est, y_est, u_est, v_est,
                    x_wst, y_wst, u_wst, v_wst,
                    x_dat, y_dat, u_dat, v_dat,
                )

            # grad to weights
            grad_glb = tp.gradient(loss_glb, self._weights[l])
            grad_mmt = tp.gradient(loss_mmt, self._weights[l])
            grad_div = tp.gradient(loss_div, self._weights[l])
            grad_bc  = tp.gradient(loss_bc,  self._weights[l])
            grad_dat = tp.gradient(loss_dat, self._weights[l])

            grad_glb = tf.cast(grad_glb, dtype=tf.float32)
            grad_mmt = tf.cast(grad_mmt, dtype=tf.float32)
            grad_div = tf.cast(grad_div, dtype=tf.float32)
            grad_bc  = tf.cast(grad_bc,  dtype=tf.float32)
            grad_dat = tf.cast(grad_dat, dtype=tf.float32)

            _grad_glb = tf.concat([_grad_glb, tf.reshape(grad_glb, [-1])], axis=0)
            _grad_mmt = tf.concat([_grad_mmt, tf.reshape(grad_mmt, [-1])], axis=0)
            _grad_div = tf.concat([_grad_div, tf.reshape(grad_div, [-1])], axis=0)
            _grad_bc  = tf.concat([_grad_bc,  tf.reshape(grad_bc,  [-1])], axis=0)
            _grad_dat = tf.concat([_grad_dat, tf.reshape(grad_dat, [-1])], axis=0)

            # grad to biases (last bias may not be tracked due to the problem setup)
            try:
                grad_glb = tp.gradient(loss_glb, self._biases[l])
                grad_mmt = tp.gradient(loss_mmt, self._biases[l])
                grad_div = tp.gradient(loss_div, self._biases[l])
                grad_bc  = tp.gradient(loss_bc,  self._biases[l])
                grad_dat = tp.gradient(loss_dat, self._biases[l])

                grad_glb = tf.cast(grad_glb, dtype=tf.float32)
                grad_mmt = tf.cast(grad_mmt, dtype=tf.float32)
                grad_div = tf.cast(grad_div, dtype=tf.float32)
                grad_bc  = tf.cast(grad_bc,  dtype=tf.float32)
                grad_dat = tf.cast(grad_dat, dtype=tf.float32)

                _grad_glb = tf.concat([_grad_glb, tf.reshape(grad_glb, [-1])], axis=0)
                _grad_mmt = tf.concat([_grad_mmt, tf.reshape(grad_mmt, [-1])], axis=0)
                _grad_div = tf.concat([_grad_div, tf.reshape(grad_div, [-1])], axis=0)
                _grad_bc  = tf.concat([_grad_bc,  tf.reshape(grad_bc,  [-1])], axis=0)
                _grad_dat = tf.concat([_grad_dat, tf.reshape(grad_dat, [-1])], axis=0)
            except:
                pass
            del tp

        # Lp norm (p = 1, 2, ..., np.inf)
        p = 2
        _norm_glb = tf.norm(_grad_glb, ord=p)
        _norm_mmt = tf.norm(_grad_mmt, ord=p)
        _norm_div = tf.norm(_grad_div, ord=p)
        _norm_bc  = tf.norm(_grad_bc,  ord=p)
        _norm_dat = tf.norm(_grad_dat, ord=p)

        # compute gamma
        gamma_bc  = _norm_mmt / _norm_bc
        gamma_dat = _norm_mmt / _norm_dat
        gamma_div = _norm_mmt / _norm_div

        # exponential decay
        gamma_bc = tf.cast(gamma_bc, dtype=tf.float64)
        gamma_bc = beta * self.gamma_bc + (1. - beta) * gamma_bc
        gamma_dat = tf.cast(gamma_dat, dtype=tf.float64)
        gamma_dat = beta * self.gamma_dat + (1. - beta) * gamma_dat
        gamma_div = tf.cast(gamma_div, dtype=tf.float64)
        gamma_div = beta * self.gamma_div + (1. - beta) * gamma_div

        # .assign() to update
        self.gamma_bc.assign(gamma_bc)
        self.gamma_dat.assign(gamma_dat)
        self.gamma_div.assign(gamma_div)

        # bias correction
        t = epoch / tau + 1.
        gamma_bc_hat  = gamma_bc  / (1. - beta**t)
        gamma_dat_hat = gamma_dat / (1. - beta**t)
        gamma_div_hat = gamma_div / (1. - beta**t)

        # .assign() to update
        self.gamma_bc_hat.assign(gamma_bc_hat)
        self.gamma_dat_hat.assign(gamma_dat_hat)
        self.gamma_div_hat.assign(gamma_div_hat)

        return



################################################################################
# Hessian computation
# reference: Wang+2021 (https://doi.org/10.1137/20M1318043)
#            Nilsen+2021 (https://doi.org/10.48550/arxiv.1905.05559)
#            Pealmutter1994 (https://doi.org/10.1162/neco.1994.6.1.147)
################################################################################

    def feed(
        self, 
        x_pde, y_pde, 
        x_bc,  y_bc,  u_bc
    ):
        # pde residual evaluation
        self.x_pde = x_pde
        self.y_pde = y_pde

        # bc evaluation
        self.x_bc = x_bc
        self.y_bc = y_bc
        self.u_bc = u_bc

    def flatten(self, vs):
        """
        flattens a list of tensors (vs = [v1, v2, ..., vN]) into a 1D tensor
        """
        return tf.concat([tf.reshape(v, [-1]) for v in vs], axis=0)

    @tf.function
    def compute_H(self):
        """
        computes the Hessian, applies `Hv_op`'s to `v` via tf.map_fn
        """

        # # of parameters (or # of weights)
        P = self._n_params
        print(f">>> computing the Hessian of size {P}")

        # map
        H_glb = tf.map_fn(self.Hv_op_glb, tf.eye(P, P, dtype=self.d_type), fn_output_signature=self.d_type)
        H_pde = tf.map_fn(self.Hv_op_pde, tf.eye(P, P, dtype=self.d_type), fn_output_signature=self.d_type)
        H_bc  = tf.map_fn(self.Hv_op_bc,  tf.eye(P, P, dtype=self.d_type), fn_output_signature=self.d_type)
        return H_glb, H_pde, H_bc

    @tf.function
    def Hv_op_glb(self, v):
        """
        Hessian-vector product operation, used via tf.map_fn
        """
        with tf.GradientTape(persistent=True) as tp2:
            with tf.GradientTape(persistent=True) as tp1:
                loss_glb, loss_pde, loss_bc = self.loss_glb(
                    self.x_pde, self.y_pde, 
                    self.x_bc,  self.y_bc,  self.u_bc
                )

            # 1st order derivatives
            grad_glb = tp1.gradient(loss_glb, self._params)
            del tp1

            # flatten
            grad_glb = self.flatten(grad_glb)

            # product with v
            vprod_glb = tf.multiply(grad_glb, tf.stop_gradient(v))

        # 2nd order derivatives
        Hv_glb = tp2.gradient(vprod_glb, self._params)
        del tp2

        # flatten
        Hv_glb = self.flatten(Hv_glb)

        return Hv_glb

    @tf.function
    def Hv_op_pde(self, v):
        with tf.GradientTape(persistent=True) as tp2:
            with tf.GradientTape(persistent=True) as tp1:
                loss_glb, loss_pde, loss_bc = self.loss_glb(
                    self.x_pde, self.y_pde, 
                    self.x_bc,  self.y_bc,  self.u_bc
                )

            # 1st order derivatives
            grad_pde = tp1.gradient(loss_pde, self._params)
            del tp1

            # flatten
            grad_pde = self.flatten(grad_pde)

            # product with v
            vprod_pde = tf.multiply(grad_pde, tf.stop_gradient(v))

        # 2nd order derivatives
        Hv_pde = tp2.gradient(vprod_pde, self._params)
        del tp2

        # flatten
        Hv_pde = self.flatten(Hv_pde)

        return Hv_pde

    @tf.function
    def Hv_op_bc(self, v):
        with tf.GradientTape(persistent=True) as tp2:
            with tf.GradientTape(persistent=True) as tp1:
                loss_glb, loss_pde, loss_bc = self.loss_glb(
                    self.x_pde, self.y_pde, 
                    self.x_bc,  self.y_bc,  self.u_bc
                )

            # 1st order derivatives
            grad_bc = tp1.gradient(loss_bc, self._params)
            del tp1

            # flatten
            grad_bc = self.flatten(grad_bc)

            # product with v
            vprod_bc = tf.multiply(grad_bc, tf.stop_gradient(v))

        # 2nd order derivatives
        Hv_bc = tp2.gradient(vprod_bc, self._params)
        del tp2

        # flatten
        Hv_bc = self.flatten(Hv_bc)

        return Hv_bc

    @tf.function
    def compute_max_lr(self, track_all=False):
        """
        maximum learning rate bounded by the eigenvalus of the Hessian
        """

        with tf.GradientTape(persistent=True) as tp:
            loss_glb, loss_pde, loss_bc = self.loss_glb(
                self.x_pde, self.y_pde, 
                self.x_bc,  self.y_bc,  self.u_bc
            )

        # gradient
        grad_glb = tp.gradient(loss_glb, self._params)
        grad_pde = tp.gradient(loss_pde, self._params)
        grad_bc  = tp.gradient(loss_bc,  self._params)
        del tp

        # flatten
        grad_glb = self.flatten(grad_glb)
        grad_pde = self.flatten(grad_pde)
        grad_bc  = self.flatten(grad_bc)

        # normalized gradient (x = grad / ||grad||2)
        _x_glb = grad_glb / tf.linalg.norm(grad_glb, ord=2)
        _x_pde = grad_pde / tf.linalg.norm(grad_pde, ord=2)
        _x_bc  = grad_bc  / tf.linalg.norm(grad_bc,  ord=2)

        # Hessian
        H_glb, H_pde, H_bc = self.compute_H()

        if track_all:
            # eigen decomposition of the Hessian (H = QLQ^{transpose})
            eigval_glb, eigvec_glb = tf.linalg.eigh(H_glb)
            eigval_pde, eigvec_pde = tf.linalg.eigh(H_pde)
            eigval_bc,  eigvec_bc  = tf.linalg.eigh(H_bc)

            # y = Qx
            _y_glb = tf.linalg.matvec(eigvec_glb, _x_glb)
            _y_pde = tf.linalg.matvec(eigvec_pde, _x_pde)
            _y_bc  = tf.linalg.matvec(eigvec_bc,  _x_bc)

            # z = Ly^2
            _z_glb = tf.multiply(eigval_glb, tf.square(_y_glb))
            _z_pde = tf.multiply(eigval_pde, tf.square(_y_pde))
            _z_bc  = tf.multiply(eigval_bc,  tf.square(_y_bc))

            # sum
            _s_glb = tf.reduce_sum(_z_glb)
            _s_pde = tf.reduce_sum(_z_pde)
            _s_bc  = tf.reduce_sum(_z_bc)

            # maximul lr allowed by the Hessian
            _r_glb = 2. * tf.math.reciprocal(_s_glb)
            _r_pde = 2. * tf.math.reciprocal(_s_pde)
            _r_bc  = 2. * tf.math.reciprocal(_s_bc)

            return _r_glb, _r_pde, _r_bc

        else:
            # eigen decomposition of the Hessian (H = QLQ^{transpose})
            eigval_glb, eigvec_glb = tf.linalg.eigh(H_glb)
            # eigval_pde, eigvec_pde = tf.linalg.eigh(H_pde)
            # eigval_bc,  eigvec_bc  = tf.linalg.eigh(H_bc)

            # y = Qx
            _y_glb = tf.linalg.matvec(eigvec_glb, _x_glb)
            # _y_pde = tf.linalg.matvec(eigvec_pde, _x_pde)
            # _y_bc  = tf.linalg.matvec(eigvec_bc,  _x_bc)

            # z = Ly^2
            _z_glb = tf.multiply(eigval_glb, tf.square(_y_glb))
            # _z_pde = tf.multiply(eigval_pde, tf.square(_y_pde))
            # _z_bc  = tf.multiply(eigval_bc,  tf.square(_y_bc))

            # sum
            _s_glb = tf.reduce_sum(_z_glb)
            # _s_pde = tf.reduce_sum(_z_pde)
            # _s_bc  = tf.reduce_sum(_z_bc)

            # maximul lr allowed by the Hessian
            _r_glb = 2. * tf.math.reciprocal(_s_glb)
            # _r_pde = 2. * tf.math.reciprocal(_s_pde)
            # _r_bc  = 2. * tf.math.reciprocal(_s_bc)
            _r_pde = 9999.
            _r_bc  = 9999.

            return _r_glb, _r_pde, _r_bc



################################################################################
# exact imposition of boundary conditions
# reference: Sukumar+2022 (https://doi.org/10.1016/j.cma.2021.114333)
# the following is implemented in utils.py
################################################################################

    @tf.function
    def adf_line_segment(self, x, y, P1, P2):
        """
        approximate distance function to a line segment

        args:
            x, y: coordinates of the point
            P1, P2: coordinates of the end points of the line segment

        returns:
            phi: approximate distance function
        """

        # get positions
        x1, y1 = P1
        x2, y2 = P2

        # length of the line segment
        L = tf.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        L = tf.cast(L, tf.float64)

        # center of the line segment
        xc = (x1 + x2) / 2.
        yc = (y1 + y2) / 2.

        # unit tangential vector
        tx = (x2 - x1) / L
        ty = (y2 - y1) / L

        # unit normal vector
        nx = -ty
        ny =  tx

        # signed distance function
        # implicit representation of the line passing through P1 and P2
        s = 1. / L * ((x - x1) * (y2 - y1) - (y - y1) * (x2 - x1))

        # trimming function
        # implicit representation of the circle centered at (xc, yc) with radius L/2
        t = 1. / L * ((L / 2.)**2 - (x - xc)**2 - (y - yc)**2)

        # approximation of the distance function to the segment with end points P1 and P2
        phi = (s**2 + (((s**4 + t**2)**.5 - t) / 2.)**2)**.5

        return phi

    def merge_two_adfs(self, phi1, phi2, m):
        """
        merge two ADFs

        args:
            phi1, phi2: ADFs
            m: normalization parameter

        return:
            phi: merged ADF
        """

        phi = phi1 * phi2 / (phi1**m + phi2**m)**(1. / m)
        return phi

    def merge_multiple_adfs(self, phi_list, m):
        """
        merge multiple ADFs

        args:
            phi_list: list of ADFs
            m: normalization parameter

        return:
            phi: merged ADF
        """

        phi = phi_list[0]
        for i in range(1, len(phi_list)):
            phi = self.merge_two_adfs(phi, phi_list[i], m)
        return phi

    def transfinite_interpolation(self, x, y, psi_list, phi_list, mu_list):
        """
        transfinite interpolation

        args:
            x, y: coordinates of the point
            psi_list: list of Dirichlet boundary conditions
            phi_list: list of ADFs
            mu_list: list of mu, the normalization parameter

        return:
            psi: interpolated Dirichlet boundary condition
        """

        # psi = sum_i w_i * psi_i, where w_i = phi_i**(-mu_i) / sum_j phi_j**(-mu_j)
        denominator = 0.
        for i in range(len(psi_list)):
            denominator += phi_list[i]**(-mu_list[i])

        psi = 0.
        for i in range(len(psi_list)):
            numerator = phi_list[i]**(-mu_list[i])
            w_i = numerator / denominator
            psi_i = psi_list[i]
            psi += w_i * psi_i

        return psi

    def dist(self, x1, y1, x2, y2):
        return tf.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def linseg(self, x, y, x1, y1, x2, y2):
        L = self.dist(x1, y1, x2, y2)
        xc = (x1 + x2) / 2.
        yc = (y1 + y2) / 2.

        # signed distance function from x to the line that passes through x1 and x2
        f = 1. / L * ((x - x1) * (y2 - y1) - (y - y1) * (x2 - x1))

        # trimming
        t = 1. / L * ((L / 2.)**2 - self.dist(x, y, xc, yc)**2)

        # approximation of the distance function to the segment with end points x1 and x2
        vphi = tf.sqrt(t**2 + f**4)
        phi = tf.sqrt(f**2 + 1. / 4. * (vphi - t)**2)
        return f, t, vphi, phi

    def segments(self):
        # this is a little more readable, but need to remove `tf.function`
        # segments = tf.constant(
        #     [
        #         [self.lb[0].numpy(), self.lb[1].numpy(), self.lb[0].numpy(), self.ub[1].numpy()],
        #         [self.lb[0].numpy(), self.ub[1].numpy(), self.ub[0].numpy(), self.ub[1].numpy()],
        #         [self.ub[0].numpy(), self.ub[1].numpy(), self.ub[0].numpy(), self.lb[1].numpy()],
        #         [self.ub[0].numpy(), self.lb[1].numpy(), self.lb[0].numpy(), self.lb[1].numpy()]
        #     ], 
        #     dtype=tf.float64
        # )

        # this is faster, but not very readable and need careful check
        segments = tf.constant(
            [
                [0., 0., 0., 1.],
                [0., 1., 1., 1.],
                [1., 1., 1., 0.],
                [1., 0., 0., 0.]
            ], 
            dtype=tf.float64
        )

        return segments

    @tf.function
    def phi(self, x, y, m=1):
        """
        approximate distance function to the boundary

        args:
            m: normalization parameter (Sukumar+2022: m=1, comput mech community: m=2)
                higher m gives a better approximation of the exact distance
                also strong normalization properties in the vicinity
                however, higher order functions become oscillatory
        """

        segments = self.segments()

        R = 0.
        for i in range(len(segments[:,0])):
            f, t, vphi, phi = self.linseg(
                x, y, 
                segments[i, 0], segments[i, 1], segments[i, 2], segments[i, 3]
            )
            R += 1. / phi**m
        R = 1. / R**(1 / m)
        return R



    def Dirichlet_BC(self, x, y):
        # prescribed Dirichlet boundary condition
        numerator = (tf.exp(-np.pi * y) + tf.exp(np.pi * y)) * tf.sin(np.pi * x)
        denominator = tf.exp(-np.pi) + tf.exp(np.pi)
        u = tf.cast(numerator, dtype=tf.float64) / tf.cast(denominator, dtype=tf.float64)
        return u


    # @tf.function
    # def psi(self, x, y, m=1):
    #     """
    #     approximate distance function to the boundary

    #     args:
    #         m: normalization parameter (Sukumar+2022: m=1, comput mech community: m=2)
    #             higher m gives a better approximation of the exact distance
    #             also strong normalization properties in the vicinity
    #             however, higher order functions become oscillatory
    #     """

    #     segments = self.segments()

    #     R = 0.
    #     for i in range(len(segments[:,0])):
    #         f, t, vphi, phi = self.linseg(
    #             x, y, 
    #             segments[i, 0], segments[i, 1], segments[i, 2], segments[i, 3]
    #         )
    #         R += 1. / phi**m
    #     R = 1. / R**(1 / m)
    #     return R

    @tf.function
    def psi(self, x, y, m=1):
        """
        extension of homogeneous / inhomogeneous Dirichlet boundary condition

        args:
            m: regularity parameter (denoted as mu in Sukumar+2022)
                m = 1: Dirichlet condition g is interpolated
                m = 2: Dirichlet condition g and its normal derivative dg/dn are interpolated
        """

        segments = self.segments()

        # denominator of weights
        W = 0.
        for i in range(len(segments[:,0])):
            f, t, vphi, phi = self.linseg(
                x, y, 
                segments[i, 0], segments[i, 1], segments[i, 2], segments[i, 3]
            )
            W += phi**(-m)

        # extension of the Dirichlet boundary condition
        S = 0.
        for i in range(len(segments[:,0])):
            f, t, vphi, phi = self.linseg(
                x, y, 
                segments[i, 0], segments[i, 1], segments[i, 2], segments[i, 3]
            )
            w = phi**(-m) / W
            s = self.Dirichlet_BC(x, y)
            # s = self.Dirichlet_BC(x - segments[i, 0], y - segments[i, 1])

            S += w * s
        return S
