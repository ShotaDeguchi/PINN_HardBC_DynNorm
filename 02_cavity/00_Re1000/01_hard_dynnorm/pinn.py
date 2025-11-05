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
        hard BC imposition
        """
        with tf.GradientTape(persistent=True) as tp1:
            tp1.watch([x, y])
            with tf.GradientTape(persistent=True) as tp2:
                tp2.watch([x, y])
                with tf.GradientTape(persistent=True) as tp3:
                    tp3.watch([x, y])
                    uvp = self.forward_pass(tf.concat([x, y], axis=1))
                    u = uvp[:, 0:1]
                    v = uvp[:, 1:2]
                    p = uvp[:, 2:3]

                    # ADFs
                    eps = 1e-8
                    phi_nth = rvachev.adf_line_segment(x, y, P1=(0.-eps, 1.+eps), P2=(1.+eps, 1.+eps), dtype=self.d_type)
                    phi_sth = rvachev.adf_line_segment(x, y, P1=(0.-eps, 0.-eps), P2=(1.+eps, 0.-eps), dtype=self.d_type)
                    phi_est = rvachev.adf_line_segment(x, y, P1=(1.+eps, 0.-eps), P2=(1.+eps, 1.+eps), dtype=self.d_type)
                    phi_wst = rvachev.adf_line_segment(x, y, P1=(0.-eps, 0.-eps), P2=(0.-eps, 1.+eps), dtype=self.d_type)

                    # ADF to Gamma (= Gamma_Dir \cup Gamma_Neu)
                    m = 1   # normalization parameter
                    # phi = rvachev.join_multiple_adfs(
                    #     phis=[phi_nth, phi_sth, phi_est, phi_wst],
                    #     m=m
                    # )

                    # ADFs
                    phi_Dir_u = rvachev.join_multiple_adfs(
                        phis=[phi_nth, phi_sth, phi_est, phi_wst],
                        m=m
                    )
                    phi_Dir_v = rvachev.join_multiple_adfs(
                        phis=[phi_nth, phi_sth, phi_est, phi_wst],
                        m=m
                    )
                    phi_Neu_p = rvachev.join_multiple_adfs(
                        phis=[phi_nth, phi_sth, phi_est, phi_wst],
                        m=m
                    )

                # gradint of p
                p_x = tp3.gradient(p, x)
                p_y = tp3.gradient(p, y)

                # gradint of phi_Neu_p
                phi_Neu_p_x = tp3.gradient(phi_Neu_p, x)
                phi_Neu_p_y = tp3.gradient(phi_Neu_p, y)
                del tp3

                # Dirichlet BC
                g_Dir_u = rvachev.transfinite_interpolation(
                    gs=[1., 0., 0., 0.],
                    phis=[phi_nth, phi_sth, phi_est, phi_wst],
                    mus=[1, 1, 1, 1],
                    method=2
                )
                g_Dir_v = rvachev.transfinite_interpolation(
                    gs=[0., 0., 0., 0.],
                    phis=[phi_nth, phi_sth, phi_est, phi_wst],
                    mus=[1, 1, 1, 1],
                    method=2
                )
                g_Neu_p = 0.   # homogeneous Neumann BC

                # solutions
                u = g_Dir_u + phi_Dir_u * u
                v = g_Dir_v + phi_Dir_v * v
                # no change in pressure so its more silimar to coupled formulation
                # p = p - phi_Neu_p * (p_x * phi_Neu_p_x + p_y * phi_Neu_p_y) - phi_Neu_p * g_Neu_p + phi_Neu_p**2 * p

            u_x = tp2.gradient(u, x); u_y = tp2.gradient(u, y)
            v_x = tp2.gradient(v, x); v_y = tp2.gradient(v, y)
            p_x = tp2.gradient(p, x); p_y = tp2.gradient(p, y)
            del tp2
        u_xx = tp1.gradient(u_x, x); u_yy = tp1.gradient(u_y, y)
        v_xx = tp1.gradient(v_x, x); v_yy = tp1.gradient(v_y, y)
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

    def loss_pde(self, x, y):
        *_, r0_, r1_, r2_ = self.compute_pde(x, y)
        loss_div = tf.reduce_mean(tf.square(r0_))
        loss_mmt = tf.reduce_mean(tf.square(r1_)) \
                    + tf.reduce_mean(tf.square(r2_))
        return loss_mmt, loss_div

    def loss_dat(self, x, y, u, v):
        u_, v_, *_ = self.compute_pde(x, y)
        loss = tf.reduce_mean(tf.square(u_ - u)) \
                + tf.reduce_mean(tf.square(v_ - v))
        return loss

    # def loss_bc_Dir(self, x, y, u):
    #     u_, *_ = self.compute_pde(x, y)
    #     loss = tf.reduce_mean(tf.square(u_ - u))
    #     return loss

    # def loss_bc_Neu_x(self, x, y, u_x):
    #     u_, u_x_, u_y_, u_xx_, u_yy_, r_ = self.compute_pde(x, y)
    #     loss = tf.reduce_mean(tf.square(u_x_ - u_x))
    #     return loss

    # def loss_bc_Neu_y(self, x, y, u_y):
    #     u_, u_x_, u_y_, u_xx_, u_yy_, r_ = self.compute_pde(x, y)
    #     loss = tf.reduce_mean(tf.square(u_y_ - u_y))
    #     return loss

    def loss_sr(self, alphas, weight):
        """
        slope recovery for L-LAAF (Jagtap+2020)
        weight = 1., 10., 20., etc.
        """
        loss = 1. / tf.reduce_mean(tf.exp(alphas))
        loss *= weight
        return loss

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
        loss_bc  = (loss_nth + loss_sth + loss_est + loss_wst) / 4. * 0.
        loss_dat = self.loss_dat(x_dat, y_dat, u_dat, v_dat)
        loss_glb = loss_mmt + loss_div + loss_bc + loss_dat

        if self.d_norm:
            loss_glb = loss_mmt \
                        + self.gamma_div_hat * loss_div \
                        + self.gamma_bc_hat * loss_bc \
                        + self.gamma_dat_hat * loss_dat
        if self.l_laaf:
            loss_glb += self.loss_sr(self._alphas, weight=1.)
        if self.g_enhc:
            loss_glb += self.loss_ge(x_pde, y_pde, weight=1e-4)
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
# bias-corrected dynamic normalization
# reference: Deguchi+2023 (https://doi.org/10.1088/2399-6528/ace416)
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

        _grad_glb = tf.zeros(shape=(0))
        _grad_mmt = tf.zeros(shape=(0))
        _grad_div = tf.zeros(shape=(0))
        _grad_bc  = tf.zeros(shape=(0))
        _grad_dat = tf.zeros(shape=(0))
        for l in range(self.depth):
            with tf.GradientTape(persistent=True) as tp:
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
        # _norm_bc  = tf.norm(_grad_bc,  ord=p)
        _norm_dat = tf.norm(_grad_dat, ord=p)

        # compute gamma
        eps = 1e-8
        # gamma_bc  = _norm_pde / (_norm_bc + eps)
        gamma_dat = _norm_mmt / (_norm_dat + eps)
        gamma_div = _norm_mmt / (_norm_div + eps)

        # exponential decay
        # gamma_bc = tf.cast(gamma_bc, dtype=tf.float64)
        # gamma_bc = beta * self.gamma_bc + (1. - beta) * gamma_bc
        gamma_dat = tf.cast(gamma_dat, dtype=tf.float64)
        gamma_dat = beta * self.gamma_dat + (1. - beta) * gamma_dat
        gamma_div = tf.cast(gamma_div, dtype=tf.float64)
        gamma_div = beta * self.gamma_div + (1. - beta) * gamma_div

        # .assign() to update
        # self.gamma_bc.assign(gamma_bc)
        self.gamma_dat.assign(gamma_dat)
        self.gamma_div.assign(gamma_div)

        # bias correction
        t = epoch / tau + 1.
        # gamma_bc_hat  = gamma_bc  / (1. - beta**t)
        gamma_dat_hat = gamma_dat / (1. - beta**t)
        gamma_div_hat = gamma_div / (1. - beta**t)

        # .assign() to update
        # self.gamma_bc_hat.assign(gamma_bc_hat)
        self.gamma_dat_hat.assign(gamma_dat_hat)
        self.gamma_div_hat.assign(gamma_div_hat)

        return
