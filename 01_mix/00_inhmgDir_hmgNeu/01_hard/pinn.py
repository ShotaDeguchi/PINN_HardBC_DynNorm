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
        self.a0  = a0
        self.a1  = a1
        if self.inv:
            print(">>> inverse analysis, adding trainable variables...")
            self.a0 = tf.Variable(a0, dtype=self.d_type)
            self.a1 = tf.Variable(a1, dtype=self.d_type)
            self._params.append(self.a0)
            self._params.append(self.a1)
        else:
            print(">>> forward analysis, using constant values...")
            self.a0 = tf.constant(a0, dtype=self.d_type)
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
        # unbiased estimator
        self.gamma_bc_hat = tf.Variable(0., dtype=self.d_type)
        self.gamma_dat_hat = tf.Variable(0., dtype=self.d_type)

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

    def feature_scaling(self, x):
        if self.f_scale == None or self.f_scale == "linear":
            z = tf.identity(x)
        elif self.f_scale == "minmax":
            a, b = -1., 1
            z = (b - a) * (x - self.f_in_lb) / (self.f_in_ub - self.f_in_lb) + a
        elif self.f_scale == "std":
            z = (x - self.f_in_mean) / self.f_in_std
        else:
            raise NotImplementedError(">>>>> feature_scaling")
        return z

    def forward_pass(self, x):
        # feature scaling
        z = self.feature_scaling(x)

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

                    # raw mlp output
                    u = self.forward_pass(tf.concat([x, y], axis=1))

                    # ADFs
                    phi_nth = rvachev.adf_line_segment(x, y, P1=(0., 1.), P2=(1., 1.), dtype=self.d_type)
                    phi_sth = rvachev.adf_line_segment(x, y, P1=(0., 0.), P2=(1., 0.), dtype=self.d_type)
                    phi_est = rvachev.adf_line_segment(x, y, P1=(1., 0.), P2=(1., 1.), dtype=self.d_type)
                    phi_wst = rvachev.adf_line_segment(x, y, P1=(0., 0.), P2=(0., 1.), dtype=self.d_type)

                    # ADF to Gamma (= Gamma_Dir \cup Gamma_Neu)
                    m = 1   # normalization parameter
                    phi = rvachev.join_multiple_adfs(
                        phis=[phi_nth, phi_sth, phi_est, phi_wst],
                        m=m
                    )

                    # ADF to Gamma_Dir
                    phi_Dir = rvachev.join_multiple_adfs(
                        phis=[phi_nth, phi_est, phi_wst],
                        m=m
                    )

                    # ADF to Gamma_Neu
                    phi_Neu = phi_sth

                # gradint of u
                u_x = tp3.gradient(u, x)
                u_y = tp3.gradient(u, y)

                # gradint of phi_Neu
                phi_Neu_x = tp3.gradient(phi_Neu, x)
                phi_Neu_y = tp3.gradient(phi_Neu, y)
                del tp3

                # Dirichlet BC
                g_Dir = rvachev.transfinite_interpolation(
                    gs=[tf.sin(np.pi * x), 0., 0.],
                    phis=[phi_nth, phi_est, phi_wst],
                    mus=[1, 1, 1],   # mu = 1 or 2
                    method=2
                )

                # Neumann BC
                g_Neu = 0.   # homogeneous Neumann BC

                # solution structure for Dirichlet and Neumann BC
                u_Dir = g_Dir
                u_Neu = u - phi_Neu * (u_x * phi_Neu_x + u_y * phi_Neu_y) - phi_Neu * g_Neu

                # transfinite interpolation of u_Dir and u_Neu for mixed BC
                mu_Dir = 1   # interpolant is differentiable (mu_i - 1) times at Gamma_i
                mu_Neu = 2
                w_Dir = phi_Neu**mu_Neu / (phi_Dir**mu_Dir + phi_Neu**mu_Neu)
                w_Neu = phi_Dir**mu_Dir / (phi_Dir**mu_Dir + phi_Neu**mu_Neu)
                u = w_Dir * u_Dir \
                    + w_Neu * u_Neu \
                    + phi_Dir**mu_Dir * phi_Neu**mu_Neu * u

            u_x = tp2.gradient(u, x)
            u_y = tp2.gradient(u, y)
            del tp2
        u_xx = tp1.gradient(u_x, x)
        u_yy = tp1.gradient(u_y, y)
        del tp1
        lap_u = u_xx + u_yy
        f = tf.sin(2. * np.pi * (x + y))   # known source term
        # r = - lap_u - f
        r =   lap_u - f
        return u, u_x, u_y, u_xx, u_yy, r
        # return u, u_x, u_y, u_xx, u_yy, phi, phi_Dir, phi_Neu, phi_Neu_x, phi_Neu_y, r

    def loss_pde(self, x, y):
        *_, r_ = self.compute_pde(x, y)
        loss = tf.reduce_mean(tf.square(r_))
        return loss

    def loss_bc_Dir(self, x, y, u):
        u_, *_ = self.compute_pde(x, y)
        loss = tf.reduce_mean(tf.square(u_ - u))
        return loss

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
            *_, r_ = self.compute_pde(x, y)
        r_x_ = tp.gradient(r_, x)
        r_y_ = tp.gradient(r_, y)
        del tp
        loss = tf.reduce_mean(tf.square(r_x_)) \
                + tf.reduce_mean(tf.square(r_y_))
        loss *= weight
        return loss

    def loss_glb(
        self, 
        x_pde, y_pde, 
        x_nth, y_nth, u_nth,
        x_sth, y_sth, u_sth,
        x_est, y_est, u_est,
        x_wst, y_wst, u_wst,
    ):
        loss_pde = self.loss_pde(x_pde, y_pde)
        # loss_nth = self.loss_bc_Dir(x_nth, y_nth, u_nth)
        # loss_sth = self.loss_bc_Dir(x_sth, y_sth, u_sth)
        # loss_est = self.loss_bc_Dir(x_est, y_est, u_est)
        # loss_wst = self.loss_bc_Dir(x_wst, y_wst, u_wst)
        # loss_bc  = (loss_nth + loss_sth + loss_est + loss_wst) / 4.
        loss_bc = 0.
        loss_glb = loss_pde + loss_bc
        if self.d_norm:
            loss_glb = loss_pde + self.gamma_bc_hat * loss_bc
        if self.l_laaf:
            loss_glb += self.loss_sr(self._alphas, weight=1.)
        if self.g_enhc:
            loss_glb += self.loss_ge(x_pde, y_pde, weight=1e-4)
        return loss_glb, loss_pde, loss_bc

    @tf.function
    def train_step(
        self, 
        x_pde, y_pde, 
        x_nth, y_nth, u_nth,
        x_sth, y_sth, u_sth,
        x_est, y_est, u_est,
        x_wst, y_wst, u_wst,
    ):
        with tf.GradientTape(persistent=True) as tp:
            loss_glb, loss_pde, loss_bc = self.loss_glb(
                x_pde, y_pde, 
                x_nth, y_nth, u_nth,
                x_sth, y_sth, u_sth,
                x_est, y_est, u_est,
                x_wst, y_wst, u_wst,
            )
        grad = tp.gradient(loss_glb, self._params)
        # # flatten
        # grad_flat = self.flatten(grad)
        # # # count the number of NaNs in grad
        # nan_grad = tf.math.count_nonzero(tf.math.is_nan(grad_flat))
        # print(f'>>> nan_grad: {nan_grad} | number of parameters: {self._n_params}')
        # # print(f">>> number of parameters: {self._n_params} | nan_grad: {nan_grad}")
        # # print(f">>> grad: {grad}")
        del tp
        self.optimizer.apply_gradients(zip(grad, self._params))
        return loss_glb, loss_pde, loss_bc

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
        x_nth, y_nth, u_nth,
        x_sth, y_sth, u_sth,
        x_est, y_est, u_est,
        x_wst, y_wst, u_wst,
    ):
        """
        Update gamma's (weights of loss components) using gradient norms
        """

        _grad_glb = tf.zeros(shape=(0))
        _grad_pde = tf.zeros(shape=(0))
        _grad_bc  = tf.zeros(shape=(0))
        for l in range(self.depth):
            with tf.GradientTape(persistent=True) as tp:
                loss_glb, loss_pde, loss_bc = self.loss_glb(
                    x_pde, y_pde,
                    x_nth, y_nth, u_nth,
                    x_sth, y_sth, u_sth,
                    x_est, y_est, u_est,
                    x_wst, y_wst, u_wst,
                )

            # grad to weights
            grad_glb = tp.gradient(loss_glb, self._weights[l])
            grad_pde = tp.gradient(loss_pde, self._weights[l])
            grad_bc  = tp.gradient(loss_bc,  self._weights[l])
            grad_glb = tf.cast(grad_glb, dtype=tf.float32)
            grad_pde = tf.cast(grad_pde, dtype=tf.float32)
            grad_bc  = tf.cast(grad_bc,  dtype=tf.float32)
            _grad_glb = tf.concat([_grad_glb, tf.reshape(grad_glb, [-1])], axis=0)
            _grad_pde = tf.concat([_grad_pde, tf.reshape(grad_pde, [-1])], axis=0)
            _grad_bc  = tf.concat([_grad_bc,  tf.reshape(grad_bc,  [-1])], axis=0)

            # grad to biases (last bias may not be tracked due to the problem setup)
            try:
                grad_glb = tp.gradient(loss_glb, self._biases[l])
                grad_pde = tp.gradient(loss_pde, self._biases[l])
                grad_bc  = tp.gradient(loss_bc,  self._biases[l])
                grad_glb = tf.cast(grad_glb, dtype=tf.float32)
                grad_pde = tf.cast(grad_pde, dtype=tf.float32)
                grad_bc  = tf.cast(grad_bc,  dtype=tf.float32)
                _grad_glb = tf.concat([_grad_glb, tf.reshape(grad_glb, [-1])], axis=0)
                _grad_pde = tf.concat([_grad_pde, tf.reshape(grad_pde, [-1])], axis=0)
                _grad_bc  = tf.concat([_grad_bc,  tf.reshape(grad_bc,  [-1])], axis=0)
            except:
                pass
            del tp

        # Lp norm (p = 1, 2, ..., np.inf)
        p = 2
        _norm_glb = tf.norm(_grad_glb, ord=p)
        _norm_pde = tf.norm(_grad_pde, ord=p)
        _norm_bc  = tf.norm(_grad_bc,  ord=p)

        # compute gamma
        gamma_bc = _norm_pde / _norm_bc

        # exponential decay
        gamma_bc = tf.cast(gamma_bc, dtype=tf.float64)
        gamma_bc = beta * self.gamma_bc + (1. - beta) * gamma_bc

        # .assign() to update
        self.gamma_bc.assign(gamma_bc)

        # bias correction
        t = epoch / tau + 1.
        gamma_bc_hat = gamma_bc / (1. - beta**t)

        # .assign() to update
        self.gamma_bc_hat.assign(gamma_bc_hat)

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



