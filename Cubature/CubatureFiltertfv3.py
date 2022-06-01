import numpy as np
from . import cubaturetfv3 as cb
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
tf.random.set_seed(0)

debug = False
class CubatureFilter:

    def __init__(self, f, h, x0, P0, Q0, R0, s_dim, reg_const=0.01):
        """
        class CubatureFilter
        :param f:
        :param h:
        :param x0:
        :param P0:
        :param Q0:
        :param R0:
        :param reg_const:
        """
        self.f = f
        def f_x_u(x,u):
            temp = self.f(x, u)
            return tf.tensordot(temp, temp, axes=0)
        self.fft = f_x_u
        self.h = h
        def h_x(x):
            temp = self.h(x)
            return tf.tensordot(temp, temp, axes=0)
        self.hht = h_x
        self.x = x0
        self.P = P0
        self.Q = Q0
        self.R = R0
        self.s_dim = s_dim
        self.dim_x = len(self.x)
        self.cov_type = "diag"
        if callable(R0):
            self.dim_y = R0().shape[0]
        else:
            self.dim_y = R0.shape[0]
        self.n_cubature_points = 2*self.dim_x
        self.reg_mat = reg_const*tf.keras.backend.eye(len(x0))

    def predict(self,t, u=None):
        # x_pred, cubature_points , P_pred_temp = cb.cubature_int(self.f, self.x, self.P, return_cubature_points=True, u=u)
        x_pred, cubature_points , P_pred_temp = cb.cubature_int_f(self.f, t, self.x, self.P, return_cubature_points=True, u=u)
        debug = False
        if debug:
            print("\n cubature_points", cubature_points)
        eps = 1e-5
        # multiplier = np.ones(len(cubature_points)//2,dtype=np.float64)
        # multiplier[-2] = -1
        # multiplier = tf.constant(multiplier )#Force first and last states to be positive, second state to be negative
        # cubature_points = cubature_points * multiplier 
        # cubature_points  = tf.nn.relu(cubature_points - eps) + eps
        # # s_k = tf.keras.activations.softplus(s_k )
        # cubature_points = cubature_points * multiplier 
        
        P_pred = P_pred_temp - tf.tensordot(x_pred, x_pred,axes=0) + self.Q
        return x_pred, P_pred

    def update(self, y, t,u=None):
        x_pred, P_pred = self.predict(t,u=u)
        x_cubature_points = cb.gen_cubature_points(x_pred, P_pred, self.dim_x, self.n_cubature_points)
        # print("\n x_cubature_points",x_cubature_points)
        eps = 1e-5
        # multiplier = np.ones(len(x_cubature_points)//2,dtype=np.float64)
        # multiplier[-2] = -1
        # multiplier = tf.constant(multiplier )#Force first and last states to be positive, second state to be negative

        # x_cubature_points  = x_cubature_points * multiplier #Force first and last states to be positive, second state to be negative
        # x_cubature_points  = tf.nn.relu(x_cubature_points - eps) + eps
        # x_cubature_points  = x_cubature_points * multiplier #Force first and last states to be positive, second state to be negative

        temp = [self.h(x_cubature_points[i]) for i in range(x_cubature_points.shape[0])]
        y_cubature_points = tf.constant(temp)
        if debug:
            print("\n y_cubature_points", y_cubature_points)
        y_pred = tf.math.reduce_mean(y_cubature_points, axis=0)
        if callable(self.R):
            R = self.R(x_pred)
        else:
            R = self.R
        P_yy = cb.cubature_int(self.hht, x_pred, P_pred, cubature_points=x_cubature_points) - tf.tensordot(y_pred, y_pred,axes = 0) + R
        if debug:
            print("\n y_pred", y_pred)
        # print("\n P_yy", P_yy)
        P_xy = tf.math.reduce_mean([tf.tensordot(x_cubature_points[i], y_cubature_points[i],axes=0) for i in range(self.n_cubature_points)], axis=0) - tf.tensordot(x_pred, y_pred,axes=0)
        Kg = tf.transpose(tf.linalg.matvec(P_xy, cb.inv_pd_mat(P_yy)))
        self.x = x_pred + tf.linalg.matvec(Kg, (y - y_pred))
        self.P = P_pred - tf.linalg.matmul(tf.linalg.matmul(Kg, P_yy), tf.transpose(Kg))
        # self.P = tf.Variable(self.P)
        return y_pred, (y - y_pred)

    def test_forward(self, y, u=None):
        x_pred, cubature_points , P_pred_temp = cb.cubature_int_f(self.f, self.x[-self.s_dim:], self.P[-self.s_dim:,-self.s_dim:], return_cubature_points=True, u=u[0:self.s_dim*2,:])
        P_pred = P_pred_temp - tf.tensordot(x_pred, x_pred,axes=0) + self.Q
        x_cubature_points = cb.gen_cubature_points(x_pred, P_pred, self.s_dim, 2*self.s_dim)#instead of self.n_cubature_points
        y_cubature_points = tf.constant([self.h(x_cubature_points[i]) for i in range(len(x_cubature_points))])
        y_pred = tf.math.reduce_mean(y_cubature_points, axis=0)
        if callable(self.R):
            R = self.R(x_pred)
        else:
            R = self.R
        P_yy = cb.cubature_int(self.hht, x_pred, P_pred, cubature_points=x_cubature_points) - tf.tensordot(y_pred, y_pred,axes = 0) + R
        P_xy = tf.math.reduce_mean([tf.tensordot(x_cubature_points[i], y_cubature_points[i],axes=0) for i in range(self.n_cubature_points)], axis=0) - tf.tensordot(x_pred, y_pred,axes=0)
        Kg = tf.transpose(tf.linalg.matvec(P_xy, cb.inv_pd_mat(P_yy)))
        self.x = x_pred + tf.linalg.matvec(Kg, (y - y_pred))
        self.P = P_pred - tf.linalg.matmul(tf.linalg.matmul(Kg, P_yy), tf.transpose(Kg))
        # self.P = tf.Variable(self.P)
        return y_pred, (y - y_pred)


    def open_loop_forward(self, y, u=None):
        x_pred, cubature_points = cb.cubature_int(self.f, self.x[-self.s_dim:], self.P[-self.s_dim:,-self.s_dim:], return_cubature_points=True, u=u)
        P_pred = cb.cubature_int(self.fft, self.x[-self.s_dim:], self.P[-self.s_dim:,-self.s_dim:], cubature_points=cubature_points, u=u) - np.outer(x_pred, x_pred) + self.Q[-self.s_dim:,-self.s_dim:]
        x_cubature_points = cb.gen_cubature_points(x_pred, P_pred, self.s_dim, 2*self.s_dim)#instead of self.n_cubature_points

        y_cubature_points = np.array([self.h(x_cubature_points[i]) for i in range(len(x_cubature_points))])
        y_pred = np.mean(y_cubature_points, axis=0)

        self.x[-self.s_dim:] = x_pred
        self.P[-self.s_dim:,-self.s_dim:] = P_pred
        #no correction term
        return y_pred, (y - y_pred)

    def reset_states(self):
        self.x[-self.s_dim:] *= 0
        
    def get_states(self):
        return self.x


class SQRCubatureFilter:
    def __init__(self, f, h, x0, P0, Q0, R0, s_dim, reg_const=0.01):
        """
        class CubatureFilter
        :param f:
        :param h:
        :param x0:
        :param P0:
        :param Q0:
        :param R0:
        :param reg_const:
        """
        self.f = f
        def f_x_u(x,u):
            temp = self.f(x, u)
            return tf.tensordot(temp, temp, axes=0)
        self.fft = f_x_u
        self.h = h
        def h_x(x):
            temp = self.h(x)
            return tf.tensordot(temp, temp, axes=0)
        self.hht = h_x
        self.x = x0
        self.P = P0
        self.Q = Q0
        self.R = R0
        self.s_dim = s_dim
        self.dim_x = len(self.x)
        self.cov_type = "diag"
        if callable(R0):
            self.dim_y = R0().shape[0]
        else:
            self.dim_y = R0.shape[0]
        self.n_cubature_points = 2*self.dim_x
        self.reg_mat = reg_const*tf.keras.backend.eye(len(x0))

    def predict(self, u=None):
        x_pred, cubature_points , P_pred_temp = cb.cubature_int_f(self.f, self.x, self.P, return_cubature_points=True, u=u)
        P_pred = P_pred_temp - tf.tensordot(x_pred, x_pred,axes=0) + self.Q
        P_pred  = self.P
        return x_pred, P_pred

    def update(self, y, u=None):
        x_pred, P_pred = self.predict(u=u)
        x_cubature_points = cb.gen_cubature_points(x_pred, P_pred, self.dim_x, self.n_cubature_points)
        temp = [self.h(x_cubature_points[i]) for i in range(x_cubature_points.shape[0])]
        y_cubature_points = tf.constant(temp)
        y_pred = tf.math.reduce_mean(y_cubature_points, axis=0)
        if callable(self.R):
            R = self.R(x_pred)
        else:
            R = self.R
        P_yy = cb.cubature_int(self.hht, x_pred, P_pred, cubature_points=x_cubature_points) - tf.tensordot(y_pred, y_pred,axes = 0) + R
        P_xy = tf.math.reduce_mean([tf.tensordot(x_cubature_points[i], y_cubature_points[i],axes=0) for i in range(self.n_cubature_points)], axis=0) - tf.tensordot(x_pred, y_pred,axes=0)
        Kg = tf.transpose(tf.linalg.matvec(P_xy, cb.inv_pd_mat(P_yy)))
        self.x = x_pred + tf.linalg.matvec(Kg, (y - y_pred))
        # self.P = P_pred - tf.linalg.matmul(tf.linalg.matmul(Kg, P_yy), tf.transpose(Kg))
        # self.P = tf.Variable(self.P)
        return y_pred, (y - y_pred)

    def test_forward(self, y, u=None):
        x_pred, cubature_points , P_pred_temp = cb.cubature_int_f(self.f, self.x[-self.s_dim:], self.P[-self.s_dim:,-self.s_dim:], return_cubature_points=True, u=u[0:self.s_dim*2,:])
        P_pred = P_pred_temp - tf.tensordot(x_pred, x_pred,axes=0) + self.Q
        
        x_cubature_points = cb.gen_cubature_points(x_pred, P_pred, self.s_dim, 2*self.s_dim)#instead of self.n_cubature_points
        y_cubature_points = tf.constant([self.h(x_cubature_points[i]) for i in range(len(x_cubature_points))])
        y_pred = tf.math.reduce_mean(y_cubature_points, axis=0)
        if callable(self.R):
            R = self.R(x_pred)
        else:
            R = self.R
        P_yy = cb.cubature_int(self.hht, x_pred, P_pred, cubature_points=x_cubature_points) - tf.tensordot(y_pred, y_pred,axes = 0) + R
        P_xy = tf.math.reduce_mean([tf.tensordot(x_cubature_points[i], y_cubature_points[i],axes=0) for i in range(self.n_cubature_points)], axis=0) - tf.tensordot(x_pred, y_pred,axes=0)
        Kg = tf.transpose(tf.linalg.matvec(P_xy, cb.inv_pd_mat(P_yy)))
        self.x = x_pred + tf.linalg.matvec(Kg, (y - y_pred))
        self.P = P_pred - tf.linalg.matmul(tf.linalg.matmul(Kg, P_yy), tf.transpose(Kg))
        self.P = tf.Variable(self.P)
        return y_pred, (y - y_pred)

    def open_loop_forward(self, y, u=None):
        x_pred, cubature_points = cb.cubature_int(self.f, self.x[-self.s_dim:], self.P[-self.s_dim:,-self.s_dim:], return_cubature_points=True, u=u)
        P_pred = cb.cubature_int(self.fft, self.x[-self.s_dim:], self.P[-self.s_dim:,-self.s_dim:], cubature_points=cubature_points, u=u) - np.outer(x_pred, x_pred) + self.Q[-self.s_dim:,-self.s_dim:]
        x_cubature_points = cb.gen_cubature_points(x_pred, P_pred, self.s_dim, 2*self.s_dim)#instead of self.n_cubature_points

        y_cubature_points = np.array([self.h(x_cubature_points[i]) for i in range(len(x_cubature_points))])
        y_pred = np.mean(y_cubature_points, axis=0)

        self.x[-self.s_dim:] = x_pred
        self.P[-self.s_dim:,-self.s_dim:] = P_pred
        #no correction term
        return y_pred, (y - y_pred)

    def reset_states(self):
        self.x[-self.s_dim:] *= 0
        
    def get_states(self):
        return self.x
    
    
    
    
    
    
#     class CubatureFilter:

#     def __init__(self, f, h, x0, P0, Q0, R0, reg_const=0.01):
#         """
#         class CubatureFilter
#         :param f:
#         :param h:
#         :param x0:
#         :param P0:
#         :param Q0:
#         :param R0:
#         :param reg_const:
#         """
#         self.f = f
#         self.fft = lambda x, u: np.outer(self.f(x, u), self.f(x, u))
#         self.h = h
#         self.hht = lambda x: np.outer(self.h(x), self.h(x))
#         self.x = x0
#         self.P = P0
#         self.Q = Q0
#         self.R = R0
#         self.S = np.linalg.qr(P0)[1].T
#         # self.Q = Q0
#         self.S_Q = np.linalg.qr(Q0)[1].T
#         # self.R = R0
#         self.S_R = np.linalg.qr(R0)[1].T
#         self.dim_x = len(self.x)
#         if callable(R0):
#             self.dim_y = R0().shape[0]
#         else:
#             self.dim_y = R0.shape[0]
#         self.n_cubature_points = 2*self.dim_x
#         self.reg_mat = reg_const*np.eye(len(x0))

#     def predict(self, u=None):
#         x_pred, cubature_points = cb.cubature_int(self.f, self.x, self.P, return_cubature_points=True, u=u)
#         P_pred = cb.cubature_int(self.fft, self.x, self.P, cubature_points=cubature_points, u=u) - \
#                  np.outer(x_pred, x_pred) + self.Q

#         return x_pred, P_pred

#     def update(self, y, u=None):
#         x_pred, P_pred = self.predict(u=u)

#         # P_pred = P_pred + self.reg_mat
#         x_cubature_points = cb.gen_cubature_points(x_pred, P_pred, self.dim_x, self.n_cubature_points)

#         # y_pred = cb.cubature_int(self.h, x_pred, P_pred, cubature_points=x_cubature_points)
#         y_cubature_points = np.array([self.h(x_cubature_points[i]) for i in range(len(x_cubature_points))])
#         y_pred = np.mean(y_cubature_points, axis=0)

#         if callable(self.R):
#             R = self.R(x_pred)
#         else:
#             R = self.R

#         P_yy = cb.cubature_int(self.hht, x_pred, P_pred, cubature_points=x_cubature_points) - \
#                np.outer(y_pred, y_pred) + R

#         P_xy = np.mean([np.outer(x_cubature_points[i], y_cubature_points[i]) for i in range(self.n_cubature_points)],
#                     axis=0) - np.outer(x_pred, y_pred)

#         Kg = np.dot(P_xy, cb.inv_pd_mat(P_yy))

#         self.x = x_pred + np.dot(Kg, (y - y_pred))
#         # self.P = P_pred - np.dot(Kg, P_yy).dot(Kg.T)

#         return y_pred, (y - y_pred)


# class SquareRootCubatureFilter:


#     def predict(self, u=None):

#         cubature_points = cb.gen_cubature_points(self.x, self.S, self.dim_x, self.n_cubature_points, sqr=True)
#         if u is not None:
#             X_ = np.concatenate([self.f(xi, u).reshape(1, -1) for xi in cubature_points], axis=0)
#         else:
#             X_ = np.concatenate([self.f(xi).reshape(1, -1) for xi in cubature_points], axis=0)

#         x_pred = np.mean(X_, axis=0)

#         XX = ((X_ - x_pred)/np.sqrt(self.n_cubature_points)).T
#         # getting R^T (lower triangular matrix)
#         S_pred = np.linalg.qr(np.concatenate((XX, self.S_Q), axis=1).T)[1].T

#         return x_pred, S_pred

#     def update(self, y, u=None):

#         x_pred, S_pred = self.predict(u=u)

#         # P_pred = P_pred + self.reg_mat
#         x_cubature_points = cb.gen_cubature_points(x_pred, S_pred, self.dim_x, self.n_cubature_points, sqr=True)

#         # y_pred = cb.cubature_int(self.h, x_pred, P_pred, cubature_points=x_cubature_points)
#         # y_cubature_points = np.array([self.h(x_cubature_points[i]) for i in range(len(x_cubature_points))])
#         # y_pred = np.mean(y_cubature_points, axis=0)
#         Y_cubature_points = np.concatenate([self.h(xcb).reshape(1, -1) for xcb in x_cubature_points], axis=0)
#         y_pred = np.mean(Y_cubature_points, axis=0)

#         YY = ((Y_cubature_points - y_pred)/np.sqrt(self.n_cubature_points)).T
#         S_yy = np.linalg.qr(np.concatenate((YY, self.S_R), axis=1).T)[1].T
#         XX = ((x_cubature_points - x_pred)/np.sqrt(self.n_cubature_points)).T

#         P_xy = XX.dot(YY.T)

#         # Kalman Gain:
#         S_yy_inv = np.linalg.inv(S_yy.T)
#         # Kg = P_xy.dot(np.linalg.inv(S_yy.T)).dot(np.linalg.inv(S_yy))
#         Kg = P_xy.dot(S_yy_inv.T).dot(S_yy_inv)

#         self.x = x_pred + np.dot(Kg, (y - y_pred))
#         self.S = np.linalg.qr(np.concatenate((XX - Kg.dot(YY), Kg.dot(self.S_R)), axis=1).T)[1].T

#         return y_pred, (y - y_pred)
