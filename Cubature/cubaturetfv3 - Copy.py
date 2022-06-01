import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

# from scipy.linalg import cholesky, cho_solve, solve_triangular
import time

debug = False
debug_mod = False
def cubature_int_f(func, mean, cov, return_cubature_points=False, cubature_points=None, u=None, sqr=False):
    """
   Computes the cubature integral of the type "\int func(x) p(x) dx", where func is an arbitrary function of x, p(x) is
   considered a Gaussian distribution with mean given by 'mean', and covariance matrix 'cov'. The integral is performed
   numerically as func_int = sum(func(sig_points(i)))/N
   The outputs are the mean vector ('func_int') and the cubature points matrix ('cubature_points').
    :param func: function to be integrated.
    :param mean: (d,) mean numpy array
    :param cov: (d,d) numpy array (covariance matrix)
    :param cubature_points: (d, 2*d) numpy array containing the 2*d cubature_points (this depends on the distribution p(x) if
                        it changes new points need to be generated.
    :param u: input signal in case of f(x,u) (default u=None, f(x) ).
    :param sqr: Boolean defining if running a square-root kf. If True than cov should be a lower triangular
    decomposition of the covariance matrix! That is cov = cholesky(P).
    :return: func_int
    -------------------------------------------------------------------
   e.g.
       cov = np.array([[1.2769, 0.0843], [0.0843, 1.0725]])
        mean = np.array([0.1, 2])
        fh = lambda x: x**2 + 10
        mean, cubature_points = cubature_int(fh, mean, cov)
        print(mean, cubature_points)
    -------------------------------------------------------------------
    Author: Tales Imbiriba
    Modified by: Ahmet Demirkaya
    Last modified in Feb 2022.
    """
    
    # x_pred, cubature_points = cb.cubature_int(self.f,   self.x, self.P, return_cubature_points=True, u=u)
    # P_pred =                  cb.cubature_int(self.fft, self.x, self.P, cubature_points=cubature_points, u=u)         - tf.tensordot(x_pred, x_pred,axes=0) + self.Q
        
    d = len(mean)
    n_points = 2 * d

    if cubature_points is None:
        # create cubature points;
        print("\n cov", cov)
        cubature_points = gen_cubature_points(mean, cov, d, n_points, sqr)
        print("\n cubature_points", cubature_points)

    int_ = func(cubature_points, u)
    
    P_pred_temp_arr = [tf.tensordot(x, x, axes=0) for x in int_]
    # print("min eigs of all p", [np.amin(np.linalg.eig(m)[0]) for m in P_pred_temp_arr])
    x_pred = tf.math.reduce_mean(int_, axis=0)
    P_pred_temp = tf.math.reduce_mean(P_pred_temp_arr, axis = 0)
    # print("min eig of mean P", np.amin(np.linalg.eig(P_pred_temp)[0]))
    
    return x_pred, cubature_points, P_pred_temp
 
def cubature_int(func, mean, cov, return_cubature_points=False, cubature_points=None, u=None, sqr=False):
    """
    Computes the cubature integral of the type "\int func(x) p(x) dx", where func is an arbitrary function of x, p(x) is
    considered a Gaussian distribution with mean given by 'mean', and covariance matrix 'cov'. The integral is performed
    numerically as func_int = sum(func(sig_points(i)))/N
    The outputs are the mean vector ('func_int') and the cubature points matrix ('cubature_points').
    :param func: function to be integrated.
    :param mean: (d,) mean numpy array
    :param cov: (d,d) numpy array (covariance matrix)
    :param cubature_points: (d, 2*d) numpy array containing the 2*d cubature_points (this depends on the distribution p(x) if
                        it changes new points need to be generated.
    :param u: input signal in case of f(x,u) (default u=None, f(x) ).
    :param sqr: Boolean defining if running a square-root kf. If True than cov should be a lower triangular
    decomposition of the covariance matrix! That is cov = cholesky(P).
    :return: func_int
    -------------------------------------------------------------------
    e.g.
        cov = np.array([[1.2769, 0.0843], [0.0843, 1.0725]])
        mean = np.array([0.1, 2])
        fh = lambda x: x**2 + 10
        mean, cubature_points = cubature_int(fh, mean, cov)
        print(mean, cubature_points)
    -------------------------------------------------------------------
    Author: Tales Imbiriba
    Last modified in Jan 2021.
    """
    
    # x_pred, cubature_points = cb.cubature_int(self.f, self.x, self.P, return_cubature_points=True, u=u)
    # P_pred = cb.cubature_int(self.fft, self.x, self.P, cubature_points=cubature_points, u=u) - tf.tensordot(x_pred, x_pred,axes=0) + self.Q

    if debug_mod:
        start= time.time()
        
    d = len(mean)
    n_points = 2 * d

    if cubature_points is None:
        # create cubature points;
        cubature_points = gen_cubature_points(mean, cov, d, n_points, sqr)

    int_mean = int_func(func, cubature_points, u)

    if return_cubature_points:
        return int_mean, cubature_points
    else:
        return int_mean

def gen_cubature_points(mean, cov, d, n_points, sqr=False):
    cubature_points = tf.keras.backend.zeros((n_points, d))
    # print("gen_cubature_points eigs of conv: ", np.amin(np.linalg.eig(cov.numpy())[0]))
    if sqr:
        L = cov
    else:
        reg = 1e-6
        if debug:
            print("\n cov",cov)
            print("eig cov", np.linalg.eigh(cov.numpy()))
        L = tf.linalg.cholesky(cov + reg * tf.keras.backend.eye(cov.shape[0]))

    num = tf.sqrt(n_points / 2)
    num = tf.cast(num, tf.float64)
    num_eye = num * tf.keras.backend.eye(d)
    xi = tf.keras.backend.concatenate((num_eye, -num_eye ), axis=1)
    xi = tf.keras.backend.transpose(xi)
    cubature_points = mean + tf.linalg.matvec(L, xi)
    return cubature_points

def int_func(func, cubature_points, u=None):
    if u is None:
        return tf.math.reduce_mean([func(x)    for x in cubature_points], axis=0)
    else:
        return tf.math.reduce_mean([func(x, u) for x in cubature_points], axis=0)

def inv_pd_mat(K, reg=1e-5):
    """
    Usage: inv_pd_mat(self, K, reg=1e-5)
    Invert (Squared) Positive Definite matrix using Cholesky decomposition.
    :param K: Positive definite matrix. (ndarray).
    :param reg: a regularization parameter (default: reg = 1e-6).
    :return: the inverse of K.
    """
    # compute inverse K_inv of K based on its Cholesky
    # decomposition L and its inverse L_inv
    K = tf.identity(K) + reg * tf.keras.backend.eye(len(K))
    L = tf.linalg.cholesky(K)
    L_inv = tf.linalg.triangular_solve(tf.transpose(L), tf.keras.backend.eye(L.shape[0]), lower=False)
    inv = tf.tensordot(L_inv, tf.transpose(L_inv), axes = 1)
    return inv 
    
