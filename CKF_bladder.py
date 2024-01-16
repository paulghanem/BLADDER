#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:43:21 2022

@author: paul
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:45:45 2022

@author: paul
"""

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import math
import tensorflow as tf
from scipy.integrate import odeint
import sys
sys.path.insert(
    1, '/home/paul/MSCKF/multistream_Kalman_filter/fast_Bladder_model')
from plotting import *
from functions import *
from parameters import *
from timeit import  time
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist


# tf.compat.v1.disable_eager_execution()


def cubaturepoints(P, xk):

    L = np.linalg.cholesky(P)  # generate cholewsky matrix
    d = len(xk)
    m = 2*d
    num = np.sqrt(m/2)

    xi = np.concatenate((num*np.identity(d), -num*np.identity(d)), axis=0)
    cubature_points = xk + np.matmul(L, xi.T)  # get cubature points
    return cubature_points


sol = np.load("ode_solution_20000.npy")[0:, :]
# Initial conditions
VB0 = 5e-3  # [ml]
TB0 = 1e-3  # [Pa]
NBa0 = 1e-3  # [muV]

# Initial conditions
ODE0 = np.array([VB0, TB0, NBa0])
nx = ODE0.size
ODE0 = ODE0.reshape((nx, 1))


# generate data\
# Time range

t = sol[0:, 0]
dt = t[1]


# simulation length

# sample rate
# neural  net parameters setup

ny = 3

# creating model object
model = Sequential()
# adding layers and forming the model
model.add(Dense(5, input_shape=(1,3), activation='sigmoid', name='first-Layer'))
model.add(Dense(5, activation='sigmoid', name='second-Layer'))
model.add(Dense(1, activation=None, name='output-Layer'))


model.compile(optimizer="adam", loss='mean_squared_error',
              metrics=["accuracy"])



W = []
for j, layer in enumerate(model.layers):
    weights = layer.get_weights()[0]
    weights = np.reshape(weights, (weights.size,))
    bias = layer.get_weights()[1]
    Wb = np.concatenate((weights, bias), axis=0)
    W.append(Wb)


Wk = np.concatenate(W, axis=0)
Wk = np.reshape(Wk, (Wk.size, 1))
xk = np.concatenate([Wk, ODE0], axis=0)
nw = Wk.size


# initalize cubature kalman filter parameters
n = len(xk)  # length of neural net weights vector

m = 2*n  # set number of cubature points


batch_size = 1
# covariance matrices
P = 1e-5*np.random.rand(n)
P[:nw] = np.ones(nw)*1e2
Q = 1e-5*np.ones(n)
Q[:nw] = np.ones(nw)*1e8
R = 1e-1*np.identity(ny*(batch_size))
cubature_points = np.zeros((n, 2))
Xk_c = np.zeros((n, 2))


iterations = t.size

mse = np.zeros(iterations)


for i, time1 in enumerate(t):
    if i < (len(t)-2):
        t1 = t[i:i+2]
    else:
        t1 = t[i:len(t)]

    #smpl = np.random.choice(t, size=batch_size)

    y_epoch = np.reshape(sol[i, 1:], (3, 1))
    # cubature kalman algorithm
    tic = time.time()
    d = len(xk)
    L = np.zeros(n)
    for j in range(n):
        L[j] = np.sqrt(np.maximum(0, P[j]))  # generate cholewsky matrix

    m = 2*d
    num = np.sqrt(m/2)

    xi = np.concatenate((num*np.identity(d), -num*np.identity(d)), axis=0)
    cubature_points = xk + np.transpose(L*xi)  # get cubature points
    toc = time.time()
    print("1", toc-tic)

    
    # if i >10 :
    #     a=np.argsort(np.abs(P))
    #     a=np.array(a[0:int(0.2*P.size)])
    # else :
    #     a=np.argsort(P)
    #     a=np.array(a[0:int(P.size)])

    # a=np.concatenate((a,a+n))

    f = np.zeros([n, m])

    for c in range(m):
        if cubature_points[-3, c] < 3e-3:
            cubature_points[-3, c] = 3e-3
        if cubature_points[-2, c] < 0:
            cubature_points[-2, c] = 0
        if cubature_points[-1, c] < 0:
            cubature_points[-1, c] = 0
        k = 0
        for layer in model.layers:
            weights_size = layer.get_weights()[0].size
            weights_shape = layer.get_weights()[0].shape

            bias_size = layer.get_weights()[1].size

            weights = tf.reshape(cubature_points[k:weights_size+k, c], weights_shape)
            k = weights_size+k
            bias = cubature_points[k:bias_size+k, c]
            k = bias_size+k
            layer.set_weights([weights, bias])
        out = odeint(ODE, cubature_points[k:, c], t1, args=(model,), mxstep=5)
        # calculate neural network output of each cubature point
        f[:, c] = np.concatenate([cubature_points[0:k, c], out[-1, :].reshape((nx,))])
    xstar=f
    x_hat = np.mean(xstar, axis=1)
    x_hat = np.reshape(x_hat, (x_hat.size, 1))
    for j in range(n):
        P[j] = 1/(m)*np.dot(xstar[j, :], xstar[j, :].T)-(x_hat[j, :]
                                                         * x_hat[j, :].T) + Q[j]  # update covariance matrix

     # evaluate cubature points
    L = np.zeros(n)
    for j in range(n):
        L[j] = np.sqrt(np.maximum(0, P[j]))  # generate cholewsky matrix

    m = 2*d
    num = np.sqrt(m/2)

    xi = np.concatenate((num*np.identity(d), -num*np.identity(d)), axis=0)
    Xk_c = x_hat + np.transpose(L*xi)
    z = []
    z_hat = []

    z = Xk_c[k:, :]

    z_hat = np.mean(z, axis=1)

    # update covariances
    x_hat = np.array(x_hat)
    x_hat = np.reshape(x_hat, (x_hat.size, 1))
    z_hat = np.array(z_hat)
    z_hat = np.reshape(z_hat, (z_hat.size, 1))
    Xk_c = np.array(Xk_c)
    Pzz = 1/(m)*np.matmul(z, z.T) - np.matmul(z_hat, z_hat.T) + R
    Pxz = 1/(m)*np.matmul(Xk_c, z.T) - np.matmul(x_hat, z_hat.T)
    # ucalculate kalman gain
    Kk = np.matmul(Pxz, np.linalg.inv(Pzz))
    # update weights wk+1
    xk = x_hat+np.matmul(Kk, (y_epoch-z_hat))
    tic = time.time()
    # update covariance matrix
    Pzz_1 = np.linalg.inv(Pzz)

    for j in range(n):
        P[j] = P[j]-np.matmul(np.matmul(Pxz[j, :], Pzz_1), Pxz[j, :].T)
    toc = time.time()
    print("3", toc-tic)


# store resulting estimated weights and neural net output
    # data.Wk(:,int16(i))=Wk;
    # data.f(int16(i/params.dt))=z_hat(end);
    # data.P(:,:,i)=P;

    # if i>=2*params.dt
    #     data.deltaW(:,int16(i/params.dt))=data.Wk(:,int16(i/params.dt))-data.Wk(:,int16(i/params.dt-1));
    #     deltaP=data.P(:,:,i)-data.P(:,:,i-1);

    k = 0
    for layer in model.layers:
        weights_size = layer.get_weights()[0].size
        weights_shape = layer.get_weights()[0].shape

        bias_size = layer.get_weights()[1].size
        bias_shape = layer.get_weights()[1].shape

        weights = tf.reshape(xk[k:weights_size+k], weights_shape)
        k = weights_size+k
        bias = xk[k:bias_size+k]
        bias = tf.reshape(bias, bias_shape)
        k = bias_size+k
        layer.set_weights([weights, bias])

    # f=model(t) #calculate neural net output

    mse[i] = np.mean(abs(y_epoch-z_hat))
    print(mse[i], Q[1])
    print(m)

    if (np.remainder(i, 200) == 0):
        Q[:nw] = 1e-1*Q[:nw]


plt.plot(t, y, t, model(t))
