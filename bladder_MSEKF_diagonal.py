#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 16:39:43 2022

@author: paul
"""

import tensorflow as tf 
import tensorflow.compat.v1 as tfc
import numpy as np 
import sys 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,SimpleRNN
import math 
from timeit import  time
from tqdm import tqdm
from keras import backend as ker
tf.compat.v1.disable_eager_execution() 


sys.path.insert(0, '/Users/siliconsynapse/Desktop/multistream_Kalman_filter/multistream_Kalman_filter/bladder-soft-model')

from scipy.integrate import odeint

#Local imports
from parameters import *
from functions import *
from plotting import *




class LUT:
    def __init__(self):
        
        self.Vb=5e-2
        self.Tb=1e-2
        self.Nba=1e-2
        
        
        
        
        
    def forward(self,VB0,TB0,NBa0,t):
       
       ODE0=[VB0,TB0,NBa0] 
       sol = odeint(ODE, ODE0, t)

       self.Vb = sol[-1, 0]
       self.Tb = sol[-1, 1]
       self.Nba = sol[-1, 2]
       
       return self.Vb, self.Tb, self.Nba

    def forward_rnn(self,model,ODE0,t,window_size):
        
        
        ODE0=ODE0.reshape(3,) 
        sol = odeint(ODE_nn, ODE0, t,args=(model,window_size))

        self.Vb = sol[-1, 0]
        self.Tb = sol[-1, 1]
        self.Nba = sol[-1, 2]
         
        return self.Vb, self.Tb, self.Nba



def shift_elements(arr, num, fill_value):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def P_dot(y,t,F3,F4,Qw,Qx):
    Pw=y[0:nw]
    Pw=Pw.reshape(nw,1)
    P12=y[nw:nw+F3.size]
    Px=y[nw+F3.size:]
    Px=Px.reshape(Qx.shape)
    
    P_dot_11=2*Pw 
    P_dot_12=np.multiply(Pw,F3.T) 
    P_dot_22=np.matmul(F4,Px) + np.matmul(Px,F4.T) 
    
    P_dot_11=P_dot_11.flatten()
    P_dot_12=P_dot_12.flatten()
    P_dot_22=P_dot_22.flatten()
    
    ODE=np.block([P_dot_11,P_dot_12,P_dot_22]).T
    
    return ODE


sess = tfc.InteractiveSession()

sess.run(tfc.initialize_all_variables())


#generate data
ny=3
samples_per_minute = 2000
t_final = 15 #[min]
t=np.linspace(0,t_final,t_final*samples_per_minute)
LUT1=LUT()
iterations=t_final*samples_per_minute
z=np.zeros((iterations-1,ny))
jac=jacobians(LUT1.Vb,LUT1.Tb,LUT1.Nba)
ODE0=[LUT1.Vb,LUT1.Tb,LUT1.Nba]

for i in range(iterations-1):
    t1=t[i:i+2]
     
    LUT1.forward(LUT1.Vb,LUT1.Tb,LUT1.Nba,t1)  
    z[i,:]=[LUT1.Vb+0*np.random.normal(0, 1e-2, 1), LUT1.Tb+ 0*np.random.normal(0, 1e-2, 1), LUT1.Nba+ 0*np.random.normal(0, 1e-2, 1)]




window_size=1





model=Sequential()
#adding layers and forming the model
model.add(Dense(units = 10,activation='sigmoid' ,input_shape=(window_size,ny)))
model.add(Dense(units = 5,activation='sigmoid'))





model.add(Dense(1))


model.compile(optimizer="adam",loss='mean_squared_error',metrics=["accuracy"])
W=[]


for j,layer in enumerate(model.layers):
    weights=layer.get_weights()[0]
    weights=np.reshape(weights,(weights.size,))
    weights_h=layer.get_weights()[1]
    weights_h=np.reshape(weights_h,(weights_h.size,))
   
    Wb=np.concatenate((weights,weights_h),axis=0)
        
    W.append(Wb)

batch_size=99
Wk=np.concatenate(W,axis=0)
Wk=np.reshape(Wk,(Wk.size,1))
x_hat=Wk
LUTs=[]

for j in range (batch_size):
    LUTs.append(LUT())
    [Vb,Tb,Nba]=np.array([LUTs[j].Vb,LUTs[j].Tb,LUTs[j].Nba])
    x_hat=np.vstack([x_hat,Vb,Tb,Nba])


#initalize cubature kalman filter parameters
n=len(x_hat) #length of neural net weights vector
nw=len(Wk)
nx=len(x_hat[-3*batch_size:])



#covariance matrices



Pw=1e2*np.ones((nw,1))
Px=1e-2*np.identity(nx)

Qw=1e-1*np.ones((nw,1))
Qx=1e-5*np.identity(nx)



R=1e-10*np.identity(ny*(batch_size))
S=np.zeros((ny*(batch_size),ny*(batch_size)))


mse=np.zeros((iterations,ny))
epochs=40
mse_epochs=np.zeros((epochs,ny))


dfi_dw=np.zeros((nw))
dfi_dx=np.zeros((nx))

dt=t[1]

t_batch= np.array_split(t, batch_size+1)
t_batch=np.array(t_batch)
mse=np.zeros((t_batch.shape[1],nx))
F=np.zeros((nx,n))

output_tensor = model.output
listOfVariableTensors = model.trainable_weights
gradients_w_raw = ker.gradients(output_tensor, listOfVariableTensors)
listOfVariableTensors = model.input
gradients_x_raw = ker.gradients(output_tensor, listOfVariableTensors)

for i in range(epochs):
    for o in range(batch_size):
        [LUTs[o].Vb,LUTs[o].Tb,LUTs[o].Nba]=z[int(t_batch[o,0]/dt),:]
        x_hat[nw+ny*o:nw+ny*(o+1)]=np.reshape(z[int(t_batch[o,0]/dt),:],(ny,1))
    for m in t_batch[:,1:].T:
        index=(m/dt).astype(int)
       
           
            #xtrain[:,window_size-1,:]=x_hat[-2:].T
            
     
       
        #print(sf
        tic = time.time()
        #mse[index[0],:]=abs(z[index,:].reshape(nx,1)-x_hat[-nx:])
      

        
        
        for o in range(batch_size):
            
            gradients_w = sess.run(gradients_w_raw, feed_dict={model.input: x_hat[nw+ny*o:nw+ny*(o+1)].reshape(1,window_size,ny)})
            gradients_x = sess.run(gradients_x_raw, feed_dict={model.input: x_hat[nw+ny*o:nw+ny*(o+1)].reshape(1,window_size,ny)})
            
            gradients_x=gradients_x[0]
            
            jacob=jacobians(x_hat[nw+ny*o],x_hat[nw+ny*o+1],x_hat[nw+ny*o+2])
           
            

            k=0
            for j in range(len(gradients_w)):
                weights=gradients_w[j]
                weights=np.reshape(weights,(weights.size,))
                 
                dfi_dw[k:weights.size+k]=weights
                k=weights.size+k
            dVb_dx=[jacob[0],jacob[1],jacob[2]]
            dTb_dx=[jacob[3],jacob[4],jacob[5]]
            dfi_dx=np.reshape(gradients_x[0,-1,:],(int(nx/batch_size),))
            dfi_dx[0]=dfi_dx[0]
            dfi_dx[1]=dfi_dx[1]
            dfi_dx[2]=dfi_dx[2]
            a=np.block([np.zeros(nw),jacob[0:3],np.zeros(nx-int(nx/batch_size))])
            a=shift_elements(a, ny*o, 0)
            b=np.block([np.zeros(nw),jacob[3:],np.zeros(nx-int(nx/batch_size))])
            b=shift_elements(b, ny*o, 0)
            c=np.block([dfi_dx,np.zeros(nx-int(nx/batch_size))])
            c=shift_elements(c, ny*o, 0)
            
            d=np.block([dfi_dw,c])
            
            F[ny*o,:]=a
            F[ny*o+1,:]=b
            F[ny*o+2,:]=d
            
           
            
        toc = time.time()
        #print("1",toc-tic)
        F3=F[:,0:nw]
        F4=F[:,nw:nw+nx]
        
        tic= time.time()
        
        for j in range(batch_size):
        #model_b.fit(xtrain, z[m,1].reshape((1,)), epochs=1,batch_size=1)
            t1=[index[j]*dt,index[j]*dt +dt]
            sf= LUTs[j].forward_rnn(model,x_hat[nw+ny*j:nw+ny*(j+1)],t1,window_size)
            sf=np.array(sf)
            sf=sf.reshape((ny,1))
        
            x_hat[nw+ny*j:nw+ny*(j+1)]=sf
        
        toc = time.time()
        
        #print("2",toc-tic)
        
        
        tic= time.time()
        
        
        P11_0=Pw
        
        P12_0=np.zeros(F3.shape).T
        P22_0=Px 
        
        P11_0_f=P11_0.flatten()
        P12_0_f=P12_0.flatten()
        P22_0_f=P22_0.flatten()
        
        ODE0=np.block([P11_0_f,P12_0_f,P22_0_f])
        
        sol = odeint(P_dot, ODE0, t1, args=(F3,F4,Qw,Qx))
        
        
        
        
        a=sol[1,0:P11_0.size].reshape(P11_0.shape) + Qw
        c=sol[1,P11_0.size:P11_0.size+P12_0.size].reshape(P12_0.shape)
        b=sol[1,P11_0.size+P12_0.size:].reshape(P22_0.shape)+   Qx
        P12_0=c
        print(a[0])
        
        S=b+R
        Sinv=np.linalg.inv(S)
        K=np.block([[np.matmul(c,Sinv)],[np.matmul(b,Sinv)]])
           
          
        e=z[index[0:-1],:].reshape(nx,1)-x_hat[-nx:]
        x_hat=x_hat+ np.matmul(K,(e))
       
        #K_c=np.matmul(K[0:nw,:],c.T)
        K_c=np.zeros(nw)
        for j in range(nw):
            K_c[j]=np.matmul(K[j,:],c[j,:])
        
        Pw= a-np.reshape(K_c,(nw,1))
        toc = time.time()   
        
        Px=b-np.matmul(K[nw:nw+nx,:],b)
        
        #print("3",toc-tic)
        
        for o in range(batch_size):
            [LUTs[o].Vb,LUTs[o].Tb,LUTs[o].Nba]=x_hat[nw+ny*o:nw+ny*(o+1)]
            
        
       
     
    
      
       
       
        k=0
        for j,layer in enumerate(model.layers):
             
             weights_size=layer.get_weights()[0].size
             weights_shape=layer.get_weights()[0].shape
             weights=np.reshape(x_hat[k:weights_size+k],weights_shape)
             k=weights_size+k
             
             weights_h_size=layer.get_weights()[1].size
             weights_h_shape=layer.get_weights()[1].shape
             weights_h=np.reshape(x_hat[k:weights_h_size+k],weights_h_shape)
             k=weights_h_size+k
             
             
             layer.set_weights([weights,weights_h])
                
        
        
        
        mse[index[0],:]=abs(e.reshape(nx,))
        print(mse[index[0],:],Qw[0,0])
    
    
    #mse_epochs[i,:]=np.mean(abs(mse))
    #print(mse_epochs[i,:],Q[1,1])
    
    
    if (np.remainder(i,5)==0 and Qw[0,0] >=1e-2):
             Qw=1e-1*Qw
     #     R=1e-1*R
        
         

LUT_nn=LUT()
[Vb,Tb,Nba]=np.array([LUT_nn.Vb,LUT_nn.Tb,LUT_nn.Nba])
x_end=np.zeros((ny,iterations))
x_end[:,0]=[Vb,Tb,Nba]
xtrain=np.zeros((1,window_size,ny))
for m in range(iterations-window_size): 
    t1=t[m:m+2]
    if m < window_size:
        xtrain[:,m,:]=x_end[-ny:,m].T
    else:
        xtrain[:,0:window_size-1,:]=xtrain[:,1:window_size,:]
        xtrain[:,window_size-1,:]=x_end[-ny:,m].T
        
    sf=LUT_nn.forward_rnn(model,xtrain,t1,window_size)
    sf=np.array(sf)
    sf=sf.reshape((ny,))
    x_end[-ny:,m+1]=sf
        
     

#plt.plot(t,z,t,model.predict(t))

     
        







 
 