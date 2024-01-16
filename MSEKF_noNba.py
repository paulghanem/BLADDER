# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 15:00:37 2023

@author: siliconsynapse
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:31:31 2022

@author: paul
"""



import tensorflow as tf 
import tensorflow.compat.v1 as tfc
import numpy as np 
import sys 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,SimpleRNN,LSTM
import math 
from timeit import  time
from tqdm import tqdm
from keras import backend as ker

tf.compat.v1.disable_eager_execution() 
tf.compat.v1.experimental.output_all_intermediates(True)

sys.path.insert(0, '/Users/siliconsynapse/Desktop/multistream_Kalman_filter/multistream_Kalman_filter/bladder-soft-model')
sys.path.insert(0, '/home/paul/MSCKF/multistream_Kalman_filter/bladder-soft-model')


from scipy.integrate import odeint

#Local imports
from parameters import *
from functions import *
from plotting import *




class LUT:
    def __init__(self):
        
        self.Vb=5e-2
        self.Tb=1e-2
        self.Nba=1e-3
        
        
        
        
        
    def forward(self,VB0,TB0,NBa0,t):
       
       ODE0=[VB0,TB0,NBa0] 
       sol = odeint(ODE, ODE0, t)

       self.Vb = sol[-1, 0]
       self.Tb = sol[-1, 1]
       self.Nba = sol[-1, 2]
       
       return self.Vb, self.Tb, self.Nba

    def forward_rnn(self,model,xtrain,t,window_size):
        
        
        ODE0=xtrain[window_size-1,:].copy().reshape(3,) 
        xtrain[:,1]=(1/100)*xtrain[:,1]
        sol = odeint(ODE_nn, ODE0, t,args=(model,xtrain, window_size))
        xtrain[:,1]=(100)*xtrain[:,1]

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
dt=t[1]

for i in range(iterations-1):
    t1=t[i:i+2]
     
    LUT1.forward(LUT1.Vb,LUT1.Tb,LUT1.Nba,t1)  
    z[i,:]=[LUT1.Vb+0*np.random.normal(0, 1e-2, 1), LUT1.Tb+ 0*np.random.normal(0, 1e-2, 1), LUT1.Nba+ 0*np.random.normal(0, 1e-2, 1)]

SAI=np.zeros((iterations-2,ny))
for i in range (iterations-2):
    SAI[i,:]=1/np.power(np.abs(z[i,:]),2)*np.power(((z[i+1,:]-z[i,:])/dt),2)


ztrain=z.copy()

a=[]

a.append([i for i,v in enumerate(SAI[1000:,1]) if v > 1])
a=np.array(a)
a=a.flatten()

# z=z[a,:]
# a=a*dt
# t=np.linspace(0,len(a)*dt,len(a))
window_size=5
batch_size=99
t_batch= np.array_split(t, batch_size+1)
t_batch=np.array(t_batch)

    



model=Sequential()
#adding layers and forming the model
model.add(LSTM(units = 10,return_sequences=(True),input_shape=(window_size,ny)))
model.add(Dropout(0.2))

model.add(LSTM(units = 10,return_sequences=(False)))
model.add(Dropout(0.2))

model.add(Dense(1))


model.compile(optimizer="adam",loss='mean_squared_error',metrics=["accuracy"])
W=[]


for j,layer in enumerate(model.layers):
    weights=layer.get_weights()
    for ik in range(len(weights)):
        Wint=weights[ik].flatten()
        W.append(Wint)
        
    


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
nm=int(nx*2/3)


#covariance matrices

#covariance matrices
P=1e-2*np.identity(n)
P[0:nw,0:nw]=1e0*np.identity(nw)
Q=1e-1*np.identity(n)
Q[0:nw,0:nw]=1e-2*np.identity(nw)


# Pw=1e2*np.ones((nw,1))
# Px=1e-2*np.identity(nx)

# Qw=1e-1*np.ones((nw,1))
# Qx=1e-5*np.identity(nx)



R=1e-10*np.identity(int(ny*2/3)*(batch_size))
S=np.zeros((ny*(batch_size),ny*(batch_size)))


mse=np.zeros((iterations,nm))
epochs=12
mse_epochs=np.zeros((epochs,ny))

xtrain=np.zeros((batch_size,window_size,ny))
dfi_dw=np.zeros((nw))
dfi_dx=np.zeros((nx))




mse=np.zeros((t_batch.shape[1],nm))
F=np.zeros((n,n))
F[0:nw,:]=np.block([[np.identity(nw), np.zeros((nw,nx))]])
#F=np.zeros((nx,n))
output_tensor=model.output
listOfVariableTensors=model.trainable_weights
gradients_w_raw=ker.gradients(output_tensor,listOfVariableTensors)
listOfVariableTensors=model.input
gradients_x_raw=ker.gradients(output_tensor,listOfVariableTensors)

d=np.block([[1, 0 , 0], [0, 1, 0],[0, 0 , 0]])
H2=np.kron(np.eye(batch_size,dtype=int),d)
H2 = H2[~np.all(H2 == 0, axis=1)]

for i in range(epochs):
    for o in range(batch_size):
        [LUTs[o].Vb,LUTs[o].Tb,LUTs[o].Nba]=z[int(t_batch[o,0]/dt),:]
        x_hat[nw+ny*o:nw+ny*(o+1)]=np.reshape(z[int(t_batch[o,0]/dt),:],(ny,1))
    if i>0:
        xtrain=mem
    for m in t_batch[:,1:].T:
        
        index=(m/dt).astype(int)
 
        if (index[0]-1) < window_size:
            for j in range(batch_size):
                xtrain[j,:,:]=np.matmul(np.ones((window_size,1)),x_hat[nw+ny*j:nw+ny*(j+1)].T)
                mem=xtrain
        else:
            for j in range(batch_size):
                xtrain[j,0:window_size-1,:]=xtrain[j,1:window_size,:]
                xtrain[j,window_size-1,:]=x_hat[nw+ny*j:nw+ny*(j+1)].T
       
       
           
        xtrain[:,:,1]=(1/100)*xtrain[:,:,1]
        #print(sf)
        tic = time.time()
        #mse[index[0],:]=abs(z[index,:].reshape(nx,1)-x_hat[-nx:])
        
        
        
        
        for o in range(batch_size):
            
            gradients_w = sess.run(gradients_w_raw, feed_dict={model.input: xtrain[o,:,:].reshape(1,window_size,ny)})
            gradients_x = sess.run(gradients_x_raw, feed_dict={model.input: xtrain[o,:,:].reshape(1,window_size,ny)})
            gradients_x=gradients_x[0]
            
            jacob=jacobians(x_hat[nw+ny*o][0],x_hat[nw+ny*o+1][0],x_hat[nw+ny*o+2][0])
           
            

            k=0
            for j in range(len(gradients_w)):
                weights=gradients_w[j]
                weights=np.reshape(weights,(weights.size,))
                 
                dfi_dw[k:weights.size+k]=weights
                k=weights.size+k
                
            dVb_dx=np.array([1+dt*jacob[0,0],dt*jacob[0,1],dt*jacob[0,2]])    
            dTb_dx=np.array([dt*jacob[1,0],1+dt*jacob[1,1],dt*jacob[1,2]])
            dfi_dx=np.reshape(gradients_x[0,-1,:],(int(nx/batch_size),))
            dfi_dx[0]=dfi_dx[0]*dt
            dfi_dx[1]=dfi_dx[1]*dt
            dfi_dx[2]=dfi_dx[2]*dt +1
            a=np.block([np.zeros(nw),dVb_dx,np.zeros(nx-int(nx/batch_size))])
            a=shift_elements(a, ny*o, 0)
            b=np.block([np.zeros(nw),dTb_dx,np.zeros(nx-int(nx/batch_size))])
            b=shift_elements(b, ny*o, 0)
            c=np.block([dfi_dx,np.zeros(nx-int(nx/batch_size))])
            c=shift_elements(c, ny*o, 0)
            
            d=np.block([dt*dfi_dw,c])
            
            F[nw+ny*o,:]=a
            F[nw+ny*o+1,:]=b
            F[nw+ny*o+2,:]=d
   
            
        toc = time.time()
        #print("1",toc-tic)
        F3=F[:,0:nw]
        F4=F[:,nw:nw+nx]
        
        tic= time.time()
        xtrain[:,:,1]=(100)*xtrain[:,:,1]
        for j in range(batch_size):
        #model_b.fit(xtrain, z[m,1].reshape((1,)), epochs=1,batch_size=1)
            t1=t[index[j]:index[j]+2]
            sf= LUTs[j].forward_rnn(model,xtrain[j,:,:],t1,window_size)
            sf=np.array(sf)
            sf=sf.reshape((ny,1))
        
            x_hat[nw+ny*j:nw+ny*(j+1)]=sf
        toc = time.time()
        
        #print("2",toc-tic)
        
        # tic= time.time()
        # a=Pw+Qw
        
        # c=np.multiply(Pw,F3.T)
        # b=np.matmul(F3,c) + np.matmul(F4,np.matmul(Px,F4.T)) +Qx 
         
        # S=np.matmul(H2,np.matmul(b,H2.T))+R
        # Sinv=np.linalg.inv(S)
        # K=np.block([[np.matmul(c,np.matmul(H2.T,Sinv))],[np.matmul(b,np.matmul(H2.T,Sinv))]])
        # inter=x_hat[-nx:]
        # meas=np.array([])
        # for ik in range(batch_size):
        #       meas=np.append(meas,inter[[ny*ik,ny*ik+1]])
        # meas=meas.reshape((nm,1))
        # e1=z[index[0:-1],:].reshape(nx,1)-x_hat[-nx:]
        # e=z[index[0:-1],0:2].reshape(nm,1)-meas
        # x_hat=x_hat+ np.matmul(K,(e))
       
        # #K_c=np.matmul(K[0:nw,:],c.T)
        # c=np.matmul(c,H2.T)
        # K_c=np.zeros(nw)
        # for j in range(nw):
        #     K_c[j]=np.matmul(K[j,:],c[j,:])
        
        # Pw=a-np.reshape(K_c,(nw,1))
        # toc = time.time()   
        
        # Px=b-np.matmul(K[nw:nw+nx,:],np.matmul(H2,b))
        
        # print("3",toc-tic)
        tic = time.time()
        
        P=np.matmul(F,np.matmul(P,F.T)) + Q 
        H=np.block([[np.block([np.zeros((nm,nw)), H2])]])
        
        
        S=np.matmul(H,np.matmul(P,H.T)) + R
        Sinv=np.linalg.inv(S)
        K=np.matmul(P,np.matmul(H.T,Sinv))
        inter=x_hat[-nx:]
        meas=np.array([])
        for ik in range(batch_size):
            meas=np.append(meas,inter[[ny*ik,ny*ik+1]])
        meas=meas.reshape((nm,1))
        e1=z[index[0:-1],:].reshape(nx,1)-x_hat[-nx:]
        e=z[index[0:-1],0:2].reshape(nm,1)-meas
        x_hat=x_hat+ np.matmul(K,(e))
        for j in range(batch_size):
            ztrain[index[j],2]=x_hat[nw+ny*(j+1)-1]
        P= P-np.matmul(K,np.matmul(H,P))
        toc = time.time()
        print("3",toc-tic)
        
        for o in range(batch_size):
            [LUTs[o].Vb,LUTs[o].Tb,LUTs[o].Nba]=x_hat[nw+ny*o:nw+ny*(o+1)]
            
        
       
     
    
      
       
       
        k=0
        for j,layer in enumerate(model.layers):
            weights_set=[]
            weights=layer.get_weights()
            for ik in range(len(weights)):
                Wint=weights[ik]
               
                weights_size=Wint.size
                weights_shape=Wint.shape
                weights_app=np.reshape(x_hat[k:weights_size+k],weights_shape)
                k=weights_size+k
                weights_set.append(weights_app)
            layer.set_weights(weights_set)
               
        
        
        
        #mse[index[0],:]=abs(e.reshape(nm,))
        print(e1,Q[0,0])
    
    
    #mse_epochs[i,:]=np.mean(abs(mse))
    #print(mse_epochs[i,:],Q[1,1])
    
    
    if (np.remainder(i,1)==0 and Q[0,0] >=1e-1):
             Q=1e-1*Q
     #     R=1e-1*R
        
         

LUT_nn=LUT()
[Vb,Tb,Nba]=np.array([LUT_nn.Vb,LUT_nn.Tb,LUT_nn.Nba])
x_end=np.zeros((ny,iterations))
x_end[:,0]=[Vb,Tb,Nba]
xtrain=np.zeros((1,window_size,ny))
xtrain=mem[0].reshape((1,mem[0].shape[0],mem[0].shape[1],))
for m in range(iterations-window_size): 
    t1=t[m:m+2]
    if m < window_size:
        xtrain[:,m,:]=x_end[-ny:,m].T
    else:
        xtrain[:,0:window_size-1,:]=xtrain[:,1:window_size,:]
        xtrain[:,window_size-1,:]=x_end[-ny:,m].T
        
    sf=LUT_nn.forward_rnn(model,xtrain.reshape((window_size,3)),t1,window_size)
    sf=np.array(sf)
    sf=sf.reshape((ny,))
    x_end[-ny:,m+1]=sf
       
     

#plt.plot(t,z,t,model.predict(t))

     
        







 
 