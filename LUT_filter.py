# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:18:16 2023

@author: siliconsynapse
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
import scipy.io as spio

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
        self.Nba=1e-2
        
        
        
        
        
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




V = spio.loadmat('V.mat', squeeze_me=True)
V=V['V']
P = spio.loadmat('P.mat', squeeze_me=True)
P=P['P']
t= spio.loadmat('t.mat', squeeze_me=True)
t=t['t']
dt=t[1]
iterations=len(t)
z=np.zeros((iterations-1,ny))
T=np.zeros((iterations-1))



for i in range(iterations-1):
    T[i]=np.divide(P[i]*RB(V[i])-Pabd,2*hB)
    T[i]=T[i]*133.322
    z[i,0]=V[i]
    z[i,1]=T[i]
    z[i,2]=0.01

z=z[3000:,:]
t=t[3000:]
iterations=len(t)
window_size=5
x_filt=np.zeros((iterations,ny))

    



model= tf.keras.models.load_model("LUT_model.keras")
x_hat=[]
LUTs=[]


LUTs.append(LUT())
[Vb,Tb,Nba]=np.array([LUTs[0].Vb,LUTs[0].Tb,LUTs[0].Nba])
x_hat=np.vstack([Vb,Tb,Nba])


#initalize cubature kalman filter parameters
n=len(x_hat) #length of neural net weights vector
nx=len(x_hat[-ny:])
nm=int(nx*2/3)


#covariance matrices

#covariance matrices
# P=1e-2*np.identity(n)
# P[0:nw,0:nw]=1e0*np.identity(nw)
# Q=1e-1*np.identity(n)
# Q[0:nw,0:nw]=1e-2*np.identity(nw)


Px=1e-1*np.identity(nx)

Qx=1e-2*np.identity(nx)
Qx[1,1]=1e0
Qx[2,2]=1e-2



R=1e-10*np.identity(int(ny*2/3))
R[1,1]=1e2
S=np.zeros((ny,ny))


mse=np.zeros((iterations,nm))
epochs=6
mse_epochs=np.zeros((epochs,ny))

xtrain=np.zeros((1,window_size,ny))
dfi_dx=np.zeros((nx))


F=np.zeros((n,n))
#F=np.zeros((nx,n))
output_tensor=model.output
listOfVariableTensors=model.input
gradients_x_raw=ker.gradients(output_tensor,listOfVariableTensors)

d=np.block([[1, 0 , 0], [0, 1, 0]])
H2=d


[LUTs[0].Vb,LUTs[0].Tb,LUTs[0].Nba]=z[0,:]
x_hat=np.reshape(z[0,:],(ny,1))

for m in range(iterations-1):
 
    if m < window_size:
        xtrain[0,:,:]=np.matmul(np.ones((window_size,1)),x_hat.T)
        
    else:
            xtrain[0,0:window_size-1,:]=xtrain[0,1:window_size,:]
            xtrain[0,window_size-1,:]=x_hat.T
   
   
       
    xtrain[:,:,1]=(1/100)*xtrain[:,:,1]
    #print(sf)
    tic = time.time()
    #mse[index[0],:]=abs(z[index,:].reshape(nx,1)-x_hat[-nx:])
    
    
    
    
    
    
    gradients_x = sess.run(gradients_x_raw, feed_dict={model.input: xtrain.reshape(1,window_size,ny)})
    gradients_x=gradients_x[0]
    
    jacob=jacobians(x_hat[0][0],x_hat[1][0],x_hat[2][0])
   
    

    dVb_dx=np.array([1+dt*jacob[0,0],dt*jacob[0,1],dt*jacob[0,2]])    
    dTb_dx=np.array([dt*jacob[1,0],1+dt*jacob[1,1],dt*jacob[1,2]])
    dfi_dx=np.reshape(gradients_x[0,-1,:],(nx,))
    dfi_dx[0]=dfi_dx[0]*dt
    dfi_dx[1]=dfi_dx[1]*dt
    dfi_dx[2]=dfi_dx[2]*dt +1
    a=dVb_dx
    b=dTb_dx
    c=dfi_dx
    
    F[0,:]=a
    F[1,:]=b
    F[2,:]=c
    
   
     
    toc = time.time()
    #print("1",toc-tic)
    F4=F
    
    tic= time.time()
   
    xtrain[:,:,1]=(100)*xtrain[:,:,1]
    #model_b.fit(xtrain, z[m,1].reshape((1,)), epochs=1,batch_size=1)
    t1=t[m:m+2]
    sf= LUTs[0].forward_rnn(model,xtrain[0,:,:],t1,window_size)
    sf=np.array(sf)
    sf=sf.reshape((ny,1))
    
    x_hat=sf
    
    
    #print("2",toc-tic)
    A=F4
    Px=np.matmul(A,np.matmul(Px,A.T)) + Qx
    
    
    C=H2
    #C_kw=np.matmul(C,F3)
    Sx=np.matmul(C,np.matmul(Px,C.T)) + R
    Sinv=np.linalg.inv(Sx)
    Kx=np.matmul(Px,np.matmul(C.T,Sinv))
     
    inter=x_hat[-nx:]
    meas=np.array([])
    meas=np.append(meas,inter[[0,1]])
    meas=meas.reshape((nm,1))
    e1=z[m+1,:].reshape(nx,1)-x_hat[-nx:]
    e=z[m+1,0:2].reshape(nm,1)-meas
    x_=x_hat[-nx:].copy()
    x_hat[-nx:]=x_hat[-nx:]+ np.matmul(Kx,(e))
    Px= Px-np.matmul(Kx,np.matmul(C,Px))
   
    
    
    # print("3",toc-tic)
    tic = time.time()
    
    # P=np.matmul(F,np.matmul(P,F.T)) + Q 
    # H=np.block([[np.block([np.zeros((nm,nw)), H2])]])
    
    
    # S=np.matmul(H,np.matmul(P,H.T)) + R
    # Sinv=np.linalg.inv(S)
    # K=np.matmul(P,np.matmul(H.T,Sinv))
    # inter=x_hat[-nx:]
    # meas=np.array([])
    # for ik in range(batch_size):
    #     meas=np.append(meas,inter[[ny*ik,ny*ik+1]])
    # meas=meas.reshape((nm,1))
    # e1=z[index[0:-1],:].reshape(nx,1)-x_hat[-nx:]
    # e=z[index[0:-1],0:2].reshape(nm,1)-meas
    # x_hat=x_hat+ np.matmul(K,(e))
    # for j in range(batch_size):
    #     ztrain[index[j],2]=x_hat[nw+ny*(j+1)-1]
    # P= P-np.matmul(K,np.matmul(H,P))
    # toc = time.time()
    # print("3",toc-tic)
    
   
    [LUTs[0].Vb,LUTs[0].Tb,LUTs[0].Nba]=x_hat
        
    x_filt[m,:]=x_hat.reshape((ny,))
   
 

  
   
   
    
           
    
    
    
    #mse[index[0],:]=abs(e.reshape(nm,))
    print(e1)
    
    
    #mse_epochs[i,:]=np.mean(abs(mse))
    #print(mse_epochs[i,:],Q[1,1])
    
     #     R=1e-1*R
        
spio.savemat('filtered_signal.mat', {'X': x_filt})          
         
model.save("LUT_model.keras")
LUT_nn=LUT()
[Vb,Tb,Nba]=np.array([LUT_nn.Vb,LUT_nn.Tb,LUT_nn.Nba])
x_end=np.zeros((ny,iterations))
x_end[:,0]=[Vb,Tb,Nba]
xtrain=np.zeros((1,window_size,ny))
xtrain=mem[0].reshape((1,mem[0].shape[0],mem[0].shape[1],))
for m in range(iterations-window_size): 
    t1=t[m:m+2]
    if m < window_size:
        xtrain[:,:,:]=x_end[-ny:,m].T
    else:
        xtrain[:,0:window_size-1,:]=xtrain[:,1:window_size,:]
        xtrain[:,window_size-1,:]=x_end[-ny:,m].T
        
    sf=LUT_nn.forward_rnn(model,xtrain.reshape((window_size,3)),t1,window_size)
    sf=np.array(sf)
    sf=sf.reshape((ny,))
    x_end[-ny:,m+1]=sf
       
     

#plt.plot(t,z,t,model.predict(t))

     
        







 
 