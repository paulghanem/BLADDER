# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:55:58 2023

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

sys.path.insert(0, '/Users/siliconsynapse/Desktop/multistream_Kalman_filter/multistream_Kalman_filter/Python_LUT_Model1.4_10282023/Python_LUT_Model1.4_10282023')



from scipy.integrate import odeint

#Local imports
from parameters import *
from functions import *
from plotting import *




class LUT:
    def __init__(self):
        
        self.Vb=0.0498
        self.Pb=728.59
        self.Nba=0.0098
        
        
        
        
        
    def forward(self,VB0,PB0,NBa0,t):
       
       ODE0=[VB0,PB0,NBa0] 
       sol = odeint(ODE, ODE0, t)

       self.Vb = sol[-1, 0]
       self.Pb = sol[-1, 1]
       self.Nba = sol[-1, 2]
       
       return self.Vb, self.Pb, self.Nba

    def forward_rnn(self,model,xtrain,t,window_size):
        
        
        ODE0=xtrain[window_size-1,:].copy().reshape(3,) 
        xtrain[:,1]=(1/1000)*xtrain[:,1]
        sol = odeint(ODE_nn, ODE0, t,args=(model,xtrain, window_size))
        xtrain[:,1]=(1000)*xtrain[:,1]

        self.Vb = sol[-1, 0]
        self.Pb = sol[-1, 1]
        #self.Nba = np.maximum(0,sol[-1, 2])
        self.Nba = sol[-1, 2]
         
        return self.Vb, self.Pb, self.Nba



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

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

sess = tfc.InteractiveSession()

sess.run(tfc.initialize_all_variables())

#generate data
ny=3
t= spio.loadmat('t.mat', squeeze_me=True)
t=t['t']



V = spio.loadmat('V.mat', squeeze_me=True)
V=V['V']
P = spio.loadmat('P.mat', squeeze_me=True)
P=P['P']

V=V[9000:]
P=P[9000:]
P=P*133.322
t=t[9000:]
t=t-t[0]
dt=t[1]

LUT1=LUT()
iterations=len(t)
z=np.zeros((iterations-1,ny))
jac=jacobians(LUT1.Vb,LUT1.Pb,LUT1.Nba)
ODE0=[LUT1.Vb,LUT1.Pb,LUT1.Nba]
dt=t[1]

for i in range(iterations-1):
    t1=t[i:i+2]
     
    LUT1.forward(LUT1.Vb,LUT1.Pb,LUT1.Nba,t1)  
    z[i,:]=[LUT1.Vb+0*np.random.normal(0, 1e-2, 1), LUT1.Pb+ 0*np.random.normal(0, 1e-2, 1), LUT1.Nba+ 0*np.random.normal(0, 1e-2, 1)]


z[:,0]=V[1:]
z[:,1]=P[1:]


ztrain=z.copy()



window_size=1
batch_size=1
t_batch= np.array_split(t, batch_size+1)
t_batch=np.array(t_batch)

    



model=Sequential()
#adding layers and forming the model
model.add(Dense(units = 30,activation='tanh',input_shape=(window_size,ny)))

model.add(Dense(units = 30,activation='tanh'))


model.add(Dense(units=1))


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
    [Vb,Pb,Nba]=np.array([LUTs[j].Vb,LUTs[j].Pb,LUTs[j].Nba])
    x_hat=np.vstack([x_hat,Vb,Pb,Nba])


#initalize cubature kalman filter parameters
n=len(x_hat) #length of neural net weights vector
nw=len(Wk)
nx=len(x_hat[-3*batch_size:])
nm=int(nx*2/3)


#covariance matrices

#covariance matrices
# P=1e-2*np.identity(n)
# P[0:nw,0:nw]=1e0*np.identity(nw)
# Q=1e-1*np.identity(n)
# Q[0:nw,0:nw]=1e-2*np.identity(nw)


Pw=1e2*np.identity((nw))
Px=1e-0*np.identity(nx)

Qw=1e1*np.identity((nw))
Qx=1e-1*np.identity(nx)
Qx[2,2]=1e-1



R=1e1*np.identity(int(ny*2/3)*(batch_size))
S=np.zeros((ny*(batch_size),ny*(batch_size)))


mse=np.zeros((iterations,nm))
epochs=30
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

x_filt=np.zeros((iterations,ny))

for i in range(epochs):
    for o in range(batch_size):
        [LUTs[o].Vb,LUTs[o].Pb,LUTs[o].Nba]=z[int(t_batch[o,0]/dt),:]
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
       
       
           
        xtrain[:,:,1]=(1/1000)*xtrain[:,:,1]
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
            dPb_dx=np.array([dt*jacob[1,0],1+dt*jacob[1,1],dt*jacob[1,2]])
            dfi_dx=np.reshape(gradients_x[0,-1,:],(int(nx/batch_size),))
            dfi_dx[0]=dfi_dx[0]*dt
            dfi_dx[1]=dfi_dx[1]*dt
            dfi_dx[2]=dfi_dx[2]*dt +1
            a=np.block([np.zeros(nw),dVb_dx,np.zeros(nx-int(nx/batch_size))])
            a=shift_elements(a, ny*o, 0)
            b=np.block([np.zeros(nw),dPb_dx,np.zeros(nx-int(nx/batch_size))])
            b=shift_elements(b, ny*o, 0)
            c=np.block([dfi_dx,np.zeros(nx-int(nx/batch_size))])
            c=shift_elements(c, ny*o, 0)
            
            d=np.block([dt*dfi_dw,c])
            
            F[nw+ny*o,:]=a
            F[nw+ny*o+1,:]=b
            F[nw+ny*o+2,:]=d
            if o ==0:
            # c=np.block([pends[o].dt*dfi_dw,b])
                
                C_kw=np.block([[np.zeros((nw))],[np.zeros((nw))],[dt*dfi_dw]])
                
            else:
                
                C_kw=np.block([[C_kw],[np.zeros((nw))],[np.zeros((nw))],[dt*dfi_dw]])
   
         
        toc = time.time()
        #print("1",toc-tic)
        F3=F[nw:,0:nw]
        F4=F[nw:,nw:nw+nx]
        
        tic= time.time()
        Pw=Pw+Qw
        xtrain[:,:,1]=(1000)*xtrain[:,:,1]
        for j in range(batch_size):
        #model_b.fit(xtrain, z[m,1].reshape((1,)), epochs=1,batch_size=1)
            t1=t[index[j]:index[j]+2]
            sf= LUTs[j].forward_rnn(model,xtrain[j,:,:],t1,window_size)
            sf=np.array(sf)
            sf=sf.reshape((ny,1))
        
            x_hat[nw+ny*j:nw+ny*(j+1)]=sf
        
        
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
        for ik in range(batch_size):
              meas=np.append(meas,inter[[ny*ik,ny*ik+1]])
        meas=meas.reshape((nm,1))
        e1=z[index[0:-1],:].reshape(nx,1)-x_hat[-nx:]
        e=z[index[0:-1],0:2].reshape(nm,1)-meas
        x_=x_hat[-nx:].copy()
        x_hat[-nx:]=x_hat[-nx:]+ np.matmul(Kx,(e))
        #relu
        x_hat[-1]=np.maximum(0.5,np.minimum(2,x_hat[-1]))
        
        Px= Px-np.matmul(Kx,np.matmul(C,Px))
       
        Sw=np.matmul(C_kw,np.matmul(Pw,C_kw.T)) + Qx
        Sinv=np.linalg.inv(Sw)
        Kw=np.matmul(Pw,np.matmul(C_kw.T,Sinv))
       
        e_=x_hat[-nx:]-x_
        x_hat[:nw]=x_hat[:nw]+ np.matmul(Kw,(e_))
        Pw= Pw-np.matmul(Kw,np.matmul(C_kw,Pw))
        
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
        
        for o in range(batch_size):
             [LUTs[o].Vb,LUTs[o].Tb,LUTs[o].Nba]=x_hat[nw+ny*o:nw+ny*(o+1)]
             
             x_filt[index[o],:]=x_hat[nw+ny*o:nw+ny*(o+1)].reshape((ny,))
            
        
       
     
    
      
       
       
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
        print(e1,Qw[0,0])
    
    
    #mse_epochs[i,:]=np.mean(abs(mse))
    #print(mse_epochs[i,:],Q[1,1])
    
    
    if (np.remainder(i,2)==0 and Qw[0,0] >=1e-1):
             Qw=1e-1*Qw
     #     R=1e-1*R
        
         
model.save("LUT_model_1.4_1.keras")
LUT_nn=LUT()
[LUT_nn.Vb,LUT_nn.Pb,LUT_nn.Nba]=z[0,-ny:]
[Vb,Pb,Nba]=np.array([LUT_nn.Vb,LUT_nn.Pb,LUT_nn.Nba])
x_end=np.zeros((ny,iterations))
x_end[:,0]=[Vb,Pb,Nba]
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

     
        







 
 