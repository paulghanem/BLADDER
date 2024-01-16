#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 17:19:55 2022

@author: paul
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:41:09 2022

@author: paul
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 13:51:33 2022

@author: paul
"""


import tensorflow as tf 
import tensorflow.compat.v1 as tfc
import numpy as np 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,SimpleRNN
import math 
from timeit import  time
from tqdm import tqdm
from keras import backend as ker
tf.compat.v1.disable_eager_execution() 
tf.compat.v1.experimental.output_all_intermediates(True)




class pend:
    def __init__(self):
        self.m=1
        self.l=1
        self.c = 0.3
        self.g = 9.8
        self.th=2.5
        self.w=0
        self.z=[self.th,self.w]
        self.dt=0.001
        
        
        
        
    def forward(self):
       self.dth=self.w
       self.dw=-self.g/self.l*math.sin(self.th)-self.c/(self.m*math.pow(self.l,2))*self.w
       
        
       self.th=self.w*self.dt+self.th
       self.w= self.w+self.dt*(-self.g/self.l*math.sin(self.th)-self.c/(self.m*math.pow(self.l,2))*self.w)

    def forward_rnn(self,nn,inp):
        
        self.dth=self.w
        self.dw=nn.predict(inp,verbose=0)
        
         
        self.th=self.w*self.dt+self.th
        self.w= self.w+self.dt*self.dw
         
        return self.th, self.w



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
ny=2
pendul1=pend()
iterations=30000
z=np.zeros((iterations,ny))
for i in range(iterations):
    pendul1.forward()  
    z[i,:]=[pendul1.th+0*np.random.normal(0, 1e-2, 1),pendul1.w+ 0*np.random.normal(0, 1e-2, 1)]




window_size=1


#creating model object
model_b=Sequential()
#adding layers and forming the model
model_b.add(Dense(units = 50, input_shape=(window_size,2)))
model_b.add(Dense(units = 50))
model_b.add(Dense(1))
model_b.compile(optimizer="adam",loss='mean_squared_error',metrics=["accuracy"])



model=Sequential()
#adding layers and forming the model
model.add(Dense(units = 500,activation='tanh' ,input_shape=(window_size,2)))
model.add(Dense(units = 40,activation='tanh'))


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
pends=[]

for j in range (batch_size):
    pends.append(pend())
    [theta,omega]=np.array([pends[j].th,pends[j].w])
    x_hat=np.vstack([x_hat,theta,omega])


#initalize cubature kalman filter parameters
n=len(x_hat) #length of neural net weights vector
nw=len(Wk)
nx=len(x_hat[-2*batch_size:])



#covariance matrices

#covariance matrices
# P=1e2*np.identity(n)

# Q=1e-5*np.identity(n)
# Q[0:nw,0:nw]=1e3*np.identity(nw)

Pw=1e2*np.ones((nw,1))
Px=1e-2*np.identity(nx)
Pwx=np.zeros((nw,nx))

Qw=1e-2*np.ones((nw,1))
Qx=1e-5*np.identity(nx)



R=1e-10*np.identity(ny*(batch_size))
S=np.zeros((ny*(batch_size),ny*(batch_size)))


mse=np.zeros((iterations,ny))
epochs=200
mse_epochs=np.zeros((epochs,ny))


dfi_dw=np.zeros((nw))
dfi_dx=np.zeros((nx))

dt=pends[0].dt
t=np.linspace(0,iterations*dt,iterations)
t_batch= np.array_split(t, batch_size+1)
t_batch=np.array(t_batch)
mse=np.zeros((t_batch.shape[1],nx))
F=np.zeros((nx,n))
output_tensor=model.output
listOfVariableTensors=model.trainable_weights
gradients_w_raw=ker.gradients(output_tensor,listOfVariableTensors)
listOfVariableTensors=model.input
gradients_x_raw=ker.gradients(output_tensor,listOfVariableTensors)


# F[0:nw,:]=np.block([[np.identity(nw), np.zeros((nw,nx))]])
for i in range(epochs):
    for o in range(batch_size):
        [pends[o].th,pends[o].w]=z[int(t_batch[o,0]/dt),:]
        x_hat[nw+2*o:nw+2*(o+1)]=np.reshape(z[int(t_batch[o,0]/dt),:],(2,1))
    for m in t_batch[:,1:].T:
        
        index=(m/dt).astype(int)
       
           
            #xtrain[:,window_size-1,:]=x_hat[-2:].T
            
     
       
        #print(sf
        tic = time.time()
        #mse[index[0],:]=abs(z[index,:].reshape(nx,1)-x_hat[-nx:])
        
        
        
        
        for o in range(batch_size):
            
            gradients_w = sess.run(gradients_w_raw, feed_dict={model.input: x_hat[nw+2*o:nw+2*(o+1)].reshape(1,window_size,2)})
            gradients_x = sess.run(gradients_x_raw, feed_dict={model.input: x_hat[nw+2*o:nw+2*(o+1)].reshape(1,window_size,2)})
            gradients_x=gradients_x[0]
            
           
            

            k=0
            for j in range(len(gradients_w)):
                weights=gradients_w[j]
                weights=np.reshape(weights,(weights.size,))
                 
                dfi_dw[k:weights.size+k]=weights
                k=weights.size+k
         
            dfi_dx=np.reshape(gradients_x[0,-1,:],(int(nx/batch_size),))
            dfi_dx[0]=dfi_dx[0]*pends[o].dt
            dfi_dx[1]=dfi_dx[1]*pends[o].dt +1
            a=np.block([np.zeros(nw),1, pends[o].dt,np.zeros(nx-int(nx/batch_size))])
            a=shift_elements(a, 2*o, 0)
            b=np.block([dfi_dx,np.zeros(nx-int(nx/batch_size))])
            b=shift_elements(b, 2*o, 0)
            
            c=np.block([pends[o].dt*dfi_dw,b])
            
            F[2*o,:]=a
            F[2*o+1,:]=c
            
        toc = time.time()
        print("1",toc-tic)
        F3=F[:,0:nw]
        F4=F[:,nw:nw+nx]
        
        tic= time.time()
        for j in range(batch_size):
        #model_b.fit(xtrain, z[m,1].reshape((1,)), epochs=1,batch_size=1)
            sf=pends[j].forward_rnn(model,x_hat[nw+2*j:nw+2*(j+1)].T.reshape(1,window_size,2))
            sf=np.array(sf)
            sf=sf.reshape((2,1))
        
            x_hat[nw+2*j:nw+2*(j+1)]=sf
        toc = time.time()
        
        print("2",toc-tic)
        
        
        tic= time.time()
        a=Pw+Qw
        
        c=np.multiply(Pw,F3.T) 
        b=np.matmul(F3,c) + np.matmul(F4,np.matmul(Px,F4.T)) +Qx 
         
        S=b+R
        Sinv=np.linalg.inv(S)
        K=np.block([[np.matmul(c,Sinv)],[np.matmul(b,Sinv)]])
           
          
        e=z[index[0:-1],:].reshape(nx,1)-x_hat[-nx:]
        x_hat=x_hat+ np.matmul(K,(e))
       
        #K_c=np.matmul(K[0:nw,:],c.T)
        K_c=np.zeros(nw)
        for j in range(nw):
            K_c[j]=np.matmul(K[j,:],c[j,:])
        
        Pw=a-np.reshape(K_c,(nw,1))
        toc = time.time()   
        
        Px=b-np.matmul(K[nw:nw+nx,:],b)
        
        print("3",toc-tic)
        # tic = time.time()
        # P=np.matmul(F,np.matmul(P,F.T)) + Q 
        # H=np.block([[np.block([np.zeros((nx,nw)), np.identity(nx)])]])
        
        
        # S=np.matmul(H,np.matmul(P,H.T)) + R
        # Sinv=np.linalg.inv(S)
        # K=np.matmul(P,np.matmul(H.T,Sinv))
        # e=z[index[0:-1],:].reshape(nx,1)-x_hat[-nx:]
        # x_hat=x_hat+ np.matmul(K,(e))
        # P= P-np.matmul(K,np.matmul(H,P))
        # toc = time.time()
        # print("3",toc-tic)
        
        for o in range(batch_size):
            [pends[o].th,pends[o].w]=x_hat[nw+2*o:nw+2*(o+1)]
            
        
       
     
    
      
       
       
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
    
    
    if (np.remainder(i,1)==0 and Qw[0,0] >=1e-1):
             Qw=1e-1*Qw
    #  #     R=1e-1*R
    #        print(Qw[0,0])
        
         

pend_nn=pend()
[theta,omega]=np.array([pend_nn.th,pend_nn.w])
x_end=np.zeros((ny,5*iterations))
x_end[:,0]=[theta,omega]
xtrain=np.zeros((1,window_size,2))
for m in range(5*iterations-window_size): 
    if m < window_size:
        xtrain[:,m,:]=x_end[-2:,m].T
    else:
        xtrain[:,0:window_size-1,:]=xtrain[:,1:window_size,:]
        xtrain[:,window_size-1,:]=x_end[-2:,m].T
        
    sf=pend_nn.forward_rnn(model,xtrain)
    sf=np.array(sf)
    sf=sf.reshape((2,))
    x_end[-2:,m+1]=sf
        
     

#plt.plot(t,z,t,model.predict(t))

     
        







 
 