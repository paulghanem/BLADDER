# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 17:22:51 2023

@author: siliconsynapse
"""


from autograd import grad, jacobian
import tensorflow as tf 
import tensorflow.compat.v1 as tfc
import autograd.numpy as np
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
pi=math.pi
tf.compat.v1.disable_eager_execution() 
tf.compat.v1.experimental.output_all_intermediates(True)

from scipy.integrate import odeint
from scipy.linalg import block_diag




class LUT:
    def __init__(self):
        
       
        
        #Conversion factors
        self.fac_cmH20_Pa = 98.0665  # 1 cmH20 = 98.0665 Pa
        self.fac_mmHg_Pa =133.322    # 1 mmHg  = 133.322 Pa
        
        #Initial conditions
        self.Vb=0.0498 
        self.Tb=9.925e+03 
        self.Nba=0.0098
        
        # inflow rate 
        self.Q_in    = 0.998/3600; #[ml/s]
        
        #Abdominal pressure
        self.Pabd    = 0.0
        
        #Reference size of the bladder
        self.VB0star= 0.0197 # [ml]
        self.RB0 = np.cbrt((3*self.VB0star)/(4*pi)) # [cm]
        self.LB0=2*pi*self.RB0 #[cm]
        self.hB= self.RB0/20 #Bladder thickness [cm]
        
        
        
        
        
        
        #Fluid flow in/out of the bladder
        self.alpa = 1.8e-6  # urethral resistance [ml/(s Pa)] -- [GR]
        self.Pc   = 19.53 * self.fac_mmHg_Pa  # cutoff urethral pressure [Pa] -- [GR]
        #C_QK = 5 * 1e-8  # kidney --> bladder flow rate [m^3/s] -- [B]
        #Ptheta = 30 * fac_cmH20_Pa  # cutoff ureters pressure [Pa] -- [B] (in-flow is constant in this implementation)
        
        #Abdominal pressure
        self.Pabd    = 0     #[Pa]
        
        # time steps for euler method 
        
        #----------------- NEW -- PASSIVE PART -----------------
        
        # E1 new function 
        self.E10      = 63239.2  #[Pa]
        self.VBstar  = 0.020
        self.s       = 10.18
        
        self.E2      = 395112.7  #[Pa]
        self.eta     = 1499087.2  #[Pa s]
        #
        
        #----------------- NEW -- ACTIVE PART -----------------
        self.gama    = 5.5e5  # parameter for TA [Pa/uV]
        self.NBath   = 0.81 #Activation threshold for efferent signal[uV]
        self.bn      = 3.5    # Sensitivity of efferent signal to afferent signal [-]
        
        self.k       = 30.0  #[1/uV/s]
        self.v       = 0.02  #[uV]
        self.m1      = 0.0019*self.fac_mmHg_Pa #[mmHg]
        self.m2      = 0.4 #[-]
        
    #Radius of bladder
    def RB(self,VB):
      a= (3*VB/(4*pi))**(1/3)
      return a



    #Circumferencial stretch of the bladder
    def lambdaB(self, VB): 
        a= (VB/self.VB0star)**(1/3)-1
        return a    
        
    def QB(self,TB,VB):
       if isinstance(self.PB(TB,VB),(list, tuple, set, np.ndarray)):
            a= self.alpa*(self.PB(TB,VB)-self.Pc) 
            a[a<0]=0
            return a
       else:
            return max(0,self.alpa*(self.PB(TB,VB)-self.Pc))
       
    def E1(self,VB):
        fexp = np.exp(-self.s*(VB-self.VBstar))
        fVB  = 1/(1+fexp)
        return self.E10 * fVB

    def dE1(self,VB): 
        fexp = np.exp(-self.s*(VB-self.VBstar))
        fVB  = 1/(1+fexp)
        dfVB = (-(1+fexp)**(-2)) * fexp * (-self.s)
        return self.E10 * dfVB 

    #Efferent signal
    def NBe(self,NBa):
      if isinstance(NBa,(list, tuple, set, np.ndarray)):
        a= self.bn * (NBa-self.NBath)
        a[a<0]=0
        return  a
      else:
        return max(0,self.bn * (NBa-self.NBath))
    
    def TA(self,VB,NBa):
        return self.gama * self.lambdaB(VB) * self.NBe(NBa) 

    def PB (self,TB,VB):
        PB = (2* TB * self.hB)/ self.RB(VB)
        return PB
    
    def ODE(self,y,t):

        VB = y[0] 
        TB = y[1] 
        NBa = y[2]
        
        a1 = -(self.E1(VB)/self.eta)
        a2 = - (self.E2/self.eta)
        a3 = (self.Q_in-self.QB(TB,VB))*self.dE1(VB)/self.E1(VB)
        
        a  = a1 + a2 + a3 
        
        
        
        b1 = (self.E1(VB)/(3*self.RB0*VB)) * (self.Q_in-self.QB(TB,VB)) *((VB/self.VB0star)**(-2/3))
        
        b2 = ((self.E1(VB)*self.E2)/(self.eta)) * ((VB/self.VB0star)**(1/3)-1);
        b  = b1 + b2
        
        c  =(self.E1(VB)/self.eta)
        
        ODE1 = self.Q_in - self.QB(TB,VB)
        ODE2 = a*TB + b + c* self.TA(VB,NBa)
        #ODE2=ODE2._value
        
        #print(k, NBa, ODE1, ODE2, PB, QB(PB) )
        ODE3 = self.k*NBa*(self.v*(self.PB(TB,VB)/self.m1)**self.m2-NBa)

        return [ODE1, ODE2, ODE3]

    def ODE_nn(self,y, t, model,x_train, window_size):
        VB = y[0] 
        TB = y[1] 
        NBa = y[2]
        
        a1 = -(self.E1(VB)/self.eta)
        a2 = - (self.E2/self.eta)
        a3 = (self.Q_in-self.QB(TB,VB))*self.dE1(VB)/self.E1(VB)
        
        a  = a1 + a2 + a3 
        
        
        
        b1 = (self.E1(VB)/(3*self.RB0*VB)) * (self.Q_in-self.QB(TB,VB)) *((VB/self.VB0star)**(-2/3))
        
        b2 = ((self.E1(VB)*self.E2)/(self.eta)) * ((VB/self.VB0star)**(1/3)-1);
        b  = b1 + b2
        
        c  =(self.E1(VB)/self.eta)
        
        ODE1 = self.Q_in - self.QB(TB,VB)
        ODE2 = a*TB + b + c* self.TA(VB,NBa)

        ODE3 = model.predict(x_train.reshape((1,window_size,3))) 
        #ODE3 = k*NBa*(v*(PB(VB,TB)/m1)**m2-NBa)
        return [ODE1, ODE2, ODE3] 

    def ODE_jac(self,y):
       
        self.s=y[3]
        self.alpa=y[4]
        self.bn =y[5]
        self.VBstar=y[6]
        self.eta =y[7]
        self.NBath=y[8]
        self.VB0star=y[9]

        VB = y[0] 
        TB = y[1] 
        NBa = y[2]
        
        a1 = -(self.E1(VB)/self.eta)
        a2 = - (self.E2/self.eta)
        a3 = (self.Q_in-self.QB(TB,VB))*self.dE1(VB)/self.E1(VB)
        
        a  = a1 + a2 + a3 
        
        
        
        b1 = (self.E1(VB)/(3*self.RB0*VB)) * (self.Q_in-self.QB(TB,VB)) *((VB/self.VB0star)**(-2/3))
        
        b2 = ((self.E1(VB)*self.E2)/(self.eta)) * ((VB/self.VB0star)**(1/3)-1);
        b  = b1 + b2
        
        c  =(self.E1(VB)/self.eta)
        
        ODE1 = self.Q_in - self.QB(TB,VB)
        ODE2 = a*TB + b + c* self.TA(VB,NBa)
        
        #print(k, NBa, ODE1, ODE2, PB, QB(PB) )
        ODE3 = self.k*NBa*(self.v*(self.PB(TB,VB)/self.m1)**self.m2-NBa)
        
        return np.array([ODE1, ODE2, ODE3]) 
        
    def jacobians(self,VB,TB,NBa,s,alpa,bn,VBstar,eta,NBath, VB0star):
        jac_x=jacobian(self.ODE_jac)
        res=jac_x(np.array([VB,TB,NBa,s,alpa,bn,VBstar,eta,NBath, VB0star],dtype=float))
        res_x=res[0:3,0:3]
        res_p=res[0:3,3:]
        return res_x,res_p
    

    
    def forward(self,VB0,TB0,NBa0,t):
       
       ODE0=[VB0,TB0,NBa0] 
       sol = odeint(self.ODE, ODE0, t)

       self.Vb = sol[-1, 0]
       self.Tb = sol[-1, 1]
       self.Nba = sol[-1, 2]
       
       return self.Vb, self.Tb, self.Nba

    def forward_rnn(self,model,xtrain,t,window_size):
        
        
        ODE0=xtrain[window_size-1,:].copy().reshape(3,) 
        xtrain[:,1]=(1/10000)*xtrain[:,1]
        sol = odeint(self.ODE_nn, ODE0, t,args=(model,xtrain, window_size))
        xtrain[:,1]=(10000)*xtrain[:,1]

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
npar=7
LUT_0=LUT()

jac_x,jac_p=LUT_0.jacobians(LUT_0.Vb,LUT_0.Tb,LUT_0.Nba,LUT_0.s,LUT_0.alpa,LUT_0.bn,LUT_0.VBstar,LUT_0.eta,LUT_0.NBath, LUT_0.VB0star )

[LUT_0.s,LUT_0.alpa,LUT_0.bn,LUT_0.VBstar,LUT_0.eta,LUT_0.NBath, LUT_0.VB0star ]=[LUT_0.s._value,LUT_0.alpa._value,LUT_0.bn._value,LUT_0.VBstar._value,LUT_0.eta._value,LUT_0.NBath._value, LUT_0.VB0star._value ]
ODE0=[LUT_0.Vb,LUT_0.Tb,LUT_0.Nba]
#samples_per_minute = 100
#t_final = 1500 #[min]
#t=np.linspace(0,t_final,t_final*samples_per_minute)
#iterations=t_final*samples_per_minute


#for i in range(iterations-1):
 #   t1=t[i:i+2]
     
  
    #LUT_0.forward(LUT_0.Vb,LUT_0.Tb,LUT_0.Nba,t1)  
    #z[i,:]=[LUT_0.Vb+0*np.random.normal(0, 1e-2, 1), LUT_0.Tb+ 0*np.random.normal(0, 1e-2, 1), LUT_0.Nba+ 0*np.random.normal(0, 1e-2, 1)]




V = spio.loadmat('V.mat', squeeze_me=True)
V=V['V']
P = spio.loadmat('P.mat', squeeze_me=True)
P=P['P']
t= spio.loadmat('t.mat', squeeze_me=True)
t=t['t']

iterations=len(t)
z=np.zeros((iterations-1,ny+npar))
T=np.zeros((iterations-1))
for i in range(iterations-1):
    T[i]=np.divide(P[i]*LUT_0.RB(V[i]),2*LUT_0.hB)
    T[i]=T[i]*133.322
    z[i,npar+0]=V[i]
    z[i,npar+1]=T[i]
    z[i,npar+2]=0.01
    z[i,0]=LUT_0.s
    z[i,1]=LUT_0.alpa
    z[i,2]=LUT_0.bn
    z[i,3]=LUT_0.VBstar
    z[i,4]=LUT_0.eta
    z[i,5]=LUT_0.NBath
    z[i,6]=LUT_0.VB0star 


z=z[9000:,:]
t=t[9000:]
t=t-t[0]
dt=t[1]
ztrain=z.copy()



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
par=np.array([LUT_0.s,LUT_0.alpa,LUT_0.bn,LUT_0.VBstar,LUT_0.eta,LUT_0.NBath, LUT_0.VB0star ])


for j,layer in enumerate(model.layers):
    weights=layer.get_weights()
    for ik in range(len(weights)):
        Wint=weights[ik].flatten()
        W.append(Wint)
        
W.append(par)    


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
nm=int(npar+nx*2/3)


#covariance matrices

#covariance matrices
# P=1e-2*np.identity(n)
# P[0:nw,0:nw]=1e0*np.identity(nw)
# Q=1e-1*np.identity(n)
# Q[0:nw,0:nw]=1e-2*np.identity(nw)


Pw=1e2*np.identity((nw-npar))
Px=1e-1*np.identity(nx+npar)

Qw=1e0*np.identity((nw-npar))
Qx=1e-2*np.identity(nx+npar)
Px[2,2]=1e3
Qx[2,2]=1e0

lamda=1e5
Rpar=1/lamda*np.identity(npar)

R=1e-10*np.identity(int(ny*2/3)*(batch_size))
R=block_diag(Rpar,R)



mse=np.zeros((iterations,nm))
epochs=6
mse_epochs=np.zeros((epochs,ny))

xtrain=np.zeros((batch_size,window_size,ny))
dfi_dw=np.zeros((nw-npar))
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
Hpar=np.identity(npar)
H2=block_diag(Hpar,H2)
for i in range(epochs):
    for o in range(batch_size):
        [LUTs[o].Vb,LUTs[o].Tb,LUTs[o].Nba]=z[int(t_batch[o,0]/dt),-ny:]
        x_hat[nw+ny*o:nw+ny*(o+1)]=np.reshape(z[int(t_batch[o,0]/dt),-ny:],(ny,1))
        [LUTs[o].s,LUTs[o].alpa,LUTs[o].bn,LUTs[o].VBstar,LUTs[o].eta,LUTs[o].NBath, LUTs[o].VB0star ]=x_hat[nw-npar:nw]
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
                mem=xtrain
       
       
           
        xtrain[:,:,1]=(1/10000)*xtrain[:,:,1]
        #print(sf)
        tic = time.time()
        #mse[index[0],:]=abs(z[index,:].reshape(nx,1)-x_hat[-nx:])
        
        
        
        
        for o in range(batch_size):
            
            gradients_w = sess.run(gradients_w_raw, feed_dict={model.input: xtrain[o,:,:].reshape(1,window_size,ny)})
            gradients_x = sess.run(gradients_x_raw, feed_dict={model.input: xtrain[o,:,:].reshape(1,window_size,ny)})
            gradients_x=gradients_x[0]
            
            jacob_x,jacob_p=LUTs[o].jacobians(x_hat[nw+ny*o][0],x_hat[nw+ny*o+1][0],x_hat[nw+ny*o+2][0],x_hat[nw-7],x_hat[nw-6],x_hat[nw-5],x_hat[nw-4],x_hat[nw-3],x_hat[nw-2],x_hat[nw-1])
            [LUTs[o].s,LUTs[o].alpa,LUTs[o].bn,LUTs[o].VBstar,LUTs[o].eta,LUTs[o].NBath, LUTs[o].VB0star ]=[LUTs[o].s._value,LUTs[o].alpa._value,LUTs[o].bn._value,LUTs[o].VBstar._value,LUTs[o].eta._value,LUTs[o].NBath._value, LUTs[o].VB0star._value ]
            

            k=0
            for j in range(len(gradients_w)):
                weights=gradients_w[j]
                weights=np.reshape(weights,(weights.size,))
                 
                dfi_dw[k:weights.size+k]=weights
                k=weights.size+k
                
            dVb_dx=np.array([1+dt*jacob_x[0,0],dt*jacob_x[0,1],dt*jacob_x[0,2]])    
            dTb_dx=np.array([dt*jacob_x[1,0],1+dt*jacob_x[1,1],dt*jacob_x[1,2]])
            dfi_dx=np.reshape(gradients_x[0,-1,:],(int(nx/batch_size),))
            dfi_dx[0]=dfi_dx[0]*dt
            dfi_dx[1]=dfi_dx[1]*dt
            dfi_dx[2]=dfi_dx[2]*dt +1
            a=np.block([np.zeros(nw-npar),jacob_p[0,:],dVb_dx,np.zeros(nx-int(nx/batch_size))])
            a=shift_elements(a, ny*o, 0)
            b=np.block([np.zeros(nw-npar),jacob_p[1,:],dTb_dx,np.zeros(nx-int(nx/batch_size))])
            b=shift_elements(b, ny*o, 0)
            c=np.block([dfi_dx,np.zeros(nx-int(nx/batch_size))])
            c=shift_elements(c, ny*o, 0)
            
            d=np.block([dt*dfi_dw,np.zeros(npar),c])
            
            F[nw+ny*o,:]=a
            F[nw+ny*o+1,:]=b
            F[nw+ny*o+2,:]=d
            if o ==0:
            # c=np.block([pends[o].dt*dfi_dw,b])
                
                C_kw=np.block([[np.block([np.zeros(nw-npar)])],[np.block([np.zeros(nw-npar)])],[np.block([dt*dfi_dw])]])
                
            else:
                
                C_kw=np.block([[C_kw],[np.block([np.zeros(nw-npar)])],[np.block([np.zeros(nw-npar)])],[np.block([dt*dfi_dw])]])
   
        C_kw=np.concatenate((np.zeros((npar,nw-npar)),C_kw))
        toc = time.time()
        #print("1",toc-tic)
        F3=F[nw-npar:,0:nw]
        F4=F[nw-npar:,nw-npar:]
        
        tic= time.time()
        Pw=Pw+Qw
        xtrain[:,:,1]=(10000)*xtrain[:,:,1]
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
         
        inter=x_hat[-(nx+npar):]
        meas=np.array([])
        meas=np.append(meas,inter[0:npar])
        for ik in range(batch_size):
              meas=np.append(meas,inter[[npar+ny*ik,npar+ny*ik+1]])
        meas=meas.reshape((nm,1))
        e1=z[index[0:-1],-ny:].reshape(nx,1)-x_hat[-(nx):]
        e1=np.block([z[0,:npar],e1[:,0]])
        
        zint=z[index[0:-1],npar:npar+2].reshape(nm-npar,1)
        zint=np.block([z[0,:npar],zint[:,0]])
        zint=zint.reshape((zint.shape[0],1))
        e=zint-meas
        x_=x_hat[-(nx+npar):].copy()
        x_hat[-(nx+npar):]=x_hat[-(nx+npar):]+ np.matmul(Kx,(e))
        Px= Px-np.matmul(Kx,np.matmul(C,Px))
       
        Sw=np.matmul(C_kw,np.matmul(Pw,C_kw.T)) + Qx
        Sinv=np.linalg.inv(Sw)
        P_int= np.matmul(C_kw.T,np.matmul(Sinv,C_kw))
        Pw=Pw-np.matmul(Pw,np.matmul(P_int,Pw))
        Kw=np.matmul(Pw,C_kw.T)
       
        e_=x_hat[-(nx+npar):]-x_
        x_hat[:nw-npar]=x_hat[:nw-npar]+ np.matmul(Kw,(e_))
        Pw=Pw+Qw
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
             [LUTs[o].s,LUTs[o].alpa,LUTs[o].bn,LUTs[o].VBstar,LUTs[o].eta,LUTs[o].NBath, LUTs[o].VB0star ]=x_hat[nw-npar:nw]
        
       
     
    
      
       
       
        k=npar
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
        print(x_hat[nw-npar:nw])
    
    
    #mse_epochs[i,:]=np.mean(abs(mse))
    #print(mse_epochs[i,:],Q[1,1])
    
    
    if (np.remainder(i,1)==0 and Qw[0,0] >=1e-1):
             Qw=1e-1*Qw
     #     R=1e-1*R
        
         

LUT_nn=LUT()
[Vb,Tb,Nba]=np.array([LUT_nn.Vb,LUT_nn.Tb,LUT_nn.Nba])
x_end=np.zeros((ny,iterations))
x_end[:,0]=[Vb,Tb,Nba]
xtrain=np.zeros((1,window_size,ny))
#xtrain=mem[0].reshape((1,mem[0].shape[0],mem[0].shape[1],))
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

     
        







 
 