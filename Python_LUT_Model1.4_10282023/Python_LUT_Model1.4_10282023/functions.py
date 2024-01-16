import autograd.numpy as np
import autograd
from autograd import grad, jacobian
from scipy.integrate import odeint
import tensorflow as tf 
#Local Imports
from parameters import *

#Radius of bladder
def RB(VB):
  a= np.power(3*VB/(4*pi),1/3)
  return a



#Circumferencial stretch of the bladder
def lambdaB(VB): 
    a= np.power((VB/VB0star),1/3)-1
    return a

#Bladder tension
#def TB(VB,phiB):
 # if isinstance(VB,(list, tuple, set, np.ndarray)) | isinstance(phiB,(list, tuple, set, np.ndarray)):
 #   a=(phiB + aL * lambdaB(VB))/aT
 #   a[a<0]=0
 #   return a
 # else:
 #   return max(0,(phiB + aL * lambdaB(VB))/aT)

#Bladder pressure 
#def PB(VB,phiB):
#  if isinstance(VB,(list, tuple, set, np.ndarray)) | isinstance(phiB,(list, tuple, set, np.ndarray)):
#    a=Pabd + np.divide(2 * hB * TB(VB,phiB),RB(VB))
#    a[a<0]=0
#    return a
#  else:
#    return max(0,Pabd + (2 * hB * TB(VB,phiB))/RB(VB))

#Out-flow rate 
def QB(PB):
   if isinstance(PB,(list, tuple, set, np.ndarray)):
        a= alpa*(PB-Pc) 
        a[a<0]=0
        return a
   else:
        return max(0,alpa*(PB-Pc))


'''def QB(PB):
    if PB > Pc:
        a= alpa*(PB-Pc)
        return a
    else:
        a= 0
        return a'''
# non linear elastance E1 and its derivative
def E1(VB):
    fexp = np.exp(-s*(VB-VBstar))
    fVB  = 1/(1+fexp)
    return E10 * fVB

def dE1(VB): 
    fexp = np.exp(-s*(VB-VBstar))
    fVB  = 1/(1+fexp)
    dfVB = (-(1+fexp)**(-2)) * fexp * (-s)
    return E10 * dfVB 

#Efferent signal
def NBe(NBa):
  if isinstance(NBa,(list, tuple, set, np.ndarray)):
    a= bn * (NBa-NBath)
    a[a<0]=0
    return  a
  else:
    return max(0,bn * (NBa-NBath))


#Active tension
def TA(VB,NBa):
  return gama * lambdaB(VB) * NBe(NBa) 

def TB (PB,VB):
    TB = (PB * RB(VB))/(2 *hB)
    return TB
    
#Infusion rate
#def QI(t):
#  return 0

#ODE
def ODE(y,t):

    VB = y[0] 
    PB = y[1] 
    NBa = y[2]
    
    a1 = -(E1(VB)/eta)
    a2 = - (E2/eta)
    a3 = (Q_in-QB(PB))*dE1(VB)/E1(VB)
    a4 = -(Q_in-QB(PB))/(3*VB)
    a  = a1 + a2 + a3 + a4
    
    b1 = ((2*hB*E1(VB))/(3*RB0*VB)) * (Q_in-QB(PB))
    b2 = ((2*hB*E1(VB)*E2)/(eta*RB(VB))) * (np.power((VB/VB0star),1/3)-1)
    b  = b1 + b2
    
    c  =((2*hB*E1(VB)*TA(VB,NBa))/(eta*RB(VB)))
    
    ODE1 = Q_in - QB(PB)
    ODE2 = a*PB + b + c
    
    #print(k, NBa, ODE1, ODE2, PB, QB(PB) )
    ODE3 = k*NBa*(v*(PB/m1)**m2-NBa)

    return [ODE1, ODE2, ODE3]

def ODE_nn(y, t, model,x_train, window_size):

    VB = y[0] 
    PB = y[1] 
    NBa = y[2]
    
    a1 = -(E1(VB)/eta)
    a2 = - (E2/eta)
    a3 = (Q_in-QB(PB))*dE1(VB)/E1(VB)
    a4 = -(Q_in-QB(PB))/(3*VB)
    a  = a1 + a2 + a3 + a4
    
    b1 = ((2*hB*E1(VB))/(3*RB0*VB)) * (Q_in-QB(PB))
    b2 = ((2*hB*E1(VB)*E2)/(eta*RB(VB))) * (np.power((VB/VB0star),1/3)-1)
    b  = b1 + b2
    
    c  =((2*hB*E1(VB)*TA(VB,NBa))/(eta*RB(VB)))
    
    ODE1 = Q_in - QB(PB)
    ODE2 = a*PB + b + c
    
    #print(k, NBa, ODE1, ODE2, PB, QB(PB) )
    ODE3 = model.predict(x_train.reshape((1,window_size,3))) 

    return [ODE1, ODE2, ODE3]
    
def ODE_jac(y):

    VB = y[0] 
    PB = y[1] 
    NBa = y[2]
    
    a1 = -(E1(VB)/eta)
    a2 = - (E2/eta)
    a3 = (Q_in-QB(PB))*dE1(VB)/E1(VB)
    a4 = -(Q_in-QB(PB))/(3*VB)
    a  = a1 + a2 + a3 + a4
    
    b1 = ((2*hB*E1(VB))/(3*RB0*VB)) * (Q_in-QB(PB))
    b2 = ((2*hB*E1(VB)*E2)/(eta*RB(VB))) * (np.power((VB/VB0star),1/3)-1)
    b  = b1 + b2
    
    c  =((2*hB*E1(VB)*TA(VB,NBa))/(eta*RB(VB)))
    
    ODE1 = Q_in - QB(PB)
    ODE2 = a*PB + b + c
    
    #print(k, NBa, ODE1, ODE2, PB, QB(PB) )
    ODE3 = k*NBa*(v*(PB/m1)**m2-NBa)

    return np.array([ODE1, ODE2, ODE3]) 

def jacobians(VB,PB,NBa):
    jac=jacobian(ODE_jac)
    res=jac(np.array([VB,PB,NBa],dtype=float))
    return res

