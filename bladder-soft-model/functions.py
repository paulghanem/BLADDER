
import autograd.numpy as np
import autograd
from autograd import grad, jacobian
from scipy.integrate import odeint
import tensorflow as tf 
#Local Imports
from parameters import *

def Softplus(x,beta): 
    return 1*(np.log(1 + np.exp(beta*x)) )

#Bladder stretch
def lambdaB(VB):
    lambdaB = RB(VB)/RBstar-1
    return lambdaB

#Derivative of lambdaB w.r.t VB
def dlambdaBdVB(VB):
    dlambdadVB = np.where(RB(VB)/RBstar-1<0, 0, 1/(3*np.power(VBstar*(VB**2),1/3)))
    return dlambdadVB 

#Radius of bladder
def RB(VB):
    RB = np.power(3*VB/(4*pi),1/3)
    return RB

#Circumference of the bladder
def LB(VB):
    LB = 2*pi*RB(VB)
    return LB

#Circumference of the bladder
def E1(lambdaB):
    E1 = E10/(1+np.exp(-s*(VBstar*(lambdaB+1)**3-VBth)))
    return E1

#Bladder Pressure
def PB(VB,TB):
    PB = Pabd + np.divide(2*hB*TB,RB(VB))
    return PB

#Bladder outflow
def QB(VB,TB):
    QB=np.where(PB(VB,TB)-Pc<0, 0, alpha*(PB(VB,TB)-Pc))
    #QB=alpha*Softplus(PB(VB,TB)-Pc,15)
    return QB

#Efferent signal
def NBe(NBa):
    NBe=np.where(NBa-NBath<0, 0, bn*(NBa-NBath))
    #NBe=bn*Softplus(NBa-NBath,15)
    return NBe

#Active Tension
def TA(VB,NBa):
    TA = gamma*lambdaB(VB)*NBe(NBa)
    return TA

#Vector field
def ODE(y, t):

    VB = y[0] 
    TB = y[1] 
    NBa = y[2]
    
    #Combination of parameters to simplify the presentation
    a1 = -(E1(lambdaB(VB))/eta) * (L2star/L1star)
    a2 = -(E2/eta)
    a3 = (Q_in-QB(VB,TB))/E1(lambdaB(VB))
    a  = a1 + a2 + a3

    b1 = ((E1(lambdaB(VB))*LBstar)/(3*VBstar*L1star)) * (Q_in-QB(VB,TB)) * (VB/VBstar)**(-2/3)
    b2 = ((E1(lambdaB(VB))*E2*LBstar)/(eta*L1star)) * ((VB/VBstar)**(1/3)-1)
    b  = b1 + b2
    
    c  = (L2star/L1star)*(E1(lambdaB(VB))/eta)

    ODE1 = Q_in - QB(VB,TB) 
# =============================================================================
    ODE2 = a*TB+b*lambdaB(VB)+c*TA(VB,NBa)      
# =============================================================================
    ODE3 = k*NBa*(v*(PB(VB,TB)/m1)**m2-NBa)
    
    return [ODE1, ODE2, ODE3] 


def ODE_nn(y, t, model,x_train, window_size):
    VB = y[0] 
    TB = y[1] 
    NBa = y[2]
    
    ##Combination of parameters to simplify the presentation
    a1 = -(E1(lambdaB(VB))/eta) * (L2star/L1star)
    a2 = -(E2/eta)
    a3 = (Q_in-QB(VB,TB))/E1(lambdaB(VB))
    a  = a1 + a2 + a3

    b1 = ((E1(lambdaB(VB))*LBstar)/(3*VBstar*L1star)) * (Q_in-QB(VB,TB)) * (VB/VBstar)**(-2/3)
    b2 = ((E1(lambdaB(VB))*E2*LBstar)/(eta*L1star)) * ((VB/VBstar)**(1/3)-1)
    b  = b1 + b2
    
    c  = (L2star/L1star)*(E1(lambdaB(VB))/eta)

    ODE1 = Q_in - QB(VB,TB)  
    ODE2 = a*TB+b*lambdaB(VB)+c*TA(VB,NBa)    

    ODE3 = model.predict(x_train.reshape((1,window_size,3))) 
    #ODE3 = k*NBa*(v*(PB(VB,TB)/m1)**m2-NBa)
    return [ODE1, ODE2, ODE3] 

def ODE_jac(y):

   VB = y[0] 
   TB = y[1] 
   NBa = y[2]
   
   #Combination of parameters to simplify the presentation
   a1 = -(E1(lambdaB(VB))/eta) * (L2star/L1star)
   a2 = -(E2/eta)
   a3 = (Q_in-QB(VB,TB))/E1(lambdaB(VB))
   a  = a1 + a2 + a3

   b1 = ((E1(lambdaB(VB))*LBstar)/(3*VBstar*L1star)) * (Q_in-QB(VB,TB)) * (VB/VBstar)**(-2/3)
   b2 = ((E1(lambdaB(VB))*E2*LBstar)/(eta*L1star)) * ((VB/VBstar)**(1/3)-1)
   b  = b1 + b2
   
   c  = (L2star/L1star)*(E1(lambdaB(VB))/eta)

   ODE1 = Q_in - QB(VB,TB) 
   ODE2 = a*TB+b*lambdaB(VB)+c*TA(VB,NBa)      
   ODE3 = k*NBa*(v*(PB(VB,TB)/m1)**m2-NBa)
   
    
   return np.array([ODE1, ODE2, ODE3]) 


def jacobians(VB,TB,NBa):
    jac=jacobian(ODE_jac)
    res=jac(np.array([VB,TB,NBa],dtype=float))
    return res

VB=.1
TB=500
NBa=1

a=jacobians(VB,TB,NBa)


