import math
import autograd.numpy as np
#Mathematical constants
pi=math.pi

#Conversion factors
fac_cmH20_Pa = 98.0665  # 1 cmH20 = 98.0665 Pa
fac_mmHg_Pa =133.322    # 1 mmHg  = 133.322 Pa

#Initial conditions
VB0  = 0.0498  # [ml] [VD]
PB0 = 728.59 #no prestress, no prestretch
NBa0 = 0.0098 #no afferent signal

# inflow rate 
Q_in    = 0.998/3600; #[ml/s]

#Abdominal pressure
Pabd    = 0.0

#Reference size of the bladder
VB0star= 0.0197 # [ml]
RB0 = np.power(((3*VB0star)/(4*pi)),1/3) # [cm]
LB0=2*pi*RB0 #[cm]
hB= RB0/20 #Bladder thickness [cm]






#Fluid flow in/out of the bladder
alpa = 1.8e-6  # urethral resistance [ml/(s Pa)] -- [GR]
Pc   = 19.53 * fac_mmHg_Pa  # cutoff urethral pressure [Pa] -- [GR]
#C_QK = 5 * 1e-8  # kidney --> bladder flow rate [m^3/s] -- [B]
#Ptheta = 30 * fac_cmH20_Pa  # cutoff ureters pressure [Pa] -- [B] (in-flow is constant in this implementation)

#Abdominal pressure
Pabd    = 0     #[Pa]

# time steps for euler method 

#----------------- NEW -- PASSIVE PART -----------------

# E1 new function 
E10      = 63239.2  #[Pa]
VBstar  = 0.020
s       = 10.18

E2      = 395112.7  #[Pa]
eta     = 1499087.2  #[Pa s]
#

#----------------- NEW -- ACTIVE PART -----------------
gama    = 5.5e5  # parameter for TA [Pa/uV]
NBath   = 0.81 #Activation threshold for efferent signal[uV]
bn      = 3.5    # Sensitivity of efferent signal to afferent signal [-]

k       = 30.0  #[1/uV/s]
v       = 0.02  #[uV]
m1      = 0.0019*fac_mmHg_Pa #[mmHg]
m2      = 0.4 #[-]