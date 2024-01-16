import math

#######################
#Parameters are chosen to model a rodent bladder
#######################

#######################
#UNITS
#######################

#Length: cm
#Volume: ml
#Time: min
#Pressure: Pa
#Voltage: muV

#This library contains all the constants on which the model depends.

#Mathematical constants
pi = math.pi

#Conversion factors
fac_cmH20_Pa = 98.0665  # 1 cmH20 = 98.0665 Pa
fac_mmHg_Pa = 133.322    # 1 mmHg  = 133.322 Pa

#----------------- Physiological quantities -----------------

#Reference size of the bladder
VBstar = 3e-3 #[ml]
RBstar = ((3*VBstar)/(4*pi))**(1/3)
LBstar = 2*pi*RBstar

#Fluid flow in/out of the bladder
Q_in = 0.0258 # inflow rate [ml/min] 

alpha = 0.5e-3 #urethral resistance [ml/(min Pa)]
Pc = 29 * fac_mmHg_Pa #23 * fac_cmH20_Pa [Pa]

#Abdominal pressure
Pabd =  0 #[Pa]

#Bladder thickness
hB = 6e-1 #[cm]

#----------------- PASSIVE PART -----------------
r = 0.3 
L1star = r * LBstar #[cm]
L2star = (1-r) * LBstar #[cm]

#Stiffness of E1
E10 = 0.5e2 #[Pa]
VBth = 5e-2 #[ml]
s = 15 #[1/ml]

#Cosntant stiffness spring
E2 = 1e3  #[Pa]

#Viscous parameter
eta =100 #[Pa min]

#------------------ ACTIVE PART -----------------

gamma = 6e3 #Reference value [Pa/uV]
NBath = 0.9 #Activation threshold for efferent signal[uV]
bn = 4 #Sensitivity of efferent signal to afferent signal [-]

#Parameters of the differential equation for NBa
k  = 30*60  #[uV^-1 min^-1]
v  = 0.02  #[uV]
m1 = 0.0019*fac_mmHg_Pa #[Pa]
m2 = 0.4 #[-]