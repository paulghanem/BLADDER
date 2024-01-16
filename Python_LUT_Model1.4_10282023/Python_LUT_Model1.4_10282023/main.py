import numpy as np
import math
from scipy.integrate import odeint
from parameters import *
from functions import *
from plotting import *
#from singular_point import *
from matplotlib import gridspec
#from post_proccesing import *
#Initial conditions
import matplotlib.pyplot as plt
plt.close('all')

ODE0=[VB0,PB0,NBa0]

#Time range


#integrate ODE
#sol = odeint(ODE, ODE0, t, atol=1e-13, rtol=[1e-6, 1e-10, 1e-6], mxstep=5000)


samples_per_minute = 100
t_final = 1700 #[sec] 25 min   
t=np.linspace(0,t_final,t_final*samples_per_minute)

#integrate ODE
sol = odeint(ODE, ODE0, t)

# postprocessing 
#post_proccesing
VBs = sol[:, 0]
PBs = sol[:, 1]
NBas = sol[:, 2]
  
  

implemented_funcs = ["VB","NBa","PB","TB","TA","QB","lambdaB", "NBe","E1"]

#if not (func_name in implemented_funcs):
#  print("Invalid function")
# return

#General settings
"""plt.style.use('seaborn-poster')

#Create a figure and axis
fig = plt.figure(figsize=(12, 4))
ax = plt.subplots(3,3)

 # if func_name == "VB":
  
    #Volume vs. time
#plt.subplot(3,3,1)
ax[0,0].plot(t/60, VBs*1e6)
ax[0,0].set_ylabel('$V_B$ [ml]')
ax[0,0].set_xlabel('$t$ [s]')

#Pressure vs. time
plt.subplot(3,3,2)
ax.plot(t/60, PBs/fac_mmHg_Pa)
ax.set_ylabel('$P_B$ [mmHg]')
ax.set_xlabel('$t$ [s]')

#flow vs. time
plt.subplot(3,3,3)
ax.plot(t/60, QB(VBs)*1e6)
ax.set_ylabel('$lambda_B$ [-]')
ax.set_xlabel('$t$ [s]')

#lamdbaB vs. time
plt.subplot(3,3,4)
ax.plot(t/60, lambdaB(VBs)*1e6)
ax.set_ylabel('$Q_B$ [ml/s]')
ax.set_xlabel('$t$ [s]')

#TA vs. time
plt.subplot(3,3,5)
ax.plot(t/60, TA(PBs,NBas)/fac_mmHg_Pa)
ax.set_ylabel('$T_A$ [mmHg]')
ax.set_xlabel('$t$ [s]')

#TB vs. time
plt.subplot(3,3,6)
ax.plot(t/60, TB(PBs)/fac_mmHg_Pa)
ax.set_ylabel('$T_B$ [mmHg]')
ax.set_xlabel('$t$ [s]')

#NBa vs. time
plt.subplot(3,3,7)
ax.plot(t/60, NBas)
ax.set_ylabel('$NBa$ [\mu V]')
ax.set_xlabel('$t$ [s]')

#NBe vs. time
plt.subplot(3,3,8)
ax.plot(t/60, NBe(NBas))
ax.set_ylabel('$NBe$ [\mu V]')
ax.set_xlabel('$t$ [s]')

#E1 vs. time
plt.subplot(3,3,9)
ax.plot(t/60, E1(VBs)/fac_mmHg_Pa)
ax.set_ylabel('$E1$ [mmHg]')
ax.set_xlabel('$t$ [s]')"""

#Plots
plot_vs_time("VB", t, sol)
plot_vs_time("PB", t, sol)
plot_vs_time("NBa", t, sol)
plot_vs_time("TB",t,sol)
