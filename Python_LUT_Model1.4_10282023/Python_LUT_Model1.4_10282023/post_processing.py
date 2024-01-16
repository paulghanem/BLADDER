import matplotlib.pyplot as plt
import numpy as np
from parameters import *
from functions import *

#Plots the given function along the solution to the ODE

  #Arrays of independent variables
  VBs = sol[:, 0]
  PBs = sol[:, 1]
  NBas = sol[:, 2]
  
  

  implemented_funcs = ["VB","NBa","PB","TB","TA","QB","lambdaB", "NBe","E1"]

  #if not (func_name in implemented_funcs):
  #  print("Invalid function")
   # return
    
  #General settings
  plt.style.use('seaborn-poster')

  #Create figure and axis
  fig = plt.figure(figsize=(12, 4))
  #ax = fig.add_subplot(3,3,1)

 # if func_name == "VB":
  
    #Volume vs. time
    plt.subplot(3,3,1)
    ax.plot(t, VBs*1e6)
    ax.set_ylabel('$V_B$ [ml]')
    ax.set_xlabel('$t$ [s]')
   
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
    ax.set_xlabel('$t$ [s]')
    
  """#phiB vs. time
  if func_name == "phiB":
    plt.plot(t/60, phiBs)
    plt.ylabel('$\phi_B$ [?]')

  #NBa vs. time
  if func_name == "NBas":
    plt.plot(t/60, NBas)
    plt.ylabel('$N_{B,a}$ [uV]')

  #Pressure vs. time
  if func_name == "PB":
    ax.plot(t/60, PB(VBs,phiBs)*0.00750062)
    ax.set_ylabel('$P_B$ [mmHg]')

  #Tension vs. time
  if func_name == "TB":
    plt.plot(t/60, TB(VBs,phiBs)*0.00750062)
    plt.ylabel('$T_B$ [mmHg]')

  #QB vs. time
  if func_name == "QB":
    plt.plot(t/60, QB(VBs,phiBs)*1e6)
    plt.ylabel('$Q_B$ [ml/s]')

  #LambdaB vs. time
  if func_name == "lambdaB":
    plt.plot(t/60, lambdaB(VBs))
    plt.ylabel('$\lambda_B$ [-]')

  #NBe vs. time
  if func_name == "NBe":
    plt.plot(t/60, NBe(NBas))
    plt.ylabel('$N_{B,e}$ [-]')

  #t-axis label
  plt.xlabel('time [min]')
  #
  plt.tight_layout()
  plt.show()
"""
