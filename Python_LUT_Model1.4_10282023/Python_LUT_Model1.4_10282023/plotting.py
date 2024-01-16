import matplotlib.pyplot as plt
import numpy as np
from parameters import *
from functions import *
plt.close('all')

#Plots the given function along the solution to the ODE
def plot_vs_time(func_name, t, sol):

  #Arrays of independent variables
  VBs = sol[:, 0]
  PBs = sol[:, 1]
  NBas = sol[:, 2]

  implemented_funcs = ["VB","NBa","PB","TB","TA","QB","lambdaB", "NBe","E1"]

  if not (func_name in implemented_funcs):
    print("Invalid function")
    return
    
  #General settings
  plt.style.use('seaborn-poster')

  #Create figure and axis
  fig = plt.figure(figsize=(12, 4))
  ax = fig.add_subplot(1,1,1)

  if func_name == "VB":
    #Volume vs. time
    ax.plot(t, VBs)
    ax.set_ylabel('$V_B$ [ml]')

 
  #NBa vs. time
  if func_name == "NBa":
    plt.plot(t, NBas)
    plt.ylabel('$N_{B,a}$ [uV]')
    #plt.ylim([0, 2])

  #Pressure vs. time
  if func_name == "PB":
    ax.plot(t, PBs/fac_mmHg_Pa)
    ax.set_ylabel('$P_B$ [mmHg]')

  #Tension vs. time
  if func_name == "TB":
    plt.plot(t, TB(PBs,VBs)/fac_mmHg_Pa)
    plt.ylabel('$T_B$ [mmHg]')

  #QB vs. time
  if func_name == "QB":
    plt.plot(t, QB(PBs))
    plt.ylabel('$Q_B$ [ml/s]')

  #LambdaB vs. time
  if func_name == "lambdaB":
    plt.plot(t, lambdaB(VBs))
    plt.ylabel('$\lambda_B$ [-]')

  #NBe vs. time
  if func_name == "NBe":
    plt.plot(t, NBe(NBas))
    plt.ylabel('$N_{B,e}$ [-]')
   # plt.ylim([0, 4])

  #t-axis label
  plt.xlabel('time [sec]')
  #
  plt.tight_layout()
  plt.show()

