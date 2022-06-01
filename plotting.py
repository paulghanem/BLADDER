import matplotlib.pyplot as plt
import numpy as np

from PhysiologicalComponent.Bladder import *

#Plots the given function along the solution to the ODE
def plot_vs_time(func_name, t, sol, constants):

  #Arrays of independent variables
  VBs = sol[:, 0]
  phiBs = sol[:, 1]
  NBas = sol[:, 2]

  implemented_funcs = ["VB","phiB","NBa","PB","TB","QB","lambdaB", "NBe", "NBas"]

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
    ax.plot(t/60, VBs*1e6)
    ax.set_ylabel('$V_B$ [ml]')

  #phiB vs. time
  elif func_name == "phiB":
    plt.plot(t/60, phiBs)
    plt.ylabel('$\phi_B$ [?]')

  #NBa vs. time
  elif func_name == "NBas":
    plt.plot(t/60, NBas)
    plt.ylabel('$N_{B,a}$ [uV]')

  #Pressure vs. time
  elif func_name == "PB":
    plt.plot(t/60, PB(VBs,phiBs, constants)*0.00750062)
    plt.set_ylabel('$P_B$ [mmHg]')

  #Tension vs. time
  elif func_name == "TB":
    plt.plot(t/60, TB(VBs,phiBs, constants)*0.00750062)
    plt.ylabel('$T_B$ [mmHg]')

  #QB vs. time
  elif func_name == "QB":
    plt.plot(t/60, QB(VBs,phiBs, constants)*1e6)
    plt.ylabel('$Q_B$ [ml/s]')

  #LambdaB vs. time
  elif func_name == "lambdaB":
    plt.plot(t/60, lambdaB(VBs, constants))
    plt.ylabel('$\lambda_B$ [-]')

  #NBe vs. time
  elif func_name == "NBe":
    plt.plot(t/60, NBe(NBas, constants))
    plt.ylabel('$N_{B,e}$ [-]')

  else:
    plt.plot(t/60, sol)

      
  #t-axis label
  plt.xlabel('time [min]')
  plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5)
  plt.minorticks_on()
  plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
  plt.tight_layout()
  plt.savefig('Figures/{}.png'.format(func_name), bbox_inches='tight', dpi=500) 
  plt.show()
  # plt.savefig("rk45_result.png")

