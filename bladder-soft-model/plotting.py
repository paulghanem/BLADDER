import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from parameters import *
from functions import *

#Plots the given fuction along the solution to the ODE
def plot_vs_time(func_name, t, sol):

    #Arrays of state variables
    VBs = sol[:, 0]
    TBs = sol[:, 1]
    NBas = sol[:, 2]

    implemented_funcs = ["VB","TB","NBa","PB","TB","QB","lambdaB", "NBe"]

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

    #Pressure vs. time
    if func_name == "PB":
        ax.plot(t, PB(VBs,TBs)*0.00750062)
        ax.set_ylabel('$P_B$ [mmHg]')

    #Tension vs. time
    if func_name == "TB":
        plt.plot(t, TBs*0.00750062)
        plt.ylabel('$T_B$ [mmHg]')

    #QB vs. time
    if func_name == "QB":
        plt.plot(t, QB(VBs,TBs))
        plt.ylabel('$Q_B$ [ml/min]')

    #LambdaB vs. time
    if func_name == "lambdaB":
        plt.plot(t, lambdaB(VBs))
        plt.ylabel('$\lambda_B$ [-]')

    #NBe vs. time
    if func_name == "NBe":
        plt.plot(t, NBe(NBas))
        plt.ylabel('$N_{B,e}$ [$\mu V]')

    #t-axis label
    plt.xlabel('time [min]')
    #
    plt.tight_layout()
    plt.show()