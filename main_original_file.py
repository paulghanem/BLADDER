#Library imports
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
import math
import pickle
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
import time
import sys
import argparse
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nns
from scipy.special import expit
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#Imports from our files
from LUT.LUT import LUT
from Cubature import cubaturetfv3 as cb
from Cubature.CubatureFiltertfv3 import CubatureFilter
from ODEComponent import *
from ODEComponent.BladderVolume import BladderVolume
from ODEComponent.BladderTension import BladderTension
from ODEComponent.BladderAfferent import BladderAfferent
from PhysiologicalComponent.PhysiologicalComponent import PhysiologicalComponent
from PhysiologicalComponent.Bladder import Bladder
from LUT.Integrator import Integrator
from plotting import *
from ANN import *
from read_data import *
from ANN.LearnableDenseBlock import LearnableDenseBlock
from utils import *
# from PhysiologicalComponent import functions
import datetime
simulation_date = str(datetime.datetime.now().time()).replace(":",".")
# if True:
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("p0_multiplier", type=float, default=1.0)
    parser.add_argument("q0_multiplier", type=float, default=0)
    parser.add_argument("mode", type=float, default=1000)
    args = parser.parse_args()
 
    ''' Specify which states will be learnable. If True, neural 
        networks will be learning otherwise the equations 
        provided by Prof. Guidoboni will be used'''
    is_learnable_array = np.array([False, False, False])
    
    is_learnable = is_learnable_array
    is_observable = np.invert(is_learnable)
    
    dataset_dir = "C:/Users/ahmet/OneDrive - Northeastern University/BLADDER/LUT_Data_Archive/NEU"
    
    experiment = "E011"
    session = "02"
    # dataset = read_session(experiment, session, dataset_dir)
    
    markers = ['v','s', 'o', '*','P']

    ''' Creates directories to save the plots if directories do not exist '''
    if not os.path.exists('Figures'):
        os.mkdir('Figures')

    ''' Starts from zero initial condition for learnable states '''
    zero_initial_conditions = False
    
    step_size = np.linspace(0,300*60,200000)[1]
    
    solution = np.load("ode_solution_200000.npy")[0:,:]
    solution[:,0] *= 1000
    #Initial conditions
    VB0  = solution[0,0]   # [m^3] [VD]
    phiB0 = solution[0,1]; #no prestress, no prestretch
    NBa0 = solution[0,2]; #no afferent signal
    initial_condition = {"Bladder" : 
                             {'Bladder Volume': VB0, 
                              'Bladder Tension': phiB0, 
                              'Bladder Afferent': NBa0}
                        }
    
    #Start initializing the model
    integrator_type = "euler"

    lut = LUT(step_size, initial_condition=initial_condition , integrator_type=integrator_type)
    
    bladder = Bladder(initial_condition = initial_condition)
    
    bladder_volume   = BladderVolume(bladder, initial_condition = VB0, is_learnable = {"BladderVolume":False,
                                                                                       'QB': False})
    bladder_tension  = BladderTension(bladder, initial_condition = phiB0,  is_learnable = {'TB': False,
                                                                                           'lambdaB': False,
                                                                                           'PB': False,
                                                                                           'LB': False,
                                                                                           'TA': False,
                                                                                           'RB': False})
    bladder_afferent = BladderAfferent(bladder, initial_condition = NBa0, is_learnable = {'NBe': True})

    
    bladder.ode_component_dict["Bladder Volume"] = bladder_volume
    bladder.ode_component_dict["Bladder Tension"] = bladder_tension
    bladder.ode_component_dict["Bladder Afferent"] = bladder_afferent
    
    lut.physiological_component_dict["Bladder"]  = bladder
    lut.get_num_of_learnables()
    s_dim = 3#number of latent states
    
    lut.compile(is_observable, s_dim, lut.dictionary_to_array(initial_condition))#creates learnable parameters and pointers from learnable parts
    
    lut.integrator.inputs = bladder.constants

    # solution = lut.integrator.rk45(init_cond = lut.dictionary_to_array(initial_condition))
    # np.save("ode_solution.npy", solution)
    # =============================================================================

    # Cubature Kalman filter training
    # =============================================================================
    lut.z_dim = lut.get_num_of_learnables() + lut.s_dim
    # x0 = tf.keras.backend.concatenate((lut.weights[0,:], lut.init_cond[0,:,0]))    #np.zeros((z_dim,))
    debug = True
    x0 = tf.keras.backend.concatenate((lut.weights[0,:], lut.dictionary_to_array(initial_condition)))
    # p0_multiplier = 1e-10
    # p0_multiplier = argsp0_multiplier
    lut.y_dim = 3
    P0 = np.eye(lut.z_dim)*args.p0_multiplier
    # if args.mode == 0:
    P0[-3:,-3:] *= 10**args.mode
    
    # P0[-lut.s_dim:,-lut.s_dim:] = np.diag(np.square(lut.dictionary_to_array(initial_condition)))
    P0 = tf.Variable(P0)
    Q0 = np.eye(lut.z_dim)*args.q0_multiplier #process noise
    constant = 0
    noise_pw  = 1e-5
    R0 = noise_pw * tf.keras.backend.eye(lut.y_dim)* constant
    Q0 = tf.Variable(Q0)
    nncbf = CubatureFilter(lut.f, lut.h, x0, P0, Q0, R0, lut.s_dim)
    N = len(solution)
    ydata= solution
    # run_model(lut, nncbf, N, solution, mode='train')

    print_progress = True
    mode = "train"
# def run_model(rnn, nncbf, N, ydata, mode='train', print_progress = True):
    save_e = tf.keras.backend.zeros((N+1, 1))
    save_w = tf.keras.backend.zeros((N, lut.get_num_of_learnables()))
    save_d = tf.keras.backend.zeros_like(lut.y_dim)
    save_e_cbf = np.zeros((N+1, lut.y_dim))
    save_d_cbf = np.zeros(save_e_cbf.shape)
    
    all_states_for_plotting = []

    iterations = N
    Qi = 0
    training_input = tf.keras.backend.tile(np.array(Qi)[np.newaxis][np.newaxis][np.newaxis], [1, lut.weights.shape[0],1])
    training_input = tf.keras.backend.expand_dims(training_input)
    training_input = tf.cast(training_input, dtype="float64")

    training_weights_1 = np.zeros((lut.weights.shape[1], iterations))
    all_states_for_plotting = np.zeros((lut.s_dim, iterations+1))
    for t in range(iterations):
        start = time.time()
        # nncbf.Q.assign(nncbf.Q.numpy() * 0)

        if t % N == 0:
            lut.init_cond = np.expand_dims(np.repeat(lut.dictionary_to_array(initial_condition).reshape(1,-1), 2*lut.z_dim, axis=0),axis=2)
            lut.init_cond[:,is_learnable,:] = 0
            all_states_for_plotting[:,t] = lut.init_cond[0,:, 0]
        xi = training_input
        xi = bladder.constants

        if mode=='train':
            yi_cbf, ei_cbf = nncbf.update(ydata[t], u=xi)
        elif mode== 'test':
            yi_cbf, ei_cbf = nncbf.test_forward(ydata[t],  u=xi)
        else:
            print("mode not found in options")

        all_states_for_plotting[:,t+1] = nncbf.x[-s_dim:]
        save_d_cbf[t] = yi_cbf
        save_e_cbf[t] = tf.keras.backend.abs(ei_cbf)
        
        training_weights_1[:, t] = lut.weights[0].numpy() #save the weights to analyze later
        if ((t+1) % 10 )== 0:
            if mode=='train':
                if sum(is_learnable) > 0:
                    plt.figure(figsize=(3,2))
                    plt.plot(training_weights_1.T)
                    plt.title('Weight 1')
                    plt.ylabel('Weights')
                    plt.xlabel('sample')
                    plt.show()
        plot_freq = 10
        if print_progress:
            print((time.time()-start))
            print(t, "/", N, 'rnn.latest_states ', nncbf.x[-s_dim:].numpy(), np.diag(nncbf.Q[-s_dim:]), is_learnable)
        if ((t) % plot_freq )== 0 or t==iterations-1:
            #%%
            markersize = 3
            fig, ax = plt.subplots(figsize = (10,4))
            unknown_state = np.asarray(all_states_for_plotting).T
            ax2 = ax.twinx()
            linewidth = 2
            ax.plot(ydata[0:, 0], label=r'$V_{Bladder}$ - Ground Truth', marker=markers[0], linewidth=linewidth, markevery=5000, color='yellowgreen', markersize = markersize)
            ax.plot(unknown_state[0:t,0], ':',linewidth=linewidth, label=r'$\hat{V}_{Bladder}$ - CKF',          marker=markers[0], markevery=5000, color='darkgreen', markersize = markersize)
            ax2.plot(ydata[0:, 1]*5, label=r'$\phi_B$ - Ground Truth', marker=markers[0], linewidth=linewidth, markevery=5000, color='orange', markersize = markersize)
            ax2.plot(unknown_state[0:t,1]*5, ':',linewidth=linewidth, label=r'$\hat{\phi}_{B}$ - CKF',          marker=markers[0], markevery=5000, color='red', markersize = markersize)
            ax2.plot(ydata[0:, 2], label=r'$N_{B,a}$ - Ground Truth', marker=markers[0], linewidth=linewidth, markevery=5000, color='lightblue', markersize = markersize)
            ax2.plot(unknown_state[0:t,2], ':',linewidth=linewidth, label=r'$\hat{N}_{B,a}$ - CKF',          marker=markers[0], markevery=5000, color='darkblue', markersize = markersize)

            ax.set_ylabel(r'$V_{B}$ [ml]')
            ax2.set_ylabel(r'$\phi_B*5$, $N_{B,a} [uV]$')
            plt.xticks(np.arange(0,len(solution) + len(solution)//10,len(solution)//10),np.arange(0,len(solution) + len(solution)//10,len(solution)//10))
            ax.set_xlabel('time (min)')
            ax.grid(visible=True, which='major', color='#999999', linestyle='-', alpha=0.5)
            ax.minorticks_on()
            ax.grid(visible=True, which='minor', color='#BBBBBB', linestyle='-', alpha=0.2)
            plt.xlim([0, t*20])
            fig.legend(ncol=3,loc='lower center', bbox_to_anchor=(0.5, -0.3))
            err = np.mean((ydata[0:t,:] - unknown_state[0:t,:])**2)
            plt.savefig('Figures/{}_model_err_{}_t_{}_p_{}_mode_{}_q_{}.png'.format(simulation_date,
                                                                "{:.7f}".format(err), 
                                                                str(t), 
                                                                str(args.p0_multiplier), 
                                                                str(args.mode), 
                                                                str(args.q0_multiplier)), bbox_inches='tight', dpi=500) 
            plt.show()
        # break
            #%%
    # np.save("hh_Figures_{}/Kalman_{}_errors_seq_length_{}_noise_{}_comp_{}_with_unknown_state_{}.npy".format(is_learnable_str, mode,input_sequence_length_before_compression, noise_pw, compress_ratio, now), save_e_cbf)
    # train_absolute_error = save_e_cbf
    # save_e_cbf = np.zeros((N, np.sum(is_observable)))
    # save_d_cbf = np.zeros_like(save_y)
    # return rnn, nncbf, save_e_cbf, save_d_cbf, unknown_state, ydata, save_y


    #Plots
    plot_vs_time("VB",   lut.integrator.t[-solution.shape[0]:], solution, bladder.constants)
    plot_vs_time("phiB", lut.integrator.t[-solution.shape[0]:], solution, bladder.constants)
    plot_vs_time("NBas", lut.integrator.t[-solution.shape[0]:], solution, bladder.constants)
    
    save_synthetic_data(lut.integrator.t[-solution.shape[0]:], solution)
    
    
    