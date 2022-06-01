import tensorflow as tf
from ODEComponent.ODEComponent import ODEComponent
from PhysiologicalComponent.Bladder import *
import numpy as np
from ANN.LearnableDenseBlock import LearnableDenseBlock
from ANN.utils import soft_plus

debug = False
class BladderVolume(ODEComponent):
    """ 
        BladderVolume: 
            
        build:     Built is used by Tensorflow to determine if layer is built properly
        call:      Inference function used by tensorflow when backpropagation is in use, calls the forward function 
        forward:   Inference function used by cubature Kalman filter and backpropagation updates
        get_param: Returns all trainable parameters so that cubature Kalman filters can calculate the next weights
        set_param: Sets all trainable parameters after cubature Kalman filter updates
    """
    def __init__(self, physiological_component, initial_condition, is_learnable):
        self.is_learnable = is_learnable
        self.num_of_learnables = 0
        self.learnable_components = []
        self.physiological_component = physiological_component
        self.lut = None
        super().__init__(physiological_component, initial_condition, self.is_learnable)

    def forward(self, inputs, states=None):
        """
            inputs: Inputs fed at time t
            states: Outputs at time t-1 are passed as states at time t
        """
        VB =   states["Bladder Volume"]
        phiB = states["Bladder Tension"]
        NBa =  states["Bladder Afferent"]

        constants = inputs
        if self.is_learnable['BladderVolume']:
            nn_inputs = tf.keras.backend.stack((VB, phiB),axis = 1)
            ODE1 = self.BladderVolumeForward.forward(nn_inputs)[:,0]
        else:
            ODE1 = constants['C_QK'] + QI() - self.QB(VB, phiB, constants) #VB # pass t
        return ODE1

    def get_param(self):
        " Returns all trainable parameters of the component"
        
    def set_params(self, params):
        " Sets all trainable parameters of the component"
        
    def get_num_of_learnables(self):
        self.num_of_learnables = 0
        for comp in self.learnable_components:
            self.num_of_learnables += comp.get_num_of_learnables()
        return self.num_of_learnables
    
    def compile(self, lut):
        self.lut = lut
        if self.is_learnable['BladderVolume']:
            self.BladderVolumeForward = LearnableDenseBlock(lut, input_shape=2, output_shape=1, layer_count = 1)
            self.learnable_components.append(self.BladderVolumeForward)
        if self.is_learnable['QB']:
            self.QB_nn = LearnableDenseBlock(lut, input_shape=2, output_shape=1, layer_count = 1)
            self.learnable_components.append(self.QB_nn)
        
    #Out-flow rate 
    def QB(self, VB, phiB, constants):
        if self.is_learnable['QB']:
            nn_inputs = tf.keras.backend.stack((VB, phiB),axis = 1)
            # nn_inputs = tf.expand_dims(nn_inputs, axis=1)
            nn_out = self.QB_nn.forward(nn_inputs)[:,0]
            if debug:
                print("\n \n \n model out", np.maximum(0, constants['alpa' ] * (self.physiological_component.ode_component_dict["Bladder Tension"].PB(VB,phiB, constants)-constants['Pc'])))
            # nn_out = tf.keras.backend.relu(nn_out)
            if debug:
                print("\n \n nn out", nn_out)
            # nn_out = soft_plus(nn_out, stiffness=10000)
            # print("\n \n nn out", nn_out)
            nn_out = tf.keras.backend.clip(nn_out, min_value = 0, max_value=11.06)
            return nn_out
        else:
            # if isinstance(VB,(list, tuple, set, np.ndarray)) | isinstance(phiB,(list, tuple, set, np.ndarray)):
            #     a=constants['alpa'] * (self.physiological_component["Bladder Tension"].PB(VB,phiB, constants)-constants['Pc'])
            #     a[a<0]=0
            #     return  a
            # else:
                
            return np.maximum(0, constants['alpa'] * (self.physiological_component.ode_component_dict["Bladder Tension"].PB(VB,phiB, constants)-constants['Pc']))
        
   

    
            
