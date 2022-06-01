from ODEComponent.ODEComponent import ODEComponent
import numpy as np
from PhysiologicalComponent.Bladder import *
from ANN.LearnableDenseBlock import LearnableDenseBlock
import tensorflow as tf
from ANN.utils import soft_plus

debug = False
class BladderAfferent(ODEComponent):
    """ 
        BladderAfferent: 
            
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
        self.NBe_outs = []
        self.NBe_nn_outs = []
        super().__init__(physiological_component, initial_condition, self.is_learnable)

    def forward(self, inputs, states=None):
        """
            inputs: Inputs fed at time t
            states: Outputs at time t-1 are passed as states at time t
        """
        VB = states["Bladder Volume"]
        phiB = states["Bladder Tension"]
        NBa = states["Bladder Afferent"]


        # print("NBa ", NBa )
        constants = inputs
        ODE3 = constants['k']*NBa*(constants['v']*(self.physiological_component.ode_component_dict["Bladder Tension"].PB(VB,phiB, constants)/constants['m1'])**constants['m2']-NBa)#NBa
        # print("ODE3", ODE3)
        return ODE3
        
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
        if self.is_learnable['NBe']:
            self.NBe_nn = LearnableDenseBlock(lut, input_shape=1, output_shape=1, layer_count = 3)
            self.learnable_components.append(self.NBe_nn)
        
    #Efferent signal
    def NBe(self, NBa, constants):
        if self.is_learnable['NBe']:
            if debug:
                print("\n NBa.shape",NBa.shape)
            nn_inputs = tf.keras.backend.stack((NBa,),axis=-1)
            nn_out = self.NBe_nn.forward(nn_inputs)[:,0]
            a = np.maximum(0,constants['bn'] * (NBa-constants['NBath'])) 
            if debug:
                print("\n correct out NBe", a)
                print("\n out nn_out", nn_out)
            # nn_out = tf.keras.activations.sigmoid(nn_out)
            # nn_out = tf.keras.activations.softplus(nn_out)
            # nn_out = soft_plus(nn_out , stiffness = 100)
                print("\n out nn_out", nn_out)
            # nn_out = tf.keras.backend.clip(nn_out, min_value = 0, max_value=0.83776)
            self.NBe_outs.append(a)
            self.NBe_nn_outs.append(nn_out)
            return nn_out 
        else:
            if isinstance(NBa,(list, tuple, set, np.ndarray)):
                a = constants['bn'] * (NBa-constants['NBath'])
                a[a<0]=0
                return a
            else:
                return np.maximum(0,constants['bn'] * (NBa-constants['NBath']))
    
    
    
