from ODEComponent.ODEComponent import ODEComponent
import tensorflow as tf
import numpy as np
from PhysiologicalComponent.Bladder import *
from ANN.LearnableDenseBlock import LearnableDenseBlock
debug = False
class BladderTension(ODEComponent):
    """ 
        BladderTension: 
            
        build:     Built is used by Tensorflow to determine if layer is built properly
        call:      Inference function used by tensorflow when backpropagation is in use, calls the forward function 
        forward:   Inference function used by cubature Kalman filter and backpropagation updates
        get_param: Returns all trainable parameters so that cubature Kalman filters can calculate the next weights
        set_param: Sets all trainable parameters after cubature Kalman filter updates
    """
    def __init__(self, physiological_component, initial_condition, is_learnable):
        self.is_learnable = is_learnable
        self.physiological_component = physiological_component
        self.num_of_learnables = 0
        self.learnable_components = []
        
        super().__init__(physiological_component, initial_condition, self.is_learnable)

    def forward(self, inputs, states=None):
        """
            inputs: Inputs fed at time t
            states: Outputs at time t-1 are passed as states at time t
        """
        VB = states["Bladder Volume"]
        phiB = states["Bladder Tension"]
        NBa = states["Bladder Afferent"]

        # print("\n \n \n phiB ", phiB )

        constants = inputs
        ODE2 = -constants['cT'] * self.TB(VB, phiB, constants) + constants['cL'] * self.lambdaB(VB, constants) + self.TA(VB,NBa,constants)#phiB
        # print("\n \n -constants['cT'] * self.TB(VB, phiB, constants) ", -constants['cT'] * self.TB(VB, phiB, constants) )
        # print("\n \n constants['cL'] * self.lambdaB(VB, constants) ", constants['cL'] * self.lambdaB(VB, constants) )#outputs the maximum
        # print("\n \n self.TA(VB,NBa,constants)",self.TA(VB,NBa,constants))
        # print("\n \n ODE2", ODE2)
        return ODE2

    def get_param(self):
        " Returns all trainable parameters of the component"
        # trainable_vars = np.array([])
        # for param in self.trainable_variables:
        #     trainable_vars = np.append(trainable_vars, param.numpy().flatten()[:])#1003
        # return trainable_vars

    def set_params(self, params):
        " Sets all trainable parameters of the component"
        # starting_index = 0
        # model_weights = self.get_weights()
        # for param in range(len(self.trainable_variables)):
        #     model_weights[param] = params[starting_index:starting_index+self.trainable_variables[param].numpy().flatten().shape[0]].reshape(self.trainable_variables[param].shape)
        #     starting_index += self.trainable_variables[param].numpy().flatten().shape[0]
        # self.set_weights(model_weights)
    
    def get_num_of_learnables(self):
        self.num_of_learnables = 0
        for comp in self.learnable_components:
            self.num_of_learnables += comp.get_num_of_learnables()
        return self.num_of_learnables
  

    def compile(self, lut):
        self.lut = lut
        if self.is_learnable['TB']:
            self.TB_nn = LearnableDenseBlock(lut, input_shape=2, output_shape=1, layer_count = 1)
            self.learnable_components.append(self.TB_nn)
        if self.is_learnable['lambdaB']:
            self.lambdaB_nn = LearnableDenseBlock(lut, input_shape=1, output_shape=1, layer_count = 1)
            self.learnable_components.append(self.lambdaB_nn)
        if self.is_learnable['PB']:
            self.PB_nn = LearnableDenseBlock(lut, input_shape=2, output_shape=1, layer_count = 1)
            self.learnable_components.append(self.PB_nn)
        if self.is_learnable['LB']:
            self.LB_nn = LearnableDenseBlock(lut, input_shape=1, output_shape=1, layer_count = 1)
            self.learnable_components.append(self.LB_nn)
        if self.is_learnable['TA']:
            self.TA_nn = LearnableDenseBlock(lut, input_shape=2, output_shape=1, layer_count = 1)
            self.learnable_components.append(self.TA_nn)
        if self.is_learnable['RB']:
            self.RB_nn = LearnableDenseBlock(lut, input_shape=1, output_shape=1, layer_count = 1)
            self.learnable_components.append(self.RB_nn)


    #Bladder tension
    def TB(self, VB, phiB, constants):
        if self.is_learnable['TB']:
            nn_inputs = tf.keras.backend.stack((VB,phiB),axis = 1)
            return self.TB_nn.forward(nn_inputs)[:,0]
        else:
            # if isinstance(VB,(list, tuple, set, np.ndarray)) | isinstance(phiB,(list, tuple, set, np.ndarray)):
                # a=(phiB + constants['aL'] * self.lambdaB(VB))/constants['aT']
                # a[a<0]=0
                # return a
            # else:
            return np.maximum(0,(phiB + constants['aL'] * self.lambdaB(VB, constants))/constants['aT'])
    
    #Circumferencial stretch of the bladder
    def lambdaB(self, VB, constants): 
        if self.is_learnable['lambdaB']:
            nn_inputs = tf.keras.backend.stack((VB,),axis = 1)
            return self.lambdaB_nn.forward(nn_inputs)[:,0]
        else:
            # if isinstance(VB,(list, tuple, set, np.ndarray)):
            #     a=self.RB(VB)/constants['RBstar']-1
            #     a[a<0]=0
            #     if debug:
            #         print("self.RB(VB, constants)/constants['RBstar']-1", a)
            #     return a
            # else:
            if debug:
                print("self.RB(VB, constants)/constants['RBstar']-1", self.RB(VB, constants)/constants['RBstar']-1)
            model_out = np.clip(np.maximum(0,self.RB(VB, constants)/constants['RBstar']-1)  ,a_min=0,a_max=0.355013)
            return model_out 
  
    #Bladder pressure 
    def PB(self, VB, phiB, constants):
        if self.is_learnable['PB']:
            nn_inputs = tf.keras.backend.stack((VB,phiB),axis = 1)
            nn_out = self.PB_nn.forward(nn_inputs)[:,0]
            if debug:
                print(" \n nmodel out", np.maximum(0,constants['Pabd'] + (2 * constants['hB'] * self.TB(VB,phiB, constants))/self.RB(VB, constants)))
                print(" \n nn_out ", nn_out)
            return nn_out 
        else:
            if isinstance(VB,(list, tuple, set, np.ndarray)) | isinstance(phiB,(list, tuple, set, np.ndarray)):
                a=constants['Pabd'] + np.divide(2 * constants['hB'] * self.TB(VB,phiB, constants),self.RB(VB, constants))
                a[a<0]=0
                return a
            else:
                return np.maximum(0,constants['Pabd'] + (2 * constants['hB'] * self.TB(VB,phiB, constants))/self.RB(VB, constants))
    
    #Circumference of the bladder
    def LB(self, VB, constants):
        if self.is_learnable['LB']:
            nn_inputs = tf.keras.backend.stack((VB,),axis = 1)
            return self.LB_nn.forward(nn_inputs)[:,0]
        else:
            return 2*math.pi*self.RB(VB, constants)
    
    #Active tension
    def TA(self, VB,NBa, constants):
        if self.is_learnable['TA']:
            nn_inputs = tf.keras.backend.stack((VB,NBa),axis = 1)
            nn_out = self.TA_nn.forward(nn_inputs)[:,0]
            self.physiological_component.ode_component_dict["Bladder Afferent"].NBe_outs.append(self.physiological_component.ode_component_dict["Bladder Afferent"].NBe(NBa, constants))
            self.physiological_component.ode_component_dict["Bladder Afferent"].NBe_nn_outs.append(nn_out)
            if debug:
                print(" \n nn_out ", nn_out)
                
                print(" \n model_out ", constants['TAstar'] * self.lambdaB(VB, constants) * self.physiological_component.ode_component_dict["Bladder Afferent"].NBe(NBa, constants)) 

            return nn_out
        else:
            ta = constants['TAstar'] * self.lambdaB(VB, constants) * self.physiological_component.ode_component_dict["Bladder Afferent"].NBe(NBa, constants) 
            if debug:
                print("\n TA", ta)
            return ta

    #Bladder radius
    def RB(self, VB, constants):
        if self.is_learnable['RB']:
            nn_inputs = tf.keras.backend.stack((VB,),axis = 1)
            return self.RB_nn.forward(nn_inputs)[:,0]
        else:
            if isinstance(VB,(list, tuple, set, np.ndarray)):
                return np.cbrt(3*VB/(4*math.pi))
            else:
                return np.cbrt(3*VB/(4*math.pi))
                

