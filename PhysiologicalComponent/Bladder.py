import numpy as np
from PhysiologicalComponent.PhysiologicalComponent import PhysiologicalComponent
import math 

class Bladder(PhysiologicalComponent):
    """
        Bladder: Extends PhysiologicalComponent.  

        build:     Built is used by Tensorflow to determine if layer is built properly
        call:      Inference function used by tensorflow when backpropagation is in use, calls the forward function 
        forward:   Inference function used by cubature Kalman filter and backpropagation updates
        get_param: Returns all trainable parameters so that cubature Kalman filters can calculate the next weights
        set_param: Sets all trainable parameters after cubature Kalman filter updates
    """
    def __init__(self, initial_condition):
        self.ode_component_dict = {}
        self.constants = self.set_constants()
        self.initial_condition = initial_condition
        super(Bladder).__init__()
        
    def forward(self, inputs, states=None):
        """
            inputs: Inputs fed at time t
            states: Outputs at time t-1 are passed as states at time t
        """
        outputs = {}
        for key, ode_component in self.ode_component_dict.items():
            outputs[key] = ode_component.forward(inputs, states)
        return outputs
    
    def get_param(self):
        """ Returns all trainable parameters of the component """
        trainable_vars = np.array([])
        for param in self.trainable_variables:
            trainable_vars = np.append(trainable_vars, param.numpy().flatten()[:])#1003
        return trainable_vars
    
    def get_num_of_learnables(self):
        self.num_of_learnables = 0
        for key, ode_component in self.ode_component_dict.items():
            self.num_of_learnables += ode_component.get_num_of_learnables()
        return self.num_of_learnables
    
    def set_params(self, params):
        """ Sets all trainable parameters of the component """
        starting_index = 0
        model_weights = self.get_weights()
        for param in range(len(self.trainable_variables)):
            model_weights[param] = params[starting_index:starting_index+self.trainable_variables[param].numpy().flatten().shape[0]].reshape(self.trainable_variables[param].shape)
            starting_index += self.trainable_variables[param].numpy().flatten().shape[0]
        self.set_weights(model_weights)
        
    def set_constants(self):
        # #Initial conditions
        VB0  = 25e-6  # [m^3] [VD]
        phiB0 = 0; #no prestress, no prestretch
        NBa0 = 0.01; #no afferent signal
        #Reference size of the bladder
        VBstar=8*VB0 #This amounts to doubling the initial radius
        RBstar=((3*VBstar)/(4*math.pi))**(1/3)
        LBstar=2*math.pi*RBstar
        
        #Conversion factors
        fac_cmH20_Pa = 98.0665  # 1 cmH20 = 98.0665 Pa
        fac_mmHg_Pa =133.322    # 1 mmHg  = 133.322 Pa
        
        #Fluid flow in/out of the bladder
        alpa = 1 * 1e-6 / fac_cmH20_Pa  # urethral resistance [m^3/(s Pa)] -- [GR]
        Pc   = 25 * fac_cmH20_Pa  # cutoff urethral pressure [Pa] -- [GR]
        C_QK = 5 * 1e-8  # kidney --> bladder flow rate [m^3/s] -- [B]
        #Ptheta = 30 * fac_cmH20_Pa  # cutoff ureters pressure [Pa] -- [B] (in-flow is constant in this implementation)
        
        #Abdominal pressure
        Pabd    = 0     #[Pa]
        
        #Bladder thickness
        b0      = 6e-3  #[m]
        #b1      = 1e-3   #[-]
        hB=b0 #Thickness is constant in this implementation
        
        #----------------- NEW -- PASSIVE PART -----------------
        r       = 0.3 
        LB0 = 2*math.pi*(3*VB0/(4*math.pi))**(1/3) #initial circumference of the bladder
        L1star  = r * LB0
        L2star  = (1-r) * LB0 
        #
        E1      = 3.5e4  #[Pa]
        E2      = 1e4  #[Pa]
        eta     = 1e-1  #[Pa s]
        #
        aT      = (L1star/L2star) * (eta/E1)  #[s]
        aL      = (LBstar/L2star) * eta       #[Pa s]
        cT      = 1 + (L1star/L2star) * (E2/E1)  #[-]
        cL      = (LBstar/L2star)*E2  #[Pa]
        
        #----------------- NEW -- ACTIVE PART -----------------
        TAstar  = 2e5 #[Pa]
        NBath   = 0.75 #[-]
        
        bn=1
        
        k       = 30  #[1/uV/s]
        v       = 0.02  #[uV]
        m1      = 0.0019*fac_mmHg_Pa #[mmHg]
        m2      = 0.4 #[-]

        self.constants = {"VBstar": VBstar,
                          "RBstar": RBstar,
                          "LBstar": LBstar,
                          "fac_cmH20_Pa": fac_cmH20_Pa,
                          "fac_mmHg_Pa" : fac_mmHg_Pa,
                          "alpa" : alpa,
                          "Pc" : Pc,
                          "C_QK" : C_QK,
                          "Pabd" : Pabd,
                          "b0" : b0,
                          "hB" : hB,
                          "r" : r,
                          "LB0" : LB0,
                          "L1star" : L1star,
                          "L2star" : L2star,
                          "E1" : E1,
                          "E2" : E2,
                          "eta" : eta,
                          "aT" : aT,
                          "aL" : aL,
                          "cT" : cT,
                          "cL" : cL,
                          "TAstar" : TAstar,
                          "NBath" : NBath,
                          "bn" : bn,
                          "k" : k,
                          "v" : v,
                          "m1" : m1,
                          "m2" : m2}
        return self.constants

#Infusion rate
def QI(t=None):
  return 0
