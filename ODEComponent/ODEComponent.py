from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

class ODEComponent(ABC):
    """ 
        ODEComponent: An abstract class, so some of the functions are implemented but some of them needs to be implemented by the extending class. Shouldnt be instantiated.
    
        build:     Built is used by Tensorflow to determine if layer is built properly
        call:      Inference function used by tensorflow when backpropagation is in use, calls the forward function 
        forward:   Inference function used by cubature Kalman filter and backpropagation updates
        get_param: Returns all trainable parameters so that cubature Kalman filters can calculate the next weights
        set_param: Sets all trainable parameters after cubature Kalman filter updates
    """
    def __init__(self, physiological_component, initial_condition, is_learnable):
        """
            Constructor for the ODEComponent
        """
        self.initial_condition = initial_condition
        self.is_learnable = is_learnable
        self.physiological_component = physiological_component
    
    @abstractmethod
    def forward(self, inputs, states=None):
        """
            inputs: Inputs fed at time t
            states: Outputs at time t-1 are passed as states at time t
        """

    @abstractmethod
    def get_param(self):
        " Returns all trainable parameters of the component"
    
    @abstractmethod
    def get_num_of_learnables(self):
        """ Returns number of trainable parameters of the component """

    @abstractmethod
    def set_params(self, params):
        " Sets all trainable parameters of the component"
    