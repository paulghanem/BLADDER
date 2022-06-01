from abc import ABC, abstractmethod
from ANN.utils import *

import tensorflow as tf

class LearnableBlock(ABC):
    
    @abstractmethod
    def forward(self, x):
        """
        Parameters
        ----------
        x : input to the first layer
        
        Returns
        -------
        x : input to the first layer will be iteratively 
            feeded to all layers and output will be x too
        """
    @abstractmethod
    def get_layer_count(self):
        """
        Returns
        -------
        Number of layers in the given block.

        """
        
    @abstractmethod
    def get_hidden_unit_count(self):
        """
        Returns
        -------
        Number of hidden units per layer in the given block.

        """
        
    @abstractmethod
    def get_activation_function(self):
        """
        Returns
        -------
        Returns activation function of the intermediate layers

        """
        
    @abstractmethod
    def get_weight_pointer(self):
        """
        Returns
        -------
        Pointer on the array where all learnable parameters are stored

        """
    
    @abstractmethod
    def get_num_of_learnables(self):
        """
        Returns
        -------
        Number of learnable parameters in the given block

        """
        
    
        
