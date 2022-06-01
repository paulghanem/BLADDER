import tensorflow as tf
from ANN.LearnableBlock import LearnableBlock 
from ANN.utils import *

class LearnableDenseBlock(LearnableBlock):
    def __init__(self, rnn, input_shape, output_shape, weight_pointer=0, activation_function = None, layer_count=3):
        self.hidden_units = 5
        self.layers = []
        self.rnn = rnn
        self.activation_function = activation_function
        self.weight_pointer = weight_pointer
        for l in range(layer_count): #append all layers except first and last
        
            if layer_count == 1: #First layer
                layer_input_shape = input_shape
                layer_output_shape = output_shape
                layer_activation_function = self.activation_function
            elif l == 0: #First layer
                layer_input_shape = input_shape
                layer_output_shape = self.hidden_units
                layer_activation_function = self.activation_function
            elif l == layer_count - 1: #last layer
                layer_input_shape = self.hidden_units
                layer_output_shape = output_shape
                layer_activation_function = False
            else: #Intermediate layer
                layer_input_shape = self.hidden_units
                layer_output_shape = self.hidden_units
                layer_activation_function = self.activation_function
            self.layers.append(dense_layer(input_shape=layer_input_shape, output_shape=layer_output_shape, weight_pointer=self.weight_pointer, activation_function = layer_activation_function))
            self.weight_pointer += self.layers[-1].num_of_learnables
        self.num_of_learnables = sum([x.num_of_learnables for x in self.layers])

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
        for layer in self.layers:
            x = layer.forward(self.rnn, x)
        return x

    def get_layer_count(self):
        """
        Returns
        -------
        Number of layers in the given block.

        """
        return len(self.layers)
        
    def get_hidden_unit_count(self):
        """
        Returns
        -------
        Number of hidden units per layer in the given block.

        """
        return self.hidden_units
    
    def get_activation_function(self):
        """
        Returns
        -------
        Returns activation function of the intermediate layers

        """
        return self.activation_function

    def get_weight_pointer(self):
        """
        Returns
        -------
        Pointer on the array where all learnable parameters are stored

        """
        return self.weight_pointer
        
    def get_num_of_learnables(self):
        """
        Returns
        -------
        Number of learnable parameters in the given block

        """
        return self.num_of_learnables