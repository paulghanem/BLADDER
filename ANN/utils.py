import tensorflow as tf

class dense_layer():
    def __init__(self, input_shape, output_shape, weight_pointer=0, activation_function = None):
        self.w_indices    = [weight_pointer, weight_pointer+ input_shape * output_shape ]
        self.bias_indices = [weight_pointer+ input_shape * output_shape, weight_pointer+ input_shape * output_shape + output_shape]
        self.num_of_learnables = (input_shape + 1) * output_shape
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation_function = activation_function
                
    def forward(self, rnn, input):
        output = tf.keras.backend.batch_dot(tf.keras.backend.reshape(rnn.weights[:,self.w_indices[0]:self.w_indices[1]], shape = (rnn.weights.shape[0], self.output_shape, self.input_shape)), input) 
        output = output + rnn.weights[:, self.bias_indices[0]:self.bias_indices[1]]
        if self.activation_function:
            return tf.keras.backend.relu(output, alpha=0.1)
        return output
    
    
def soft_plus(value, stiffness):
    return tf.math.log(1+tf.math.exp(value*stiffness))/stiffness