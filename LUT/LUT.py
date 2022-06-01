import numpy as np
import tensorflow as tf
from LUT.Integrator import Integrator
from scipy.integrate import odeint

class LUT():
    """ 
        creates a structure containing the following variables that
        describe a LUT model:
            
        LUT is an RNN for keeping all states that are feedforward layers
        
    """
    def __init__(self, delta_time, initial_condition, integrator_type, **kwargs):
        
        # self.step_size = step_size
        # self.is_learnable = is_learnable
        self.physiological_component_dict = {}
        self.delta_time = delta_time
        self.initial_condition = initial_condition
        self.states = self.initial_condition
        self.integrator = Integrator(self.delta_time, integrator_type)
        self.weights = None
        self.num_of_learnables = 0
        self.s_dim = None
        self.z_dim = None
        if integrator_type == "rk45":
            self.integrator.forward_function = self.forward_rk45
            self.integrator.inputs = None
        super(LUT, self).__init__(**kwargs)
    
    def forward(self, inputs, states=None):
        """
        Parameters
        ----------
        inputs : Inputs can be constants or any other inputs to be passed to the system
        states : The output coming from the previous iteration. The state vector that all \ 
                 ODEs will be contributing is self.states. We update self.states \ 
                 when a new state is passed. 
        
        Returns
        -------
        outputs : 
            Output for all ODEs/neural networks
        list
            This output will be needed if backpropagation is used for training
        """
        outputs = {}
        states1=states
        self.update_states(states)
        
        # self.states = states
##        if states == None:
##            self.states = self.initial_condition
##            states = self.states
##            self.update_states(states)
##        else:
##            self.update_states(states)
        

            # print(key)

        # for key, physiological_component in self.physiological_component_dict.items():
        for key, physiological_component in self.physiological_component_dict.items():
            outputs[key] = physiological_component.forward(inputs, self.states[key])
            #integration is handled in the integration object
            if self.integrator.integrator_type == 'euler':
                outputs[key] = self.integrator.euler_integration(outputs[key],self.states[key])
            else:
                pass
        return outputs
    
    def forward_rk45(self, states, t):
        """
        Parameters
        ----------
        states : integration output at the previous iteration is passed by odeint.\
                 states is 
        t : integration time points 

        Returns
        -------
        out : output of the ODEs. 
        """
        self.states = self.update_states(states)# update the states and convert it to dictionary format
        out_dict = self.forward(self.integrator.inputs, self.states)
        out = self.dictionary_to_array(out_dict)
        return out
    def forward_lsoda(self,states,t):
        """
        Parameters
        ----------
        states : integration output at the previous iteration is passed by odeint.\
                 states is 
        t : integration time points 

        Returns
        -------
        out : output of the ODEs. 
        """
        
        out_dict = self.forward(self.integrator.inputs, states)
        out = self.dictionary_to_array(out_dict)
        return out
    
    def get_param(self):
        trainable_vars = np.array([])
        for param in self.trainable_variables:
            trainable_vars = np.append(trainable_vars, param.numpy().flatten()[:])#1003
        return trainable_vars

    def dictionary_to_array(self, out_dict):
        out = np.asarray(list(list(out_dict.values())[0].values())).T
        return out
    
    def update_states(self, states_arr):
        index = 0
        
        for phys_key, physiological_component in self.states.items():
            for ode_key, ode_component in self.states[phys_key].items():
                self.states[phys_key][ode_key] = states_arr[index]
                index += 1 

    def get_num_of_learnables(self):
        self.num_of_learnables = 0
        for key, physiological_component in self.physiological_component_dict.items():
            self.num_of_learnables += physiological_component.get_num_of_learnables()
        return self.num_of_learnables
    
    def compile(self, is_observable, s_dim, initial_condition):
        self.s_dim = s_dim
        for key_i, physiological_component in self.physiological_component_dict.items():
            for key_j, ode_component in physiological_component.ode_component_dict.items():
                ode_component.compile(self)
        self.num_of_learnables = self.get_num_of_learnables()
        self.weights = tf.keras.backend.random_normal((2*self.num_of_learnables + 2*len(is_observable), self.num_of_learnables), mean=0.0, stddev=0.0)

    def f(self, z,t ,input):
        """
            The f function is used by Cubature's update functions. 
            It forwards the model one step, using 
            concatenated weights and states contained in z.
        """
        if not len(z) == self.s_dim:
            self.weights = z[ :-self.s_dim]#weights
        s_k = z[ -self.s_dim:]  #states
        eps = 1e-5
        #s_k = s_k * tf.constant([[1.0 , -1, 1]], dtype='float64')#Force first and last states to be positive, second state to be negative
        #s_k = tf.nn.relu(s_k)
        #s_k = tf.keras.activations.softplus(s_k )
        #s_k = s_k * tf.constant([[1.0 , -1, 1]], dtype='float64')
        self.update_states(s_k)
        x_k = input 
        #s_k_1 = self.forward(x_k, s_k)
        sol=odeint(self.forward_lsoda,s_k, t, atol=1e-13,rtol=[1e-6,1e-10,1e-6],mxstep=5000)
        s_k_1_arr=sol[1,:]
        #s_k_1_arr = self.dictionary_to_array(s_k_1)
        #s_k_1_arr = s_k_1_arr * tf.constant([[1.0 , -1, 1]],dtype='float64')#Force first and last states to be positive, second state to be negative
        #s_k_1_arr = tf.nn.relu(s_k_1_arr - eps)+eps
        # s_k = tf.keras.activations.softplus(s_k )
        #s_k_1_arr = s_k_1_arr * tf.constant([[1.0 , -1, 1]],dtype='float64')
        # print("s_k_1", s_k_1)
        if not len(z) == self.s_dim:
            z = tf.keras.backend.concatenate((self.weights, s_k_1_arr), axis=1)
        else:
            z = s_k_1_arr
        return z
    
    def h(self, z):
        """ 
            h is the function that decides which states are observable, which states are not
        """
        
        return z[-self.s_dim:].numpy()
    
    # def run_model(self, nncbf, N, ydata, mode='train'):
        
        # observation_matrix = np.zeros((np.sum(is_observable),self.s_dim))
        # for index, idx in enumerate(occurrences(is_observable, True)):
        #     observation_matrix[index,idx] = 1 #observation matrix is the states that are not learnable
        # all_states = z[-4:]
        # observable_states = np.dot(observation_matrix,all_states)
        # return observable_states
