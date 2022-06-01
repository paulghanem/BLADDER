import tensorflow as tf
import numpy as np
from scipy.integrate import odeint

class Integrator():
    """ 
        Integrator class holds all possible integrators we may implement in the future. 
            adder: 
            delta_time: 
            multiplier: 
            
        Contains methods:      
            euler_integration: Simple first order integrator
            rk45:  Runge-Kutta type of integrator     
    """
    
    def __init__(self, delta_time, integrator_type, **kwargs):
        """
        Parameters
        ----------
        delta_time : integration time step
        integrator_type : euler or
        """
        self.adder = tf.keras.layers.Add(input_shape=(1,))
        #Time range
        self.t = np.linspace(0,300*60,20000)
        self.delta_time = delta_time
        self.forward_function = None
        self.integrator_type = integrator_type
        if self.integrator_type == "lsoda":
            lsoda = None
        self.inputs = None


        super(Integrator, self).__init__(**kwargs)
    
    def euler_integration(self, current_output, previous_output):
        """
        Parameters
        ----------
        current_output : derivative found at the current step
        previous_output : integration result calculated at the previous step (state)

        Returns
        -------
        current_output : Current output is now integration result of the current step
        """
        for sub_key, ode_component in current_output.items():
            current_output[sub_key] = previous_output[sub_key] + current_output[sub_key]*self.delta_time#rk45 will only need the delta change, it will handle the integration without an integrator call at this line

        # current_output = self.multiplier([current_output, self.delta_time])
        # current_output = self.adder([current_output, previous_output])
        return current_output
    
    def lsoda_integration(self, current_output, previous_output):
        """
        Parameters
        ----------
        current_output : derivative found at the current step
        previous_output : integration result calculated at the previous step (state)

        Returns
        -------
        current_output : Current output is now integration result of the current step
        """
        for sub_key, ode_component in current_output.items():
            current_output[sub_key] = previous_output[sub_key] + current_output[sub_key]*self.delta_time#rk45 will only need the delta change, it will handle the integration without an integrator call at this line

        # current_output = self.multiplier([current_output, self.delta_time])
        # current_output = self.adder([current_output, previous_output])
        return current_output
    
    def rk45(self, init_cond):
        """
        Parameters
        ----------
        init_cond : rk45 type of integrator needs the initial condition

        Returns
        -------
        sol : output matrix for all states and all time steps
        """
        sol = odeint(self.forward_function, init_cond, self.t, atol=1e-13, rtol=[1e-6, 1e-10, 1e-6], mxstep=50000, printmessg = True)#function, initial condition, time
        return sol
