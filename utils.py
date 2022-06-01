import tensorflow as tf
import numpy as np

def logit(x):
    """ Computes the logit function, i.e. the logistic sigmoid inverse. """
    return - tf.keras.backend.log(1. / x - 1.) + 0.5

def init_ones(shape, dtype=None):
    initializer = tf.keras.initializers.RandomUniform(minval = 1/shape[1]/2, maxval = 1/shape[1]*2)
    return initializer(shape)

def init_zero(shape, dtype=None):
    initializer = tf.keras.initializers.RandomUniform(minval = -0.0001, maxval = 0.0001)
    return initializer(shape)

def my_mse(y_true, y_pred):
    """
        Loss function can be modified by modifying this function
    """
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`

def occurrences(arr, element):
    """
        Returns list of occurrences of an element
    """
    temp = []
    for i in range(len(arr)):
        if arr[i] == element:
            temp.append(i)
    return temp

def save_synthetic_data(time, solution):
    np.save("Data/rk45_solution_t_VB_PhiB_Nba.npy", np.concatenate([np.expand_dims(time, axis=1), solution], axis = 1))
    print("Saved Synthetic Data")
    

# def clip(ydata, is_learnable):
#     for index, pressure in enumerate(occurrences(is_learnable, False)):
#         if not pressure == 0:# do not clip the membrane potential
#             ydata[:,index] = torch.clip(ydata[:,index],0,1)
            
#     return ydata



