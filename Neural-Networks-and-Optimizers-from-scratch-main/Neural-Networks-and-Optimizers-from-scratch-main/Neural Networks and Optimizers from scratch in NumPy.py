#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow.keras.datasets.mnist as mnist

import sys
sys.path.append('../Neural_Network_from_Scratch')
import optimizers as opt

import matplotlib.pyplot as plt
import plotly.graph_objs as go


# In[2]:


# Load data from tensorflow
data = mnist.load_data()


# In[3]:


(x_train, y_train), (x_test, y_test) = data


# In[4]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[5]:


plt.imshow(x_train[0])


# In[6]:


# Preprocess data
# Reshape (flatten)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
print(x_train.shape, x_test.shape)

# Normalize data within {0,1} + dtype conversion
x_train = np.array(x_train/255., dtype=np.float32)
x_test = np.array(x_test/255., dtype=np.float32)


# In[7]:


# Function to convert labels to one-hot encodings
def one_hot(Y):
    num_labels = len(set(Y))
    new_Y = []
    for label in Y:
        encoding = np.zeros(num_labels)
        encoding[label] = 1.
        new_Y.append(encoding)
    return np.array(new_Y)


# In[8]:


y_train = one_hot(y_train)
y_test = one_hot(y_test)
y_train.shape, y_test.shape


# In[9]:


# Layer object to handle weights, biases, activation (if any) of a layer

class Layer:
    def __init__(self, hidden_units: int, activation:str=None):
        self.hidden_units = hidden_units
        self.activation = activation
        self.W = None
        self.b = None
        
    def initialize_params(self, n_in, hidden_units):
        # set seed for reproducibility 
        np.random.seed(42)
        self.W = np.random.randn(n_in, hidden_units) * np.sqrt(2/n_in) 
        np.random.seed(42)
        self.b = np.zeros((1, hidden_units))
    
    def forward(self, X):
        self.input = np.array(X, copy=True)
        if self.W is None:
            self.initialize_params(self.input.shape[-1], self.hidden_units)

        self.Z = X @ self.W + self.b
        
        if self.activation is not None:
            self.A = self.activation_fn(self.Z)
            return self.A
        return self.Z
    
    def activation_fn(self, z, derivative=False):
        if self.activation == 'relu':
            if derivative:
                return self.drelu(z)
            return self.relu(z)
        if self.activation == 'sigmoid':
            if derivative:
                return self.dsigmoid(z)
            return self.sigmoid(z)
        if self.activation == 'softmax':
            if derivative: 
                return self.dsoftmax(z)
            return self.softmax(z)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def dsigmoid(z):
        return Layer.sigmoid(z) * (1-Layer.sigmoid(z))

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def drelu(z):
        return np.where(z<=0,0,1)

    @staticmethod
    def softmax(x):  # numerically stable version of softmax 
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    @staticmethod
    def dsoftmax(x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True)) 
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))

    def __repr__(self):
        return str(f'''Hidden Units={self.hidden_units}; Activation={self.activation}''')


# In[1]:


class Model:
    def __init__(self):
        self.layers = dict()
        self.cache = dict()
        self.grads = dict()
        
    def add(self, layer):
        self.layers[len(self.layers)+1] = layer
        
    def set_config(self, epochs, learning_rate, optimizer=None):
        self.epochs = epochs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        if not not self.optimizer:
            self.optimizer.config(self.layers)
            self.optimizer.epochs = self.epochs
            self.optimizer.learning_rate = self.learning_rate

    def forward(self, x):
        for idx, layer in self.layers.items():
            x = layer.forward(x)
            self.cache[f'W{idx}'] = layer.W
            self.cache[f'Z{idx}'] = layer.Z
            self.cache[f'A{idx}'] = layer.A
        return x

    def backward(self, y):
        last_layer_idx = max(self.layers.keys())
        m = y.shape[0]
        # back prop through all dZs 
        for idx in reversed(range(1, last_layer_idx+1)):
            if idx == last_layer_idx:
                # e.g. dZ3 = y_pred - y_true for a 3 layer network 
                self.grads[f'dZ{idx}'] = self.cache[f'A{idx}'] - y
            else:
                # dZn = dZ(n+1) dot W(n+1) * inverse derivative of activation function of Layer n, with Zn as input 
                self.grads[f'dZ{idx}'] = self.grads[f'dZ{idx+1}'] @ self.cache[f'W{idx+1}'].T *                                        self.layers[idx].activation_fn(self.cache[f'Z{idx}'], derivative=True)
            self.grads[f'dW{idx}'] = 1 / m * self.layers[idx].input.T @ self.grads[f'dZ{idx}']
            self.grads[f'db{idx}'] = 1 / m * np.sum(self.grads[f'dZ{idx}'], axis=0, keepdims=True)
            
            assert self.grads[f'dW{idx}'].shape == self.cache[f'W{idx}'].shape

    def update_params(self, epoch_num, steps):
        for idx in self.layers.keys():
            if self.optimizer is None:
                self.optimize(idx)
            else:
                self.optimizer.optimize(idx, self.layers, self.grads, epoch_num, steps)                
 
    def optimize(self, idx):  
        # Vanilla minibatch gradient descent
        self.layers[idx].W -= self.learning_rate * self.grads[f'dW{idx}']
        self.layers[idx].b -= self.learning_rate * self.grads[f'db{idx}']
        
    def fit(self, x_train, y_train, x_test=None, y_test=None, batch_size=32):
        '''Training cycle of the model object'''
        losses = []
        train_accs = []
        val_accs = []

        for i in range(1, self.epochs+1):
            print(f'Epoch {i}')
            batches = self.create_batches(x_train, y_train, batch_size)
            epoch_loss = []
            steps = 0
            
            for x, y in batches:
                steps += 1
                preds = self.forward(x)
                loss = self.compute_loss(y, preds)
                epoch_loss.append(loss)

                # Backward propagation - calculation of gradients 
                self.backward(y)
                
                # update weights and biases of each layer
                self.update_params(i, steps)
                
            loss = sum(epoch_loss) / len(epoch_loss)
            losses.append(loss)

            # Predict with network on x_train
            train_preds = self.forward(x_train)
            c = np.argmax(train_preds, axis=1) == np.argmax(y_train, axis=1)
            train_acc = list(c).count(True) / len(c) * 100
            train_accs.append(train_acc)
            
            # Predict with network on x_test

            test_preds = self.forward(x_test)
            c = np.argmax(test_preds, axis=1) == np.argmax(y_test, axis=1)
            val_acc = list(c).count(True)/len(c) * 100
            val_accs.append(val_acc)

            print(f'Loss:{loss} Train Acc: {train_acc} Val Acc: {val_acc}')            
                
        self.history = {'train_loss': losses, 'train_acc': train_accs, 'val_acc': val_accs}

    @staticmethod
    def compute_loss(Y, Y_hat):
        m = Y.shape[0]
        L = -1./m * np.sum(Y * np.log(Y_hat))
        return L

    @staticmethod
    def mse(Y, Y_hat):
        m = Y.shape[0]
        return np.mean((Y- Y_hat)**2)
    
    @staticmethod
    def create_batches(x, y, batch_size):
        m = x.shape[0]
        num_batches = m / batch_size
        batches = []
        for i in range(int(num_batches+1)):
            batch_x = x[i*batch_size:(i+1)*batch_size]
            batch_y = y[i*batch_size:(i+1)*batch_size]
            batches.append((batch_x, batch_y))
        
        # without this, batch sizes that are perfectly divisible will create an 
        # empty array at index -1
        if m % batch_size == 0:
            batches.pop(-1)

        return batches
    
    def __repr__(self):
        return str(self.layers)


# In[11]:


epochs = 600
lr = 1e-3
batch_size = 128
lr_decay = False


# In[12]:


def run_model(optimizer=None):
    model = Model()
    model.add(Layer(200, activation='relu'))
    model.add(Layer(200, activation='relu'))
    model.add(Layer(10, activation='softmax'))
    model.set_config(epochs=epochs, learning_rate=lr, optimizer=optimizer)

    model.fit(x_train, y_train, x_test, y_test, batch_size=batch_size)
    return model.history


# In[15]:


# Demon stands for decaying momentum from this paper : https://arxiv.org/pdf/1910.04952v3.pdf
# Optimizers : refer to code in link below for implementation
# https://github.com/timvvvht/Neural-Networks-and-Optimizers-from-scratch/blob/main/optimizers.py

sgdm = opt.SGDM(lr, name='SGDM')
demonSGDM = opt.SGDM(lr, demon=True, name='DemonSGDM')
qhm = opt.QHM(lr, name='QHM')
nesterov = opt.Nesterov(lr, name='Nesterov')
demonNesterov = opt.Nesterov(lr, demon=True, name='DemonNesterov')

adagrad = opt.Adagrad(lr, name='Adagrad')
rmsprop = opt.RMSprop(lr, name='RMSprop')
adadelta = opt.Adadelta(lr, name='Adadelta')


# In[16]:


# Adam family
# Weight_decay is the implementation in the AdamW paper 
adam = opt.Adam(lr, name='Adam')
adamW = opt.Adam(lr, weight_decay=True, name='AdamW')
demonAdam = opt.Adam(lr, demon=True, name='DemonAdam')
demonAdamW = opt.Adam(lr, weight_decay=True, demon=True, name='DemonAdamW')

NAdam = opt.Nadam(lr, name='NAdam')
NAdamW = opt.Nadam(lr, weight_decay=True, name='NAdamW')
demonNAdam = opt.Nadam(lr, demon=True, name='DemonNAdam')
demonNAdamW = opt.Nadam(lr, weight_decay=True, demon=True, name='DemonNAdamW')

adamax = opt.Adamax(lr, name='Adamax')
adamaxW = opt.Adamax(lr, weight_decay=True, name='AdamaxW')
demonAdamax = opt.Adamax(lr, demon=True, name='DemonAdamax')
demonAdamaxW = opt.Adamax(lr, weight_decay=True, demon=True, name='DemonAdamaxW')

QHAdam = opt.QHAdam(lr, name='QHAdam')
QHAdamW = opt.QHAdam(lr, weight_decay=True, name='QHAdamW')
demonQHAdam = opt.QHAdam(lr, demon=True, name='DemonQHAdam')
demonQHAdamW = opt.QHAdam(lr, weight_decay=True, demon=True, name='DemonQHAdamW')


# In[17]:


opts1 = [sgdm, demonSGDM, qhm, nesterov, demonNesterov, adagrad, rmsprop, adadelta]


# In[18]:


opts2 = [adam, adamW, demonAdam, demonAdamW, NAdam, NAdamW, demonNAdam, demonNAdamW, 
         adamax, adamaxW, demonAdamax, demonAdamaxW, QHAdam, QHAdamW, demonQHAdam, demonQHAdamW]


opts3=[adam]
# In[23]:


#opts = opts1 + opts2
opts=opts3
opts


# In[24]:


opts_history = {i.name: run_model(i) for i in opts}


# In[25]:


opts_history['Gradient_Descent'] = (run_model(None))


# In[35]:


opts


# In[62]:


from plotly.subplots import make_subplots

num_epochs = [i for i in range(epochs)]
fig = go.Figure(layout_title_text="Loss")

lst = ['Gradient_Descent', 'SGDM','Nesterov', 'RMSprop', 'Adam', 'Adagrad', 'QHAdamW', 'NAdam']


for i in lst:
    fig.add_trace(go.Scatter(x=num_epochs, y=opts_history[i]['train_loss'], name=i))
    
fig.update_xaxes(title_text='Epochs')
fig.update_yaxes(title_text='Loss')
fig.show()


# In[64]:


fig = go.Figure(layout_title_text="Validation Accuracy")

for i in lst:
    fig.add_trace(go.Scatter(x=num_epochs, y=opts_history[i]['val_acc'], name=i))
    
fig.update_xaxes(title_text='Epochs')
fig.update_yaxes(title_text='Accuracy')
fig.show()


# In[ ]:




