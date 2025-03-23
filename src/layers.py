import numpy as np
import math
from .activations import ReLU, Sigmoid, Softmax, Tanh


ACTIVATION_MAP = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'softmax': Softmax,
}

class Layer():
    def __init__(self, num_neurons=1, prev_layer=None, activation:bool=False, init_weights:bool=False, activation_function:str=None, init_method:str=None):
        self.valid_activations = ["relu", "sigmoid", "softmax"]
        self.valid_intializations = ["glorot", "he"]
        self.num_neurons = num_neurons
        self.prev_layer = prev_layer
        self.biases = np.zeros((self.num_neurons, 1))
    
        if self.prev_layer:
            self.prev_layer.next_layer = self
        
        if activation:
            self.activation_function = activation_function
        
        if init_weights:
            self.init_method = init_method
        
    @property
    def activation_function(self):
        return self._activation_function
    
    @activation_function.setter
    def activation_function(self, activation_function:str) -> None:
        if activation_function is None:
            self._activation_function = ACTIVATION_MAP["relu"]()
        elif activation_function not in self.valid_activations:
            raise ValueError(f"Invalid activation function: {activation_function}. Please select from {self.valid_activations}")
        else:
            self._activation_function = ACTIVATION_MAP[activation_function]()
        return
    
    @property
    def next_layer(self):
        return self._next_layer
    
    @next_layer.setter
    def next_layer(self, layer):
        self._next_layer = layer

        if self.prev_layer is None:
            return
        else:
            self.initialize_weights()
        return

    def initialize_weights(self):
        fan_in = self.prev_layer.num_neurons
        fan_out = self.num_neurons
        

        if self._init_method == "glorot":
            self.glorot_init(fan_in, fan_out)
        
        elif self._init_method == "he":
            self.he_init(fan_in, fan_out)

        return
    
    def glorot_init(self, fan_in, fan_out):
        self.weights = np.random.randn(fan_in, fan_out)*math.sqrt(2/fan_in + fan_out)
        return
    
    def he_init(self, fan_in, fan_out):
        self.weights = np.random.randn(fan_in, fan_out)*math.sqrt(2/fan_in)
        return
        
    @property
    def init_method(self):
        return self._init_method
    
    @init_method.setter
    def init_method(self, init_method:str):
        if init_method is None:
            self._init_method = "glorot"
        elif init_method not in self.valid_intializations:
            raise ValueError(f"Invalid weight initialization method: {init_method}. Please select from {self.valid_intializations}")
        else:
            self._init_method = init_method
        return
    
    def linear(self) -> np.ndarray:
        self.A_prev = self.prev_layer.activate() # Storing previous layer's activation ouput for backprop
        self.Z = np.dot(self.A_prev, self.weights) + self.biases.T # Also storing the current layer's linear output (inputs*weights)
        return self.Z
    
    def activate(self) -> np.ndarray:
        Z = self.linear()

        return self._activation_function(Z)
    
    def backward(self, dA:np.ndarray) -> np.ndarray:

        if isinstance(self._activation_function, Softmax):
            raise RuntimeError("Softmax derivative must be calculated by cross-entropy loss function.")

        dZ = dA*self._activation_function.derivative(self.Z)
        m = dZ.shape[0]

        self.dW = self.A_prev.T@dZ/m
        self.db = np.mean(dZ, axis=0).reshape(-1, 1)

        dA_prev = dZ@self.weights.T

        return self.prev_layer.backward(dA_prev)

class InputLayer(Layer):
    def __init__(self, input_shape:np.ndarray):
        super().__init__(num_neurons=input_shape[0], activation=False, init_weights=False)
        self.inputs = None
        self.input_shape = input_shape
        
    def activate(self) -> np.ndarray:
        return self.inputs.reshape(self.inputs.shape[0], -1)
    
    def backward(self, dA):
        return dA
    

class HiddenLayer(Layer):
    def __init__(self, num_neurons, prev_layer, activation_function:str=None, init_method:str=None):
        super().__init__(num_neurons=num_neurons, prev_layer=prev_layer, activation=True, init_weights=True, activation_function=activation_function, init_method=init_method)
        
class OutputLayer(Layer):
    def __init__(self, num_neurons, prev_layer, activation_function:str=None, init_method:str=None):
        super().__init__(num_neurons=num_neurons, prev_layer=prev_layer, activation=True, init_weights=True, activation_function=activation_function, init_method=init_method)
        self.next_layer = None
    
    def backward(self, dA):
        
        if isinstance(self._activation_function, Softmax):
            dZ = dA
        else:
            dZ = dA*self._activation_function.derivative(self.Z)
        
        m = dZ.shape[0]

        self.dW = self.A_prev.T@dZ/m
        self.db = np.mean(dZ, axis=0).reshape(-1, 1)

        dA_prev = dZ@self.weights.T

        return self.prev_layer.backward(dA_prev)

class BatchNormLayer(Layer):
    def __init__(self, num_neurons=1, prev_layer=None, activation = False, init_weights = False, activation_function = None, init_method = None):
        super().__init__(num_neurons=num_neurons, prev_layer=prev_layer, activation=True, init_weights=True, activation_function=activation_function, init_method=init_method)
        self._gamma = np.ones(num_neurons,)
        self._beta = np.zeros(num_neurons,)
        self._running_mean = np.zeros(num_neurons,)
        self._running_var = np.ones(num_neurons,)
    
    def linear(self) -> np.ndarray:
        self.A_prev = self.prev_layer.activate() # Storing previous layer's activation ouput for backprop
        batch_size = self.A_prev.shape[0]
        self.Z = self.batch_norm(np.dot(self.A_prev, self.weights), batch_size)# Also storing the current layer's linear output (inputs*weights)

        return self.Z
    
    def batch_norm(self, Z, batch_size):
        eps = 1e-5
        self.avg = np.mean(Z, axis=0)
        self.var = np.var(Z, axis=0)

        Z_norm = (Z - self.avg)/np.sqrt(self.var + eps)

        return self._gamma*Z_norm + self._beta



