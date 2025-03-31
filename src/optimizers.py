import numpy as np

class Optimizer():
    def update_weights(self, layer):
        raise NotImplementedError("Weight updates must be implemented by subclasses")

class SGD(Optimizer):
    def __init__(self, eta):
        self.eta = eta
        
    def update_weights(self, layer):
        if hasattr(layer, "weights") and hasattr(layer, "dW"):
            layer.weights -= self.eta * layer.dW
            layer.biases -= self.eta * layer.db
            
        elif hasattr(layer, "_gamma") and hasattr(layer, "_beta"):
            layer._gamma -= self.eta*layer.dGamma
            layer._beta -= self.eta*layer.dBeta

        return
    
class Momentum(Optimizer):
    def __init__(self, eta, beta):
        self.beta = beta
        self.eta = eta
        self.momentum_vector = {}

    def update_weights(self, layer):
        layer_id = id(layer)
        
        if layer_id not in self.momentum_vector:
            if hasattr(layer, "weights") and hasattr(layer, "biases"):
                self.momentum_vector[layer_id] = { 
                    "mW": np.zeros_like(layer.weights),
                    "mb": np.zeros_like(layer.biases)  
                }

            if hasattr(layer, "_gamma") and hasattr(layer, "_beta"):
                self.momentum_vector[layer_id].update({
                    "mGamma": np.zeros_like(layer._gamma),
                    "mBeta": np.zeros_like(layer._beta)
                })

         
        
        if hasattr(layer, "weights") and hasattr(layer, "dW"):
            self.momentum_vector[layer_id]["mW"] = self.beta*self.momentum_vector[layer_id]["mW"] - self.eta*layer.dW
            self.momentum_vector[layer_id]["mb"] = self.beta*self.momentum_vector[layer_id]["mb"] - self.eta*layer.db
            layer.weights += self.momentum_vector[layer_id]["mW"]
            layer.biases += self.momentum_vector[layer_id]["mb"]
        
        elif hasattr(layer, "_gamma") and hasattr(layer, "_beta"):
            self.momentum_vector[layer_id]["mGamma"] = self.beta*self.momentum_vector[layer_id]["mGamma"] - self.eta*layer.dGamma
            self.momentum_vector[layer_id]["mBeta"] = self.beta*self.momentum_vector[layer_id]["mBeta"] - self.eta*layer.dBeta
            layer._beta += self.momentum_vector[layer_id]["mBeta"]
            layer._gamma += self.momentum_vector[layer_id]["mGamma"]

        return

        


