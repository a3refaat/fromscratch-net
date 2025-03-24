import numpy as np

from .losses import CrossEntropyLoss, MSELoss
from .layers import Layer

LOSS_MAP = {
    'cross_entropy': CrossEntropyLoss,
    'mse': MSELoss,
}

class NeuralNetwork():
    def __init__(self, layers:list[Layer], epochs:int, eta:float, loss_func=None):
        self.layers = layers
        self.epochs = epochs
        self.eta = eta
        self.valid_loss_funcs = list(LOSS_MAP.keys())
        self.loss_func = loss_func

    @property
    def loss_func(self):
        return self._loss_func
    
    @loss_func.setter
    def loss_func(self, loss_func:str) -> None:
        if loss_func is None:
            self._loss_func = LOSS_MAP['mse']()
        elif loss_func not in self.valid_loss_funcs:
            raise ValueError(f"Invalid loss function: {loss_func}. Please select from {self.valid_loss_funcs}")
        else:
            self._loss_func = LOSS_MAP[loss_func]()
        
    
    def fit(self, X:np.ndarray, y:np.ndarray, batch_size:int): ## Optimizers to be added later
        for layer in self.layers:
            layer.training = True

        for i in range(self.epochs):
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            batch_losses = []
            for j in range(0, len(X), batch_size):


                X_set = X[j:j + batch_size]
                y_set = y[j:j + batch_size]
                self.layers[0].inputs = X_set

                y_pred = self.__forward_pass()
                loss = self.__compute_loss(y_pred, y_set, batch_size)
                batch_losses.append(loss)
                gradient = self.__backpropagate(y_pred, y_set)
                self.__update_weights()

            epoch_loss = np.mean(batch_losses)
            print(f"Epoch {i + 1}/{self.epochs} - Loss: {epoch_loss:.4f}")


        return
        
    def __compute_loss(self, y_pred:np.ndarray, y_true:np.ndarray, batch_size:int) -> float:
        return self._loss_func.compute_loss(y_pred, y_true, batch_size)

    def __forward_pass(self, ) -> np.ndarray:
        return self.layers[-1].activate()
    
    def __backpropagate(self, y_pred:np.ndarray, y_true:np.ndarray) -> np.ndarray:
        loss_derivative = self._loss_func.derivative(y_pred, y_true)
        return self.layers[-1].backward(loss_derivative)
    
    def __update_weights(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "weights") and hasattr(layer, "dW"):
                layer.weights -= self.eta * layer.dW
                layer.biases -= self.eta * layer.db
            
            elif hasattr(layer, "_gamma") and hasattr(layer, "_beta"):
                layer._gamma -= self.eta*layer.dGamma
                layer._beta -= self.eta*layer.dBeta
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.layers[0].inputs = X
        for layer in self.layers:
            training = False
        return self.__forward_pass()
        
