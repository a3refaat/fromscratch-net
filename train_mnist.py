import numpy as np

from sklearn.datasets import fetch_openml
from src import NeuralNetwork, InputLayer, HiddenLayer, OutputLayer
from visualize import NeuralNetworkVisualizer


if __name__ == "__main__":
    inputs = InputLayer(input_shape=[784,])
    mylayer = HiddenLayer(50, prev_layer=inputs, activation_function="relu", init_method="glorot")
    layer2 = HiddenLayer(75, prev_layer=mylayer, activation_function="relu", init_method="glorot")
    outputs = OutputLayer(10, prev_layer=layer2, activation_function="softmax", init_method="glorot")
    mnist = fetch_openml("mnist_784", version=1, parser='pandas')

    X = mnist.data.to_numpy()  # Convert to NumPy array
    X = X / 255.0 # Scale pixel values
    y = mnist.target.to_numpy().astype(int)

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    layers = []
    layers.append(inputs)
    layers.append(mylayer)
    layers.append(layer2)
    layers.append(outputs)
    neural_net = NeuralNetwork(layers, epochs=100, eta=1e-4, loss_func="cross_entropy")
    NeuralNetworkVisualizer(neural_net).visualize_static()
    neural_net.fit(X_train, y_train, batch_size=32)

    
