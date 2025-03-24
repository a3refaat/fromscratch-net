import numpy as np

from sklearn.datasets import fetch_openml
from src import NeuralNetwork, InputLayer, HiddenLayer, OutputLayer, BatchNormLayer
from visualize import NeuralNetworkVisualizer


if __name__ == "__main__":
    inputs = InputLayer(input_shape=[784,])
    layer1 = BatchNormLayer(25, prev_layer=inputs)
    layer2 = BatchNormLayer(25, prev_layer=layer1)
    layer3 = HiddenLayer(50, prev_layer=layer2, activation_function="relu", init_method="glorot")
    layer4 = HiddenLayer(75, prev_layer=layer3, activation_function="relu", init_method="glorot")
    
    outputs = OutputLayer(10, prev_layer=layer4, activation_function="softmax", init_method="glorot")
    mnist = fetch_openml("mnist_784", version=1, parser='pandas')

    X = mnist.data.to_numpy()  # Convert to NumPy array
    X = X / 255.0 # Scale pixel values
    y = mnist.target.to_numpy().astype(int)

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    layers = []
    layers.append(inputs)
    layers.append(layer1)
    layers.append(layer2)
    layers.append(layer3)
    layers.append(layer4)
    layers.append(outputs)
    neural_net = NeuralNetwork(layers, epochs=100, eta=1e-3, loss_func="cross_entropy")
    neural_net.fit(X_train, y_train, batch_size=32)

    
