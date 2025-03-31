import numpy as np

from sklearn.datasets import fetch_openml
from src import NeuralNetwork, InputLayer, HiddenLayer, OutputLayer, BatchNormLayer, SGD, Momentum
from visualize import NeuralNetworkVisualizer


if __name__ == "__main__":
    inputs = InputLayer(input_shape=[28, 28], flatten=True)
    layer1 = HiddenLayer(50, activation_function="leaky_relu", init_method="he")
    layer2 = HiddenLayer(50, activation_function="leaky_relu", init_method="he")
    layer3 = HiddenLayer(50, activation_function="leaky_relu", init_method="he")
    layer4 = HiddenLayer(50, activation_function="leaky_relu", init_method="he")
    layer5 = HiddenLayer(50, activation_function="leaky_relu", init_method="he")
    layer6 = HiddenLayer(50, activation_function="leaky_relu", init_method="he")
    
    outputs = OutputLayer(10, activation_function="softmax", init_method="he")
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
    layers.append(layer5)
    layers.append(layer6)
    layers.append(outputs)

    momentum_optimizer = Momentum(eta=5e-3, beta=0.8)

    neural_net = NeuralNetwork(layers, epochs=100, eta=1e-5, loss_func="cross_entropy", optimizer=momentum_optimizer)
    neural_net.fit(X_train, y_train, batch_size=32, X_val=X_test, y_val=y_test, plot_curves=True)

    
