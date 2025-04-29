import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from src import Model, InputLayer, HiddenLayer, OutputLayer, BatchNormLayer, SGD, Momentum, Adam

if __name__ == "__main__":

    # Load California Housing Dataset
    X, y = fetch_california_housing(return_X_y=True)
    
    # Optional: Standardize features (helps convergence)
    X = np.asarray((X - np.mean(X, axis=0)) / np.std(X, axis=0))
    y = np.asarray(y).reshape(-1, 1)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define model layers
    layers = [
    InputLayer(input_shape=[8], flatten=False),
    
    # Block 1: Wide feature expansion
    BatchNormLayer(512, activation_function="leaky_relu", init_method="he"),
    HiddenLayer(512, activation_function="leaky_relu", init_method="he"),
    
    # Block 2: Feature transformation
    BatchNormLayer(256, activation_function="leaky_relu", init_method="he"),
    HiddenLayer(256, activation_function="leaky_relu", init_method="he"),
    
    # Block 3: Feature compression
    BatchNormLayer(128, activation_function="leaky_relu", init_method="he"),
    HiddenLayer(128, activation_function="leaky_relu", init_method="he"),
    
    # Block 4: Further compression
    BatchNormLayer(64, activation_function="leaky_relu", init_method="he"),
    HiddenLayer(64, activation_function="leaky_relu", init_method="he"),
    
    # Block 5: Penultimate low-dimensional representation
    BatchNormLayer(32, activation_function="leaky_relu", init_method="he"),
    HiddenLayer(32, activation_function="leaky_relu", init_method="he"),
    
    # Final output layer
    OutputLayer(1, activation_function=None, init_method="he")  # Regression output
    ]

    # Choose optimizer
    momentum_optimizer = Momentum(eta=1e-3, beta=0.8)
    adam = Adam(eta=5e-5, beta1=0.8, beta2=0.999)

    # Instantiate model
    model = Model(layers, epochs=500, eta=5e-5, loss_func="mse", optimizer=adam)

    # Move model to GPU
    model.to('cuda')
    print(f"Model using device: {model.device_manager.device}")

    # Train the model
    model.fit(X_train, y_train, batch_size=1024, X_val=X_test, y_val=y_test, plot_curves=True)
    
