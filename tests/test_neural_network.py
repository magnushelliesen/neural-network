import numpy as np
from  neural_network import NeuralNetwork
import pytest

def test_neural_network():
    # Make deterministic part of two different arrays
    array1 = np.linspace(0, 1, 100)
    array2 = np.linspace(1, 0, 100)

    # Make noisy arrays
    type1_data = [[array1+np.random.normal(size=100)/10, np.array([1, 0])] for _ in range(250)]
    type2_data = [[array2+np.random.normal(size=100)/10, np.array([0, 1])] for _ in range(250)]
    data = [*type1_data, *type2_data]

    # Setup neural network
    nn = NeuralNetwork(100, 50, 2, 2)

    # Train neural network 10 000 times
    nn.batch_train(data, 10_000, 10)

    # Check prediction of array 1
    prediction = nn.predict(array1)
    assert prediction[0] > 0.9, "Prediction for array1 is wrong"

    # Check prediction of array 2
    prediction = nn.predict(array2)
    assert prediction[1] > 0.9, "Prediction for array2 is wrong"