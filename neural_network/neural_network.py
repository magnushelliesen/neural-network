"""
By: Magnus Kvåle Helliesen
"""

import numpy as np


class NeuralNetwork():
    """
    Docstring will come
    """

    def __init__(self,
                 dim_input: int,
                 dim_hidden: int,
                 n_hidden: int,
                 dim_output: int
                 ):
        """
        Docstring will come
        """

        self._dim_input = dim_input
        self._dim_hidden = dim_hidden
        self._n_hidden = n_hidden
        self._dim_output = dim_output

        # Setup weights and biases
        self._weights = []
        self._biases = []

        # Setup weights and biases from input layer to first hidden layer
        self._weights += (np.random.rand(dim_hidden, dim_input)-0.5)/self.dim_hidden,
        self._biases += (np.random.rand(dim_hidden)-0.5)/self.dim_hidden,

        # Setup weights and biases between hidden layers
        for i in range(self.n_hidden-1):
            self._weights += (np.random.rand(dim_hidden, dim_hidden)-0.5)/self.dim_hidden,
            self._biases += (np.random.rand(dim_hidden)-0.5)/self.dim_hidden,

        # Setup weights and biases from last hidden layer to output layer
        self._weights += (np.random.rand(dim_output, dim_hidden)-0.5)/self.dim_output,
        self._biases += (np.random.rand(dim_output)-0.5)/self.dim_output,

        # Storing initial weigths and biases
        self._weights0 = tuple(x.copy() for x in self.weights)
        self._biases0 = tuple(x.copy() for x in self.biases)

    @property
    def dim_input(self):
        return self._dim_input

    @property
    def dim_hidden(self):
        return self._dim_hidden

    @property
    def n_hidden(self):
        return self._n_hidden

    @property
    def dim_output(self):
        return self._dim_output

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases

    @property
    def weights0(self):
        return self._weights0

    @property
    def biases0(self):
        return self._biases0

    @staticmethod
    def _sigmoid(x):
        """
        Sigmoid activation function
        """

        return 1/(1+np.exp(-x))

    @staticmethod
    def _relu(x):
        """
        ReLU activation function
        """

        return np.maximum(x, 0)

    @staticmethod
    def _softmax(x):
        """
        Softmax activation function
        """

        return np.exp(x)/np.exp(x).sum()

    def _activations(self, input: np.ndarray):
        """
        Docstring will come
        """

        if input.shape[0] != self.dim_input:
            raise IndexError(f'Input must have {self.dim_input} rows, not {input.shape[0]}')
        if len(input.shape) != 1:
            raise IndexError('Input must be 1d array')

        # We store all the activation from the different layers in a tuple
        x = input
        activation = tuple()

        # Forwardpropagation
        for i, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            if i < self.n_hidden:
                x = self._relu(weights.dot(x)+biases)
            else:
                x = self._softmax(weights.dot(x)+biases)

            activation += x,

        return activation

    def predict(self, input: np.ndarray):
        """
        Docstring will come
        """

        return self._activations(input)[-1]

    def train(self, data: tuple(tuple()), step=1):
        """
        Docstring will come
        """

        n = len(data)
        activations = tuple(self._activations(x[0]) for x in data)

        delta = []
        for i in reversed(range(self.n_hidden+1)):
            for j, ((_, target), activation) in enumerate(zip(data, activations)):
                if i == self.n_hidden:
                    delta += activation[i]-target,
                else:
                    delta[j] = delta[j].dot(self.weights[i+1]).T*(activation[i]>0)

            self._biases[i] -= step*sum(delta)/n
            if i == 0:
                self._weights[i] -= step*np.outer(sum(delta), sum(x[0] for x in data))/n
            else:
                self._weights[i] -= step*np.outer(sum(delta), sum(x[i-1] for x in activations))/n
