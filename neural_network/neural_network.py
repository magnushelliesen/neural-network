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
        self._weights += np.random.normal(size=(dim_hidden, dim_input))/self.dim_input,
        self._biases += np.random.normal(size=dim_hidden)/self.dim_input,

        # Setup weights and biases between hidden layers
        for i in range(self.n_hidden-1):
            self._weights += np.random.normal(size=(dim_hidden, dim_hidden))/self.dim_hidden,
            self._biases += np.random.normal(size=dim_hidden)/self.dim_hidden,

        # Setup weights and biases from last hidden layer to output layer
        self._weights += np.random.normal(size=(dim_output, dim_hidden))/self.dim_hidden,
        self._biases += np.random.normal(size=dim_output)/self.dim_hidden,

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
        return self.dim_output

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases

    @staticmethod
    def _actiavtion_function(x):
        """
        Sigmoid activation function
        """
        return 1/(1+np.exp(-x))

    def _activation(self, input: np.ndarray):
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

        for weights, biases in zip(self.weights, self.biases):
            x = self._actiavtion_function(weights.dot(x)+biases)
            activation += x,

        return activation

    def predict(self, input: np.ndarray):
        """
        Docstring will come
        """

        return self._activation(input)[-1]

    def train(self, input: np.ndarray, target: np.ndarray, step=1):
        """
        Docstring will come
        """
        
        act = self._activation(input)
        act = tuple(input)+act

        i = self.n_hidden
        while True:
            if i == 0:
                break
            if i == self.n_hidden:
                delta = (act[i+1]-target)*act[i+1]*(1-act[i+1])
            else:
                delta = delta.dot(self.weights[i+1]).T*act[i+1]*(1-act[i+1])

            Delta = np.outer(delta, act[i])

            self._weights[i] -= step*Delta
            self._biases[i] -= step*delta

            i -= 1
