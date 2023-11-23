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
        self._weights += np.zeros((dim_hidden, dim_input)),
        self._biases += np.zeros(dim_hidden),

        # Setup weights and biases between hidden layers
        for i in range(self.n_hidden-1):
            self._weights += np.zeros((dim_hidden, dim_hidden)),
            self._biases += np.zeros(dim_hidden),

        # Setup weights and biases from last hidden layer to output layer
        self._weights += np.zeros((dim_output, dim_hidden)),
        self._biases += np.zeros(dim_output),

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
        
        activation = self._activation(input)

        output = activation[-1]

        # This is ugly and needs refactoring, but is close
        ##################################################
        for i, layers in enumerate(activation[::-1]):
            if i == 0:
                delta_o = (output-target)*output*(1-output)
            else:
                delta_o = (delta_o.dot(self.weights[-i])).T*activation[-1-i]*(1-activation[-1-i])

            try:
                activation_o = activation[-2-i]
            except IndexError:
                activation_o = input
            delta_loss = np.outer(delta_o, activation_o)

            # Update weights
            self._weights[-i-1] -= step*delta_loss
            
            # What about the bias? I think this is it
            self._biases[-i-1] -= step*delta_o
        ##################################################