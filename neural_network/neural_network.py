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

        # Setup weights and biases for input layer
        self._w_i = np.ones((dim_hidden, dim_input))
        self._b_i = np.zeros(dim_input)

        # Setup weights and biases for hidden layers
        self._w_h = {}
        self._b_h = {}
        for i in range(self.n_hidden-1):
            self._w_h[i] = np.ones((dim_hidden, dim_hidden))
            self._b_h[i] = np.zeros(dim_hidden)

        # Setup weights and biases for output layer
        self._w_o = np.zeros((dim_output, dim_hidden))
        self._b_o = np.zeros(dim_output)

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
    def w_i(self):
        return self._w_i

    @property
    def b_i(self):
        return self._b_i

    @property
    def w_h(self):
        return self._w_h

    @property
    def b_h(self):
        return self._b_h

    @property
    def w_o(self):
        return self._w_o

    @property
    def b_o(self):
        return self._b_o

    @staticmethod
    def actiavtion_function(x):
        return 1/(1+np.exp(-x))

    def predict(self, input: np.ndarray):
        """
        Docstring will come
        """
        if input.shape[0] != self.dim_input:
            raise IndexError(f'Input has dimenson {input.shape} and not {self.dim_input}')

        # I will have to think about what to store of the calculations
        x = self.actiavtion_function(self.w_i.dot(input)+self.b_i)

        for i in range(self.n_hidden-1):
            x = self.actiavtion_function(self.w_h.get(i).dot(x)+self.b_h.get(i))

        return self.actiavtion_function(self.w_o.dot(x)+self.b_o)

    def train(self, data: dict):
        # I think the training data will be a dict of np.ndarrays
        pass