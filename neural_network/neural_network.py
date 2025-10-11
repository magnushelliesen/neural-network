"""
By: Magnus Kvåle Helliesen
"""

import numpy as np
import pandas as pd # type: ignore
from random import choices
from typing import Union, List, Tuple


class NeuralNetwork():
    """
    A simple neural network class with multiple hidden layers.
    """

    def __init__(
            self,
            dim_input: int,
            dim_hidden: int,
            n_hidden: int,
            dim_output: int
            ) -> None:
        """
        Initializes the neural network with given dimensions for input, hidden, and output layers.
        Randomly initializes weights and biases for each layer.

        Parameters
        ----------
        dim_input : int
            Dimension of the input layer.
        dim_hidden : int
            Dimension of each hidden layer.
        n_hidden : int
            Number of hidden layers.
        dim_output : int
            Dimension of the output layer.
        """

        self._training = 0

        self._dim_input = dim_input
        self._dim_hidden = dim_hidden
        self._n_hidden = n_hidden
        self._dim_output = dim_output

        self._last_input: np.ndarray | None = None
        self._last_activations: Tuple[np.ndarray , ...] | None = None

        # Setup weights and biases
        self._weights: Tuple[np.ndarray, ...] = tuple()
        self._biases: Tuple[np.ndarray, ...] = tuple()

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
    def training(self) -> int:
        return self._training

    @property
    def dim_input(self) -> int:
        return self._dim_input

    @property
    def dim_hidden(self) -> int:
        return self._dim_hidden

    @property
    def n_hidden(self) -> int:
        return self._n_hidden

    @property
    def dim_output(self) -> int:
        return self._dim_output

    @property
    def weights(self) -> Tuple[np.ndarray, ...]:
        return self._weights

    @property
    def biases(self) -> Tuple[np.ndarray, ...]:
        return self._biases

    @property
    def weights0(self) -> Tuple[np.ndarray, ...]:
        return self._weights0

    @property
    def biases0(self) -> Tuple[np.ndarray, ...]:
        return self._biases0

    @property
    def delta_weights(self) -> Tuple[np.ndarray, ...]:
        return tuple(x-y for x, y in zip(self.weights, self.weights0))

    @property
    def delta_biases(self) -> Tuple[np.ndarray]:
        return tuple(x-y for x, y in zip(self.biases, self.biases0))

    @property
    def last_input(self) -> np.ndarray | None:
        return self._last_input

    @property
    def last_activations(self) -> Tuple[np.ndarray, ...] | None:
        return self._last_activations

    def __repr__(self):
        return f'NeuralNetwork({self.dim_input}, {self.dim_hidden}, {self.n_hidden}, {self.dim_output})'

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        Parameters
        ----------
        x : numpy.ndarray
            Input array.

        Returns
        -------
        numpy.ndarray
            Output after applying the sigmoid function element-wise.
        """

        return 1/(1+np.exp(-x))

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        """
        ReLU activation function.

        Parameters
        ----------
        x : numpy.ndarray
            Input array.

        Returns
        -------
        numpy.ndarray
            Output after applying the ReLU function element-wise.
        """

        return np.maximum(x, 0)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """
        Softmax activation function.

        Parameters
        ----------
        x : numpy.ndarray
            Input array.

        Returns
        -------
        numpy.ndarray
            Output after applying the softmax function.
        """

        return np.exp(x)/np.exp(x).sum()

    def _activations(self, input: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Calculates activations through the network layers.

        Parameters
        ----------
        input : numpy.ndarray
            Input array.

        Returns
        -------
        tuple
            Activations from each layer stored in a tuple.
        """

        if input.shape[0] != self.dim_input:
            raise IndexError(f'Input must have {self.dim_input} rows, not {input.shape[0]}')
        if len(input.shape) != 1:
            raise IndexError('Input must be 1d array')

        x = input
        activations: Tuple[np.ndarray, ...] = tuple()

        # Forwardpropagation
        for i, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            if i == self.n_hidden:
                x = self._softmax(weights.dot(x)+biases)
            elif i == 0:
                x = self._relu(weights.dot(x)+biases)
            else:
                x = self._sigmoid(weights.dot(x)+biases)

            if all(np.isfinite(x)) is False:
                raise ValueError('Weights and biases give np.nan or np.inf')

            activations += x,

        self._last_input = input
        self._last_activations = activations

        return activations

    def predict(self, input: np.ndarray) -> np.ndarray:
        """
        Predicts the output for a given input using the neural network.

        Parameters
        ----------
        input : numpy.ndarray
            Input array.

        Returns
        -------
        numpy.ndarray
            Predicted output.
        """

        return self._activations(input)[-1]

    def train(
            self,
            data: Union[Tuple[Tuple[np.ndarray, np.ndarray]], Tuple[Tuple[np.ndarray, np.ndarray]]],
            n: int,
            step: float=0.1
            ) -> None:
        """
        Trains the neural network using backpropagation.

        Parameters
        ----------
        data : list[list[numpy.ndarray, numpy.ndarray]]
            Training data pairs of input and target output.
        n : int
            Number of training iterations.
        step : float, optional
            Learning rate (default is 0.1).
        """

        if isinstance(data, (list, tuple)):
            random_data = choices(data, k=n)
        elif isinstance(data, pd.DataFrame):
            random_df = data.sample(n=n, replace=True)
            raise RuntimeError('No support for dataframe yet')

        weights, biases = tuple(x.copy() for x in self.weights), tuple(x.copy() for x in self.biases)

        for input, target in random_data:
            try:
                delta_weights, delta_biases = self.backpropagation(input, target, step)

                for weight, delta_weight in zip(self._weights, delta_weights):
                    weight += delta_weight

                for bias, delta_bias in zip(self._biases, delta_biases):
                    bias += delta_bias
            except ValueError:
                self._weights, self._biases = weights, biases
                print('Try reducing learning rate')
                return

        self._training += n

    def batch_train(
            self,
            data: Union[Tuple[Tuple[np.ndarray, np.ndarray]], Tuple[Tuple[np.ndarray, np.ndarray]]],
            n: int,
            batch_size: int=10,
            step: float=0.1
            ) -> None:
        """
        Trains the neural network using backpropagation.

        Parameters
        ----------
        data : list[list[numpy.ndarray, numpy.ndarray]]
            Training data pairs of input and target output.
        n : int
            Number of training iterations.
        step : float, optional
            Learning rate (default is 0.1).
        """

        if isinstance(data, (list, tuple)):
            random_data_batches = self.batchify(choices(data, k=n), batch_size)
        elif isinstance(data, pd.DataFrame):
            random_df = data.sample(n=n, replace=True)
            raise RuntimeError('No support for dataframe yet')

        weights, biases = tuple(x.copy() for x in self.weights), tuple(x.copy() for x in self.biases)

        for random_data_batch in list(random_data_batches):

            delta_weights = tuple(np.zeros_like(x) for x in self.weights)
            delta_biases = tuple(np.zeros_like(x) for x in self.biases)

            for random_data in random_data_batch:
                input, target = random_data
                try:
                    delta_weights_update, delta_biases_update = self.backpropagation(input, target, step)

                    for delta_weight, delta_weight_update in zip(delta_weights, delta_weights_update):
                        delta_weight += delta_weight_update

                    for delta_biase, delta_biase_update in zip(delta_biases, delta_biases_update):
                        delta_biase += delta_biase_update

                except ValueError:
                    self._weights, self._biases = weights, biases
                    print('Try reducing learning rate')
                    return

            for weight, delta_weight in zip(self._weights, delta_weights):
                weight += delta_weight

            for bias, delta_bias in zip(self._biases, delta_biases):
                bias += delta_bias

        self._training += n

    def backpropagation(
            self,
            input: np.ndarray,
            target: np.ndarray,
            step: float=0.1
            ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Performs backpropagation to update weights and biases based on the input and target output.

        Parameters
        ----------
        input : numpy.ndarray
            Input array.
        target : numpy.ndarray
            Target output.
        step : float, optional
            Learning rate (default is 0.1).
        """
        
        activations = self._activations(input)
        delta_weights = [np.zeros_like(x) for x in self.weights]
        delta_biases = [np.zeros_like(x) for x in self.biases]

        # Backpropagation
        for i in reversed(range(self.n_hidden+1)):
            if i == self.n_hidden:
                delta = activations[i]-target
                delta_weights[i] -= step*np.outer(delta, activations[i-1])
            elif i == 0:
                delta = delta.dot(self.weights[i+1]).T*(activations[i]>0)
                delta_weights[i] -= step*np.outer(delta, input)
            else:
                delta = delta.dot(self.weights[i+1]).T*activations[i]*(1-activations[i])
                delta_weights[i] -= step*np.outer(delta, activations[i-1])
            delta_biases[i] -= step*delta

        return delta_weights, delta_biases

    @staticmethod
    def batchify(x: list, n: int):
        for i in range(0, len(x), n):
            yield x[i:i+n]
