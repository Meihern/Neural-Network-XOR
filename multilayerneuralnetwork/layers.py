import numpy as np

from multilayerneuralnetwork import functions


class Layer:
    def __init__(self, dims: int, input_dims: int, activation_function=functions.sigmoid, weights=None, bias=None):
        self.input_dims = input_dims
        self.activation_function = activation_function
        self.activation_function_derivative = self.__set_activation_function_derivative()
        self.dims = dims
        self.bias = bias
        self.weights = weights
        self.output = None
        if weights is None:
            self.weights = np.random.uniform(-1, 1, (self.input_dims, self.dims))
        if bias is None:
            self.bias = np.random.uniform(-1, 1, (1, dims))

    def calcul_output(self, input_data):
        self.output = self.activation_function(self.__calcul_input(input_data))

    def __calcul_input(self, input_data):
        return np.dot(input_data, self.weights) + self.bias

    def __set_activation_function_derivative(self):
        if self.activation_function == functions.sigmoid:
            return functions.sigmoid_derivative
