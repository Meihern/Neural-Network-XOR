from multilayerneuralnetwork import losses, metrics
from multilayerneuralnetwork.layers import Layer
import numpy as np


class MultiLayerNN:

    def __init__(self, hidden_layer: Layer, output_layer: Layer, loss_function=losses.mean_squared_error, metric=metrics.accuracy):
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.loss_function = loss_function
        self.loss_function_derivative = self.__set_loss_function_derivative()
        self.metric = metric

    def fit(self, x, y, learning_rate=0.001, epochs=1):
        for i in range(epochs):
            y_predicted = self.__forward_pass(x)
            self.__back_propagation(x, y_predicted, y, learning_rate)
            print('Epoch: {},\t accuracy = {}'.format(i, self.metric(y, y_predicted)))

    def predict(self, x):
        return self.__forward_pass(x)

    def __set_loss_function_derivative(self):
        if self.loss_function == losses.mean_squared_error:
            return losses.mean_squared_error_derivative
        if self.loss_function == losses.squared_error:
            return losses.squared_error_derivative

    def __forward_pass(self, input_data):
        self.hidden_layer.calcul_output(input_data)
        hidden_layer_output_data = self.hidden_layer.output
        self.output_layer.calcul_output(hidden_layer_output_data)
        return self.output_layer.output

    def __back_propagation(self, x, y_predicted, y, learning_rate):
        dloss_dw_h = x.T.dot(((self.loss_function_derivative(y_predicted,
                                                             y) * self.output_layer.activation_function_derivative(
            y_predicted)).dot(self.output_layer.weights.T)) * (
                                 self.hidden_layer.activation_function_derivative(self.hidden_layer.output)))

        dloss_dbias_h = self.loss_function_derivative(y_predicted,
                                                      y) * self.output_layer.activation_function_derivative(
            y_predicted).dot(
            self.output_layer.weights.T) * self.hidden_layer.activation_function_derivative(self.hidden_layer.output)

        dloss_dw_out = self.hidden_layer.output.T.dot(
            self.loss_function_derivative(y_predicted, y) * self.hidden_layer.activation_function_derivative(
                y_predicted))

        dloss_dbias_out = (y_predicted - y) * self.output_layer.activation_function_derivative(y_predicted)

        self.hidden_layer.weights -= dloss_dw_h * learning_rate
        self.hidden_layer.bias -= np.sum(dloss_dbias_h, axis=0, keepdims=True) * learning_rate

        self.output_layer.weights -= dloss_dw_out * learning_rate
        self.output_layer.bias -= np.sum(dloss_dbias_out, axis=0, keepdims=True) * learning_rate
