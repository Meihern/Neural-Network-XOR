import numpy as np


def mean_squared_error(y_predicted, y):
    return np.mean(np.square(y_predicted, y))


def mean_squared_error_derivative(y_predicted, y):
    return (2 / len(y)) * np.sum(y_predicted - y)


def squared_error(y_predicted, y):
    return np.square(y - y_predicted)/2


def squared_error_derivative(y_predicted, y):
    return y_predicted - y


def mean_absolute_error():
    pass


def binary_cross_entropy(y, y_predicted):
    return -np.mean(y * np.log(y_predicted) + ((1 - y) * np.log(1 - y_predicted)))
