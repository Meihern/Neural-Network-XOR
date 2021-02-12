import numpy as np


def sigmoid(x) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu() -> np.ndarray:
    pass


def leaky_relu() -> np.ndarray:
    pass


def softmax() -> np.ndarray:
    pass
