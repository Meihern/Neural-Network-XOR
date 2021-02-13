import numpy as np


def sigmoid(x) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x) -> np.ndarray:
    return x * (1 - x)


def relu(x) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_derivative(x) -> np.ndarray:
    return np.where(x > 0, 1, 0)


def leaky_relu(x, a=0.01) -> np.ndarray:
    return np.where(x > 0, x, x * a)


def leaky_relu_derivative(x, a=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = a
    return dx


def softmax() -> np.ndarray:
    pass
