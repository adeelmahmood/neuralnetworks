import numpy as np


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def sigmoid_derv(s):
    return s * (1 - s)


def relu(x):
    s = np.maximum(0, x)
    return s


def relu_derv(x):
    return (x > 0)


def tanh_derv(x):
    return (1 - np.power(x, 2))


def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)
