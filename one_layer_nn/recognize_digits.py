import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import sys
sys.path.append(".")
from utils.data_utils import *

# np.random.seed(4)

def sigmoid(z):
  s = 1 / (1 + np.exp(-z))
  return s

def sigmoid_derv(s):
    return s * (1 - s)

def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

def initialize_parameters(X, n_hidden, Y):
    W1 = np.random.randn(X.shape[1], n_hidden) * 0.01
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, Y.shape[1] ) * 0.01
    b2 = np.zeros((1, Y.shape[1]))

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def forward(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

#     (1617, 64)
#     (64, 128)
#     (1617, 128)
#     (1, 128)

    Z1 = np.matmul(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.matmul(A1, W2) + b2
    A2 = softmax(Z2)

    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

def cost(A2, Y):
    m = Y.shape[1]
    logprobs = (1/m)*np.sum(np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),(1-Y)))
    cost = - np.sum(logprobs)
    return cost

def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss

def backward(X, Y, parameters, activations):
    m = X.shape[1]
    W2 = parameters["W2"]
    A1 = activations["A1"]
    A2 = activations["A2"]

    dZ2 = A2 - Y
    dW2 = (1/m) * np.matmul(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis = 0, keepdims = True)
    dZ1 = np.matmul(dZ2, W2.T) * sigmoid_derv(A1)
    dW1 = (1/m) * np.matmul(X.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis = 0, keepdims = True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

def update_parameters(parameters, grads, learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - dW1 * learning_rate
    b1 = b1 - db1 * learning_rate
    W2 = W2 - dW2 * learning_rate
    b2 = b2 - db2 * learning_rate

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def predict(X, parameters):
    activations = forward(X, parameters)
    A2 = activations["A2"]

    predictions = np.argmax(A2)
    return predictions

def model(X, Y, hidden_layers, iterations):
#     n_input = X.shape[0]
#     n_output = Y.shape[0]
    parameters = initialize_parameters(X, hidden_layers, Y)

    for i in range(iterations):
        activations = forward(X, parameters)
        cst = cost(activations["A2"], Y)
        grads = backward(X, Y, parameters, activations)
        parameters = update_parameters(parameters, grads, learning_rate = 0.5)

        if(i % 100 == 0):
            print("cost after {} iterations {}".format(i, str(cst)))

    return parameters


digits = load_digits()
digits_binary_rep = pd.get_dummies(digits.target)
training_set, test_set, training_labels, test_labels = train_test_split(digits.data, digits_binary_rep, test_size=0.1, random_state=22)

parameters = model(training_set/16, np.array(training_labels), 128, 1500)

def get_acc(x, y, parameters):
    acc = 0
#     count = 1
    for xx,yy in zip(x, y):
        print(yy)
        s = predict(xx, parameters)
        print(s)
        if s == np.argmax(yy):
            acc +=1
#         if(count > 0):
#             break
#         count = count + 1
    return acc/len(x)*100

# print("Training accuracy : ", get_acc(training_set/16, np.array(training_labels), parameters))
print("Test accuracy : ", get_acc(test_set/16, np.array(test_labels), parameters))

