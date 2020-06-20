import numpy as np
import sys
sys.path.append(".")
from utils.data_utils import *
from utils.activation_functions import *
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def generate_mini_batches(X, Y, batch_size=64, seed=0):
    batches = []
    m = X.shape[0]

    np.random.seed(seed)
    permutation = list(np.random.permutation(m))
    shuffleX = X[permutation, :]
    shuffleY = Y[permutation, :].reshape(m, 1)

    num_batches = math.floor(m/batch_size)
    for i in range(0, num_batches):
        batchX = shuffleX[i*batch_size : (i+1)*batch_size, :]
        batchY = shuffleY[i*batch_size : (i+1)*batch_size, :]
        batch = (batchX, batchY)
        batches.append(batch)

    if (m % batch_size != 0):
        batchX = shuffleX[num_batches*batch_size : , :]
        batchY = shuffleY[num_batches*batch_size : , :]
        batch = (batchX, batchY)
        batches.append(batch)

    return batches

def initialize_weights_and_biases(layers_dim, seed=0):
    parameters = {}
    np.random.seed(seed)
    for i in range(len(layers_dim)-1):
        parameters["W" + str(i+1)] = 2 * (np.random.randn(layers_dim[i], layers_dim[i+1]) if (i<len(layers_dim)-2) else np.random.rand(layers_dim[i], layers_dim[i+1])) - 1
        parameters["b" + str(i+1)] = np.zeros((1, layers_dim[i+1]))
    return parameters

def initialize_momentum(parameters):
    v = {}
    for i in range(len(parameters)//2):
        v["dW" + str(i+1)] = np.zeros_like(parameters["W" + str(i+1)])
        v["db" + str(i+1)] = np.zeros_like(parameters["b" + str(i+1)])
    return v

def initialize_adam(parameters):
    v = {}
    s = {}
    for i in range(len(parameters)//2):
        v["dW" + str(i+1)] = np.zeros_like(parameters["W" + str(i+1)])
        v["db" + str(i+1)] = np.zeros_like(parameters["b" + str(i+1)])
        s["dW" + str(i+1)] = np.zeros_like(parameters["W" + str(i+1)])
        s["db" + str(i+1)] = np.zeros_like(parameters["b" + str(i+1)])
    return v, s

def forward(parameters, X):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = np.matmul(X, W1) + b1
    A1 = np.tanh(Z1)

    Z2 = np.matmul(A1, W2) + b2
    A2 = np.tanh(Z2)

    Z3 = np.matmul(A2, W3) + b3
    A3 = sigmoid(Z3)

    return {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3}

def forward_with_dropout(parameters, X, keep_prob=0.5, seed=0):
    np.random.seed(seed)

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = np.matmul(X, W1) + b1
    A1 = np.tanh(Z1)
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    D1 = (D1 < keep_prob).astype(int)
    A1 = A1 * D1
    A1 = A1 / keep_prob

    Z2 = np.matmul(A1, W2) + b2
    A2 = np.tanh(Z2)
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = (D2 < keep_prob).astype(int)
    A2 = A2 * D2
    A2 = A2 / keep_prob

    Z3 = np.matmul(A2, W3) + b3
    A3 = sigmoid(Z3)

    return {'Z1': Z1, 'D1': D1, 'A1': A1, 'Z2': Z2, 'D2': D2, 'A2': A2, 'Z3': Z3, 'A3': A3}

def backprop(X, Y, parameters, activations):
    m = X.shape[0]
    W3 = parameters['W3']
    W2 = parameters['W2']
    A3 = activations['A3']
    A2 = activations['A2']
    A1 = activations['A1']

    dZ3 = A3 - Y
    dW3 = (1/m) * np.matmul(A2.T, dZ3)
    db3 = (1/m) * np.sum(dZ3, axis=0)

    dZ2 = np.matmul(dZ3, W3.T) * tanh_derv(A2)
    dW2 = (1/m) * np.matmul(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0)

    dZ1 = np.matmul(dZ2, W2.T) * tanh_derv(A1)
    dW1 = (1/m) * np.matmul(X.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0)

    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}

def backprop_with_regularization(X, Y, parameters, activations, lambd):
    m = X.shape[0]
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    A3 = activations['A3']
    A2 = activations['A2']
    A1 = activations['A1']

    dZ3 = A3 - Y
    dW3 = (1/m) * np.matmul(A2.T, dZ3) + (lambd * W3)/m
    db3 = (1/m) * np.sum(dZ3, axis=0)

    dZ2 = np.matmul(dZ3, W3.T) * tanh_derv(A2)
    dW2 = (1/m) * np.matmul(A1.T, dZ2) + (lambd * W2)/m
    db2 = (1/m) * np.sum(dZ2, axis=0)

    dZ1 = np.matmul(dZ2, W2.T) * tanh_derv(A1)
    dW1 = (1/m) * np.matmul(X.T, dZ1) + (lambd * W1)/m
    db1 = (1/m) * np.sum(dZ1, axis=0)

    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}

def backprop_with_dropout(X, Y, parameters, activations, keep_prob=0.5):
    m = X.shape[0]
    W3 = parameters['W3']
    W2 = parameters['W2']
    A3 = activations['A3']
    A2 = activations['A2']
    A1 = activations['A1']
    D1 = activations['D1']
    D2 = activations['D2']

    dZ3 = A3 - Y
    dW3 = (1/m) * np.matmul(A2.T, dZ3)
    db3 = (1/m) * np.sum(dZ3, axis=0)

    dA2 = np.matmul(dZ3, W3.T)
    dA2 = dA2 * D2
    dA2 = dA2 / keep_prob

    dZ2 = dA2 * tanh_derv(A2)
    dW2 = (1/m) * np.matmul(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0)

    dA1 = np.matmul(dZ2, W2.T)
    dA1 = dA1 * D1
    dA1 = dA1 / keep_prob

    dZ1 = dA1 * tanh_derv(A1)
    dW1 = (1/m) * np.matmul(X.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0)

    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}


def update_parameters(parameters, grads, learning_rate):
    for i in range(len(parameters)//2):
        parameters["W" + str(i+1)] = parameters["W" + str(i+1)] - grads["dW" + str(i+1)] * learning_rate
        parameters["b" + str(i+1)] = parameters["b" + str(i+1)] - grads["db" + str(i+1)] * learning_rate
    return parameters

def update_parameters_with_momentum(parameters, grads, learning_rate, v, beta):
    for i in range(len(parameters)//2):
        v["dW" + str(i+1)] = beta * v["dW" + str(i+1)] + (1 - beta) * grads["dW" + str(i+1)]
        v["db" + str(i+1)] = beta * v["db" + str(i+1)] + (1 - beta) * grads["db" + str(i+1)]

        parameters["W" + str(i+1)] = parameters["W" + str(i+1)] - v["dW" + str(i+1)] * learning_rate
        parameters["b" + str(i+1)] = parameters["b" + str(i+1)] - v["db" + str(i+1)] * learning_rate
    return parameters, v

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
    v_c = {}
    s_c = {}
    for i in range(len(parameters)//2):
        v["dW" + str(i+1)] = beta1 * v["dW" + str(i+1)] + (1 - beta1) * grads["dW" + str(i+1)]
        v["db" + str(i+1)] = beta1 * v["db" + str(i+1)] + (1 - beta1) * grads["db" + str(i+1)]

        v_c["dW" + str(i+1)] = v["dW" + str(i+1)] / (1 - beta1**t)
        v_c["db" + str(i+1)] = v["db" + str(i+1)] / (1 - beta1**t)

        s["dW" + str(i+1)] = beta2 * s["dW" + str(i+1)] + (1 - beta2) * grads["dW" + str(i+1)]**2
        s["db" + str(i+1)] = beta2 * s["db" + str(i+1)] + (1 - beta2) * grads["db" + str(i+1)]**2

        s_c["dW" + str(i+1)] = s["dW" + str(i+1)] / (1 - beta2**t)
        s_c["db" + str(i+1)] = s["db" + str(i+1)] / (1 - beta2**t)

        parameters["W" + str(i+1)] = parameters["W" + str(i+1)] - learning_rate * (v_c["dW" + str(i+1)] / (np.sqrt(s_c["dW" + str(i+1)]) + epsilon))
        parameters["b" + str(i+1)] = parameters["b" + str(i+1)] - learning_rate * (v_c["db" + str(i+1)] / (np.sqrt(s_c["db" + str(i+1)]) + epsilon))
    return parameters, v, s

def cost(A, Y):
    m = Y.shape[0]
    logprobs = (1/m)*np.sum(np.multiply(np.log(A),Y) + np.multiply(np.log(1-A),(1-Y)))
    cost = - np.sum(logprobs)
    return cost

def cost_with_regularization(A, Y, parameters, lambd):
    m = Y.shape[0]
    logprobs = (1/m)*np.sum(np.multiply(np.log(A),Y) + np.multiply(np.log(1-A),(1-Y)))
    cost = - np.sum(logprobs)
    L2_regularization_cost = (1/m) * (lambd / 2) * (np.sum(np.square(parameters['W1'])) + np.sum(np.square(parameters['W2'])) + np.sum(np.square(parameters['W3'])))
    return cost + L2_regularization_cost

def predict(parameters, X, Y):
    activations = forward(parameters, X)
    A3 = activations["A3"]
    predictions = (A3 > 0.5).astype(int)
    print("modal accuracy with given set: {} %".format(100 - np.mean(np.abs(predictions - Y)) * 100))
    return predictions

def predict_s(parameters, X):
    activations = forward(parameters, X)
    A3 = activations["A3"]
    predictions = (A3 > 0.5).astype(int)
    return predictions

def train(X, Y, n_hidden, learning_rate, n_iterations, keep_prob=1, lambd=0, optimizer="gd", batch_size=64, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, printCost=False):
    n_input = X.shape[1]
    n_output = Y.shape[1]
    t = 0

    parameters = initialize_weights_and_biases([n_input, n_hidden, n_hidden, n_output])

    if (optimizer == "momentum"):
        v = initialize_momentum(parameters)
    elif (optimizer == "adam"):
        v, s = initialize_adam(parameters)

    errors = []
    for i in range(n_iterations):
        if(keep_prob == 1):
            activations = forward(parameters, X)
        elif(keep_prob != 0):
            activations = forward_with_dropout(parameters, X, keep_prob)

        if(lambd == 0 and keep_prob == 1):
            grads = backprop(X, Y, parameters, activations)
        elif(lambd != 0):
            grads = backprop_with_regularization(X, Y, parameters, activations, lambd)
        elif(keep_prob < 1):
            grads = backprop_with_dropout(X, Y, parameters, activations, keep_prob)

        if (optimizer == "gd"):
            parameters = update_parameters(parameters, grads, learning_rate)
        elif (optimizer == "momentum"):
            parameters, v = update_parameters_with_momentum(parameters, grads, learning_rate, v, beta)
        elif (optimizer == "adam"):
            t = t + 1
            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)

        if(printCost == True and i % 1000 == 0):
            if(lambd == 0):
                print("cost after {} iterations {}".format(i, str(cost(activations['A3'], Y))))
            elif(lambd != 0):
                print("cost after {} iterations {}".format(i, str(cost_with_regularization(activations['A3'], Y, parameters, lambd))))

    return parameters, errors

n_features = 8
n_samples = 1000
seed = 0

np.random.seed(seed)

set = np.random.randn(n_samples, n_features)
lbls = compute_labels(set, 1).reshape(n_samples, 1)

training_set, training_labels, test_set, test_labels = split_train_test_dataset(set, lbls, split_by = "rows", perc = 0.1)


for opt in ["gd", "momentum", "adam"]:
    print('\n:: gradient descent - optimizer = ' + opt)
    parameters, errors = train(training_set, training_labels, n_hidden=5, learning_rate = 0.07, n_iterations=10000, optimizer=opt)
    predict(parameters, test_set, test_labels)

    print(':: gradient descent with regularization')
    parameters, errors = train(training_set, training_labels, n_hidden=5, learning_rate = 0.07, n_iterations=10000, optimizer=opt, lambd=0.1)
    predict(parameters, test_set, test_labels)

    print(':: gradient descent with drop out')
    parameters, errors = train(training_set, training_labels, n_hidden=5, learning_rate = 0.07, n_iterations=10000, optimizer=opt, keep_prob=0.8)
    predict(parameters, test_set, test_labels)

