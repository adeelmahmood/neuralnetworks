import sys

import pandas as pd

sys.path.append(".")
from utils.activation_functions import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('three_layer_nn/wines-data.csv')
print(df.head())

labels = df[['Cultivar 1', 'Cultivar 2', 'Cultivar 3']].values

training_set = df.drop(['Cultivar 1', 'Cultivar 2', 'Cultivar 3'], axis=1)

training_set = training_set.values

x_train, x_val, y_train, y_val = train_test_split(training_set, labels, test_size=0.1, random_state=20)

np.random.seed(0)


def initialize_weights_and_biases(n_input, n_hidden, n_output):
    W1 = 2 * np.random.randn(n_input, n_hidden) - 1
    b1 = np.zeros((1, n_hidden))

    W2 = 2 * np.random.randn(n_hidden, n_hidden) - 1
    b2 = np.zeros((1, n_hidden))

    W3 = 2 * np.random.rand(n_hidden, n_output) - 1
    b3 = np.zeros((1, n_output))

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}


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
    A3 = softmax(Z3)

    return {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3}


def backprop(X, Y, parameters, activations):
    m = X.shape[0]
    W3 = parameters['W3']
    W2 = parameters['W2']
    A3 = activations['A3']
    A2 = activations['A2']
    A1 = activations['A1']

    dZ3 = A3 - Y
    dW3 = (1 / m) * np.matmul(A2.T, dZ3)
    db3 = (1 / m) * np.sum(dZ3, axis=0)

    dZ2 = np.matmul(dZ3, W3.T) * tanh_derv(A2)
    dW2 = (1 / m) * np.matmul(A1.T, dZ2)
    db2 = (1 / m) * np.sum(dZ2, axis=0)

    dZ1 = np.matmul(dZ2, W2.T) * tanh_derv(A1)
    dW1 = (1 / m) * np.matmul(X.T, dZ1)
    db1 = (1 / m) * np.sum(dZ1, axis=0)

    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}


def update_parameters(parameters, grads, learning_rate):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    dW1 = grads['dW1']
    dW2 = grads['dW2']
    dW3 = grads['dW3']

    db1 = grads['db1']
    db2 = grads['db2']
    db3 = grads['db3']

    W1 = W1 - dW1 * learning_rate
    W2 = W2 - dW2 * learning_rate
    W3 = W3 - dW3 * learning_rate

    b1 = b1 - db1 * learning_rate
    b2 = b2 - db2 * learning_rate
    b3 = b3 - db3 * learning_rate

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}


def cost(A3, Y):
    m = Y.shape[0]
    logprobs = (1 / m) * np.sum(np.multiply(np.log(A3), Y) + np.multiply(np.log(1 - A3), (1 - Y)))
    cost = - np.sum(logprobs)
    return cost


def predict(parameters, X):
    z = forward(parameters, X)
    preds = np.argmax(z['A3'], axis=1)
    return preds


def train(X, Y, n_hidden, learning_rate, n_iterations):
    n_input = X.shape[1]
    n_output = Y.shape[1]

    parameters = initialize_weights_and_biases(n_input, n_hidden, n_output)

    errors = []

    for i in range(n_iterations):
        activations = forward(parameters, X)
        grads = backprop(X, Y, parameters, activations)
        parameters = update_parameters(parameters, grads, learning_rate)

        if (i % 500 == 0):
            print("cost after {} iterations {}".format(i, str(cost(activations['A3'], Y))))
            pred = predict(parameters, X)
            y_true = Y.argmax(axis=1)
            print("accuracy after {} iterations {}%".format(i, accuracy_score(y_pred=pred, y_true=y_true) * 100))
            errors.append(accuracy_score(y_pred=pred, y_true=y_true) * 100)

    return parameters, errors


parameters, errors = train(x_train, y_train, n_hidden=5, learning_rate=0.07, n_iterations=5000)
p = predict(parameters, x_val)
l = np.argmax(y_val, axis=1)
print(p)
print(l)
print("modal accuracy with given set: {} %".format(100 - np.mean(np.abs(p - l)) * 100))

# plt.plot(errors)
# plt.show()
