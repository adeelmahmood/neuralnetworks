import sys

sys.path.append(".")
from utils.data_utils import *
from utils.activation_functions import *

n_features = 13
n_samples = 100

np.random.seed(0)

set = np.random.randn(n_samples, n_features)
lbls = compute_labels(set, 1).reshape(n_samples, 1)

training_set, training_labels, test_set, test_labels = split_train_test_dataset(set, lbls, split_by="rows", perc=0.1)


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
    A3 = sigmoid(Z3)

    return {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3}


def forward_with_dropout(parameters, X, keep_prob=0.5):
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
    dW3 = (1 / m) * np.matmul(A2.T, dZ3)
    db3 = (1 / m) * np.sum(dZ3, axis=0)

    dZ2 = np.matmul(dZ3, W3.T) * tanh_derv(A2)
    dW2 = (1 / m) * np.matmul(A1.T, dZ2)
    db2 = (1 / m) * np.sum(dZ2, axis=0)

    dZ1 = np.matmul(dZ2, W2.T) * tanh_derv(A1)
    dW1 = (1 / m) * np.matmul(X.T, dZ1)
    db1 = (1 / m) * np.sum(dZ1, axis=0)

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
    dW3 = (1 / m) * np.matmul(A2.T, dZ3) + (lambd * W3) / m
    db3 = (1 / m) * np.sum(dZ3, axis=0)

    dZ2 = np.matmul(dZ3, W3.T) * tanh_derv(A2)
    dW2 = (1 / m) * np.matmul(A1.T, dZ2) + (lambd * W2) / m
    db2 = (1 / m) * np.sum(dZ2, axis=0)

    dZ1 = np.matmul(dZ2, W2.T) * tanh_derv(A1)
    dW1 = (1 / m) * np.matmul(X.T, dZ1) + (lambd * W1) / m
    db1 = (1 / m) * np.sum(dZ1, axis=0)

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
    dW3 = (1 / m) * np.matmul(A2.T, dZ3)
    db3 = (1 / m) * np.sum(dZ3, axis=0)

    dA2 = np.matmul(dZ3, W3.T)
    dA2 = dA2 * D2
    dA2 = dA2 / keep_prob

    dZ2 = dA2 * tanh_derv(A2)
    dW2 = (1 / m) * np.matmul(A1.T, dZ2)
    db2 = (1 / m) * np.sum(dZ2, axis=0)

    dA1 = np.matmul(dZ2, W2.T)
    dA1 = dA1 * D1
    dA1 = dA1 / keep_prob

    dZ1 = dA1 * tanh_derv(A1)
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


def cost(A, Y):
    m = Y.shape[0]
    logprobs = (1 / m) * np.sum(np.multiply(np.log(A), Y) + np.multiply(np.log(1 - A), (1 - Y)))
    cost = - np.sum(logprobs)
    return cost


def cost_with_regularization(A, Y, parameters, lambd):
    m = Y.shape[0]
    logprobs = (1 / m) * np.sum(np.multiply(np.log(A), Y) + np.multiply(np.log(1 - A), (1 - Y)))
    cost = - np.sum(logprobs)
    L2_regularization_cost = (1 / m) * (lambd / 2) * (
            np.sum(np.square(parameters['W1'])) + np.sum(np.square(parameters['W2'])) + np.sum(
        np.square(parameters['W3'])))
    return cost + L2_regularization_cost


def predict(parameters, X, Y):
    activations = forward(parameters, X)
    A3 = activations["A3"]
    predictions = (A3 > 0.5).astype(int)
    print("modal accuracy with given set: {} %".format(100 - np.mean(np.abs(predictions - Y)) * 100))
    #     print(Y)
    #     print(predictions)
    return predictions


def train(X, Y, n_hidden, learning_rate, n_iterations, keep_prob=1, lambd=0):
    n_input = X.shape[1]
    n_output = Y.shape[1]

    parameters = initialize_weights_and_biases(n_input, n_hidden, n_output)

    errors = []

    for i in range(n_iterations):
        if (keep_prob == 1):
            activations = forward(parameters, X)
        elif (keep_prob != 0):
            activations = forward_with_dropout(parameters, X, keep_prob)

        if (lambd == 0 and keep_prob == 1):
            grads = backprop(X, Y, parameters, activations)
        elif (lambd != 0):
            grads = backprop_with_regularization(X, Y, parameters, activations, lambd)
        elif (keep_prob < 1):
            grads = backprop_with_dropout(X, Y, parameters, activations, keep_prob)

        parameters = update_parameters(parameters, grads, learning_rate)

        if (i % 1000 == 0):
            if (lambd == 0):
                print("cost after {} iterations {}".format(i, str(cost(activations['A3'], Y))))
            elif (lambd != 0):
                print("cost after {} iterations {}".format(i, str(
                    cost_with_regularization(activations['A3'], Y, parameters, lambd))))

    return parameters, errors


parameters, errors = train(training_set, training_labels, n_hidden=5, learning_rate=0.07, n_iterations=10000, lambd=0.1)
print('predicting with training set')
predictions = predict(parameters, training_set, training_labels)
print('predicting with test set')
predictions = predict(parameters, test_set, test_labels)
print(test_labels.T)
print(predictions.T)
