import numpy as np
import math

# np.random.seed(5)

def sigmoid(z):
  s = 1 / (1 + np.exp(-z))
  return s

def initialize_parameters(n_input, n_hidden, n_output):
    W1 = np.random.randn(n_hidden, n_input) * 0.01
    b1 = np.zeros((n_hidden, 1))
    W2 = np.random.randn(n_output, n_hidden) * 0.01
    b2 = np.zeros((n_output, 1))

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def forward(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.matmul(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.matmul(W2, A1) + b2
    A2 = sigmoid(Z2)

    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

def cost(A2, Y):
    m = Y.shape[1]
    logprobs = (1/m)*np.sum(np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),(1-Y)))
    cost = - np.sum(logprobs)

    return cost

def backward(X, Y, parameters, activations):
    m = X.shape[1]
    W2 = parameters["W2"]
    A1 = activations["A1"]
    A2 = activations["A2"]

    dZ2 = A2 - Y
    dW2 = (1/m) * np.matmul(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.matmul(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1/m) * np.matmul(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis = 1, keepdims = True)

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


def model(X, Y, hidden_layers, iterations):
    n_input = X.shape[0]
    n_output = Y.shape[0]
    parameters = initialize_parameters(n_input, hidden_layers, n_output)

    for i in range(iterations):
        activations = forward(X, parameters)
        cst = cost(activations["A2"], Y)
        grads = backward(X, Y, parameters, activations)
        parameters = update_parameters(parameters, grads, learning_rate = 1.2)

        if(i % 1000 == 0):
            print("cost after {} iterations {}".format(i, str(cst)))

    return parameters

def predict(parameters, X, Y):
    activations = forward(X, parameters)
    A2 = activations["A2"]
    predictions = (A2 > 0.5)
    print("modal accuracy with given set: {} %".format(100 - np.mean(np.abs(predictions - Y)) * 100))
#     print(Y)
#     print(predictions)
    return predictions

n_features = 2
n_samples = 100
n_hidden = 4
n_iterations = 10000
verbose = True

def compute_labels(set):
     sums = np.sum(set, axis=0)
     labels = list(map(lambda x: 1 if int(x*100) % 2 == 0 else 0, sums))
     return np.array(labels).reshape(1, set.shape[1])

# create model data
set = np.power(np.random.rand(n_features, n_samples), 2)
labels = compute_labels(set)

test_size = math.ceil((n_samples*10)/100)

# splice the sample dataset to extract training and test sets
training_set = set[:,:-test_size]
training_labels = labels[:,:-test_size]

test_set = set[:,-test_size:]
test_labels = labels[:,-test_size:]

if verbose:
  print('training set shape ' + str(training_set.shape))
  print('training labels shape ' + str(training_labels.shape))
  print('test set shape ' + str(test_set.shape))
  print('test labels shape ' + str(test_labels.shape))

parameters = model(training_set, training_labels, n_hidden, n_iterations)
predictions = predict(parameters, test_set, test_labels)