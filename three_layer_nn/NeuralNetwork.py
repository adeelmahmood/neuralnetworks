import numpy as np
import sys
sys.path.append(".")
from utils.data_utils import *
from utils.activation_functions import *
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    def __init__(self, seed):
        self.seed = seed

    def generate_mini_batches(self, X, Y, batch_size=64):
        batches = []
        m = X.shape[0]

        np.random.seed(self.seed)

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

    def initialize_weights_and_biases(self, layers_dim, seed=0):
        parameters = {}
        np.random.seed(self.seed)
        for i in range(len(layers_dim)-1):
            parameters["W" + str(i+1)] = 2 * (np.random.randn(layers_dim[i], layers_dim[i+1]) if (i<len(layers_dim)-2) else np.random.rand(layers_dim[i], layers_dim[i+1])) - 1
            parameters["b" + str(i+1)] = np.zeros((1, layers_dim[i+1]))
        return parameters

    def initialize_momentum(self, parameters):
        v = {}
        for i in range(len(parameters)//2):
            v["dW" + str(i+1)] = np.zeros_like(parameters["W" + str(i+1)])
            v["db" + str(i+1)] = np.zeros_like(parameters["b" + str(i+1)])
        return v

    def initialize_adam(self, parameters):
        v = {}
        s = {}
        for i in range(len(parameters)//2):
            v["dW" + str(i+1)] = np.zeros_like(parameters["W" + str(i+1)])
            v["db" + str(i+1)] = np.zeros_like(parameters["b" + str(i+1)])
            s["dW" + str(i+1)] = np.zeros_like(parameters["W" + str(i+1)])
            s["db" + str(i+1)] = np.zeros_like(parameters["b" + str(i+1)])
        return v, s

    def forward(self, parameters, X, activation_functions):
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        W3 = parameters['W3']
        b3 = parameters['b3']

        Z1 = np.matmul(X, W1) + b1
        A1 = activation_functions[0](Z1)

        Z2 = np.matmul(A1, W2) + b2
        A2 = activation_functions[1](Z2)

        Z3 = np.matmul(A2, W3) + b3
        A3 = activation_functions[2](Z3)

        return {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3}

    def forward_with_dropout(self, parameters, X, activation_functions, keep_prob=0.5, seed=0):
        np.random.seed(self.seed)

        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        W3 = parameters['W3']
        b3 = parameters['b3']

        Z1 = np.matmul(X, W1) + b1
        A1 = activation_functions[0](Z1)
        D1 = np.random.rand(A1.shape[0], A1.shape[1])
        D1 = (D1 < keep_prob).astype(int)
        A1 = A1 * D1
        A1 = A1 / keep_prob

        Z2 = np.matmul(A1, W2) + b2
        A2 = activation_functions[1](Z2)
        D2 = np.random.rand(A2.shape[0], A2.shape[1])
        D2 = (D2 < keep_prob).astype(int)
        A2 = A2 * D2
        A2 = A2 / keep_prob

        Z3 = np.matmul(A2, W3) + b3
        A3 = activation_functions[2](Z3)

        return {'Z1': Z1, 'D1': D1, 'A1': A1, 'Z2': Z2, 'D2': D2, 'A2': A2, 'Z3': Z3, 'A3': A3}

    def backprop(self, X, Y, parameters, activations, activation_functions_dervs):
        m = X.shape[0]
        W3 = parameters['W3']
        W2 = parameters['W2']
        A3 = activations['A3']
        A2 = activations['A2']
        A1 = activations['A1']

        dZ3 = A3 - Y
        dW3 = (1/m) * np.matmul(A2.T, dZ3)
        db3 = (1/m) * np.sum(dZ3, axis=0)

        dZ2 = np.matmul(dZ3, W3.T) * activation_functions_dervs[0](A2)
        dW2 = (1/m) * np.matmul(A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0)

        dZ1 = np.matmul(dZ2, W2.T) * activation_functions_dervs[1](A1)
        dW1 = (1/m) * np.matmul(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0)

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}

    def backprop_with_regularization(self, X, Y, parameters, activations, activation_functions_dervs, lambd):
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

        dZ2 = np.matmul(dZ3, W3.T) * activation_functions_dervs[0](A2)
        dW2 = (1/m) * np.matmul(A1.T, dZ2) + (lambd * W2)/m
        db2 = (1/m) * np.sum(dZ2, axis=0)

        dZ1 = np.matmul(dZ2, W2.T) * activation_functions_dervs[1](A1)
        dW1 = (1/m) * np.matmul(X.T, dZ1) + (lambd * W1)/m
        db1 = (1/m) * np.sum(dZ1, axis=0)

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}

    def backprop_with_dropout(self, X, Y, parameters, activations, activation_functions_dervs, keep_prob=0.5):
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

        dZ2 = dA2 * activation_functions_dervs[0](A2)
        dW2 = (1/m) * np.matmul(A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0)

        dA1 = np.matmul(dZ2, W2.T)
        dA1 = dA1 * D1
        dA1 = dA1 / keep_prob

        dZ1 = dA1 * activation_functions_dervs[1](A1)
        dW1 = (1/m) * np.matmul(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0)

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}


    def update_parameters(self, parameters, grads, learning_rate):
        for i in range(len(parameters)//2):
            parameters["W" + str(i+1)] = parameters["W" + str(i+1)] - grads["dW" + str(i+1)] * learning_rate
            parameters["b" + str(i+1)] = parameters["b" + str(i+1)] - grads["db" + str(i+1)] * learning_rate
        return parameters

    def update_parameters_with_momentum(self, parameters, grads, learning_rate, v, beta):
        for i in range(len(parameters)//2):
            v["dW" + str(i+1)] = beta * v["dW" + str(i+1)] + (1 - beta) * grads["dW" + str(i+1)]
            v["db" + str(i+1)] = beta * v["db" + str(i+1)] + (1 - beta) * grads["db" + str(i+1)]

            parameters["W" + str(i+1)] = parameters["W" + str(i+1)] - v["dW" + str(i+1)] * learning_rate
            parameters["b" + str(i+1)] = parameters["b" + str(i+1)] - v["db" + str(i+1)] * learning_rate
        return parameters, v

    def update_parameters_with_adam(self, parameters, grads, v, s, t, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
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

    def cost(self, A, Y):
        m = Y.shape[0]
        logprobs = (1/m)*np.sum(np.multiply(np.log(A),Y) + np.multiply(np.log(1-A),(1-Y)))
        cost = - np.sum(logprobs)
        return cost

    def cost_with_regularization(self, A, Y, parameters, lambd):
        m = Y.shape[0]
        logprobs = (1/m)*np.sum(np.multiply(np.log(A),Y) + np.multiply(np.log(1-A),(1-Y)))
        cost = - np.sum(logprobs)
        L2_regularization_cost = (1/m) * (lambd / 2) * (np.sum(np.square(parameters['W1'])) + np.sum(np.square(parameters['W2'])) + np.sum(np.square(parameters['W3'])))
        return cost + L2_regularization_cost

    def predict(self, parameters, X, Y, activation_functions, predic_func):
        activations = self.forward(parameters, X, activation_functions)
        A3 = activations["A3"]
#         predictions = (A3 > 0.5).astype(int)
        predictions = predic_func(A3)
        print("modal accuracy with given set: {} %".format(100 - np.mean(np.abs(predictions - Y)) * 100))
        return predictions

    def predict_s(self, parameters, X, activation_functions, predic_func):
        activations = self.forward(parameters, X, activation_functions)
        A3 = activations["A3"]
#         predictions = (A3 > 0.5).astype(int)
        predictions = predic_func(A3)
        return predictions

    def train(self,
        X, Y, n_hidden, learning_rate, n_iterations,
        activation_functions, activation_functions_dervs,
        keep_prob=1, lambd=0, optimizer="gd",
        batch_size=64, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8,
        printCost=False):

        n_input = X.shape[1]
        n_output = Y.shape[1]
        t = 0

        parameters = self.initialize_weights_and_biases([n_input, n_hidden, n_hidden, n_output])

        if (optimizer == "momentum"):
            v = self.initialize_momentum(parameters)
        elif (optimizer == "adam"):
            v, s = self.initialize_adam(parameters)

        errors = []
        for i in range(n_iterations):
            if(keep_prob == 1):
                activations = self.forward(parameters, X, activation_functions)
            elif(keep_prob != 0):
                activations = self.forward_with_dropout(parameters, X, activation_functions, keep_prob)

            if(lambd == 0 and keep_prob == 1):
                grads = self.backprop(X, Y, parameters, activations, activation_functions_dervs)
            elif(lambd != 0):
                grads = self.backprop_with_regularization(X, Y, parameters, activations, activation_functions_dervs, lambd)
            elif(keep_prob < 1):
                grads = self.backprop_with_dropout(X, Y, parameters, activations, activation_functions_dervs, keep_prob)

            if (optimizer == "gd"):
                parameters = self.update_parameters(parameters, grads, learning_rate)
            elif (optimizer == "momentum"):
                parameters, v = self.update_parameters_with_momentum(parameters, grads, learning_rate, v, beta)
            elif (optimizer == "adam"):
                t = t + 1
                parameters, v, s = self.update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)

            if(printCost == True and i % 1000 == 0):
                if(lambd == 0):
                    print("cost after {} iterations {}".format(i, str(self.cost(activations['A3'], Y))))
                elif(lambd != 0):
                    print("cost after {} iterations {}".format(i, str(self.cost_with_regularization(activations['A3'], Y, parameters, lambd))))

        return parameters, errors
