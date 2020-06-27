import sys

import pandas as pd

sys.path.append(".")
from utils.activation_functions import *
from sklearn.model_selection import train_test_split
from three_layer_nn.NeuralNetwork import NeuralNetwork

df = pd.read_csv('three_layer_nn/wines-data.csv')
# print(df.head())

labels = df[['Cultivar 1', 'Cultivar 2', 'Cultivar 3']].values

training_set = df.drop(['Cultivar 1', 'Cultivar 2', 'Cultivar 3'], axis=1)

training_set = training_set.values

x_train, x_val, y_train, y_val = train_test_split(training_set, labels, test_size=0.2, random_state=20)

nn = NeuralNetwork(0)

activation_functions = [np.tanh, np.tanh, softmax]
activation_functions_dervs = [tanh_derv, tanh_derv]

n_hidden = 5
learning_rate = 0.07
n_iterations = 5000
opt = "gd"

l = np.argmax(y_val, axis=1)

for train in [nn.train, nn.train_in_batches]:
    for opt in ["gd", "momentum", "adam"]:
        print('\n:: gradient descent - optimizer = ' + opt + ' - train = ' + str(train.__name__))
        parameters, errors = train(x_train, y_train, n_hidden=n_hidden,
                                   learning_rate=learning_rate, n_iterations=n_iterations,
                                   activation_functions=activation_functions,
                                   activation_functions_dervs=activation_functions_dervs,
                                   optimizer=opt, printCost=False)
        p = nn.predict_s(parameters, x_val, activation_functions, lambda x: np.argmax(x, axis=1))
        print("modal accuracy with given set: {} %".format(100 - np.mean(np.abs(p - l)) * 100))

        print(':: gradient descent with regularization')
        parameters, errors = nn.train(x_train, y_train, n_hidden=n_hidden,
                                      learning_rate=learning_rate, n_iterations=n_iterations,
                                      activation_functions=activation_functions,
                                      activation_functions_dervs=activation_functions_dervs,
                                      optimizer=opt, printCost=False, lambd=0.1)
        p = nn.predict_s(parameters, x_val, activation_functions, lambda x: np.argmax(x, axis=1))
        print("modal accuracy with given set: {} %".format(100 - np.mean(np.abs(p - l)) * 100))

        print(':: gradient descent with drop out')
        parameters, errors = nn.train(x_train, y_train, n_hidden=n_hidden,
                                      learning_rate=learning_rate, n_iterations=n_iterations,
                                      activation_functions=activation_functions,
                                      activation_functions_dervs=activation_functions_dervs,
                                      optimizer=opt, printCost=False, keep_prob=0.8)
        p = nn.predict_s(parameters, x_val, activation_functions, lambda x: np.argmax(x, axis=1))
        print("modal accuracy with given set: {} %".format(100 - np.mean(np.abs(p - l)) * 100))
