import sys

sys.path.append(".")
from utils.data_utils import *
from utils.activation_functions import *
from three_layer_nn.NeuralNetwork import NeuralNetwork

n_features = 8
n_samples = 1000
n_iterations = 10000
learning_rate = 0.1
seed = 0

nn = NeuralNetwork(seed)

activation_functions = [np.tanh, np.tanh, sigmoid]
activation_functions_dervs = [tanh_derv, tanh_derv]

np.random.seed(seed)

set = np.random.randn(n_samples, n_features)
lbls = compute_labels(set, 1).reshape(n_samples, 1)

training_set, training_labels, test_set, test_labels = split_train_test_dataset(set, lbls, split_by="rows", perc=0.1)

for opt in ["gd", "momentum", "adam"]:
    print('\n:: gradient descent - optimizer = ' + opt)
    parameters, errors = nn.train(training_set, training_labels, layers=[3, 3],
                                  learning_rate=learning_rate, n_iterations=n_iterations,
                                  activation_functions=activation_functions,
                                  activation_functions_dervs=activation_functions_dervs,
                                  optimizer=opt)
    nn.predict(parameters, test_set, test_labels, activation_functions, lambda x: (x > 0.5).astype(int))

    print(':: gradient descent with regularization')
    parameters, errors = nn.train(training_set, training_labels, layers=[3, 3],
                                  learning_rate=learning_rate, n_iterations=n_iterations,
                                  activation_functions=activation_functions,
                                  activation_functions_dervs=activation_functions_dervs,
                                  optimizer=opt, lambd=0.1)
    nn.predict(parameters, test_set, test_labels, activation_functions, lambda x: (x > 0.5).astype(int))

    print(':: gradient descent with drop out')
    parameters, errors = nn.train(training_set, training_labels, layers=[3, 3],
                                  learning_rate=learning_rate, n_iterations=n_iterations,
                                  activation_functions=activation_functions,
                                  activation_functions_dervs=activation_functions_dervs,
                                  optimizer=opt, keep_prob=0.8)
    nn.predict(parameters, test_set, test_labels, activation_functions, lambda x: (x > 0.5).astype(int))
