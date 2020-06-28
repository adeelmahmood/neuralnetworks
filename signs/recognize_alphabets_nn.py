import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import sys

sys.path.append(".")
from three_layer_nn.NeuralNetwork import NeuralNetwork
from utils.activation_functions import *

sys.path.append(".")
from utils.data_utils import *

train_data = pd.read_csv('~/Downloads/signs-dataset/sign_mnist_train.csv')
test_data = pd.read_csv('~/Downloads/signs-dataset/sign_mnist_test.csv')

train_labels = train_data["label"].values.ravel()
train_data.drop("label", axis=1, inplace=True)
train_labels = np.array(pd.get_dummies(train_labels))
train_data = train_data.values / 255.

test_labels = test_data["label"].values.ravel()
test_data.drop("label", axis=1, inplace=True)
test_labels = np.array(pd.get_dummies(test_labels))
test_data = test_data.values / 255.

n_iterations = 200
learning_rate = 0.07
seed = 0

nn = NeuralNetwork(seed)

activation_functions = [relu, relu, softmax]
activation_functions_dervs = [relu_derv, relu_derv]

np.random.seed(seed)

parameters, errors = nn.train_in_batches(train_data, train_labels, layers=[25, 12],
                                         learning_rate=learning_rate, n_iterations=n_iterations,
                                         activation_functions=activation_functions,
                                         activation_functions_dervs=activation_functions_dervs,
                                         optimizer="adam", lambd=0.1, print_cost=True, print_cost_freq=5)

plt.plot(errors)
plt.show()

l = np.argmax(train_labels, axis=1)
p = nn.predict_s(parameters, train_data, activation_functions, lambda x: np.argmax(x, axis=1))
print("modal accuracy with train set: {} %".format(np.mean(np.abs(p - l))))

l = np.argmax(test_labels, axis=1)
p = nn.predict_s(parameters, test_data, activation_functions, lambda x: np.argmax(x, axis=1))
print("modal accuracy with test set: {} %".format(np.mean(np.abs(p - l))))
