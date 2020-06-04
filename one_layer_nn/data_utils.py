import numpy as np
import math
from sklearn.datasets import load_digits

def compute_labels(set):
     sums = np.sum(set, axis=0)
     labels = list(map(lambda x: 1 if int(x*100) % 2 == 0 else 0, sums))
     return np.array(labels).reshape(1, set.shape[1])

def random_data_set(n_features, n_samples, verbose = False):
    # create model data
    set = np.random.rand(n_features, n_samples)
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

    return training_set, training_labels, test_set, test_labels

def digits_data_set():
    return load_digits()
