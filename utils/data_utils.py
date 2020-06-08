import numpy as np
import math
from sklearn.datasets import load_digits

def compute_labels(set, sumAxis=0):
     sums = np.sum(set, sumAxis)
     labels = list(map(lambda x: 1 if int(x*100) % 2 == 0 else 0, sums))
     return np.array(labels)

def split_train_test_dataset(set, labels, split_by = "rows", perc = 0.1, verbose = False):
    if(split_by == "rows"):
        test_size = math.ceil(set.shape[0]*perc)
        train_set = set[:-test_size]
        test_set = set[-test_size:]
        train_labels = labels[:-test_size]
        test_labels = labels[-test_size:]
    else:
        test_size = math.ceil(set.shape[1]*perc)
        train_set = set[:,:-test_size]
        train_labels = labels[:,:-test_size]
        test_set = set[:,-test_size:]
        test_labels = labels[:,-test_size:]

    if verbose:
      print('training set shape ' + str(train_set.shape))
      print('training labels shape ' + str(train_labels.shape))
      print('test set shape ' + str(test_set.shape))
      print('test labels shape ' + str(test_labels.shape))

    return train_set, train_labels, test_set, test_labels