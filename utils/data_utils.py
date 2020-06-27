import math

import matplotlib.pyplot as plt
import numpy as np


def compute_labels(set, sumAxis=0):
    sums = np.sum(set, sumAxis)
    labels = list(map(lambda x: 1 if int(x * 100) % 2 == 0 else 0, sums))
    return np.array(labels)


def generate_mini_batches(X, Y, batch_size=64, seed=0):
    batches = []
    m = X.shape[0]

    np.random.seed(seed)

    permutation = list(np.random.permutation(m))
    shuffleX = X[permutation, :]
    shuffleY = Y[permutation, :].reshape(m, Y.shape[1])

    num_batches = math.floor(m / batch_size)
    for i in range(0, num_batches):
        batchX = shuffleX[i * batch_size: (i + 1) * batch_size, :]
        batchY = shuffleY[i * batch_size: (i + 1) * batch_size, :]
        batch = (batchX, batchY)
        batches.append(batch)

    if m % batch_size != 0:
        batchX = shuffleX[num_batches * batch_size:, :]
        batchY = shuffleY[num_batches * batch_size:, :]
        batch = (batchX, batchY)
        batches.append(batch)

    return batches


def split_train_test_dataset(set, labels, split_by="rows", perc=0.1, verbose=False):
    if split_by == "rows":
        test_size = math.ceil(set.shape[0] * perc)
        train_set = set[:-test_size]
        test_set = set[-test_size:]
        train_labels = labels[:-test_size]
        test_labels = labels[-test_size:]
    else:
        test_size = math.ceil(set.shape[1] * perc)
        train_set = set[:, :-test_size]
        train_labels = labels[:, :-test_size]
        test_set = set[:, -test_size:]
        test_labels = labels[:, -test_size:]

    if verbose:
        print('training set shape ' + str(train_set.shape))
        print('training labels shape ' + str(train_labels.shape))
        print('test set shape ' + str(test_set.shape))
        print('test labels shape ' + str(test_labels.shape))

    return train_set, train_labels, test_set, test_labels


def plot_decision_boundary(model, X, Y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    print(xx.shape)
    print(yy.shape)

    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    print(Z.shape)

    plt.contourf(xx, yy, Z)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)
    plt.show()
