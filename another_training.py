import numpy as np


def sigmoid(z, deriv=False):
    if (deriv == True):
        return z * (1 - z)

    s = 1 / (1 + np.exp(-z))
    return s


x = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# define training set and labels
train_set = np.random.rand(3, 4)
labels = np.random.rand(4, 1)

# train the model
for i in range(10000):

    # activations
    l0 = x
    l1 = sigmoid(np.dot(l0, train_set))
    l2 = sigmoid(np.dot(l1, labels))

    # find error
    l2_error = y - l2

    if (i % 1000 == 0):
        print("Error: " + str(np.mean(np.abs(l2_error))))

    # which direction to fix the error
    l2_delta = l2_error * sigmoid(l2, deriv=True)

    # prop back to previous layer
    l1_error = l2_delta.dot(labels.T)

    # which direction for prevoius layer
    l1_delta = l1_error * sigmoid(l1, deriv=True)

    # update weights
    train_set += l1.T.dot(l2_delta)
    labels += l0.T.dot(l1_delta)
