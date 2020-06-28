import pandas as pd
import tensorflow as tf
import sys
from tensorflow.python.framework import ops

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


def create_placeholders(n_input, n_output):
    X = tf.placeholder(tf.float32, shape=[None, n_input], name="X")
    Y = tf.placeholder(tf.float32, shape=[None, n_output], name="Y")
    return X, Y


def initialize_parameters():
    W1 = tf.get_variable("W1", [784, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    W2 = tf.get_variable("W2", [25, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    W3 = tf.get_variable("W3", [12, 24], initializer=tf.contrib.layers.xavier_initializer(seed=1))

    b1 = tf.get_variable("b1", [1, 25], initializer=tf.zeros_initializer())
    b2 = tf.get_variable("b2", [1, 12], initializer=tf.zeros_initializer())
    b3 = tf.get_variable("b3", [1, 24], initializer=tf.zeros_initializer())

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    return parameters


def forward(X, parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]

    Z1 = tf.add(tf.matmul(X, W1), b1)
    A1 = tf.nn.relu(Z1)

    Z2 = tf.add(tf.matmul(A1, W2), b2)
    A2 = tf.nn.relu(Z2)

    Z3 = tf.add(tf.matmul(A2, W3), b3)

    return Z3


def compute_cost(Z3, Y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))


def model(train_s, train_l, test_s, test_l, learning_rate=0.0001, n_iterations=1500, batch_size=64,
          print_cost=True):
    (m, n_input) = train_s.shape
    n_output = train_l.shape[1]
    costs = []
    seed = 0

    ops.reset_default_graph()
    tf.set_random_seed(1)

    (X, Y) = create_placeholders(n_input, n_output)

    parameters = initialize_parameters()

    Z3 = forward(X, parameters)

    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(n_iterations):
            iter_cost = 0
            n_batches = (m / batch_size)
            seed = seed + 1

            batches = generate_mini_batches(train_s, train_l, batch_size=batch_size, seed=seed)

            for batch in batches:
                (batchX, batchY) = batch

                _, batch_cost = sess.run([optimizer, cost], feed_dict={X: batchX, Y: batchY})

                iter_cost += batch_cost / n_batches

            if print_cost == True and i % 5 == 0:
                print('cost after iteration %i: %f' % (i, iter_cost))
                costs.append(iter_cost)

        plt.plot(np.squeeze(costs))
        plt.show()

        parameters = sess.run(parameters)

        pred = tf.equal(tf.argmax(Y, 1), tf.argmax(Z3, 1))
        accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

        print("Train accuracy ", accuracy.eval({X: train_s, Y: train_l}))
        print("Test accuracy ", accuracy.eval({X: test_s, Y: test_l}))

    return Z3, parameters


model(train_data, train_labels, test_data, test_labels, n_iterations=100)
