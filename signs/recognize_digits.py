import sys

import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops

sys.path.append(".")
from utils.data_utils import *

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def create_placeholders(n_input, n_output):
    X = tf.placeholder(tf.float32, shape=[None, n_input], name="X")
    Y = tf.placeholder(tf.float32, shape=[None, n_output], name="Y")
    return X, Y


def initialize_parameters():
    W1 = tf.get_variable("W1", [784, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    W2 = tf.get_variable("W2", [25, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    W3 = tf.get_variable("W3", [12, 10], initializer=tf.contrib.layers.xavier_initializer(seed=1))

    b1 = tf.get_variable("b1", [1, 25], initializer=tf.zeros_initializer())
    b2 = tf.get_variable("b2", [1, 12], initializer=tf.zeros_initializer())
    b3 = tf.get_variable("b3", [1, 10], initializer=tf.zeros_initializer())

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
    seed = 0

    ops.reset_default_graph()
    # tf.set_random_seed(1)

    (X, Y) = create_placeholders(n_input, n_output)

    parameters = initialize_parameters()

    Z3 = forward(X, parameters)

    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    pred = tf.equal(tf.argmax(Z3, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(n_iterations):
            seed = seed + 1

            batches = generate_mini_batches(train_s, train_l, batch_size=batch_size, seed=seed)

            for batch in batches:
                (batchX, batchY) = batch

                sess.run([optimizer], feed_dict={X: batchX, Y: batchY})

            if print_cost == True and i % 100 == 0:
                batch_cost, batch_accuracy = sess.run([cost, accuracy], feed_dict={X: batchX, Y: batchY})
                print('Iteration %i\t| Loss = %f\t| Accuracy = %f' % (i, batch_cost, batch_accuracy))

        # plt.plot(np.squeeze(costs))
        # plt.show()

        parameters = sess.run(parameters)

        print("Final  accuracy with test set ", sess.run([cost, accuracy], feed_dict={X: test_s, Y: test_l}))

    return Z3, parameters


model(mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels, n_iterations=400)
