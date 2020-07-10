import sys

import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.preprocessing import LabelBinarizer

sys.path.append(".")
from utils.data_utils import *

train_data = pd.read_csv('~/Downloads/signs-dataset/sign_mnist_train.csv')
test_data = pd.read_csv('~/Downloads/signs-dataset/sign_mnist_test.csv')

lb = LabelBinarizer()

train_labels = train_data['label']
train_labels = lb.fit_transform(train_labels)
del train_data['label']

train_data = train_data.values
train_data = np.array([np.reshape(i, (28, 28, 1)) for i in train_data])
train_data = train_data / 255

test_labels = test_data['label']
test_labels = lb.fit_transform(test_labels)
del test_data['label']

test_data = test_data.values
test_data = np.array([np.reshape(i, (28, 28, 1)) for i in test_data])
test_data = test_data / 255

# f, ax = plt.subplots(2,5)
# f.set_size_inches(10, 10)
# k = 0
# for i in range(2):
#     for j in range(5):
#         ax[i,j].imshow(train_data[k] , cmap = "gray")
#         k += 1
#     plt.tight_layout()

def create_placeholders(image_height, image_width, image_channels, output_classes):
    X = tf.placeholder(tf.float32, shape=[None, image_height, image_width, image_channels], name="X")
    Y = tf.placeholder(tf.float32, shape=[None, output_classes], name="Y")
    return X, Y


def initialize_parameters():
    W1 = tf.get_variable("W1", [3, 3, 1, 32], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    W2 = tf.get_variable("W2", [3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer(seed=1))

    parameters = {"W1": W1, "W2": W2}
    return parameters


def forward(X, parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]

#     CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    F = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(F, 24, activation_fn=None)

    return Z3


def compute_cost(Z3, Y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))


def model(train_s, train_l, test_s, test_l, learning_rate=0.009, n_iterations=100, batch_size=64,
          print_cost=True):
    seed = 0

    ops.reset_default_graph()
    # tf.set_random_seed(1)

    (X, Y) = create_placeholders(28, 28, 1, 24)

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

            if print_cost == True and i % 5 == 0:
                batch_cost, batch_accuracy = sess.run([cost, accuracy], feed_dict={X: batchX, Y: batchY})
                print('Iteration %i\t| Loss = %f\t| Accuracy = %f' % (i, batch_cost, batch_accuracy))

        # plt.plot(np.squeeze(costs))
        # plt.show()

        parameters = sess.run(parameters)

        print("Final  accuracy with test set ", sess.run([cost, accuracy], feed_dict={X: test_s, Y: test_l}))

    return Z3, parameters


model(train_data, train_labels, test_data, test_labels, n_iterations=20)
