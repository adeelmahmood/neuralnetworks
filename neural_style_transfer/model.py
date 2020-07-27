import os
import sys
import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from nst_utils import *
import imageio
from PIL import Image
import shutil

"""
Input Parameters - Start
"""

content_img_path = '/Users/adeelqureshi/Downloads/Photos/IMG_6230.jpg'
style_img_path = '/Users/adeelqureshi/Downloads/picaso2-style.jpg'

image_width = 400
image_height = 300
noise_threshold = 0.2

"""
Input Parameters - End
"""

images_dir = '/Users/adeelqureshi/neuralnetworks/neural_style_transfer/images/'

shutil.rmtree(images_dir)
os.mkdir(images_dir)

content_image = Image.open(content_img_path)
if content_image.size[0] != image_width or content_image.size[1] != image_height:
    new_content_image = content_image.resize((image_width, image_height))
    new_content_image.save(images_dir + 'content.jpg')
else:
    content_image.save(images_dir + 'content.jpg')

style_image = Image.open(style_img_path)
if style_image.size[0] != image_width or style_image.size[1] != image_height:
    new_style_image = style_image.resize((image_width, image_height))
    new_style_image.save(images_dir + 'style.jpg')
else:
    style_image.save(images_dir + 'style.jpg')

content_img = imageio.imread(images_dir + 'content.jpg')
style_img = imageio.imread(images_dir + 'style.jpg')


def compute_content_cost(a_C, a_G):
    """
    :param a_C: shape = [m, n_H, n_C, n_W]
    :param a_G: shape = [m, n_H, n_C, n_W]
    """
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    cost = (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(a_C - a_G))
    return cost


def gram_matrix(A):
    """
    :param A: shape = [n_C, n_H*n_W]
    """
    return tf.matmul(A, tf.transpose(A))


def compute_layer_style_cost(a_S, a_G):
    """
    :param a_C: shape = [m, n_H, n_C, n_W]
    :param a_G: shape = [m, n_H, n_C, n_W]
    """
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_S = tf.transpose(tf.reshape(a_S, shape=[-1, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, shape=[-1, n_C]))

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    cost = 1 / (2 * n_C * n_H * n_W) ** 2 * tf.reduce_sum(tf.square(GS - GG))
    return cost


def compute_layer_cost(model, LAYERS):
    cost = 0
    for layer, weight in LAYERS:
        out = model[layer]
        a_S = sess.run(out)
        a_G = out

        layer_cost = compute_layer_style_cost(a_S, a_G)
        cost += layer_cost * weight

    return cost


def total_cost(content_cost, style_cost, alpha=10, beta=40):
    return alpha * content_cost + beta * style_cost


tf.reset_default_graph()
sess = tf.InteractiveSession()

content_img = reshape_and_normalize_image(content_img)
style_img = reshape_and_normalize_image(style_img)
gen_image = generate_noise_image(content_img, image_width, image_height, noise_threshold)

model = load_vgg_model(image_width, image_height)

# -- compute content cost from content image --
sess.run(model['input'].assign(content_img))

out = model['conv4_2']

# activations from selected layer
a_C = sess.run(out)
# not evaluated yet - should be activations from given layer
a_G = out

# compute the content cost
content_cost = compute_content_cost(a_C, a_G)

# -- compute style cost from style image --
sess.run(model['input'].assign(style_img))

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

# compute the style cost
style_cost = compute_layer_cost(model, STYLE_LAYERS)

# total cost from both functions
total_cost = total_cost(content_cost, style_cost)

# define optimizer
optimizer = tf.train.AdamOptimizer(2.0)
train_step = optimizer.minimize(total_cost)


def model_nn(sess, input_img, n_iterations=200):
    sess.run(tf.global_variables_initializer())

    sess.run(model['input'].assign(input_img))

    for i in range(n_iterations):
        sess.run(train_step)
        generated_image = sess.run(model['input'])

        if i % 20 == 0:
            tc, cc, sc = sess.run([total_cost, content_cost, style_cost])
            print('Iteration %i\t\t| Content Cost = %f\t\t| Style Cost = %f\t\t| Total Cost = %f' % (i, cc, sc, tc))
            # save_image(images_dir + 'out-' + str(i) + '.png', generated_image)

    save_image(images_dir + 'final.png', generated_image)


model_nn(sess, gen_image)
