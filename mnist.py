from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf


BATCH_SIZE = 128
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
IMAGE_DEPTH = 1
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH = 55000


def convolution(inputs, kernel_size):
    kernel = tf.get_variable("kernel",
                             shape=kernel_size,
                             initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    bias = tf.get_variable("bias",
                           shape=[kernel_size[-1]],
                           initializer=tf.constant_initializer(0., dtype=tf.float32))
    conv = tf.nn.conv2d(inputs,
                        filter=kernel,
                        strides=[1, 1, 1, 1],
                        padding="SAME",
                        name="conv")
    conv_bias = tf.nn.bias_add(conv, bias)
    relu = tf.nn.relu(conv_bias, name="relu")
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding="SAME",
                          name="pool")
    return pool


def inference(inputs):
    with tf.variable_scope("conv1"):
        conv_1 = convolution(inputs, [5, 5, 1, 64])

    with tf.variable_scope("conv2"):
        conv_2 = convolution(conv_1, [5, 5, 64, 128])

    reshape = tf.reshape(conv_2, [BATCH_SIZE, -1])
    dim = reshape.get_shape()[-1].value

    with tf.variable_scope("fully_connection"):
        weights = tf.get_variable("weights",
                                  shape=[dim, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable("bias",
                               shape=[256],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0., dtype=tf.float32))
        fully_con1 = tf.nn.xw_plus_b(reshape, weights, bias, name="output")

    with tf.variable_scope("output"):
        weights = tf.get_variable("weights",
                                  shape=[256, NUM_CLASSES],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable("bias",
                               shape=[NUM_CLASSES],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0., dtype=tf.float32))
        logits = tf.nn.xw_plus_b(fully_con1, weights, bias, name="logits")

    return logits


def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    loss = tf.reduce_mean(cross_entropy, name="loss")
    return loss


def accuracy(logits, labels):
    equal = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy_score = tf.reduce_mean(tf.cast(equal, tf.float32), name="accuracy")
    return accuracy_score


def train(loss, learning_rate=0.0001):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op