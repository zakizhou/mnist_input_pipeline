from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import tensorflow as tf
import mnist
from mnist_input import mnist_inputs

NUM_STEPS = 300
LEARNING_RATE = 0.0001
NUM_GPUS = 2
TOWER_NAME = "tower"
FRACTION = 0.4
MIN_AFTER_DEQUEUE = int(mnist.NUM_EXAMPLES_PER_EPOCH * FRACTION)


def tower_loss(scope):
    images, labels = mnist_inputs(min_after_dequeue=MIN_AFTER_DEQUEUE)
    logits = mnist.inference(images)
    _ = mnist.loss(logits, labels)
    loss = tf.get_collection("loss", scope=scope)[0]
    return loss


def averaged_gradients(tower_gradients):
    gradients_mean = []
    for gradient_variable in zip(*tower_gradients):
        gradients = []
        for gradient, _ in gradient_variable:
            expanded_gradient = tf.expand_dims(gradient, 0)
            gradients.append(expanded_gradient)
        concat = tf.concat(0, gradients)
        averaged_gradient = tf.reduce_mean(concat, reduction_indices=0)
        grad_var_tuple = (averaged_gradient, gradient_variable[0][1])
        gradients_mean.append(grad_var_tuple)
    return gradients_mean


def main():
    with tf.Graph().as_default(), tf.device("/cpu:0"):
        tower_gradients = []
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        for gpu_index in range(NUM_GPUS):
            gpu_name = "/gpu:"+str(gpu_index)
            with tf.device(gpu_name):
                with tf.name_scope("%s_%d" % (TOWER_NAME, gpu_index)) as scope:
                    loss = tower_loss(scope)
                    tf.get_variable_scope().reuse_variables()
                    gradient = optimizer.compute_gradients(loss)
                    tower_gradients.append(gradient)
        gradient_mean = averaged_gradients(tower_gradients)
        train_step = optimizer.apply_gradients(gradient_mean)
        init = tf.group(tf.initialize_all_variables(),
                        tf.initialize_local_variables())
        with tf.Session() as sess:
            sess.run(init)
            for index in range(NUM_STEPS):
                _, loss_value = sess.run([train_step, loss])
                print("step: "+str(loss_value))


if __name__ == "__main__":
    main()
