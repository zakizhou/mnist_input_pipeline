from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import tensorflow as tf
import mnist
from mnist_input import mnist_inputs
from datetime import datetime
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
        validation_x, validation_y = mnist_inputs(MIN_AFTER_DEQUEUE, train=False, num_epochs=None)
        validation_logits = mnist.inference(validation_x)
        validation_accuracy = mnist.accuracy(validation_logits, validation_y)
        gradient_mean = averaged_gradients(tower_gradients)
        train_step = optimizer.apply_gradients(gradient_mean)
        init = tf.group(tf.initialize_all_variables(),
                        tf.initialize_local_variables())
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        start_time = datetime.now()
        try:
            index = 1
            while not coord.should_stop():
                _, loss_value = sess.run([train_step, loss])
                print("step: " + str(index) + " loss:" + str(loss_value))
                if index % 10 == 0:
                    accuracy = sess.run(validation_accuracy)
                    print("validation accuracy: "+str(accuracy))
                index += 1
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
            end_time = datetime.now()
            print("Time Consumption: " + str(end_time - start_time))
        except KeyboardInterrupt:
            print("keyboard interrupt detected, stop running")
            del sess

        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)

        sess.close()
        del sess


if __name__ == "__main__":
    main()
