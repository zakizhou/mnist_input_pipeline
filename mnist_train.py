from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist
from mnist_input import mnist_inputs, inputs
NUM_STEPS = 300
mnist_data = input_data.read_data_sets("/home/windows98/TensorFlow/mnist_data/", one_hot=True)


def main():
    images, labels = inputs()
    reshaped_images = tf.reshape(images, [mnist.BATCH_SIZE,
                                          mnist.IMAGE_HEIGHT,
                                          mnist.IMAGE_WIDTH,
                                          mnist.IMAGE_DEPTH])
    logits = mnist.inference(reshaped_images)
    loss = mnist.loss(logits, labels)
    accuracy = mnist.accuracy(logits, labels)
    train_op = mnist.train(loss)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        for index in range(NUM_STEPS):
            batch_x, batch_y = mnist_data.train.next_batch(mnist.BATCH_SIZE)
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict={
                                         images: batch_x,
                                         labels: batch_y
                                     })
            print("step:"+str(index+1)+" loss: "+str(loss_value))
            if (index+1) % 10 == 0:
                validation_x, validation_y = mnist_data.validation.next_batch(mnist.BATCH_SIZE)
                accuracy_score = sess.run(accuracy,
                                          feed_dict={
                                              images: validation_x,
                                              labels: validation_y
                                          })
                print("accuracy : "+str(accuracy_score))
if __name__ == "__main__":
    main()