from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import tensorflow as tf
import mnist
from mnist_input import mnist_inputs
NUM_STEPS = 300


def main():
    fraction = 0.4
    min_after_dequeue = int(mnist.NUM_EXAMPLES_PER_EPOCH * fraction)
    images, labels = mnist_inputs(min_after_dequeue)
    validation_images, validation_labels = mnist_inputs(2000, train=False, num_epochs=None)
    with tf.variable_scope("inference") as scope:
        logits = mnist.inference(images)
        scope.reuse_variables()
        validation_logits = mnist.inference(validation_images)
    loss = mnist.loss(logits, labels)
    tf.scalar_summary("cross_entropy", loss)
    accuracy = mnist.accuracy(validation_logits, validation_labels)
    tf.scalar_summary("validation_accuracy", accuracy)
    train_op = mnist.train(loss)
    sess = tf.Session()
    sess.run(tf.initialize_local_variables())
    sess.run(tf.initialize_all_variables())
    tf.train.start_queue_runners(sess=sess)
    merge = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("/home/windows98/PycharmProjects/mnist/Summary/")
    for index in range(NUM_STEPS):
        _, loss_value, summary = sess.run([train_op, loss, merge])
        writer.add_summary(summary, index+1)
        # accuracy_score, summary = sess.run([accuracy, summary])
        # writer.add_summary(summary, index+1)
        print("step:"+str(index+1)+" loss: "+str(loss_value))
        # print("step:"+str(index+1)+" loss: "+str(loss_value)+" accuracy: "+str(accuracy_score))


if __name__ == "__main__":
    main()