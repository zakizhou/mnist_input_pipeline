from mnist_input import mnist_inputs, read_and_decode
import tensorflow as tf


# def main():
input_queue = tf.train.string_input_producer(["/home/windows98/TensorFlow/mnist_tfrecords/train.tfrecords"])
# images, labels = mnist_inputs(10000)
# image, label = read_and_decode(input_queue)
images, labels, reshaped_lbs = mnist_inputs(10000)
sess = tf.Session()
sess.run(tf.initialize_local_variables())
sess.run(tf.initialize_local_variables())
tf.train.start_queue_runners(sess=sess)
img, lb, rsp_lbs = sess.run([images, labels, reshaped_lbs])


# if __name__ == "__main__":
#     main()
