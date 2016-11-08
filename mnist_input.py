from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import mnist


def inputs():
    batch_inputs = tf.placeholder(tf.float32,
                                  shape=[mnist.BATCH_SIZE, mnist.IMAGE_HEIGHT * mnist.IMAGE_WIDTH],
                                  name="inputs")

    batch_labels = tf.placeholder(tf.float32,
                                  shape=[mnist.BATCH_SIZE, mnist.NUM_CLASSES],
                                  name="labels")
    return batch_inputs, batch_labels


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(queue=filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64)
        })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    # height = tf.cast(features['height'], tf.int32)
    # width = tf.cast(features['width'], tf.int32)
    # depth = tf.cast(features['depth'], tf.int32)
    casted_image = tf.cast(image, tf.float32)
    casted_image.set_shape([mnist.IMAGE_HEIGHT * mnist.IMAGE_WIDTH])
    reshaped_image = tf.reshape(casted_image, [mnist.IMAGE_HEIGHT, mnist.IMAGE_WIDTH, mnist.IMAGE_DEPTH])
    return reshaped_image, label


def mnist_inputs(min_after_dequeue, train=True, num_epochs=1):
    train_or_validate = {True: "train.tfrecords",
                         False: "validation.tfrecords"}
    filename_queue = tf.train.string_input_producer(["/home/windows98/TensorFlow/mnist_tfrecords/"+train_or_validate[train]],
                                                    num_epochs=num_epochs)
    image, label = read_and_decode(filename_queue)
    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=mnist.BATCH_SIZE,
                                            capacity=min_after_dequeue + 3 * mnist.BATCH_SIZE,
                                            min_after_dequeue=min_after_dequeue)
    return images, tf.one_hot(tf.reshape(labels, [mnist.BATCH_SIZE]), depth=mnist.NUM_CLASSES, dtype=tf.float32)