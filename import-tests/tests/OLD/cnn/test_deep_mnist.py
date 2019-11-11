import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph

MNIST = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)


class MnistImages(TestGraph):
    def __init__(self, test_size=100, *args, **kwargs):
        super(MnistImages, self).__init__(*args, **kwargs)
        self.test_input = MNIST.test.images[:test_size].reshape((test_size, 28, 28, 1))
        self.test_target = MNIST.test.labels[:test_size]

    def list_inputs(self):
        return ["input", "keep_prob"]

    def get_placeholder_input(self, name):
        if name == "input":
            return self.test_input
        if name == "keep_prob":
            return 1.0
        if name == "target":
            return self.test_target

    def _get_placeholder_shape(self, name):
        if name == "input":
            return [None, 28, 28, 1]
        if name == "keep_prob":
            return []
        if name == "target":
            return [None, 10]


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def test_deep_mnist():
    mnist_images = MnistImages(seed=13, test_size=100)
    x_image = mnist_images.get_placeholder("input", data_type=tf.float32)
    y_ = mnist_images.get_placeholder("target", data_type=tf.float32)
    keep_prob = mnist_images.get_placeholder("keep_prob", data_type=tf.float32)
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    out_node = tf.nn.softmax(y_conv, name="output")

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    training_steps = 500
    display_step = 100
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    batch_size = 128
    for step in range(1, training_steps + 1):
        batch_x, batch_y = MNIST.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((-1, 28, 28, 1))
        # Run optimization op (backprop)
        sess.run(train_step, feed_dict={x_image: batch_x, y_: batch_y, keep_prob: 0.6})  # dropout tied to 0.6 for train
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            train_accuracy = sess.run(accuracy, feed_dict={
                x_image: batch_x, y_: batch_y, keep_prob: 0.6})
            print('step %d, training accuracy %g' % (step, train_accuracy))
    print("Optimization Finished!")
    print('test accuracy %g' % sess.run(accuracy, feed_dict={
        x_image: mnist_images.test_input, y_: mnist_images.test_target,
        keep_prob: mnist_images.get_placeholder_input("keep_prob")}))

    out_after_train = sess.run([out_node], feed_dict={x_image: mnist_images.test_input,
                                                      keep_prob: mnist_images.get_placeholder_input("keep_prob")})

    placeholders = [x_image, keep_prob]
    predictions = [out_node]
    tfp = TensorFlowPersistor(save_dir="deep_mnist", verbose=False)
    predictions_after_freeze = tfp \
        .set_training_sess(sess) \
        .set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(mnist_images.get_test_data()) \
        .build_save_frozen_graph()
    for before, after in zip(out_after_train, predictions_after_freeze):
        np.testing.assert_equal(before, after)


if __name__ == "main":
    test_deep_mnist()
