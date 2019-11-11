import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph
from tensorflow.examples.tutorials.mnist import input_data

MNIST = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)


class ModifiedMNIST(TestGraph):
    def get_placeholder_input(self, name):
        if name == "input":
            return MNIST.test.images[:100, :]

    def _get_placeholder_shape(self, name):
        if name == "input":
            return [None, 784]
        if name == "target":
            return [None, 10]


def test_modified_mnist():
    modified_mnist = ModifiedMNIST(seed=1337)
    image_in = modified_mnist.get_placeholder("input", data_type=tf.float32)
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.bias_add(tf.matmul(image_in, W), b)
    image_label = modified_mnist.get_placeholder("target", data_type=tf.float32)
    cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=image_label, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy_mean)
    softmax_out = tf.nn.softmax(y, name="output")

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    for _ in range(50):
        batch_xs, batch_ys = MNIST.train.next_batch(100)
        sess.run(train_step, feed_dict={image_in: batch_xs, image_label: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(image_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={image_in: MNIST.test.images,
                                        image_label: MNIST.test.labels}))

    placeholders = [image_in]
    predictions = [softmax_out]
    out_after_train = sess.run(predictions, feed_dict={image_in: modified_mnist.get_placeholder_input("input")})

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="mnist_00", verbose=False)
    predictions_after_freeze = tfp \
        .set_training_sess(sess) \
        .set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(modified_mnist.get_test_data()) \
        .build_save_frozen_graph()

    for before, after in zip(out_after_train, predictions_after_freeze):
        np.testing.assert_equal(before, after)


if __name__ == '__main__':
    test_modified_mnist()
