import numpy as np
import math
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph


class SimpleAE(TestGraph):
    def __init__(self, *args, **kwargs):
        super(SimpleAE, self).__init__(*args, **kwargs)
        self.train_input, self.train_output = generate_input_output()
        self.n_input = self.train_input.shape[1]
        self.n_hidden = 2

    def get_placeholder_input(self, name):
        if name == "input":
            return self.train_input
        if name == "target":
            return self.train_output

    def _get_placeholder_shape(self, name):
        if name == "input":
            return [None, self.n_input]
        if name == "target":
            return [None, self.n_input]


def generate_input_output():
    np.random.seed(13)
    my_input = np.array([[2.0, 1.0, 1.0, 2.0],
                         [-2.0, 1.0, -1.0, 2.0],
                         [0.0, 1.0, 0.0, 2.0],
                         [0.0, -1.0, 0.0, -2.0],
                         [0.0, -1.0, 0.0, -2.0]])

    my_output = np.array([[2.0, 1.0, 1.0, 2.0],
                          [-2.0, 1.0, -1.0, 2.0],
                          [0.0, 1.0, 0.0, 2.0],
                          [0.0, -1.0, 0.0, -2.0],
                          [0.0, -1.0, 0.0, -2.0]])
    noisy_input = my_input + .2 * np.random.random_sample((my_input.shape)) - .1
    # Scale to [0,1]
    scaled_input_1 = np.divide((noisy_input - noisy_input.min()), (noisy_input.max() - noisy_input.min()))
    scaled_output_1 = np.divide((my_output - my_output.min()), (my_output.max() - my_output.min()))
    # Scale to [-1,1]
    input_data = (scaled_input_1 * 2) - 1
    output_data = (scaled_output_1 * 2) - 1
    return input_data, output_data


def test_simple_ae():
    # Placeholders
    simple_ae = SimpleAE(seed=1337)
    in_node = simple_ae.get_placeholder("input")
    expected_out = simple_ae.get_placeholder("target")
    # Hidden layer
    Wh = tf.Variable(tf.random_uniform((simple_ae.n_input, simple_ae.n_hidden), -1.0 / math.sqrt(simple_ae.n_input),
                                       1.0 / math.sqrt(simple_ae.n_input), dtype=tf.float64))
    bh = tf.Variable(tf.random_normal([simple_ae.n_hidden], dtype=tf.float64))
    h = tf.nn.tanh(tf.nn.bias_add(tf.matmul(in_node, Wh), bh))
    # Output layer
    Wo = tf.transpose(Wh)  # tied weights
    bo = tf.Variable(tf.zeros([simple_ae.n_input], dtype=tf.float64))
    out_node = tf.nn.tanh(tf.nn.bias_add(tf.matmul(h, Wo), bo), name="output")
    meansq = tf.reduce_mean(tf.square(expected_out - out_node))
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(meansq)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(10):
        sess.run(train_step, feed_dict={in_node: simple_ae.train_input, expected_out: simple_ae.train_output})

    placeholders = [in_node]
    predictions = [out_node]

    out_after_train = sess.run(predictions, feed_dict={in_node: simple_ae.train_input})

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="ae_00")
    predictions_after_freeze = tfp \
        .set_training_sess(sess) \
        .set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(simple_ae.get_test_data()) \
        .build_save_frozen_graph()

    for before, after in zip(out_after_train, predictions_after_freeze):
        np.testing.assert_equal(before, after)


if __name__ == '__main__':
    test_simple_ae()
