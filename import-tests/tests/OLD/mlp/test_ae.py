import numpy as np
import math
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph


class SelfCodingNetwork(TestGraph):
    def __init__(self, n_input=10, n_hidden_1=10, n_hidden_2=5, *args, **kwargs):
        super(SelfCodingNetwork, self).__init__(*args, **kwargs)
        self.n_input = n_input
        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
            'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
        }
        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([n_input]))
        }
        self.n_input = n_input
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2

    def get_placeholder_input(self, name):
        if name == "input":
            return np.reshape(np.linspace(1, self.n_input, self.n_input), (1, self.n_input))

    def _get_placeholder_shape(self, name):
        if name == "input":
            return [None, self.n_input]

    def encoder(self, x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']),
                                       self.biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
                                       self.biases['encoder_b2']), name="output")
        return layer_2

    def decoder(self, x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']),
                                       self.biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
                                       self.biases['decoder_b2']))
        return layer_2


def test_self_coding_network():
    self_coding_network = SelfCodingNetwork(n_input=676, seed=123, n_hidden_1=60, n_hidden_2=2)
    in_node = self_coding_network.get_placeholder("input", data_type=tf.float32)
    # define model
    encoder_op = self_coding_network.encoder(in_node)
    decoder_op = self_coding_network.decoder(encoder_op)

    placeholders = [in_node]
    predictions = [encoder_op]

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    out_after_train = sess.run(predictions, feed_dict={in_node: self_coding_network.get_placeholder_input("input")})

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="ae")
    predictions_after_freeze = tfp \
        .set_training_sess(sess) \
        .set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(self_coding_network.get_test_data()) \
        .build_save_frozen_graph()

    for before, after in zip(out_after_train, predictions_after_freeze):
        np.testing.assert_equal(before, after)


if __name__ == '__main__':
    test_self_coding_network()
