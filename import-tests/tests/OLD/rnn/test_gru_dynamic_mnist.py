import functools
import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph
from tensorflow.examples.tutorials.mnist import input_data

'''
#Adapated from https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/
'''

MNIST = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return wrapper


class SequenceClassification(TestGraph):
    def __init__(self, num_hidden=200, num_layers=3, *args, **kwargs):
        super(SequenceClassification, self).__init__(*args, **kwargs)
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.input_placeholder = self.get_placeholder("input", data_type=tf.float32)
        self.target_placeholder = self.get_placeholder("target", data_type=tf.float32)
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def prediction(self):
        # Recurrent network.
        cells = []
        for _ in range(self._num_layers):
            cell = tf.contrib.rnn.GRUCell(self._num_hidden)  # Or LSTMCell(num_units)
            cells.append(cell)
        network = tf.contrib.rnn.MultiRNNCell(cells)
        output, _ = tf.nn.dynamic_rnn(network, self.input_placeholder, dtype=tf.float32)
        # Select last output.
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)
        # Softmax layer.
        weight, bias = self._weight_and_bias(
            self._num_hidden, int(self.target_placeholder.get_shape()[1]))
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias, name="output")
        return prediction

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target_placeholder * tf.log(self.prediction))
        return cross_entropy

    @lazy_property
    def optimize(self):
        learning_rate = 0.003
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target_placeholder, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    def get_placeholder_input(self, name):
        test_len = 100
        if name == "input":
            return MNIST.test.images[:test_len].reshape((-1, 28, 28))  # time_steps, num_input
        if name == "target":
            return MNIST.test.labels[:test_len]

    def _get_placeholder_shape(self, name):
        if name == "input":
            return [None, 28, 28]
        if name == "target":
            return [None, 10]


def test_sequence_classification():
    sequence_classifier = SequenceClassification(seed=713, num_hidden=20)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    test_data = sequence_classifier.get_placeholder_input("input")
    test_target = sequence_classifier.get_placeholder_input("target")
    for epoch in range(10):
        for _ in range(100):
            # reshape mnist data as a timeseries
            batch_data, batch_target = MNIST.train.next_batch(10)
            sess.run(sequence_classifier.optimize, {
                sequence_classifier.input_placeholder: batch_data.reshape((-1, 28, 28)),
                sequence_classifier.target_placeholder: batch_target})
        error = sess.run(sequence_classifier.error, {
            sequence_classifier.input_placeholder: test_data, sequence_classifier.target_placeholder: test_target})
        print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))

    placeholders = [sequence_classifier.input_placeholder]
    predictions = [sequence_classifier.prediction]
    out_after_train = sess.run(predictions, feed_dict={sequence_classifier.input_placeholder: test_data})

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="gru_dynamic_mnist")
    predictions_after_freeze = tfp \
        .set_training_sess(sess) \
        .set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(sequence_classifier.get_test_data()) \
        .build_save_frozen_graph()

    for before, after in zip(out_after_train, predictions_after_freeze):
        np.testing.assert_equal(before, after)


if __name__ == '__main__':
    test_sequence_classification()
