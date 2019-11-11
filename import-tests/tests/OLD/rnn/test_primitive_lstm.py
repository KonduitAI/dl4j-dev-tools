import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph
from tensorflow.contrib import rnn

'''
Toy network for testing a primitve lstm - completely static
'''


class PrimitiveLSTM(TestGraph):
    def __init__(self, feature_size=5, time_steps=9, *args, **kwargs):
        super(PrimitiveLSTM, self).__init__(*args, **kwargs)
        self.feature_size = feature_size
        self.time_steps = time_steps
        self.train_input, self.train_target = self._generate_time_seq(feature_size=feature_size, time_steps=time_steps)

    def _generate_time_seq(self, feature_size, time_steps, mini_batch=5):
        self.feature_size = 2
        time_seq = np.zeros((mini_batch, feature_size, time_steps))
        time_seq += np.linspace(0, time_steps - 1, time_steps).reshape(1, time_steps) / 100  # 0.01 increments
        time_seq += np.random.uniform(0, 0.5, mini_batch * feature_size).reshape(-1, feature_size, 1)  # random offset
        return time_seq[:, :, :-1], time_seq[:, :, -1].reshape((mini_batch, feature_size))

    def get_placeholder_input(self, name):
        if name == "input":
            return self.train_input
        if name == "target":
            return self.train_target

    def _get_placeholder_shape(self, name):
        if name == "input":
            return [None, self.feature_size, self.time_steps - 1]
        if name == "target":
            return [None, self.feature_size]


def RNNBasic(x, weights, biases, num_hidden):
    x = tf.unstack(x, axis=2)
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # inputs: A length T list of inputs,
    #  each a Tensor of shape [batch_size, input_size], or a nested tuple of such elements.
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float64)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def test_primitive_lstm():
    feature_size = 2
    num_hidden = 3
    primitive_lstm = PrimitiveLSTM(feature_size=feature_size, time_steps=7)
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, feature_size], dtype=tf.float64))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([feature_size], dtype=tf.float64))
    }
    in_node = primitive_lstm.get_placeholder("input")
    target_node = primitive_lstm.get_placeholder("target")
    preout = RNNBasic(in_node, weights, biases, num_hidden)
    out_node = tf.identity(preout, name="output")

    # Define loss and optimizer
    learning_rate = 0.001
    loss_op = tf.reduce_sum(tf.pow(out_node - target_node, 2)) / (2 * primitive_lstm.train_input.shape[0])  # MSE
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    training_steps = 500
    display_step = 100
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for step in range(1, training_steps + 1):
        sess.run(train_op,
                 feed_dict={in_node: primitive_lstm.train_input, target_node: primitive_lstm.train_target})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss = sess.run(loss_op, feed_dict={in_node: primitive_lstm.train_input,
                                                target_node: primitive_lstm.train_target})
            print("Step " + str(step) + ", Loss= " + \
                  "{:.4f}".format(loss))
    print("Optimization Finished!")

    placeholders = [in_node]
    predictions = [out_node]

    out_after_train = sess.run(predictions, feed_dict={in_node: primitive_lstm.train_input})
    tfp = TensorFlowPersistor(save_dir="primitive_lstm")
    predictions_after_freeze = tfp \
        .set_training_sess(sess) \
        .set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(primitive_lstm.get_test_data()) \
        .build_save_frozen_graph()
    print(predictions_after_freeze)
    for before, after in zip(out_after_train, predictions_after_freeze):
        np.testing.assert_equal(before, after)


if __name__ == '__main__':
    test_primitive_lstm()
