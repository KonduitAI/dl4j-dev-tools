import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph

'''
Toy network for test
'''


class RNNSimpleDynamic(TestGraph):
    def __init__(self, feature_size=5, time_steps=9, *args, **kwargs):
        super(RNNSimpleDynamic, self).__init__(*args, **kwargs)
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


def test_RNN_Simple_Dynamic():
    num_hidden = 3
    num_layers = 5
    rnn_simple_dynamic = RNNSimpleDynamic(seed=1337, feature_size=2)
    in_node = rnn_simple_dynamic.get_placeholder("input", data_type=tf.float32)
    target_node = rnn_simple_dynamic.get_placeholder("target", data_type=tf.float32)
    cells = []
    for _ in range(num_layers):
        cell = tf.contrib.rnn.GRUCell(num_hidden)  # Or LSTMCell(num_units)
        cells.append(cell)
    network = tf.contrib.rnn.MultiRNNCell(cells)
    output_rnn, _ = tf.nn.dynamic_rnn(network, in_node, dtype=tf.float32)
    output_rnn = tf.transpose(output_rnn, [1, 0, 2])
    last = tf.gather(output_rnn, int(output_rnn.get_shape()[0]) - 1)
    weight = tf.Variable(tf.truncated_normal([num_hidden, int(target_node.get_shape()[1])], stddev=0.01))
    bias = tf.Variable(tf.constant(0.1, shape=[int(target_node.get_shape()[1])]))
    out_node = tf.identity(tf.matmul(last, weight) + bias, name="output")

    # Define loss and optimizer
    learning_rate = 0.001
    loss_op = tf.reduce_sum(tf.pow(out_node - target_node, 2)) / (2 * rnn_simple_dynamic.train_input.shape[0])  # MSE
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    training_steps = 500
    display_step = 100
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for step in range(1, training_steps + 1):
        sess.run(train_op,
                 feed_dict={in_node: rnn_simple_dynamic.train_input, target_node: rnn_simple_dynamic.train_target})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss = sess.run(loss_op, feed_dict={in_node: rnn_simple_dynamic.train_input,
                                                target_node: rnn_simple_dynamic.train_target})
            print("Step " + str(step) + ", Loss= " + \
                  "{:.4f}".format(loss))
    print("Optimization Finished!")

    placeholders = [in_node]
    predictions = [out_node]

    out_after_train = sess.run(predictions, feed_dict={in_node: rnn_simple_dynamic.train_input})
    tfp = TensorFlowPersistor(save_dir="primitive_gru_dynamic")
    predictions_after_freeze = tfp \
        .set_training_sess(sess) \
        .set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(rnn_simple_dynamic.get_test_data()) \
        .build_save_frozen_graph()
    print(predictions_after_freeze)
    for before, after in zip(out_after_train, predictions_after_freeze):
        np.testing.assert_equal(before, after)


if __name__ == '__main__':
    test_RNN_Simple_Dynamic()
