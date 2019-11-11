import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

MNIST = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)


class SequenceClassification(TestGraph):
    def __init__(self, test_size=100, *args, **kwargs):
        super(SequenceClassification, self).__init__(*args, **kwargs)
        self.test_input = MNIST.test.images[:test_size].reshape((-1, 28, 28))
        self.test_target = MNIST.test.labels[:test_size]

    def get_placeholder_input(self, name):
        if name == "input":
            return self.test_input
        if name == "target":
            return self.test_target

    def _get_placeholder_shape(self, name):
        if name == "input":
            return [None, 28, 28]
        if name == "target":
            return [None, 10]


def RNNBasic(x, weights, biases, num_hidden):
    x = tf.unstack(x, 28, 1)
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # inputs: A length T list of inputs,
    #  each a Tensor of shape [batch_size, input_size], or a nested tuple of such elements.
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float64)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def test_primitive_lstm_mnist():
    feature_size = 10  # mnist
    num_hidden = 128
    sequence_classification = SequenceClassification(seed=713)
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, feature_size], dtype=tf.float64))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([feature_size], dtype=tf.float64))
    }
    in_node = sequence_classification.get_placeholder("input")
    target_node = sequence_classification.get_placeholder("target")
    logits = RNNBasic(in_node, weights, biases, num_hidden)
    out_node = tf.nn.softmax(logits, name="output")

    # Define loss and optimizer
    learning_rate = 0.001
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=target_node))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(out_node, 1), tf.argmax(target_node, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float64))

    training_steps = 500
    display_step = 100
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    batch_size = 128
    for step in range(1, training_steps + 1):
        batch_x, batch_y = MNIST.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, 28, 28))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={in_node: batch_x, target_node: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={in_node: batch_x,
                                                                 target_node: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    placeholders = [in_node]
    predictions = [out_node]

    out_after_train = sess.run(predictions, feed_dict={in_node: sequence_classification.test_input})
    tfp = TensorFlowPersistor(save_dir="lstm_mnist")
    predictions_after_freeze = tfp \
        .set_training_sess(sess) \
        .set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(sequence_classification.get_test_data()) \
        .build_save_frozen_graph()
    print(predictions_after_freeze)
    for before, after in zip(out_after_train, predictions_after_freeze):
        np.testing.assert_equal(before, after)


if __name__ == '__main__':
    test_primitive_lstm_mnist()
