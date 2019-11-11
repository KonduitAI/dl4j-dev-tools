import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph


class MathOpsNine(TestGraph):
    def __init__(self, *args, **kwargs):
        super(MathOpsNine, self).__init__(*args, **kwargs)
        self.input_1 = np.random.uniform(size=(3, 10))
        self.input_2 = np.random.uniform(size=(2, 10))
        self.input_3 = np.random.uniform(size=(5, 10))

    def list_inputs(self):
        return ["input_1", "input_2", "input_3"]

    def get_placeholder_input(self, name):
        if name == "input_1":
            return self.input_1
        if name == "input_2":
            return self.input_2
        if name == "input_3":
            return self.input_3

    def _get_placeholder_shape(self, name):
        return [None, 10]


def test_mathops_nine():
    mathops_9 = MathOpsNine(seed=19)
    in_node_1 = mathops_9.get_placeholder("input_1")
    in_node_2 = mathops_9.get_placeholder("input_2")
    in_node_3 = mathops_9.get_placeholder("input_3")
    n1 = tf.nn.softsign(in_node_1)
    n2 = tf.nn.softplus(in_node_2)
    n3 = tf.concat([n1, n2, in_node_3], axis=0)
    n4 = tf.nn.softmax(n3)
    w = tf.Variable(tf.random_normal([10, 10], dtype=tf.float64), name="w")
    n5 = tf.nn.softmax(w)
    n6 = tf.nn.softmax_cross_entropy_with_logits(labels=n5, logits=n4)
    n7 = tf.nn.log_softmax(n6)
    n8 = tf.nn.sigmoid_cross_entropy_with_logits(labels=n5, logits=n4)
    n9 = tf.nn.weighted_cross_entropy_with_logits(targets=n5, logits=n4, pos_weight=10)

    out_node_1 = tf.identity(n7, name="output_1")
    out_node_2 = tf.identity(n8, name="output_2")
    out_node_3 = tf.identity(n8, name="output_3")
    placeholders = [in_node_1, in_node_2, in_node_3]
    predictions = [out_node_1, out_node_2, out_node_3]
    # Run and persist
    tfp = TensorFlowPersistor(save_dir="g_09")
    predictions_after_freeze = tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(mathops_9.get_test_data()) \
        .build_save_frozen_graph()
    print(predictions_after_freeze[0].shape)
    print(predictions_after_freeze[1].shape)
    print(predictions_after_freeze[2].shape)


if __name__ == '__main__':
    test_mathops_nine()
