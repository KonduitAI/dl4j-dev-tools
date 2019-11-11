import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph


class BiasAdd(TestGraph):
    def get_placeholder_input(self, name):
        if name == "input":
            input_0 = np.linspace(1, 40, 40).reshape(10, 4)
            return input_0

    def _get_placeholder_shape(self, name):
        if name == "input":
            return [None, 4]

    def list_inputs(self):
        return ["input"]


def test_bias_add():
    bias_add = BiasAdd(seed=1337)
    in_node = bias_add.get_placeholder("input")
    biases = tf.Variable(tf.lin_space(1.0, 4.0, 4), name="bias")
    out_node = tf.nn.bias_add(in_node, tf.cast(biases, dtype=tf.float64), name="output")

    placeholders = [in_node]
    predictions = [out_node]

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="bias_add")
    predictions = tfp \
        .set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(bias_add.get_test_data()) \
        .build_save_frozen_graph()
    print(predictions)


if __name__ == '__main__':
    test_bias_add()
