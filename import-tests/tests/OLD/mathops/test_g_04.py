import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph


class MathOpsFour(TestGraph):
    def __init__(self, *args, **kwargs):
        super(MathOpsFour, self).__init__(*args, **kwargs)
        self.input_1 = np.random.uniform(size=(16, 16))
        self.input_2 = np.random.uniform(size=(16, 16)) + np.random.uniform(size=(16, 16))

    def list_inputs(self):
        return ["input_1", "input_2"]

    def get_placeholder_input(self, name):
        if name == "input_1":
            return self.input_1
        if name == "input_2":
            return self.input_2

    def _get_placeholder_shape(self, name):
        if name == "input_1":
            return [16, 16]
        if name == "input_2":
            return [16, 16]


def test_mathops_four():
    mathops_4 = MathOpsFour(seed=19)
    in_node_1 = mathops_4.get_placeholder("input_1")
    in_node_2 = mathops_4.get_placeholder("input_2")
    k0 = tf.Variable(tf.random_normal([8, 1, 8], dtype=tf.float64), name="in0")
    n1 = tf.concat([in_node_1, in_node_2], axis=-2)
    n3 = tf.reshape(n1, [8, 8, 8])
    n4 = tf.pow(n3, n3)
    n5 = tf.tan(n4)
    n6 = tf.negative(n5)
    n7 = tf.multiply(n6, n4)
    out_node = tf.subtract(n7, k0, name="output")

    placeholders = [in_node_1, in_node_2]
    predictions = [out_node]
    # Run and persist
    tfp = TensorFlowPersistor(save_dir="g_04")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(mathops_4.get_test_data()) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_mathops_four()
