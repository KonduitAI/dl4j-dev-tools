import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph


class MathOpsThree(TestGraph):
    def __init__(self, *args, **kwargs):
        super(MathOpsThree, self).__init__(*args, **kwargs)
        self.input_0 = np.random.uniform(size=(4, 4, 16, 16))

    def list_inputs(self):
        return ["input_0"]

    def get_placeholder_input(self, name):
        if name == "input_0":
            return self.input_0

    def _get_placeholder_shape(self, name):
        if name == "input_0":
            return [4, 4, 16, 16]


def test_mathops_three():
    mathops_3 = MathOpsThree(seed=19)
    in_node_0 = mathops_3.get_placeholder("input_0")
    k0 = tf.Variable(tf.random_normal([8, 8], dtype=tf.float64), name="in0")
    n4 = tf.depth_to_space(in_node_0, block_size=4)
    n5 = tf.cumsum(n4, axis=-3, exclusive=True, reverse=True)
    n6 = tf.diag_part(tf.reshape(n5, [8, 8, 8, 8]))
    n7 = tf.diag(n6)
    out_node = tf.add(n7, k0, name="output")

    placeholders = [in_node_0]
    predictions = [out_node]
    # Run and persist
    tfp = TensorFlowPersistor(save_dir="g_03")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(mathops_3.get_test_data()) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_mathops_three()
