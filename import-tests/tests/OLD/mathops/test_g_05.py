import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph


class MathOpsFive(TestGraph):
    def __init__(self, *args, **kwargs):
        super(MathOpsFive, self).__init__(*args, **kwargs)
        self.input_1 = np.random.uniform(size=(2, 4, 3, 2))
        self.input_2 = np.random.uniform(size=(3, 2))

    def list_inputs(self):
        return ["input_1", "input_2"]

    def get_placeholder_input(self, name):
        if name == "input_1":
            return self.input_1
        if name == "input_2":
            return self.input_2

    def _get_placeholder_shape(self, name):
        if name == "input_1":
            return [2, 4, 3, 2]
        if name == "input_2":
            return [3, 2]


def test_mathops_five():
    mathops_5 = MathOpsFive(seed=19)
    in_node_1 = mathops_5.get_placeholder("input_1")
    in_node_2 = mathops_5.get_placeholder("input_2")
    k0 = tf.Variable(tf.random_normal([3, 2], dtype=tf.float64), name="in0")
    n0 = tf.gather(in_node_1, [1, 0], axis=-2)  # 2,4,2,2
    n1 = tf.gather_nd(n0, [[0, 2, 1], [0, 1, 0], [1, 3, 1]])  # 3,2
    out_node = tf.stack([n1, k0, in_node_2], axis=-1, name="output")  # 3, 2, 2

    placeholders = [in_node_1, in_node_2]
    predictions = [out_node]
    # Run and persist
    tfp = TensorFlowPersistor(save_dir="g_05")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(mathops_5.get_test_data()) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_mathops_five()
