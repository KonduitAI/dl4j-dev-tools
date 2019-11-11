import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph


class MathOpsSeven(TestGraph):
    def __init__(self, *args, **kwargs):
        super(MathOpsSeven, self).__init__(*args, **kwargs)
        self.input_1 = np.array([9, 4, 2, 3, 1, 5, 7, 0, 6, 8])
        self.input_2 = np.random.uniform(size=(10, 8))

    def list_inputs(self):
        return ["input_1", "input_2"]

    def get_placeholder_input(self, name):
        if name == "input_1":
            return self.input_1
        if name == "input_2":
            return self.input_2

    def _get_placeholder_shape(self, name):
        if name == "input_1":
            return [10]
        if name == "input_2":
            return [None, 8]


def test_mathops_seven():
    mathops_7 = MathOpsSeven(seed=19)
    in_node_1 = mathops_7.get_placeholder("input_1", data_type=tf.int32)
    in_node_2 = mathops_7.get_placeholder("input_2")
    w = tf.Variable(tf.random_normal([8, 10], dtype=tf.float64), name="w")
    b = tf.cast(tf.invert_permutation(in_node_1), dtype=tf.float64)
    n1 = tf.nn.xw_plus_b(in_node_2, w, b)
    n2 = tf.cast(tf.fill([10, 10], 1.2345), dtype=tf.float64)
    n3 = tf.add(n1, n2)
    n4 = tf.nn.relu6(n3)
    n5 = tf.nn.moments(n4, axes=[1, 0], keep_dims=True)
    n6 = tf.meshgrid(n5, tf.Variable(tf.random_normal([2, 1, 1], dtype=tf.float64)))
    n7 = tf.parallel_stack([n6[1], n6[0], n6[1]])  # (3,2,2)
    n8 = tf.nn.normalize_moments(n7[0], n7[1], n7[2], None)  # (2,2,2)
    out_node = tf.pad(n8, tf.constant([[1, 1], [1, 1], [1, 1]]), "REFLECT", name="output")

    placeholders = [in_node_1, in_node_2]
    predictions = [out_node]
    # Run and persist
    tfp = TensorFlowPersistor(save_dir="g_07")
    predictions_after_freeze = tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(mathops_7.get_test_data()) \
        .build_save_frozen_graph()
    print(predictions_after_freeze[0].shape)


if __name__ == '__main__':
    test_mathops_seven()
