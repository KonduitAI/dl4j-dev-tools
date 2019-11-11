import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph


class MathOpsTen(TestGraph):
    def __init__(self, *args, **kwargs):
        super(MathOpsTen, self).__init__(*args, **kwargs)
        self.input_1 = np.random.uniform(size=(4, 3))

    def list_inputs(self):
        return ["input_1"]

    def get_placeholder_input(self, name):
        if name == "input_1":
            return self.input_1

    def _get_placeholder_shape(self, name):
        if name == "input_1":
            return [None, 3]


def test_mathops_ten():
    mathops_10 = MathOpsTen(seed=19)
    in_node_1 = mathops_10.get_placeholder("input_1")
    #in_node_2 = mathops_8.get_placeholder("input_2")
    #n0 = tf.is_finite(in_node_1)
    #n1 = tf.reduce_all(n0)
    #n2 = tf.cast(n0, dtype=tf.float64)
    #n3 = tf.cast(n1, dtype=tf.float64)
    n4 = tf.add(in_node_1, in_node_1)
    #n5 = tf.cast(tf.truncatediv(tf.cast(n4, dtype=tf.int32), 3), dtype=tf.float64)
    n6 = tf.reciprocal(n4)  # should be inf now
    #n7 = tf.cast(tf.is_inf(n6), dtype=tf.float64)
    #n8 = tf.cast(tf.is_nan(n6), dtype=tf.float64)
    n9 = tf.squared_difference(n4, n6)
    w = tf.Variable(tf.random_normal([4, 3], dtype=tf.float64), name="w")
    n10 = tf.reverse(w, axis=[-1])
    n11 = tf.add(n10, n9)
    #n12 = tf.reciprocal(tf.multiply(n11, [[0, 1, 1], [1, 1, 1], [0, 1, 0], [1, 0, 0]]))
    #n13 = tf.reduce_any(tf.is_inf(n12))
    #n14 = tf.cast(n13, dtype=tf.float64)

    out_node = tf.identity(n11, name="output")
    placeholders = [in_node_1]
    predictions = [out_node]
    # Run and persist
    tfp = TensorFlowPersistor(save_dir="g_11")
    predictions_after_freeze = tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(mathops_10.get_test_data()) \
        .build_save_frozen_graph()
    print(predictions_after_freeze[0].shape)


if __name__ == '__main__':
    test_mathops_ten()
