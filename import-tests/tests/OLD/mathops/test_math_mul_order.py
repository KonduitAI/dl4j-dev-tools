import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph

'''
One off test for matmul
'''


class MatMulOrder(TestGraph):
    def list_inputs(self):
        return ["input_0", "input_1"]

    def get_placeholder_input(self, name):
        if name == "input_0":
            input_0 = np.random.uniform(size=(3, 3))
            return input_0
        if name == "input_1":
            input_1 = np.random.uniform(size=(3, 3)) + np.random.uniform(size=(3, 3))
            return input_1

    def _get_placeholder_shape(self, name):
        if name == "input_0" or name == "input_1":
            return [None, 3]


def test_mat_mul_order():
    simple_m = MatMulOrder(seed=713)
    in0 = simple_m.get_placeholder("input_0")
    in1 = simple_m.get_placeholder("input_1")
    k0 = tf.Variable(tf.random_normal([3, 3], dtype=tf.float64), name="in0")
    out_node = tf.matmul(k0, tf.matmul(in0, in1), name="output")

    placeholders = [in0, in1]
    predictions = [out_node]
    # Run and persist
    tfp = TensorFlowPersistor(save_dir="math_mul_order")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(simple_m.get_test_data()) \
        .build_save_frozen_graph()

if __name__ == '__main__':
    test_mat_mul_order()
