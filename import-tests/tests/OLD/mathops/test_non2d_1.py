import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph


class NonTwoDOne(TestGraph):
    def __init__(self, *args, **kwargs):
        super(NonTwoDOne, self).__init__(*args, **kwargs)
        self.input_scalar = np.random.uniform()
        self.input_vector = np.random.uniform(size=2)

    def list_inputs(self):
        return ["scalar", "vector"]

    def get_placeholder_input(self, name):
        if name == "scalar":
            return self.input_scalar
        if name == "vector":
            return self.input_vector

    def _get_placeholder_shape(self, name):
        if name == "scalar":
            return []
        if name == "vector":
            return [2]


def test_nontwod_one():
    non_twod_1 = NonTwoDOne(seed=13)
    in_node_a = non_twod_1.get_placeholder("scalar", data_type=tf.float32)
    in_node_b = non_twod_1.get_placeholder("vector", data_type=tf.float32)
    k0 = tf.Variable(tf.random_normal([2, 1]), name="someweight", dtype=tf.float32)

    i0 = tf.reshape(tf.reduce_sum(in_node_b), [])
    i1 = in_node_a + in_node_b + i0
    out_node = tf.matmul(tf.expand_dims(i1, 0), k0, name="output")

    placeholders = [in_node_a, in_node_b]
    predictions = [out_node]

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="non2d_1")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(non_twod_1.get_test_data()) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_nontwod_one()
