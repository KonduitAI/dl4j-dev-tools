import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph


class NonTwoDZero(TestGraph):
    def __init__(self, *args, **kwargs):
        super(NonTwoDZero, self).__init__(*args, **kwargs)
        self.input_scalar = np.random.uniform()

    def list_inputs(self):
        return ["scalar"]

    def get_placeholder_input(self, name):
        if name == "scalar":
            return self.input_scalar

    def _get_placeholder_shape(self, name):
        if name == "scalar":
            return []


def test_nontwod_zero():
    non_twod_0 = NonTwoDZero(seed=13)
    in_node = non_twod_0.get_placeholder("scalar", data_type=tf.float32)
    k0 = tf.Variable(tf.random_normal([2, 1]), name="someweight", dtype=tf.float32)
    a = tf.reduce_sum(in_node + k0)  # gives a scalar
    out_node = tf.reduce_sum(a + k0, name="output", axis=0)  # gives a vector

    placeholders = [in_node]
    predictions = [out_node]

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="non2d_0")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(non_twod_0.get_test_data()) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_nontwod_zero()
