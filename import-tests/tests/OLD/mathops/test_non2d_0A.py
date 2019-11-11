import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph


class NonTwoDZeroA(TestGraph):
    def __init__(self, *args, **kwargs):
        super(NonTwoDZeroA, self).__init__(*args, **kwargs)
        self.input_scalar_a = np.random.random_integers(1, 6)
        self.input_scalar_b = np.random.random_integers(1, 4)

    def list_inputs(self):
        return ["scalarA", "scalarB"]

    def get_placeholder_input(self, name):
        if name == "scalarA":
            return self.input_scalar_a
        if name == "scalarB":
            return self.input_scalar_b

    def _get_placeholder_shape(self, name):
        if name == "scalarA" or name == "scalarB":
            return []


def test_nontwod_zero_a():
    non_twod_0_a = NonTwoDZeroA(seed=13)
    in_node_a = non_twod_0_a.get_placeholder("scalarA", data_type=tf.int32)
    in_node_b = non_twod_0_a.get_placeholder("scalarB", data_type=tf.int32)

    some_vector = tf.stack([in_node_a, in_node_b])  # [2,] shape with value [5,2]
    i0 = tf.Variable(np.random.uniform(size=(3, 4)), dtype=tf.float32)  # shape [3,4]
    out_node = tf.tile(i0, some_vector, name="output")

    placeholders = [in_node_a, in_node_b]
    predictions = [out_node]

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="non2d_0A")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(non_twod_0_a.get_test_data()) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_nontwod_zero_a()
