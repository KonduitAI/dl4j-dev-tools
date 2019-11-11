import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph


class ExpandDimT(TestGraph):
    def __init__(self, *args, **kwargs):
        super(ExpandDimT, self).__init__(*args, **kwargs)
        self.input_0 = np.random.uniform(size=(3, 4))

    def list_inputs(self):
        return ["input_0"]

    def get_placeholder_input(self, name):
        if name == "input_0":
            return self.input_0

    def _get_placeholder_shape(self, name):
        if name == "input_0":
            return [3, 4]


def test_expand_dim():
    expand_dim_t = ExpandDimT(seed=19)
    in_node_0 = expand_dim_t.get_placeholder("input_0")
    k0 = tf.Variable(tf.random_normal([3, 1, 4], dtype=tf.float64), name="in0")
    in0_expanded = tf.expand_dims(in_node_0, axis=-2)
    out_node = tf.add(in0_expanded, k0, name="output")

    placeholders = [in_node_0]
    predictions = [out_node]
    # Run and persist
    tfp = TensorFlowPersistor(save_dir="expand_dim")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(expand_dim_t.get_test_data()) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_expand_dim()
