import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph


class MultipleOuts(TestGraph):
    def __init__(self, *args, **kwargs):
        super(MultipleOuts, self).__init__(*args, **kwargs)
        self.input_0 = np.linspace(1, 120, 120).reshape(2, 3, 4, 5)

    def list_inputs(self):
        return ["input_0"]

    def get_placeholder_input(self, name):
        if name == "input_0":
            return self.input_0

    def _get_placeholder_shape(self, name):
        if name == "input_0":
            return [2, 3, 4, 5]


def test_multiple_outs():
    multiple_out_test = MultipleOuts(seed=913)
    in_node_0 = multiple_out_test.get_placeholder("input_0", data_type=tf.float32)
    unstacked = tf.unstack(in_node_0, axis=-2)
    unstack1 = unstacked[0]
    unstack2 = unstacked[1]  # 2x3x5 now
    n1 = unstack1 + tf.Variable(tf.zeros([2, 3, 5]))
    n2 = unstack2
    out_node = tf.stack([n1, n2, unstacked[2], unstacked[3]], axis=-4, name="output")

    placeholders = [in_node_0]
    predictions = [out_node]

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="node_multiple_out")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(multiple_out_test.get_test_data()) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_multiple_outs()
