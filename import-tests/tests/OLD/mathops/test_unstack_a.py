import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph


class MultipleOutsA(TestGraph):
    def __init__(self, *args, **kwargs):
        super(MultipleOutsA, self).__init__(*args, **kwargs)
        self.input_0 = np.linspace(1, 24, 24).reshape(2, 3, 4)

    def list_inputs(self):
        return ["input_0"]

    def get_placeholder_input(self, name):
        if name == "input_0":
            return self.input_0

    def _get_placeholder_shape(self, name):
        if name == "input_0":
            return [2, 3, 4]


def test_multiple_outs_a():
    multiple_out_test = MultipleOutsA(seed=913)
    in_node = multiple_out_test.get_placeholder("input_0", data_type=tf.float32)
    in_node_0 = in_node + tf.Variable(tf.zeros([2, 3, 4])) #Graph won't save without some variable present
    out_node_a = tf.unstack(in_node_0, axis=2, name="outputA") # 4 of size 2x3
    out_node_b = tf.unstack(in_node_0, axis=1, name="outputB") # 3 of size 2x4

    placeholders = [in_node]
    predictions = [out_node_a, out_node_b] #out_node_a and out_node_b are lists

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="multiple_outs_a")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(multiple_out_test.get_test_data()) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_multiple_outs_a()
