import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph


class MultipleOutsB(TestGraph):
    def __init__(self, *args, **kwargs):
        super(MultipleOutsB, self).__init__(*args, **kwargs)
        self.input_0 = np.linspace(1, 8, 8).reshape(2, 2, 2)

    def list_inputs(self):
        return ["input_0"]

    def get_placeholder_input(self, name):
        if name == "input_0":
            return self.input_0

    def _get_placeholder_shape(self, name):
        if name == "input_0":
            return [2, 2, 2]


def test_multiple_outs_b():
    multiple_out_test = MultipleOutsB(seed=913)
    in_node = multiple_out_test.get_placeholder("input_0", data_type=tf.float32)
    in_node_0 = in_node + tf.Variable(tf.zeros([2, 2, 2])) #Graph won't save without some variable present
    out_node_a = tf.unstack(in_node_0, axis=2,name='outputA') # 2 of size 2x2
    a_node = tf.add(in_node_0,out_node_a[0])
    out_node_b = tf.unstack(a_node,axis=0,name="outputB")

    placeholders = [in_node]
    predictions = [out_node_a[1], out_node_b] #out_node_b is a list

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="multiple_outs_b")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(multiple_out_test.get_test_data()) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_multiple_outs_b()
