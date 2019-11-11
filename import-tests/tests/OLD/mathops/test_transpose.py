import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph

'''
One off test for tranpose 
'''


class SimpleTranspose(TestGraph):
    def get_placeholder_input(self, name):
        if name == "input":
            return np.random.uniform(size=(3, 3))

    def _get_placeholder_shape(self, name):
        if name == "input":
            return [None, 3]


def test_simple_transpose():
    simple_t = SimpleTranspose(seed=713)
    in0 = simple_t.get_placeholder("input")

    k0 = tf.Variable(tf.random_normal([3, 3], dtype=tf.float64), name="k0")
    in1 = tf.transpose(in0, name="input_1")
    out_node = tf.add(in1, k0, name="output")

    placeholders = [in0]
    predictions = [out_node]

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="transpose")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(simple_t.get_test_data()) \
        .build_save_frozen_graph()

if __name__ == '__main__':
    test_simple_transpose()