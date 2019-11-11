import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph


class MathOpsTwelve(TestGraph):
    def list_inputs(self):
        return []


def test_mathops_twelve():
    mathops_0 = MathOpsTwelve(seed=19)
    in_node_0 = tf.Variable(tf.random_normal([3, 3]), name="in_0", dtype=tf.float32)
    n0 = tf.add(np.arange(-4., 5., 1.).astype(np.float32).reshape(3, 3), in_node_0)
    n1 = tf.abs(n0)
    n2 = tf.rsqrt(n1)
    a_node = tf.tanh(n2)
    out_node = tf.arg_max(a_node,1,name='output')
    predictions = [out_node]
    # Run and persist
    tfp = TensorFlowPersistor(save_dir="g_12")
    tfp.set_placeholders([]) \
        .set_output_tensors(predictions) \
        .set_test_data(mathops_0.get_test_data()) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_mathops_twelve()