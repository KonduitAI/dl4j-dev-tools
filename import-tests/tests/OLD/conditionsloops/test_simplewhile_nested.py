from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph
import tensorflow as tf
import numpy as np


class TwoInputs(TestGraph):
    def __init__(self, *args, **kwargs):
        super(TwoInputs, self).__init__(*args, **kwargs)
        self.input_0 = np.linspace(1, 4, 4).reshape(2, 2)
        self.input_1 = np.ones([3, 3])

    def list_inputs(self):
        return ["input_0", "input_1"]

    def get_placeholder_input(self, name):
        if name == "input_0":
            return self.input_0
        if name == "input_1":
            return self.input_1

    def _get_placeholder_shape(self, name):
        if name == "input_0":
            return [2, 2]
        if name == "input_1":
            return [3, 3]


def test_simplewile_nested():
    two_inputs = TwoInputs(seed=13)
    in0 = two_inputs.get_placeholder("input_0", data_type=tf.float32)
    in1 = two_inputs.get_placeholder("input_1", data_type=tf.float32)
    c0 = tf.Variable(tf.constant(10.0, shape=[], name="addVal0"))
    c1 = tf.Variable(tf.constant(2.0, shape=[], name="addVal1"))
    c2 = tf.Variable(tf.constant(1.0, shape=[], name="addVal2"))

    def outer_body_cond(x0, x1):
        return tf.less(tf.reduce_mean(x0), c0)

    def outer_body_fn(x0, x1):
        def inner_body_cond(x0, x1):
            return tf.less_equal(tf.reduce_sum(x1), tf.reduce_sum(x0))

        def inner_body_fn(x0, x1):
            x1 = tf.add(x1, c2)
            return [x0, x1]

        x0 = tf.add(x0, c1)
        x0, x1 = tf.while_loop(inner_body_cond, inner_body_fn, [x0, x1])
        x1 = tf.subtract(x1, c2)
        return [x0, x1]

    in0p, in1p = tf.while_loop(outer_body_cond, outer_body_fn, [in0, in1])
    inp = tf.add(in0p, tf.reduce_mean(in1p))
    out_node = tf.identity(inp, name="output")

    placeholders = [in0, in1]
    predictions = [out_node]

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="simplewhile_nested")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(two_inputs.get_test_data()) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_simplewile_nested()
