import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph


class AssertTrue(TestGraph):
    def get_placeholder_input(self, name):
        if name == "input":
            return np.random.uniform(size=(3, 3))

    def _get_placeholder_shape(self, name):
        if name == "input":
            return [None, 3]


def test_assert_true():
    assertTrue = AssertTrue(seed=713)

    x = assertTrue.get_placeholder("input")
    k0 = tf.Variable(tf.random_normal([3, 3], dtype=tf.float64), name="k0")
    assert_op = tf.Assert(tf.less_equal(tf.reduce_max(x), 100.), [k0])

    with tf.control_dependencies([assert_op]):
        in1 = tf.transpose(x, name="input_1")
    out_node = tf.add(in1, k0, name="output")

    placeholders = [x]
    predictions = [out_node]

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="assert_true")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(assertTrue.get_test_data()) \
        .build_save_frozen_graph()

if __name__ == '__main__':
    test_assert_true()
