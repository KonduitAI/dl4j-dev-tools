import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.math_ops import DifferentiableMathOps
from tfoptests.test_graph import TestGraph


class AddN(TestGraph):
    def __init__(self, *args, **kwargs):
        super(AddN, self).__init__(*args, **kwargs)
        self.input_0 = np.random.uniform(size=(3, 3))
        self.input_1 = np.random.uniform(size=(3, 3)) + np.random.uniform(size=(3, 3))

    def list_inputs(self):
        return ["input_0", "input_1"]

    def get_placeholder_input(self, name):
        if name == "input_0":
            return self.input_0
        if name == "input_1":
            return self.input_1

    def _get_placeholder_shape(self, name):
        if name == "input_0" or name == "input_1":
            return [None, 3]


def test_add_n():
    ops = ["add", "add_n"]
    addn = AddN(seed=13)
    in_node_0 = addn.get_placeholder("input_0")
    in_node_1 = addn.get_placeholder("input_1")
    k0 = tf.Variable(tf.random_normal([3, 3]), name="in0", dtype=tf.float32)

    constr = DifferentiableMathOps(in_node_0, in_node_1)

    for op in ops:
        print("Running " + op)
        answer = constr.execute(op)
        print(answer)
        constr.set_a(answer)

    out_node = tf.rsqrt(answer, name="output")

    placeholders = [in_node_0, in_node_1]
    predictions = [out_node]
    # Run and persist
    tfp = TensorFlowPersistor(save_dir="add_n")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(addn.get_test_data()) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_add_n()
