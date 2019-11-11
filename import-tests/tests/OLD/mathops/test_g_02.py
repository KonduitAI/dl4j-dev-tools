import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph
from tfoptests.math_ops import DifferentiableMathOps


class MathOpsTwo(TestGraph):
    def __init__(self, *args, **kwargs):
        super(MathOpsTwo, self).__init__(*args, **kwargs)
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
            return [3, 3]


def test_mathops_two():
    ops = ["acos"
        , "sin"
        , "asin"
        , "sinh"
        , "floor"
        , "asinh"
        , "min"
        , "cos"
        , "add"
        , "acosh"
        , "atan"
        , "atan2"
        , "add"
        , "elu"
        , "cosh"
        , "mod"
        , "cross"
           # , "diagpart"
           # , "diag"
        , "expm"
        , "asinh"
        , "atanh"
           ]
    mathops_2 = MathOpsTwo(seed=19)
    in_node_0 = mathops_2.get_placeholder("input_0", data_type=tf.float32)
    in_node_1 = mathops_2.get_placeholder("input_1", data_type=tf.float32)
    k0 = tf.Variable(tf.random_normal([8, 8]), name="in0")
    constr = DifferentiableMathOps(in_node_0, in_node_1)

    for op in ops:
        print("Running " + op)
        answer = constr.execute(op)
        print(answer)
        constr.set_a(answer)

    out_node = tf.floormod(constr.a, constr.b, name="output")

    placeholders = [in_node_0, in_node_1]
    predictions = [out_node]
    # Run and persist
    tfp = TensorFlowPersistor(save_dir="g_02")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(mathops_2.get_test_data()) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_mathops_two()
