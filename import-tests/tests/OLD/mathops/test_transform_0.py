import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph
from tfoptests.math_ops import DifferentiableMathOps


class MathTransform(TestGraph):
    def __init__(self, *args, **kwargs):
        super(MathTransform, self).__init__(*args, **kwargs)
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


def test_mathtransform():
    ops = ["add"
           # , "add_n"
        , "max"
        , "min"
        , "abs"
        , "cos"
        , "acos"
        , "add"
        , "max"
        , "min"
        , "abs"
        , "ceil"
        , "min"
           # , "cross"
        , "exp"
        , "log"
           # , "log1p"
           # , "mod"
           # , "mathmul"
           # , "cumprod"
           # , "cumsum"
           # , "erf"
           # , "count_nonzero"
           # , "greater"
           # , "greater_equal"
           # , "equal"
           ]
    math_transform = MathTransform(seed=19)
    in_node_0 = math_transform.get_placeholder("input_0", data_type=tf.float32)
    in_node_1 = math_transform.get_placeholder("input_1", data_type=tf.float32)
    k0 = tf.Variable(tf.random_normal([8, 8]), name="in0")
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
    tfp = TensorFlowPersistor(save_dir="transform_0")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(math_transform.get_test_data()) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_mathtransform()
