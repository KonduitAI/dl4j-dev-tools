import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph

'''
No training.
Bunch of ops - prob need to make simpler graphs
'''


class TensorOpsMisc(TestGraph):
    def __init__(self, feature_size_a=[7, 2, 3, 4], feature_size_b=[5, 3, 4, 5], *args, **kwargs):
        super(TensorOpsMisc, self).__init__(*args, **kwargs)
        self.feature_size_a = feature_size_a
        self.feature_size_b = feature_size_b
        self.a_data, self.b_data = self.generate_data()

    def list_inputs(self):
        return ["input_a", "input_b"]

    def generate_data(self):
        return np.random.uniform(size=self.feature_size_a), np.random.uniform(size=self.feature_size_b)

    def get_placeholder_input(self, name):
        if name == "input_a":
            return self.a_data
        if name == "input_b":
            return self.b_data

    def _get_placeholder_shape(self, name):
        if name == "input_a":
            return [None] + self.feature_size_a[1:]
        if name == "input_b":
            return [None] + self.feature_size_b[1:]


def test_tensor_misc():
    tensor_scatter_misc = TensorOpsMisc(seed=713, feature_size_a=[12, 3, 4, 3], feature_size_b=[3, 3])
    in_node_a = tensor_scatter_misc.get_placeholder("input_a")  # 12,3,4,5
    in_node_a_erf = tf.erf(in_node_a)
    in_node_b = tensor_scatter_misc.get_placeholder("input_b")
    a_reversed_seq = tf.reverse_sequence(in_node_a_erf, batch_axis=2, seq_axis=3, seq_lengths=[2, 1, 3, 2])
    reduced = tf.reduce_sum(tf.round(a_reversed_seq), axis=(0, 2))
    erfc_plus = tf.erfc(in_node_b) + reduced + tf.cast(tf.eye(3), dtype=tf.float64)
    '''
    # Can't feeze graphs with scatter because of the same issue as that with batch norm
    some_var = tf.Variable(tf.random_normal(shape=[12, 3, 3], dtype=tf.float64), name="some_3x3")
    scatter_add_var = tf.scatter_add(some_var, indices=[2, 1, 0, 0],
                                     updates=tf.constant(np.random.uniform(size=(4, 3, 3)), dtype=tf.float64))
    after_scatter = tf.reduce_sum(scatter_add_var, axis=0) + tf.log1p(erfc_plus)
    scatter_nd_var = tf.scatter_nd([[0], [1], [3], [2]], updates=tf.constant(np.random.uniform(size=(4, 3, 2))),
                                   shape=tf.constant([5, 3, 2]))
    out_node = tf.concat([tf.reshape(scatter_nd_var, shape=[10, 3]), after_scatter], axis=0, name="output")
    '''
    some_var = tf.Variable(tf.random_normal(shape=[3, 3], dtype=tf.float64))
    out_node = tf.add(tf.log1p(erfc_plus), some_var, name="output")

    placeholders = [in_node_a, in_node_b]
    predictions = [out_node]

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="tensor_ops_misc")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(tensor_scatter_misc.get_test_data()) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_tensor_misc()
