import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph

'''
No training.
Graph with a few ops and wierd shapes to hit edge cases 
'''


class TensorDotMisc(TestGraph):
    def __init__(self, feature_size_a=[7, 2, 3, 4], feature_size_b=[5, 3, 4, 5], *args, **kwargs):
        super(TensorDotMisc, self).__init__(*args, **kwargs)
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


def test_tensor_dot_misc():
    tensor_dot_misc = TensorDotMisc(seed=713, feature_size_a=[36, 3, 4, 5], feature_size_b=[5, 5, 3, 4])
    in_node_a = tensor_dot_misc.get_placeholder("input_a")
    in_node_b = tensor_dot_misc.get_placeholder("input_b")
    tensor_dot_node = tf.tensordot(in_node_a, in_node_b, axes=[[3, 1], [1, 2]])  # 36,4,5,4
    permute_axis = tf.transpose(tensor_dot_node, perm=[0, 1, 3, 2])  # 36,4,4,5
    batch_to_space_node_a = tf.batch_to_space_nd(permute_axis, block_shape=(1, 4), crops=[[0, 0], [1, 2]])  # 9,4,13,5
    batch_to_space_node_b = tf.batch_to_space(batch_to_space_node_a, block_size=3, crops=[[1, 5], [4, 3]])  # 1,6,32,5
    space_to_depth_node = tf.round(tf.space_to_depth(batch_to_space_node_b, block_size=2))  # 1,3,16,20
    some_add = tf.add(tf.Variable(tf.random_normal((16, 20), dtype=tf.float64)), space_to_depth_node)  # broadcast
    out_node = tf.round(some_add, name="output")

    placeholders = [in_node_a, in_node_b]
    predictions = [out_node]

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="tensor_dot_misc")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(tensor_dot_misc.get_test_data()) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_tensor_dot_misc()
