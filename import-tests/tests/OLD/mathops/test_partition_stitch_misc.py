import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph

'''
No training.
Tensor Transforms with rearranging values and some random ops
'''


class TensorRearrange(TestGraph):
    def __init__(self, *args, **kwargs):
        super(TensorRearrange, self).__init__(*args, **kwargs)
        self.a = np.random.uniform(size=(2, 5, 4))
        self.b = np.random.uniform(size=(2, 3, 5, 4))
        self.c = np.random.uniform(size=(3, 1, 5, 4))

    def list_inputs(self):
        return ["input_0", "input_1", "input_2"]

    def get_placeholder_input(self, name):
        if name == "input_0":
            return self.a
        if name == "input_1":
            return self.b
        if name == "input_2":
            return self.c

    def _get_placeholder_shape(self, name):
        if name == "input_0":
            return self.a.shape
        if name == "input_1":
            return self.b.shape
        if name == "input_2":
            return self.c.shape


def test_tensor_rearrange():
    tensor_rearrange = TensorRearrange(seed=713)
    in_node_a = tensor_rearrange.get_placeholder("input_0")
    in_node_b = tensor_rearrange.get_placeholder("input_1")
    in_node_c = tensor_rearrange.get_placeholder("input_2")
    stitched = tf.dynamic_stitch([[1, 10], [[0, 7, 9], [5, 8, 3]], [[6], [4], [2]]],
                                 [in_node_a, in_node_b, in_node_c])  # should be 11,5,4
    list_of_parts = tf.dynamic_partition(tf.transpose(stitched, perm=[1, 2, 0]),
                                         [[0, 1, 2, 3], [1, 0, 2, 3], [2, 3, 1, 0], [2, 1, 0, 3], [0, 1, 2, 3]],
                                         num_partitions=4)  # after permute becomes 5,4,11, return all partitions 5,11
    node_a = tf.div(list_of_parts[0], list_of_parts[1])
    node_b = tf.divide(list_of_parts[2], list_of_parts[3])
    trace_node = tf.trace(node_a) + node_b  # there is a broadcast here
    out_node = tf.cast(tf.count_nonzero(trace_node), dtype=tf.float32) + tf.Variable(tf.random_normal(shape=(2, 3)))

    placeholders = [in_node_a, in_node_b, in_node_c]
    predictions = [out_node]

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="partition_stitch_misc")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(tensor_rearrange.get_test_data()) \
        .build_save_frozen_graph()

if __name__ == '__main__':
    test_tensor_rearrange()