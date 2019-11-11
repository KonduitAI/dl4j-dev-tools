import tensorflow as tf
import numpy as np

from tfoptests.test_graph import TestGraph
from tfoptests.persistor import TensorFlowPersistor

'''
graph configuration is simply with the intent of including ops for import test...
'''


class PoolND(TestGraph):
    def __init__(self, *args, **kwargs):
        super(PoolND, self).__init__(*args, **kwargs)
        self.input_0 = np.random.uniform(size=(1, 1, 2, 2, 2))

    def list_inputs(self):
        return ["input"]

    def get_placeholder_input(self, name):
        if name == "input":
            return self.input_0

    def _get_placeholder_shape(self, name):
        if name == "input":
            # Tensor of rank N+2, of shape [batch_size] + input_spatial_shape + [num_channels]
            return [None, 1, 2, 2, 2]


def test_conv_5():
    pool_nd = PoolND(seed=124)
    in_node = pool_nd.get_placeholder("input", data_type=tf.float32)
    w = tf.Variable(tf.random_normal([2, 2]), name="in0", dtype=tf.float32)
    # in1 = tf.nn.pool(input=in_node, window_shape=[1, 2, 2], pooling_type="MAX", padding="SAME")
    #for so - called "global normalization", used with convolutional filters with shape[batch, height, width, depth], pass axes=[0, 1, 2].
    in2 = tf.nn.relu_layer(tf.reshape(in_node, [4, 2]), w, tf.random_normal([2]))
    in3 = tf.reciprocal(in2)
    in4 = tf.squared_difference(in2, in3)
    out_node = tf.nn.l2_loss(in4, name="output")
    placeholders = [in_node]
    predictions = [out_node]

    tfp = TensorFlowPersistor(save_dir="conv_5")
    predictions_after_freeze = tfp \
        .set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(pool_nd.get_test_data()) \
        .build_save_frozen_graph()
    print(predictions_after_freeze[0].shape)

if __name__ == '__main__':
    test_conv_5()
