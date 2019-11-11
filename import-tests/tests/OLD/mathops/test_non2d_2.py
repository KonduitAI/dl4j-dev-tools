import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph


class NonTwoDTwo(TestGraph):
    def __init__(self, *args, **kwargs):
        super(NonTwoDTwo, self).__init__(*args, **kwargs)
        self.input_rank2df = np.random.uniform(size=(1, 2))
        self.input_rank2db = np.random.uniform(size=(2, 1))
        self.input_rank3d = np.random.uniform(size=(1, 3, 2))
        self.input_rank3db = np.random.uniform(size=(3, 1, 2))

    def list_inputs(self):
        return ["rank2df", "rank2db", "rank3d", "rank3db"]

    def get_placeholder_input(self, name):
        if name == "rank2df":
            return self.input_rank2df
        if name == "rank2db":
            return self.input_rank2db
        if name == "rank3d":
            return self.input_rank3d
        if name == "rank3db":
            return self.input_rank3db

    def _get_placeholder_shape(self, name):
        if name == "rank2df":
            return self.input_rank2df.shape
        if name == "rank2db":
            return self.input_rank2db.shape
        if name == "rank3d":
            return self.input_rank3d.shape
        if name == "rank3db":
            return self.input_rank3db.shape


def test_nontwod_two():
    non_twod_2 = NonTwoDTwo(seed=13)
    in_node_0 = non_twod_2.get_placeholder("rank2df")  # [1,2]
    in_node_1 = non_twod_2.get_placeholder("rank2db")
    in_node_2 = non_twod_2.get_placeholder("rank3d")
    in_node_3 = non_twod_2.get_placeholder("rank3db")

    i0 = tf.squeeze(in_node_0)
    k0 = tf.Variable(np.random.uniform(size=(2, 1)), name="someweight", dtype=tf.float64)

    i1 = tf.stack([in_node_0, tf.transpose(in_node_1)])
    i2 = tf.multiply(i1, i0)
    i3 = tf.tile(i1, [1, 3, 1])  # 2,3,2
    i4 = tf.transpose(in_node_2) + i3  # 2,3,2
    i5 = tf.reduce_sum(i4)  # now a scalar

    i6 = tf.squeeze(in_node_2 + i5)  # 3,2
    i7 = in_node_3 + i6  # 3,3,2

    i8 = tf.unstack(k0, axis=0)[0]  # vector (1,)
    i9 = tf.unstack(k0, axis=1)[0]  # vector (2,)
    i10 = i9 + i8 + i7

    out_node = tf.identity(i10, name="output")

    placeholders = [in_node_0, in_node_1, in_node_2, in_node_3]
    predictions = [out_node]

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="non2d_2")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(non_twod_2.get_test_data()) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_nontwod_two()
