import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor


def test_unstack():
    arrs = tf.Variable(tf.constant(np.reshape(np.linspace(1, 25, 25), (5, 5))))
    unstack_list = tf.unstack(arrs, axis=0)
    out_node = tf.reduce_sum(unstack_list, axis=0, name="output")
    # Run and persist
    tfp = TensorFlowPersistor(save_dir="unstack")
    tfp.set_placeholders([]) \
        .set_output_tensors([out_node]) \
        .set_test_data({}) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_unstack()
