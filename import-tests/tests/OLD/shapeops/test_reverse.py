import tensorflow as tf
import numpy as np
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph


class ReverseTest(TestGraph):
    def list_inputs(self):
        return []


def test_reverse():
    shape = [[1], [5], [3,4], [3,4,5]]

    for s in shape:
        rank = len(s)
        for axis1 in range(-rank, rank):
            test = ReverseTest(seed=13)
            #tf.reset_default_graph()
            node = tf.range(np.prod(s), dtype=tf.float32)
            node2 = tf.reshape(node, s)
            node2 = tf.Variable(node2)
            #node2 = tf.Variable(tf.constant(5, dtype=tf.float32, shape=s))
            out_node = tf.reverse(node2, [axis1])

            placeholders = []
            predictions = [out_node]
            outPath = "reverse/shape" + ','.join(str(x) for x in s) + '-axis' + str(axis1)
            print("Output path: " + outPath)
            tfp = TensorFlowPersistor(save_dir=outPath)
            tfp.set_placeholders(placeholders) \
                    .set_output_tensors(predictions) \
                    .set_test_data(test.get_test_data()) \
                    .build_save_frozen_graph()

if __name__ == '__main__':
    test_reverse()
