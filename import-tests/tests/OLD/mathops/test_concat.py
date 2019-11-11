import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph


class ConcatTest(TestGraph):
    def list_inputs(self):
        return []


def test_concat_one():
    concat_test = ConcatTest(seed=13)
    arrs = []
    for i in range(1, 5, 1):
        arrs.append(tf.Variable(tf.constant(5, dtype=tf.float32, shape=(1, 1), name=str(str(i) + '_num'))))
    out_node = tf.concat(arrs, 0, name='output')

    placeholders = []
    predictions = [out_node]

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="concat")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(concat_test.get_test_data()) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_concat_one()
