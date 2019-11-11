import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor


def test_stack_scalar():
    arrs = []
    for i in range(1, 5, 1):
        arrs.append(tf.Variable(tf.constant(5, dtype=tf.float32, shape=([]), name=str(str(i) + '_num'))))

    out_node = tf.stack(arrs, 0, name='output')
    predictions = [out_node]

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="stack_scalar")
    tfp.set_placeholders([]) \
        .set_output_tensors(predictions) \
        .set_test_data({}) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_stack_scalar()
