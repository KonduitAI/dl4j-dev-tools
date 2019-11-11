import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor

tf.set_random_seed(1)


def test_simple_while():
    i1 = tf.Variable(tf.constant(0), name='loop_var')
    c = lambda i: tf.less(i, 10)
    b = lambda i: tf.add(i, 1)
    r = tf.while_loop(c, b, [i1])
    out_node = tf.identity(r, name="output")
    predictions = [out_node]

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="simple_while")
    tfp.set_placeholders([]) \
        .set_output_tensors(predictions) \
        .set_test_data({}) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_simple_while()
