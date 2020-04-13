import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor

tf.random.set_seed(1)


def test_simplecond():
    in0 = tf.Variable(np.linspace(1, 4, 4) + 1, name='greater')
    in1 = tf.Variable(np.linspace(1, 4, 4), name='lesser')

    def f1(): return in0 / tf.Variable(2.0, name='div_f1_constant', dtype=tf.float64)

    def f2(): return in1 * tf.Variable(4.0, name='mul_f2_constant', dtype=tf.float64)

    def check(): return tf.reduce_sum(in0 - in1) < 2

    r_node = tf.cond(tf.reduce_sum(in0 - in1) < 2, true_fn=lambda: f1(), false_fn=lambda: f2(), name='cond5')
    r2 = tf.cond(tf.reduce_sum(in0 - in1) < 2, true_fn=lambda: f1(), false_fn=lambda: f2(), name='cond6')

    last_result = tf.add(r_node, tf.constant(1.0, dtype=tf.float64), name='first_output_input')
    last_result2 = tf.add(r2, tf.constant(1.0, dtype=tf.float64), name='second_output_input')
    out_node = tf.add(last_result, last_result2, name='output')

    predictions = [out_node]

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="simple_cond")
    tfp.set_placeholders([]) \
        .set_output_tensors(predictions) \
        .set_test_data({}) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_simplecond()
