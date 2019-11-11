import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph

n_hidden_1 = 10
num_input = 5
mini_batch = 4
num_classes = 3


class VanillaMLP(TestGraph):
    def list_inputs(self):
        return ["input"]

    def get_placeholder_input(self, name):
        if name == "input":
            input_0 = np.random.uniform(size=(mini_batch, num_input))
            return input_0

    def _get_placeholder_shape(self, name):
        if name == "input":
            return [None, num_input]


def test_vanilla_mlp():
    vanilla_mlp = VanillaMLP(seed=1337)
    in_node = vanilla_mlp.get_placeholder("input")
    weights = dict(
        h1=tf.Variable(tf.random_normal([num_input, n_hidden_1], dtype=tf.float64),
                       name="l0W"),
        out=tf.Variable(tf.random_normal([n_hidden_1, num_classes], dtype=tf.float64),
                        name="l1W")
    )
    biases = dict(
        b1=tf.Variable(tf.random_normal([n_hidden_1], dtype=tf.float64), name="l0B"),
        out=tf.Variable(tf.random_normal([num_classes], dtype=tf.float64), name="l1B")
    )
    # Define model
    layer_1 = tf.nn.bias_add(tf.matmul(in_node, weights['h1']), biases['b1'], name="l0Preout")
    layer_1_post_actv = tf.abs(layer_1, name="l0Out")
    logits = tf.nn.bias_add(tf.matmul(layer_1_post_actv, weights['out']), biases['out'], name="l1PreOut")
    out_node = tf.nn.softmax(logits, name='output')

    placeholders = [in_node]
    predictions = [out_node]

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="mlp_00")
    predictions = tfp \
        .set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(vanilla_mlp.get_test_data()) \
        .build_save_frozen_graph()
    print(predictions)


if __name__ == '__main__':
    test_vanilla_mlp()
