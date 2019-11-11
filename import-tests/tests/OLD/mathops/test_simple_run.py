import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph

'''
This is meant to be a simple placeholder to implement ONE op at a time in a graph and write out pbs to 
test op mapping in nd4j samediff import
Donot check this in to the main branch as this is a one time thing...
Implement your graph with the op in question under def test_simple and run it
Will write pbs and relevant files to $DL4J_TEST_RESOURCES/src/main/resources/tf_graphs/examples/simple_run
Run mvn clean install in dl4j-test-resources and then run TFGraphTestList import test in nd4j with model_name set to simple_run
Make sure to run with both libnd4j executor and samediff executor
eg. below is with one hot op and no placeholders 
'''


class SimpleRun(TestGraph):
    def __init__(self, *args, **kwargs):
        super(SimpleRun, self).__init__(*args, **kwargs)
        # self.input_1 = [[0, 2], [1, -1]]

    def list_inputs(self):
        return []


'''
    def get_placeholder_input(self, name):
        if name == "input_1":
            return self.input_1

    def _get_placeholder_shape(self, name):
        if name == "input_1":
            return [2, 2]
'''


def test_simple():
    simple_run = SimpleRun(seed=19)
    in_node_1 = tf.Variable([[0, 2], [1, -1]])
    out_node = tf.one_hot(in_node_1, 3, axis=1, off_value=-2.0, name="output")

    predictions = [out_node]
    # Run and persist
    tfp = TensorFlowPersistor(save_dir="simple_run")
    tfp.set_placeholders([]) \
        .set_output_tensors(predictions) \
        .set_test_data(simple_run.get_test_data()) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_simple()
