import tensorflow as tf
import numpy as np


class TestGraph(object):
    def __init__(self, seed=None, verbose=True):
        tf.set_random_seed(1)
        seed = 713 if seed is None else seed
        np.random.seed(seed=seed)
        self.verbose = verbose
        self.seed = seed

    def get_placeholder_input(self, name):
        '''Get input tensor for given node name'''
        return None

    def _get_placeholder_shape(self, name):
        '''Get input tensor shape for given node name'''
        return None

    def list_inputs(self):
        '''List names of input nodes'''
        return ["input"]

    def get_placeholder(self, name, data_type="float64"):
        return tf.placeholder(dtype=data_type, shape=self._get_placeholder_shape(name), name=name)

    def get_test_data(self):
        test_dict = {}
        for an_input in self.list_inputs():
            test_dict[an_input] = self.get_placeholder_input(an_input)
        return test_dict
