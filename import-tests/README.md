# TFOpTests - Generate, persist and load tensorflow graphs for tests [![Build Status](https://travis-ci.org/deeplearning4j/TFOpTests.svg?branch=master)]

## Setup

To get started with this project first clone the DL4J test resources repository into
a folder of your choice:
```bash
git clone https://github.com/deeplearning4j/dl4j-test-resources
cd dl4j-test-resources
```

Next, put `DL4J_TEST_RESOURCES` on your path, for instance by adding it to your `.zshrc` (or `.bashrc` etc.):

```bash
echo "export DL4J_TEST_RESOURCES=$(pwd)" >> $HOME/.zshrc
```

This is the path used in this project to generate test resources. If you wish to store results in
another path, just change the above resource path accordingly.

Finally, install this library locally by running `python setup.py develop`. It is recommended to
work with a Python virtual environment.

## Usage

To run a single test, for instance for a simple MLP with non-trivial bias terms in layers, run the following:

```python
python tests/mlp/test_bias_add.py
```

To generate all test resources, simply run:
```python
python -m pytest
```

## Adding new tests

The base for adding any new tests is extending `TestGraph`, which is defined in `tfoptests.test_graph`. 
You will have to override functionality, as suitable, for all the methods except the `get_test_data` and the `get_placeholder` methods:

```python
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
```

These methods specify the input data and its shape for the graph we want to run, persist and test.

Below we walk through a very elementary example where we set up a graph that contains a matrix multiply and run it through some methods with the `TensorFlowPersistor`

```python
class MatMulOrder(TestGraph):
    def list_inputs(self):
        return ["input_0", "input_1"]

    def get_placeholder_input(self, name):
        if name == "input_0":
            input_0 = np.random.uniform(size=(3, 3))
            return input_0
        if name == "input_1":
            input_1 = np.random.uniform(size=(3, 3)) + np.random.uniform(size=(3, 3))
            return input_1

    def _get_placeholder_shape(self, name):
        if name == "input_0" or name == "input_1":
            return [None, 3]
```

The class above inherits from TestGraph and all we do is specify input names and input shapes.

And to test we add in functionality and ops as needed as shown below:

```python
def test_mat_mul_order():
    simple_m = MatMulOrder(seed=713)
    in0 = simple_m.get_placeholder("input_0")
    in1 = simple_m.get_placeholder("input_1")
    k0 = tf.Variable(tf.random_normal([3, 3], dtype=tf.float64), name="in0")
    out_node = tf.matmul(k0, tf.matmul(in0, in1), name="output")

    placeholders = [in0, in1]
    predictions = [out_node]
    # Run and persist
    tfp = TensorFlowPersistor(save_dir="math_mul_order")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(simple_m.get_test_data()) \
        .build_save_frozen_graph()
```

Note the `TensorFlowPersistor` (TFP) method call. 
Given the input/placeholder tensors along with their values and the output tensors under the hood it will:
- persist the tf graph after freezing
- write graph inputs, graph outputs and intermediate node results
- run asserts to ensure that predictions before and after freezing are the same
etc.

It is the .get_test_data() method implemented in TestGraph that provides the TFP with the placeholder name-value dict it needs to run through the graph.


These graphs are then used in integration tests for our _tensorflow model import_ . Graphs are imported into samediff and checks are run to ensure the correctness of the libnd4j implementation and its mapping.

Checks on the java side can be found in the nd4j repository in this [package](https://github.com/deeplearning4j/nd4j/tree/master/nd4j-backends/nd4j-tests/src/test/java/org/nd4j/imports/TFGraphs). Take a look at [TFGraphTestAllSameDiff](https://github.com/deeplearning4j/nd4j/tree/master/nd4j-backends/nd4j-tests/src/test/java/org/nd4j/imports/TFGraphs/TFGraphTestAllSameDiff.java) to see how checks are run with the SameDiff executor.

## Contributing

If there are missing operations or architectures that need to be covered, make sure to file an issue or open a pull request.  
