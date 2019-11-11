import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph
from tfoptests.reduce_ops import ReduceOps


class Reductions(TestGraph):
    def __init__(self, numInputs=1, *args, **kwargs):
        super(Reductions, self).__init__(*args, **kwargs)
        self.numInputs = numInputs
        #self.innames = ["in_" + str(i) for i in range(numInputs)]
        self.innames = []
        self.inshapes = {}
        self.invals = {}

    def list_inputs(self):
        return self.innames

    def _get_placeholder_shape(self, name):
        '''Get input tensor shape for given node name'''
        return self.inshapes[name]

    def get_placeholder_input(self, name):
        '''Get input tensor for given node name'''
        return self.invals[name]

    def getVars(self, shapes, dtypes, init, isPlaceholder):
        print("shapes: ", shapes)
        print("dtypes: ", dtypes)
        out = []
        # for(s in shapes):
        for i in range(len(shapes)):
            s = shapes[i]
            d = tf.float32
            if(dtypes is not None):
                d = dtypes[i]

            n = "in_" + str(i)

            if(isPlaceholder is not None and isPlaceholder[i] is True):
                out.append(tf.placeholder(dtype=d, shape=shapes[i], name=n))

                if(d == tf.bool):
                    self.invals[n] = (np.random.random(s) >= 0) != 0
                elif(d == tf.float32):
                    set = False
                    if(init is not None):
                        if(len(init) > i and init[i] is not None):
                            if(init[i] == "uniform"):
                                self.invals[n] = np.random.random(size=s)
                                set = True
                            elif(init[i] == "uniform10"):
                                self.invals[n] = np.random.random(size=s) * 10.0
                                set = True
                    if(set != True):
                        self.invals[n] = np.random.normal(size=s)
                elif(d == tf.int32):
                    self.invals[n] = np.random.randint(low=0, high=10, size=s, dtype=np.int32)
                else:
                    raise Exception("Datatype not implemented for placeholder")

                print("Appended placeholder: " + str(i) + " - " + n + " - shape " + str(self.invals[n]))
            else:
                if(d == tf.bool):
                    out.append(tf.Variable(tf.random_normal(s) >= 0, tf.bool, name=n))
                elif(d == tf.float32):
                    set = False
                    if(init is not None):
                        if(len(init) > i and init[i] is not None):
                            if(init[i] == "uniform"):
                                out.append(tf.Variable(tf.random_uniform(s), tf.float32, name=n))
                                set = True
                            elif(init[i] == "uniform10"):
                                out.append(tf.Variable(tf.random_uniform(s, minval=0.0, maxval=10.0), tf.float32, name=n))
                                set = True
                    if(set != True):
                        out.append(tf.Variable(tf.random_normal(s), tf.float32, name=n))
                elif(d == tf.int32):
                    set = False
                    if(init is not None):
                        if(len(init) > i and init[i] is not None):
                            if(init[i] == "uniform5"):
                                out.append(tf.Variable(tf.random_uniform(s, minval=0.0, maxval=5.0), tf.int32, name=n))
                            if(init[i] == "uniform10"):
                                out.append(tf.Variable(tf.random_uniform(s, minval=0.0, maxval=10.0), tf.int32, name=n))
                    if(set != True):
                        out.append(tf.Variable(tf.random_uniform(s, minval=1, maxval=10, dtype=tf.int32), name=n))
                else:
                    raise Exception("Datatype not implemented for variable")

                print("Appended variable: " + str(i) + " - " + n)

        return out


def test_mathtransform():
    ops = [
        #Format: [opName, testName, inputShapes, inputTypes, axis, extra, random_init, placeholder]
        # ["reduce_sum", "sum_0", [[3,4]], None, [0], {"keepdims":False}],
        # ["reduce_sum", "sum_1keep", [[3,4]], None, [1], {"keepdims":True}],
        # ["reduce_sum", "sum_01", [[3,4]], None, [0,1], {"keepdims":False}],
        # ["reduce_sum", "sum_012keep", [[3,4,5]], None, [0,1,2], {"keepdims":True}],
        # ["reduce_sum", "sum_all", [[3,4,5]], None, None, {"keepdims":False}],
        # ["reduce_sum", "sum_scalar", [[]], None, None, {"keepdims":False}],
        # ["reduce_max", "max_0", [[3,4]], None, [0], {"keepdims":False}],
        # ["reduce_max", "max_1keep", [[3,4]], None, [1], {"keepdims":True}],
        # ["reduce_max", "max_01", [[3,4]], None, [0,1], {"keepdims":False}],
        # ["reduce_max", "max_012keep", [[3,4,5]], None, [0,1,2], {"keepdims":True}],
        # ["reduce_max", "max_all", [[3,4,5]], None, None, {"keepdims":False}],
        # ["reduce_max", "max_scalar", [[]], None, None, {"keepdims":False}],
        # ["reduce_min", "min_0", [[3,4]], None, [0], {"keepdims":False}],
        # ["reduce_min", "min_1keep", [[3,4]], None, [1], {"keepdims":True}],
        # ["reduce_min", "min_01", [[3,4]], None, [0,1], {"keepdims":False}],
        # ["reduce_min", "min_012keep", [[3,4,5]], None, [0,1,2], {"keepdims":True}],
        # ["reduce_min", "min_all", [[3,4,5]], None, None, {"keepdims":False}],
        # ["reduce_min", "min_scalar", [[]], None, None, {"keepdims":False}],
        # ["reduce_mean", "mean_1keep", [[3,4]], None, [1], {"keepdims":True}],
        # ["reduce_mean", "mean_01", [[3,4]], None, [0,1], {"keepdims":False}],
        # ["reduce_mean", "mean_012keep", [[3,4,5]], None, [0,1,2], {"keepdims":True}],
        # ["reduce_mean", "mean_all", [[3,4,5]], None, None, {"keepdims":False}],
        # ["reduce_mean", "mean_scalar", [[]], None, None, {"keepdims":False}],
        # ["reduce_prod", "prod_1keep", [[3,4]], None, [1], {"keepdims":True}],
        # ["reduce_prod", "prod_01", [[3,4]], None, [0,1], {"keepdims":False}],
        # ["reduce_prod", "prod_012keep", [[3,4,5]], None, [0,1,2], {"keepdims":True}],
        # ["reduce_prod", "prod_all", [[3,4,5]], None, None, {"keepdims":False}],
        # ["reduce_prod", "prod_scalar", [[]], None, None, {"keepdims":False}]
        # ["argmax", "argmax3,4_0", [[3,4]], None, 0, None],
        # ["argmax", "argmax3,4_1", [[3,4]], None, 1, None],
        # ["argmax", "argmax3,4_-2", [[3,4]], None, -2, None],
        # ["argmax", "argmax3,4,5_-1", [[3,4,5]], None, -1, None],
        # ["argmin", "argmin3,4_0", [[3,4]], None, 0, None],
        # ["argmin", "argmin3,4_1", [[3,4]], None, 1, None],
        # ["argmin", "argmin3,4_-1", [[3,4]], None, -1, None],
        # ["argmin", "argmin3,4,5_-2", [[3,4,5]], None, -2, None],
        # ["add_n", "add_n", [[3,4], [3,4], [3,4]], None, None, None],
        # ["add_n", "add_n_single", [[3,4]], None, None, None],
        # ["add_n", "add_n_single_scalar", [[]], None, None, None]
        # ["normalize_moments", "normalize_moments", [[], [5], [5]], [tf.float32, tf.float32, tf.float32], None, ["uniform10", None, "uniform"]],  #Args: count, mean_ss, variance_ss, shift
        # ["normalize_moments", "normalize_moments_shift", [[], [5], [5], []], [tf.float32, tf.float32, tf.float32, tf.float32], None, None, ["uniform10", None, "uniform", "uniform"]]
        # ["normalize_moments", "normalize_moments_rank2shift", [[], [5,5], [5,5], []], [tf.float32, tf.float32, tf.float32, tf.float32], None, None, ["uniform10", None, "uniform", "uniform"]]
        # ["normalize_moments", "normalize_moments_rank3", [[], [3,4,5], [3,4,5]], [tf.float32, tf.float32, tf.float32], None, None, ["uniform10", None, "uniform"]]
        # ["count_nonzero", "count_nonzero_0", [[3,4]], None, [0], {"keepdims":False}],
        # ["count_nonzero", "count_nonzero_1", [[3,4]], None, [1], {"keepdims":False}],
        # ["count_nonzero", "count_nonzero_1keep", [[3,4]], None, [1], {"keepdims":True}],
        # ["count_nonzero", "count_nonzero_all", [[3,4,5]], None, None, {"keepdims":False}],
        # ["count_nonzero", "count_nonzero_scalar", [[]], None, None, {"keepdims":False}],
        # ["count_nonzero", "count_nonzero_345_-1", [[3,4,5]], None, [-1], {"keepdims":False}],
        # ["moments", "moments0", [[3,4]], None, [0], {"keepdims":False} ],
        # ["moments", "moments1", [[3,4]], None, [1], {"keepdims":False} ],
        # ["moments", "moments01", [[3,4]], None, [0,1], {"keepdims":False} ],
        # ["moments", "moments1keep", [[3,4]], None, [1], {"keepdims":True} ],
        # ["moments", "moments345-02", [[3,4,5]], None, [0,2], {"keepdims":False} ],
        # ["moments", "moments2345-023", [[2,3,4,5]], None, [0,2,3], {"keepdims":True} ]

        #Problem here: ref should be a variable, indices/updates should be placeholder not variable
        #Order of args: ref, indices, updates
        # ["scatter_add", "scatter_add_scalar", [[10,3], [], [3]], [tf.float32, tf.int32, tf.float32], None, None, None, [False, True, True]],
        # ["scatter_add", "scatter_add_vector", [[10,3], [2], [2,3]], [tf.float32, tf.int32, tf.float32], None, None, None, [False, True, True]],
        # ["scatter_sub", "scatter_sub_scalar", [[10,3], [], [3]], [tf.float32, tf.int32, tf.float32], None, None, None, [False, True, True]],
        # ["scatter_sub", "scatter_sub_vector", [[10,3], [2], [2,3]], [tf.float32, tf.int32, tf.float32], None, None, None, [False, True, True]],
        # ["scatter_mul", "scatter_mul_scalar", [[10,3], [], [3]], [tf.float32, tf.int32, tf.float32], None, None, None, [False, True, True]],
        # ["scatter_mul", "scatter_mul_vector", [[10,3], [2], [2,3]], [tf.float32, tf.int32, tf.float32], None, None, None, [False, True, True]],
        # ["scatter_div", "scatter_div_scalar", [[10,3], [], [3]], [tf.float32, tf.int32, tf.float32], None, None, None, [False, True, True]],
        # ["scatter_div", "scatter_div_vector", [[10,3], [2], [2,3]], [tf.float32, tf.int32, tf.float32], None, None, None, [False, True, True]],
        # ["scatter_update", "scatter_update_scalar", [[10,3], [], [3]], [tf.float32, tf.int32, tf.float32], None, None, None, [False, True, True]],
        # ["scatter_update", "scatter_update_vector", [[10,3], [2], [2,3]], [tf.float32, tf.int32, tf.float32], None, None, None, [False, True, True]],
        # ["scatter_max", "scatter_max_scalar", [[10,3], [], [3]], [tf.float32, tf.int32, tf.float32], None, None, None, [False, True, True]],
        # ["scatter_max", "scatter_max_vector", [[10,3], [2], [2,3]], [tf.float32, tf.int32, tf.float32], None, None, None, [False, True, True]],
        # ["scatter_min", "scatter_min_scalar", [[10,3], [], [3]], [tf.float32, tf.int32, tf.float32], None, None, None, [False, True, True]],
        # ["scatter_min", "scatter_min_vector", [[10,3], [2], [2,3]], [tf.float32, tf.int32, tf.float32], None, None, None, [False, True, True]],
        # # TODO Matrix case is giving obscure error: ValueError: Shapes must be equal rank, but are 3 and 4 for 'scatter_add-1' (op: 'ScatterAdd') with input shapes: [5,5,3], [2,2], [2,2,3]
        #["scatter_add", "scatter_add_matrix", [[5,5,3], [2,2], [2,2,3]], [tf.float32, tf.int32, tf.float32], None, None, [None, "uniform5", None], [False, True, True]]

        #Not exporting correctly?
        #import/is_non_decreasing-1/cond/Switch:1
        # Unexpected error: <class 'tensorflow.python.framework.errors_impl.InvalidArgumentError'>
        # Tensor("import/is_non_decreasing-1/cond/Switch:1", shape=(), dtype=bool)
        # SKIPPING
        #["is_non_decreasing", "is_non_decreasing_3-4", [[3,4]], None, None, None],
        #["is_non_decreasing", "is_non_decreasing_scalar", [[]], None, None, None]
           ]

    # max, mean, min, prod, sum



    for op in ops:
        tf.reset_default_graph()
        print("Running " + str(op))
        test = Reductions(seed=19, numInputs=len(op[2]))

        # print("op[2]: ", op[2])
        # print("op[3]: ", op[3])

        init = None
        if(len(op) > 6):
            init = op[6]

        isPlaceholder = None
        if(len(op) > 7):
            isPlaceholder = op[7]

        vars = test.getVars(op[2], op[3], init, isPlaceholder)

        placeholders = []
        if(isPlaceholder is not None):
            for i in range(len(isPlaceholder)):
                if(isPlaceholder[i] is True):
                    placeholders.append(vars[i])
                    n = "in_" + str(i)
                    test.inshapes[n] = op[2][i]
                    test.innames.append(n)

        reduction = ReduceOps(vars, op[2], op[3], op[4], op[5])
        out = reduction.execute(op[0])

        print(out)

        # Run and persist
        testName = "reductions/" + op[1]
        tfp = TensorFlowPersistor(save_dir=testName)
        tfp.set_placeholders(placeholders) \
            .set_output_tensors(out) \
            .set_test_data(test.get_test_data()) \
            .build_save_frozen_graph()


if __name__ == '__main__':
    test_mathtransform()
