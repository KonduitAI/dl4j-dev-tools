

import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph
from tfoptests.reduce_ops import ReduceOps
from tfoptests.ops import OpCreator
from tfoptests.var_initializer import VarInitializer

class OpTest(TestGraph):
    def __init__(self, op, *args, **kwargs):
        super(OpTest, self).__init__(*args, **kwargs)
        self.op = op

    def list_inputs(self):
        return self.op.get("phNames", [])

    def _get_placeholder_shape(self, name):
        '''Get input tensor shape for given node name'''
        return self.op.get("phShapes", {})

    def get_placeholder_input(self, name):
        '''Get input tensor for given node name'''
        return self.invals[name]

    def createVars(self, shapes, dtypes, init):
        print("Creating vars: shapes=", shapes, ", dtypes=", dtypes, ", init=", init)
        out = []
        initializer = VarInitializer()
        # for(s in shapes):
        for i in range(len(shapes)):
            s = shapes[i]
            d = tf.float32
            if(dtypes is not None):
                d = dtypes[i]

            n = "in_" + str(i)

            varInit = "uniform"
            if(init is not None and init[i] is not None):
                varInit = init[i]

            out.append(initializer.newVar(varInit, s, d, n))

        return out

    def createPlaceholders(self, shapes, dtypes, init):
        print("Creating vars: shapes=", shapes, ", dtypes=", dtypes, ", init=", init)
        out = []
        initializer = VarInitializer()
        for i in range(len(shapes)):
            s = shapes[i]
            d = tf.float32
            if(dtypes is not None):
                d = dtypes[i]

            n = "in_ph_" + str(i)

            varInit = "uniform"
            if(init is not None and init[i] is not None):
                varInit = init[i]

            out.append(initializer.newPlaceholder(varInit, s, d, n))

        return out

def test_mathtransform():
    ops = [
        #Format:
        #{"opName": "segment_max", "outName": "segment/segment_max_rank1", "varShapes":[[20], [20]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment5"]},
        # {"opName": "segment_mean", "outName": "segment/segment_mean_rank1", "varShapes":[[20], [20]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment5"]},
        # {"opName": "segment_mean", "outName": "segment/segment_mean_rank2", "varShapes":[[6,3], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment3"]},

        # {"opName": "segment_min", "outName": "segment/segment_min_rank1", "varShapes":[[20], [20]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment5"]},

        # {"opName": "segment_prod", "outName": "segment/segment_prod_rank1", "varShapes":[[20], [20]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment5"]},

        # {"opName": "segment_sum", "outName": "segment/segment_sum_rank1", "varShapes":[[20], [20]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment5"]},

        # {"opName": "segment_max", "outName": "segment/segment_max_rank2", "varShapes":[[6,3], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment3"]},

        # {"opName": "segment_min", "outName": "segment/segment_min_rank2", "varShapes":[[6,3], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment3"]},
        # {"opName": "segment_prod", "outName": "segment/segment_prod_rank2", "varShapes":[[6,3], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment3"]},
        # {"opName": "segment_sum", "outName": "segment/segment_sum_rank2", "varShapes":[[6,3], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment3"]},
        # {"opName": "segment_max", "outName": "segment/segment_max_rank3", "varShapes":[[6,3,2], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment3"]},
        # {"opName": "segment_mean", "outName": "segment/segment_mean_rank3", "varShapes":[[6,3,2], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment3"]},
        # {"opName": "segment_min", "outName": "segment/segment_min_rank3", "varShapes":[[6,3,2], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment3"]},
        # {"opName": "segment_prod", "outName": "segment/segment_prod_rank3", "varShapes":[[6,3,2], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment3"]},
        # {"opName": "segment_sum", "outName": "segment/segment_sum_rank3", "varShapes":[[6,3,2], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment3"]},
        # {"opName": "space_to_batch", "outName": "space_to_batch/rank4nhwc", "varShapes":[[2,4,4,4], [2,2]], "varTypes":["float32", "int32"], "varInit":["range", "zero"]},
        # {"opName": "space_to_batch", "outName": "space_to_batch/rank4nhwc_pad", "varShapes":[[2,2,2,4], [2,2]], "varTypes":["float32", "int32"], "varInit":["range", "one"]},
        # {"opName": "space_to_depth", "outName": "space_to_depth/rank4nhwc", "varShapes":[[2,4,4,4]], "varTypes":["float32", "int32"], "varInit":["range", "zero"], "data_format":"NHWC"},
        # {"opName": "space_to_depth", "outName": "space_to_depth/rank4nchw", "varShapes":[[2,4,4,4]], "varTypes":["float32", "int32"], "varInit":["range", "zero"], "data_format":"NCHW"},
        # {"opName": "batch_to_space", "outName": "batch_to_space/rank4nhwc", "varShapes":[[8,2,2,4], [2,2]], "varTypes":["float32", "int32"], "varInit":["range", "zero"]},
        # {"opName": "batch_to_space", "outName": "batch_to_space/rank4nhwc_crop", "varShapes":[[8,2,2,4], [2,2]], "varTypes":["float32", "int32"], "varInit":["range", "one"]},
        # {"opName": "depth_to_space", "outName": "depth_to_space/rank4nhwc", "varShapes":[[2,4,4,4]], "varTypes":["float32", "int32"], "varInit":["range", "zero"], "data_format":"NHWC"},
        #{"opName": "depth_to_space", "outName": "depth_to_space/rank4nchw", "varShapes":[[2,4,4,4]], "varTypes":["float32", "int32"], "varInit":["range", "zero"], "data_format":"NCHW"},  #Only NHWC format supported on CPU!?
        # {"opName": "size", "outName": "size_rank2", "varShapes":[[3,4]], "varTypes":["float32"]},
        # {"opName": "size", "outName": "size_rank3", "varShapes":[[2,3,4]], "varTypes":["float32"]},
        # {"opName": "shape", "outName": "shape_rank2", "varShapes":[[3,4]], "varTypes":["float32"]},
        # {"opName": "shape", "outName": "shape_rank3", "varShapes":[[2,3,4]], "varTypes":["float32"]}
        # {"opName": "shapen", "outName": "shapen_3x2", "varShapes":[[3,4], [1,2], [2,4]], "varTypes":["float32", "float32", "float32"]},
        # {"opName": "shapen", "outName": "shapen_3x3", "varShapes":[[2,3,4], [1,2,3], [2,1,2]], "varTypes":["float32", "float32", "float32"]}
        # {"opName": "matrix_inverse", "outName": "matrix_inverse/rank2", "varShapes":[[3,3]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName": "matrix_inverse", "outName": "matrix_inverse/rank3", "varShapes":[[2,3,3]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName": "matrix_inverse", "outName": "matrix_inverse/rank4", "varShapes":[[2,2,3,3]], "varTypes":["float32"], "varInit":["uniform"]}
        # {"opName": "pad", "outName": "pad/rank1Pzero_const0", "varShapes":[[5],[1,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","zero","zero"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank1Pzero_const10", "varShapes":[[5],[1,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","zero","ten"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank1Pone_const0", "varShapes":[[5],[1,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","one","zero"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank1Pone_const10", "varShapes":[[5],[1,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","one","ten"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank1Pone_reflect", "varShapes":[[5],[1,2]], "varTypes":["float32", "int32"], "varInit":["uniform","one"], "mode":"REFLECT"},
        # {"opName": "pad", "outName": "pad/rank1Pone_symmetric", "varShapes":[[5],[1,2]], "varTypes":["float32", "int32"], "varInit":["uniform","one"], "mode":"SYMMETRIC"}
        # {"opName": "pad", "outName": "pad/rank2Pzero_const0", "varShapes":[[3,4],[2,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","zero","zero"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank2Pzero_const10", "varShapes":[[3,4],[2,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","zero","ten"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank2Pone_const0", "varShapes":[[3,4],[2,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","one","zero"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank2Pone_const10", "varShapes":[[3,4],[2,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","one","ten"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank2Pone_reflect", "varShapes":[[3,4],[2,2]], "varTypes":["float32", "int32"], "varInit":["uniform","one"], "mode":"REFLECT"},
        # {"opName": "pad", "outName": "pad/rank2Pone_symmetric", "varShapes":[[3,4],[2,2]], "varTypes":["float32", "int32"], "varInit":["uniform","one"], "mode":"SYMMETRIC"},
        # {"opName": "pad", "outName": "pad/rank3Pzero_const0", "varShapes":[[2,3,4],[3,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","zero","zero"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank3Pzero_const10", "varShapes":[[2,3,4],[3,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","zero","ten"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank3Pone_const0", "varShapes":[[2,3,4],[3,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","one","zero"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank3Pone_const10", "varShapes":[[2,3,4],[3,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","one","ten"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank3Pone_reflect", "varShapes":[[2,3,4],[3,2]], "varTypes":["float32", "int32"], "varInit":["uniform","one"], "mode":"REFLECT"},
        # {"opName": "pad", "outName": "pad/rank3Pone_symmetric", "varShapes":[[2,3,4],[3,2]], "varTypes":["float32", "int32"], "varInit":["uniform","one"], "mode":"SYMMETRIC"},
        # {"opName": "unique", "outName": "unique10-5", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform_int5"]},
        # {"opName": "unique_with_counts", "outName": "uniqueWithCounts10-5", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform_int5"]},
        # {"opName": "topk", "outName": "topk/rank1_k1", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform"], "k":1, "sorted":False},
        # {"opName": "topk", "outName": "topk/rank1_k1_sorted", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform"], "k":1, "sorted": True},
        # {"opName": "topk", "outName": "topk/rank1_k5", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform"], "k":5, "sorted":False},
        # {"opName": "topk", "outName": "topk/rank1_k5_sorted", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform"], "k":5, "sorted":True},
        # {"opName": "topk", "outName": "topk/rank2_k1", "varShapes":[[3,6]], "varTypes":["float32"], "varInit":["uniform"], "k":1, "sorted":False},
        # {"opName": "topk", "outName": "topk/rank2_k1_sorted", "varShapes":[[3,6]], "varTypes":["float32"], "varInit":["uniform"], "k":1, "sorted": True},
        # {"opName": "topk", "outName": "topk/rank2_k5", "varShapes":[[3,6]], "varTypes":["float32"], "varInit":["uniform"], "k":5, "sorted":False},
        # {"opName": "topk", "outName": "topk/rank2_k5_sorted", "varShapes":[[3,6]], "varTypes":["float32"], "varInit":["uniform"], "k":5, "sorted":True},
        # {"opName": "topk", "outName": "topk/rank3_k3", "varShapes":[[3,4,5]], "varTypes":["float32"], "varInit":["uniform"], "k":3, "sorted":False},
        # {"opName": "topk", "outName": "topk/rank3_k3_sorted", "varShapes":[[3,4,5]], "varTypes":["float32"], "varInit":["uniform"], "k":3, "sorted":True}
        # {"opName": "in_top_k", "outName": "in_top_k/test_4,5_k1", "varShapes":[[4,5], [4]], "varTypes":["float32", "int32"], "varInit":["uniform", "uniform_int5"], "k":1},
        # {"opName": "in_top_k", "outName": "in_top_k/test_4,5_k3", "varShapes":[[4,5], [4]], "varTypes":["float32", "int32"], "varInit":["uniform", "uniform_int5"], "k":3}
        # {"opName": "matrix_determinant", "outName": "matrix_determinant/rank2_5,5", "varShapes":[[5,5]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName": "matrix_determinant", "outName": "matrix_determinant/rank3_2,3,3", "varShapes":[[2,3,3]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName": "matrix_determinant", "outName": "matrix_determinant/rank4_2,2,3,3", "varShapes":[[2,2,3,3]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName": "matrix_set_diag", "outName": "matrix_set_diag/rank2_5,5", "varShapes":[[5,5], [5]], "varTypes":["float32", "float32"], "varInit":["zeros", "uniform"]},
        # {"opName": "matrix_set_diag", "outName": "matrix_set_diag/rank2_5,4", "varShapes":[[5,4], [4]], "varTypes":["float32", "float32"], "varInit":["zeros", "uniform"]},
        # {"opName": "matrix_set_diag", "outName": "matrix_set_diag/rank2_4,5", "varShapes":[[5,4], [4]], "varTypes":["float32", "float32"], "varInit":["zeros", "uniform"]},
        # {"opName": "matrix_set_diag", "outName": "matrix_set_diag/rank3_2,3,3", "varShapes":[[2,3,3], [2,3]], "varTypes":["float32", "float32"], "varInit":["zeros", "uniform"]},
        # {"opName": "matrix_set_diag", "outName": "matrix_set_diag/rank3_2,3,4", "varShapes":[[2,3,4], [2,3]], "varTypes":["float32", "float32"], "varInit":["zeros", "uniform"]},
        # {"opName": "matrix_set_diag", "outName": "matrix_set_diag/rank4_2,2,3,3", "varShapes":[[2,2,3,3], [2,2,3]], "varTypes":["float32", "float32"], "varInit":["zeros", "uniform"]}
        # {"opName": "identity_n", "outName": "identity_n_2", "varShapes":[[2,3], [2]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"]},
        # {"opName": "identity_n", "outName": "identity_n_4", "varShapes":[[2,3], [2], [], [2,1,3]], "varTypes":["float32", "float32", "float32", "float32"], "varInit":["uniform", "uniform", "uniform", "uniform"]}
        # {"opName": "zeta", "outName": "zeta_rank2", "varShapes":[[2,3], [2,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"]},
        # {"opName": "zeta", "outName": "zeta_rank3", "varShapes":[[2,3,2], [2,3,2]], "varTypes":["float32", "float32", "float32"], "varInit":["uniform", "uniform", "uniformt"]},
        #{"opName": "confusion_matrix", "outName": "confusion/no_num_classes", "varShapes":[[5], [5]], "varTypes":["int32", "int32"], "varInit":["uniform_int5", "uniform_int5"], "num_classes":None},
        #{"opName": "confusion_matrix", "outName": "confusion/with_num_classes", "varShapes":[[5], [5]], "varTypes":["int32", "int32"], "varInit":["uniform_int5", "uniform_int5"], "num_classes":5},
        #{"opName": "confusion_matrix", "outName": "confusion/no_num_classes_with_weights", "varShapes":[[5], [5], [5]], "varTypes":["int32", "int32", "float32"], "varInit":["uniform_int5", "uniform_int5", "uniform"], "num_classes":None},
        #{"opName": "confusion_matrix", "outName": "confusion/with_num_classes_with_weights", "varShapes":[[5], [5], [5]], "varTypes":["int32", "int32", "float32"], "varInit":["uniform_int5", "uniform_int5", "uniform"], "num_classes":5}
        # {"opName": "stack", "outName": "stack/rank0_axis-1", "varShapes":[[], []], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":-1},
        # {"opName": "stack", "outName": "stack/rank0_axis0", "varShapes":[[], []], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":0},
        # {"opName": "stack", "outName": "stack/rank1_axis-2", "varShapes":[[3], [3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":-2},
        # {"opName": "stack", "outName": "stack/rank1_axis-1", "varShapes":[[3], [3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":-1},
        # {"opName": "stack", "outName": "stack/rank1_axis-0", "varShapes":[[3], [3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":0},
        # {"opName": "stack", "outName": "stack/rank1_axis1", "varShapes":[[3], [3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":1},
        # {"opName": "stack", "outName": "stack/rank2_axis-3", "varShapes":[[2,3], [2,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":-3},
        # {"opName": "stack", "outName": "stack/rank2_axis-2", "varShapes":[[2,3], [2,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":-2},
        # {"opName": "stack", "outName": "stack/rank2_axis-1", "varShapes":[[2,3], [2,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":-1},
        # {"opName": "stack", "outName": "stack/rank2_axis-0", "varShapes":[[2,3], [2,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":0},
        # {"opName": "stack", "outName": "stack/rank2_axis1", "varShapes":[[2,3], [2,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":1},
        # {"opName": "stack", "outName": "stack/rank2_axis2", "varShapes":[[2,3], [2,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":2},
        # {"opName": "stack", "outName": "stack/rank3_axis-2", "varShapes":[[2,1,3], [2,1,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":-2},
        # {"opName": "stack", "outName": "stack/rank3_axis0", "varShapes":[[2,1,3], [2,1,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":0},
        # {"opName": "stack", "outName": "stack/rank3_axis3", "varShapes":[[2,1,3], [2,1,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":3},
        #Note that parallel_stack doesn't support axis arg - equivalent to stack with axis=0
        # {"opName": "parallel_stack", "outName": "parallel_stack/rank0", "varShapes":[[], []], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"]},
        # {"opName": "parallel_stack", "outName": "parallel_stack/rank1", "varShapes":[[3], [3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"]},
        # {"opName": "parallel_stack", "outName": "parallel_stack/rank2", "varShapes":[[2,3], [2,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"]},
        # {"opName": "parallel_stack", "outName": "parallel_stack/rank3", "varShapes":[[2,1,3], [2,1,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"]},
        # {"opName": "accumulate_n", "outName": "accumulate_n/rank0", "varShapes":[[], [], []], "varTypes":["float32", "float32", "float32"], "varInit":["uniform", "uniform", "uniform"]},
        # {"opName": "accumulate_n", "outName": "accumulate_n/rank1", "varShapes":[[3], [3], [3]], "varTypes":["float32", "float32", "float32"], "varInit":["uniform", "uniform", "uniform"]},
        # {"opName": "accumulate_n", "outName": "accumulate_n/rank2", "varShapes":[[2,3], [2,3], [2,3]], "varTypes":["float32", "float32", "float32"], "varInit":["uniform", "uniform", "uniform"]},
        # {"opName": "accumulate_n", "outName": "accumulate_n/rank3", "varShapes":[[2,3,4], [2,3,4], [2,3,4]], "varTypes":["float32", "float32", "float32"], "varInit":["uniform", "uniform", "uniform"]},
        #{"opName": "angle", "outName": "angle_scalar", "varShapes":[[]], "varTypes":["float32"], "varInit":["uniform"]},
        #{"opName": "angle", "outName": "angle_rank1", "varShapes":[[5]], "varTypes":["float32"], "varInit":["uniform"]},
        #{"opName": "angle", "outName": "angle_rank2", "varShapes":[[3,4]], "varTypes":["float32"], "varInit":["uniform"]},
        #TODO how to create ApproximateEqual class??
        # {"opName": "approximate_equal", "outName": "approximate_equal_scalar", "varShapes":[[],[]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "tolerance":0.1},
        # {"opName": "matmul", "outName": "matmul/rank2", "varShapes":[[3,4],[4,5]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "transpose_a":False, "transpose_b":False},
        # {"opName": "matmul", "outName": "matmul/emptyArrayTest/rank2", "varShapes":[[0,4],[4,0]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"], "transpose_a":False, "transpose_b":False},
        # {"opName": "matmul", "outName": "matmul/rank2_ta", "varShapes":[[4,3],[4,5]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "transpose_a":True, "transpose_b":False},
        # {"opName": "matmul", "outName": "matmul/rank2_tb", "varShapes":[[3,4],[5,4]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "transpose_a":False, "transpose_b":True},
        # {"opName": "matmul", "outName": "matmul/rank3_batch1", "varShapes":[[1,3,4],[1,4,5]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"]},
        # {"opName": "matmul", "outName": "matmul/rank3_batch2", "varShapes":[[2,3,4],[2,4,5]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"]},
        # {"opName": "matmul", "outName": "matmul/rank3_batch2_ta", "varShapes":[[2,4,3],[2,4,5]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "transpose_a":True, "transpose_b":False},
        # {"opName": "matmul", "outName": "matmul/rank3_batch2_tb", "varShapes":[[2,3,4],[2,5,4]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "transpose_a":False, "transpose_b":True},
        # {"opName": "matmul", "outName": "matmul/rank3_batch2_ta_tb", "varShapes":[[2,4,3],[2,5,4]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "transpose_a":True, "transpose_b":True},
        # {"opName": "matmul", "outName": "matmul/rank3_batch3", "varShapes":[[3,3,4],[3,4,5]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"]},
        # {"opName": "matmul", "outName": "matmul/rank4_batch2,2", "varShapes":[[2,2,3,4],[2,2,4,5]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"]},
        # {"opName": "matmul", "outName": "matmul/rank4_batch2,2_ta", "varShapes":[[2,2,4,3],[2,2,4,5]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "transpose_a":True, "transpose_b":False},
        # {"opName": "matmul", "outName": "matmul/rank4_batch2,2_tb", "varShapes":[[2,2,3,4],[2,2,5,4]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "transpose_a":False, "transpose_b":True},
        # {"opName": "matmul", "outName": "matmul/rank4_batch2,2_ta_tb", "varShapes":[[2,2,4,3],[2,2,5,4]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "transpose_a":True, "transpose_b":True},
        # {"opName": "matmul", "outName": "matmul/emptyArrayTest/rank4", "varShapes":[[2,2,4,0],[2,2,0,4]], "varTypes":["float64", "float64"], "varInit":["empty", "empty"], "transpose_a":True, "transpose_b":True},
        # {"opName": "matmul", "outName": "matmul/rank5_batch2,2,2", "varShapes":[[2,2,2,3,4],[2,2,2,4,5]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "transpose_a":False, "transpose_b":False},
        # {"opName": "matmul", "outName": "matmul/rank5_batch2,2,2_ta_tb", "varShapes":[[2,2,2,4,3],[2,2,2,5,4]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "transpose_a":True, "transpose_b":True},
        # {"opName": "matrix_diag_part", "outName": "matrix_diag_part/rank2", "varShapes":[[4,4]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName": "matrix_diag_part", "outName": "matrix_diag_part/rank3", "varShapes":[[3,4,4]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName": "matrix_diag_part", "outName": "matrix_diag_part/rank4", "varShapes":[[2,2,4,4]], "varTypes":["float32"], "varInit":["uniform"]},
        #{"opName": "svd", "outName": "svd/rank2_3,3_noFull_noUv", "varShapes":[[3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":False},
        #{"opName": "svd", "outName": "svd/rank2_3,3_full_noUv", "varShapes":[[3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        #{"opName": "svd", "outName": "svd/rank2_3,3_noFull_uv", "varShapes":[[3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":True},
        #{"opName": "svd", "outName": "svd/rank2_3,3_full_uv", "varShapes":[[3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        #{"opName": "svd", "outName": "svd/rank2_4,3_noFull_noUv", "varShapes":[[4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":False},
        #{"opName": "svd", "outName": "svd/rank2_4,3_full_noUv", "varShapes":[[4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        #{"opName": "svd", "outName": "svd/rank2_4,3_noFull_uv", "varShapes":[[4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":True},
        #{"opName": "svd", "outName": "svd/rank2_4,3_full_uv", "varShapes":[[4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        #{"opName": "svd", "outName": "svd/rank3_2,3,3_noFull_noUv", "varShapes":[[2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":False},
        #{"opName": "svd", "outName": "svd/rank3_2,3,3_full_noUv", "varShapes":[[2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        #{"opName": "svd", "outName": "svd/rank3_2,3,3_noFull_uv", "varShapes":[[2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":True},
        #{"opName": "svd", "outName": "svd/rank3_2,3,3_full_uv", "varShapes":[[2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        #{"opName": "svd", "outName": "svd/rank3_2,4,3_noFull_noUv", "varShapes":[[2,4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":False},
        #{"opName": "svd", "outName": "svd/rank3_2,4,3_full_noUv", "varShapes":[[2,4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        #{"opName": "svd", "outName": "svd/rank3_2,4,3_noFull_uv", "varShapes":[[2,4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":True},
        #{"opName": "svd", "outName": "svd/rank3_2,4,3_full_uv", "varShapes":[[2,4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        #{"opName": "svd", "outName": "svd/rank4_2,2,3,3_noFull_noUv", "varShapes":[[2,2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":False},
        #{"opName": "svd", "outName": "svd/rank4_2,2,3,3_full_noUv", "varShapes":[[2,2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        #{"opName": "svd", "outName": "svd/rank4_2,2,3,3_noFull_uv", "varShapes":[[2,2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":True},
        #{"opName": "svd", "outName": "svd/rank4_2,2,3,3_full_uv", "varShapes":[[2,2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},

        # {"opName": "pow", "outName": "pow/rank0", "varShapes": [[], []],"varTypes": ["float32", "float32"], "varInit": ["uniform", "uniform"]},
        # {"opName": "pow", "outName": "pow/rank1", "varShapes":[[2], [2]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"]},
        # {"opName": "pow", "outName": "pow/rank2", "varShapes":[[3,2], [3,2]], "varTypes":["float64", "float64"], "varInit":["uniform", "uniform"]},
        # {"opName": "pow", "outName": "pow/rank2bc0", "varShapes":[[2,5], []], "varTypes":["float64", "float64"], "varInit":["uniform", "uniform"]},
        # {"opName": "pow", "outName": "pow/rank2bc", "varShapes":[[2,5], [2, 1]], "varTypes":["float64", "float64"], "varInit":["uniform", "uniform"]},

        # {"opName": "mean_squared_error", "outName": "losses/mse_rank0", "varShapes":[[],[]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "mean_squared_error", "outName": "losses/mse_rank1", "varShapes":[[5],[5]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "mean_squared_error", "outName": "losses/mse_rank2", "varShapes":[[3,4],[3,4]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "mean_squared_error", "outName": "losses/mse_rank3", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "mean_squared_error", "outName": "losses/mse_rank1_weights_1", "varShapes":[[5],[5],[]], "varTypes":["float32","float32", "float32"], "varInit":["uniform","uniform", "uniform"]},
        # {"opName": "mean_squared_error", "outName": "losses/mse_rank2_weights_1", "varShapes":[[3,4],[3,4],[]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "mean_squared_error", "outName": "losses/mse_rank2_weights_2", "varShapes":[[3,4],[3,4],[1,4]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "mean_squared_error", "outName": "losses/mse_rank2_weights_3", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "mean_squared_error", "outName": "losses/mse_rank3_weights_1", "varShapes":[[2,3,4],[2,3,4],[]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "mean_squared_error", "outName": "losses/mse_rank3_weights_2", "varShapes":[[2,3,4],[2,3,4],[1,1,4]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "mean_squared_error", "outName": "losses/mse_rank3_weights_3", "varShapes":[[2,3,4],[2,3,4],[2,1,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "mean_squared_error", "outName": "losses/mse_rank3_weights_4", "varShapes":[[2,3,4],[2,3,4],[2,3,4]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]}
        # {"opName": "absolute_difference", "outName": "losses/absdiff_rank0", "varShapes":[[],[]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "absolute_difference", "outName": "losses/absdiff_rank1", "varShapes":[[5],[5]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "absolute_difference", "outName": "losses/absdiff_rank2", "varShapes":[[3,4],[3,4]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "absolute_difference", "outName": "losses/absdiff_rank3", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "absolute_difference", "outName": "losses/absdiff_rank0_weights", "varShapes":[[],[],[]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "absolute_difference", "outName": "losses/absdiff_rank1_weights_1", "varShapes":[[5],[5],[]], "varTypes":["float32","float32", "float32"], "varInit":["uniform","uniform", "uniform"]},
        # {"opName": "absolute_difference", "outName": "losses/absdiff_rank2_weights_1", "varShapes":[[3,4],[3,4],[]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "absolute_difference", "outName": "losses/absdiff_rank2_weights_2", "varShapes":[[3,4],[3,4],[1,4]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "absolute_difference", "outName": "losses/absdiff_rank3_weights_1", "varShapes":[[2,3,4],[2,3,4],[]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "absolute_difference", "outName": "losses/absdiff_rank3_weights_2", "varShapes":[[2,3,4],[2,3,4],[2,1,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},

        #{"opName": "cosine_distance", "outName": "losses/cosine_diff_rank0", "varShapes":[[],[]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"], "axis":None},     #Cosine doesn't like rank 0 input, it seems...
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank1", "varShapes":[[5],[5]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"], "axis":0},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank2_axis0_SUM", "varShapes":[[3,4],[3,4],[1,4]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.SUM, "axis":0},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank2_axis1_NONE", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.NONE, "axis":1},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank2_axis1_SUM", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.SUM, "axis":1},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank2_axis1_MEAN", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.MEAN, "axis":1},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank2_axis1_SUM_OVER_BATCH_SIZE", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_OVER_BATCH_SIZE, "axis":1},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank2_axis1_SUM_BY_NONZERO_WEIGHTS", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS, "axis":1},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank3", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"], "axis":0},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank3_axis1", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"], "axis":1},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank3_axis2", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"], "axis":2},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank3_weights0", "varShapes":[[2,3,4],[2,3,4],[1,1,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"], "axis":0},  #Can't have weights [2,1,1]? Maybe weights must match post reduce...
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank3_weights1", "varShapes":[[2,3,4],[2,3,4],[1,3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"], "axis":0},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank3_weights2", "varShapes":[[2,3,4],[2,3,4],[1,1,4]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"], "axis":0},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank3_weightsAll", "varShapes":[[2,3,4],[2,3,4],[1,3,4]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"], "axis":0},

        #Hinge: need bernoulli (0 or 1) labels, and 0 centered predictions (<0 negative, > 0 positive)
        # {"opName": "hinge_loss", "outName": "losses/hinge_rank1", "varShapes":[[5],[5]], "varTypes":["float32","float32"], "varInit":["bernoulli","uniform_m1_1"]},
        # {"opName": "hinge_loss", "outName": "losses/hinge_rank2_NONE", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["bernoulli","uniform_m1_1","uniform_sparse"], "reduction":tf.losses.Reduction.NONE},
        # {"opName": "hinge_loss", "outName": "losses/hinge_rank2_SUM", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["bernoulli","uniform_m1_1","uniform_sparse"], "reduction":tf.losses.Reduction.SUM},
        # {"opName": "hinge_loss", "outName": "losses/hinge_rank2_MEAN", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["bernoulli","uniform_m1_1","uniform_sparse"], "reduction":tf.losses.Reduction.MEAN},
        # {"opName": "hinge_loss", "outName": "losses/hinge_rank2_SUM_OVER_BATCH_SIZE", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["bernoulli","uniform_m1_1","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_OVER_BATCH_SIZE},
        # {"opName": "hinge_loss", "outName": "losses/hinge_rank2_SUM_BY_NONZERO_WEIGHTS", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["bernoulli","uniform_m1_1","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS},
        # {"opName": "hinge_loss", "outName": "losses/hinge_rank3", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["bernoulli","uniform_m1_1"]},
        # {"opName": "hinge_loss", "outName": "losses/hinge_rank3_weights0", "varShapes":[[2,3,4],[2,3,4],[2,1,1]], "varTypes":["float32","float32","float32"], "varInit":["bernoulli","uniform_m1_1","uniform"]},
        # {"opName": "hinge_loss", "outName": "losses/hinge_rank3_weights1", "varShapes":[[2,3,4],[2,3,4],[1,3,1]], "varTypes":["float32","float32","float32"], "varInit":["bernoulli","uniform_m1_1","uniform"]},
        # {"opName": "hinge_loss", "outName": "losses/hinge_rank3_weights2", "varShapes":[[2,3,4],[2,3,4],[1,1,4]], "varTypes":["float32","float32","float32"], "varInit":["bernoulli","uniform_m1_1","uniform"]},
        # {"opName": "hinge_loss", "outName": "losses/hinge_rank3_weightsAll", "varShapes":[[2,3,4],[2,3,4],[2,3,4]], "varTypes":["float32","float32","float32"], "varInit":["bernoulli","uniform_m1_1","uniform"]},

        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank0", "varShapes":[[],[]], "varTypes":["float32","float32"], "varInit":["stdnormal","stdnormal"]},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank0", "varShapes":[[],[]], "varTypes":["float32","float32"], "varInit":["stdnormal","stdnormal"]},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank1", "varShapes":[[5],[5]], "varTypes":["float32","float32"], "varInit":["stdnormal","stdnormal"]},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank1_d05", "varShapes":[[5],[5]], "varTypes":["float32","float32"], "varInit":["stdnormal","stdnormal"],"delta":0.5},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank1_d2", "varShapes":[[5],[5]], "varTypes":["float32","float32"], "varInit":["stdnormal","stdnormal"],"delta":2.0},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank2_axis0_SUM", "varShapes":[[3,4],[3,4],[1,4]], "varTypes":["float32","float32","float32"], "varInit":["stdnormal","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank2_axis1_NONE", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["stdnormal","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.NONE},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank2_axis1_SUM", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["stdnormal","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank2_axis1_MEAN", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["stdnormal","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.MEAN},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank2_axis1_SUM_OVER_BATCH_SIZE", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["stdnormal","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_OVER_BATCH_SIZE},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank2_axis1_SUM_BY_NONZERO_WEIGHTS", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["stdnormal","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank3", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["stdnormal","stdnormal"]},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank3_axis1", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["stdnormal","stdnormal"]},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank3_axis2", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["stdnormal","stdnormal"]},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank3_weights0", "varShapes":[[2,3,4],[2,3,4],[1,1,1]], "varTypes":["float32","float32","float32"], "varInit":["stdnormal","stdnormal","uniform"]},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank3_weights1", "varShapes":[[2,3,4],[2,3,4],[1,3,1]], "varTypes":["float32","float32","float32"], "varInit":["stdnormal","stdnormal","uniform"]},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank3_weights2", "varShapes":[[2,3,4],[2,3,4],[1,1,4]], "varTypes":["float32","float32","float32"], "varInit":["stdnormal","stdnormal","uniform"]},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank3_weightsAll", "varShapes":[[2,3,4],[2,3,4],[1,3,4]], "varTypes":["float32","float32","float32"], "varInit":["stdnormal","stdnormal","uniform"]},

        # {"opName": "log_loss", "outName": "losses/log_loss_rank0", "varShapes":[[],[]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank0", "varShapes":[[],[]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank1", "varShapes":[[5],[5]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank1_eps01", "varShapes":[[5],[5]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"],"epsilon":0.1},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank2_axis0_SUM", "varShapes":[[3,4],[3,4],[1,4]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.SUM},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank2_axis1_NONE", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.NONE},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank2_axis1_SUM", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.SUM},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank2_axis1_MEAN", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.MEAN},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank2_axis1_SUM_OVER_BATCH_SIZE", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_OVER_BATCH_SIZE},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank2_axis1_SUM_BY_NONZERO_WEIGHTS", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank3", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank3_axis1", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank3_axis2", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank3_weights0", "varShapes":[[2,3,4],[2,3,4],[1,1,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank3_weights1", "varShapes":[[2,3,4],[2,3,4],[1,3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank3_weights2", "varShapes":[[2,3,4],[2,3,4],[1,1,4]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank3_weightsAll", "varShapes":[[2,3,4],[2,3,4],[1,3,4]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},

        #sigmoid_cross_entropy: seems to only support [batch_size, num_classes] shapes?
        # {"opName": "sigmoid_cross_entropy", "outName": "losses/sigmoid_ce", "varShapes":[[3,4],[3,4]], "varTypes":["float32","float32"], "varInit":["uniform","stdnormal"]},
        # {"opName": "sigmoid_cross_entropy", "outName": "losses/sigmoid_ce_smooth01", "varShapes":[[3,4],[3,4]], "varTypes":["float32","float32"], "varInit":["uniform","stdnormal"],"label_smoothing":0.1},
        # {"opName": "sigmoid_cross_entropy", "outName": "losses/sigmoid_ce_smooth05", "varShapes":[[3,4],[3,4]], "varTypes":["float32","float32"], "varInit":["uniform","stdnormal"],"label_smoothing":0.5},
        # {"opName": "sigmoid_cross_entropy", "outName": "losses/sigmoid_ce_NONE", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.NONE},
        # {"opName": "sigmoid_cross_entropy", "outName": "losses/sigmoid_ce_SUM", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM},
        # {"opName": "sigmoid_cross_entropy", "outName": "losses/sigmoid_ce_MEAN", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.MEAN},
        # {"opName": "sigmoid_cross_entropy", "outName": "losses/sigmoid_ce_SUM_OVER_BATCH_SIZE", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_OVER_BATCH_SIZE},
        # {"opName": "sigmoid_cross_entropy", "outName": "losses/sigmoid_ce_SUM_BY_NONZERO_WEIGHTS", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS},

        # {"opName": "softmax_cross_entropy", "outName": "losses/softmax_ce", "varShapes":[[10,4],[10,4]], "varTypes":["float32","float32"], "varInit":["onehot","stdnormal"]},
        # {"opName": "softmax_cross_entropy", "outName": "losses/softmax_ce_smooth01", "varShapes":[[10,4],[10,4]], "varTypes":["float32","float32"], "varInit":["onehot","stdnormal"],"label_smoothing":0.1},
        # {"opName": "softmax_cross_entropy", "outName": "losses/softmax_ce_smooth05", "varShapes":[[10,4],[10,4]], "varTypes":["float32","float32"], "varInit":["onehot","stdnormal"],"label_smoothing":0.5},
        # {"opName": "softmax_cross_entropy", "outName": "losses/softmax_ce_NONE", "varShapes":[[10,4],[10,4],[10]], "varTypes":["float32","float32","float32"], "varInit":["onehot","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.NONE},
        # {"opName": "softmax_cross_entropy", "outName": "losses/softmax_ce_SUM", "varShapes":[[10,4],[10,4],[10]], "varTypes":["float32","float32","float32"], "varInit":["onehot","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM},
        # {"opName": "softmax_cross_entropy", "outName": "losses/softmax_ce_MEAN", "varShapes":[[10,4],[10,4],[10]], "varTypes":["float32","float32","float32"], "varInit":["onehot","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.MEAN},
        # {"opName": "softmax_cross_entropy", "outName": "losses/softmax_ce_SUM_OVER_BATCH_SIZE", "varShapes":[[10,4],[10,4],[10]], "varTypes":["float32","float32","float32"], "varInit":["onehot","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_OVER_BATCH_SIZE},
        # {"opName": "softmax_cross_entropy", "outName": "losses/softmax_ce_SUM_BY_NONZERO_WEIGHTS", "varShapes":[[10,4],[10,4],[10]], "varTypes":["float32","float32","float32"], "varInit":["onehot","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS},

        # {"opName": "sparse_softmax_cross_entropy", "outName": "losses/sparse_softmax_ce", "varShapes":[[10],[10,5]], "varTypes":["int32","float32"], "varInit":["uniform_int5","stdnormal"]},
        # {"opName": "sparse_softmax_cross_entropy", "outName": "losses/sparse_softmax_ce_NONE", "varShapes":[[10],[10,5],[10,1]], "varTypes":["int32","float32","float32"], "varInit":["uniform_int5","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.NONE},
        # {"opName": "sparse_softmax_cross_entropy", "outName": "losses/sparse_softmax_ce_SUM", "varShapes":[[10],[10,5],[10,1]], "varTypes":["int32","float32","float32"], "varInit":["uniform_int5","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM},
        # {"opName": "sparse_softmax_cross_entropy", "outName": "losses/sparse_softmax_ce_MEAN", "varShapes":[[10],[10,5],[10,1]], "varTypes":["int32","float32","float32"], "varInit":["uniform_int5","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.MEAN},
        # {"opName": "sparse_softmax_cross_entropy", "outName": "losses/sparse_softmax_ce_SUM_OVER_BATCH_SIZE", "varShapes":[[10],[10,5],[10]], "varTypes":["int32","float32","float32"], "varInit":["uniform_int5","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_OVER_BATCH_SIZE},
        # {"opName": "sparse_softmax_cross_entropy", "outName": "losses/sparse_softmax_ce_SUM_BY_NONZERO_WEIGHTS", "varShapes":[[10],[10,5],[10]], "varTypes":["int32","float32","float32"], "varInit":["uniform_int5","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS},
        # {"opName": "sparse_softmax_cross_entropy", "outName": "losses/sparse_softmax_ce_SUM_BY_NONZERO_WEIGHTS_1", "varShapes":[[10],[10,5],[1]], "varTypes":["int32","float32","float32"], "varInit":["uniform_int5","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS},

        # {"opName":"l2_loss", "outName":"losses/l2_loss_rank0", "varShapes":[[]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName":"l2_loss", "outName":"losses/l2_loss_rank1", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName":"l2_loss", "outName":"losses/l2_loss_rank2", "varShapes":[[3,4]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName":"l2_loss", "outName":"losses/l2_loss_rank3", "varShapes":[[2,3,4]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName":"l2_loss", "outName":"losses/l2_loss_rank4", "varShapes":[[2,3,4,5]], "varTypes":["float32"], "varInit":["uniform"]}

        #tf.nn.conv1d
        #CNN 1D layers: value, filters
        #value: NCW: [batch, channels, width], NWC: [batch, width, channels]
        #filters are [kernel, inChannels, outChannels] for both
        #Can't run the ncw tests: "UnimplementedError (see above for traceback): Generic conv implementation only supports NHWC tensor format for now" :/
        # {"opName":"nn_cnn1d", "outName":"cnn1d_nn/ncw_b1_k2_s1_SAME", "varShapes":[[1, 2, 5], [5, 2, 3]], "varTypes":["float32","float32"], "stride":1, "padding":"SAME", "data_format":"NCW"},
        # {"opName":"nn_cnn1d", "outName":"cnn1d_nn/ncw_b2_k2_s1_SAME", "varShapes":[[2, 2, 5], [5, 2, 3]], "varTypes":["float32","float32"], "stride":1, "padding":"SAME", "data_format":"NCW"},
        # {"opName":"nn_cnn1d", "outName":"cnn1d_nn/ncw_b2_k2_s1_VALID", "varShapes":[[2, 2, 5], [5, 2, 3]], "varTypes":["float32","float32"], "stride":1, "padding":"VALID", "data_format":"NCW"},
        # {"opName":"nn_cnn1d", "outName":"cnn1d_nn/ncw_b1_k2_s2_SAME", "varShapes":[[1, 2, 5], [5, 2, 3]], "varTypes":["float32","float32"], "stride":2, "padding":"SAME", "data_format":"NCW"},
        # {"opName":"nn_cnn1d", "outName":"cnn1d_nn/nwc_b1_k2_s1_SAME", "varShapes":[[1, 5, 2], [5, 2, 3]], "varTypes":["float32","float32"], "stride":1, "padding":"SAME", "data_format":"NWC"},
        # {"opName":"nn_cnn1d", "outName":"cnn1d_nn/nwc_b2_k2_s1_SAME", "varShapes":[[2, 5, 2], [5, 2, 3]], "varTypes":["float32","float32"], "stride":1, "padding":"SAME", "data_format":"NWC"},
        # {"opName":"nn_cnn1d", "outName":"cnn1d_nn/nwc_b2_k2_s1_VALID", "varShapes":[[2, 5, 2], [5, 2, 3]], "varTypes":["float32","float32"], "stride":1, "padding":"VALID", "data_format":"NWC"},
        # {"opName":"nn_cnn1d", "outName":"cnn1d_nn/nwc_b1_k2_s2_SAME", "varShapes":[[1, 5, 2], [5, 2, 3]], "varTypes":["float32","float32"], "stride":2, "padding":"SAME", "data_format":"NWC"},

        #tf.layers.conv1d
        #Note that the tf.layers version seems to add the variables directly - you don't provide the kernel params as a variable...
        #Also can't run channels_first here: "Generic conv implementation only supports NHWC tensor format for now." :/
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_first_b1_k2_s1_d1_SAME", "varShapes":[[1, 2, 5]], "varTypes":["float32","float32"], "filters":2, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_first", "dilation_rate":1},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_first_b1_k3_s1_d1_SAME", "varShapes":[[1, 2, 5]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_first", "dilation_rate":1},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_first_b2_k2_s1_d2_SAME", "varShapes":[[2, 2, 5]], "varTypes":["float32","float32"], "filters":4, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_first", "dilation_rate":2},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_first_b2_k2_s1_d1_VALID", "varShapes":[[2, 2, 5]], "varTypes":["float32","float32"], "filters":1, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_first", "dilation_rate":1},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_first_b1_k2_s2_d1_SAME", "varShapes":[[1, 2, 5]], "varTypes":["float32","float32"], "filters":2, "kernel_size":2, "strides":2, "padding":"SAME", "data_format":"channels_first", "dilation_rate":1},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b1_k2_s1_d1_SAME", "varShapes":[[1, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b1_k2_s1_d2_SAME", "varShapes":[[1, 5, 2]], "varTypes":["float32","float32"], "filters":2, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":2},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b2_k2_s1_SAME", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b2_k2_s1_VALID", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":2, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_last", "dilation_rate":1},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b1_k2_s2_SAME", "varShapes":[[1, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":2, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b1_k2_s1_d1_SAME_sigmoid", "varShapes":[[1, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "activation":tf.nn.relu},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b1_k2_s1_d1_SAME_sigmoid", "varShapes":[[1, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "activation":tf.nn.sigmoid},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b1_k2_s1_d1_SAME_elu", "varShapes":[[1, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "activation":tf.nn.elu},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b1_k2_s1_d1_SAME_relu6", "varShapes":[[1, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "activation":tf.nn.relu6},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b1_k2_s1_d1_SAME_relu6", "varShapes":[[1, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "activation":tf.nn.selu},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b1_k2_s1_d1_SAME_crelu", "varShapes":[[1, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "activation":tf.nn.crelu},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b2_k2_s1_SAME_regularizers", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1,
        #     "kernel_regularizer":tf.contrib.layers.l2_regularizer(scale=0.1), "bias_regularizer":tf.contrib.layers.l1_regularizer(scale=0.2), "activity_regularizer":tf.contrib.layers.l1_l2_regularizer(scale_l1=0.1,scale_l2=0.2)},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b2_k2_s1_SAME_constraints1", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1,
        #     "kernel_constraint":tf.keras.constraints.MaxNorm(max_value=2), "bias_constraint":tf.keras.constraints.NonNeg()},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b2_k2_s1_SAME_constraints2", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1,
        #     "kernel_constraint":tf.keras.constraints.MinMaxNorm(min_value=1, max_value=2)},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b2_k2_s1_SAME_constraints3", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1,
        #     "kernel_constraint":tf.keras.constraints.UnitNorm()},
        #TODO TF constraints don't appear to get saved with the model...

        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b1_k2_s1_d1_SAME_dm1", "varShapes":[[1, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":1},
        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b2_k2_s1_d1_SAME_dm2", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2},
        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b2_k2_s2_d1_SAME_dm2", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":2, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2},
        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b2_k2_s1_d2_SAME_dm2", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":2, "depth_multiplier":2},
        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b2_k2_s1_d1_SAME_dm1_sigm", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":1, "activation":tf.nn.tanh},
        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b2_k2_s1_d1_SAME_dm2_sigm_nobias", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2, "activation":tf.nn.tanh, "use_bias":False},
        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b1_k2_s1_d1_VALID_dm1", "varShapes":[[1, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":1},
        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b2_k2_s1_d1_VALID_dm2", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2},
        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b2_k2_s2_d1_VALID_dm2", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":2, "padding":"VALID", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2},
        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b2_k2_s1_d2_VALID_dm2", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_last", "dilation_rate":2, "depth_multiplier":2},
        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b2_k2_s1_d1_VALID_dm1_sigm", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":1, "activation":tf.nn.tanh},
        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b2_k2_s1_d1_VALID_dm2_sigm_nobias", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2, "activation":tf.nn.tanh, "use_bias":False},


        #Max/avg pool 1d:
        #channels_first:    "Default MaxPoolingOp only supports NHWC on device type CPU"        :/
        # {"opName":"max_pooling1d", "outName":"max_pooling1d/channels_first_b1_k2_s1_SAME", "varShapes":[[1, 2, 5]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_first"},
        # {"opName":"max_pooling1d", "outName":"max_pooling1d/channels_first_b1_k3_s1_SAME", "varShapes":[[1, 2, 5]], "varTypes":["float32"], "pooling_size":3, "stride":1, "padding":"SAME", "data_format":"channels_first"},
        # {"opName":"max_pooling1d", "outName":"max_pooling1d/channels_first_b2_k2_s1_SAME", "varShapes":[[2, 2, 5]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_first"},
        # {"opName":"max_pooling1d", "outName":"max_pooling1d/channels_first_b2_k2_s1_VALID", "varShapes":[[2, 2, 5]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"VALID", "data_format":"channels_first"},
        # {"opName":"max_pooling1d", "outName":"max_pooling1d/channels_first_b1_k2_s2_SAME", "varShapes":[[1, 2, 5]], "varTypes":["float32"], "pooling_size":2, "stride":2, "padding":"SAME", "data_format":"channels_first"},
        # {"opName":"max_pooling1d", "outName":"max_pooling1d/channels_last_b1_k2_s1_SAME", "varShapes":[[1, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_last"},
        # {"opName":"max_pooling1d", "outName":"max_pooling1d/channels_last_b2_k2_s1_SAME", "varShapes":[[2, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_last"},
        # {"opName":"max_pooling1d", "outName":"max_pooling1d/channels_last_b2_k2_s1_VALID", "varShapes":[[2, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"VALID", "data_format":"channels_last"},
        # {"opName":"max_pooling1d", "outName":"max_pooling1d/channels_last_b1_k2_s2_SAME", "varShapes":[[1, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":2, "padding":"SAME", "data_format":"channels_last"},

        #{"opName":"max_pool_with_argmax", "outName":"max_pool_with_argmax/float32_to_int32_nhwc", "varShapes":[[1, 2, 2, 5]], "varTypes":["int32"], "varInit":["uniform_int10"], "ksizes":[1,2,1,1],"strides":[1,1,2,1], "padding":"SAME", "data_format":"NHWC", "output_dtype":tf.int32},
        #{"opName": "max_pool_with_argmax", "outName": "max_pool_with_argmax/float32_to_int64", "varShapes": [[1, 2, 2, 5]], "varTypes": ["float32"], "varInit": ["uniform"], "ksizes": [1,2,1,1], "strides": [1,1,2,1], "padding": "SAME",  "data_format": "NHWC", "output_dtype": tf.int64},
        #{"opName": "max_pool_with_argmax", "outName": "max_pool_with_argmax/float64_to_int64", "varShapes": [[1, 2, 2, 5]], "varTypes": ["float64"], "varInit": ["uniform"], "ksizes": [1, 2,1,1], "strides": [1, 1,2,1], "padding": "VALID", "data_format": "NHWC", "output_dtype": tf.int64},
        #{"opName": "max_pool_with_argmax", "outName": "max_pool_with_argmax/int32_to_int64", "varShapes": [[1, 2, 2, 5]], "varTypes": ["int32"], "varInit": ["uniform"], "ksizes": [1, 2,1,1], "strides": [1, 1,2,1], "padding": "VALID","data_format": "NHWC", "output_dtype": tf.int64},
        #{"opName": "max_pool_with_argmax", "outName": "max_pool_with_argmax/int64_to_int32", "varShapes": [[1, 2, 2, 5]], "varTypes": ["int64"], "varInit": ["uniform"], "ksizes": [1, 2,1,1], "strides": [1, 1,2,1], "padding": "VALID", "data_format": "NHWC", "output_dtype": tf.int32},
        #{"opName": "max_pool_with_argmax", "outName": "max_pool_with_argmax/bfloat16_to_int32", "varShapes": [[1, 2, 2, 5]], "varTypes": ["bfloat16"], "varInit": ["uniform"], "ksizes": [1, 2,1,1], "strides": [1, 1,2,1], "padding": "VALID", "data_format": "NHWC", "output_dtype": tf.int32},
        #{"opName": "max_pool_with_argmax", "outName": "max_pool_with_argmax/half_to_int32", "varShapes": [[1, 2, 2, 5]], "varTypes": ["half"], "varInit": ["uniform"], "ksizes": [1, 2,1,1], "strides": [1, 1,1,2], "padding": "VALID","data_format": "NHWC", "output_dtype": tf.int32},

        #Default AvgPoolingOp only supports NHWC on device type CPU
        # {"opName":"avg_pooling1d", "outName":"avg_pooling1d/channels_first_b1_k2_s1_SAME", "varShapes":[[1, 2, 5]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_first"},
        # {"opName":"avg_pooling1d", "outName":"avg_pooling1d/channels_first_b1_k2_s1_SAME", "varShapes":[[1, 2, 5]], "varTypes":["float32"], "pooling_size":3, "stride":1, "padding":"SAME", "data_format":"channels_first"},
        # {"opName":"avg_pooling1d", "outName":"avg_pooling1d/channels_first_b2_k2_s1_SAME", "varShapes":[[2, 2, 5]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_first"},
        # {"opName":"avg_pooling1d", "outName":"avg_pooling1d/channels_first_b2_k2_s1_VALID", "varShapes":[[2, 2, 5]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"VALID", "data_format":"channels_first"},
        # {"opName":"avg_pooling1d", "outName":"avg_pooling1d/channels_first_b1_k2_s2_SAME", "varShapes":[[1, 2, 5]], "varTypes":["float32"], "pooling_size":2, "stride":2, "padding":"SAME", "data_format":"channels_first"},
        # {"opName":"avg_pooling1d", "outName":"avg_pooling1d/channels_last_b1_k2_s1_SAME", "varShapes":[[1, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_last"},
        # {"opName":"avg_pooling1d", "outName":"avg_pooling1d/channels_last_b2_k2_s1_SAME", "varShapes":[[2, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_last"},
        # {"opName":"avg_pooling1d", "outName":"avg_pooling1d/channels_last_b2_k2_s1_VALID", "varShapes":[[2, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"VALID", "data_format":"channels_last"},
        # {"opName":"avg_pooling1d", "outName":"avg_pooling1d/channels_last_b1_k2_s2_SAME", "varShapes":[[1, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":2, "padding":"SAME", "data_format":"channels_last"},

        # {"opName":"dense", "outName":"dense/dense5", "varShapes":[[5,4]], "varTypes":["float32"], "units":5, "activation":None, "use_bias":True, "kernel_regularizer":None, "bias_regularizer":None},
        # {"opName":"dense", "outName":"dense/dense5_sigmoid_nobias", "varShapes":[[5,4]], "varTypes":["float32"], "units":5, "activation":tf.nn.sigmoid, "use_bias":False, "kernel_regularizer":None, "bias_regularizer":None},
        # {"opName":"dense", "outName":"dense/dense5_tanh_regularizer", "varShapes":[[5,4]], "varTypes":["float32"], "units":5, "activation":tf.nn.tanh, "use_bias":True, "kernel_regularizer":tf.contrib.layers.l2_regularizer(scale=0.1), "bias_regularizer":None},
        # {"opName":"flatten", "outName":"flatten/rank2", "varShapes":[[3,4]], "varTypes":["float32"]},
        # {"opName":"flatten", "outName":"flatten/rank3", "varShapes":[[2,3,4]], "varTypes":["float32"]},
        # {"opName":"flatten", "outName":"flatten/rank4", "varShapes":[[2,3,2,4]], "varTypes":["float32"]},
        # {"opName":"flatten", "outName":"flatten/rank5", "varShapes":[[2,3,2,4,2]], "varTypes":["float32"]},

        # NHWC format: kernel format is [kH,kW,cIn,cOut]
        #Also, strides and dilation are 4d for some reason, strides should be [1, sH, sW, 1]
        # {"opName":"nn_conv2d", "outName":"cnn2d_nn/nhwc_b1_k2_s1_d1_SAME", "varShapes":[[1, 5, 5, 2], [2, 2, 2, 3]], "varTypes":["float32","float32"], "strides":[1,1,1,1], "padding":"SAME", "data_format":"NHWC"},
        # {"opName":"nn_conv2d", "outName":"cnn2d_nn/nhwc_b2_k3_s1_d1_SAME", "varShapes":[[2, 5, 5, 2], [3, 3, 2, 3]], "varTypes":["float32","float32"], "strides":[1,1,1,1], "padding":"SAME", "data_format":"NHWC"},
        # {"opName":"nn_conv2d", "outName":"cnn2d_nn/nhwc_b2_k2_s1_d1_SAME", "varShapes":[[2, 5, 5, 2], [2, 2, 2, 3]], "varTypes":["float32","float32"], "strides":[1,1,1,1], "padding":"SAME", "data_format":"NHWC"},
        # {"opName":"nn_conv2d", "outName":"cnn2d_nn/nhwc_b2_k2_s1_d2_SAME", "varShapes":[[2, 5, 5, 2], [2, 2, 2, 3]], "varTypes":["float32","float32"], "strides":[1,1,1,1], "padding":"SAME", "data_format":"NHWC", "dilation":[1,2,2,1]},
        # {"opName":"nn_conv2d", "outName":"cnn2d_nn/nhwc_b2_k2_s1_d1_VALID", "varShapes":[[2, 5, 5, 2], [2, 2, 2, 3]], "varTypes":["float32","float32"], "strides":[1,1,1,1], "padding":"VALID", "data_format":"NHWC"},
        # {"opName":"nn_conv2d", "outName":"cnn2d_nn/nhwc_b1_k2_s2_SAME", "varShapes":[[2, 5, 5, 2], [2, 2, 2, 3]], "varTypes":["float32","float32"], "strides":[1,2,2,1], "padding":"SAME", "data_format":"NHWC"},
        # {"opName":"nn_conv2d", "outName":"cnn2d_nn/nchw_b1_k2_s1_d1_SAME", "varShapes":[[1, 2, 5, 5], [2, 2, 2, 3]], "varTypes":["float32","float32"], "strides":[1,1,1,1], "padding":"SAME", "data_format":"NCHW"},
        # {"opName":"nn_conv2d", "outName":"cnn2d_nn/nchw_b2_k3_s1_d1_SAME", "varShapes":[[2, 2, 5, 5], [3, 3, 2, 3]], "varTypes":["float32","float32"], "strides":[1,1,1,1], "padding":"SAME", "data_format":"NCHW"},
        # {"opName":"nn_conv2d", "outName":"cnn2d_nn/nchw_b2_k2_s1_d1_SAME", "varShapes":[[2, 2, 5, 5], [2, 2, 2, 3]], "varTypes":["float32","float32"], "strides":[1,1,1,1], "padding":"SAME", "data_format":"NCHW"},
        # {"opName":"nn_conv2d", "outName":"cnn2d_nn/nchw_b2_k2_s1_d2_SAME", "varShapes":[[2, 2, 5, 5], [2, 2, 2, 3]], "varTypes":["float32","float32"], "strides":[1,1,1,1], "padding":"SAME", "data_format":"NCHW", "dilation":[1,1,2,2]},
        # {"opName":"nn_conv2d", "outName":"cnn2d_nn/nchw_b2_k2_s1_d1_VALID", "varShapes":[[2, 2, 5, 5], [2, 2, 2, 3]], "varTypes":["float32","float32"], "strides":[1,1,1,1], "padding":"VALID", "data_format":"NCHW"},
        # {"opName":"nn_conv2d", "outName":"cnn2d_nn/nchw_b1_k2_s2_SAME", "varShapes":[[2, 2, 5, 5], [2, 2, 2, 3]], "varTypes":["float32","float32"], "strides":[1,1,1,2], "padding":"SAME", "data_format":"NCHW"},

        # {"opName":"nn_conv2d", "outName":"cnn2d_nn/nhwc_35_35_32_b1_k3_s1_SAME", "varShapes":[[1, 35, 35, 3], [3, 3, 3, 192]], "varTypes":["float32","float32"], "strides":[1,1,1,1], "padding":"SAME", "data_format":"NHWC"},

        #Again, no channels_first on CPU
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b1_k2_s1_d1_SAME", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1]},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b1_k2_s1_d2_SAME", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":2, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[2,2]},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b2_k2_s1_SAME_nobias", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1], "use_bias":False},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b2_k2_s1_VALID", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":2, "kernel_size":[2,2], "strides":[1,1], "padding":"VALID", "data_format":"channels_last", "dilation_rate":[1,1]},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b1_k2_s2_SAME", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[2,2], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1]},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b1_k2_s1_d1_SAME_sigmoid", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1], "activation":tf.nn.relu},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b1_k2_s1_d1_SAME_sigmoid", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1], "activation":tf.nn.sigmoid},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b1_k2_s1_d1_SAME_elu", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1], "activation":tf.nn.elu},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b1_k2_s1_d1_SAME_relu6", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1], "activation":tf.nn.relu6},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b1_k2_s1_d1_SAME_selu_nobias", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1], "activation":tf.nn.selu, "use_bias":False},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b1_k2_s1_d1_SAME_crelu", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1], "activation":tf.nn.crelu},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b2_k2_s1_SAME_regularizers", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1],
        #     "kernel_regularizer":tf.contrib.layers.l2_regularizer(scale=0.1), "bias_regularizer":tf.contrib.layers.l1_regularizer(scale=0.2), "activity_regularizer":tf.contrib.layers.l1_l2_regularizer(scale_l1=0.1,scale_l2=0.2)},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b2_k2_s1_SAME_constraints1", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1],
        #     "kernel_constraint":tf.keras.constraints.MaxNorm(max_value=2), "bias_constraint":tf.keras.constraints.NonNeg()},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b2_k2_s1_SAME_constraints2", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1],
        #     "kernel_constraint":tf.keras.constraints.MinMaxNorm(min_value=1, max_value=2)},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b2_k2_s1_SAME_constraints3", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1],
        #     "kernel_constraint":tf.keras.constraints.UnitNorm()},

        # {"opName":"layers_conv2d_transpose", "outName":"conv2d_transpose/channels_last_b1_k2_s1_SAME", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":2, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last"},
        # {"opName":"layers_conv2d_transpose", "outName":"conv2d_transpose/channels_last_b2_k2_s1_SAME_nobias", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "use_bias":False},
        # {"opName":"layers_conv2d_transpose", "outName":"conv2d_transpose/channels_last_b2_k2_s1_VALID", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":2, "kernel_size":[2,2], "strides":[1,1], "padding":"VALID", "data_format":"channels_last"},
        # {"opName":"layers_conv2d_transpose", "outName":"conv2d_transpose/channels_last_b1_k2_s2_SAME", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[2,2], "padding":"SAME", "data_format":"channels_last"},
        # {"opName":"layers_conv2d_transpose", "outName":"conv2d_transpose/channels_last_b1_k2_s1_SAME_sigmoid", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "activation":tf.nn.relu},
        # {"opName":"layers_conv2d_transpose", "outName":"conv2d_transpose/channels_last_b1_k2_s1_SAME_sigmoid", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "activation":tf.nn.sigmoid},
        # {"opName":"layers_conv2d_transpose", "outName":"conv2d_transpose/channels_last_b1_k2_s1_SAME_elu", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "activation":tf.nn.elu},
        # {"opName":"layers_conv2d_transpose", "outName":"conv2d_transpose/channels_last_b1_k2_s1_SAME_relu6", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "activation":tf.nn.relu6},
        # {"opName":"layers_conv2d_transpose", "outName":"conv2d_transpose/channels_last_b1_k2_s1_SAME_selu_nobias", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "activation":tf.nn.selu, "use_bias":False},
        # {"opName":"layers_conv2d_transpose", "outName":"conv2d_transpose/channels_last_b1_k2_s1_SAME_crelu", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "activation":tf.nn.crelu},
        # {"opName":"layers_conv2d_transpose", "outName":"conv2d_transpose/channels_last_b2_k2_s1_SAME_regularizers", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last",
        #  "kernel_regularizer":tf.contrib.layers.l2_regularizer(scale=0.1), "bias_regularizer":tf.contrib.layers.l1_regularizer(scale=0.2), "activity_regularizer":tf.contrib.layers.l1_l2_regularizer(scale_l1=0.1,scale_l2=0.2)},
        # {"opName":"Conv2DTranspose", "outName":"Conv2DTranspose", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "kernel_size":[2,2], "filters":2},

        # Data format: ch_last: NDHWC, ch_first: NCDHW
        # "CPU implementation of Conv3D currently only supports the NHWC tensor format."
        #{"opName":"layers_conv3d", "outName":"cnn3d_layers/channels_first_b1_k2_s1_d1_SAME", "varShapes":[[1, 2, 5, 5, 5]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2,2], "strides":[1,1,1], "padding":"SAME", "data_format":"channels_first", "dilation_rate":[1,1,1]},
        # {"opName":"layers_conv3d", "outName":"cnn3d_layers/channels_last_b1_k2_s1_d1_SAME", "varShapes":[[1, 5, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2,2], "strides":[1,1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1,1]},
        # {"opName":"layers_conv3d", "outName":"cnn3d_layers/channels_last_b1_k2_s1_d2_SAME", "varShapes":[[1, 5, 5, 5, 2]], "varTypes":["float32","float32"], "filters":2, "kernel_size":[2,2,2], "strides":[1,1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[2,2,2]},
        # {"opName":"layers_conv3d", "outName":"cnn3d_layers/channels_last_b2_k3_s1_SAME_nobias", "varShapes":[[2, 5, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[3,3,3], "strides":[1,1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1,1], "use_bias":False},
        # {"opName":"layers_conv3d", "outName":"cnn3d_layers/channels_last_b2_k2_s1_VALID", "varShapes":[[2, 5, 5, 5, 2]], "varTypes":["float32","float32"], "filters":2, "kernel_size":[2,2,2], "strides":[1,1,1], "padding":"VALID", "data_format":"channels_last", "dilation_rate":[1,1,1]},
        # {"opName":"layers_conv3d", "outName":"cnn3d_layers/channels_last_b1_k2_s2_SAME", "varShapes":[[1, 5, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2,2], "strides":[2,2,2], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1,1]},
        # {"opName":"layers_conv3d", "outName":"cnn3d_layers/channels_last_b1_k2_s1_d1_SAME_sigmoid", "varShapes":[[1, 5, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2,2], "strides":[1,1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1,1], "activation":tf.nn.relu},
        # {"opName":"layers_conv3d", "outName":"cnn3d_layers/channels_last_b2_k2_s1_SAME_regularizers", "varShapes":[[2, 5, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2,2], "strides":[1,1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1,1],
        #     "kernel_regularizer":tf.contrib.layers.l2_regularizer(scale=0.1), "bias_regularizer":tf.contrib.layers.l1_regularizer(scale=0.2), "activity_regularizer":tf.contrib.layers.l1_l2_regularizer(scale_l1=0.1,scale_l2=0.2)},
        # {"opName":"layers_conv3d", "outName":"cnn3d_layers/channels_last_b2_k2_s1_SAME_constraints1", "varShapes":[[2, 5, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2,2], "strides":[1,1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1,1],
        #     "kernel_constraint":tf.keras.constraints.MaxNorm(max_value=2), "bias_constraint":tf.keras.constraints.NonNeg()},

        # Max/avg pool 3d:
        # channels_first:    "Default MaxPoolingOp only supports NHWC on device type CPU"        :/
        # {"opName":"max_pooling3d", "outName":"max_pooling3d/channels_first_b1_k2_s1_SAME", "varShapes":[[1, 2, 5, 5, 5]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_first"},
        # {"opName":"max_pooling3d", "outName":"max_pooling3d/channels_last_b1_k2_s1_SAME", "varShapes":[[1, 5, 5, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_last"},
        # {"opName":"max_pooling3d", "outName":"max_pooling3d/channels_last_b2_k2_s1_SAME", "varShapes":[[2, 5, 5, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_last"},
        # {"opName":"max_pooling3d", "outName":"max_pooling3d/channels_last_b2_k2_s1_VALID", "varShapes":[[2, 5, 5, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"VALID", "data_format":"channels_last"},
        # {"opName":"max_pooling3d", "outName":"max_pooling3d/channels_last_b1_k2_s2_SAME", "varShapes":[[1, 5, 5, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":2, "padding":"SAME", "data_format":"channels_last"},
        #
        # #Default AvgPoolingOp only supports NHWC on device type CPU
        # {"opName":"avg_pooling3d", "outName":"avg_pooling3d/channels_first_b1_k2_s1_SAME", "varShapes":[[1, 2, 5, 5, 5]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_first"},
        # {"opName":"avg_pooling3d", "outName":"avg_pooling3d/channels_last_b1_k2_s1_SAME", "varShapes":[[1, 5, 5, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_last"},
        # {"opName":"avg_pooling3d", "outName":"avg_pooling3d/channels_last_b2_k2_s1_SAME", "varShapes":[[2, 5, 5, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_last"},
        # {"opName":"avg_pooling3d", "outName":"avg_pooling3d/channels_last_b2_k2_s1_VALID", "varShapes":[[2, 5, 5, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"VALID", "data_format":"channels_last"},
        # {"opName":"avg_pooling3d", "outName":"avg_pooling3d/channels_last_b1_k2_s2_SAME", "varShapes":[[1, 5, 5, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":2, "padding":"SAME", "data_format":"channels_last"},

        # {"opName":"conv3d_transpose_layers", "outName":"conv3d_transpose_layers/NDHWC_b1_k2_s1_SAME", "varShapes":[[1, 5, 5, 5, 2]], "varTypes":["float32","float32"], "filters":2, "kernel_size":[2,2,2], "strides":[1,1,1], "padding":"SAME", "data_format":"channels_last"},
        # {"opName":"conv3d_transpose_layers", "outName":"conv3d_transpose_layers/NCDHW_b2_k2_s1_SAME_nobias", "varShapes":[[2, 2, 5, 5, 5]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2,2], "strides":[1,1,1], "padding":"SAME", "data_format":"channels_first", "use_bias":False},
        # {"opName":"conv3d_transpose_layers", "outName":"conv3d_transpose_layers/NDHWC_b2_k2_s1_VALID", "varShapes":[[2, 5, 5, 5, 2]], "varTypes":["float32","float32"], "filters":2, "kernel_size":[2,2,2], "strides":[1,1,1], "padding":"VALID", "data_format":"channels_last"},
        # {"opName":"conv3d_transpose_layers", "outName":"conv3d_transpose_layers/NDHWC_b1_k2_s2_SAME", "varShapes":[[1, 5, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2,2], "strides":[2,2,2], "padding":"SAME", "data_format":"channels_last"},
        # {"opName":"conv3d_transpose_layers", "outName":"conv3d_transpose_layers/NDHWC_b1_k2_s1_SAME_sigmoid", "varShapes":[[1, 5, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2,2], "strides":[1,1,1], "padding":"SAME", "data_format":"channels_last", "activation":tf.nn.relu},
        # {"opName":"conv3d_transpose_layers", "outName":"conv3d_transpose_layers/NCDHW_b1_k2_s2_SAME_sigmoid", "varShapes":[[1, 2, 6, 6, 6,]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2,2], "strides":[2,2,2], "padding":"SAME", "data_format":"channels_first", "activation":tf.nn.sigmoid},
        # {"opName":"conv3d_transpose_layers", "outName":"conv3d_transpose_layers/NDHWC_b1_k2_s1_SAME_elu", "varShapes":[[1, 5, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2,2], "strides":[1,1,1], "padding":"SAME", "data_format":"channels_first", "activation":tf.nn.elu},
        # {"opName":"conv3d_transpose_layers", "outName":"conv3d_transpose_layers/NCDHW_b1_k3_s2_SAME_relu6", "varShapes":[[1, 2, 6, 6, 6]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[3,3,3], "strides":[2,2,2], "padding":"SAME", "data_format":"channels_last", "activation":tf.nn.relu6},
        # {"opName":"conv3d_transpose_layers", "outName":"conv3d_transpose_layers/NDHWC_b1_k2_s1_SAME_selu_nobias", "varShapes":[[1, 5, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2,2], "strides":[1,1,1], "padding":"SAME", "data_format":"channels_last", "activation":tf.nn.selu, "use_bias":False},
        # {"opName":"conv3d_transpose_layers", "outName":"conv3d_transpose_layers/NCDHW_b1_k2_s1_VALID_crelu", "varShapes":[[1, 2, 6, 6, 6]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2,2], "strides":[1,1,1], "padding":"VALID", "data_format":"channels_first", "activation":tf.nn.crelu},
        # {"opName":"conv3d_transpose_layers", "outName":"conv3d_transpose_layers/NDHWC_b2_k2_s1_SAME_regularizers", "varShapes":[[2, 5, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2,2], "strides":[1,1,1], "padding":"SAME", "data_format":"channels_last",
        #  "kernel_regularizer":tf.contrib.layers.l2_regularizer(scale=0.1), "bias_regularizer":tf.contrib.layers.l1_regularizer(scale=0.2), "activity_regularizer":tf.contrib.layers.l1_l2_regularizer(scale_l1=0.1,scale_l2=0.2)},

        # variables: value (NCDHW / NDHWC) and filter ([kD, kH, kW, oC, iC])
        # {"opName":"conv3d_transpose_nn", "outName":"conv3d_transpose_nn/NDHWC_b1_k2_s1_d1_SAME", "varShapes":[[1, 3, 3, 3, 2], [2, 2, 2, 3, 2]], "varTypes":["float32","float32"], "output_shape":[1,3,3,3,3], "strides":[1,1,1,1,1], "padding":"SAME", "data_format":"NDHWC", "dilations":1},
        # # Next 2: "Operation received an exception:Status: 3, message: could not create a convolution backward data descriptor, in file tensorflow/core/kernels/mkl_conv_grad_input_ops.cc:456"
        # # {"opName":"conv3d_transpose_nn", "outName":"conv3d_transpose_nn/NCDHW_b2_k2_s2_d1_VALID_out8", "varShapes":[[2, 2, 4, 4, 4], [2, 2, 2, 3, 2]], "varTypes":["float32","float32"], "output_shape":[1,3,8,8,8], "strides":[1,1,2,2,2], "padding":"VALID", "data_format":"NCDHW", "dilations":1},
        # # {"opName":"conv3d_transpose_nn", "outName":"conv3d_transpose_nn/NDHWC_b2_k2_s2_d1_VALID_out9", "varShapes":[[2, 4, 4, 4, 2], [2, 2, 2, 3, 2]], "varTypes":["float32","float32"], "output_shape":[1,9,9,9,3], "strides":[1,2,2,2,1], "padding":"VALID", "data_format":"NDHWC", "dilations":1},
        # {"opName":"conv3d_transpose_nn", "outName":"conv3d_transpose_nn/NDHWC_b1_k2_s2_d1_SAME", "varShapes":[[1, 2, 2, 2, 2], [2, 2, 2, 3, 2]], "varTypes":["float32","float32"], "output_shape":[1,4,4,4,3], "strides":[1,2,2,2,1], "padding":"SAME", "data_format":"NDHWC", "dilations":[1,1,1,1,1]},
        # {"opName":"conv3d_transpose_nn", "outName":"conv3d_transpose_nn/NCDHW_b1_k2_s2_d1_SAME_out6", "varShapes":[[1, 2, 3, 3, 3,], [2, 2, 2, 3, 2]], "varTypes":["float32","float32"], "output_shape":[1,3,6,6,6], "strides":[1,1,2,2,2], "padding":"SAME", "data_format":"NCDHW", "dilations":[1,1,1]},
        # {"opName":"conv3d_transpose_nn", "outName":"conv3d_transpose_nn/NCDHW_b1_k2_s2_d1_SAME_out5", "varShapes":[[1, 2, 3, 3, 3,], [2, 2, 2, 3, 2]], "varTypes":["float32","float32"], "output_shape":[1,3,5,5,5], "strides":[1,1,2,2,2], "padding":"SAME", "data_format":"NCDHW", "dilations":[1,1,1]},
        # {"opName":"conv3d_transpose_nn", "outName":"conv3d_transpose_nn/NCDHW_b1_k3_s3_d1_SAME", "varShapes":[[1, 2, 1, 1, 1], [3, 3, 3, 3, 2]], "varTypes":["float32","float32"], "output_shape":[1,3,3,3,3], "strides":[1,1,3,3,3], "padding":"SAME", "data_format":"NCDHW", "dilations":[1,1,1]},

        #Separable conv 2d - channels_last = NHWC
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b1_k2_s1_d1_SAME_dm1", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":1},
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b2_k2_s1_d1_SAME_dm2", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2},
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b2_k2_s2_d1_SAME_dm2", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":2, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2},
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b2_k2_s1_d2_SAME_dm2", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":2, "depth_multiplier":2},
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b2_k2_s1_d1_SAME_dm1_sigm", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":1, "activation":tf.nn.tanh},
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b2_k2_s1_d1_SAME_dm2_sigm_nobias", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2, "activation":tf.nn.tanh, "use_bias":False},
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b1_k2_s1_d1_VALID_dm1", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":1},
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b2_k2_s1_d1_VALID_dm2", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2},
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b2_k2_s2_d1_VALID_dm2", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":2, "padding":"VALID", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2},
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b2_k2_s1_d2_VALID_dm2", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_last", "dilation_rate":2, "depth_multiplier":2},
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b2_k2_s1_d1_VALID_dm1_sigm", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":1, "activation":tf.nn.tanh},
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b2_k2_s1_d1_VALID_dm2_sigm_nobias", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2, "activation":tf.nn.tanh, "use_bias":False},


        #Batch norm - 2d
        # {"opName":"batchnorm", "outName":"batchnorm/rank2_batch2_sz5_fused", "varShapes":[[2, 5]], "varTypes":["float32"], "axis":1, "fused":True},
        # {"opName":"batchnorm", "outName":"batchnorm/rank2_batch4_sz5_noFused", "varShapes":[[4, 5]], "varTypes":["float32"], "axis":1, "fused":False},
        # {"opName":"batchnorm", "outName":"batchnorm/rank2_batch3_sz5_fused_m50_e01", "varShapes":[[3, 5]], "varTypes":["float32"], "axis":1, "fused":True, "momentum":0.5, "epsilon":0.1},
        # {"opName":"batchnorm", "outName":"batchnorm/rank2_batch4_sz5_noFused_m50e01", "varShapes":[[4, 5]], "varTypes":["float32"], "axis":1, "fused":False, "momentum":0.5, "epsilon":0.1},
        #
        # #Batch norm - 3d input (time series) - NCW = [mb, size, length]
        # {"opName":"batchnorm", "outName":"batchnorm/rank3_ncw_batch2_sz5_fused", "varShapes":[[2, 5, 5]], "varTypes":["float32"], "axis":1, "fused":True},
        # {"opName":"batchnorm", "outName":"batchnorm/rank3_ncw_batch4_sz5_noFused", "varShapes":[[4, 5, 5]], "varTypes":["float32"], "axis":1, "fused":False},
        # {"opName":"batchnorm", "outName":"batchnorm/rank3_ncw_batch3_sz5_fused_m50_e01", "varShapes":[[3, 5, 3]], "varTypes":["float32"], "axis":1, "fused":True, "momentum":0.5, "epsilon":0.1},
        # {"opName":"batchnorm", "outName":"batchnorm/rank3_ncw_batch4_sz5_noFused_m50e01", "varShapes":[[4, 5, 1]], "varTypes":["float32"], "axis":1, "fused":False, "momentum":0.5, "epsilon":0.1},
        # {"opName":"batchnorm", "outName":"batchnorm/rank3_nwc_batch2_sz5_fused", "varShapes":[[2, 5, 5]], "varTypes":["float32"], "axis":2, "fused":True},
        # {"opName":"batchnorm", "outName":"batchnorm/rank3_nwc_batch4_sz5_noFused", "varShapes":[[4, 5, 5]], "varTypes":["float32"], "axis":2, "fused":False},
        # {"opName":"batchnorm", "outName":"batchnorm/rank3_nwc_batch3_sz5_fused_m50_e01", "varShapes":[[3, 5, 3]], "varTypes":["float32"], "axis":2, "fused":True, "momentum":0.5, "epsilon":0.1},
        # {"opName":"batchnorm", "outName":"batchnorm/rank3_nwc_batch4_sz5_noFused_m50e01", "varShapes":[[4, 5, 1]], "varTypes":["float32"], "axis":2, "fused":False, "momentum":0.5, "epsilon":0.1},

        #Batch norm - 4d input (2d CNN)
        #Can't do fused + nchw(axis=1)
        # # {"opName":"batchnorm", "outName":"batchnorm/rank4_nchw_batch2_sz5_fused", "varShapes":[[2, 5, 5, 3]], "varTypes":["float32"], "axis":1, "fused":True},
        # {"opName":"batchnorm", "outName":"batchnorm/rank4_nchw_batch4_sz5_noFused", "varShapes":[[4, 5, 5, 3]], "varTypes":["float32"], "axis":1, "fused":False},
        # # {"opName":"batchnorm", "outName":"batchnorm/rank4_nchw_batch3_sz5_fused_m50_e01", "varShapes":[[3, 5, 3, 5]], "varTypes":["float32"], "axis":1, "fused":True, "momentum":0.5, "epsilon":0.1},
        # {"opName":"batchnorm", "outName":"batchnorm/rank4_nchw_batch4_sz5_noFused_m50e01", "varShapes":[[4, 5, 5, 1]], "varTypes":["float32"], "axis":1, "fused":False, "momentum":0.5, "epsilon":0.1},
        # {"opName":"batchnorm", "outName":"batchnorm/rank4_nhwc_batch2_sz5_fused", "varShapes":[[2, 5, 5, 5]], "varTypes":["float32"], "axis":3, "fused":True},
        # {"opName":"batchnorm", "outName":"batchnorm/rank4_nhwc_batch4_sz5_noFused", "varShapes":[[4, 5, 5, 5]], "varTypes":["float32"], "axis":3, "fused":False},
        # {"opName":"batchnorm", "outName":"batchnorm/rank4_nhwc_batch3_sz5_fused_m50_e01", "varShapes":[[3, 5, 5, 3]], "varTypes":["float32"], "axis":3, "fused":True, "momentum":0.5, "epsilon":0.1},
        # {"opName":"batchnorm", "outName":"batchnorm/rank4_nhwc_batch4_sz5_noFused_m50e01", "varShapes":[[4, 5, 5, 1]], "varTypes":["float32"], "axis":3, "fused":False, "momentum":0.5, "epsilon":0.1},

        #Leaky RELU
        #{"opName":"leaky_relu", "outName":"leaky_relu/rank2_a0", "varShapes":[[4, 5]], "varTypes":["float32"], "varInit":["stdnormal"], "alpha":0.0},
        #{"opName":"leaky_relu", "outName":"leaky_relu/rank4_a05", "varShapes":[[4, 5]], "varTypes":["float32"], "varInit":["stdnormal"], "alpha":0.5},
        #{"opName":"leaky_relu", "outName":"leaky_relu/rank4_a0", "varShapes":[[4, 5, 5, 1]], "varTypes":["float32"], "varInit":["stdnormal"], "alpha":0.0},
        #{"opName":"leaky_relu", "outName":"leaky_relu/rank4_a02", "varShapes":[[4, 5, 5, 1]], "varTypes":["float32"], "varInit":["stdnormal"], "alpha":0.2},

        #Embedding lookup
        # {"opName":"embedding_lookup", "outName":"embedding_lookup/rank2_single_div_nomaxnorm", "varShapes":[[10, 5],[4]], "varTypes":["float32","int32"], "varInit":["uniform","uniform_int10"], "partition_strategy":"div", "max_norm":None},
        # {"opName":"embedding_lookup", "outName":"embedding_lookup/rank2_single_mod_maxnorm1", "varShapes":[[10, 5],[4]], "varTypes":["float32","int32"], "varInit":["uniform","uniform_int10"], "partition_strategy":"mod", "max_norm":1.0},
        # {"opName":"embedding_lookup", "outName":"embedding_lookup/rank2_multiple_div_nomaxnorm", "varShapes":[[4, 5],[3,5],[3,5],[4]], "varTypes":["float32","float32","float32","int32"], "varInit":["uniform","uniform","uniform","uniform_int10"], "partition_strategy":"div", "max_norm":None},
        # {"opName":"embedding_lookup", "outName":"embedding_lookup/rank2_multiple_mod_maxnorm1", "varShapes":[[4, 5],[3,5],[3,5],[4]], "varTypes":["float32","float32","float32","int32"], "varInit":["uniform","uniform","uniform","uniform_int10"], "partition_strategy":"mod", "max_norm":1.0},
        # {"opName":"embedding_lookup", "outName":"embedding_lookup/rank4_single_div_nomaxnorm", "varShapes":[[10, 5],[4]], "varTypes":["float32","int32"], "varInit":["uniform","uniform_int10"], "partition_strategy":"div", "max_norm":None},
        # {"opName":"embedding_lookup", "outName":"embedding_lookup/rank4_single_mod_maxnorm1", "varShapes":[[10, 2, 3, 4],[4]], "varTypes":["float32","int32"], "varInit":["uniform","uniform_int10"], "partition_strategy":"mod", "max_norm":1.0},
        # {"opName":"embedding_lookup", "outName":"embedding_lookup/rank4_multiple_div_nomaxnorm", "varShapes":[[4,2,3,4],[3,2,3,4],[3,2,3,4],[4]], "varTypes":["float32","float32","float32","int32"], "varInit":["uniform","uniform","uniform","uniform_int10"], "partition_strategy":"div", "max_norm":None},
        # {"opName":"embedding_lookup", "outName":"embedding_lookup/rank4_multiple_mod_maxnorm1", "varShapes":[[4,2,3,4],[3,2,3,4],[3,2,3,4],[4]], "varTypes":["float32","float32","float32","int32"], "varInit":["uniform","uniform","uniform","uniform_int10"], "partition_strategy":"mod", "max_norm":1.0},

        # {"opName":"l2_normalize", "outName":"l2_normalize/rank2_e0", "varShapes":[[4, 5]], "varTypes":["float32"], "varInit":["uniform"], "axis":1, "epsilon":0.0},
        # {"opName":"l2_normalize", "outName":"l2_normalize/rank2_e05", "varShapes":[[4, 5]], "varTypes":["float32"], "varInit":["uniform"], "axis":1, "epsilon":0.5},
        # {"opName":"l2_normalize", "outName":"l2_normalize/rank4_e0_d1", "varShapes":[[3,2,3,4]], "varTypes":["float32"], "varInit":["uniform"], "axis":1, "epsilon":0.0},
        # {"opName":"l2_normalize", "outName":"l2_normalize/rank4_e05_d123", "varShapes":[[3,3,4,5]], "varTypes":["float32"], "varInit":["uniform"], "axis":[1,2,3], "epsilon":0.5}

        # {"opName":"lrn", "outName":"lrn/dr5_b1_a1_b05", "varShapes":[[2, 4, 4, 8]], "varTypes":["float32"], "varInit":["uniform"], "depth_radius":5, "bias":1.0, "alpha":1.0, "beta":0.5},
        # {"opName":"lrn", "outName":"lrn/dr3_b05_a05_b02", "varShapes":[[2, 4, 4, 8]], "varTypes":["float32"], "varInit":["uniform"], "depth_radius":3, "bias":0.5, "alpha":0.5, "beta":0.2},

        #Dropouts. Note that due to random nature - we need to validate these differently than simply "samediff output equals tensorflow output"
        # {"opName":"layers_dropout", "outName":"layers_dropout/rank2_d05_train", "varShapes":[[4,5]], "varTypes":["float32"], "varInit":["uniform"], "rate":0.5, "training":True},
        # {"opName":"layers_dropout", "outName":"layers_dropout/rank2_d05_test", "varShapes":[[4,5]], "varTypes":["float32"], "varInit":["uniform"], "rate":0.5, "training":False},
        # {"opName":"layers_dropout", "outName":"layers_dropout/rank2_d01_train", "varShapes":[[4,5]], "varTypes":["float32"], "varInit":["uniform"], "rate":0.1, "training":True},
        # {"opName":"layers_dropout", "outName":"layers_dropout/rank2_d01_test", "varShapes":[[4,5]], "varTypes":["float32"], "varInit":["uniform"], "rate":0.1, "training":False},
        # {"opName":"layers_dropout", "outName":"layers_dropout/rank2_d09_train", "varShapes":[[4,5]], "varTypes":["float32"], "varInit":["uniform"], "rate":0.9, "training":True},
        # {"opName":"layers_dropout", "outName":"layers_dropout/rank3_d05_train_mask1", "varShapes":[[4,5,6]], "varTypes":["float32"], "varInit":["uniform"], "rate":0.5, "training":True, "noise_shape":[4,1,6]},
        # {"opName":"layers_dropout", "outName":"layers_dropout/rank3_d05_train_mask2", "varShapes":[[4,5,6]], "varTypes":["float32"], "varInit":["uniform"], "rate":0.5, "training":True, "noise_shape":[4,5,1]},
        # {"opName":"layers_dropout", "outName":"layers_dropout/rank3_d05_test", "varShapes":[[4,5,6]], "varTypes":["float32"], "varInit":["uniform"], "rate":0.5, "training":False},
        # {"opName":"layers_dropout", "outName":"layers_dropout/rank4_d05_train", "varShapes":[[2,5,5,3]], "varTypes":["float32"], "varInit":["uniform"], "rate":0.5, "training":True},
        # {"opName":"layers_dropout", "outName":"layers_dropout/rank4_d05_train_mask", "varShapes":[[2,5,5,3]], "varTypes":["float32"], "varInit":["uniform"], "rate":0.5, "training":True, "noise_shape":[2,1,1,3]},

        # {"opName":"contrib_nn_alpha_dropout", "outName":"alpha_dropout/rank2_p05", "varShapes":[[4,5]], "varTypes":["float32"], "varInit":["uniform"], "keep_prob":0.5},
        # {"opName":"contrib_nn_alpha_dropout", "outName":"alpha_dropout/rank2_p01", "varShapes":[[4,5]], "varTypes":["float32"], "varInit":["uniform"], "keep_prob":0.1},
        # {"opName":"contrib_nn_alpha_dropout", "outName":"alpha_dropout/rank2_p09", "varShapes":[[4,5]], "varTypes":["float32"], "varInit":["uniform"], "keep_prob":0.9},
        # {"opName":"contrib_nn_alpha_dropout", "outName":"alpha_dropout/rank3_p05_mask1", "varShapes":[[4,5,6]], "varTypes":["float32"], "varInit":["uniform"], "keep_prob":0.5, "noise_shape":[4,1,6]},
        # {"opName":"contrib_nn_alpha_dropout", "outName":"alpha_dropout/rank3_p05_mask2", "varShapes":[[4,5,6]], "varTypes":["float32"], "varInit":["uniform"], "keep_prob":0.5, "noise_shape":[4,5,1]},
        # {"opName":"contrib_nn_alpha_dropout", "outName":"alpha_dropout/rank4_p05", "varShapes":[[2,5,5,3]], "varTypes":["float32"], "varInit":["uniform"], "keep_prob":0.5},
        # {"opName":"contrib_nn_alpha_dropout", "outName":"alpha_dropout/rank4_p05_mask", "varShapes":[[2,5,5,3]], "varTypes":["float32"], "varInit":["uniform"], "keep_prob":0.5, "noise_shape":[2,1,1,3]},

        #Meshgrid - seems like TF doesn't like things like [[3], [4]] - "ValueError: Dimension 0 in both shapes must be equal, but are 3 and 4. Shapes are [3] and [4]."
        # {"opName":"meshgrid", "outName":"meshgrid/n1_xy", "varShapes":[[3]], "varTypes":["float32"], "varInit":["uniform"], "indexing":"xy"},
        # {"opName":"meshgrid", "outName":"meshgrid/n1_ij", "varShapes":[[3]], "varTypes":["float32"], "varInit":["uniform"], "indexing":"ij"},
        # {"opName":"meshgrid", "outName":"meshgrid/n2_xy", "varShapes":[[3],[3]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"], "indexing":"xy"},
        # {"opName":"meshgrid", "outName":"meshgrid/n2_ij", "varShapes":[[3],[3]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"], "indexing":"ij"},
        # {"opName":"meshgrid", "outName":"meshgrid/n3_xy", "varShapes":[[3],[3],[3]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"], "indexing":"xy"},
        # {"opName":"meshgrid", "outName":"meshgrid/n3_ij", "varShapes":[[3],[3],[3]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"], "indexing":"ij"},
        # {"opName":"meshgrid", "outName":"meshgrid/n4_xy", "varShapes":[[3],[3],[3],[3]], "varTypes":["float32","float32","float32","float32"], "varInit":["uniform","uniform","uniform","uniform"],"indexing":"xy"},
        # {"opName":"meshgrid", "outName":"meshgrid/n4_ij", "varShapes":[[3],[3],[3],[3]], "varTypes":["float32","float32","float32","float32"], "varInit":["uniform","uniform","uniform","uniform"],"indexing":"ij"}

        #{"opName":"eye", "outName":"eye/e22", "varShapes":[], "varTypes":[], "varInit":[], "num_rows":2, "num_columns":2},
        #{"opName":"eye", "outName":"eye/e23", "varShapes":[], "varTypes":[], "varInit":[], "num_rows":2, "num_columns":3},
        #{"opName":"eye", "outName":"eye/e32", "varShapes":[], "varTypes":[], "varInit":[], "num_rows":3, "num_columns":2},
        #{"opName":"eye", "outName":"eye/e22_b1", "varShapes":[[1]], "varTypes":["int32"], "varInit":["one"], "num_rows":2, "num_columns":2},
        #{"opName":"eye", "outName":"eye/e23_b2", "varShapes":[[1]], "varTypes":["int32"], "varInit":["two"], "num_rows":2, "num_columns":3},
        #{"opName":"eye", "outName":"eye/e32_b22", "varShapes":[[2]], "varTypes":["int32"], "varInit":["two"], "num_rows":3, "num_columns":2},


        #log determinant op: requires SYMMETRIC matrix. slogdet uses log determinant internally - so same thing...
        # {"opName":"log_determinant", "outName":"log_determinant/rank2", "varShapes":[[3,3]], "varTypes":["float32"], "varInit":["positive_def_symmetric_33"]},
        # {"opName":"log_determinant", "outName":"log_determinant/rank3", "varShapes":[[2,3,3]], "varTypes":["float32"], "varInit":["positive_def_symmetric_233"]},
        # {"opName":"slog_determinant", "outName":"slogdet/rank2", "varShapes":[[3,3]], "varTypes":["float32"], "varInit":["positive_def_symmetric_33"]},
        # {"opName":"slog_determinant", "outName":"slogdet/rank3", "varShapes":[[2,3,3]], "varTypes":["float32"], "varInit":["positive_def_symmetric_233"]},

        # {"opName":"sequence_mask", "outName":"sequence_mask/rank1_auto_maxlen", "varShapes":[[5]], "varTypes":["int32"], "varInit":["uniform_int5"]},
        # {"opName":"sequence_mask", "outName":"sequence_mask/rank1_provided_maxlen", "varShapes":[[5], []], "varTypes":["int32", "int32"], "varInit":["uniform_int5", "ten"]},
        # {"opName":"sequence_mask", "outName":"sequence_mask/rank2_auto_maxlen", "varShapes":[[3,4]], "varTypes":["int32"], "varInit":["uniform_int5"]},
        # {"opName":"sequence_mask", "outName":"sequence_mask/rank2_provided_maxlen", "varShapes":[[3,4], []], "varTypes":["int32", "int32"], "varInit":["uniform_int5", "ten"]},
        # {"opName":"sequence_mask", "outName":"sequence_mask/rank3_auto_maxlen", "varShapes":[[2,3,4]], "varTypes":["int32"], "varInit":["uniform_int5"]},
        # {"opName":"sequence_mask", "outName":"sequence_mask/rank3_provided_maxlen", "varShapes":[[2,3,4], []], "varTypes":["int32", "int32"], "varInit":["uniform_int5", "ten"]},

        # {"opName":"rint", "outName":"rint/rank0", "varShapes":[[]], "varTypes":["float32"], "varInit":["stdnormal"]},
        # {"opName":"rint", "outName":"rint/rank1", "varShapes":[[10]], "varTypes":["float32"], "varInit":["stdnormal"]},
        # {"opName":"rint", "outName":"rint/rank2", "varShapes":[[3,4]], "varTypes":["float32"], "varInit":["stdnormal"]},

        # {"opName":"histogram_fixed_width", "outName":"histogram_fixed_width/rank0", "varShapes":[[], [2]], "varTypes":["float32", "float32"], "varInit":["stdnormal", "fixed_m1_1"], "nbins":5},
        # {"opName":"histogram_fixed_width", "outName":"histogram_fixed_width/rank1", "varShapes":[[10], [2]], "varTypes":["float32", "float32"], "varInit":["stdnormal", "fixed_m1_1"], "nbins":5},
        # {"opName":"histogram_fixed_width", "outName":"histogram_fixed_width/rank2", "varShapes":[[10,10], [2]], "varTypes":["float32", "float32"], "varInit":["stdnormal", "fixed_m1_1"], "nbins":20}

        # {"opName":"bincount", "outName":"bincount/rank0", "varShapes":[[]], "varTypes":["int32"], "varInit":["uniform_int10"], "minlength":None, "maxlength":None},
        # {"opName":"bincount", "outName":"bincount/rank0_minmax", "varShapes":[[]], "varTypes":["int32"], "varInit":["uniform_int10"], "minlength":3, "maxlength":8},
        # {"opName":"bincount", "outName":"bincount/rank0_weights", "varShapes":[[],[]], "varTypes":["int32", "float32"], "varInit":["uniform_int10","uniform"], "minlength":None, "maxlength":None},
        # {"opName":"bincount", "outName":"bincount/rank1", "varShapes":[[10]], "varTypes":["int32"], "varInit":["uniform_int10"], "minlength":None, "maxlength":None},
        # {"opName":"bincount", "outName":"bincount/rank1_minmax", "varShapes":[[10]], "varTypes":["int32"], "varInit":["uniform_int10","uniform"], "minlength":3, "maxlength":8},
        # {"opName":"bincount", "outName":"bincount/rank1_min10", "varShapes":[[10]], "varTypes":["int32"], "varInit":["uniform_int10","uniform"], "minlength":10, "maxlength":None},
        # {"opName":"bincount", "outName":"bincount/rank1_max5", "varShapes":[[10]], "varTypes":["int32"], "varInit":["uniform_int10","uniform"], "minlength":None, "maxlength":5},
        # {"opName":"bincount", "outName":"bincount/rank1_weights", "varShapes":[[10],[10]], "varTypes":["int32", "float32"], "varInit":["uniform_int10","uniform"], "minlength":None, "maxlength":None},
        # #TF bug? Next one doess't work
        # #{"opName":"bincount", "outName":"bincount/rank1_minmax_weights", "varShapes":[[10],[10]], "varTypes":["int32", "float32"], "varInit":["uniform_int10","uniform"], "minlength":3, "maxlength":8},
        # #TF bug? Next one doess't work
        # # {"opName":"bincount", "outName":"bincount/rank2_minmax_weights", "varShapes":[[5,5],[5,5]], "varTypes":["int32", "float32"], "varInit":["uniform_int10","uniform"], "minlength":3, "maxlength":8},
        # {"opName":"bincount", "outName":"bincount/rank2_weights", "varShapes":[[5,5],[5,5]], "varTypes":["int32", "float32"], "varInit":["uniform_int10","uniform"], "minlength":None, "maxlength":None},

        #Scatter ND: arrays are indices, updates. Updates has shape indices.shape[:-1] + shape[indices.shape[-1]:]
            #This case: update has shape [4]+[] = 4
        # {"opName":"scatter_nd", "outName":"scatter_nd/rank1shape_1indices", "varShapes":[[4,1],[4]], "varTypes":["int32", "float32"], "varInit":["fixed_3_1_4_2", "uniform"], "shape":[10]},
        #     #This case: 2 indices, 2 shape -> updates are individual elements
        # {"opName":"scatter_nd", "outName":"scatter_nd/rank2shape_2indices", "varShapes":[[4,2],[4]], "varTypes":["int32", "float32"], "varInit":["unique_rand_10", "uniform"], "shape":[10,10]},
        #     #This case: 1 indices, 2 shape -> updates are slices from shape [4]+[7]=[4,7]
        # {"opName":"scatter_nd", "outName":"scatter_nd/rank2shape_1indices", "varShapes":[[4,1],[4,7]], "varTypes":["int32", "float32"], "varInit":["unique_rand_10", "uniform"], "shape":[10,7]},
        #     #This case: 2 indices, 3 shape -> updates are slices of shape [2]+[5] = [2,5]
        # {"opName":"scatter_nd", "outName":"scatter_nd/rank3shape_2indices", "varShapes":[[2,2],[2,5]], "varTypes":["int32", "float32"], "varInit":["unique_rand_5", "uniform"], "shape":[10,7,5]},
        #     #This case: 1 indices, 3 shape -> updates are slices of shape [4]+[7,5] = [4,7,5]
        # {"opName":"scatter_nd", "outName":"scatter_nd/rank3shape_1indices", "varShapes":[[4,1],[4,7,5]], "varTypes":["int32", "float32"], "varInit":["uniform_int10", "uniform"], "shape":[10,7,5]},

        # #Scatter ND ADD: arrays are ref, indices, updates. Updates has shape indices.shape[:-1] + shape[indices.shape[-1]:]
        # #This case: update has shape [4]+[] = 4
        # {"opName":"scatter_nd_add", "outName":"scatter_nd_add/locking/rank1shape_1indices", "varShapes":[[10], [4,1],[4]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "uniform_int10", "uniform"], "use_locking":True},
        # {"opName":"scatter_nd_add", "outName":"scatter_nd_add/unique_idxs/rank1shape_1indices", "varShapes":[[10], [4,1],[4]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "unique_rand_10", "uniform"], "use_locking":False},
        # #This case: 2 indices, 2 shape -> updates are individual elements
        # {"opName":"scatter_nd_add", "outName":"scatter_nd_add/locking/rank2shape_2indices", "varShapes":[[10,10], [4,2],[4]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "uniform_int10", "uniform"], "use_locking":True},
        # {"opName":"scatter_nd_add", "outName":"scatter_nd_add/unique_idxs/rank2shape_2indices", "varShapes":[[10,10], [4,2],[4]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "unique_rand_10", "uniform"], "use_locking":False},
        # #This case: 1 indices, 2 shape -> updates are slices of shape [4]+[7]
        # {"opName":"scatter_nd_add", "outName":"scatter_nd_add/locking/rank2shape_1indices", "varShapes":[[10,7], [4,1],[4,7]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "uniform_int10", "uniform"], "use_locking":True},
        # {"opName":"scatter_nd_add", "outName":"scatter_nd_add/unique_idxs/rank2shape_1indices", "varShapes":[[10,7], [4,1],[4,7]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "unique_rand_10", "uniform"], "use_locking":False},
        # # This case: 2 indices, 3 shape -> updates are slices of shape [4]+[5] = [4,5]
        # {"opName":"scatter_nd_add", "outName":"scatter_nd_add/locking/rank3shape_2indices", "varShapes":[[10,7,5], [4,2],[4,5]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "uniform_int10", "uniform"], "use_locking":True},
        # {"opName":"scatter_nd_add", "outName":"scatter_nd_add/unique_idxs/rank3shape_2indices", "varShapes":[[10,7,5], [4,2],[4,5]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "unique_rand_10", "uniform"], "use_locking":False},
        # #This case: 1 indices, 3 shape -> updates are slices of shape [4]+[7,5] = [4,7,5]
        # {"opName":"scatter_nd_add", "outName":"scatter_nd_add/locking/rank3shape_1indices", "varShapes":[[10,7,5], [4,1],[4,7,5]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "uniform_int10", "uniform"], "use_locking":True},
        # {"opName":"scatter_nd_add", "outName":"scatter_nd_add/unique_idxs/rank3shape_1indices", "varShapes":[[10,7,5], [4,1],[4,7,5]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "unique_rand_10", "uniform"], "use_locking":False},
        #
        # {"opName":"scatter_nd_sub", "outName":"scatter_nd_sub/locking/rank1shape_1indices", "varShapes":[[10], [4,1],[4]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "uniform_int10", "uniform"], "use_locking":True},
        # {"opName":"scatter_nd_sub", "outName":"scatter_nd_sub/locking/rank2shape_2indices", "varShapes":[[10,10], [4,2],[4]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "uniform_int10", "uniform"], "use_locking":True},
        # {"opName":"scatter_nd_sub", "outName":"scatter_nd_sub/locking/rank2shape_1indices", "varShapes":[[10,7], [4,1],[4,7]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "uniform_int10", "uniform"], "use_locking":True},
        # {"opName":"scatter_nd_sub", "outName":"scatter_nd_sub/locking/rank3shape_2indices", "varShapes":[[10,7,5], [4,2],[4,5]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "uniform_int10", "uniform"], "use_locking":True},
        # {"opName":"scatter_nd_sub", "outName":"scatter_nd_sub/locking/rank3shape_1indices", "varShapes":[[10,7,5], [4,1],[4,7,5]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "uniform_int10", "uniform"], "use_locking":True},
        # {"opName":"scatter_nd_sub", "outName":"scatter_nd_sub/unique_idxs/rank1shape_1indices", "varShapes":[[10], [4,1],[4]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "unique_rand_10", "uniform"], "use_locking":False},
        # {"opName":"scatter_nd_sub", "outName":"scatter_nd_sub/unique_idxs/rank2shape_2indices", "varShapes":[[10,10], [4,2],[4]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "unique_rand_10", "uniform"], "use_locking":False},
        # {"opName":"scatter_nd_sub", "outName":"scatter_nd_sub/unique_idxs/rank2shape_1indices", "varShapes":[[10,7], [4,1],[4,7]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "unique_rand_10", "uniform"], "use_locking":False},
        # {"opName":"scatter_nd_sub", "outName":"scatter_nd_sub/unique_idxs/rank3shape_2indices", "varShapes":[[10,7,5], [4,2],[4,5]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "unique_rand_10", "uniform"], "use_locking":False},
        # {"opName":"scatter_nd_sub", "outName":"scatter_nd_sub/unique_idxs/rank3shape_1indices", "varShapes":[[10,7,5], [4,1],[4,7,5]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "unique_rand_10", "uniform"], "use_locking":False},

        # {"opName":"scatter_nd_update", "outName":"scatter_nd_update/locking/rank1shape_1indices", "varShapes":[[10], [4,1],[4]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "uniform_int10", "uniform"], "use_locking":True},
        # {"opName":"scatter_nd_update", "outName":"scatter_nd_update/locking/rank2shape_2indices", "varShapes":[[10,10], [4,2],[4]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "uniform_int10", "uniform"], "use_locking":True},
        # {"opName":"scatter_nd_update", "outName":"scatter_nd_update/locking/rank2shape_1indices", "varShapes":[[10,7], [4,1],[4,7]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "uniform_int10", "uniform"], "use_locking":True},
        # {"opName":"scatter_nd_update", "outName":"scatter_nd_update/locking/rank3shape_2indices", "varShapes":[[10,7,5], [4,2],[4,5]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "uniform_int10", "uniform"], "use_locking":True},
        # {"opName":"scatter_nd_update", "outName":"scatter_nd_update/locking/rank3shape_1indices", "varShapes":[[10,7,5], [4,1],[4,7,5]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "uniform_int10", "uniform"], "use_locking":True},
        # {"opName":"scatter_nd_update", "outName":"scatter_nd_update/unique_idxs/rank1shape_1indices", "varShapes":[[10], [4,1],[4]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "unique_rand_10", "uniform"], "use_locking":False},
        # {"opName":"scatter_nd_update", "outName":"scatter_nd_update/unique_idxs/rank2shape_2indices", "varShapes":[[10,10], [4,2],[4]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "unique_rand_10", "uniform"], "use_locking":False},
        # {"opName":"scatter_nd_update", "outName":"scatter_nd_update/unique_idxs/rank2shape_1indices", "varShapes":[[10,7], [4,1],[4,7]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "unique_rand_10", "uniform"], "use_locking":False},
        # {"opName":"scatter_nd_update", "outName":"scatter_nd_update/unique_idxs/rank3shape_2indices", "varShapes":[[10,7,5], [4,2],[4,5]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "unique_rand_10", "uniform"], "use_locking":False},
        # {"opName":"scatter_nd_update", "outName":"scatter_nd_update/unique_idxs/rank3shape_1indices", "varShapes":[[10,7,5], [4,1],[4,7,5]], "varTypes":["float32", "int32", "float32"], "varInit":["one", "unique_rand_10", "uniform"], "use_locking":False},

        #Seems like shift=None is NOT supported: "TypeError: Fetch argument None has invalid type <class 'NoneType'>"
        # {"opName":"sufficient_statistics", "outName":"sufficient_statistics/rank1", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform"], "axes":[0], "shift":0.0, "keep_dims":False},
        # {"opName":"sufficient_statistics", "outName":"sufficient_statistics/rank1_keep", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform"], "axes":[0], "shift":0.0, "keep_dims":True},
        # {"opName":"sufficient_statistics", "outName":"sufficient_statistics/rank1_keep_shift05", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform"], "axes":[0], "shift":0.5, "keep_dims":True},
        # {"opName":"sufficient_statistics", "outName":"sufficient_statistics/rank2_d0", "varShapes":[[5,5]], "varTypes":["float32"], "varInit":["uniform"], "axes":[0], "shift":0.0, "keep_dims":False},
        # {"opName":"sufficient_statistics", "outName":"sufficient_statistics/rank2_d1_keep", "varShapes":[[5,5]], "varTypes":["float32"], "varInit":["uniform"], "axes":[1], "shift":0.0, "keep_dims":True},
        # {"opName":"sufficient_statistics", "outName":"sufficient_statistics/rank2_d01_keep_shift05", "varShapes":[[5,5]], "varTypes":["float32"], "varInit":["uniform"], "axes":[0,1], "shift":0.5, "keep_dims":True},
        # {"opName":"sufficient_statistics", "outName":"sufficient_statistics/rank3_d0", "varShapes":[[3,4,5]], "varTypes":["float32"], "varInit":["uniform"], "axes":[0], "shift":0.0, "keep_dims":False},
        # {"opName":"sufficient_statistics", "outName":"sufficient_statistics/rank3_d1", "varShapes":[[3,4,5]], "varTypes":["float32"], "varInit":["uniform"], "axes":[1], "shift":0.0, "keep_dims":False},
        # {"opName":"sufficient_statistics", "outName":"sufficient_statistics/rank3_d2", "varShapes":[[3,4,5]], "varTypes":["float32"], "varInit":["uniform"], "axes":[2], "shift":0.0, "keep_dims":False},

        # {"opName":"split", "outName":"split/rank1_10_num2_axis-1", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform"], "num_or_size_split":2, "axis":-1},
        # {"opName":"split", "outName":"split/rank1_9_num3_axis0", "varShapes":[[9]], "varTypes":["float32"], "varInit":["uniform"], "num_or_size_split":3, "axis":0},
        # {"opName":"split", "outName":"split/rank2_3,8_num2_axis-1", "varShapes":[[3,8]], "varTypes":["float32"], "varInit":["uniform"], "num_or_size_split":2, "axis":-1},
        # {"opName":"split", "outName":"split/rank2_8,4_num2_axis0", "varShapes":[[8,4]], "varTypes":["float32"], "varInit":["uniform"], "num_or_size_split":2, "axis":0},
        # {"opName":"split", "outName":"split/rank2_3,8_num2_axis1", "varShapes":[[3,8]], "varTypes":["float32"], "varInit":["uniform"], "num_or_size_split":2, "axis":1},
        # {"opName":"split", "outName":"split/rank2_8,7_sz5,3_axis-2", "varShapes":[[8,7]], "varTypes":["float32"], "varInit":["uniform"], "num_or_size_split":[5,3], "axis":-2},
        # {"opName":"split", "outName":"split/rank2_8,7_sz5,3_axis0", "varShapes":[[8,7]], "varTypes":["float32"], "varInit":["uniform"], "num_or_size_split":[5,3], "axis":0},
        # {"opName":"split", "outName":"split/rank2_8,7_sz2,1,4_axis-1", "varShapes":[[8,7]], "varTypes":["float32"], "varInit":["uniform"], "num_or_size_split":[2,1,4], "axis":-1},
        # {"opName":"split", "outName":"split/rank2_8,7_sz2,1,4_axis1", "varShapes":[[8,7]], "varTypes":["float32"], "varInit":["uniform"], "num_or_size_split":[2,1,4], "axis":1},
        # {"opName":"split", "outName":"split/rank3_9,3,4_num3_axis0", "varShapes":[[9,3,4]], "varTypes":["float32"], "varInit":["uniform"], "num_or_size_split":3, "axis":0},
        # {"opName":"split", "outName":"split/rank3_3,9,4_num3_axis1", "varShapes":[[3,9,4]], "varTypes":["float32"], "varInit":["uniform"], "num_or_size_split":3, "axis":1},
        # {"opName":"split", "outName":"split/rank3_3,4,9_num3_axis2", "varShapes":[[3,4,9]], "varTypes":["float32"], "varInit":["uniform"], "num_or_size_split":3, "axis":2},
        # {"opName":"split", "outName":"split/rank3_10,3,4_sz2,3,5_axis0", "varShapes":[[10,3,4]], "varTypes":["float32"], "varInit":["uniform"], "num_or_size_split":[2,3,5], "axis":0},

        # {"opName":"reduce_logsumexp", "outName":"logsumexp/rank0", "varShapes":[[]], "varTypes":["float32"], "varInit":["uniform"], "axis":None, "keep_dims":False},
        # {"opName":"reduce_logsumexp", "outName":"logsumexp/rank1", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform"], "axis":None, "keep_dims":False},
        # {"opName":"reduce_logsumexp", "outName":"logsumexp/rank1_d0", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform"], "axis":0, "keep_dims":False},
        # {"opName":"reduce_logsumexp", "outName":"logsumexp/rank1_d-1", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform"], "axis":-1, "keep_dims":False},
        # {"opName":"reduce_logsumexp", "outName":"logsumexp/rank1_d0_keep", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform"], "axis":0, "keep_dims":True},
        # {"opName":"reduce_logsumexp", "outName":"logsumexp/rank1_keep", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform"], "axis":None, "keep_dims":True},
        # {"opName":"reduce_logsumexp", "outName":"logsumexp/rank2", "varShapes":[[4,5]], "varTypes":["float32"], "varInit":["uniform"], "axis":None, "keep_dims":False},
        # {"opName":"reduce_logsumexp", "outName":"logsumexp/rank2_d0", "varShapes":[[4,5]], "varTypes":["float32"], "varInit":["uniform"], "axis":0, "keep_dims":False},
        # {"opName":"reduce_logsumexp", "outName":"logsumexp/rank2_d1", "varShapes":[[4,5]], "varTypes":["float32"], "varInit":["uniform"], "axis":1, "keep_dims":False},
        # {"opName":"reduce_logsumexp", "outName":"logsumexp/rank2_d-1", "varShapes":[[4,5]], "varTypes":["float32"], "varInit":["uniform"], "axis":-1, "keep_dims":False},
        # {"opName":"reduce_logsumexp", "outName":"logsumexp/rank2_d0_keep", "varShapes":[[4,5]], "varTypes":["float32"], "varInit":["uniform"], "axis":0, "keep_dims":True},
        # {"opName":"reduce_logsumexp", "outName":"logsumexp/rank2_keep", "varShapes":[[4,5]], "varTypes":["float32"], "varInit":["uniform"], "axis":None, "keep_dims":True},

        # {"opName":"nth_element", "outName":"nth_element/rank1_n0", "varShapes":[[10],[]], "varTypes":["float32","int32"], "varInit":["uniform","zero"], "reverse":False},
        # {"opName":"nth_element", "outName":"nth_element/rank1_n3_reverse", "varShapes":[[10],[]], "varTypes":["float32","int32"], "varInit":["uniform","three"], "reverse":True},
        # {"opName":"nth_element", "outName":"nth_element/rank2_n0", "varShapes":[[10],[]], "varTypes":["float32","int32"], "varInit":["uniform","zero"], "reverse":False},
        # {"opName":"nth_element", "outName":"nth_element/rank2_n4", "varShapes":[[5,6],[]], "varTypes":["float32","int32"], "varInit":["uniform","four"], "reverse":False},
        # {"opName":"nth_element", "outName":"nth_element/rank2_n3_reverse", "varShapes":[[5,5],[]], "varTypes":["float32","int32"], "varInit":["uniform","three"], "reverse":True},
        # {"opName":"nth_element", "outName":"nth_element/rank3_n2", "varShapes":[[2,3,4],[]], "varTypes":["float32","int32"], "varInit":["uniform","two"], "reverse":False},
        #
        # {"opName":"reduce_any", "outName":"reduce_any/rank0", "varShapes":[[]], "varTypes":["bool"], "varInit":["boolean"], "axis":None, "keep_dims":False},
        # {"opName":"reduce_any", "outName":"reduce_any/rank1", "varShapes":[[3]], "varTypes":["bool"], "varInit":["boolean"], "axis":None, "keep_dims":False},
        # {"opName":"reduce_any", "outName":"reduce_any/rank1_d0_keep", "varShapes":[[3]], "varTypes":["bool"], "varInit":["boolean"], "axis":0, "keep_dims":True},
        # {"opName":"reduce_any", "outName":"reduce_any/rank2", "varShapes":[[3,4]], "varTypes":["bool"], "varInit":["boolean"], "axis":None, "keep_dims":False},
        # {"opName":"reduce_any", "outName":"reduce_any/rank2_d0_keep", "varShapes":[[3,4]], "varTypes":["bool"], "varInit":["boolean"], "axis":0, "keep_dims":True},
        # {"opName":"reduce_any", "outName":"reduce_any/rank2_d1_keep", "varShapes":[[3,4]], "varTypes":["bool"], "varInit":["boolean"], "axis":1, "keep_dims":False},
        # {"opName":"reduce_any", "outName":"reduce_any/rank3_d01_keep", "varShapes":[[2,3,4]], "varTypes":["bool"], "varInit":["boolean"], "axis":[0,1], "keep_dims":True},
        #
        # {"opName":"reduce_all", "outName":"reduce_all/rank0", "varShapes":[[]], "varTypes":["bool"], "varInit":["boolean"], "axis":None, "keep_dims":False},
        # {"opName":"reduce_all", "outName":"reduce_all/rank1", "varShapes":[[3]], "varTypes":["bool"], "varInit":["boolean"], "axis":None, "keep_dims":False},
        # {"opName":"reduce_all", "outName":"reduce_all/rank1_d0_keep", "varShapes":[[3]], "varTypes":["bool"], "varInit":["boolean"], "axis":0, "keep_dims":True},
        # {"opName":"reduce_all", "outName":"reduce_all/rank2", "varShapes":[[3,4]], "varTypes":["bool"], "varInit":["boolean"], "axis":None, "keep_dims":False},
        # {"opName":"reduce_all", "outName":"reduce_all/rank2_d0_keep", "varShapes":[[3,4]], "varTypes":["bool"], "varInit":["boolean"], "axis":0, "keep_dims":True},
        # {"opName":"reduce_all", "outName":"reduce_all/rank2_d1_keep", "varShapes":[[3,4]], "varTypes":["bool"], "varInit":["boolean"], "axis":1, "keep_dims":False},
        # {"opName":"reduce_all", "outName":"reduce_all/rank3_d01_keep", "varShapes":[[2,3,4]], "varTypes":["bool"], "varInit":["boolean"], "axis":[0,1], "keep_dims":True},
        #{"opName": "reduce_all", "outName": "reduce_all/empty_axis0", "varShapes": [[0, 0, 0]], "varTypes": ["bool"], "varInit": ["boolean"], "axis": [0], "keep_dims": True},
        #{"opName": "reduce_all", "outName": "reduce_all/empty_axis1", "varShapes": [[0, 0, 3]], "varTypes": ["bool"], "varInit": ["boolean"], "axis": [1], "keep_dims": True},
        #{"opName":"reduce_all", "outName":"reduce_all/empty_axis2", "varShapes":[[2,0,3]], "varTypes":["bool"], "varInit":["boolean"], "axis":[2], "keep_dims":True},

        # {"opName":"boolean_mask", "outName":"boolean_mask/rank1_mask1", "varShapes":[[10],[10]], "varTypes":["float32", "bool"], "varInit":["uniform", "boolean"]},
        # {"opName":"boolean_mask", "outName":"boolean_mask/rank2_mask1", "varShapes":[[5,4],[5]], "varTypes":["float32", "bool"], "varInit":["uniform", "boolean"]},
        # {"opName":"boolean_mask", "outName":"boolean_mask/rank2_mask2", "varShapes":[[5,4],[5,4]], "varTypes":["float32", "bool"], "varInit":["uniform", "boolean"]},
        # {"opName":"boolean_mask", "outName":"boolean_mask/rank3_mask1", "varShapes":[[5,4,3],[5]], "varTypes":["float32", "bool"], "varInit":["uniform", "boolean"]},
        # {"opName":"boolean_mask", "outName":"boolean_mask/rank3_mask2", "varShapes":[[5,4,3],[5,4]], "varTypes":["float32", "bool"], "varInit":["uniform", "boolean"]},
        # {"opName":"boolean_mask", "outName":"boolean_mask/rank3_mask2", "varShapes":[[5,4,3],[5,4,3]], "varTypes":["float32", "bool"], "varInit":["uniform", "boolean"]},
        #
        # {"opName":"where", "outName":"where/cond_only_rank1", "varShapes":[[5]], "varTypes":["float32"], "varInit":["fixed_5"]},
        # {"opName":"where", "outName":"where/cond_only_rank2", "varShapes":[[2,3]], "varTypes":["float32"], "varInit":["fixed_2_3"]},
        # {"opName":"where", "outName":"where/cond_only_rank3", "varShapes":[[2,3,4]], "varTypes":["float32"], "varInit":["fixed_2_3_4"]},
        # {"opName":"where", "outName":"where/cond_rank1_xy_rank1", "varShapes":[[5],[5],[5]], "varTypes":["bool", "float32", "float32"], "varInit":["boolean", "uniform", "uniform"]},
        # {"opName":"where", "outName":"where/cond_rank2_xy_rank2", "varShapes":[[3,4],[3,4],[3,4]], "varTypes":["bool", "float32", "float32"], "varInit":["boolean", "uniform", "uniform"]},
        # {"opName":"where", "outName":"where/cond_rank3_xy_rank3", "varShapes":[[2,3,4],[2,3,4],[2,3,4]], "varTypes":["bool", "float32", "float32"], "varInit":["boolean", "uniform", "uniform"]},
        #"If condition is a vector and x and y are higher rank matrices, then it chooses which row (outer dimension) to copy from x and y"
        #"If condition is rank 1, x may have higher rank, but its first dimension must match the size of condition."
        # {"opName":"where", "outName":"where/cond_rank1_xy_rank2", "varShapes":[[5],[5,4],[5,4]], "varTypes":["bool", "float32", "float32"], "varInit":["boolean", "uniform", "uniform"]},
        # {"opName":"where", "outName":"where/cond_rank1_xy_rank3", "varShapes":[[5],[5,3,4],[5,3,4]], "varTypes":["bool", "float32", "float32"], "varInit":["boolean", "uniform", "uniform"]},

        # {"opName":"broadcast_dynamic_shape", "outName":"broadcast_dynamic_shape/1_4", "varShapes":[[1],[1]], "varTypes":["int32", "int32"], "varInit":["one", "four"]},
        # {"opName":"broadcast_dynamic_shape", "outName":"broadcast_dynamic_shape/1,1_4", "varShapes":[[2],[1]], "varTypes":["int32", "int32"], "varInit":["one", "four"]},
        # {"opName":"broadcast_dynamic_shape", "outName":"broadcast_dynamic_shape/2,2_1", "varShapes":[[2],[1]], "varTypes":["int32", "int32"], "varInit":["two", "one"]},
        # {"opName":"broadcast_dynamic_shape", "outName":"broadcast_dynamic_shape/2,1_4", "varShapes":[[2],[1]], "varTypes":["int32", "int32"], "varInit":["fixed_2_1", "four"]},
        # {"opName":"broadcast_dynamic_shape", "outName":"broadcast_dynamic_shape/2,1,4_2,2,4", "varShapes":[[3],[3]], "varTypes":["int32", "int32"], "varInit":["fixed_2_1_4", "fixed_2_2_4"]},

        # {"opName":"broadcast_to", "outName":"broadcast_to/1_4", "varShapes":[[],[1]], "varTypes":["int32", "int32"], "varInit":["uniform", "four"]},
        # {"opName":"broadcast_to", "outName":"broadcast_to/1_3,3", "varShapes":[[1],[2]], "varTypes":["int32", "int32"], "varInit":["uniform", "three"]},
        # {"opName":"broadcast_to", "outName":"broadcast_to/1_2,1", "varShapes":[[],[2]], "varTypes":["int32", "int32"], "varInit":["uniform", "fixed_2_1"]},
        # {"opName":"broadcast_to", "outName":"broadcast_to/3_5,3", "varShapes":[[3],[2]], "varTypes":["int32", "int32"], "varInit":["uniform", "fixed_5_3"]},
        # {"opName":"broadcast_to", "outName":"broadcast_to/1,3_5,3", "varShapes":[[1,3],[2]], "varTypes":["int32", "int32"], "varInit":["uniform", "fixed_5_3"]},
        # {"opName":"broadcast_to", "outName":"broadcast_to/2,1,4_2,2,4", "varShapes":[[2,1,4],[3]], "varTypes":["int32", "int32"], "varInit":["uniform", "fixed_2_2_4"]},

        # {"opName": "unsorted_segment_max", "outName": "unsorted_segment/unsorted_segment_max_rank1", "varShapes":[[20], [20]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "uniform_int5"], "num_segments":5},
        # {"opName": "unsorted_segment_mean", "outName": "unsorted_segment/unsorted_segment_mean_rank1", "varShapes":[[20], [20]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "uniform_int5"], "num_segments":5},
        # {"opName": "unsorted_segment_min", "outName": "unsorted_segment/unsorted_segment_min_rank1", "varShapes":[[20], [20]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "uniform_int5"], "num_segments":5},
        # {"opName": "unsorted_segment_prod", "outName": "unsorted_segment/unsorted_segment_prod_rank1", "varShapes":[[20], [20]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "uniform_int5"], "num_segments":5},
        # {"opName": "unsorted_segment_sum", "outName": "unsorted_segment/unsorted_segment_sum_rank1", "varShapes":[[20], [20]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "uniform_int5"], "num_segments":5},
        # {"opName": "unsorted_segment_sqrt_n", "outName": "unsorted_segment/unsorted_segment_sqrt_n_rank1", "varShapes":[[20], [20]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "uniform_int5"], "num_segments":5},
        # {"opName": "unsorted_segment_max", "outName": "unsorted_segment/unsorted_segment_max_rank2", "varShapes":[[6,3], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "uniform_int3"], "num_segments":3},
        # {"opName": "unsorted_segment_mean", "outName": "unsorted_segment/unsorted_segment_mean_rank2", "varShapes":[[6,3], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "uniform_int3"], "num_segments":3},
        # {"opName": "unsorted_segment_min", "outName": "unsorted_segment/unsorted_segment_min_rank2", "varShapes":[[6,3], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "uniform_int3"], "num_segments":3},
        # {"opName": "unsorted_segment_prod", "outName": "unsorted_segment/unsorted_segment_prod_rank2", "varShapes":[[6,3], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "uniform_int3"], "num_segments":3},
        # {"opName": "unsorted_segment_sum", "outName": "unsorted_segment/unsorted_segment_sum_rank2", "varShapes":[[6,3], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "uniform_int3"], "num_segments":3},
        # {"opName": "unsorted_segment_sqrt_n", "outName": "unsorted_segment/unsorted_segment_sqrt_n_rank2", "varShapes":[[6,3], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "uniform_int3"], "num_segments":3},
        # {"opName": "unsorted_segment_max", "outName": "unsorted_segment/unsorted_segment_max_rank3", "varShapes":[[6,3,2], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "uniform_int3"], "num_segments":3},
        # {"opName": "unsorted_segment_mean", "outName": "unsorted_segment/unsorted_segment_mean_rank3", "varShapes":[[6,3,2], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "uniform_int3"], "num_segments":3},
        # {"opName": "unsorted_segment_min", "outName": "unsorted_segment/unsorted_segment_min_rank3", "varShapes":[[6,3,2], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "uniform_int3"], "num_segments":3},
        # {"opName": "unsorted_segment_prod", "outName": "unsorted_segment/unsorted_segment_prod_rank3", "varShapes":[[6,3,2], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "uniform_int3"], "num_segments":3},
        # {"opName": "unsorted_segment_sum", "outName": "unsorted_segment/unsorted_segment_sum_rank3", "varShapes":[[6,3,2], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "uniform_int3"], "num_segments":3},
        # {"opName": "unsorted_segment_sqrt_n", "outName": "unsorted_segment/unsorted_segment_sqrt_n_rank3", "varShapes":[[6,3,2], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "uniform_int3"], "num_segments":3},
        #
        # {"opName":"truncatemod", "outName":"truncatemod/1_4", "varShapes":[[],[4]], "varTypes":["float32", "float32"], "varInit":["stdnormal", "stdnormal"]},
        # {"opName":"truncatemod", "outName":"truncatemod/4_4", "varShapes":[[4],[4]], "varTypes":["float32", "float32"], "varInit":["stdnormal", "stdnormal"]},
        # {"opName":"truncatemod", "outName":"truncatemod/3,4_3,1", "varShapes":[[3,4],[3,1]], "varTypes":["float32", "float32"], "varInit":["stdnormal", "stdnormal"]},
        #
        # #axes=1: equivalent to mmul, as is axes=[[1],[0]]
        # {"opName":"tensordot", "outName":"tensordot/4,3_3,2_a1", "varShapes":[[4,3],[3,2]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axes":1},
        # {"opName":"tensordot", "outName":"tensordot/emptyArrayTest/4,3_3,2_a1", "varShapes":[[0,3],[3,0]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"], "axes":0},
        # {"opName":"tensordot", "outName":"tensordot/4,3_3,2_a1-0", "varShapes":[[4,3],[3,2]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axes":[[1],[0]]},
        # {"opName":"tensordot", "outName":"tensordot/4,3_2,3_a1-1", "varShapes":[[4,3],[2,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axes":[[1],[1]]},
        # {"opName":"tensordot", "outName":"tensordot/3,4_2,3_a1-1", "varShapes":[[4,3],[2,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axes":[[1],[1]]},
        # {"opName":"tensordot", "outName":"tensordot/4,3,2_2,5,4_a2,0-0,2", "varShapes":[[4,3,2],[2,5,4]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axes":[[2,0],[0,2]]},

        # {"opName":"assert_equal", "outName":"assert_equal/scalar_float32", "varShapes":[[],[]], "varTypes":["float32", "float32"], "varInit":["one", "one"]},
        # {"opName":"assert_equal", "outName":"assert_equal/scalar_int32", "varShapes":[[],[]], "varTypes":["int32", "int32"], "varInit":["one", "one"]},
        # {"opName":"assert_equal", "outName":"assert_equal/scalar_rank1_float32", "varShapes":[[],[1]], "varTypes":["float32", "float32"], "varInit":["two", "two"]},
        # {"opName":"assert_equal", "outName":"assert_equal/scalar_rank1_int32", "varShapes":[[1],[]], "varTypes":["int32", "int32"], "varInit":["one", "one"]},
        # {"opName":"assert_equal", "outName":"assert_equal/3,4_3,4_float32", "varShapes":[[3,4],[3,4]], "varTypes":["float32", "float32"], "varInit":["four", "four"]},
        # {"opName":"assert_equal", "outName":"assert_equal/3,4_1,4_int32", "varShapes":[[3,4],[1,4]], "varTypes":["int32", "int32"], "varInit":["three", "three"]},

        # {"opName":"assert_greater", "outName":"assert_greater/scalar_float32", "varShapes":[[],[]], "varTypes":["float32", "float32"], "varInit":["two", "one"]},
        # {"opName":"assert_greater", "outName":"assert_greater/scalar_int32", "varShapes":[[],[]], "varTypes":["int32", "int32"], "varInit":["two", "one"]},
        # {"opName":"assert_greater", "outName":"assert_greater/scalar_rank1_float32", "varShapes":[[],[1]], "varTypes":["float32", "float32"], "varInit":["three", "two"]},
        # {"opName":"assert_greater", "outName":"assert_greater/scalar_rank1_int32", "varShapes":[[1],[]], "varTypes":["int32", "int32"], "varInit":["one", "zero"]},
        # {"opName":"assert_greater", "outName":"assert_greater/3,4_3,4_float32", "varShapes":[[3,4],[3,4]], "varTypes":["float32", "float32"], "varInit":["four", "one"]},
        # {"opName":"assert_greater", "outName":"assert_greater/3,4_1,4_int32", "varShapes":[[3,4],[1,4]], "varTypes":["int32", "int32"], "varInit":["three", "one"]},
        #
        # {"opName":"assert_greater_equal", "outName":"assert_greater_equal/scalar_float32", "varShapes":[[],[]], "varTypes":["float32", "float32"], "varInit":["two", "one"]},
        # {"opName":"assert_greater_equal", "outName":"assert_greater_equal/scalar_int32", "varShapes":[[],[]], "varTypes":["int32", "int32"], "varInit":["one", "one"]},
        # {"opName":"assert_greater_equal", "outName":"assert_greater_equal/scalar_rank1_float32", "varShapes":[[],[1]], "varTypes":["float32", "float32"], "varInit":["three", "two"]},
        # {"opName":"assert_greater_equal", "outName":"assert_greater_equal/scalar_rank1_int32", "varShapes":[[1],[]], "varTypes":["int32", "int32"], "varInit":["one", "zero"]},
        # {"opName":"assert_greater_equal", "outName":"assert_greater_equal/3,4_3,4_float32", "varShapes":[[3,4],[3,4]], "varTypes":["float32", "float32"], "varInit":["four", "four"]},
        # {"opName":"assert_greater_equal", "outName":"assert_greater_equal/3,4_1,4_int32", "varShapes":[[3,4],[1,4]], "varTypes":["int32", "int32"], "varInit":["three", "one"]},
        #
        # {"opName":"assert_less", "outName":"assert_less/scalar_float32", "varShapes":[[],[]], "varTypes":["float32", "float32"], "varInit":["zero", "one"]},
        # {"opName":"assert_less", "outName":"assert_less/scalar_int32", "varShapes":[[],[]], "varTypes":["int32", "int32"], "varInit":["one", "two"]},
        # {"opName":"assert_less", "outName":"assert_less/scalar_rank1_float32", "varShapes":[[],[1]], "varTypes":["float32", "float32"], "varInit":["zero", "two"]},
        # {"opName":"assert_less", "outName":"assert_less/scalar_rank1_int32", "varShapes":[[1],[]], "varTypes":["int32", "int32"], "varInit":["one", "three"]},
        # {"opName":"assert_less", "outName":"assert_less/3,4_3,4_float32", "varShapes":[[3,4],[3,4]], "varTypes":["float32", "float32"], "varInit":["two", "four"]},
        # {"opName":"assert_less", "outName":"assert_less/3,4_1,4_int32", "varShapes":[[3,4],[1,4]], "varTypes":["int32", "int32"], "varInit":["three", "four"]},
        #
        # {"opName":"assert_less_equal", "outName":"assert_less_equal/scalar_float32", "varShapes":[[],[]], "varTypes":["float32", "float32"], "varInit":["zero", "one"]},
        # {"opName":"assert_less_equal", "outName":"assert_less_equal/scalar_int32", "varShapes":[[],[]], "varTypes":["int32", "int32"], "varInit":["one", "one"]},
        # {"opName":"assert_less_equal", "outName":"assert_less_equal/scalar_rank1_float32", "varShapes":[[],[1]], "varTypes":["float32", "float32"], "varInit":["two", "two"]},
        # {"opName":"assert_less_equal", "outName":"assert_less_equal/scalar_rank1_int32", "varShapes":[[1],[]], "varTypes":["int32", "int32"], "varInit":["one", "three"]},
        # {"opName":"assert_less_equal", "outName":"assert_less_equal/3,4_3,4_float32", "varShapes":[[3,4],[3,4]], "varTypes":["float32", "float32"], "varInit":["two", "four"]},
        # {"opName":"assert_less_equal", "outName":"assert_less_equal/3,4_1,4_int32", "varShapes":[[3,4],[1,4]], "varTypes":["int32", "int32"], "varInit":["two", "three"]},
        #
        # {"opName":"assert_none_equal", "outName":"assert_none_equal/scalar_float32", "varShapes":[[],[]], "varTypes":["float32", "float32"], "varInit":["one", "two"]},
        # {"opName":"assert_none_equal", "outName":"assert_none_equal/scalar_int32", "varShapes":[[],[]], "varTypes":["int32", "int32"], "varInit":["two", "one"]},
        # {"opName":"assert_none_equal", "outName":"assert_none_equal/scalar_rank1_float32", "varShapes":[[],[1]], "varTypes":["float32", "float32"], "varInit":["two", "three"]},
        # {"opName":"assert_none_equal", "outName":"assert_none_equal/scalar_rank1_int32", "varShapes":[[1],[]], "varTypes":["int32", "int32"], "varInit":["one", "four"]},
        # {"opName":"assert_none_equal", "outName":"assert_none_equal/3,4_3,4_float32", "varShapes":[[3,4],[3,4]], "varTypes":["float32", "float32"], "varInit":["zero", "four"]},
        # {"opName":"assert_none_equal", "outName":"assert_none_equal/3,4_1,4_int32", "varShapes":[[3,4],[1,4]], "varTypes":["int32", "int32"], "varInit":["three", "zero"]},

        # {"opName":"assert_integer", "outName":"assert_integer/scalar_int32", "varShapes":[[]], "varTypes":["int32"], "varInit":["one"]},
        # {"opName":"assert_integer", "outName":"assert_integer/scalar_int64", "varShapes":[[]], "varTypes":["int64"], "varInit":["two"]},
        # {"opName":"assert_integer", "outName":"assert_integer/scalar_rank1_int32", "varShapes":[[3]], "varTypes":["int32"], "varInit":["three"]},
        # {"opName":"assert_integer", "outName":"assert_integer/3,4_3,4_int64", "varShapes":[[3,4]], "varTypes":["int64"], "varInit":["four"]},
        #
        # {"opName":"assert_negative", "outName":"assert_negative/scalar_int32", "varShapes":[[]], "varTypes":["int32"], "varInit":["minus_one"]},
        # {"opName":"assert_negative", "outName":"assert_negative/scalar_int64", "varShapes":[[]], "varTypes":["int64"], "varInit":["minus_two"]},
        # {"opName":"assert_negative", "outName":"assert_negative/scalar_rank1_float32", "varShapes":[[3]], "varTypes":["float32"], "varInit":["uniform_m1_0"]},
        # {"opName":"assert_negative", "outName":"assert_negative/3,4_3,4_float64", "varShapes":[[3,4]], "varTypes":["float64"], "varInit":["uniform_m1_0"]},

        # {"opName":"assert_positive", "outName":"assert_positive/scalar_float32", "varShapes":[[]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName":"assert_positive", "outName":"assert_positive/scalar_int64", "varShapes":[[]], "varTypes":["int64"], "varInit":["two"]},
        # {"opName":"assert_positive", "outName":"assert_positive/rank1_int32", "varShapes":[[3]], "varTypes":["int32"], "varInit":["uniform_int3"]},
        # {"opName":"assert_positive", "outName":"assert_positive/3,4_3,4_int64", "varShapes":[[3,4]], "varTypes":["int64"], "varInit":["uniform"]},

        # {"opName":"assert_rank", "outName":"assert_rank/rank0_int32", "varShapes":[[], []], "varTypes":["int32", "int32"], "varInit":["one", "zero"]},
        # {"opName":"assert_rank", "outName":"assert_rank/rank1_float64", "varShapes":[[3], []], "varTypes":["float64", "int32"], "varInit":["uniform", "one"]},
        # {"opName":"assert_rank", "outName":"assert_rank/rank2_float32", "varShapes":[[2,3], []], "varTypes":["float32", "int32"], "varInit":["three", "two"]},

        # {"opName":"assert_rank_at_least", "outName":"assert_rank_at_least/rank0_int32", "varShapes":[[], []], "varTypes":["int32", "int32"], "varInit":["one", "zero"]},
        # {"opName":"assert_rank_at_least", "outName":"assert_rank_at_least/rank1_float64", "varShapes":[[3], []], "varTypes":["float64", "int32"], "varInit":["uniform", "zero"]},
        # {"opName":"assert_rank_at_least", "outName":"assert_rank_at_least/rank1_float32", "varShapes":[[3], []], "varTypes":["float32", "int32"], "varInit":["uniform", "one"]},
        # {"opName":"assert_rank_at_least", "outName":"assert_rank_at_least/rank2_float32", "varShapes":[[2,3], []], "varTypes":["float32", "int32"], "varInit":["three", "two"]},
        # {"opName":"assert_rank_at_least", "outName":"assert_rank_at_least/rank2_float64", "varShapes":[[2,3], []], "varTypes":["float64", "int32"], "varInit":["three", "one"]},
        # {"opName":"assert_rank_at_least", "outName":"assert_rank_at_least/rank2_float32", "varShapes":[[2,3], []], "varTypes":["float32", "int32"], "varInit":["three", "zero"]},

        # {"opName":"assert_type", "outName":"assert_type/rank0_int32", "varShapes":[[]], "varTypes":["int32"], "varInit":["one"], "tf_type":tf.int32},
        # {"opName":"assert_type", "outName":"assert_type/rank1_float32", "varShapes":[[3]], "varTypes":["float32"], "varInit":["uniform"], "tf_type":tf.float32},
        # {"opName":"assert_type", "outName":"assert_type/rank2_float32", "varShapes":[[2,3]], "varTypes":["float32"], "varInit":["three"], "tf_type":tf.float32},
        # {"opName":"assert_type", "outName":"assert_type/rank2_int64", "varShapes":[[2,2]], "varTypes":["int64"], "varInit":["zero"], "tf_type":tf.int64},

        # {"opName":"cond", "outName":"cond/cond_true", "varShapes":[[]], "varTypes":["bool"], "varInit":["booleanTrue"]},
        # {"opName":"cond", "outName":"cond/cond_false", "varShapes":[[]], "varTypes":["bool"], "varInit":["booleanFalse"]},

        # {"opName":"case", "outName":"case/cond_1", "varShapes":[[]], "varTypes":["float32"], "varInit":["zero"]},
        # {"opName":"case", "outName":"case/cond_1b", "varShapes":[[]], "varTypes":["float32"], "varInit":["one"]},
        # {"opName":"case", "outName":"case/cond_2", "varShapes":[[]], "varTypes":["float32"], "varInit":["two"]},
        # {"opName":"case", "outName":"case/cond_3", "varShapes":[[]], "varTypes":["float32"], "varInit":["three"]},
        # {"opName":"case", "outName":"case/cond_default", "varShapes":[[]], "varTypes":["float32"], "varInit":["four"]},

        # {"opName":"while1", "outName":"while1/iter_1", "varShapes":[[]], "varTypes":["float32"], "varInit":["one"]},
        # {"opName":"while1", "outName":"while1/iter_3", "varShapes":[[]], "varTypes":["float32"], "varInit":["three"]},

        # {"opName":"while2", "outName":"while2/a", "varShapes":[[2],[2]], "varTypes":["float32", "float32"], "varInit":["one", "fixed_5_3"]},
        # {"opName":"while2", "outName":"while2/b", "varShapes":[[2,3],[2,3]], "varTypes":["float32", "float32"], "varInit":["two", "ten"]},

        #Reductions with dynamic reduce dimensions
        # {"opName":"sum_dynamic_axis", "outName":"reduce_dynamic_axis/sum_rank2_argmin_shape", "varShapes":[[3,4]], "varTypes":["float32"], "varInit":["uniform"], "axistype":"argmin", "keepdims":False},
        # {"opName":"sum_dynamic_axis", "outName":"reduce_dynamic_axis/sum_rank2_argmax_shape", "varShapes":[[3,4]], "varTypes":["float32"], "varInit":["uniform"], "axistype":"argmax", "keepdims":True}


        #TensorArray - get and set ops
        # {"opName":"tensorarray_identity", "outName":"tensor_array/getset_sz1_float32_nodynamic_noname_noshape", "varShapes":[[2,3]], "varTypes":["float32"], "varInit":["uniform"],\
        #     "dtype":tf.float32, "size":1, "dynamic_size":False, "tensor_array_name":None, "element_shape":None},
        # {"opName":"tensorarray_getset", "outName":"tensor_array/getset_sz1_int64_nodynamic_noname_shape2-3", "varShapes":[[2,3]], "varTypes":["int64"], "varInit":["uniform_int5"], \
        #  "dtype":tf.int64, "size":1, "dynamic_size":False, "tensor_array_name":None, "element_shape":[2,3]},
        # {"opName":"tensorarray_getset", "outName":"tensor_array/getset_sz2_float64_nodynamic_name_noshape", "varShapes":[[2,3],[2,3]], "varTypes":["float64","float64"], "varInit":["uniform","uniform"], \
        #  "dtype":tf.float64, "size":2, "dynamic_size":False, "tensor_array_name":None, "element_shape":None},
        # #Next test: initially size 1, but dynamic so add 3
        # {"opName":"tensorarray_getset", "outName":"tensor_array/getset_sz3-1_int32_dynamic_name_shape", "varShapes":[[2,3],[2,3],[2,3]], "varTypes":["int32","int32","int32"], "varInit":["uniform_int10","uniform_int10","uniform_int10"], \
        #  "dtype":tf.int32, "size":1, "dynamic_size":True, "tensor_array_name":None, "element_shape":[2,3]},

        #TensorArray - size
        # {"opName":"tensorarray_size", "outName":"tensor_array/size_sz1_float32_nodynamic_noname_noshape", "varShapes":[[2,3]], "varTypes":["float32"], "varInit":["uniform"], \
        #  "dtype":tf.float32, "size":1, "dynamic_size":False, "tensor_array_name":None, "element_shape":None},
        # {"opName":"tensorarray_size", "outName":"tensor_array/size_sz1_int64_nodynamic_noname_shape2-3", "varShapes":[[2,3]], "varTypes":["int64"], "varInit":["uniform_int5"], \
        #  "dtype":tf.int64, "size":1, "dynamic_size":False, "tensor_array_name":None, "element_shape":[2,3]},
        # {"opName":"tensorarray_size", "outName":"tensor_array/size_sz2_float64_nodynamic_name_noshape", "varShapes":[[2,3],[2,3]], "varTypes":["float64","float64"], "varInit":["uniform","uniform"], \
        #  "dtype":tf.float64, "size":2, "dynamic_size":False, "tensor_array_name":None, "element_shape":None},
        # {"opName":"tensorarray_size", "outName":"tensor_array/size_sz3-1_int32_dynamic_name_shape", "varShapes":[[2,3],[2,3],[2,3]], "varTypes":["int32","int32","int32"], "varInit":["uniform_int10","uniform_int10","uniform_int10"], \
        #  "dtype":tf.int32, "size":1, "dynamic_size":True, "tensor_array_name":None, "element_shape":[2,3]},

        #TensorArray - Concat (note that shapes must match, but first dim can differ - but not possible in practice, just gives inconsistent shapes exception???)
        # {"opName":"tensorarray_concat", "outName":"tensor_array/concat_sz1_float32_nodynamic_noname_noshape", "varShapes":[[2,3]], "varTypes":["float32"], "varInit":["uniform"], \
        #  "dtype":tf.float32, "size":1, "dynamic_size":False, "tensor_array_name":None, "element_shape":None},
        # {"opName":"tensorarray_concat", "outName":"tensor_array/concat_sz1_int64_nodynamic_noname_shape2-3", "varShapes":[[2,3]], "varTypes":["int64"], "varInit":["uniform_int5"], \
        #  "dtype":tf.int64, "size":1, "dynamic_size":False, "tensor_array_name":None, "element_shape":[2,3]},
        # {"opName":"tensorarray_concat", "outName":"tensor_array/concat_sz2_float64_nodynamic_name_noshape", "varShapes":[[2,3],[2,3]], "varTypes":["float64","float64"], "varInit":["uniform","uniform"], \
        #  "dtype":tf.float64, "size":2, "dynamic_size":False, "tensor_array_name":None, "element_shape":None},
        # {"opName":"tensorarray_concat", "outName":"tensor_array/concat_sz3-1_int32_dynamic_name_shape", "varShapes":[[2,3],[2,3],[2,3]], "varTypes":["int32","int32","int32"], "varInit":["uniform_int10","uniform_int10","uniform_int10"], \
        #  "dtype":tf.int32, "size":1, "dynamic_size":True, "tensor_array_name":None, "element_shape":[2,3], "infer_shape":False},

        #TensorArray - Stack
        # {"opName":"tensorarray_stack", "outName":"tensor_array/stack_sz1_float32_nodynamic_noname_noshape", "varShapes":[[2,3]], "varTypes":["float32"], "varInit":["uniform"], \
        #  "dtype":tf.float32, "size":1, "dynamic_size":False, "tensor_array_name":None, "element_shape":None},
        # {"opName":"tensorarray_stack", "outName":"tensor_array/stack_sz1_int64_nodynamic_noname_shape2-3", "varShapes":[[2,3]], "varTypes":["int64"], "varInit":["uniform_int5"], \
        #  "dtype":tf.int64, "size":1, "dynamic_size":False, "tensor_array_name":None, "element_shape":[2,3]},
        # {"opName":"tensorarray_stack", "outName":"tensor_array/stack_sz2_float64_nodynamic_name_noshape", "varShapes":[[2,3],[2,3]], "varTypes":["float64","float64"], "varInit":["uniform","uniform"], \
        #  "dtype":tf.float64, "size":2, "dynamic_size":False, "tensor_array_name":None, "element_shape":None},
        # {"opName":"tensorarray_stack", "outName":"tensor_array/stack_sz3-1_int32_dynamic_name_shape", "varShapes":[[2,3],[2,3],[2,3]], "varTypes":["int32","int32","int32"], "varInit":["uniform_int10","uniform_int10","uniform_int10"], \
        #  "dtype":tf.int32, "size":1, "dynamic_size":True, "tensor_array_name":None, "element_shape":[2,3], "infer_shape":False},

        #TensorArray - Unstack
         #{"opName":"tensorarray_unstack", "outName":"tensor_array/unstack_sz1_float32_nodynamic_noname_noshape", "varShapes":[[2,3]], "varTypes":["float32"], "varInit":["uniform"], \
          #"dtype":tf.float32, "size":1, "dynamic_size":False, "tensor_array_name":None, "element_shape":None},
         #{"opName":"tensorarray_unstack", "outName":"tensor_array/unstack_sz1_int64_nodynamic_noname_shape2-3", "varShapes":[[2,3]], "varTypes":["int64"], "varInit":["uniform_int5"], \
          #"dtype":tf.int64, "size":1, "dynamic_size":False, "tensor_array_name":None, "element_shape":[2,3]},
         #{"opName":"tensorarray_unstack", "outName":"tensor_array/unstack_sz2_float64_nodynamic_name_noshape", "varShapes":[[2,3],[2,3]], "varTypes":["float64","float64"], "varInit":["uniform","uniform"], \
         # "dtype":tf.float64, "size":2, "dynamic_size":False, "tensor_array_name":None, "element_shape":None},
         #{"opName":"tensorarray_unstack", "outName":"tensor_array/unstack_sz3-1_int32_dynamic_name_shape", "varShapes":[[2,3],[2,3],[2,3]], "varTypes":["int32","int32","int32"], "varInit":["uniform_int10","uniform_int10","uniform_int10"], \
         # "dtype":tf.int32, "size":1, "dynamic_size":True, "tensor_array_name":None, "element_shape":[2,3], "infer_shape":False},

        #TensorArray - identity (set, identity, get)
        # {"opName":"tensorarray_identity", "outName":"tensor_array/identity_sz1_float32_nodynamic_noname_noshape", "varShapes":[[2,3]], "varTypes":["float32"], "varInit":["uniform"],\
        #     "dtype":tf.float32, "size":1, "dynamic_size":False, "tensor_array_name":None, "element_shape":None},
        # {"opName":"tensorarray_identity", "outName":"tensor_array/identity_sz1_int64_nodynamic_noname_shape2-3", "varShapes":[[2,3]], "varTypes":["int64"], "varInit":["uniform_int5"], \
        #  "dtype":tf.int64, "size":1, "dynamic_size":False, "tensor_array_name":None, "element_shape":[2,3]},
        # {"opName":"tensorarray_identity", "outName":"tensor_array/identity_sz2_float64_nodynamic_name_noshape", "varShapes":[[2,3],[2,3]], "varTypes":["float64","float64"], "varInit":["uniform","uniform"], \
        #  "dtype":tf.float64, "size":2, "dynamic_size":False, "tensor_array_name":None, "element_shape":None},
        # #Next test: initially size 1, but dynamic so add 3
        # {"opName":"tensorarray_identity", "outName":"tensor_array/identity_sz3-1_int32_dynamic_name_shape", "varShapes":[[2,3],[2,3],[2,3]], "varTypes":["int32","int32","int32"], "varInit":["uniform_int10","uniform_int10","uniform_int10"], \
        #  "dtype":tf.int32, "size":1, "dynamic_size":True, "tensor_array_name":None, "element_shape":[2,3]},

        # #TensorArray - Split (basically, standard Split op + tensor array scatter  - i.e., split then put results into TensorArray)
        # # In these tests, first variable is the values, second is the lengths of each split. Note that size of each has to be same size...
        # {"opName":"tensorarray_split", "outName":"tensor_array/split_sz1_float32_nodynamic_noname_noshape", "varShapes":[[2,3],[1]], "varTypes":["float32", "int32"], "varInit":["uniform", "two"],\
        #     "dtype":tf.float32, "size":1, "dynamic_size":False, "tensor_array_name":None, "element_shape":None},
        # {"opName":"tensorarray_split", "outName":"tensor_array/split_sz2_float64_nodynamic_name_noshape", "varShapes":[[4,3],[2]], "varTypes":["float64","int32"], "varInit":["uniform","two"], \
        #  "dtype":tf.float64, "size":2, "dynamic_size":False, "tensor_array_name":None, "element_shape":None},
        # {"opName":"tensorarray_split", "outName":"tensor_array/split_sz3-1_int32_dynamic_name_shape", "varShapes":[[9,3],[3]], "varTypes":["int32","int32"], "varInit":["uniform_int10","three"], \
        #  "dtype":tf.int32, "size":1, "dynamic_size":True, "tensor_array_name":None, "element_shape":[3,3]},

        # TensorArray - close
        # {"opName":"tensorarray_close", "outName":"tensor_array/close_sz1_float32_nodynamic_noname_noshape", "varShapes":[[2,3]], "varTypes":["float32"], "varInit":["uniform"],\
        #     "dtype":tf.float32, "size":1, "dynamic_size":False, "tensor_array_name":None, "element_shape":None},
        # {"opName":"tensorarray_close", "outName":"tensor_array/close_sz1_int64_nodynamic_noname_shape2-3", "varShapes":[[2,3]], "varTypes":["int64"], "varInit":["uniform_int5"], \
        #  "dtype":tf.int64, "size":1, "dynamic_size":False, "tensor_array_name":None, "element_shape":[2,3]},
        # {"opName":"tensorarray_close", "outName":"tensor_array/close_sz2_float64_nodynamic_name_noshape", "varShapes":[[2,3],[2,3]], "varTypes":["float64","float64"], "varInit":["uniform","uniform"], \
        #  "dtype":tf.float64, "size":2, "dynamic_size":False, "tensor_array_name":None, "element_shape":None},

        #ExtractImagePatches
        # {"opName":"extractImagePatches", "outName":"extractImagePatches/sz1-6-6-2_float32_k3_s1_r1_SAME", "varShapes":[[1,6,6,2]], "varTypes":["float32"], "varInit":["uniform"],\
        #     "ksizes":[1,3,3,1], "strides":[1,1,1,1], "rates":[1,1,1,1], "padding":"SAME"},
        # {"opName":"extractImagePatches", "outName":"extractImagePatches/sz1-6-6-2_float32_k3_s1_r1_VALID", "varShapes":[[1,6,6,2]], "varTypes":["float32"], "varInit":["uniform"], \
        #  "ksizes":[1,3,3,1], "strides":[1,1,1,1], "rates":[1,1,1,1], "padding":"VALID"},
        # {"opName":"extractImagePatches", "outName":"extractImagePatches/sz1-8-8-2_int32_k2_s2_r1_SAME", "varShapes":[[1,8,8,2]], "varTypes":["int32"], "varInit":["uniform_int10"], \
        #  "ksizes":[1,2,2,1], "strides":[1,2,2,1], "rates":[1,1,1,1], "padding":"SAME"},
        # {"opName":"extractImagePatches", "outName":"extractImagePatches/sz1-8-8-2_int64_k2_s1_r2_SAME", "varShapes":[[1,8,8,2]], "varTypes":["int64"], "varInit":["uniform_int10"], \
        #  "ksizes":[1,2,2,1], "strides":[1,1,1,1], "rates":[1,2,2,1], "padding":"SAME"},

        #Stop gradient op
        # {"opName":"stopGradient", "outName":"stopGradient/rank0", "varShapes":[[]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName":"stopGradient", "outName":"stopGradient/rank1", "varShapes":[[3]], "varTypes":["float64"], "varInit":["uniform"]},
        # {"opName":"stopGradient", "outName":"stopGradient/rank2", "varShapes":[[3,4]], "varTypes":["float64"], "varInit":["uniform"]}

        #Note: For RNNs, TF uses [batch, seqLength, nIn]
        #LSTM - Static
        # {"opName":"lstmcell", "outName":"rnn/lstmcell/static_batch1_n5-3_tsLength4_noPH_noClip_fBias1_Tanh_noInitState_float", "varShapes":[[1,4,5]], "varTypes":["float32"], "varInit":["uniform"], "static":True, "timeSteps":4,\
        #     "num_units":3, "use_peepholes":False, "cell_clip":None, "proj_clip":None, "forget_bias":1.0, "activation":"tanh", "dtype":tf.float32},
        # {"opName":"lstmcell", "outName":"rnn/lstmcell/static_batch2_nIn2_nOut3_tsLength4_withPH_noClip_fBias1_Tanh_noInitState_double", "varShapes":[[2,4,2]], "varTypes":["float64"], "varInit":["uniform"], "static":True, "timeSteps":4, \
        #  "num_units":3, "use_peepholes":True, "cell_clip":None, "proj_clip":None, "forget_bias":1.0, "activation":"tanh", "dtype":tf.float64},
        # {"opName":"lstmcell", "outName":"rnn/lstmcell/static_batch1_n5-3_tsLength4_noPH_clip-0.3-0.4_fBias1_Tanh_noInitState_float", "varShapes":[[1,4,5]], "varTypes":["float32"], "varInit":["uniform"], "static":True, "timeSteps":4, \
        #  "num_units":3, "use_peepholes":False, "cell_clip":0.3, "proj_clip":0.4, "forget_bias":1.0, "activation":"tanh", "dtype":tf.float32},
        # {"opName":"lstmcell", "outName":"rnn/lstmcell/static_batch1_nIn5_nOut3_tsLength4_noPH_noClip_fBias1_Softsign_withInitState_float", "varShapes":[[1,4,5],[1,3],[1,3]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"],
        #     "static":True, "timeSteps":4, "num_units":3, "use_peepholes":False, "cell_clip":None, "proj_clip":None, "forget_bias":1.0, "activation":"softsign", "dtype":tf.float32},

        #LSTM - Dynamic. Supports time_major: if true, [max_time, batch_size, depth]; If false [batch_size, max_time, depth]
        # {"opName":"lstmcell", "outName":"rnn/lstmcell/dynamic_b1_nIn5_nOut3_ts4_noPH_noClip_fB1_Tanh_noInitState_float_noTM", "varShapes":[[1,4,5]], "varTypes":["float32"], "varInit":["uniform"], "static":False, "timeSteps":4,
        #     "num_units":3, "use_peepholes":False, "cell_clip":None, "proj_clip":None, "forget_bias":1.0, "activation":"tanh", "dtype":tf.float32, "time_major":False},
        # {"opName":"lstmcell", "outName":"rnn/lstmcell/dynamic_b1_nIn5_nOut3_ts4_noPH_noClip_fB1_Tanh_noIS_float_withTM", "varShapes":[[4,1,5]], "varTypes":["float32"], "varInit":["uniform"], "static":False, "timeSteps":4,
        #  "num_units":3, "use_peepholes":False, "cell_clip":None, "proj_clip":None, "forget_bias":1.0, "activation":"tanh", "dtype":tf.float32, "time_major":False},
        # {"opName":"lstmcell", "outName":"rnn/lstmcell/dynamic_b2_nIn2_nOut3_ts4_withPH_noClip_fB1_Tanh_noIS_double_noTM", "varShapes":[[2,4,2]], "varTypes":["float64"], "varInit":["uniform"], "static":False, "timeSteps":4,
        #  "num_units":3, "use_peepholes":True, "cell_clip":None, "proj_clip":None, "forget_bias":1.0, "activation":"tanh", "dtype":tf.float64, "time_major":False},
        # {"opName":"lstmcell", "outName":"rnn/lstmcell/dynamic_b1_nIn5_nOut3_ts4_noPH_clip-0.3-0.4_fB1_Tanh_noIS_float_noTM", "varShapes":[[1,4,5]], "varTypes":["float32"], "varInit":["uniform"], "static":False, "timeSteps":4,
        #  "num_units":3, "use_peepholes":False, "cell_clip":0.3, "proj_clip":0.4, "forget_bias":1.0, "activation":"tanh", "dtype":tf.float32, "time_major":False},
        # {"opName":"lstmcell", "outName":"rnn/lstmcell/dynamic_b1_nIn5_nOut3_ts4_noPH_noClip_fB2_Softsign_withIS_float_noTM", "varShapes":[[1,4,5],[1,3],[1,3]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"],
        #     "static":False, "timeSteps":4, "num_units":3, "use_peepholes":False, "cell_clip":None, "proj_clip":None, "forget_bias":2.0, "activation":"softsign", "dtype":tf.float32, "time_major":False},

        #BasicRNNCell - Static
        # {"opName":"basicrnncell", "outName":"rnn/basicrnncell/static_b1_nIn5_nOut3_ts4_tanh_noIS_float", "varShapes":[[1,4,5]], "varTypes":["float32"], "varInit":["uniform"], "static":True, "timeSteps":4,
        #      "num_units":3, "activation":"tanh", "dtype":tf.float32},
        # {"opName":"basicrnncell", "outName":"rnn/basicrnncell/static_b1_nIn5_nOut3_ts4_sigmoid_withIS_double", "varShapes":[[1,4,5], [1,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "static":True, "timeSteps":4,
        #     "num_units":3, "activation":"sigmoid", "dtype":tf.float64},

        #BasicRNNCell - dynamic
        # {"opName":"basicrnncell", "outName":"rnn/basicrnncell/dynamic_b1_nIn5_nOut3_ts4_relu_noIS_noTM_float", "varShapes":[[1,4,5]], "varTypes":["float32"], "varInit":["uniform"], "static":False, "timeSteps":4,
        #      "num_units":3, "activation":"relu", "dtype":tf.float32, "time_major":False},
        # {"opName":"basicrnncell", "outName":"rnn/basicrnncell/dynamic_b1_nIn5_nOut3_ts4_relu_noIS_withTM_float", "varShapes":[[4,1,5]], "varTypes":["float32"], "varInit":["uniform"], "static":False, "timeSteps":4,
        #  "num_units":3, "activation":"relu", "dtype":tf.float32, "time_major":True},
        # {"opName":"basicrnncell", "outName":"rnn/basicrnncell/dynamic_b1_nIn5_nOut3_ts4_sigmoid_withIS_noTM_double", "varShapes":[[1,4,5], [1,3]], "varTypes":["float64", "float64"], "varInit":["uniform", "uniform"], "static":False, "timeSteps":4,
        #  "num_units":3, "activation":"softsign", "dtype":tf.float64, "time_major":False},

        #BasicLSTMCell - Static
        # {"opName":"basiclstmcell", "outName":"rnn/basiclstmcell/static_b1_nIn5_nOut3_ts4_tanh_noIS_fb1_float", "varShapes":[[1,4,5]], "varTypes":["float32"], "varInit":["uniform"], "static":True, "timeSteps":4,
        #      "num_units":3, "activation":"tanh", "dtype":tf.float32, "forget_bias":1.0},
        # {"opName":"basiclstmcell", "outName":"rnn/basiclstmcell/static_b1_nIn5_nOut3_ts4_sigmoid_withIS_fb2_double", "varShapes":[[1,4,5], [1,3], [1,3]], "varTypes":["float32", "float32", "float32"], "varInit":["uniform", "uniform", "uniform"], "static":True, "timeSteps":4,
        #     "num_units":3, "activation":"sigmoid", "dtype":tf.float64, "forget_bias":2.0},

        #BasicLSTMCell - dynamic
        # {"opName":"basiclstmcell", "outName":"rnn/basiclstmcell/dynamic_b1_nIn5_nOut3_ts4_tanh_noIS_noTM_fb1_float", "varShapes":[[1,4,5]], "varTypes":["float32"], "varInit":["uniform"], "static":False, "timeSteps":4,
        #      "num_units":3, "activation":"tanh", "dtype":tf.float32, "time_major":False, "forget_bias":1.0},
        # {"opName":"basiclstmcell", "outName":"rnn/basiclstmcell/dynamic_b1_nIn5_nOut3_ts4_tanh_noIS_withTM_fb1_float", "varShapes":[[4,1,5]], "varTypes":["float32"], "varInit":["uniform"], "static":False, "timeSteps":4,
        #  "num_units":3, "activation":"tanh", "dtype":tf.float32, "time_major":True, "forget_bias":1.0},
        # {"opName":"basiclstmcell", "outName":"rnn/basiclstmcell/dynamic_b1_nIn5_nOut3_ts4_softsign_withIS_noTM_fb2_double", "varShapes":[[1,4,5], [1,3], [1,3]], "varTypes":["float64", "float64", "float64"], "varInit":["uniform", "uniform", "uniform"], "static":False, "timeSteps":4,
        #  "num_units":3, "activation":"softsign", "dtype":tf.float64, "time_major":False, "forget_bias":2.0},

        #GRUCell - Static
        # {"opName":"grucell", "outName":"rnn/grucell/static_b1_nIn5_nOut3_ts4_tanh_noIS_float", "varShapes":[[1,4,5]], "varTypes":["float32"], "varInit":["uniform"], "static":True, "timeSteps":4,
        #      "num_units":3, "activation":"tanh", "dtype":tf.float32},
        # {"opName":"grucell", "outName":"rnn/grucell/static_b1_nIn5_nOut3_ts4_softsign_withIS_double", "varShapes":[[1,4,5], [1,3]], "varTypes":["float64", "float64"], "varInit":["uniform", "uniform"], "static":True, "timeSteps":4,
        #     "num_units":3, "activation":"softsign", "dtype":tf.float64},

        #GRUCell - dynamic
        # {"opName":"grucell", "outName":"rnn/grucell/dynamic_b1_nIn5_nOut3_ts4_relu_noIS_noTM_float", "varShapes":[[1,4,5]], "varTypes":["float32"], "varInit":["uniform"], "static":False, "timeSteps":4,
        #      "num_units":3, "activation":"relu", "dtype":tf.float32, "time_major":False},
        # {"opName":"grucell", "outName":"rnn/grucell/dynamic_b1_nIn5_nOut3_ts4_relu_noIS_withTM_float", "varShapes":[[4,1,5]], "varTypes":["float32"], "varInit":["uniform"], "static":False, "timeSteps":4,
        #  "num_units":3, "activation":"relu", "dtype":tf.float32, "time_major":True},
        # {"opName":"grucell", "outName":"rnn/grucell/dynamic_b1_nIn5_nOut3_ts4_sigmoid_withIS_noTM_double", "varShapes":[[1,4,5], [1,3]], "varTypes":["float64", "float64"], "varInit":["uniform", "uniform"], "static":False, "timeSteps":4,
        #  "num_units":3, "activation":"softsign", "dtype":tf.float64, "time_major":False},

        #GRUBlockCellV2 - Static
        #Note: GRUBlockCellV2: "Only differs from GRUBlockCell by variable names." - GRUBlockCell is deprecated
        # {"opName":"grublockcellv2", "outName":"rnn/grublockcellv2/static_b1_n5-3_ts4_noIS_f32", "varShapes":[[1,4,5]], "varTypes":["float32"], "varInit":["uniform"], "static":True, "timeSteps":4,
        #      "num_units":3, "dtype":tf.float32},
        # {"opName":"grublockcellv2", "outName":"rnn/grublockcellv2/static_b1_n5-3_ts4_withIS_f32", "varShapes":[[1,4,5], [1,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "static":True, "timeSteps":4,
        #     "num_units":3, "dtype":tf.float32},

        #GRUBlockCellV2 - dynamic
        # {"opName":"grublockcellv2", "outName":"rnn/grublockcellv2/dynamic_b1_n3-2_ts1_noIS_noTM", "varShapes":[[1,1,3]], "varTypes":["float32"], "varInit":["uniform"], "static":False, "timeSteps":1,
        #      "num_units":2, "dtype":tf.float32, "time_major":False},
        # {"opName":"grublockcellv2", "outName":"rnn/grublockcellv2/dynamic_b1_n5-3_ts4_noIS_noTM", "varShapes":[[1,4,5]], "varTypes":["float32"], "varInit":["uniform"], "static":False, "timeSteps":4,
        #      "num_units":3, "dtype":tf.float32, "time_major":False},
        # {"opName":"grublockcellv2", "outName":"rnn/grublockcellv2/dynamic_b1_n5-3_ts4_noIS_withTM", "varShapes":[[4,1,5]], "varTypes":["float32"], "varInit":["uniform"], "static":False, "timeSteps":4,
        #  "num_units":3, "dtype":tf.float32, "time_major":True},

        #LSTMBlockCell - Static. Note: float32 and float16 only
        # {"opName":"lstmblockcell", "outName":"rnn/lstmblockcell/static_batch1_n3-2_tsLength1_noPH_noClip_fBias1_noIS", "varShapes":[[1,1,3]], "varTypes":["float32"], "varInit":["uniform"], "static":True, "timeSteps":1,
        #      "num_units":2, "use_peepholes":False, "cell_clip":None, "forget_bias":1.0, "dtype":tf.float32},
        # {"opName":"lstmblockcell", "outName":"rnn/lstmblockcell/static_batch1_n5-3_tsLength4_noPH_noClip_fBias1_noIS", "varShapes":[[1,4,5]], "varTypes":["float32"], "varInit":["uniform"], "static":True, "timeSteps":4,
        #     "num_units":3, "use_peepholes":False, "cell_clip":None, "forget_bias":1.0, "dtype":tf.float32},
        # {"opName":"lstmblockcell", "outName":"rnn/lstmblockcell/static_batch2_n2-3_tsLength4_withPH_noClip_fBias1_noIS", "varShapes":[[2,4,2]], "varTypes":["float32"], "varInit":["uniform"], "static":True, "timeSteps":4,
        #  "num_units":3, "use_peepholes":True, "cell_clip":None, "forget_bias":1.0, "dtype":tf.float32},
        # {"opName":"lstmblockcell", "outName":"rnn/lstmblockcell/static_batch1_n5-3_tsLength4_noPH_clip-0.3_fBias2_noIS", "varShapes":[[1,4,5]], "varTypes":["float32"], "varInit":["uniform"], "static":True, "timeSteps":4,
        #  "num_units":3, "use_peepholes":False, "cell_clip":0.3, "forget_bias":2.0, "dtype":tf.float32},

        #LSTMBlockCell - Dynamic. Supports time_major: if true, [max_time, batch_size, depth]; If false [batch_size, max_time, depth]
        # {"opName":"lstmblockcell", "outName":"rnn/lstmblockcell/dynamic_b1_n5-3_ts4_noPH_noClip_fB1_noIS_noTM", "varShapes":[[1,4,5]], "varTypes":["float32"], "varInit":["uniform"], "static":False, "timeSteps":4,
        #     "num_units":3, "use_peepholes":False, "cell_clip":None, "forget_bias":1.0, "dtype":tf.float32, "time_major":False},
        # {"opName":"lstmblockcell", "outName":"rnn/lstmblockcell/dynamic_b1_n5-3_ts4_noPH_noClip_fB1_noIS_withTM", "varShapes":[[4,1,5]], "varTypes":["float32"], "varInit":["uniform"], "static":False, "timeSteps":4,
        #  "num_units":3, "use_peepholes":False, "cell_clip":None, "proj_clip":None, "forget_bias":1.0, "dtype":tf.float32, "time_major":False},
        # {"opName":"lstmblockcell", "outName":"rnn/lstmblockcell/dynamic_b1_n5-3_ts4_noPH_clip-0.3-0.4_fB1_Tanh_noIS_noTM", "varShapes":[[1,4,5]], "varTypes":["float32"], "varInit":["uniform"], "static":False, "timeSteps":4,
        #  "num_units":3, "use_peepholes":False, "cell_clip":0.3, "proj_clip":0.4, "forget_bias":1.0, "dtype":tf.float32, "time_major":False},
        # {"opName":"lstmblockcell", "outName":"rnn/lstmblockcell/dynamic_b1_n5-3_ts4_noPH_noClip_fB2_withIS_noTM", "varShapes":[[1,4,5],[1,3],[1,3]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"],
        #     "static":False, "timeSteps":4, "num_units":3, "use_peepholes":False, "cell_clip":None, "proj_clip":None, "forget_bias":2.0, "dtype":tf.float32, "time_major":False},

        #SRUCell - Static
        # {"opName":"srucell", "outName":"rnn/srucell/static_b1_n5-3_tanh_ts4_noIS_f32", "varShapes":[[1,4,5]], "varTypes":["float32"], "varInit":["uniform"], "static":True, "timeSteps":4,
        #      "num_units":3, "dtype":tf.float32, "activation":tf.nn.tanh},
        # {"opName":"srucell", "outName":"rnn/srucell/static_b1_n5-3_relu_ts4_withIS_f32", "varShapes":[[1,4,5], [1,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "static":True, "timeSteps":4,
        #     "num_units":3, "dtype":tf.float32, "activation":tf.nn.relu},

        #SRUCell - dynamic
        # {"opName":"srucell", "outName":"rnn/srucell/dynamic_b1_n5-3_tanh_ts4_noIS_noTM", "varShapes":[[1,4,5]], "varTypes":["float32"], "varInit":["uniform"], "static":False, "timeSteps":4,
        #      "num_units":3, "dtype":tf.float32, "time_major":False, "activation":tf.nn.tanh},
        # {"opName":"srucell", "outName":"rnn/srucell/dynamic_b1_n5-3_elu_ts4_noIS_withTM", "varShapes":[[4,1,5]], "varTypes":["float32"], "varInit":["uniform"], "static":False, "timeSteps":4,
        #  "num_units":3, "dtype":tf.float32, "time_major":True, "activation":tf.nn.elu},

        #LSTMBlockFusedCell. Note: these don't use rnn static/dynamic, the whole RNN is one op. Also note they expect [time,batch,inSize] inputs only (not configurable)
        # {"opName":"lstmblockfusedcell", "outName":"rnn/lstmblockfusedcell/batch1_n3-2_tsLength1_noPH_noClip_fBias1_noIS", "varShapes":[[3,1,1]], "varTypes":["float32"], "varInit":["uniform"], "static":True, "timeSteps":1,
        #     "num_units":2, "use_peephole":False, "cell_clip":None, "forget_bias":1.0, "dtype":tf.float32},
        # {"opName":"lstmblockfusedcell", "outName":"rnn/lstmblockfusedcell/batch1_n5-3_tsLength4_noPH_noClip_fBias1_noIS", "varShapes":[[5,1,4]], "varTypes":["float32"], "varInit":["uniform"], "static":True, "timeSteps":4,
        #     "num_units":3, "use_peephole":False, "cell_clip":None, "forget_bias":1.0, "dtype":tf.float32},
        # {"opName":"lstmblockfusedcell", "outName":"rnn/lstmblockfusedcell/batch2_n2-3_tsLength4_withPH_noClip_fBias1_noIS", "varShapes":[[4,2,2]], "varTypes":["float32"], "varInit":["uniform"], "static":True, "timeSteps":4,
        #  "num_units":3, "use_peephole":True, "cell_clip":None, "forget_bias":1.0, "dtype":tf.float32},
        # {"opName":"lstmblockfusedcell", "outName":"rnn/lstmblockfusedcell/batch1_n5-3_tsLength4_noPH_clip-0.3_fBias2_withIS", "varShapes":[[5,1,4], [1,3], [1,3]], "varTypes":["float32", "float32", "float32"], "varInit":["uniform", "uniform", "uniform"], "static":True, "timeSteps":4,
        #  "num_units":3, "use_peephole":False, "cell_clip":0.3, "forget_bias":2.0, "dtype":tf.float32},

        # Bidirectional dynamic RNN + BasicRNNCell
        # {"opName":"bidirectional_basicrnncell", "outName":"rnn/bidir_basic/static_b1_nIn5_nOut3_ts4_tanh_noIS_float", "varShapes":[[1,4,5]], "varTypes":["float32"], "varInit":["uniform"], "static":True, "timeSteps":4,
        #      "num_units":3, "activation":"tanh", "dtype":tf.float32},
        # {"opName":"bidirectional_basicrnncell", "outName":"rnn/bidir_basic/static_b1_nIn5_nOut3_ts4_sigmoid_withIS_double", "varShapes":[[1,4,5], [1,3], [1,3]], "varTypes":["float32", "float32", "float32"], "varInit":["uniform", "uniform", "uniform"], "static":True, "timeSteps":4,
        #     "num_units":3, "activation":"sigmoid", "dtype":tf.float64},

        # Bidirectional static RNN + BasicRNNCell
        # {"opName":"bidirectional_basicrnncell", "outName":"rnn/bidir_basic/dynamic_b1_n5-3_ts4_relu_noIS_noTM_f32", "varShapes":[[1,4,5]], "varTypes":["float32"], "varInit":["uniform"], "static":False, "timeSteps":4,
        #      "num_units":3, "activation":"relu", "dtype":tf.float32, "time_major":False},
        # {"opName":"bidirectional_basicrnncell", "outName":"rnn/bidir_basic/dynamic_b1_n5-3_ts4_relu_noIS_withTM_f32", "varShapes":[[4,1,5]], "varTypes":["float32"], "varInit":["uniform"], "static":False, "timeSteps":4,
        #  "num_units":3, "activation":"relu", "dtype":tf.float32, "time_major":True},
        # {"opName":"bidirectional_basicrnncell", "outName":"rnn/bidir_basic/dynamic_b1_n5-3_ts4_sig_withIS_noTM_f64", "varShapes":[[1,4,5], [1,3], [1,3]], "varTypes":["float64", "float64", "float64"], "varInit":["uniform", "uniform", "uniform"], "static":False, "timeSteps":4,
        #  "num_units":3, "activation":"softsign", "dtype":tf.float64, "time_major":False},


        #TimeReversedFusedRNN + LSTMBlockFusedCell. Note: these don't use rnn static/dynamic, the whole RNN is one op. Also note they expect [time,batch,inSize] inputs only (not configurable)
        # {"opName":"timereversed_lstmblockfusedcell", "outName":"rnn/tr_lstmbfc/batch1_n5-3_tsLength4_noPH_noClip_fBias1_noIS", "varShapes":[[5,1,4]], "varTypes":["float32"], "varInit":["uniform"], "static":True, "timeSteps":4,
        #     "num_units":3, "use_peephole":False, "cell_clip":None, "forget_bias":1.0, "dtype":tf.float32},
        # {"opName":"timereversed_lstmblockfusedcell", "outName":"rnn/tr_lstmbfc/batch2_n2-3_tsLength4_withPH_noClip_fBias1_noIS", "varShapes":[[4,2,2]], "varTypes":["float32"], "varInit":["uniform"], "static":True, "timeSteps":4,
        #  "num_units":3, "use_peephole":True, "cell_clip":None, "forget_bias":1.0, "dtype":tf.float32},
        # {"opName":"timereversed_lstmblockfusedcell", "outName":"rnn/tr_lstmbfc/batch1_n5-3_tsLength4_noPH_clip-0.3_fBias2_withIS", "varShapes":[[5,1,4], [1,3], [1,3]], "varTypes":["float32", "float32", "float32"], "varInit":["uniform", "uniform", "uniform"], "static":True, "timeSteps":4,
        #  "num_units":3, "use_peephole":False, "cell_clip":0.3, "forget_bias":2.0, "dtype":tf.float32},

        # FusedRNNCellAdaptor + BasicRNNCell. Again, fused uses [time,batch,inSize]
        # {"opName":"fused_adaptor_basicrnncell", "outName":"rnn/fused_adapt_basic/static_b1_n5-3_ts4_tanh_noIS_float", "varShapes":[[5,1,4]], "varTypes":["float32"], "varInit":["uniform"], "timeSteps":4,
        #      "num_units":3, "activation":"tanh", "dtype":tf.float32, "use_dynamic_rnn":False},
        # {"opName":"fused_adaptor_basicrnncell", "outName":"rnn/fused_adapt_basic/static_b1_n5-3_ts4_sigmoid_withIS_double", "varShapes":[[5,1,4], [1,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "timeSteps":4,
        #     "num_units":3, "activation":"sigmoid", "dtype":tf.float64, "use_dynamic_rnn":False},
        # {"opName":"fused_adaptor_basicrnncell", "outName":"rnn/fused_adapt_basic/dynamic_b1_n5-3_ts4_relu_noIS_float", "varShapes":[[5,1,4]], "varTypes":["float32"], "varInit":["uniform"], "timeSteps":4,
        #      "num_units":3, "activation":"relu", "dtype":tf.float32, "use_dynamic_rnn":True},
        # {"opName":"fused_adaptor_basicrnncell", "outName":"rnn/fused_adapt_basic/dynamic_b1_n5-3_ts4_elu_withIS_double", "varShapes":[[5,1,4], [1,3]], "varTypes":["float64", "float64"], "varInit":["uniform", "uniform"], "timeSteps":4,
        #     "num_units":3, "activation":"elu", "dtype":tf.float64, "use_dynamic_rnn":True},

        # Stacked Bidirectional dynamic RNN + BasicRNNCell
        # {"opName":"stack_bidir_basicrnncell", "outName":"rnn/bstack/sta_b1_n5-3_ts4_tanh_noIS_f32_n3", "varShapes":[[1,4,5]], "varTypes":["float32"], "varInit":["uniform"], "static":True, "timeSteps":4,
        #      "num_units":3, "activation":"tanh", "dtype":tf.float32, "size":3},
        # {"opName":"stack_bidir_basicrnncell", "outName":"rnn/bstack/sta_b1_n5-3_ts4_sig_IS_f64_n2", "varShapes":[[1,4,5], [1,3], [1,3], [1,3], [1,3]], "varTypes":["float32", "float32", "float32", "float32", "float32"], "varInit":["uniform", "uniform", "uniform", "uniform", "uniform"], "static":True, "timeSteps":4,
        #     "num_units":3, "activation":"sigmoid", "dtype":tf.float64, "size":2},

        # Stacked Bidirectional static RNN + BasicRNNCell
        # {"opName":"stack_bidir_basicrnncell", "outName":"rnn/bstack/d_b1_n3", "varShapes":[[1,4,5]], "varTypes":["float32"], "varInit":["uniform"], "static":False, "timeSteps":4,
        #      "num_units":3, "activation":"relu", "dtype":tf.float32, "time_major":False, "size":3},
        # {"opName":"stack_bidir_basicrnncell", "outName":"rnn/bstack/d_n2", "varShapes":[[4,1,5], [1,3], [1,3], [1,3], [1,3]], "varTypes":["float32", "float32", "float32", "float32", "float32"], "varInit":["uniform", "uniform", "uniform", "uniform", "uniform"], "static":False, "timeSteps":4,
        #     "num_units":3, "activation":"relu", "dtype":tf.float32, "time_major":True, "size":2},

        # {"opName": "arg_max", "outName": "arg_max/rank1_dim0", "varShapes":[[4]], "varTypes":["float32"], "varInit":["uniform"], "dimension":0},
        # {"opName": "arg_max", "outName": "arg_max/rank2_dim1", "varShapes":[[0,1]], "varTypes":["float32"], "varInit":["uniform"], "dimension":1},
        # {"opName": "arg_min", "outName": "arg_min/rank1_dim0", "varShapes":[[3]], "varTypes":["float32"], "varInit":["uniform"], "dimension":0},
        # {"opName": "arg_min", "outName": "arg_min/rank2_dim1", "varShapes":[[2,3]], "varTypes":["float32"], "varInit":["uniform"], "dimension":1},

        # {"opName": "expand_dims", "outName": "expand_dims/rank1_axis0", "varShapes":[[1]], "varTypes":["float32"], "varInit":["uniform"], "axis":0},
        # {"opName": "expand_dims", "outName": "expand_dims/rank1_axis-1", "varShapes":[[2]], "varTypes":["float32"], "varInit":["uniform"], "axis":-1},
        # {"opName": "expand_dims", "outName": "expand_dims/rank1_axis1", "varShapes":[[3]], "varTypes":["float32"], "varInit":["uniform"], "axis":1},
        # {"opName": "expand_dims", "outName": "expand_dims/rank2_axis0", "varShapes":[[1,2]], "varTypes":["float32"], "varInit":["uniform"], "axis":0},
        # {"opName": "expand_dims", "outName": "expand_dims/rank2_axis-1", "varShapes":[[2,3]], "varTypes":["float32"], "varInit":["uniform"], "axis":-1},
        # {"opName": "expand_dims", "outName": "expand_dims/rank2_axis1", "varShapes":[[3,1]], "varTypes":["float32"], "varInit":["uniform"], "axis":1},
        # {"opName": "expand_dims", "outName": "expand_dims/rank2_axis2", "varShapes":[[1,3]], "varTypes":["float32"], "varInit":["uniform"], "axis":2},

        # {"opName": "fill", "outName": "fill/fill_3-1_val3", "varShapes":[[2], []], "varTypes":["int32", "float"], "varInit":["fixed_3_1", "three"]},

        # {"opName": "identity", "outName": "identity/identity_rank0", "varShapes":[[]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName": "identity", "outName": "identity/identity_rank1", "varShapes":[[3]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName": "identity", "outName": "identity/identity_rank2", "varShapes":[[2,3]], "varTypes":["int32"], "varInit":["uniform_int10"]},

        # {"opName": "gather", "outName": "gather/gather_3-3-1_2_axis0", "varShapes":[[3,3,1], [2]], "varTypes":["float32", "int32"], "varInit":["uniform", "uniform_int3"], "axis":0},
        # {"opName": "gather", "outName": "gather/gather_4-4_1_axis1", "varShapes":[[4,4], [0]], "varTypes":["float32", "int32"], "varInit":["uniform", "uniform_int3"], "axis":1},
        # {"opName": "gather", "outName": "gather/gather_2-2-3_3_axis-1", "varShapes":[[2,2,3], [3]], "varTypes":["float32", "int32"], "varInit":["uniform", "uniform_int3"], "axis":-1},

        # {"opName": "one", "outName": "ones/ones_rank0", "varShapes":[[1]], "varTypes":["int32"], "varInit":["one"], "dtype":tf.int64},
        # {"opName": "one", "outName": "ones/ones_rank2", "varShapes":[[2]], "varTypes":["int32"], "varInit":["fixed_2_1"], "dtype":tf.float64},

        # {"opName": "ones_like", "outName": "ones_like/rank0", "varShapes":[[]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName": "ones_like", "outName": "ones_like/rank2", "varShapes":[[2,2]], "varTypes":["float64"], "varInit":["uniform"]},

        # {"opName": "reverse", "outName": "reverse/rank1", "varShapes":[[3]], "varTypes":["float32"], "varInit":["uniform"], "axis":[0]},
        # {"opName": "reverse", "outName": "reverse/rank2_axis0", "varShapes":[[3,4]], "varTypes":["float32"], "varInit":["uniform"], "axis":[0]},
        # {"opName": "reverse", "outName": "reverse/rank2_axis1", "varShapes":[[3,4]], "varTypes":["float32"], "varInit":["uniform"], "axis":[1]},
        # {"opName": "reverse", "outName": "reverse/rank3_axis0-2", "varShapes":[[2,3,4]], "varTypes":["float32"], "varInit":["uniform"], "axis":[0,2]},


        #{"opName": "fake_quant_with_min_max_vars", "outName": "fake_quant/min_max_vars/rank1_0_1_8bit", "varShapes":[[10],[],[]], "varTypes":["float32", "float32", "float32"], "varInit":["uniform", "zero", "one"], "num_bits":8, "narrow_range":False},
        #{"opName": "fake_quant_with_min_max_vars", "outName": "fake_quant/min_max_vars/rank1_0_1_8bit_narrow", "varShapes":[[10],[],[]], "varTypes":["float32", "float32", "float32"], "varInit":["uniform", "zero", "one"], "num_bits":8, "narrow_range":True},
        #{"opName": "fake_quant_with_min_max_vars", "outName": "fake_quant/min_max_vars/rank1_0_1_4bit", "varShapes":[[10],[],[]], "varTypes":["float32", "float32", "float32"], "varInit":["uniform", "zero", "one"], "num_bits":4, "narrow_range":False},
        #{"opName": "fake_quant_with_min_max_vars", "outName": "fake_quant/min_max_vars/rank2_0_2_8bit", "varShapes":[[4,5],[],[]], "varTypes":["float32", "float32", "float32"], "varInit":["uniform", "zero", "two"], "num_bits":8, "narrow_range":False},
        #{"opName": "fake_quant_with_min_max_vars", "outName": "fake_quant/min_max_vars/rank1_1_5_4bit_narrow", "varShapes":[[5,5],[],[]], "varTypes":["float32", "float32", "float32"], "varInit":["uniform10", "one", "five"], "num_bits":4, "narrow_range":True},
        #
        #{"opName": "fake_quant_with_min_max_args", "outName": "fake_quant/min_max_args/rank1_0_1_8bit", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform"], "num_bits":8, "narrow_range":False, "min":0.0, "max":1.0},
        #{"opName": "fake_quant_with_min_max_args", "outName": "fake_quant/min_max_args/rank1_0_1_8bit_narrow", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform"], "num_bits":8, "narrow_range":True, "min":0.0, "max":1.0},
        #{"opName": "fake_quant_with_min_max_args", "outName": "fake_quant/min_max_args/rank1_0_1_4bit", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform"], "num_bits":4, "narrow_range":False, "min":0.0, "max":1.0},
        #{"opName": "fake_quant_with_min_max_args", "outName": "fake_quant/min_max_args/rank2_0_2_8bit", "varShapes":[[4,5]], "varTypes":["float32"], "varInit":["uniform"], "num_bits":8, "narrow_range":False, "min":0.0, "max":2.0},
        #{"opName": "fake_quant_with_min_max_args", "outName": "fake_quant/min_max_args/rank1_1_5_4bit_narrow", "varShapes":[[5,5]], "varTypes":["float32"], "varInit":["uniform10"], "num_bits":4, "narrow_range":True, "min":1.0, "max":5.0},

        # {"opName": "fake_quant_with_min_max_vars_per_channel", "outName": "fake_quant/min_max_args_per_channel/rank1_8bit", "varShapes":[[5], [5], [5]], "varTypes":["float32", "float32", "float32"], "varInit":["uniform", "uniform_m1_0", "uniform"], "num_bits":8, "narrow_range":False},
        # {"opName": "fake_quant_with_min_max_vars_per_channel", "outName": "fake_quant/min_max_args_per_channel/rank2_8bit_narrow", "varShapes":[[3, 5], [5], [5]], "varTypes":["float32", "float32", "float32"], "varInit":["uniform", "uniform_m1_0", "uniform"], "num_bits":8, "narrow_range":True},
        # {"opName": "fake_quant_with_min_max_vars_per_channel", "outName": "fake_quant/min_max_args_per_channel/rank4_6bit", "varShapes":[[3, 2, 2, 5], [5], [5]], "varTypes":["float32", "float32", "float32"], "varInit":["uniform", "uniform_m1_0", "uniform"], "num_bits":6, "narrow_range":False},

         #{"opName": "adjust_saturation", "outName": "adjust_saturation/rank3_float32", "varShapes": [[8, 8, 3]], "varTypes": ["float32"], "varInit": ["uniform"], "factor": 2.0},
         #{"opName": "adjust_saturation", "outName": "adjust_saturation/rank4_float32", "varShapes": [[8, 8, 2, 3]], "varTypes": ["float32"], "varInit": ["uniform"], "factor": 0.5},

         #{"opName": "adjust_contrast", "outName": "adjust_contrast_v2/rank3_float32", "varShapes": [[8,8,3]],"varTypes": ["float32"], "varInit": ["uniform"],"contrast_factor":2.0},
         #{"opName": "adjust_contrast", "outName": "adjust_contrast_v2/rank4_float32", "varShapes": [[8,8,3,1]],"varTypes": ["float32"], "varInit": ["uniform"],"contrast_factor":2.0},
         #{"opName": "adjust_contrast", "outName": "adjust_contrast_v2/rank4_float64","varShapes": [[8, 8, 3, 4]], "varTypes": ["float64"], "varInit": ["uniform"],"contrast_factor":2.0},
         #{"opName": "adjust_contrast", "outName": "adjust_contrast_v2/rank3_float32", "varShapes": [[8, 8, 3]], "varTypes": ["float32"], "varInit": ["uniform"], "contrast_factor": 2.0},
         #{"opName": "adjust_contrast", "outName": "adjust_contrast_v2/rank3_float64", "varShapes": [[8, 8, 3]],  "varTypes": ["float64"], "varInit": ["uniform"], "contrast_factor": 2.0},
         #{"opName": "adjust_contrast", "outName": "adjust_contrast_v2/emptyArrayTests/rank3_float32", "varShapes": [[0,0,0]],"varTypes": ["float32"], "varInit": ["empty"],"contrast_factor":0.0},

        #{"opName": "adjust_hue", "outName": "adjust_hue/rank3_float32", "varShapes": [[8, 8, 3]], "varTypes": ["float32"], "varInit": ["stdnormal"], "delta": 0.5},
        #{"opName": "adjust_hue", "outName": "adjust_hue/rank3_float64", "varShapes": [[8, 8, 3]], "varTypes": ["float64"], "varInit": ["uniform"], "delta": 0.5},
        #{"opName": "adjust_hue", "outName": "adjust_hue/rank3_float64_zero_delta", "varShapes": [[8, 8, 3]], "varTypes": ["float64"], "varInit": ["uniform"], "delta": 0.5},
        # {"opName": "adjust_hue", "outName": "adjust_hue/rank3_float32_test", "varShapes": [[8, 8, 3]], "varTypes": ["float32"], "varInit": ["stdnormal"], "delta": 0.5},

         #{"opName": "crop_and_resize", "outName": "crop_and_resize", "varShapes": [[1,2,2,1], [2,4], [2], [2]],"varTypes": ["float32", "float32","int32","int32"], "varInit": ["uniform", "uniform","uniform_int10","uniform_int10"], "method":"bilinear","ext_value":0},
         #{"opName": "crop_and_resize", "outName": "crop_and_resize", "varShapes": [[1,2,2,1], [2,4], [2], [2]],"varTypes": ["float32", "float32","int32","int32"], "varInit": ["uniform", "uniform","uniform_int10","uniform_int10"], "method":"nearest","ext_value":0},
         # {"opName": "crop_and_resize", "outName": "crop_and_resize", "varShapes": [[1, 2, 2, 1], [2, 4], [2], [2]], "varTypes": ["float32", "float32", "int32", "int32"], "varInit": ["uniform", "uniform", "uniform_int10", "uniform_int10"], "method": "nearest", "ext_value": 0.5},

         #{"opName": "draw_bounding_boxes", "outName": "draw_bounding_boxes/float32_input", "varShapes": [[2,5,5,1], [2,2,4], [1,2]],"varTypes": ["float32", "float32","float32"], "varInit": ["uniform", "uniform","uniform"]},
         #{"opName": "draw_bounding_boxes", "outName": "draw_bounding_boxes/half_input", "varShapes": [[2,5,5,1], [2,2,4],[1,2]],"varTypes": ["half", "float32","float32"], "varInit": ["uniform", "uniform","uniform"]},

         #{"opName": "resize_bilinear", "outName": "resize_bilinear/float32", "varShapes": [[2,5,5,3], [2]],"varTypes": ["float32", "int32"], "varInit": ["uniform", "uniform_int10"], "align_corners":True, "half_pixel_centers":False},
         #{"opName": "resize_bilinear", "outName": "resize_bilinear/float64", "varShapes": [[2, 5, 5, 1], [2]],  "varTypes": ["float64", "int32"], "varInit": ["uniform", "uniform_int10"],"align_corners":True, "half_pixel_centers":False},
         #{"opName": "resize_bilinear", "outName": "resize_bilinear/int32", "varShapes": [[2, 5, 5, 1], [2]], "varTypes": ["int32", "int32"], "varInit": ["uniform_int10", "uniform_int10"],"align_corners":False, "half_pixel_centers":True},
         #{"opName": "resize_bilinear", "outName": "resize_bilinear/float32_1", "varShapes": [[2, 5, 5, 3], [2]],   "varTypes": ["float32", "int32"], "varInit": ["uniform", "uniform_int10"],"align_corners":False, "half_pixel_centers":False},

         #{"opName": "resize_nearest_neighbor", "outName": "resize_nearest_neighbor/float32", "varShapes": [[2,5,5,3], [2]],"varTypes": ["float32", "int32"], "varInit": ["uniform", "uniform_int10"], "align_corners":True, "half_pixel_centers":False},
         #{"opName": "resize_nearest_neighbor", "outName": "resize_nearest_neighbor/float64", "varShapes": [[2,5,5,3], [2]],"varTypes": ["float64", "int32"], "varInit": ["uniform", "uniform_int10"],"align_corners":True, "half_pixel_centers":False},
         #{"opName": "resize_nearest_neighbor", "outName": "resize_nearest_neighbor/int32", "varShapes": [[2, 5, 5, 1], [2]], "varTypes": ["int32", "int32"], "varInit": ["uniform_int10", "uniform_int10"],"align_corners":False, "half_pixel_centers":True},
         #{"opName": "resize_nearest_neighbor", "outName": "resize_nearest_neighbor/float32_1", "varShapes": [[2, 5, 5, 3], [2]],   "varTypes": ["float32", "int32"], "varInit": ["uniform", "uniform_int10"],"align_corners":False, "half_pixel_centers":False},

        # {"opName": "resize_bicubic", "outName": "resize_bicubic/float32", "varShapes": [[2,5,5,3],[2]],"varTypes": ["float32","int32"], "varInit": ["uniform","uniform_int10"], "align_corners":True, "half_pixel_centers":False},
        # {"opName": "resize_bicubic", "outName": "resize_bicubic/float64", "varShapes": [[2, 5, 5, 1],[2]],  "varTypes": ["float64","int32"], "varInit": ["uniform","uniform_int10"], "size":10, "align_corners":True, "half_pixel_centers":False},
        # {"opName": "resize_bicubic", "outName": "resize_bicubic/int32", "varShapes": [[2, 5, 5, 1],[2]], "varTypes": ["int32","int32"], "varInit": ["uniform_int10","uniform_int10"], "size":0, "align_corners":False, "half_pixel_centers":True},
        # {"opName": "resize_bicubic", "outName": "resize_bicubic/float32_1", "varShapes": [[2, 5, 5, 3],[2]],   "varTypes": ["float32","int32"], "varInit": ["uniform","uniform_int10"], "size":100, "align_corners":False, "half_pixel_centers":False},

        # TODO: more tests
        # {"opName": "resize_area", "outName": "resize_area/float32", "varShapes": [[2,5,5,3],[2]],"varTypes": ["float32","int32"], "varInit": ["uniform","uniform_int10"], "align_corners":True},
        # {"opName": "resize_area", "outName": "resize_area/float64", "varShapes": [[2, 5, 5, 1],[2]],  "varTypes": ["float64","int32"], "varInit": ["uniform","uniform_int10"], "size":10, "align_corners":True},
        # {"opName": "resize_area", "outName": "resize_area/int32", "varShapes": [[2, 5, 5, 1],[2]], "varTypes": ["int32","int32"], "varInit": ["uniform_int10","uniform_int10"], "size":0, "align_corners":False},
        # {"opName": "resize_area", "outName": "resize_area/float32_1", "varShapes": [[2, 5, 5, 3],[2]],   "varTypes": ["float32","int32"], "varInit": ["uniform","uniform_int10"], "size":100, "align_corners":False},
        # {"opName": "resize_area", "outName": "resize_area/zeroArrayTest/float32", "varShapes": [[2, 5, 5, 3], [2]], "varTypes": ["float32", "int32"], "varInit": ["zeros", "uniform_int10"], "align_corners": True},

        # {"opName": "check_numerics", "outName": "check_numerics/rank1_float16", "varShapes":[[5]], "varTypes":["float16", "string"], "varInit":["uniform", "string_scalar"], "message":"This is a test string."},
        # {"opName": "check_numerics", "outName": "check_numerics/rank1_float32", "varShapes":[[5]], "varTypes":["float32", "string"], "varInit":["uniform", "string_scalar"], "message":"This is a test string."},
        # {"opName": "check_numerics", "outName": "check_numerics/rank2_float64", "varShapes":[[5]], "varTypes":["float64", "string"], "varInit":["uniform", "string_scalar"], "message":"This is a test string."},
        # {"opName": "check_numerics", "outName": "check_numerics/rank4_bfloat16", "varShapes":[[5]], "varTypes":["bfloat16", "string"], "varInit":["uniform", "string_scalar"], "message":"This is a test string."}

        # {"opName": "dropout", "outName": "dropout", "varShapes":[[1,100]], "varTypes":["float32"], "varInit":["uniform"]}



        #####################################################################################################################################
        # Empty array tests
        # {"opName": "arg_max", "outName": "emptyArrayTests/arg_max/rank1_dim0", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "dimension":0},
        # {"opName": "arg_max", "outName": "emptyArrayTests/arg_max/rank2_dim0", "varShapes":[[0,1]], "varTypes":["float32"], "varInit":["empty"], "dimension":0},
        # {"opName": "arg_min", "outName": "emptyArrayTests/arg_min/rank1_dim0", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "dimension":0},
        # {"opName": "arg_min", "outName": "emptyArrayTests/arg_min/rank2_dim1", "varShapes":[[2,0]], "varTypes":["float32"], "varInit":["empty"], "dimension":1},

        # "AttributeError: 'list' object has no attribute 'dtype'" ???
        # {"opName": "assign", "outName": "emptyArrayTests/assign/rank1", "varShapes":[[0], [0]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"]},
        # {"opName": "assign", "outName": "emptyArrayTests/assign/rank2", "varShapes":[[2,0], [0,3]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"]},

        # {"opName": "concat", "outName": "emptyArrayTests/concat/rank1_dim0", "varShapes":[[0], [0]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"], "axis":0},
        # {"opName": "concat", "outName": "emptyArrayTests/concat/rank2_dim0", "varShapes":[[0,1], [0,1]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"], "axis":0},
        # {"opName": "concat", "outName": "emptyArrayTests/concat/rank2_dim1", "varShapes":[[0,2], [0,2]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"], "axis":1},
        # {"opName": "concat", "outName": "emptyArrayTests/concat/rank2_dim0b", "varShapes":[[2,0], [2,0]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"], "axis":0},
        # {"opName": "concat", "outName": "emptyArrayTests/concat/rank2_dim1b", "varShapes":[[2,0], [3,0]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"], "axis":1},

        # {"opName": "expand_dims", "outName": "emptyArrayTests/expand_dims/rank1_axis0", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":0},
        # {"opName": "expand_dims", "outName": "emptyArrayTests/expand_dims/rank1_axis-1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":-1},
        # {"opName": "expand_dims", "outName": "emptyArrayTests/expand_dims/rank1_axis1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":1},
        # {"opName": "expand_dims", "outName": "emptyArrayTests/expand_dims/rank2_axis0", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "axis":0},
        # {"opName": "expand_dims", "outName": "emptyArrayTests/expand_dims/rank2_axis-1", "varShapes":[[2,0]], "varTypes":["float32"], "varInit":["empty"], "axis":-1},
        # {"opName": "expand_dims", "outName": "emptyArrayTests/expand_dims/rank2_axis1", "varShapes":[[3,0]], "varTypes":["float32"], "varInit":["empty"], "axis":1},
        # {"opName": "expand_dims", "outName": "emptyArrayTests/expand_dims/rank2_axis2", "varShapes":[[0,0]], "varTypes":["float32"], "varInit":["empty"], "axis":2},

        # {"opName": "fill", "outName": "emptyArrayTests/fill/fill_2-0_val3", "varShapes":[[2], []], "varTypes":["int32", "float"], "varInit":["fixed_2_0", "three"]},
        # {"opName": "fill", "outName": "emptyArrayTests/fill/fill_0-0-3_val3", "varShapes":[[3], []], "varTypes":["int32", "float"], "varInit":["fixed_0_0_3", "three"]},

        # {"opName": "gather", "outName": "emptyArrayTests/gather/gather_2-3_emptyIndicesR1_axis0", "varShapes":[[2,3], [0]], "varTypes":["float32", "int32"], "varInit":["uniform", "empty"], "axis":0},
        # {"opName": "gather", "outName": "emptyArrayTests/gather/gather_2-3_emptyIndicesR1_axis1", "varShapes":[[2,3], [0]], "varTypes":["float32", "int32"], "varInit":["uniform", "empty"], "axis":1},
        # {"opName": "gather", "outName": "emptyArrayTests/gather/gather_2-3-1_emptyIndicesR1_axis0", "varShapes":[[2,3,1], [0]], "varTypes":["float32", "int32"], "varInit":["uniform", "empty"], "axis":0},
        # {"opName": "gather", "outName": "emptyArrayTests/gather/gather_2-3-1_emptyIndicesR1_axis2", "varShapes":[[2,3,1], [0]], "varTypes":["float32", "int32"], "varInit":["uniform", "empty"], "axis":2},
        # {"opName": "gather", "outName": "emptyArrayTests/gather/gather_emptyR2_emptyR1_axis0", "varShapes":[[2,0], [0]], "varTypes":["float32", "int32"], "varInit":["uniform", "empty"], "axis":0},
        # {"opName": "gather", "outName": "emptyArrayTests/gather/gather_emptyR2_emptyR1_axis1", "varShapes":[[2,0], [0]], "varTypes":["float32", "int32"], "varInit":["uniform", "empty"], "axis":1},
        # {"opName": "gather", "outName": "emptyArrayTests/gather/gather_emptyR3_emptyR1_axis2", "varShapes":[[2,0,3], [0]], "varTypes":["float32", "int32"], "varInit":["uniform", "empty"], "axis":2},
        # {"opName": "gather", "outName": "emptyArrayTests/gather/gather_emptyR3_emptyR1_axis-1", "varShapes":[[2,0,3], [0]], "varTypes":["float32", "int32"], "varInit":["uniform", "empty"], "axis":-1},

        # {"opName": "identity", "outName": "emptyArrayTests/identity/identity_rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "identity", "outName": "emptyArrayTests/identity/identity_rank2", "varShapes":[[2,0]], "varTypes":["int32"], "varInit":["empty"]},
        # {"opName": "identity", "outName": "emptyArrayTests/identity/identity_rank3", "varShapes":[[0,1,0]], "varTypes":["int64"], "varInit":["empty"]},

        # {"opName": "identity_n", "outName": "emptyArrayTests/identity_n/rank1", "varShapes":[[0], [0]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"]},
        # {"opName": "identity_n", "outName": "emptyArrayTests/identity_n/rank2", "varShapes":[[2,0], [0,3]], "varTypes":["int32", "int32"], "varInit":["empty", "empty"]},
        # {"opName": "identity_n", "outName": "emptyArrayTests/identity_n/rank3", "varShapes":[[0,1,0], [0,0,2]], "varTypes":["int64", "int64"], "varInit":["empty", "empty"]},

        # {"opName": "one", "outName": "emptyArrayTests/ones/ones_rank1", "varShapes":[[1]], "varTypes":["int32"], "varInit":["zero"], "dtype":tf.int32},
        # {"opName": "one", "outName": "emptyArrayTests/ones/ones_rank2", "varShapes":[[2]], "varTypes":["int32"], "varInit":["fixed_2_0"], "dtype":tf.float32},
        # {"opName": "one", "outName": "emptyArrayTests/ones/ones_rank3", "varShapes":[[3]], "varTypes":["int32"], "varInit":["fixed_0_0_3"], "dtype":tf.int64},

        # {"opName": "ones_like", "outName": "emptyArrayTests/ones_like/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "ones_like", "outName": "emptyArrayTests/ones_like/rank2", "varShapes":[[2,0]], "varTypes":["float64"], "varInit":["empty"]},

        # {"opName": "range", "outName": "emptyArrayTests/range/0_0_1", "varShapes":[[],[],[]], "varTypes":["float32","float32","float32"], "varInit":["zero","zero","one"]},
        # {"opName": "range", "outName": "emptyArrayTests/range/2_2_2", "varShapes":[[],[],[]], "varTypes":["float32","float32","float32"], "varInit":["two","two","two"]},

        # {"opName": "rank", "outName": "emptyArrayTests/rank/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "rank", "outName": "emptyArrayTests/rank/rank2a", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "rank", "outName": "emptyArrayTests/rank/rank2b", "varShapes":[[1,0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "rank", "outName": "emptyArrayTests/rank/rank3", "varShapes":[[1,0,2]], "varTypes":["float32"], "varInit":["empty"]},

        # {"opName": "realdiv", "outName": "emptyArrayTests/realdiv/scalar_empty1", "varShapes":[[],[0]], "varTypes":["float32","float32"], "varInit":["two", "empty"]},
        # {"opName": "realdiv", "outName": "emptyArrayTests/realdiv/empty1_scalar", "varShapes":[[0],[]], "varTypes":["float32","float32"], "varInit":["empty", "two"]},
        # {"opName": "realdiv", "outName": "emptyArrayTests/realdiv/empty1_empty1", "varShapes":[[0],[0]], "varTypes":["float32","float32"], "varInit":["empty", "empty"]},
        # {"opName": "realdiv", "outName": "emptyArrayTests/realdiv/empty2_rank2", "varShapes":[[0,1],[1,2]], "varTypes":["float32","float32"], "varInit":["empty", "two"]},

        # {"opName": "reshape", "outName": "emptyArrayTests/reshape/rank2_shape2-0_0-1-2", "varShapes":[[2,0]], "varTypes":["float32"], "varInit":["empty", "empty"], "shape":[0,1,2]},
        # {"opName": "reshape", "outName": "emptyArrayTests/reshape/rank3_shape0-1-2_10-0", "varShapes":[[0,1,2]], "varTypes":["float32"], "varInit":["empty", "empty"], "shape":[10,0]},

        # {"opName": "reverse", "outName": "emptyArrayTests/reverse/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":[0]},
        # {"opName": "reverse", "outName": "emptyArrayTests/reverse/rank2_axis0", "varShapes":[[0,4]], "varTypes":["float32"], "varInit":["uniform"], "axis":[0]},
        # {"opName": "reverse", "outName": "emptyArrayTests/reverse/rank2_axis1", "varShapes":[[3,0]], "varTypes":["float32"], "varInit":["uniform"], "axis":[1]},
        # {"opName": "reverse", "outName": "emptyArrayTests/reverse/rank3_axis0-2", "varShapes":[[2,0,4]], "varTypes":["float32"], "varInit":["uniform"], "axis":[0,2]},

        # {"opName": "scatter_add", "outName": "emptyArrayTests/scatter_add/rank1_emptyIndices_emptyUpdates", "varShapes":[[10],[0],[0]], "varTypes":["float32","int32","float32"], "varInit":["uniform","empty","empty"]},
        # {"opName": "scatter_add", "outName": "emptyArrayTests/scatter_add/rank2_emptyIndices_emptyUpdates", "varShapes":[[3,4],[0],[0,4]], "varTypes":["float32","int32","float32"], "varInit":["uniform","empty","empty"]},
        # {"opName": "scatter_div", "outName": "emptyArrayTests/scatter_div/rank1_emptyIndices_emptyUpdates", "varShapes":[[10],[0],[0]], "varTypes":["float32","int32","float32"], "varInit":["uniform","empty","empty"]},
        # {"opName": "scatter_div", "outName": "emptyArrayTests/scatter_div/rank2_emptyIndices_emptyUpdates", "varShapes":[[3,4],[0],[0,4]], "varTypes":["float32","int32","float32"], "varInit":["uniform","empty","empty"]},
        # {"opName": "scatter_max", "outName": "emptyArrayTests/scatter_max/rank1_emptyIndices_emptyUpdates", "varShapes":[[10],[0],[0]], "varTypes":["float32","int32","float32"], "varInit":["uniform","empty","empty"]},
        # {"opName": "scatter_max", "outName": "emptyArrayTests/scatter_max/rank2_emptyIndices_emptyUpdates", "varShapes":[[3,4],[0],[0,4]], "varTypes":["float32","int32","float32"], "varInit":["uniform","empty","empty"]},
        # {"opName": "scatter_min", "outName": "emptyArrayTests/scatter_min/rank1_emptyIndices_emptyUpdates", "varShapes":[[10],[0],[0]], "varTypes":["float32","int32","float32"], "varInit":["uniform","empty","empty"]},
        # {"opName": "scatter_min", "outName": "emptyArrayTests/scatter_min/rank2_emptyIndices_emptyUpdates", "varShapes":[[3,4],[0],[0,4]], "varTypes":["float32","int32","float32"], "varInit":["uniform","empty","empty"]},
        # {"opName": "scatter_mul", "outName": "emptyArrayTests/scatter_mul/rank1_emptyIndices_emptyUpdates", "varShapes":[[10],[0],[0]], "varTypes":["float32","int32","float32"], "varInit":["uniform","empty","empty"]},
        # {"opName": "scatter_mul", "outName": "emptyArrayTests/scatter_mul/rank2_emptyIndices_emptyUpdates", "varShapes":[[3,4],[0],[0,4]], "varTypes":["float32","int32","float32"], "varInit":["uniform","empty","empty"]},
        # {"opName": "scatter_div", "outName": "emptyArrayTests/scatter_div/rank1_emptyIndices_emptyUpdates", "varShapes":[[10],[0],[0]], "varTypes":["float32","int32","float32"], "varInit":["uniform","empty","empty"]},
        # {"opName": "scatter_div", "outName": "emptyArrayTests/scatter_div/rank2_emptyIndices_emptyUpdates", "varShapes":[[3,4],[0],[0,4]], "varTypes":["float32","int32","float32"], "varInit":["uniform","empty","empty"]},
        # {"opName": "scatter_update", "outName": "emptyArrayTests/scatter_update/rank1_emptyIndices_emptyUpdates", "varShapes":[[10],[0],[0]], "varTypes":["float32","int32","float32"], "varInit":["uniform","empty","empty"]},
        # {"opName": "scatter_update", "outName": "emptyArrayTests/scatter_update/rank2_emptyIndices_emptyUpdates", "varShapes":[[3,4],[0],[0,4]], "varTypes":["float32","int32","float32"], "varInit":["uniform","empty","empty"]},

        # {"opName": "shape", "outName": "emptyArrayTests/shape/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "shape", "outName": "emptyArrayTests/shape/rank2a", "varShapes":[[0,3]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "shape", "outName": "emptyArrayTests/shape/rank2b", "varShapes":[[2,0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "shape", "outName": "emptyArrayTests/shape/rank3a", "varShapes":[[2,0,3]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "shape", "outName": "emptyArrayTests/shape/rank3b", "varShapes":[[0,0,3]], "varTypes":["float32"], "varInit":["empty"]},

        # {"opName": "shapen", "outName": "emptyArrayTests/shape_n/rank1-2-3", "varShapes":[[0],[1,0],[0,2,0]], "varTypes":["float32","float32","float32"], "varInit":["empty","empty","empty"]},

        # {"opName": "size", "outName": "emptyArrayTests/size/rank2", "varShapes":[[2,0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "size", "outName": "emptyArrayTests/size/rank3", "varShapes":[[0,0,3]], "varTypes":["float32"], "varInit":["empty"]},

        # {"opName": "slice", "outName": "emptyArrayTests/slice/rank1_size0", "varShapes":[[3],[1],[1]], "varTypes":["float32","int32","int32"], "varInit":["uniform","zero","zero"]},
        # {"opName": "slice", "outName": "emptyArrayTests/slice/rank2_size0", "varShapes":[[3,4],[2],[2]], "varTypes":["float32","int32","int32"], "varInit":["uniform","one","zero"]},
        # {"opName": "slice", "outName": "emptyArrayTests/slice/rank3_size0", "varShapes":[[3,4,2],[3],[3]], "varTypes":["float32","int32","int32"], "varInit":["uniform","one","zero"]},

        #Squeeze here is odd: input [2,1,0] axis 1 gives error: "Can not squeeze dim[1], expected a dimension of 1, got 2 for 'Squeeze' (op: 'Squeeze') with input shapes: [1,2,1,0]."
        # {"opName": "squeeze", "outName": "emptyArrayTests/squeeze/in2-1-0_axis2", "varShapes":[[2,1,0]], "varTypes":["float32"], "varInit":["empty"], "axis":2},
        # {"opName": "squeeze", "outName": "emptyArrayTests/squeeze/in1-0_axis1", "varShapes":[[1,0]], "varTypes":["float32"], "varInit":["empty"], "axis":1},

        # {"opName": "stack", "outName": "emptyArrayTests/stack/rank1_axis0", "varShapes":[[0], [0]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"], "axis":0},
        # {"opName": "stack", "outName": "emptyArrayTests/stack/rank2_shape0-1_axis0", "varShapes":[[0,2], [0,2]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"], "axis":0},
        # {"opName": "stack", "outName": "emptyArrayTests/stack/rank2_shape0-1_axis1", "varShapes":[[0,1], [0,1]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"], "axis":1},
        # {"opName": "stack", "outName": "emptyArrayTests/stack/rank2_shape2-0_axis0", "varShapes":[[2,0], [2,0]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"], "axis":0},
        # {"opName": "stack", "outName": "emptyArrayTests/stack/rank2_shape1-0_axis1", "varShapes":[[1,0], [1,0]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"], "axis":1},

        # {"opName": "strided_slice", "outName": "emptyArrayTests/strided_slice/in2-3-4_out2-0-2", "varShapes":[[2,3,4]], "varTypes":["float32"], "varInit":["uniform"], "begin":[0,0,0], "end":[2,0,2], "strides":[1,1,1],\
        #     "begin_mask":0, "end_mask":0},
        # {"opName": "strided_slice", "outName": "emptyArrayTests/strided_slice/in2-0-4_out2-0-2", "varShapes":[[2,0,4]], "varTypes":["float32"], "varInit":["empty"], "begin":[0,0,0], "end":[2,0,2], "strides":[1,1,1],\
        #     "begin_mask":0, "end_mask":0}
        # {"opName": "strided_slice", "outName": "emptyArrayTests/strided_slice/in2-0-4_out2-0-2_v2", "varShapes":[[2,0,4]], "varTypes":["float32"], "varInit":["empty"], "begin":[0,-1,0], "end":[2,-1,2], "strides":[1,1,1], \
        #     "begin_mask":2, "end_mask":2}
        # {"opName": "strided_slice", "outName": "strided_slice/in2-3-4_out2-0-2","varShapes": [[20, 30, 40]], "varTypes": ["float32"], "varInit": ["uniform"], "begin": [0, 0, 0],"end": [15,10, 12], "strides": [1, 1, 1],
        # "begin_mask": 0, "end_mask": 0},
        #{"opName": "strided_slice", "outName": "strided_slice/largein_largeout", "varShapes": [[200, 300, 400]],
        # "varTypes": ["float32"], "varInit": ["uniform"], "begin": [10, 10, 20], "end": [15, 10, 12], "strides": [10, 10, 10],
        # "begin_mask": 0, "end_mask": 0},
        #{"opName": "strided_slice", "outName": "strided_slice/largein_largeout_1", "varShapes": [[50, 30, 40]], "varTypes": ["float64"], "varInit": ["uniform"], "begin": [10, 10, 20], "end": [50, 5, 7],
        # "strides": [10, 10, 10],
        # "begin_mask": 0, "end_mask": 0},

        #{"opName": "strided_slice", "outName": "strided_slice/largein_largeout_2", "varShapes": [[1000, 200, 100]],
        # "varTypes": ["float64"], "varInit": ["uniform"], "begin": [50, 20, 20], "end": [50, 20, 40],
        # "strides": [20, 20, 20],
        # "begin_mask": 0, "end_mask": 0},


        #Again, somehow TF is appending a 1 here: InvalidArgumentError: Dimension must be 3 but is 2 for 'transpose' (op: 'Transpose') with input shapes: [1,2,0], [2].
        # {"opName": "transpose", "outName": "emptyArrayTests/transpose/in2-0_perm1-0", "varShapes":[[2,0]], "varTypes":["float32"], "varInit":["empty"], "perm":[1,0,2]},
        # {"opName": "transpose", "outName": "emptyArrayTests/transpose/in2-1-0_perm1-2-0", "varShapes":[[2,1,0]], "varTypes":["float32"], "varInit":["empty"], "perm":[1,2,0,3]},

        # {"opName": "unstack", "outName": "emptyArrayTests/unstack/in2-0-4_axis2", "varShapes":[[2,0,4]], "varTypes":["float32"], "varInit":["empty"], "num":4, "axis":2},
        # {"opName": "unstack", "outName": "emptyArrayTests/unstack/in3-0_axis0", "varShapes":[[3,0]], "varTypes":["float32"], "varInit":["empty"], "num":3, "axis":0},

        # {"opName": "zeros", "outName": "emptyArrayTests/zeros/ones_rank1", "varShapes":[[1]], "varTypes":["int32"], "varInit":["zero"], "dtype":tf.int32},
        # {"opName": "zeros", "outName": "emptyArrayTests/zeros/ones_rank2", "varShapes":[[2]], "varTypes":["int32"], "varInit":["fixed_2_0"], "dtype":tf.float32},
        # {"opName": "zeros", "outName": "emptyArrayTests/zeros/ones_rank3", "varShapes":[[3]], "varTypes":["int32"], "varInit":["fixed_0_0_3"], "dtype":tf.int64},
        # {"opName": "zeros_like", "outName": "emptyArrayTests/zeros_like/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "zeros_like", "outName": "emptyArrayTests/zeros_like/rank2", "varShapes":[[2,0]], "varTypes":["float64"], "varInit":["empty"]},
        # {"opName": "zeros_like", "outName": "zeros_like/rank2_float16", "varShapes": [[3, 12]],"varTypes": ["float16"], "varInit": ["uniform"]},
        #{"opName": "zeros_like", "outName": "zeros_like/rank2_float32", "varShapes": [[3, 12]], "varTypes": ["float32"], "varInit": ["uniform"]},
        #{"opName": "zeros_like", "outName": "zeros_like/rank2_float64", "varShapes": [[3, 12]], "varTypes": ["float64"], "varInit": ["uniform"]},
        #{"opName": "zeros_like", "outName": "zeros_like/rank2_int32", "varShapes": [[3, 10]], "varTypes": ["int32"], "varInit": ["uniform_int10"]},
        # {"opName": "zeros_like", "outName": "zeros_like/rank2_int64", "varShapes": [[3, 10]], "varTypes": ["int64"], "varInit": ["uniform_int10"]},
        #{"opName": "zeros_like", "outName": "zeros_like/rank2_float32_dtype_int8", "varShapes": [[3, 12]], "varTypes": ["float32"], "varInit": ["uniform"], "dtype": tf.int8},
        #{"opName": "zeros_like", "outName": "zeros_like/rank2_float32_dtype_int16", "varShapes": [[3, 12]],"varTypes": ["float32"], "varInit": ["uniform"], "dtype": tf.int16},
        # {"opName": "zeros_like", "outName": "zeros_like/rank2_float32_dtype_int32", "varShapes": [[3, 12]], "varTypes": ["float32"], "varInit": ["uniform"], "dtype":tf.int32},
        # {"opName": "zeros_like", "outName": "zeros_like/rank2_float32_dtype_float16", "varShapes": [[3, 12]],  "varTypes": ["float32"], "varInit": ["uniform"], "dtype": tf.float16},
        # {"opName": "zeros_like", "outName": "zeros_like/rank2_float32_dtype_float32", "varShapes": [[3, 12]],  "varTypes": ["float32"], "varInit": ["uniform"], "dtype": tf.float32},
        # {"opName": "zeros_like", "outName": "zeros_like/rank2_float32_dtype_float64", "varShapes": [[3, 12]],  "varTypes": ["float32"], "varInit": ["uniform"], "dtype": tf.float64},

        # {"opName": "cast", "outName": "emptyArrayTests/cast/rank2_shape2-0", "varShapes":[[2,0]], "varTypes":["float32"], "varInit":["empty", "empty"], "dtype":tf.int32},
        # {"opName": "cast", "outName": "emptyArrayTests/cast/rank1_shape0", "varShapes":[[0]], "varTypes":["int32"], "varInit":["empty", "empty"], "dtype":tf.int64},


        #{"opName": "abs", "outName": "emptyArrayTests/abs/rank1", "varShapes":[[0]], "varTypes":["int32"], "varInit":["empty"], "dtype":tf.int32},
        #{"opName": "abs", "outName": "emptyArrayTests/abs/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "dtype":tf.float64},
        #{"opName": "abs", "outName": "abs/rank2_float64", "varShapes": [[1, 10]], "varTypes": ["float64"], "varInit": ["uniform"], "dtype": tf.float64},
        #{"opName": "abs", "outName": "abs/rank2_float32", "varShapes": [[1, 10]], "varTypes": ["float32"], "varInit": ["uniform"], "dtype": tf.float32},
        #{"opName": "abs", "outName": "abs/rank2_float16", "varShapes": [[1, 10]], "varTypes": ["float16"], "varInit": ["uniform"], "dtype": tf.float16},
        #{"opName": "abs", "outName": "abs/rank2_int32", "varShapes": [[1, 10]], "varTypes": ["int32"], "varInit": ["uniform_int10"], "dtype": tf.int32},
        #{"opName": "abs", "outName": "abs/rank2_int64", "varShapes": [[1, 10]], "varTypes": ["int64"], "varInit": ["uniform_int10"],"dtype": tf.int64},
        #{"opName": "abs", "outName": "abs/rank3_float64", "varShapes": [[1, 10, 20]], "varTypes": ["float64"], "varInit": ["uniform"], "dtype": tf.float64},
        #{"opName": "abs", "outName": "abs/rank3_float32", "varShapes": [[1, 10, 20]], "varTypes": ["float32"], "varInit": ["uniform"], "dtype": tf.float32},
        #{"opName": "abs", "outName": "abs/rank3_float16", "varShapes": [[1, 10, 20]], "varTypes": ["float16"], "varInit": ["uniform"], "dtype": tf.float16},
        #{"opName": "abs", "outName": "abs/rank3_int32", "varShapes": [[1, 10, 20]], "varTypes": ["int32"], "varInit": ["uniform_int10"], "dtype": tf.int32},
        # {"opName": "abs", "outName": "abs/rank3_int64", "varShapes": [[1, 10, 20]], "varTypes": ["int64"], "varInit": ["uniform_int10"], "dtype": tf.int64},
        # {"opName": "abs", "outName": "abs/rank2_float32_normal", "varShapes": [[1, 10]], "varTypes": ["float32"], "varInit": ["stdnormal"], "dtype": tf.float32},
        # {"opName": "abs", "outName": "abs/rank2_float16_normal", "varShapes": [[1, 10]], "varTypes": ["float16"], "varInit": ["stdnormal"], "dtype": tf.float16},

        # {"opName": "accumulate_n", "outName": "emptyArrayTests/accumulate_n/rank1_1", "varShapes":[[0]], "varTypes":["int64"], "varInit":["empty"], "dtype":tf.int64},
        # {"opName": "accumulate_n", "outName": "emptyArrayTests/accumulate_n/rank1_3", "varShapes":[[0], [0], [0]], "varTypes":["int64", "int64", "int64"], "varInit":["empty","empty","empty"], "dtype":tf.int64},
        # {"opName": "accumulate_n", "outName": "emptyArrayTests/accumulate_n/rank2_1", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "dtype":tf.float64},
        # {"opName": "accumulate_n", "outName": "emptyArrayTests/accumulate_n/rank2_2", "varShapes":[[0,2], [0,2], [0,2]], "varTypes":["float64", "float64", "float64"], "varInit":["empty", "empty", "empty"], "dtype":tf.float64},

        # {"opName": "add", "outName": "emptyArrayTests/add/rank1", "varShapes":[[0], [0]], "varTypes":["int64", "int64"], "varInit":["empty","empty"], "dtype":tf.int64},
        # {"opName": "add", "outName": "emptyArrayTests/add/rank2", "varShapes":[[0,2], [0,2]], "varTypes":["float64", "float64"], "varInit":["empty", "empty"], "dtype":tf.float64},
        # {"opName": "sub", "outName": "emptyArrayTests/sub/rank1", "varShapes":[[0], [0]], "varTypes":["int64", "int64"], "varInit":["empty","empty"], "dtype":tf.int64},
        # {"opName": "sub", "outName": "emptyArrayTests/sub/rank2", "varShapes":[[0,2], [0,2]], "varTypes":["float64", "float64"], "varInit":["empty", "empty"], "dtype":tf.float64},
        # {"opName": "mul", "outName": "emptyArrayTests/mul/rank1", "varShapes":[[0], [0]], "varTypes":["int64", "int64"], "varInit":["empty","empty"], "dtype":tf.int64},
        # {"opName": "mul", "outName": "emptyArrayTests/mul/rank2", "varShapes":[[0,2], [0,2]], "varTypes":["float64", "float64"], "varInit":["empty", "empty"], "dtype":tf.float64},
        # TODO: Div just gives "TypeError: unsupported operand type(s) for /: 'list' and 'list'" event through other ops work :/
        # {"opName": "div", "outName": "emptyArrayTests/div/rank1", "varShapes":[[0], [0]], "varTypes":["float32", "float32"], "varInit":["empty","empty"], "dtype":tf.float32},
        # {"opName": "div", "outName": "emptyArrayTests/div/rank2", "varShapes":[[0,2], [0,2]], "varTypes":["float64", "float64"], "varInit":["empty", "empty"], "dtype":tf.float64},
        #  {"opName": "div", "outName": "div/float64", "varShapes":[[1,10], [1,10]], "varTypes":["float64", "float64"], "varInit":["uniform", "uniform"], "dtype":tf.float64},
        #  {"opName": "div", "outName": "div/float32", "varShapes": [[1, 10], [1, 10]], "varTypes": ["float32", "float32"], "varInit": ["uniform", "uniform"], "dtype": tf.float32},
        #  {"opName": "div", "outName": "div/int32", "varShapes": [[1, 10], [1, 10]], "varTypes": ["int32", "int32"],  "varInit": ["uniform_int10", "uniform_int10"], "dtype": tf.int32},

        #{"opName": "div_no_nan", "outName": "div_no_nan", "varShapes":[[1,10], [1,10]], "varTypes":["float32", "float32"], "varInit":["uniform","uniform"], "dtype":tf.float32},
        # {"opName": "div_no_nan", "outName": "div_no_nan", "varShapes":[[], [1,10]], "varTypes":["float32", "float32"], "varInit":["uniform","uniform"], "dtype":tf.float32},
        #{"opName": "div_no_nan", "outName": "div_no_nan", "varShapes": [[1, 10], [1, 10]],"varTypes": ["float64", "float64"], "varInit": ["uniform", "uniform"], "dtype": tf.float64},

        # {"opName": "add_n", "outName": "emptyArrayTests/add_n/rank1", "varShapes":[[0], [0]], "varTypes":["int64", "int64"], "varInit":["empty","empty"], "dtype":tf.int64},
        # {"opName": "add_n", "outName": "emptyArrayTests/add_n/rank2", "varShapes":[[0,2], [0,2]], "varTypes":["float64", "float64"], "varInit":["empty", "empty"], "dtype":tf.float64},

        # {"opName": "cos", "outName": "emptyArrayTests/cos/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "cos", "outName": "emptyArrayTests/cos/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},
        # {"opName": "sin", "outName": "emptyArrayTests/sin/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "sin", "outName": "emptyArrayTests/sin/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},
        # {"opName": "tan", "outName": "emptyArrayTests/tan/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "tan", "outName": "emptyArrayTests/tan/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},
        # {"opName": "cosh", "outName": "emptyArrayTests/cosh/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "cosh", "outName": "emptyArrayTests/cosh/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},
        # {"opName": "acos", "outName": "emptyArrayTests/acos/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "acos", "outName": "emptyArrayTests/acos/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},
        # {"opName": "acosh", "outName": "emptyArrayTests/acosh/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "acosh", "outName": "emptyArrayTests/acosh/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},
        # {"opName": "asin", "outName": "emptyArrayTests/asin/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "asin", "outName": "emptyArrayTests/asin/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},
        # {"opName": "asinh", "outName": "emptyArrayTests/asinh/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "asinh", "outName": "emptyArrayTests/asinh/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},
        # {"opName": "atan", "outName": "emptyArrayTests/atan/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "atan", "outName": "emptyArrayTests/atan/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},
        # {"opName": "atanh", "outName": "emptyArrayTests/atanh/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "atanh", "outName": "emptyArrayTests/atanh/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},
        # {"opName": "ceil", "outName": "emptyArrayTests/ceil/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "ceil", "outName": "emptyArrayTests/ceil/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},

        # {"opName": "count_nonzero", "outName": "emptyArrayTests/count_nonzero/rank1_axisNone", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keep_dims":False},
        # {"opName": "count_nonzero", "outName": "emptyArrayTests/count_nonzero/rank2_axis0", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":0, "keep_dims":False},
        # {"opName": "count_nonzero", "outName": "emptyArrayTests/count_nonzero/rank2_axis1", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":1, "keep_dims":False},
        # {"opName": "count_nonzero", "outName": "emptyArrayTests/count_nonzero/rank2_axis1_keep", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":1, "keep_dims":True},

        # {"opName": "cumprod", "outName": "emptyArrayTests/cumprod/rank1_axis0", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":0},
        # {"opName": "cumprod", "outName": "emptyArrayTests/cumprod/rank2_axis0", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":0},
        # {"opName": "cumprod", "outName": "emptyArrayTests/cumprod/rank2_axis1", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":1},
        # {"opName": "cumsum", "outName": "emptyArrayTests/cumsum/rank1_axis0", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":0},
        # {"opName": "cumsum", "outName": "emptyArrayTests/cumsum/rank2_axis0", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":0},
        # {"opName": "cumsum", "outName": "emptyArrayTests/cumsum/rank2_axis1", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":1},

        # {"opName": "equal", "outName": "emptyArrayTests/equal/rank1", "varShapes":[[0], [0]], "varTypes":["int64", "int64"], "varInit":["empty","empty"], "dtype":tf.int64},
        # {"opName": "equal", "outName": "emptyArrayTests/equal/rank2", "varShapes":[[0,2], [0,2]], "varTypes":["float64", "float64"], "varInit":["empty", "empty"]},
        # {"opName": "equal", "outName": "emptyArrayTests/equal/rank2bc", "varShapes":[[1,2], [0,2]], "varTypes":["float64", "float64"], "varInit":["uniform", "empty"]},

        # {"opName": "exp", "outName": "emptyArrayTests/exp/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "exp", "outName": "emptyArrayTests/exp/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},
        # {"opName": "floor", "outName": "emptyArrayTests/floor/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "floor", "outName": "emptyArrayTests/floor/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},
        # {"opName": "floordiv", "outName": "emptyArrayTests/floordiv/rank1", "varShapes":[[0], [0]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"]},
        # {"opName": "floordiv", "outName": "emptyArrayTests/floordiv/rank2", "varShapes":[[0,2], [0,2]], "varTypes":["float64", "float64"], "varInit":["empty", "empty"]},
        # {"opName": "floordiv", "outName": "emptyArrayTests/floordiv/rank2bc", "varShapes":[[1,1], [0,2]], "varTypes":["float64", "float64"], "varInit":["uniform", "empty"]},
        # {"opName": "floordiv", "outName": "emptyArrayTests/floordiv/rank2bc2", "varShapes":[[0,1], [0,2]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"]},

        # {"opName": "log", "outName": "emptyArrayTests/log/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "log", "outName": "emptyArrayTests/log/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},
        # {"opName": "log_sigmoid", "outName": "emptyArrayTests/log_sigmoid/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "log_sigmoid", "outName": "emptyArrayTests/log_sigmoid/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},
        # {"opName": "negative", "outName": "emptyArrayTests/negative/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "negative", "outName": "emptyArrayTests/negative/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},
        # {"opName": "reciprocal", "outName": "emptyArrayTests/reciprocal/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "reciprocal", "outName": "emptyArrayTests/reciprocal/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},
        # {"opName": "sign", "outName": "emptyArrayTests/sign/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "sign", "outName": "emptyArrayTests/sign/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},
        # {"opName": "softplus", "outName": "emptyArrayTests/softplus/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "softplus", "outName": "emptyArrayTests/softplus/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},
        # {"opName": "sqrt", "outName": "emptyArrayTests/sqrt/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "sqrt", "outName": "emptyArrayTests/sqrt/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},
        # {"opName": "square", "outName": "emptyArrayTests/square/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "square", "outName": "emptyArrayTests/square/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},
        # {"opName": "rsqrt", "outName": "emptyArrayTests/rsqrt/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "rsqrt", "outName": "emptyArrayTests/rsqrt/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},
        # {"opName": "round", "outName": "emptyArrayTests/round/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "round", "outName": "emptyArrayTests/round/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},
        # {"opName": "sigmoid", "outName": "emptyArrayTests/sigmoid/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "sigmoid", "outName": "emptyArrayTests/sigmoid/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},

        # {"opName": "greater", "outName": "emptyArrayTests/greater/rank1", "varShapes":[[0], [0]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"]},
        # {"opName": "greater", "outName": "emptyArrayTests/greater/rank2", "varShapes":[[0,2], [0,2]], "varTypes":["float64", "float64"], "varInit":["empty", "empty"]},
        # {"opName": "greater", "outName": "emptyArrayTests/greater/rank2bc", "varShapes":[[1,1], [0,2]], "varTypes":["float64", "float64"], "varInit":["uniform", "empty"]},
        # {"opName": "greater", "outName": "emptyArrayTests/greater/rank2bc2", "varShapes":[[0,1], [0,2]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"]},
        # {"opName": "greater_equal", "outName": "emptyArrayTests/greater_equal/rank1", "varShapes":[[0], [0]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"]},
        # {"opName": "greater_equal", "outName": "emptyArrayTests/greater_equal/rank2", "varShapes":[[0,2], [0,2]], "varTypes":["float64", "float64"], "varInit":["empty", "empty"]},
        # {"opName": "greater_equal", "outName": "emptyArrayTests/greater_equal/rank2bc", "varShapes":[[1,1], [0,2]], "varTypes":["float64", "float64"], "varInit":["uniform", "empty"]},
        # {"opName": "greater_equal", "outName": "emptyArrayTests/greater_equal/rank2bc2", "varShapes":[[0,1], [0,2]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"]},
        # {"opName": "less", "outName": "emptyArrayTests/less/rank1", "varShapes":[[0], [0]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"]},
        # {"opName": "less", "outName": "emptyArrayTests/less/rank2", "varShapes":[[0,2], [0,2]], "varTypes":["float64", "float64"], "varInit":["empty", "empty"]},
        # {"opName": "less", "outName": "emptyArrayTests/less/rank2bc", "varShapes":[[1,1], [0,2]], "varTypes":["float64", "float64"], "varInit":["uniform", "empty"]},
        # {"opName": "less", "outName": "emptyArrayTests/less/rank2bc2", "varShapes":[[0,1], [0,2]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"]},
        # {"opName": "less_equal", "outName": "emptyArrayTests/less_equal/rank1", "varShapes":[[0], [0]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"]},
        # {"opName": "less_equal", "outName": "emptyArrayTests/less_equal/rank2", "varShapes":[[0,2], [0,2]], "varTypes":["float64", "float64"], "varInit":["empty", "empty"]},
        # {"opName": "less_equal", "outName": "emptyArrayTests/less_equal/rank2bc", "varShapes":[[1,1], [0,2]], "varTypes":["float64", "float64"], "varInit":["uniform", "empty"]},
        # {"opName": "less_equal", "outName": "emptyArrayTests/less_equal/rank2bc2", "varShapes":[[0,1], [0,2]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"]},
        # {"opName": "not_equal", "outName": "emptyArrayTests/not_equal/rank1", "varShapes":[[0], [0]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"]},
        # {"opName": "not_equal", "outName": "emptyArrayTests/not_equal/rank2", "varShapes":[[0,2], [0,2]], "varTypes":["float64", "float64"], "varInit":["empty", "empty"]},
        # {"opName": "not_equal", "outName": "emptyArrayTests/not_equal/rank2bc", "varShapes":[[1,1], [0,2]], "varTypes":["float64", "float64"], "varInit":["uniform", "empty"]},
        # {"opName": "not_equal", "outName": "emptyArrayTests/not_equal/rank2bc2", "varShapes":[[0,1], [0,2]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"]},

        # {"opName": "truediv", "outName": "emptyArrayTests/truediv/rank1", "varShapes":[[0], [0]], "varTypes":["int32", "int32"], "varInit":["empty", "empty"]},
        # {"opName": "truediv", "outName": "emptyArrayTests/truediv/rank2", "varShapes":[[0,2], [0,2]], "varTypes":["int64", "int64"], "varInit":["empty", "empty"]},
        # {"opName": "truediv", "outName": "emptyArrayTests/truediv/rank2bc", "varShapes":[[1,1], [0,2]], "varTypes":["float64", "float64"], "varInit":["uniform", "empty"]},
        # {"opName": "truediv", "outName": "emptyArrayTests/truediv/rank2bc2", "varShapes":[[0,1], [0,2]], "varTypes":["int32", "int32"], "varInit":["empty", "empty"]},

        # {"opName": "zero_fraction", "outName": "emptyArrayTests/zero_fraction/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "zero_fraction", "outName": "emptyArrayTests/zero_fraction/rank2", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"]},

        #{"opName": "maximum", "outName": "emptyArrayTests/maximum/rank1", "varShapes":[[0], [0]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"]},
        # {"opName": "maximum", "outName": "emptyArrayTests/maximum/rank2", "varShapes":[[0,2], [0,2]], "varTypes":["float64", "float64"], "varInit":["empty", "empty"]},
        # {"opName": "maximum", "outName": "emptyArrayTests/maximum/rank2bc", "varShapes":[[1,1], [0,2]], "varTypes":["float64", "float64"], "varInit":["uniform", "empty"]},
        # {"opName": "maximum", "outName": "emptyArrayTests/maximum/rank2bc2", "varShapes":[[0,1], [0,2]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"]},
        # {"opName": "minimum", "outName": "emptyArrayTests/minimum/rank1", "varShapes":[[0], [0]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"]},
        # {"opName": "minimum", "outName": "emptyArrayTests/minimum/rank2", "varShapes":[[0,2], [0,2]], "varTypes":["float64", "float64"], "varInit":["empty", "empty"]},
        # {"opName": "minimum", "outName": "emptyArrayTests/minimum/rank2bc", "varShapes":[[1,1], [0,2]], "varTypes":["float64", "float64"], "varInit":["uniform", "empty"]},
        # {"opName": "minimum", "outName": "emptyArrayTests/minimum/rank2bc2", "varShapes":[[0,1], [0,2]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"]},
        # {"opName": "max", "outName": "max/rank2_float32", "varShapes":[[2,12], [2,12]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"]},
        # {"opName": "min", "outName": "min/rank2_float32", "varShapes": [[2, 12], [2, 12]], "varTypes": ["float32", "float32"], "varInit": ["uniform", "uniform"]},
        # {"opName": "mod", "outName": "mod/rank2_float32", "varShapes": [[2, 1,10], [2, 3,1]], "varTypes": ["float32", "float32"],  "varInit": ["uniform", "uniform"]},
        # {"opName": "max", "outName": "max/scalar_broadcasting_float64", "varShapes": [[], [2, 12]], "varTypes": ["float64", "float64"], "varInit": ["uniform", "uniform"]},
        # {"opName": "min", "outName": "min/scalar_broadcasting_float64", "varShapes": [[], [2, 12]], "varTypes": ["float64", "float64"], "varInit": ["uniform", "uniform"]},
        # {"opName": "mod", "outName": "mod/rank2_float64", "varShapes": [[2, 12], [2, 12]], "varTypes": ["float64", "float64"], "varInit": ["uniform", "uniform"]},
        # {"opName": "max", "outName": "max/rank2_int32", "varShapes": [[2, 12], [2, 12]], "varTypes": ["int32", "int32"], "varInit": ["uniform_int10", "uniform_int10"]},
        # {"opName": "min", "outName": "min/rank2_int32", "varShapes": [[2, 12], [2, 12]], "varTypes": ["int32", "int32"], "varInit": ["uniform_int10", "uniform_int10"]},
        # {"opName": "mod", "outName": "mod/rank2_float32", "varShapes": [[2, 12], [2, 12]], "varTypes": ["float32", "float32"], "varInit": ["uniform", "uniform"]},
        # {"opName": "mod", "outName": "mod/rank2_float64", "varShapes": [[2, 12], [2, 12]], "varTypes": ["float64", "float64"], "varInit": ["uniform", "uniform"]},
        # {"opName": "max", "outName": "max/rank3_int32", "varShapes":[[2,3,10], [2,3,10]], "varTypes":["int32", "int32"], "varInit":["uniform_int10", "uniform_int10"]},
        # {"opName": "max", "outName": "max/rank3_float32", "varShapes": [[2, 3, 12], [2, 3, 12]], "varTypes": ["float32", "float32"], "varInit": ["uniform", "uniform"]},
        # {"opName": "max", "outName": "max/rank3_float32", "varShapes": [[2, 3, 12], [2, 3, 12]], "varTypes": ["float32", "float32"],  "varInit": ["uniform", "uniform"]},
        # {"opName": "max", "outName": "max/rank3_float64", "varShapes": [[2, 3, 10], [2, 3, 10]],  "varTypes": ["float64", "float64"], "varInit": ["uniform", "uniform"]},
        # {"opName": "max", "outName": "max/rank3_float64", "varShapes": [[2, 3, 12], [2, 3, 12]], "varTypes": ["float64", "float64"], "varInit": ["uniform", "uniform"]},
        # {"opName": "max", "outName": "max/rank3_float64", "varShapes": [[2, 3, 12], [2, 3, 12]], "varTypes": ["float64", "float64"], "varInit": ["uniform", "uniform"]},
        #{"opName": "max", "outName": "max/rank3_half", "varShapes": [[2, 3, 10], [2, 3, 10]],  "varTypes": ["half", "half"], "varInit": ["uniform", "uniform"]},
        # {"opName": "max", "outName": "max/rank3_half", "varShapes": [[2, 3, 12], [2, 3, 12]], "varTypes": ["half", "half"], "varInit": ["uniform", "uniform"]},
        # {"opName": "max", "outName": "max/rank3_half", "varShapes": [[2, 3, 12], [2, 3, 12]], "varTypes": ["half", "half"], "varInit": ["uniform", "uniform"]},

        # {"opName": "pow", "outName": "emptyArrayTests/pow/rank1", "varShapes":[[0], [0]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"]},
        # {"opName": "pow", "outName": "emptyArrayTests/pow/rank2", "varShapes":[[0,2], [0,2]], "varTypes":["float64", "float64"], "varInit":["empty", "empty"]},
        # {"opName": "pow", "outName": "emptyArrayTests/pow/rank2bc", "varShapes":[[1,1], [0,2]], "varTypes":["float64", "float64"], "varInit":["uniform", "empty"]},
        # {"opName": "pow", "outName": "emptyArrayTests/pow/rank2bc2", "varShapes":[[0,1], [0,2]], "varTypes":["float32", "float32"], "varInit":["empty", "empty"]},

        #Not sure if segment ops are possible with empty inputs? "ValueError: Shape must be rank 1 but is rank 2 for 'SegmentMax' (op: 'SegmentMax') with input shapes: [1,0], [1,0]."??
        # {"opName": "segment_max", "outName": "emptyArrayTests/segment/segment_max_rank1", "varShapes":[[0], [0]], "varTypes":["float32", "int32"], "varInit":["empty", "empty"]},


        # {"opName": "logical_and", "outName": "emptyArrayTests/logical_and/rank1", "varShapes":[[0], [0]], "varTypes":["bool", "bool"], "varInit":["empty", "empty"]},
        # {"opName": "logical_and", "outName": "emptyArrayTests/logical_and/rank2", "varShapes":[[0,2], [0,2]], "varTypes":["bool", "bool"], "varInit":["empty", "empty"]},
        # {"opName": "logical_and", "outName": "emptyArrayTests/logical_and/rank2bc", "varShapes":[[1,1], [0,2]], "varTypes":["bool", "bool"], "varInit":["boolean", "empty"]},
        # {"opName": "logical_and", "outName": "emptyArrayTests/logical_and/rank2bc2", "varShapes":[[0,1], [0,2]], "varTypes":["bool", "bool"], "varInit":["empty", "empty"]},
        # {"opName": "logical_or", "outName": "emptyArrayTests/logical_or/rank1", "varShapes":[[0], [0]], "varTypes":["bool", "bool"], "varInit":["empty", "empty"]},
        # {"opName": "logical_or", "outName": "emptyArrayTests/logical_or/rank2", "varShapes":[[0,2], [0,2]], "varTypes":["bool", "bool"], "varInit":["empty", "empty"]},
        # {"opName": "logical_or", "outName": "emptyArrayTests/logical_or/rank2bc", "varShapes":[[1,1], [0,2]], "varTypes":["bool", "bool"], "varInit":["boolean", "empty"]},
        # {"opName": "logical_or", "outName": "emptyArrayTests/logical_or/rank2bc2", "varShapes":[[0,1], [0,2]], "varTypes":["bool", "bool"], "varInit":["empty", "empty"]},
        # {"opName": "logical_xor", "outName": "emptyArrayTests/logical_xor/rank1", "varShapes":[[0], [0]], "varTypes":["bool", "bool"], "varInit":["empty", "empty"]},
        # {"opName": "logical_xor", "outName": "emptyArrayTests/logical_xor/rank2", "varShapes":[[0,2], [0,2]], "varTypes":["bool", "bool"], "varInit":["empty", "empty"]},
        # {"opName": "logical_xor", "outName": "emptyArrayTests/logical_xor/rank2bc", "varShapes":[[1,1], [0,2]], "varTypes":["bool", "bool"], "varInit":["boolean", "empty"]},
        # {"opName": "logical_xor", "outName": "emptyArrayTests/logical_xor/rank2bc2", "varShapes":[[0,1], [0,2]], "varTypes":["bool", "bool"], "varInit":["empty", "empty"]},
        # {"opName": "logical_not", "outName": "emptyArrayTests/logical_not/rank1", "varShapes":[[0]], "varTypes":["bool"], "varInit":["empty"]},
        # {"opName": "logical_not", "outName": "emptyArrayTests/logical_not/rank2", "varShapes":[[0,2]], "varTypes":["bool"], "varInit":["empty"]},

        # {"opName": "reduce_all", "outName": "emptyArrayTests/reduce_all/rank1_axisNone", "varShapes":[[0]], "varTypes":["bool"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_all", "outName": "emptyArrayTests/reduce_all/rank1_axis0_keep", "varShapes":[[0]], "varTypes":["bool"], "varInit":["empty"], "axis":0, "keepdims":True},
        # {"opName": "reduce_all", "outName": "emptyArrayTests/reduce_all/rank2_axisNone", "varShapes":[[0,2]], "varTypes":["bool"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_all", "outName": "emptyArrayTests/reduce_all/rank2_axisNoneKeep", "varShapes":[[0,2]], "varTypes":["bool"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_all", "outName": "emptyArrayTests/reduce_all/rank2_axis0", "varShapes":[[0,2]], "varTypes":["bool"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_all", "outName": "emptyArrayTests/reduce_all/rank2_axis1Keep", "varShapes":[[0,2]], "varTypes":["bool"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_any", "outName": "emptyArrayTests/reduce_any/rank1_axisNone", "varShapes":[[0]], "varTypes":["bool"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_any", "outName": "emptyArrayTests/reduce_any/rank1_axis0_keep", "varShapes":[[0]], "varTypes":["bool"], "varInit":["empty"], "axis":0, "keepdims":True},
        # {"opName": "reduce_any", "outName": "emptyArrayTests/reduce_any/rank2_axisNone", "varShapes":[[0,2]], "varTypes":["bool"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_any", "outName": "emptyArrayTests/reduce_any/rank2_axisNoneKeep", "varShapes":[[0,2]], "varTypes":["bool"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_any", "outName": "emptyArrayTests/reduce_any/rank2_axis0", "varShapes":[[0,2]], "varTypes":["bool"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_any", "outName": "emptyArrayTests/reduce_any/rank2_axis1Keep", "varShapes":[[0,2]], "varTypes":["bool"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_logsumexp", "outName": "emptyArrayTests/reduce_logsumexp/rank1_axisNone", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_logsumexp", "outName": "emptyArrayTests/reduce_logsumexp/rank1_axis0_keep", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":0, "keepdims":True},
        # {"opName": "reduce_logsumexp", "outName": "emptyArrayTests/reduce_logsumexp/rank2_axisNone", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_logsumexp", "outName": "emptyArrayTests/reduce_logsumexp/rank2_axisNoneKeep", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_logsumexp", "outName": "emptyArrayTests/reduce_logsumexp/rank2_axis0", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_logsumexp", "outName": "emptyArrayTests/reduce_logsumexp/rank2_axis1Keep", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_max", "outName": "emptyArrayTests/reduce_max/rank1_axisNone", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_max", "outName": "emptyArrayTests/reduce_max/rank1_axis0_keep", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":0, "keepdims":True},
        # {"opName": "reduce_max", "outName": "emptyArrayTests/reduce_max/rank2_axisNone", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_max", "outName": "emptyArrayTests/reduce_max/rank2_axisNoneKeep", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_max", "outName": "emptyArrayTests/reduce_max/rank2_axis0", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_max", "outName": "emptyArrayTests/reduce_max/rank2_axis1Keep", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_min", "outName": "emptyArrayTests/reduce_min/rank1_axisNone", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_min", "outName": "emptyArrayTests/reduce_min/rank1_axis0_keep", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":0, "keepdims":True},
        # {"opName": "reduce_min", "outName": "emptyArrayTests/reduce_min/rank2_axisNone", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_min", "outName": "emptyArrayTests/reduce_min/rank2_axisNoneKeep", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_min", "outName": "emptyArrayTests/reduce_min/rank2_axis0", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_min", "outName": "emptyArrayTests/reduce_min/rank2_axis1Keep", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_prod", "outName": "emptyArrayTests/reduce_prod/rank1_axisNone", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_prod", "outName": "emptyArrayTests/reduce_prod/rank1_axis0_keep", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":0, "keepdims":True},
        # {"opName": "reduce_prod", "outName": "emptyArrayTests/reduce_prod/rank2_axisNone", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_prod", "outName": "emptyArrayTests/reduce_prod/rank2_axisNoneKeep", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_prod", "outName": "emptyArrayTests/reduce_prod/rank2_axis0", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_prod", "outName": "emptyArrayTests/reduce_prod/rank2_axis1Keep", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_mean", "outName": "emptyArrayTests/reduce_mean/rank1_axisNone", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_mean", "outName": "emptyArrayTests/reduce_mean/rank1_axis0_keep", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":0, "keepdims":True},
        # {"opName": "reduce_mean", "outName": "emptyArrayTests/reduce_mean/rank2_axisNone", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_mean", "outName": "emptyArrayTests/reduce_mean/rank2_axisNoneKeep", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_mean", "outName": "emptyArrayTests/reduce_mean/rank2_axis0", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_mean", "outName": "emptyArrayTests/reduce_mean/rank2_axis1Keep", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_sum", "outName": "emptyArrayTests/reduce_sum/rank1_axisNone", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_sum", "outName": "emptyArrayTests/reduce_sum/rank1_axis0_keep", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":0, "keepdims":True},
        # {"opName": "reduce_sum", "outName": "emptyArrayTests/reduce_sum/rank2_axisNone", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_sum", "outName": "emptyArrayTests/reduce_sum/rank2_axisNoneKeep", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_sum", "outName": "emptyArrayTests/reduce_sum/rank2_axis0", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_sum", "outName": "emptyArrayTests/reduce_sum/rank2_axis1Keep", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":True},

        # {"opName": "l2_normalize", "outName": "emptyArrayTests/l2_normalize/rank1", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "epsilon":0.1, "axis":None},
        # {"opName": "l2_normalize", "outName": "emptyArrayTests/l2_normalize/rank2", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "epsilon":0.1, "axis":None},
        # {"opName": "l2_normalize", "outName": "emptyArrayTests/l2_normalize/rank2_axis0", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "epsilon":0.1, "axis":0},
        # {"opName": "l2_normalize", "outName": "emptyArrayTests/l2_normalize/rank2_axis1", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "epsilon":0.1, "axis":1},
        #Not possible?
        # {"opName": "diag_part", "outName": "emptyArrayTests/diag_part/rank2", "varShapes":[[0,0]], "varTypes":["float32"], "varInit":["empty"]},
        # {"opName": "diag_part", "outName": "emptyArrayTests/diag_part/rank3", "varShapes":[[2,2,0]], "varTypes":["float32"], "varInit":["empty"]},


        # TODO BOOL

        # {"opName": "add", "outName": "dtype_tests/add/bfloat16_rank0_1", "varShapes":[[], [1]], "varTypes":["bfloat16", "bfloat16"], "varInit":["one","one"], "dtype":tf.bfloat16},
        # {"opName": "add", "outName": "dtype_tests/add/bfloat16_rank2", "varShapes":[[2,3], [2,3]], "varTypes":["bfloat16", "bfloat16"], "varInit":["one","one"], "dtype":tf.bfloat16},
        # {"opName": "add", "outName": "dtype_tests/add/double_rank0_1", "varShapes":[[], [1]], "varTypes":["double", "double"], "varInit":["one","one"], "dtype":tf.double},
        # {"opName": "add", "outName": "dtype_tests/add/double_rank2", "varShapes":[[2,3], [2,3]], "varTypes":["double", "double"], "varInit":["one","one"], "dtype":tf.double},
        # {"opName": "add", "outName": "dtype_tests/add/float16_rank0_1", "varShapes":[[], [1]], "varTypes":["float16", "float16"], "varInit":["one","one"], "dtype":tf.float16},
        # {"opName": "add", "outName": "dtype_tests/add/float16_rank2", "varShapes":[[2,3], [2,3]], "varTypes":["float16", "float16"], "varInit":["one","one"], "dtype":tf.float16},
        # {"opName": "add", "outName": "dtype_tests/add/float32_rank0_1", "varShapes":[[], [1]], "varTypes":["float32", "float32"], "varInit":["one","one"], "dtype":tf.float32},
        # {"opName": "add", "outName": "dtype_tests/add/float32_rank2", "varShapes":[[2,3], [2,3]], "varTypes":["float32", "float32"], "varInit":["one","one"], "dtype":tf.float32},
        # {"opName": "add", "outName": "dtype_tests/add/float64_rank0_1", "varShapes":[[], [1]], "varTypes":["float64", "float64"], "varInit":["one","one"], "dtype":tf.float64},
        # {"opName": "add", "outName": "dtype_tests/add/float64_rank2", "varShapes":[[2,3], [2,3]], "varTypes":["float64", "float64"], "varInit":["one","one"], "dtype":tf.float64},
        # {"opName": "add", "outName": "dtype_tests/add/half_rank0_1", "varShapes":[[], [1]], "varTypes":["half", "half"], "varInit":["one","one"], "dtype":tf.half},
        # {"opName": "add", "outName": "dtype_tests/add/half_rank2", "varShapes":[[2,3], [2,3]], "varTypes":["half", "half"], "varInit":["one","one"], "dtype":tf.half},
        # {"opName": "add", "outName": "dtype_tests/add/int16_rank0_1", "varShapes":[[], [1]], "varTypes":["int16", "int16"], "varInit":["one","one"], "dtype":tf.int16},
        # {"opName": "add", "outName": "dtype_tests/add/int16_rank2", "varShapes":[[2,3], [2,3]], "varTypes":["int16", "int16"], "varInit":["one","one"], "dtype":tf.int16},
        # {"opName": "add", "outName": "dtype_tests/add/int32_rank0_1", "varShapes":[[], [1]], "varTypes":["int32", "int32"], "varInit":["one","one"], "dtype":tf.int32},
        # {"opName": "add", "outName": "dtype_tests/add/int32_rank2", "varShapes":[[2,3], [2,3]], "varTypes":["int32", "int32"], "varInit":["one","one"], "dtype":tf.int32},
        # {"opName": "add", "outName": "dtype_tests/add/int64_rank0_1", "varShapes":[[], [1]], "varTypes":["int64", "int64"], "varInit":["one","one"], "dtype":tf.int64},
        # {"opName": "add", "outName": "dtype_tests/add/int64_rank2", "varShapes":[[2,3], [2,3]], "varTypes":["int64", "int64"], "varInit":["one","one"], "dtype":tf.int64},
        # {"opName": "add", "outName": "dtype_tests/add/int8_rank0_1", "varShapes":[[], [1]], "varTypes":["int8", "int8"], "varInit":["one","one"], "dtype":tf.int8},
        # {"opName": "add", "outName": "dtype_tests/add/int8_rank2", "varShapes":[[2,3], [2,3]], "varTypes":["int8", "int8"], "varInit":["one","one"], "dtype":tf.int8},
        # {"opName": "concat", "outName": "dtype_tests/concat/uint16_rank1", "varShapes":[[2], [1]], "varTypes":["uint16", "uint16"], "varInit":["one","one"], "dtype":tf.uint16, "axis":0},
        # {"opName": "concat", "outName": "dtype_tests/concat/uint16_rank2", "varShapes":[[2,3], [2,3]], "varTypes":["uint16", "uint16"], "varInit":["one","one"], "dtype":tf.uint16, "axis":0},
        # {"opName": "concat", "outName": "dtype_tests/concat/uint8_rank1", "varShapes":[[2], [1]], "varTypes":["uint8", "uint8"], "varInit":["one","one"], "dtype":tf.uint8, "axis":0},
        # {"opName": "concat", "outName": "dtype_tests/concat/uint8_rank2", "varShapes":[[2,3], [2,3]], "varTypes":["uint8", "uint8"], "varInit":["one","one"], "dtype":tf.uint8, "axis":0},
        #UINT32 and 64 don't support concat for some reason :/
        # {"opName": "concat", "outName": "dtype_tests/concat/uint32_rank1", "varShapes":[[2], [1]], "varTypes":["uint32", "uint32"], "varInit":["one","one"], "dtype":tf.uint32, "axis":0},
        # {"opName": "concat", "outName": "dtype_tests/concat/uint32_rank2", "varShapes":[[2,3], [2,3]], "varTypes":["uint32", "uint32"], "varInit":["one","one"], "dtype":tf.uint32, "axis":0},
        # {"opName": "concat", "outName": "dtype_tests/concat/uint64_rank1", "varShapes":[[2], [1]], "varTypes":["uint64", "uint64"], "varInit":["one","one"], "dtype":tf.uint64, "axis":0},
        # {"opName": "concat", "outName": "dtype_tests/concat/uint64_rank2", "varShapes":[[2,3], [2,3]], "varTypes":["uint64", "uint64"], "varInit":["one","one"], "dtype":tf.uint64, "axis":0},

        # This doesn't work either... :/
        # {"opName": "arg_max", "outName": "dtype_tests/argmax/uint32_rank1", "varShapes":[[4]], "varTypes":["uint32", "uint32"], "varInit":["one"], "dtype":tf.uint32, "dimension":0},
        # {"opName": "arg_max", "outName": "dtype_tests/argmax/uint32_rank2", "varShapes":[[3,4]], "varTypes":["uint32", "uint32"], "varInit":["one"], "dtype":tf.uint32, "dimension":0},
        # {"opName": "arg_max", "outName": "dtype_tests/argmax/uint64_rank1", "varShapes":[[4]], "varTypes":["uint64", "uint64"], "varInit":["one"], "dtype":tf.uint64, "dimension":0},
        # {"opName": "arg_max", "outName": "dtype_tests/argmax/uint64_rank2", "varShapes":[[3,4]], "varTypes":["uint64", "uint64"], "varInit":["one"], "dtype":tf.uint64, "dimension":0},

        #Also doesn't work: zero, range, four, uniform10, etc inits - always fail with "No OpKernel was registered to support Op 'Assign'"
        # {"opName": "arg_max", "outName": "dtype_tests/argmax/uint32_rank1", "varShapes":[[3]], "varTypes":["uint32", "uint32"], "varInit":["fixed_2_2_4"], "dtype":tf.uint32, "dimension":0},
        # {"opName": "arg_max", "outName": "dtype_tests/argmax/uint32_rank2", "varShapes":[[3,4]], "varTypes":["uint32", "uint32"], "varInit":["one"], "dtype":tf.uint32, "dimension":0},
        # {"opName": "arg_max", "outName": "dtype_tests/argmax/uint64_rank1", "varShapes":[[4]], "varTypes":["uint64", "uint64"], "varInit":["one"], "dtype":tf.uint64, "dimension":0},
        # {"opName": "arg_max", "outName": "dtype_tests/argmax/uint64_rank2", "varShapes":[[3,4]], "varTypes":["uint64", "uint64"], "varInit":["one"], "dtype":tf.uint64, "dimension":0},

        # {"opName": "cast", "outName": "dtype_tests/cast/uint32_rank1", "varShapes":[[4]], "varTypes":["int32"], "varInit":["uniform10"], "dtype":tf.int32, "dtype":tf.uint32},
        # {"opName": "cast", "outName": "dtype_tests/cast/uint64_rank1", "varShapes":[[4]], "varTypes":["int32"], "varInit":["uniform10"], "dtype":tf.int32, "dtype":tf.uint64},


        # {"opName": "reduce_all", "outName": "emptyReduceAxisTests/reduce_all/rank1_axisNone", "varShapes":[[0]], "varTypes":["bool"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_all", "outName": "emptyReduceAxisTests/reduce_all/rank1_axis0_keep", "varShapes":[[0]], "varTypes":["bool"], "varInit":["empty"], "axis":0, "keepdims":True},
        # {"opName": "reduce_all", "outName": "emptyReduceAxisTests/reduce_all/rank2_axisNone", "varShapes":[[0,2]], "varTypes":["bool"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_all", "outName": "emptyReduceAxisTests/reduce_all/rank2_axisNoneKeep", "varShapes":[[0,2]], "varTypes":["bool"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_all", "outName": "emptyReduceAxisTests/reduce_all/rank2_axis0", "varShapes":[[0,2]], "varTypes":["bool"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_all", "outName": "emptyReduceAxisTests/reduce_all/rank2_axis1Keep", "varShapes":[[0,2]], "varTypes":["bool"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_any", "outName": "emptyReduceAxisTests/reduce_any/rank1_axisNone", "varShapes":[[0]], "varTypes":["bool"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_any", "outName": "emptyReduceAxisTests/reduce_any/rank1_axis0_keep", "varShapes":[[0]], "varTypes":["bool"], "varInit":["empty"], "axis":0, "keepdims":True},
        # {"opName": "reduce_any", "outName": "emptyReduceAxisTests/reduce_any/rank2_axisNone", "varShapes":[[0,2]], "varTypes":["bool"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_any", "outName": "emptyReduceAxisTests/reduce_any/rank2_axisNoneKeep", "varShapes":[[0,2]], "varTypes":["bool"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_any", "outName": "emptyReduceAxisTests/reduce_any/rank2_axis0", "varShapes":[[0,2]], "varTypes":["bool"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_any", "outName": "emptyReduceAxisTests/reduce_any/rank2_axis1Keep", "varShapes":[[0,2]], "varTypes":["bool"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_logsumexp_dynamicaxis", "outName": "emptyReduceAxisTests/reduce_logsumexp/rank1_axisNone", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_logsumexp_dynamicaxis", "outName": "emptyReduceAxisTests/reduce_logsumexp/rank1_axis0_keep", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":0, "keepdims":True},
        # {"opName": "reduce_logsumexp_dynamicaxis", "outName": "emptyReduceAxisTests/reduce_logsumexp/rank2_axisNone", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_logsumexp_dynamicaxis", "outName": "emptyReduceAxisTests/reduce_logsumexp/rank2_axisNoneKeep", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_logsumexp_dynamicaxis", "outName": "emptyReduceAxisTests/reduce_logsumexp/rank2_axis0", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_logsumexp_dynamicaxis", "outName": "emptyReduceAxisTests/reduce_logsumexp/rank2_axis1Keep", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_max_dynamicaxis", "outName": "emptyReduceAxisTests/reduce_max/rank2_axisNone", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_max_dynamicaxis", "outName": "emptyReduceAxisTests/reduce_max/rank2_axisNoneKeep", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_max_dynamicaxis", "outName": "emptyReduceAxisTests/reduce_max/rank2_axis0", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_max_dynamicaxis", "outName": "emptyReduceAxisTests/reduce_max/rank2_axis1Keep", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_min_dynamicaxis", "outName": "emptyReduceAxisTests/reduce_min/rank1_axisNone", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_min_dynamicaxis", "outName": "emptyReduceAxisTests/reduce_min/rank1_axis0_keep", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":0, "keepdims":True},
        # {"opName": "reduce_min_dynamicaxis", "outName": "emptyReduceAxisTests/reduce_min/rank2_axis1Keep", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_prod", "outName": "emptyReduceAxisTests/reduce_prod/rank1_axisNone", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_prod", "outName": "emptyReduceAxisTests/reduce_prod/rank1_axis0_keep", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":0, "keepdims":True},
        # {"opName": "reduce_prod", "outName": "emptyReduceAxisTests/reduce_prod/rank2_axisNone", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_prod", "outName": "emptyReduceAxisTests/reduce_prod/rank2_axisNoneKeep", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_prod", "outName": "emptyReduceAxisTests/reduce_prod/rank2_axis0", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_prod", "outName": "emptyReduceAxisTests/reduce_prod/rank2_axis1Keep", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_mean", "outName": "emptyReduceAxisTests/reduce_mean/rank1_axisNone", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_mean", "outName": "emptyReduceAxisTests/reduce_mean/rank1_axis0_keep", "varShapes":[[0]], "varTypes":["float32"], "varInit":["empty"], "axis":0, "keepdims":True},
        # {"opName": "reduce_mean", "outName": "emptyReduceAxisTests/reduce_mean/rank2_axisNone", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_mean", "outName": "emptyReduceAxisTests/reduce_mean/rank2_axisNoneKeep", "varShapes":[[0,2]], "varTypes":["float64"], "varInit":["empty"], "axis":None, "keepdims":True},
        # {"opName": "reduce_mean", "outName": "emptyReduceAxisTests/reduce_mean/rank2_axis0", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":False},
        # {"opName": "reduce_mean", "outName": "emptyReduceAxisTests/reduce_mean/rank2_axis1Keep", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "axis":None, "keepdims":True},

        # {"opName": "reduce_sum", "outName": "emptyReduceAxisTests/reduce_sum/rank1", "varShapes":[[3]], "varTypes":["float32"], "varInit":["uniform_int5"], "axis":(), "keepdims":False},
        # {"opName": "reduce_sum", "outName": "emptyReduceAxisTests/reduce_sum/rank1_keep", "varShapes":[[3]], "varTypes":["float32"], "varInit":["uniform_int5"], "axis":(), "keepdims":True},
        # {"opName": "reduce_sum", "outName": "emptyReduceAxisTests/reduce_sum/rank3", "varShapes":[[2,3,4]], "varTypes":["float32"], "varInit":["uniform_int5"], "axis":(), "keepdims":False},
        # #Can't save next 2... ""No variables to save"" in TF persistor
        # {"opName": "reduce_sum", "outName": "emptyReduceAxisTests/reduce_sum/rank2_empty", "varShapes":[[2,0]], "varTypes":["float32"], "varInit":["empty"], "axis":(), "keepdims":False},
        # {"opName": "reduce_sum", "outName": "emptyReduceAxisTests/reduce_sum/rank2_empty_keep", "varShapes":[[0,2]], "varTypes":["float32"], "varInit":["empty"], "axis":(), "keepdims":True},
        # {"opName": "reduce_prod", "outName": "emptyReduceAxisTests/reduce_prod/rank1", "varShapes":[[3]], "varTypes":["float32"], "varInit":["uniform_int5"], "axis":(), "keepdims":False},
        # {"opName": "reduce_prod", "outName": "emptyReduceAxisTests/reduce_prod/rank1_keep", "varShapes":[[3]], "varTypes":["float32"], "varInit":["uniform_int5"], "axis":(), "keepdims":True},
        # {"opName": "reduce_prod", "outName": "emptyReduceAxisTests/reduce_prod/rank2", "varShapes":[[2,3]], "varTypes":["float32"], "varInit":["uniform_int5"], "axis":(), "keepdims":False},
        # {"opName": "reduce_prod", "outName": "emptyReduceAxisTests/reduce_prod/rank2_keep", "varShapes":[[2,3]], "varTypes":["float32"], "varInit":["uniform_int5"], "axis":(), "keepdims":True},
        # {"opName": "reduce_logsumexp", "outName": "emptyReduceAxisTests/reduce_logsumexp/rank1", "varShapes":[[3]], "varTypes":["float32"], "varInit":["uniform_int5"], "axis":(), "keepdims":False},
        # {"opName": "reduce_logsumexp", "outName": "emptyReduceAxisTests/reduce_logsumexp/rank2", "varShapes":[[2,3]], "varTypes":["float32"], "varInit":["uniform_int5"], "axis":(), "keepdims":False},
        # {"opName": "reduce_all", "outName": "emptyReduceAxisTests/reduce_all/rank1", "varShapes":[[3]], "varTypes":["bool"], "varInit":["boolean"], "axis":(), "keepdims":False},
        # {"opName": "reduce_all", "outName": "emptyReduceAxisTests/reduce_all/rank1_keep", "varShapes":[[3]], "varTypes":["bool"], "varInit":["boolean"], "axis":(), "keepdims":True},
        # {"opName": "reduce_all", "outName": "emptyReduceAxisTests/reduce_all/rank2", "varShapes":[[2,3]], "varTypes":["bool"], "varInit":["boolean"], "axis":(), "keepdims":False},
        # {"opName": "reduce_all", "outName": "emptyReduceAxisTests/reduce_all/rank2_keep", "varShapes":[[2,3]], "varTypes":["bool"], "varInit":["boolean"], "axis":(), "keepdims":True},
        # {"opName": "reduce_any", "outName": "emptyReduceAxisTests/reduce_any/rank1", "varShapes":[[3]], "varTypes":["bool"], "varInit":["boolean"], "axis":(), "keepdims":False},
        # {"opName": "reduce_any", "outName": "emptyReduceAxisTests/reduce_any/rank1_keep", "varShapes":[[3]], "varTypes":["bool"], "varInit":["boolean"], "axis":(), "keepdims":True},
        # {"opName": "reduce_any", "outName": "emptyReduceAxisTests/reduce_any/rank2", "varShapes":[[2,3]], "varTypes":["bool"], "varInit":["boolean"], "axis":(), "keepdims":False},
        # {"opName": "reduce_any", "outName": "emptyReduceAxisTests/reduce_any/rank2_keep", "varShapes":[[2,3]], "varTypes":["bool"], "varInit":["boolean"], "axis":(), "keepdims":True},
        # {"opName": "reduce_min", "outName": "emptyReduceAxisTests/reduce_min/rank1", "varShapes":[[3]], "varTypes":["float32"], "varInit":["uniform_int5"], "axis":(), "keepdims":False},
        # {"opName": "reduce_min", "outName": "emptyReduceAxisTests/reduce_min/rank1_keep", "varShapes":[[3]], "varTypes":["float32"], "varInit":["uniform_int5"], "axis":(), "keepdims":True},
        # {"opName": "reduce_max", "outName": "emptyReduceAxisTests/reduce_max/rank2", "varShapes":[[2,3]], "varTypes":["float32"], "varInit":["uniform_int5"], "axis":(), "keepdims":False},
        # {"opName": "reduce_max", "outName": "emptyReduceAxisTests/reduce_max/rank2_keep", "varShapes":[[2,3]], "varTypes":["float32"], "varInit":["uniform_int5"], "axis":(), "keepdims":True},

        # {"opName": "multinomial", "outName": "multinomial/logits/sample/rank1", "varShapes":[[2]], "varTypes":["float32"], "varInit":["uniform_int5"], "total_count":4., "sample_shape": 5},
        # {"opName": "multinomial", "outName": "multinomial/logits/sample/rank2", "varShapes":[[2, 3]], "varTypes":["float32"], "varInit":["uniform_int5"], "total_count":[4., 2], "sample_shape": 5},
        # {"opName": "multinomial_with_p", "outName": "multinomial/prob/sample/rank1", "varShapes":[[2]], "varTypes":["float32"], "varInit":["uniform_int5"], "total_count":4., "sample_shape": 5},
        # {"opName": "multinomial_with_p", "outName": "multinomial/prob/sample/rank2", "varShapes":[[2, 3]], "varTypes":["float32"], "varInit":["uniform_int5"], "total_count":[4., 2], "sample_shape": 5},

        # {"opName": "add", "outName": "ragged/add/2d", "varShapes":[[], []], "varTypes":["int32", "int32"], "varInit":["ragged2d", "one"]},
        # {"opName": "identity", "outName": "ragged/identity/2d", "varShapes":[[]], "varTypes":["float32"], "varInit":["ragged2d"]},
        # {"opName": "reduce_mean", "outName": "ragged/reduce_mean/2d_a0", "varShapes":[[]], "varTypes":["float32"], "varInit":["ragged2d"], "axis":0, "keepdims":False},
        # {"opName": "reduce_mean", "outName": "ragged/reduce_mean/2d_a1", "varShapes":[[]], "varTypes":["float32"], "varInit":["ragged2d"], "axis":1, "keepdims":False},
        # {"opName": "sqrt", "outName": "ragged/sqrt/2d", "varShapes":[[]], "varTypes":["float32"], "varInit":["ragged2d"]},

        # {"opName": "strings_split", "outName": "ragged/sqrt/2d", "varShapes":[[]], "varTypes":["string"], "varInit":["string2"], "split":" "},

        #{"opName": "bitcast", "outName": "bitcast/from_float32_to_int8", "varShapes":[[1]], "varTypes":["float32"], "varInit":["uniform"], "output":tf.int8},
        #{"opName": "bitcast", "outName": "bitcast/from_bfloat16_to_int8", "varShapes": [[2]], "varTypes": ["bfloat16"], "varInit": ["uniform"], "output":tf.int8},
        #{"opName": "bitcast", "outName": "bitcast/from_half_to_int8", "varShapes": [[1]], "varTypes": ["half"], "varInit": ["uniform"], "output":tf.int8},
        #{"opName": "bitcast", "outName": "bitcast/from_float64_to_int8", "varShapes": [[2]], "varTypes": ["float64"], "varInit": ["uniform"], "output":tf.int8},
        #{"opName": "bitcast", "outName": "bitcast/from_int32_to_int8", "varShapes": [[1]], "varTypes": ["int32"], "varInit": ["uniform_int2"], "output":tf.int8},
        #{"opName": "bitcast", "outName": "bitcast/from_int64_to_int8", "varShapes": [[1]], "varTypes": ["int64"], "varInit": ["uniform_int2"], "output":tf.int8},
        # {"opName": "bitcast", "outName": "bitcast/from_rank2_float32_to_int8", "varShapes":[[1,1]], "varTypes":["float32"], "varInit":["uniform"], "output":tf.int8},
        # {"opName": "bitcast", "outName": "bitcast/from_rank2_bfloat16_to_int8", "varShapes": [[2,2]], "varTypes": ["bfloat16"], "varInit": ["uniform"], "output":tf.int8},
        # {"opName": "bitcast", "outName": "bitcast/from_rank2_half_to_int8", "varShapes": [[1,1]], "varTypes": ["half"], "varInit": ["uniform"], "output":tf.int8},
        # {"opName": "bitcast", "outName": "bitcast/from_rank2_float64_to_int8", "varShapes": [[2,2]], "varTypes": ["float64"], "varInit": ["uniform"], "output":tf.int8},
        # {"opName": "bitcast", "outName": "bitcast/from_rank2_int32_to_int8", "varShapes": [[1,1]], "varTypes": ["int32"], "varInit": ["uniform_int2"], "output":tf.int8},
        # {"opName": "bitcast", "outName": "bitcast/from_rank2_int64_to_int8", "varShapes": [[1,1]], "varTypes": ["int64"], "varInit": ["uniform_int2"], "output":tf.int8},
        # {"opName": "bitcast", "outName": "bitcast/from_rank3_float32_to_int8", "varShapes":[[1,1,1]], "varTypes":["float32"], "varInit":["uniform"], "output":tf.int8},
        # {"opName": "bitcast", "outName": "bitcast/from_rank3_bfloat16_to_int8", "varShapes": [[2,2,2]], "varTypes": ["bfloat16"], "varInit": ["uniform"], "output":tf.int8},
        # {"opName": "bitcast", "outName": "bitcast/from_rank3_half_to_int8", "varShapes": [[1,1,1]], "varTypes": ["half"], "varInit": ["uniform"], "output":tf.int8},
        # {"opName": "bitcast", "outName": "bitcast/from_rank3_float64_to_int8", "varShapes": [[2,2,2]], "varTypes": ["float64"], "varInit": ["uniform"], "output":tf.int8},
        # {"opName": "bitcast", "outName": "bitcast/from_rank3_int32_to_int8", "varShapes": [[1,1,1]], "varTypes": ["int32"], "varInit": ["uniform_int2"], "output":tf.int8},
        # {"opName": "bitcast", "outName": "bitcast/from_rank3_int64_to_int8", "varShapes": [[1,1,1]], "varTypes": ["int64"], "varInit": ["uniform_int2"], "output":tf.int8},
        #{"opName": "bitcast", "outName": "bitcast/emptyArrayTest/from_uint32_to_int8", "varShapes": [[0]], "varTypes": ["uint32"], "varInit": ["empty"], "output":tf.int8},
        # {"opName": "bitcast", "outName": "bitcast/emptyArrayTest/from_rank2_uint32_to_int8", "varShapes": [[0,0]], "varTypes": ["uint32"], "varInit": ["empty"], "output":tf.int8},
        #{"opName": "bitcast", "outName": "bitcast/emptyArrayTest/from_rank3_uint32_to_int8", "varShapes": [[0, 0, 0]], "varTypes": ["uint32"], "varInit": ["empty"], "output": tf.int8},

        # {"opName": "bitcast", "outName": "bitcast/from_float32_to_int16", "varShapes":[[1]], "varTypes":["float32"], "varInit":["uniform"], "output":tf.int16},
        #  {"opName": "bitcast", "outName": "bitcast/from_bfloat16_to_int16", "varShapes": [[2]], "varTypes": ["bfloat16"], "varInit": ["uniform"], "output":tf.int16},
        #  {"opName": "bitcast", "outName": "bitcast/from_half_to_int16", "varShapes": [[1]], "varTypes": ["half"], "varInit": ["uniform"], "output":tf.int16},
        #  {"opName": "bitcast", "outName": "bitcast/from_float64_to_int16", "varShapes": [[2]], "varTypes": ["float64"], "varInit": ["uniform"], "output":tf.int16},
        #  {"opName": "bitcast", "outName": "bitcast/from_int32_to_int16", "varShapes": [[1]], "varTypes": ["int32"], "varInit": ["uniform_int2"], "output":tf.int16},
        #  {"opName": "bitcast", "outName": "bitcast/from_int64_to_int16", "varShapes": [[1]], "varTypes": ["int64"], "varInit": ["uniform_int2"], "output":tf.int16},
        #  {"opName": "bitcast", "outName": "bitcast/from_rank2_float32_to_int16", "varShapes":[[1,1]], "varTypes":["float32"], "varInit":["uniform"], "output":tf.int16},
        #  {"opName": "bitcast", "outName": "bitcast/from_rank2_bfloat16_to_int16", "varShapes": [[2,1]], "varTypes": ["bfloat16"], "varInit": ["uniform"], "output":tf.int16},
        #  {"opName": "bitcast", "outName": "bitcast/from_rank2_half_to_int16", "varShapes": [[1,1]], "varTypes": ["half"], "varInit": ["uniform"], "output":tf.int16},
        #  {"opName": "bitcast", "outName": "bitcast/from_rank2_float64_to_int16", "varShapes": [[2,1]], "varTypes": ["float64"], "varInit": ["uniform"], "output":tf.int16},
        #  {"opName": "bitcast", "outName": "bitcast/from_rank2_int32_to_int16", "varShapes": [[1,1]], "varTypes": ["int32"], "varInit": ["uniform_int2"], "output":tf.int16},
        #  {"opName": "bitcast", "outName": "bitcast/from_rank2_int64_to_int16", "varShapes": [[1,1]], "varTypes": ["int64"], "varInit": ["uniform_int2"], "output":tf.int16},
        #  {"opName": "bitcast", "outName": "bitcast/from_rank3_float32_to_int16", "varShapes": [[1, 1, 1]],  "varTypes": ["float32"], "varInit": ["uniform"], "output": tf.int16},
        #{"opName": "bitcast", "outName": "bitcast/from_rank3_bfloat16_to_int16", "varShapes": [[2, 1, 1]],  "varTypes": ["bfloat16"], "varInit": ["uniform"], "output": tf.int16},
        #{"opName": "bitcast", "outName": "bitcast/from_rank3_half_to_int16", "varShapes": [[1, 1, 1]],  "varTypes": ["half"], "varInit": ["uniform"], "output": tf.int16},
        #{"opName": "bitcast", "outName": "bitcast/from_rank3_float64_to_int16", "varShapes": [[2, 1, 1]],"varTypes": ["float64"], "varInit": ["uniform"], "output": tf.int16},
        #{"opName": "bitcast", "outName": "bitcast/from_rank3_int32_to_int16", "varShapes": [[1, 1, 1]], "varTypes": ["int32"], "varInit": ["uniform_int2"], "output": tf.int16},
        #{"opName": "bitcast", "outName": "bitcast/from_rank3_int64_to_int16", "varShapes": [[1, 1, 1]], "varTypes": ["int64"], "varInit": ["uniform_int2"], "output": tf.int16},
        # {"opName": "bitcast", "outName": "bitcast/emptyArrayTest/from_uint32_to_int16", "varShapes": [[0]], "varTypes": ["uint32"], "varInit": ["empty"], "output":tf.int16},

        # {"opName": "bitcast", "outName": "bitcast/from_float32_to_int32", "varShapes":[[2]], "varTypes":["float32"], "varInit":["uniform"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_bfloat16_to_int32", "varShapes": [[2]], "varTypes": ["bfloat16"], "varInit": ["uniform"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_half_to_int32", "varShapes": [[2]], "varTypes": ["half"], "varInit": ["uniform"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_float64_to_int32", "varShapes": [[2]], "varTypes": ["float64"], "varInit": ["uniform"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_int32_to_int32", "varShapes": [[2]], "varTypes": ["int32"], "varInit": ["uniform_int2"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_int64_to_int32", "varShapes": [[2]], "varTypes": ["int64"], "varInit": ["uniform_int2"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_float32_to_int32", "varShapes":[[2]], "varTypes":["float32"], "varInit":["uniform"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_bfloat16_to_int32", "varShapes": [[2]], "varTypes": ["bfloat16"], "varInit": ["uniform"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_half_to_int32", "varShapes": [[2]], "varTypes": ["half"], "varInit": ["uniform"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_float64_to_int32", "varShapes": [[2]], "varTypes": ["float64"], "varInit": ["uniform"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_int32_to_int32", "varShapes": [[2]], "varTypes": ["int32"], "varInit": ["uniform_int2"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_int64_to_int32", "varShapes": [[2]], "varTypes": ["int64"], "varInit": ["uniform_int2"], "output":tf.int32},
        #{"opName": "bitcast", "outName": "bitcast/from_rank2_float32_to_int32", "varShapes": [[2,2]], "varTypes": ["float32"], "varInit": ["uniform"], "output": tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_rank2_bfloat16_to_int32", "varShapes": [[2,2]], "varTypes": ["bfloat16"], "varInit": ["uniform"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_rank2_half_to_int32", "varShapes": [[2,2]], "varTypes": ["half"], "varInit": ["uniform"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_rank2_float64_to_int32", "varShapes": [[2,2]], "varTypes": ["float64"], "varInit": ["uniform"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_rank2_int32_to_int32", "varShapes": [[2,2]], "varTypes": ["int32"], "varInit": ["uniform_int2"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_rank2_int64_to_int32", "varShapes": [[2,2]], "varTypes": ["int64"], "varInit": ["uniform_int2"], "output":tf.int32},
        #{"opName": "bitcast", "outName": "bitcast/from_rank3_float32_to_int32", "varShapes": [[2,2,2]], "varTypes": ["float32"], "varInit": ["uniform"], "output": tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_rank3_bfloat16_to_int32", "varShapes": [[2,2,2]], "varTypes": ["bfloat16"], "varInit": ["uniform"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_rank3_half_to_int32", "varShapes": [[2,2,2]], "varTypes": ["half"], "varInit": ["uniform"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_rank3_float64_to_int32", "varShapes": [[2,2,2]], "varTypes": ["float64"], "varInit": ["uniform"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_rank3_int32_to_int32", "varShapes": [[2,2,2]], "varTypes": ["int32"], "varInit": ["uniform_int2"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_rank3_int64_to_int32", "varShapes": [[2,2,2]], "varTypes": ["int64"], "varInit": ["uniform_int2"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/emptyArrayTest/from_uint32_to_int32", "varShapes": [[0]], "varTypes": ["uint32"], "varInit": ["empty"], "output":tf.int32},


        #{"opName": "bitcast", "outName": "bitcast/from_float32_to_int32", "varShapes":[[2]], "varTypes":["float32"], "varInit":["uniform"], "output":tf.int32},
        #{"opName": "bitcast", "outName": "bitcast/from_bfloat16_to_int32", "varShapes": [[2]], "varTypes": ["bfloat16"], "varInit": ["uniform"], "output":tf.int32},
        #{"opName": "bitcast", "outName": "bitcast/from_half_to_int32", "varShapes": [[2]], "varTypes": ["half"], "varInit": ["uniform"], "output":tf.int32},
        #{"opName": "bitcast", "outName": "bitcast/from_float64_to_int32", "varShapes": [[2]], "varTypes": ["float64"], "varInit": ["uniform"], "output":tf.int32},
        #{"opName": "bitcast", "outName": "bitcast/from_int32_to_int32", "varShapes": [[2]], "varTypes": ["int32"], "varInit": ["uniform_int2"], "output":tf.int32},
        #{"opName": "bitcast", "outName": "bitcast/from_int64_to_int32", "varShapes": [[2]], "varTypes": ["int64"], "varInit": ["uniform_int2"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_rank2_float32_to_int32", "varShapes": [[2,2]], "varTypes": ["float32"], "varInit": ["uniform"], "output": tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_rank2_bfloat16_to_int32", "varShapes": [[2,2]], "varTypes": ["bfloat16"], "varInit": ["uniform"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_rank2_half_to_int32", "varShapes": [[2,2]], "varTypes": ["half"], "varInit": ["uniform"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_rank2_float64_to_int32", "varShapes": [[2,2]], "varTypes": ["float64"], "varInit": ["uniform"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_rank2_int32_to_int32", "varShapes": [[2,2]], "varTypes": ["int32"], "varInit": ["uniform_int2"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_rank2_int64_to_int32", "varShapes": [[2,2]], "varTypes": ["int64"], "varInit": ["uniform_int2"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_rank3_float32_to_int32", "varShapes": [[2,2,2]], "varTypes": ["float32"], "varInit": ["uniform"], "output": tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_rank3_bfloat16_to_int32", "varShapes": [[2,2,2]], "varTypes": ["bfloat16"], "varInit": ["uniform"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_rank3_half_to_int32", "varShapes": [[2,2,2]], "varTypes": ["half"], "varInit": ["uniform"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_rank3_float64_to_int32", "varShapes": [[2,2,2]], "varTypes": ["float64"], "varInit": ["uniform"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_rank3_int32_to_int32", "varShapes": [[2,2,2]], "varTypes": ["int32"], "varInit": ["uniform_int2"], "output":tf.int32},
        # {"opName": "bitcast", "outName": "bitcast/from_rank3_int64_to_int32", "varShapes": [[2,2,2]], "varTypes": ["int64"], "varInit": ["uniform_int2"], "output":tf.int32},
        #{"opName": "bitcast", "outName": "bitcast/emptyArrayTest/from_uint32_to_int32", "varShapes": [[0]], "varTypes": ["uint32"], "varInit": ["empty"], "output":tf.int32},

         #{"opName": "bitcast", "outName": "bitcast/from_float32_to_int64", "varShapes": [[2]], "varTypes": ["float32"], "varInit": ["uniform"], "output": tf.int64},
         #{"opName": "bitcast", "outName": "bitcast/from_bfloat16_to_int64", "varShapes": [[4]], "varTypes": ["bfloat16"], "varInit": ["uniform"], "output":tf.int64},
         #{"opName": "bitcast", "outName": "bitcast/from_half_to_int64", "varShapes": [[4]], "varTypes": ["half"], "varInit": ["uniform"], "output":tf.int64},
         #{"opName": "bitcast_float64", "outName": "bitcast/from_float64_to_int64", "varShapes": [[1]], "varTypes": ["float64"], "varInit": ["uniform_int10"], "output":tf.int64},
         #{"opName": "bitcast", "outName": "bitcast/from_int32_to_int64", "varShapes": [[2]], "varTypes": ["int32"], "varInit": ["uniform_int2"], "output":tf.int64},
         #{"opName": "bitcast", "outName": "bitcast/from_int64_to_int64", "varShapes": [[1]], "varTypes": ["int64"], "varInit": ["uniform_int2"], "output":tf.int64},
         #{"opName": "bitcast", "outName": "bitcast/from_rank2_float32_to_int64", "varShapes": [[2,2]], "varTypes": ["float32"], "varInit": ["uniform"], "output": tf.int64},
         #{"opName": "bitcast", "outName": "bitcast/from_rank2_bfloat16_to_int64", "varShapes": [[4,4]], "varTypes": ["bfloat16"], "varInit": ["uniform"], "output":tf.int64},
         #{"opName": "bitcast", "outName": "bitcast/from_rank2_half_to_int64", "varShapes": [[4,4]], "varTypes": ["half"], "varInit": ["uniform"], "output":tf.int64},
         #{"opName": "bitcast_float64", "outName": "bitcast/from_rank2_float64_to_int64", "varShapes": [[1,1]], "varTypes": ["float64"], "varInit": ["uniform_int10"], "output":tf.int64},
         #{"opName": "bitcast", "outName": "bitcast/from_rank2_int32_to_int64", "varShapes": [[2,2]], "varTypes": ["int32"], "varInit": ["uniform_int2"], "output":tf.int64},
         #{"opName": "bitcast", "outName": "bitcast/from_rank2_int64_to_int64", "varShapes": [[1,1]], "varTypes": ["int64"], "varInit": ["uniform_int2"], "output":tf.int64},
         #{"opName": "bitcast", "outName": "bitcast/from_rank3_float32_to_int64", "varShapes": [[2, 2,2]],"varTypes": ["float32"], "varInit": ["uniform"], "output": tf.int64},
         #{"opName": "bitcast", "outName": "bitcast/from_rank3_bfloat16_to_int64", "varShapes": [[4, 4,4]], "varTypes": ["bfloat16"], "varInit": ["uniform"], "output": tf.int64},
         #{"opName": "bitcast", "outName": "bitcast/from_rank3_half_to_int64", "varShapes": [[4, 4,4]], "varTypes": ["half"], "varInit": ["uniform"], "output": tf.int64},
         #{"opName": "bitcast", "outName": "bitcast/from_rank3_float64_to_int64", "varShapes": [[1, 1,1]], "varTypes": ["float64"], "varInit": ["uniform"], "output": tf.int64},
         #{"opName": "bitcast", "outName": "bitcast/from_rank3_int32_to_int64", "varShapes": [[2, 2,2]],  "varTypes": ["int32"], "varInit": ["uniform_int2"], "output": tf.int64},
         #{"opName": "bitcast", "outName": "bitcast/from_rank3_int64_to_int64", "varShapes": [[1, 1,1]],  "varTypes": ["int64"], "varInit": ["uniform_int2"], "output": tf.int64},
         #{"opName": "bitcast", "outName": "bitcast/emptyArrayTest/from_uint32_to_int64", "varShapes": [[0]], "varTypes": ["uint32"], "varInit": ["empty"], "output":tf.int64},

         #{"opName": "bitcast", "outName": "bitcast/from_float32_to_uint8", "varShapes": [[2]], "varTypes": ["float32"], "varInit": ["uniform"], "output": tf.uint8},
         #{"opName": "bitcast", "outName": "bitcast/from_bfloat16_to_uint8", "varShapes": [[4]], "varTypes": ["bfloat16"], "varInit": ["uniform"], "output":tf.uint8},
         #{"opName": "bitcast", "outName": "bitcast/from_half_to_uint8", "varShapes": [[4]], "varTypes": ["half"], "varInit": ["uniform"], "output":tf.uint8},
         #{"opName": "bitcast", "outName": "bitcast/from_float64_to_uint8", "varShapes": [[1]], "varTypes": ["float64"], "varInit": ["uniform"], "output":tf.uint8},
         #{"opName": "bitcast", "outName": "bitcast/from_int32_to_uint8", "varShapes": [[2]], "varTypes": ["int32"], "varInit": ["uniform_int2"], "output":tf.uint8},
         #{"opName": "bitcast", "outName": "bitcast/from_int64_to_uint8", "varShapes": [[1]], "varTypes": ["int64"], "varInit": ["uniform_int2"], "output":tf.uint8},
         #{"opName": "bitcast", "outName": "bitcast/emptyArrayTest/from_uint32_to_uint8", "varShapes": [[0]], "varTypes": ["uint32"], "varInit": ["empty"], "output":tf.uint8},

        #{"opName": "bitcast", "outName": "bitcast/from_float32_to_uint32", "varShapes": [[2]], "varTypes": ["float32"], "varInit": ["uniform"], "output": tf.uint32},
        #{"opName": "bitcast", "outName": "bitcast/from_bfloat16_to_uint32", "varShapes": [[2]], "varTypes": ["bfloat16"],"varInit": ["uniform"], "output": tf.uint32},
        #{"opName": "bitcast", "outName": "bitcast/from_half_to_uint32", "varShapes": [[2]], "varTypes": ["half"],"varInit": ["uniform"], "output": tf.uint32},
        #{"opName": "bitcast", "outName": "bitcast/from_float64_to_uint32", "varShapes": [[1]], "varTypes": ["float64"],"varInit": ["uniform"], "output": tf.uint32},
        #{"opName": "bitcast", "outName": "bitcast/from_int32_to_uint32", "varShapes": [[1]], "varTypes": ["int32"],"varInit": ["uniform_int2"], "output": tf.uint32},
        #{"opName": "bitcast", "outName": "bitcast/from_int64_to_uint32", "varShapes": [[1]], "varTypes": ["int64"],"varInit": ["uniform_int2"], "output": tf.uint32},
        #{"opName": "bitcast", "outName": "bitcast/emptyArrayTest/from_uint32_to_uint16", "varShapes": [[0]],"varTypes": ["uint32"], "varInit": ["empty"], "output": tf.uint32},

         #{"opName": "bitcast", "outName": "bitcast/from_float32_to_uint64", "varShapes": [[2]], "varTypes": ["float32"], "varInit": ["uniform"], "output": tf.uint64},
         #{"opName": "bitcast", "outName": "bitcast/from_bfloat16_to_uint64", "varShapes": [[4]], "varTypes": ["bfloat16"],"varInit": ["uniform"], "output": tf.uint64},
         #{"opName": "bitcast", "outName": "bitcast/from_half_to_uint64", "varShapes": [[4]], "varTypes": ["half"],"varInit": ["uniform"], "output": tf.uint64},
         #{"opName": "bitcast_float64", "outName": "bitcast/from_float64_to_uint64", "varShapes": [[1]], "varTypes": ["int64"],"varInit": ["uniform_int10"], "output": tf.uint64},
         #{"opName": "bitcast", "outName": "bitcast/from_int32_to_uint64", "varShapes": [[2]], "varTypes": ["int32"],"varInit": ["uniform_int2"], "output": tf.uint64},
         #{"opName": "bitcast", "outName": "bitcast/from_int64_to_uint64", "varShapes": [[1]], "varTypes": ["int64"],"varInit": ["uniform_int2"], "output": tf.uint64},
        # {"opName": "bitcast", "outName": "bitcast/emptyArrayTest/from_uint32_to_uint16", "varShapes": [[0]],"varTypes": ["uint32"], "varInit": ["empty"], "output": tf.uint32},

        # {"opName": "bitcast", "outName": "bitcast/from_float32_to_uint16", "varShapes": [[2]], "varTypes": ["float32"], "varInit": ["uniform"], "output": tf.uint16},
        # {"opName": "bitcast", "outName": "bitcast/from_bfloat16_to_uint16", "varShapes": [[4]], "varTypes": ["bfloat16"],"varInit": ["uniform"], "output": tf.uint16},
        # {"opName": "bitcast", "outName": "bitcast/from_half_to_uint16", "varShapes": [[4]], "varTypes": ["half"],"varInit": ["uniform"], "output": tf.uint16},
        # {"opName": "bitcast", "outName": "bitcast/from_float64_to_uint16", "varShapes": [[1]], "varTypes": ["float64"],"varInit": ["uniform"], "output": tf.uint16},
        # {"opName": "bitcast", "outName": "bitcast/from_int32_to_uint16", "varShapes": [[2]], "varTypes": ["int32"],"varInit": ["uniform_int2"], "output": tf.uint16},
        # {"opName": "bitcast", "outName": "bitcast/from_int64_to_uint16", "varShapes": [[1]], "varTypes": ["int64"],"varInit": ["uniform_int2"], "output": tf.uint16},
        # {"opName": "bitcast", "outName": "bitcast/emptyArrayTest/from_uint32_to_uint16", "varShapes": [[0]],"varTypes": ["uint32"], "varInit": ["empty"], "output": tf.uint16},

        #{"opName": "bitcast", "outName": "bitcast/from_float32_to_bfloat16", "varShapes": [[2]], "varTypes": ["float32"], "varInit": ["uniform"], "output": tf.bfloat16},
        #{"opName": "bitcast", "outName": "bitcast/from_bfloat16_to_bfloat16", "varShapes": [[2]], "varTypes": ["bfloat16"],"varInit": ["uniform"], "output": tf.bfloat16},
        #{"opName": "bitcast", "outName": "bitcast/from_half_to_bfloat16", "varShapes": [[2]], "varTypes": ["half"],"varInit": ["uniform"], "output": tf.bfloat16},
        #{"opName": "bitcast", "outName": "bitcast/from_float64_to_bfloat16", "varShapes": [[2]], "varTypes": ["float64"],"varInit": ["uniform"], "output": tf.bfloat16},
        #{"opName": "bitcast", "outName": "bitcast/from_int32_to_bfloat16", "varShapes": [[2]], "varTypes": ["int32"],"varInit": ["uniform_int2"], "output": tf.bfloat16},
        #{"opName": "bitcast", "outName": "bitcast/from_int64_to_bfloat16", "varShapes": [[2]], "varTypes": ["int64"],"varInit": ["uniform_int2"], "output": tf.bfloat16},
        #{"opName": "bitcast", "outName": "bitcast/emptyArrayTest/from_uint32_to_bfloat16", "varShapes": [[0]],"varTypes": ["uint32"], "varInit": ["empty"], "output": tf.bfloat16},

         #{"opName": "bitcast", "outName": "bitcast/from_float32_to_half", "varShapes": [[2]], "varTypes": ["float32"], "varInit": ["uniform"], "output": tf.half},
         #{"opName": "bitcast", "outName": "bitcast/from_bfloat16_to_half", "varShapes": [[2]], "varTypes": ["bfloat16"],"varInit": ["uniform"], "output": tf.half},
         #{"opName": "bitcast", "outName": "bitcast/from_half_to_half", "varShapes": [[2]], "varTypes": ["half"],"varInit": ["uniform"], "output": tf.half},
         #{"opName": "bitcast", "outName": "bitcast/from_float64_to_half", "varShapes": [[2]], "varTypes": ["float64"],"varInit": ["uniform"], "output": tf.half},
         #{"opName": "bitcast", "outName": "bitcast/from_int32_to_half", "varShapes": [[2]], "varTypes": ["int32"],"varInit": ["uniform_int2"], "output": tf.half},
         #{"opName": "bitcast", "outName": "bitcast/from_int64_to_half", "varShapes": [[2]], "varTypes": ["int64"],"varInit": ["uniform_int2"], "output": tf.half},
         #{"opName": "bitcast", "outName": "bitcast/emptyArrayTest/from_uint32_to_half", "varShapes": [[0]],"varTypes": ["uint32"], "varInit": ["empty"], "output": tf.half},

        #{"opName": "bitcast", "outName": "bitcast/from_float32_to_float32", "varShapes": [[2]], "varTypes": ["float32"], "varInit": ["uniform"], "output": tf.float32},
        #{"opName": "bitcast", "outName": "bitcast/from_bfloat16_to_float32", "varShapes": [[2]], "varTypes": ["bfloat16"], "varInit": ["uniform"], "output": tf.float32},
        #{"opName": "bitcast", "outName": "bitcast/from_half_to_float32", "varShapes": [[2]], "varTypes": ["half"], "varInit": ["uniform"], "output": tf.float32},
        #{"opName": "bitcast", "outName": "bitcast/from_float64_to_float32", "varShapes": [[2]], "varTypes": ["float64"], "varInit": ["uniform"], "output": tf.float32},
        #{"opName": "bitcast", "outName": "bitcast/from_int32_to_float32", "varShapes": [[2]], "varTypes": ["int32"],"varInit": ["uniform_int2"], "output": tf.float32},
        #{"opName": "bitcast", "outName": "bitcast/from_int64_to_float32", "varShapes": [[2]], "varTypes": ["int64"],"varInit": ["uniform_int2"], "output": tf.float32},
        #{"opName": "bitcast", "outName": "bitcast/emptyArrayTest/from_uint32_to_float32", "varShapes": [[0]],"varTypes": ["uint32"], "varInit": ["empty"], "output": tf.float32},

        #{"opName": "bitcast", "outName": "bitcast/from_float32_to_float64", "varShapes": [[2]], "varTypes": ["float32"], "varInit": ["uniform"], "output": tf.float64},
        #{"opName": "bitcast", "outName": "bitcast/from_bfloat16_to_float64", "varShapes": [[4]], "varTypes": ["bfloat16"],"varInit": ["uniform"], "output": tf.float64},
        #{"opName": "bitcast", "outName": "bitcast/from_half_to_float64", "varShapes": [[4]], "varTypes": ["half"], "varInit": ["uniform"], "output": tf.float64},
        #{"opName": "bitcast", "outName": "bitcast/from_float64_to_float64", "varShapes": [[1]], "varTypes": ["float64"], "varInit": ["uniform"], "output": tf.float64},
        #{"opName": "bitcast", "outName": "bitcast/from_int32_to_float64", "varShapes": [[2]], "varTypes": ["int32"], "varInit": ["uniform_int2"], "output": tf.float64},
        #{"opName": "bitcast", "outName": "bitcast/from_int64_to_float64", "varShapes": [[2]], "varTypes": ["int64"], "varInit": ["uniform_int2"], "output": tf.float64},
        #{"opName": "bitcast", "outName": "bitcast/emptyArrayTest/from_uint32_to_float64", "varShapes": [[0]], "varTypes": ["uint32"], "varInit": ["empty"], "output": tf.float64},

        # {"opName": "bitwise_and", "outName": "bitwise_and/rank2_int32", "varShapes":[[1,2], [1,2]], "varTypes":["int32", "int32"], "varInit":["uniform_int10", "uniform_int2"]},
        # {"opName": "bitwise_and", "outName": "bitwise_and/rank2_int64", "varShapes": [[1, 2], [1, 2]],"varTypes": ["int64", "int64"], "varInit": ["uniform_int10", "uniform_int2"]},
        # {"opName": "bitwise_and", "outName": "bitwise_and/rank3_int32", "varShapes":[[1,1, 2], [1,1, 2]], "varTypes":["int32", "int32"], "varInit":["uniform_int10", "uniform_int2"]},
        # {"opName": "bitwise_and", "outName": "bitwise_and/rank3_int64", "varShapes": [[1, 1, 2], [1, 1, 2]],"varTypes": ["int64", "int64"], "varInit": ["uniform_int10", "uniform_int2"]},

        # {"opName": "bitwise_or", "outName": "bitwise_or/rank2_int32", "varShapes":[[1,2], [1,2]], "varTypes":["int32", "int32"], "varInit":["uniform_int10", "uniform_int2"]},
        # {"opName": "bitwise_or", "outName": "bitwise_or/rank2_int64", "varShapes": [[1, 2], [1, 2]],"varTypes": ["int64", "int64"], "varInit": ["uniform_int10", "uniform_int2"]},
        # {"opName": "bitwise_or", "outName": "bitwise_or/rank3_int32", "varShapes":[[1,1, 2], [1,1, 2]], "varTypes":["int32", "int32"], "varInit":["uniform_int10", "uniform_int2"]},
        # {"opName": "bitwise_or", "outName": "bitwise_or/rank3_int64", "varShapes": [[1, 1, 2], [1, 1, 2]],"varTypes": ["int64", "int64"], "varInit": ["uniform_int10", "uniform_int2"]},

        #{"opName": "bitwise_or", "outName": "bitwise_or/emptyArrayTest/rank2_int32", "varShapes":[[0,0], [0,0]], "varTypes":["int32", "int32"], "varInit":["uniform_int10", "empty"]},
        #{"opName": "bitwise_or", "outName": "bitwise_or/emptyArrayTest/rank2_int64", "varShapes": [[0,0], [0,0]],"varTypes": ["int64", "int64"], "varInit": ["uniform_int10", "empty"]},
        #{"opName": "bitwise_or", "outName": "bitwise_or/emptyArrayTest/partial_rank2_int64", "varShapes": [[1,0], [0,0]],"varTypes": ["int64", "int64"], "varInit": ["uniform_int10", "empty"]},
        #{"opName": "bitwise_or", "outName": "bitwise_or/emptyArrayTest/rank3_int32", "varShapes": [[0, 0, 0], [0, 0, 0]], "varTypes": ["int32", "int32"], "varInit": ["uniform_int10", "empty"]},
        #{"opName": "bitwise_or", "outName": "bitwise_or/emptyArrayTest/partial_rank3_int32", "varShapes": [[0, 1, 0], [1, 0, 0]], "varTypes": ["int32", "int32"], "varInit": ["uniform_int10", "empty"]},
        #{"opName": "bitwise_or", "outName": "bitwise_or/emptyArrayTest/rank3_int64", "varShapes": [[0, 0, 0], [0, 0, 0]], "varTypes": ["int64", "int64"], "varInit": ["uniform_int10", "empty"]},

        #{"opName": "bitwise_xor", "outName": "bitwise_xor/rank2_int32", "varShapes":[[1,2], [1,2]], "varTypes":["int32", "int32"], "varInit":["uniform_int10", "uniform_int2"]},
        #{"opName": "bitwise_xor", "outName": "bitwise_xor/rank2_int64", "varShapes": [[1,2], [1,2]],"varTypes": ["int64", "int64"], "varInit": ["uniform_int10", "uniform_int2"]},
        # {"opName": "bitwise_xor", "outName": "bitwise_xor/partial_rank2_int64", "varShapes": [[1,0], [0,0]],"varTypes": ["int64", "int64"], "varInit": ["uniform_int10", "uniform_int2"]},
        #{"opName": "bitwise_xor", "outName": "bitwise_xor/rank3_int32", "varShapes": [[1, 1, 2], [1, 1, 2]], "varTypes": ["int32", "int32"], "varInit": ["uniform_int10", "uniform_int2"]},
        # {"opName": "bitwise_xor", "outName": "bitwise_xor/partial_rank3_int32", "varShapes": [[1, 0, 2], [0, 0, 2]], "varTypes": ["int32", "int32"], "varInit": ["uniform_int10", "uniform_int2"]},
        #{"opName": "bitwise_xor", "outName": "bitwise_xor/rank3_int64", "varShapes": [[1, 1, 2], [1, 1, 2]], "varTypes": ["int64", "int64"], "varInit": ["uniform_int10", "uniform_int2"]},

        #{"opName": "bitwise_xor", "outName": "bitwise_xor/emptyArrayTest/rank2_int32", "varShapes":[[0,0], [0,0]], "varTypes":["int32", "int32"], "varInit":["uniform_int10", "empty"]},
        #{"opName": "bitwise_xor", "outName": "bitwise_xor/emptyArrayTest/rank2_int64", "varShapes": [[0,0], [0,0]],"varTypes": ["int64", "int64"], "varInit": ["uniform_int10", "empty"]},
        #{"opName": "bitwise_xor", "outName": "bitwise_xor/emptyArrayTest/rank3_int32", "varShapes": [[0, 0, 0], [0, 0, 0]], "varTypes": ["int32", "int32"], "varInit": ["uniform_int10", "empty"]},
        #{"opName": "bitwise_xor", "outName": "bitwise_xor/emptyArrayTest/rank3_int64", "varShapes": [[0, 0, 0], [0, 0, 0]], "varTypes": ["int64", "int64"], "varInit": ["uniform_int10", "empty"]},

        #{"opName": "is_non_decreasing", "outName": "is_non_decreasing/rank2_float32", "varShapes":[[1,2]], "varTypes":["float32"], "varInit":["uniform"]},
        #{"opName": "is_non_decreasing", "outName": "is_non_decreasing/rank3_float32", "varShapes": [[1, 1, 2]], "varTypes": ["float32"], "varInit": ["uniform"]},
        #{"opName": "is_non_decreasing", "outName": "is_non_decreasing/rank2_float32", "varShapes":[[1,2]], "varTypes":["float32"], "varInit":["stdnormal"]},
        #{"opName": "is_non_decreasing", "outName": "is_non_decreasing/rank3_float32", "varShapes": [[1, 1, 2]], "varTypes": ["float32"], "varInit": ["stdnormal"]},
        #{"opName": "is_strictly_increasing", "outName": "is_strictly_increasing/rank2_float32", "varShapes":[[1,1]], "varTypes":["float32"], "varInit":["uniform"]},
        #{"opName": "is_strictly_increasing", "outName": "is_strictly_increasing/rank3_float32", "varShapes": [[1, 1, 2]],"varTypes": ["float32"], "varInit": ["uniform"]},
        #{"opName": "is_strictly_increasing", "outName": "is_strictly_increasing/rank2_float32", "varShapes":[[1,1]], "varTypes":["float32"], "varInit":["stdnormal"]},
        #{"opName": "is_strictly_increasing", "outName": "is_strictly_increasing/rank3_float32", "varShapes": [[1, 1, 2]],"varTypes": ["float32"], "varInit": ["stdnormal"]},
        #{"opName": "is_strictly_increasing", "outName": "is_strictly_increasing/emptyArrayTest/rank1_float32", "varShapes": [[0]], "varTypes": ["float32"], "varInit": ["uniform"]},
        #{"opName": "is_strictly_increasing", "outName": "is_strictly_increasing/emptyArrayTest/rank2_float32", "varShapes": [[0, 0]], "varTypes": ["float32"], "varInit": ["uniform"]},
        # {"opName": "is_strictly_increasing", "outName": "is_strictly_increasing/emptyArrayTest/rank3_float32", "varShapes": [[0, 0, 0]],"varTypes": ["float32"], "varInit": ["stdnormal"]},

        #{"opName": "log_softmax", "outName": "log_softmax/rank2_float32", "varShapes":[[1,2]], "varTypes":["float32"], "varInit":["uniform"], "axis":0},
        #{"opName": "log_softmax", "outName": "log_softmax/rank2_float64", "varShapes": [[1, 2]], "varTypes": ["float64"], "varInit": ["uniform"], "axis":0},
        #{"opName": "log_softmax", "outName": "log_softmax/rank2_half", "varShapes": [[1, 2]], "varTypes": ["half"], "varInit": ["uniform"], "axis":0},
        #{"opName": "log_softmax", "outName": "log_softmax/rank2_float32_with_axis", "varShapes": [[1, 2],[]],"varTypes": ["float32","int32"], "varInit": ["uniform","zero"], "axis":1},
        #{"opName": "log_softmax", "outName": "log_softmax/rank2_float64_with_axis", "varShapes": [[1, 2],[]],"varTypes": ["float64","int32"], "varInit": ["uniform","zero"], "axis":1},
        #{"opName": "log_softmax", "outName": "log_softmax/rank2_half_with_axis", "varShapes": [[1, 2],[]], "varTypes": ["half","int32"], "varInit": ["uniform","zero"],"axis":1},
        #{"opName": "log_softmax", "outName": "log_softmax/rank3_float32_with_axis", "varShapes": [[1, 2,3], []],"varTypes": ["float32", "int32"], "varInit": ["uniform", "zero"], "axis": 0},
        #{"opName": "log_softmax", "outName": "log_softmax/rank3_float64_with_axis", "varShapes": [[1, 2,3], []],"varTypes": ["float64", "int32"], "varInit": ["uniform", "zero"], "axis": 1},
        #{"opName": "log_softmax", "outName": "log_softmax/rank3_half_with_axis", "varShapes": [[1, 2, 3], []],"varTypes": ["half", "int32"], "varInit": ["uniform", "zero"], "axis": 2},
        #{"opName": "log_softmax", "outName": "log_softmax/rank3_float32", "varShapes": [[1, 2, 3],], "varTypes": ["float32"], "varInit": ["uniform"], "axis":0},
        #{"opName": "log_softmax", "outName": "log_softmax/rank3_float64", "varShapes": [[1, 2, 3]], "varTypes": ["float64"], "varInit": ["uniform"], "axis":0},
        #{"opName": "log_softmax", "outName": "log_softmax/rank3_half", "varShapes": [[1, 2, 3]], "varTypes": ["half"], "varInit": ["uniform"], "axis":0},
        #{"opName": "log_softmax", "outName": "log_softmax/rank4_float32", "varShapes": [[1, 2, 3, 2]], "varTypes": ["float32"], "varInit": ["uniform"], "axis":0},
        #{"opName": "log_softmax", "outName": "log_softmax/rank4_float64", "varShapes": [[1, 2, 3, 2]], "varTypes": ["float64"], "varInit": ["uniform"], "axis":0},
        #{"opName": "log_softmax", "outName": "log_softmax/rank4_half", "varShapes": [[1, 2, 3, 2]], "varTypes": ["half"], "varInit": ["uniform"], "axis":0},
        #{"opName": "log_softmax", "outName": "log_softmax/emptyArrayTest/rank4_half", "varShapes": [[0, 0, 0,0]], "varTypes": ["half"], "varInit": ["uniform"], "axis":0},
        #{"opName": "log_softmax", "outName": "log_softmax/emptyArrayTest/rank4_float32", "varShapes": [[0, 0, 0, 0]], "varTypes": ["float32"], "varInit": ["uniform"], "axis":0},
        #{"opName": "log_softmax", "outName": "log_softmax/emptyArrayTest/rank4_float64", "varShapes": [[0, 0, 0, 0]], "varTypes": ["float64"], "varInit": ["uniform"], "axis":0},
        #{"opName": "log_softmax", "outName": "log_softmax/emptyArrayTest/partial_rank4_float64", "varShapes": [[0, 1, 0, 0]], "varTypes": ["float32"], "varInit": ["uniform"], "axis":0},
        # {"opName": "log_softmax", "outName": "log_softmax/partial_rank3_float64", "varShapes": [[1, 0, 0]], "varTypes": ["float64"], "varInit": ["uniform"],"axis":0},

        #{"opName": "non_max_suppression", "outName": "non_max_suppression/float32", "varShapes": [[3, 4],[3],[]], "varTypes": ["float32","float32","int32"], "varInit": ["uniform","uniform","zero"]},
        #{"opName": "non_max_suppression_v2", "outName": "non_max_suppression_v2/float32", "varShapes": [[3, 4], [3], []], "varTypes": ["float32","float32","int32"], "varInit": ["uniform","uniform","uniform_int10"]},
        #{"opName": "non_max_suppression", "outName": "non_max_suppression/float16", "varShapes": [[3, 4], [3], []],"varTypes": ["float16", "float16", "int32"], "varInit": ["uniform", "uniform", "uniform_int10"]},
        #{"opName": "non_max_suppression_v2", "outName": "non_max_suppression_v2/float16", "varShapes": [[3, 4], [3], []],"varTypes": ["float16", "float16", "int32"], "varInit": ["uniform", "uniform", "uniform_int10"]},
        #{"opName": "non_max_suppression", "outName": "non_max_suppression/float32_with_thresholds", "varShapes": [[3, 4], [3], []], "varTypes": ["float32", "float32", "int32"], "varInit": ["uniform", "uniform", "zero"], "iou_threshold": 0.0, "score_threshold": 0.0},
        #{"opName": "non_max_suppression", "outName": "non_max_suppression/emptyArrayTest/float32_with_thresholds", "varShapes": [[0, 4], [0], []], "varTypes": ["float32", "float32", "int32"], "varInit": ["uniform", "uniform", "zero"], "iou_threshold": 0.2, "score_threshold": 0.5},
        #{"opName": "non_max_suppression", "outName": "non_max_suppression/emptyArrayTest/float16_with_thresholds", "varShapes": [[0, 4], [0], []], "varTypes": ["float16", "float16", "int32"], "varInit": ["uniform", "uniform", "zero"], "iou_threshold": 0.5, "score_threshold": 0.5},
        #{"opName": "non_max_suppression", "outName": "non_max_suppression/emptyArrayTest/float32_with_thresholds",  "varShapes": [[0, 4], [0], []], "varTypes": ["float16", "float16", "int32"], "varInit": ["uniform", "uniform", "one"], "iou_threshold": 0.4, "score_threshold": 0.4},

        # {"opName": "betainc", "outName": "betainc/rank1_float32", "varShapes":[[4],[4],[4]], "varTypes":["float32","float32","float32"], "varInit":["uniform_0_1","uniform_0_1","uniform_0_1"]},
        # {"opName": "betainc", "outName": "betainc/rank2_float32", "varShapes": [[3,4],[3,4],[3,4]], "varTypes": ["float32","float32","float32"], "varInit": ["uniform_0_1","uniform_0_1","uniform_0_1"]},
        # {"opName": "betainc", "outName": "betainc/rank1_float64", "varShapes": [[4], [4], [4]], "varTypes": ["float64", "float64", "float64"], "varInit": ["uniform_0_1", "uniform_0_1", "uniform_0_1"]},
        # {"opName": "betainc", "outName": "betainc/rank2_float64", "varShapes": [[3, 4], [3, 4], [3, 4]], "varTypes": ["float64", "float64", "float64"], "varInit": ["uniform_0_1", "uniform_0_1", "uniform_0_1"]},
        # {"opName": "betainc", "outName": "betainc/emptyArrayTest/float64", "varShapes": [[0], [0], [0]],  "varTypes": ["float64", "float64", "float64"], "varInit": ["empty", "empty", "empty"]},
        # {"opName": "betainc", "outName": "betainc/emptyArrayTest/float32", "varShapes": [[0], [0], [0]],  "varTypes": ["float64", "float64", "float64"], "varInit": ["empty", "empty", "empty"]},

         #{"opName": "matrix_band_part", "outName": "matrix_band_part/float32", "varShapes": [[3,4]],  "varTypes": ["float32"], "varInit": ["uniform"],"num_lower":1, "num_upper":-1},
         #{"opName": "matrix_band_part", "outName": "matrix_band_part/float64", "varShapes": [[3, 4]],  "varTypes": ["float64"], "varInit": ["uniform"], "num_lower": 1, "num_upper": -1},
         #{"opName": "matrix_band_part", "outName": "matrix_band_part/float32_reverse_limits", "varShapes": [[3, 4]],   "varTypes": ["float32"], "varInit": ["uniform"], "num_lower": 0, "num_upper": 1},

         #{"opName": "multinomial", "outName": "multinomial/float32", "varShapes": [[3,3]],  "varTypes": ["float32"], "varInit": ["uniform"], "num_samples":3},
         #{"opName": "multinomial", "outName": "multinomial/float64", "varShapes": [[3, 4], [3, 4]], "varTypes": ["float64", "float64"], "varInit": ["uniform10", "uniform"]},
         #{"opName": "multinomial", "outName": "multinomial/emptyArrayTest/float32", "varShapes": [[0, 0], [3, 4]],  "varTypes": ["float32", "float32"], "varInit": ["empty", "uniform"]},

         #{"opName": "polygamma", "outName": "polygamma/float32", "varShapes": [[3,4],[3,4]],  "varTypes": ["float32","float32"], "varInit": ["uniform10","uniform"]},
         #{"opName": "polygamma", "outName": "polygamma/float64", "varShapes": [[3, 4], [3, 4]], "varTypes": ["float64", "float64"], "varInit": ["uniform10", "uniform"]},
         #{"opName": "polygamma", "outName": "polygamma/emptyArrayTest/float32", "varShapes": [[0, 0], [3, 4]],  "varTypes": ["float32", "float32"], "varInit": ["empty", "uniform"]},

         #{"opName": "lgamma", "outName": "lgamma/float32", "varShapes": [[3,4]],  "varTypes": ["float32"], "varInit": ["uniform"]},
         #{"opName": "lgamma", "outName": "lgamma/float64", "varShapes": [[3, 4]], "varTypes": ["float64"], "varInit": ["uniform"]},
         #{"opName": "lgamma", "outName": "lgamma/emptyArrayTest/float32", "varShapes": [[0, 0]],  "varTypes": ["float32"], "varInit": ["empty"]},

        #{"opName": "igamma", "outName": "igamma/float32", "varShapes": [[3,4],[3,4]],  "varTypes": ["float32","float32"], "varInit": ["uniform10","uniform"]},
        #{"opName": "igamma", "outName": "igamma/float64", "varShapes": [[3, 4], [3, 4]], "varTypes": ["float64", "float64"], "varInit": ["uniform10", "uniform"]},
        #{"opName": "igamma", "outName": "igamma/emptyArrayTest/float32", "varShapes": [[0, 0], [0, 0]],  "varTypes": ["float32", "float32"], "varInit": ["empty", "uniform"]},

        #{"opName": "igammac", "outName": "igammac/float32", "varShapes": [[3, 4], [3, 4]], "varTypes": ["float32", "float32"], "varInit": ["uniform10", "uniform"]},
        #{"opName": "igammac", "outName": "igammac/float64", "varShapes": [[3, 4], [3, 4]], "varTypes": ["float64", "float64"], "varInit": ["uniform10", "uniform"]},
        #{"opName": "igammac", "outName": "igammac/emptyArrayTest/float32", "varShapes": [[0, 0], [0, 0]], "varTypes": ["float32", "float32"], "varInit": ["empty", "uniform"]},

        #{"opName": "digamma", "outName": "digamma/float32", "varShapes": [[3, 4]], "varTypes": ["float32"], "varInit": ["uniform10"]},
        #{"opName": "digamma", "outName": "digamma/float64", "varShapes": [[3, 4]], "varTypes": ["float64"], "varInit": ["uniform10"]},
        #{"opName": "digamma", "outName": "digamma/emptyArrayTest/float32", "varShapes": [[0, 0]], "varTypes": ["float32"], "varInit": ["empty"]},

        #{"opName": "lgamma", "outName": "lgamma/float32", "varShapes": [[3, 4]], "varTypes": ["float32"], "varInit": ["uniform"]},
        #{"opName": "lgamma", "outName": "lgamma/float64", "varShapes": [[3, 4]], "varTypes": ["float64"], "varInit": ["uniform"]},
        #{"opName": "lgamma", "outName": "lgamma/emptyArrayTest/float32", "varShapes": [[0, 0]], "varTypes": ["float32"], "varInit": ["empty"]},

         #{"opName": "random_crop", "outName": "random_crop/rank3_float32", "varShapes": [[8, 8, 3], [3]],   "varTypes": ["float32", "int32"], "varInit": ["stdnormal", "uniform_int2"]},
         #{"opName": "random_crop", "outName": "random_crop/rank3_float64", "varShapes": [[8, 8, 3], [3]],  "varTypes": ["float64", "int32"], "varInit": ["stdnormal", "uniform_int2"]},
         #{"opName": "random_crop", "outName": "random_crop/rank3_float64", "varShapes": [[8, 8, 3], [3]],  "varTypes": ["float64", "int32"], "varInit": ["stdnormal", "uniform_int2"]},

         #{"opName": "roll", "outName": "roll/rank1_float32", "varShapes": [[4]],"varTypes": ["float32"], "varInit": ["uniform"], "shift":2, "axis":0},
         #{"opName": "roll", "outName": "roll/rank1_float64", "varShapes": [[4]], "varTypes": ["float64"], "varInit": ["uniform"], "shift": 2, "axis": 0},
         #{"opName": "roll", "outName": "roll/rank2_float32", "varShapes": [[2,4]], "varTypes": ["float32"],  "varInit": ["uniform"], "shift": 2, "axis": 0},
         #{"opName": "roll", "outName": "roll/rank2_float32_zeroshift", "varShapes": [[2, 4]], "varTypes": ["float32"],  "varInit": ["uniform"], "shift": 0, "axis": 1},
         #{"opName": "roll", "outName": "roll/rank2_float32_axis", "varShapes": [[4,5]],"varTypes": ["float32"], "varInit": ["uniform"], "shift":2, "axis":1},
         #{"opName": "roll", "outName": "roll/rank2_float64_axis", "varShapes": [[4,5]], "varTypes": ["float64"], "varInit": ["uniform"], "shift": 2, "axis": 1},
         #{"opName": "roll", "outName": "roll/rank3_float32_axis", "varShapes": [[4, 5, 3]], "varTypes": ["float32"], "varInit": ["uniform"], "shift": 2, "axis": 2},
         #{"opName": "roll", "outName": "roll/rank3_float64_axis", "varShapes": [[4, 5, 4]], "varTypes": ["float64"],  "varInit": ["uniform"], "shift": 2, "axis": 1},
         #{"opName": "roll", "outName": "roll/rank3_int32_axis", "varShapes": [[4, 5, 4]], "varTypes": ["int32"],  "varInit": ["uniform_int10"], "shift": 2, "axis": 1},
         #{"opName": "roll", "outName": "roll/rank4_float32_axis", "varShapes": [[4, 5, 3, 4]], "varTypes": ["float32"],  "varInit": ["uniform"], "shift": 2, "axis": 3},
         #{"opName": "roll", "outName": "roll/rank4_float32_axis", "varShapes": [[4, 5, 3, 4]], "varTypes": ["half"],  "varInit": ["uniform"], "shift": 2, "axis": 2},
         #{"opName": "roll", "outName": "roll/int32_long_axis", "varShapes": [[4, 5, 3, 4]], "varTypes": ["int64"], "varInit": ["uniform_int10"], "shift": 2, "axis": 3},

         #{"opName": "roll", "outName": "roll_test/rank1_float32", "varShapes": [[1024]],"varTypes": ["float32"], "varInit": ["uniform"], "shift":2, "axis":0},
         #{"opName": "roll", "outName": "roll_test/rank2_float32", "varShapes": [[102,204]], "varTypes": ["float32","float32"], "varInit": ["uniform","uniform"], "shift": 0, "axis": 1},
         #{"opName": "roll", "outName": "roll_test/rank3_float32", "varShapes": [[102, 204, 102]],  "varTypes": ["float32", "float32", "float32"], "varInit": ["uniform", "uniform","uniform"], "shift": 0, "axis": 1},
         #{"opName": "roll", "outName": "roll_test/rank4_float32", "varShapes": [[20, 20, 30, 20]],  "varTypes": ["float32", "float32", "float32","float32"], "varInit": ["uniform", "uniform", "uniform","uniform"],"shift": 0, "axis": 1},

        #{"opName": "toggle_bits", "outName": "toggle_bits/rank1_int8", "varShapes": [[4]], "varTypes": ["int8"], "varInit": ["uniform_int10"]},
        #{"opName": "toggle_bits", "outName": "toggle_bits/rank1_int16", "varShapes": [[4]], "varTypes": ["int16"], "varInit": ["uniform_int10"]},
        #{"opName": "toggle_bits", "outName": "toggle_bits/rank1_int32", "varShapes": [[4]],"varTypes": ["int32"], "varInit": ["uniform_int10"]},
        #{"opName": "toggle_bits", "outName": "toggle_bits/rank2_int8", "varShapes": [[4,2]], "varTypes": ["int8"], "varInit": ["uniform_int10"]},
        #{"opName": "toggle_bits", "outName": "toggle_bits/rank2_int16", "varShapes": [[4,2]], "varTypes": ["int16"],"varInit": ["uniform_int10"]},
        #{"opName": "toggle_bits", "outName": "toggle_bits/rank2_int32", "varShapes": [[4,2]], "varTypes": ["int32"], "varInit": ["uniform_int10"]},
        #{"opName": "toggle_bits", "outName": "toggle_bits/rank3_int8", "varShapes": [[4, 2, 3]], "varTypes": ["int8"], "varInit": ["uniform_int10"]},
        #{"opName": "toggle_bits", "outName": "toggle_bits/rank3_int16", "varShapes": [[4, 2, 3]], "varTypes": ["int16"], "varInit": ["uniform_int10"]},
        #{"opName": "toggle_bits", "outName": "toggle_bits/rank2_int32", "varShapes": [[4, 2, 3]], "varTypes": ["int32"], "varInit": ["uniform_int10"]},

        # {"opName": "right_shift", "outName": "right_shift/rank1_int32", "varShapes": [[4],[4]],"varTypes": ["int32","int32"], "varInit": ["uniform_int10","uniform_int10"]},
        # {"opName": "right_shift", "outName": "right_shift/rank2_int32", "varShapes": [[4,2],[4,2]], "varTypes": ["int32","int32"], "varInit": ["uniform_int10","uniform_int10"]},
        # {"opName": "right_shift", "outName": "right_shift/rank2_int32", "varShapes": [[4, 2, 3],[4,2,3]], "varTypes": ["int32","int32"], "varInit": ["uniform_int10","uniform_int10"]},
        #{"opName": "right_shift", "outName": "right_shift/rank3_int8", "varShapes": [[4, 2, 3], [4, 2, 3]], "varTypes": ["int32", "int32"], "varInit": ["uniform_int10", "uniform_int10"]},
        #{"opName": "right_shift", "outName": "right_shift/rank4_int16", "varShapes": [[4, 1, 1, 2], [4, 1, 1, 2]], "varTypes": ["int32", "int32"], "varInit": ["uniform_int10", "uniform_int10"]},

        #{"opName": "left_shift", "outName": "left_shift/rank1_int32", "varShapes": [[4], [4]], "varTypes": ["int32", "int32"], "varInit": ["uniform_int10","uniform_int10"]},
        #{"opName": "left_shift", "outName": "left_shift/rank2_int32", "varShapes": [[4, 2], [4, 2]],  "varTypes": ["int32", "int32"], "varInit": ["uniform_int10","uniform_int10"]},
        #{"opName": "left_shift", "outName": "left_shift/rank2_int32", "varShapes": [[4, 2, 3], [4, 2, 3]], "varTypes": ["int32", "int32"], "varInit": ["uniform_int10","uniform_int10"]},
        #{"opName": "left_shift", "outName": "left_shift/rank3_int8", "varShapes": [[4, 2, 3], [4, 2, 3]], "varTypes": ["int32", "int32"], "varInit": ["uniform_int10", "uniform_int10"]},
        #{"opName": "left_shift", "outName": "left_shift/rank4_int16", "varShapes": [[4, 1, 1, 2], [4, 1, 1, 2]], "varTypes": ["int32", "int32"], "varInit": ["uniform_int10", "uniform_int10"]},

        #{"opName": "random_gamma", "outName": "random_gamma/rank1_float32", "varShapes": [[4], [4]], "varTypes": ["int32", "float32"], "varInit": ["uniform_int10","uniform"], "seeds":1},
        #{"opName": "random_poisson", "outName": "random_poisson/rank1_float32", "varShapes": [[4], [4]], "varTypes": ["int32", "float32"], "varInit": ["uniform_int10", "uniform"]},
        #{"opName": "random_shuffle", "outName": "random_shuffle/rank1_float32", "varShapes": [[4]], "varTypes": ["float32"], "varInit": ["uniform"]},

        #{"opName": "fused_batch_norm", "outName": "fused_batch_norm/float16_nhcw",  "varShapes": [[1, 2, 3, 4], [4], [4]], "varTypes": ["float16", "float32", "float32"],  "varInit": ["uniform", "uniform_0_1", "uniform_0_1"], "epsilon": 0.5, "data_format": "NHWC"},
        #{"opName": "fused_batch_norm", "outName": "fused_batch_norm/float32_nhwc_restr", "varShapes": [[1, 2, 3, 4], [4], [4]], "varTypes": ["float32", "float32", "float32"], "varInit": ["uniform_0_1", "uniform_0_1", "uniform_0_1"], "epsilon": 0.5 , "data_format": "NHWC"},
        # The CPU implementation of FusedBatchNorm only supports NHWC tensor format for now.
        #{"opName": "fused_batch_norm", "outName": "fused_batch_norm/float32_nchw", "varShapes": [[1, 2, 3, 4], [2], [2]], "varTypes": ["float32", "float32", "float32"], "varInit": ["uniform", "uniform", "uniform"], "epsilon": 0.5, "data_format": "NCHW"},
        #{"opName": "fused_batch_norm", "outName": "fused_batch_norm/float32_nhwc", "varShapes": [[1, 2, 3, 4], [4], [4]], "varTypes": ["float32", "float32", "float32"],  "varInit": ["uniform", "uniform", "uniform"], "epsilon": 0.0, "data_format": "NHWC"},

         # hsv_to_rgb - no registered kernels for half, float16 types.
         # {"opName": "hsv_to_rgb", "outName": "hsv_to_rgb/float16", "varShapes": [[1, 2, 3]], "varTypes": ["half"], "varInit": ["uniform"]},
         # {"opName": "hsv_to_rgb", "outName": "hsv_to_rgb/float16", "varShapes": [[1, 2, 3]], "varTypes": ["float16"], "varInit": ["uniform"]},

         #{"opName": "hsv_to_rgb", "outName": "hsv_to_rgb/float32_1", "varShapes": [[1, 1, 3]], "varTypes": ["float32"],  "varInit": ["uniform"]},
         #{"opName": "hsv_to_rgb", "outName": "hsv_to_rgb/float32", "varShapes": [[1, 2, 3]], "varTypes": ["float32"],  "varInit": ["uniform"]},
         #{"opName": "hsv_to_rgb", "outName": "hsv_to_rgb/float64", "varShapes": [[2, 4, 3]], "varTypes": ["float64"],  "varInit": ["uniform"]},
         #{"opName": "hsv_to_rgb", "outName": "hsv_to_rgb/emptyArrayTest/float64", "varShapes": [[0, 4, 3]], "varTypes": ["float64"],   "varInit": ["empty"]},
         #{"opName": "hsv_to_rgb", "outName": "hsv_to_rgb/float64_from0_to1", "varShapes": [[5, 4, 3]], "varTypes": ["float64"], "varInit": ["uniform_0_1"]},

         #{"opName": "rgb_to_hsv", "outName": "rgb_to_hsv/float32", "varShapes": [[1, 2, 3]], "varTypes": ["float32"], "varInit": ["uniform"]},
         #{"opName": "rgb_to_hsv", "outName": "rgb_to_hsv/float64", "varShapes": [[2, 4, 3]], "varTypes": ["float64"],    "varInit": ["uniform"]},
         #{"opName": "rgb_to_hsv", "outName": "rgb_to_hsv/emptyArrayTest/float64", "varShapes": [[0, 4, 3]], "varTypes": ["float64"],   "varInit": ["empty"]},
         #{"opName": "rgb_to_hsv", "outName": "rgb_to_hsv/float64_from0_to1", "varShapes": [[5, 4, 3]], "varTypes": ["float64"], "varInit": ["uniform_0_1"]},

         #{"opName": "yiq_to_rgb", "outName": "yiq_to_rgb/float32_1", "varShapes": [[1, 1, 3]], "varTypes": ["float32"],  "varInit": ["uniform"]},
         #{"opName": "yiq_to_rgb", "outName": "yiq_to_rgb/float32", "varShapes": [[1, 2, 3]], "varTypes": ["float32"],  "varInit": ["uniform"]},
         #{"opName": "yiq_to_rgb", "outName": "yiq_to_rgb/float64", "varShapes": [[2, 4, 3]], "varTypes": ["float64"],  "varInit": ["uniform"]},
         #{"opName": "yiq_to_rgb", "outName": "yiq_to_rgb/emptyArrayTest/float64", "varShapes": [[0, 4, 3]], "varTypes": ["float64"],   "varInit": ["empty"]},
         #{"opName": "yiq_to_rgb", "outName": "yiq_to_rgb/float64_from0_to1", "varShapes": [[5, 4, 3]], "varTypes": ["float64"], "varInit": ["uniform_0_1"]},

         #{"opName": "rgb_to_yiq", "outName": "rgb_to_yiq/float32", "varShapes": [[1, 2, 3]], "varTypes": ["float32"], "varInit": ["uniform"]},
         #{"opName": "rgb_to_yiq", "outName": "rgb_to_yiq/float64", "varShapes": [[2, 4, 3]], "varTypes": ["float64"],    "varInit": ["uniform"]},
         #{"opName": "rgb_to_yiq", "outName": "rgb_to_yiq/emptyArrayTest/float64", "varShapes": [[0, 4, 3]], "varTypes": ["float64"],   "varInit": ["empty"]},
         #{"opName": "rgb_to_yiq", "outName": "rgb_to_yiq/float64_from0_to1", "varShapes": [[5, 4, 3]], "varTypes": ["float64"], "varInit": ["uniform_0_1"]},

        #{"opName": "rgb_to_grayscale", "outName": "rgb_to_grayscale/float32", "varShapes": [[1, 2, 3]], "varTypes": ["float32"],  "varInit": ["uniform"]},
        #{"opName": "rgb_to_grayscale", "outName": "rgb_to_grayscale/float64", "varShapes": [[2, 4, 3]], "varTypes": ["float64"],  "varInit": ["uniform"]},
        #{"opName": "rgb_to_grayscale", "outName": "rgb_to_grayscale/emptyArrayTest/float64", "varShapes": [[0, 4, 3]], "varTypes": ["float64"], "varInit": ["empty"]},
        #{"opName": "rgb_to_grayscale", "outName": "rgb_to_grayscale/float64_from0_to1", "varShapes": [[5, 4, 3]], "varTypes": ["float64"], "varInit": ["uniform_0_1"]},

        # {"opName": "yuv_to_rgb", "outName": "yuv_to_rgb/float32_1", "varShapes": [[1, 1, 3]], "varTypes": ["float32"], "varInit": ["uniform"]},
        # {"opName": "yuv_to_rgb", "outName": "yuv_to_rgb/float32", "varShapes": [[1, 2, 3]], "varTypes": ["float32"],  "varInit": ["uniform"]},
        # {"opName": "yuv_to_rgb", "outName": "yuv_to_rgb/float64", "varShapes": [[2, 4, 3]], "varTypes": ["float64"],  "varInit": ["uniform"]},
        # {"opName": "yuv_to_rgb", "outName": "yuv_to_rgb/emptyArrayTest/float64", "varShapes": [[0, 4, 3]], "varTypes": ["float64"],   "varInit": ["empty"]},
        # {"opName": "yuv_to_rgb", "outName": "yuv_to_rgb/float64_from0_to1", "varShapes": [[5, 4, 3]], "varTypes": ["float64"], "varInit": ["uniform_0_1"]},

         #{"opName": "rgb_to_yuv", "outName": "rgb_to_yuv/float32", "varShapes": [[1, 2, 3]], "varTypes": ["float32"], "varInit": ["uniform"]},
         #{"opName": "rgb_to_yuv", "outName": "rgb_to_yuv/float64", "varShapes": [[2, 4, 3]], "varTypes": ["float64"],    "varInit": ["uniform"]},
         #{"opName": "rgb_to_yuv", "outName": "rgb_to_yuv/emptyArrayTest/float64", "varShapes": [[0, 4, 3]], "varTypes": ["float64"],   "varInit": ["empty"]},
         #{"opName": "rgb_to_yuv", "outName": "rgb_to_yuv/float64_from0_to1", "varShapes": [[5, 4, 3]], "varTypes": ["float64"], "varInit": ["uniform_0_1"]},

        #{"opName": "lu", "outName": "lu/float32_rank2", "varShapes": [[3,3]], "varTypes": ["float32"], "varInit": ["uniform"]},
        #{"opName": "lu", "outName": "lu/float32_rank3", "varShapes": [[2,2,2]], "varTypes": ["float32"], "varInit": ["uniform"]},
        #{"opName": "lu", "outName": "lu/emptyArrayTest/float32", "varShapes": [[0, 2, 2]], "varTypes": ["float32"],"varInit": ["empty"]},
        #{"opName": "lu", "outName": "lu/float32_rank3_returns_int64", "varShapes": [[2,2,2]], "varTypes": ["float32"],  "varInit": ["uniform"], "output_idx_type": "int64"}

        # {"opName": "triangular_solve", "outName": "triangular_solve/float32_rank2", "varShapes": [[3,3],[3,3]], "varTypes": ["float32","float32"], "varInit": ["uniform","uniform"], "lower":False,"adjoint":True},
        # {"opName": "triangular_solve", "outName": "triangular_solve/float32_rank3", "varShapes": [[2,2,2],[2,2,2]], "varTypes": ["float32","float32"], "varInit": ["uniform","uniform"], "lower":False,"adjoint":False},
        #{"opName": "triangular_solve", "outName": "triangular_solve/float64_rank2", "varShapes": [[3, 3], [3, 3]], "varTypes": ["float64", "float64"], "varInit": ["uniform", "uniform"], "lower": False, "adjoint": True},
        #{"opName": "triangular_solve", "outName": "triangular_solve/float64_rank3", "varShapes": [[2, 2, 2], [2, 2, 2]], "varTypes": ["float64", "float64"], "varInit": ["uniform", "uniform"], "lower": False, "adjoint": False},
        # {"opName": "triangular_solve", "outName": "triangular_solve/emptyArrayTest/float32", "varShapes": [[0, 2, 2],[0,2,2]], "varTypes": ["float32","float32"],"varInit": ["empty","empty"], "lower":True,"adjoint":True},
        # {"opName": "triangular_solve", "outName": "triangular_solve/float32_rank3_returns_int64", "varShapes": [[2,2,2],[2,2,2]], "varTypes": ["float32","float32"],  "varInit": ["uniform","uniform"], "lower":True,"adjoint":True}

        # {"opName": "lstsq", "outName": "lstsq/float32_rank2", "varShapes": [[3,3],[3,3]], "varTypes": ["float32","float32"], "varInit": ["uniform","uniform"], "l2_regularizer":0.1,"fast":True},

        #{"opName": "linear_solve", "outName": "linear_solve/float32_rank2", "varShapes": [[3, 3], [3, 3]], "varTypes": ["float32", "float32"], "varInit": ["uniform", "uniform"], "adjoint": True},
        #{"opName": "linear_solve", "outName": "linear_solve/float32_rank3", "varShapes": [[2, 2, 2], [2, 2, 2]], "varTypes": ["float32", "float32"], "varInit": ["uniform", "uniform"], "adjoint": False},
        #{"opName": "linear_solve", "outName": "linear_solve/float64_rank2", "varShapes": [[3, 3], [3, 3]],  "varTypes": ["float64", "float64"], "varInit": ["uniform", "uniform"], "adjoint": True},
        #{"opName": "linear_solve", "outName": "linear_solve/float64_rank3", "varShapes": [[2, 2, 2], [2, 2, 2]],  "varTypes": ["float64", "float64"], "varInit": ["uniform", "uniform"],  "adjoint": False},
        #{"opName": "linear_solve", "outName": "linear_solve/emptyArrayTest/float32",  "varShapes": [[0, 2, 2], [0, 2, 2]], "varTypes": ["float32", "float32"], "varInit": ["empty", "empty"], "adjoint": True},
        #{"opName": "linear_solve", "outName": "linear_solve/float32_rank3_returns_int64", "varShapes": [[2, 2, 2], [2, 2, 2]], "varTypes": ["float32", "float32"], "varInit": ["uniform", "uniform"], "adjoint": True}

        # NEWLY ADDED OPS

        # OPS works and testcase generated to dl4j-resources

        # {"opName": "adjust_contrast", "outName": "adjust_contrast/float32_rank3", "varShapes": [[16,16,3]], "varTypes": ["float32"], "varInit": ["uniform"], "contrast_factor":2},
        # {"opName":"dropout", "outName":"dropout/someoutput", "varShapes":[[1, 3, 3, 3, 2]], "varTypes":["float32"], "varInit": ["uniform"], "rate":0.2},
        # {"opName":"div", "outName":"div/someoutput", "varShapes":[[1, 2, 3],[1, 2, 3]], "varTypes":["float32","float32"], "varInit": ["uniform","uniform"]},
        # {"opName":"is_non_decreasing", "outName":"is_non_decreasing/someoutput", "varShapes":[[1, 2, 3]], "varTypes":["float32"], "varInit": ["uniform"]},
        # {"opName":"leaky_relu", "outName":"leaky_relu/someoutput", "varShapes":[[1, 2, 3]], "varTypes":["float32"], "varInit": ["uniform"], "alpha":0.2},
        # {"opName":"lgamma", "outName":"lgamma/someoutput", "varShapes":[[1, 2, 3]], "varTypes":["float32"], "varInit": ["uniform"]},
        # {"opName":"mod", "outName":"mod/someoutput", "varShapes":[[1, 2, 3],[1, 2, 3]], "varTypes":["float32", "float32"], "varInit": ["uniform", "uniform"]},
        # {"opName": "compare_and_bitpack", "outName": "compare_and_bitpack/float32", "varShapes": [[8]], "varTypes": ["float32"], "varInit": ["uniform"], "threshold":0.,},
        # {"opName":"empty", "outName":"empty/someoutput", "varShapes":[[5]], "varTypes":["int32"], "varInit": ["uniform_int10"], "dtype":tf.float32},
        # {"opName":"deep_copy", "outName":"DeepCopy/someoutput", "varShapes":[[1, 3, 3, 3, 2]], "varTypes":["float32"], "varInit": ["uniform"] },
        # {"opName":"ones_like", "outName":"ones_like/someoutput", "varShapes":[[1, 3, 3, 3, 2]], "varTypes":["float32"], "varInit": ["uniform"] },
       # {"opName": "random_crop", "outName": "random_crop/rank3_float32", "varShapes": [[8, 8, 3], [3]],   "varTypes": ["float32", "int32"], "varInit": ["stdnormal", "uniform_int2"]},
       # {"opName": "random_gamma", "outName": "random_gamma/rank1_float32", "varShapes": [[4], [4]], "varTypes": ["int32", "float32"], "varInit": ["uniform_int10","uniform"], "seed":1, "alpha":[0.5, 1.5], "dtype":tf.float32},
        # {"opName": "random_poisson", "outName": "random_poisson/rank1_float32", "varShapes": [[4], [4]], "varTypes": ["int32", "float32"], "varInit": ["uniform_int10", "uniform"], "lam":[0.5, 1.5],"dtype":tf.float32},
       # {"opName": "random_poisson_v2", "outName": "random_poisson/rank1_float32", "varShapes": [[4], [4]], "varTypes": ["int32", "float32"], "varInit": ["uniform_int10", "uniform"], "rate":[0.5, 1.5],"dtype":tf.float32},
        #  {"opName": "random_shuffle", "outName": "random_shuffle/rank1_float32", "varShapes": [[4]], "varTypes": ["float32"], "varInit": ["uniform"], "seed":12345},
#           {"opName": "random_normal", "outName": "random_shuffle/rank1_float32", "varShapes": [[4]], "varTypes": ["int32"], "varInit": ["uniform_int10"], "mean": 0., "stddev":1.0, "seed":12345, "dtype":tf.float32},
           {"opName": "random_uniform", "outName": "random_shuffle/rank1_float32", "varShapes": [[4]], "varTypes": ["int32"], "varInit": ["uniform_int10"], "maxval": 10, "minval":1, "seed":12345, "dtype":tf.float32},



        # OPS arent work for some reason
        # https://gist.github.com/atuzhykov/b9ba46de91c54eda65c24546db11b9d3 looks like problem with CPU compatible
        #  {"opName":"max_pool_with_argmax", "outName":"max_pool_with_argmax/someoutput", "varShapes":[[1, 16, 16 ,3]], "varTypes":["float32"], "varInit": ["uniform"], "ksizes": [1,1,1,1],
        #             "strides": [1,1,1,1],   "padding":"SAME", "data_format":'NHWC' ,"include_batch_in_index": True, "output_dtype": tf.int32},
        # {"opName":"Conv3DBackpropInputV2", "outName":"Conv3DBackpropInput/someoutput", "varShapes":[[5], [1, 1, 1, 3, 1], [1, 3, 3, 3, 1]], "varTypes":["int32","float32", "float32"], "varInit": ["uniform_int10","uniform","uniform"], "strides":[1,1,1,1,1], "padding":"SAME", "dilations":[1, 1, 1, 1, 1]},



     ]

    '''
    Ops requiring tests: (note that some of these might not support 0 shapes)
    *assign
    assign_add, assign_sub
    batch_to_space
    boolean_mask
    broadcast_dynamic_shape
    broadcast_static_shape
    broadcast_to
    clip_by_average_norm
    clip_by_norm
    clip_by_value
    dynamic_partition
    dynamic_stitch
    edit_distance
    eye
    floormod
    gather_nd
    meshgrid
    norm
    no_op
    one_hot
    pad
    parallel_stack
    reverse_sequence
    roll
    *scatter_* (nd, nd_add, nd_sub, nd_update)
    sequence_mask
    *split (doesn't support num_or_size_splits=0?)
    tile
    unique
    
    tf.linalg:
    det
    diag
    diag_part
    inv
    logdet
    tensor_diag
    tensor_diag_part
    trace
    
    tf.math:
    angle
    atan2
    bincount
    erf
    *invert_permutation
    in_top_k
    *scalar_mul
    *segment_* (max, mean, min, prod, sum)
    *unsorted_segment_* (max, mean, min, prod, sqrt_n, sum)
    xdivy
    xlogy
    zeta
    '''


    for op in ops:
        tf.compat.v1.reset_default_graph()
        print("Running " + str(op))
        test = OpTest(seed=19, op=op)

        opName = op["opName"]
        varShapes = op.get("varShapes")
        varTypes = op.get("varTypes")
        varInit = op.get("varInit")
        phShapes = op.get("phShapes")
        phTypes = op.get("phTypes")
        phInit = op.get("phInit")

        opCreator = OpCreator(op)

        vars = test.createVars(varShapes, varTypes, varInit)
        #ph = test.createPlaceholders(phShapes, phTypes, phInit)
        opCreator.setVars(vars)

        out = opCreator.execute(opName)

        print(out)

        # Run and persist
        testName = op["outName"]
        tfp = TensorFlowPersistor(save_dir=testName)
        tfp.set_placeholders([]) \
            .set_output_tensors(out) \
            .set_test_data(test.get_test_data()) \
            .set_verbose(True) \
            .build_save_frozen_graph()

print("TF version: " + tf.version.VERSION)
tf.compat.v1.disable_eager_execution()

if __name__ == '__main__':
    test_mathtransform()
