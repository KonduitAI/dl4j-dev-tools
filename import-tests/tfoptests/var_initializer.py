import tensorflow as tf
import numpy as np
import random


class VarInitializer:
    #def __init__(self):
    #    self.node_num = 0

    def prod(self, shape):
        prod = 1
        for v in shape:
            prod = prod * v
        return prod

    def newVar(self, initType, shape, dtype, name):
        method_name = "var_" + initType
        try:
            method = getattr(self, method_name)
        except AttributeError:
            print(method_name, "not found")
        else:
            return method(shape, dtype, name)

    def var_zeros(self, shape, dtype, n):
        return self.var_zero(shape, dtype, n)

    def var_zero(self, shape, dtype, n):
        return tf.Variable(tf.zeros(shape=shape, dtype=dtype), name=n)

    def var_one(self, shape, dtype, n):
        return tf.Variable(tf.ones(shape=shape, dtype=dtype), name=n)

    def var_two(self, shape, dtype, n):
        return tf.Variable(tf.cast(tf.fill(dims=shape, value=2), dtype=dtype), name=n)

    def var_three(self, shape, dtype, n):
        return tf.Variable(tf.cast(tf.fill(dims=shape, value=3), dtype=dtype), name=n)

    def var_four(self, shape, dtype, n):
        return tf.Variable(tf.cast(tf.fill(dims=shape, value=4), dtype=dtype), name=n)

    def var_five(self, shape, dtype, n):
        return tf.Variable(tf.cast(tf.fill(dims=shape, value=5), dtype=dtype), name=n)

    def var_ten(self, shape, dtype, n):
        return tf.Variable(tf.ones(shape=shape, dtype=dtype) * 10, name=n)

    def var_minus_one(self, shape, dtype, n):
        return tf.Variable(tf.cast(tf.fill(dims=shape, value=-1), dtype=dtype), name=n)

    def var_minus_two(self, shape, dtype, n):
        return tf.Variable(tf.cast(tf.fill(dims=shape, value=-2), dtype=dtype), name=n)

    def var_range(self, shape, dtype, n):
        return tf.Variable(tf.reshape(tf.range(start=0, limit=np.prod(shape), delta=1, dtype=dtype), shape), name=n)

    def var_stdnormal(self, shape, dtype, n):
        return tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0, dtype=dtype), dtype=dtype, name=n)

    def var_uniform(self, shape, dtype, n):
        return tf.Variable(tf.random_uniform(shape,dtype=dtype), dtype, name=n)

    def var_uniform_m1_1(self, shape, dtype, n):
        return tf.Variable(tf.random_uniform(shape, minval=-1, maxval=1), dtype, name=n)

    def var_uniform_m1_0(self, shape, dtype, n):
        return tf.Variable(tf.random_uniform(shape, minval=-1, maxval=0), dtype, name=n)

    def var_uniform10(self, shape, dtype, n):
        return tf.Variable(tf.random_uniform(shape, minval=0, maxval=10, dtype=dtype), dtype, name=n)

    def var_uniform_int2(self, shape, dtype, n):
        if(dtype == tf.int32 or dtype == tf.int64):
            return tf.Variable(tf.random_uniform(shape, minval=0, maxval=2, dtype=dtype), dtype, name=n)
        else:
            return tf.Variable(tf.floor(tf.random_uniform(shape, minval=0, maxval=2, dtype=dtype)), dtype, name=n)

    def var_uniform_int3(self, shape, dtype, n):
        if(dtype == tf.int32 or dtype == tf.int64):
            return tf.Variable(tf.random_uniform(shape, minval=0, maxval=3, dtype=dtype), dtype, name=n)
        else:
            return tf.Variable(tf.floor(tf.random_uniform(shape, minval=0, maxval=3, dtype=dtype)), dtype, name=n)

    def var_uniform_int5(self, shape, dtype, n):
        if(dtype == tf.int32 or dtype == tf.int64):
            return tf.Variable(tf.random_uniform(shape, minval=0, maxval=5, dtype=dtype), dtype, name=n)
        else:
            return tf.Variable(tf.floor(tf.random_uniform(shape, minval=0, maxval=5, dtype=dtype)), dtype, name=n)

    def var_uniform_int10(self, shape, dtype, n):
        if(dtype == tf.int32 or dtype == tf.int64):
            return tf.Variable(tf.random_uniform(shape, minval=0, maxval=10, dtype=dtype), dtype, name=n)
        else:
            return tf.Variable(tf.floor(tf.random_uniform(shape, minval=0, maxval=10, dtype=dtype)), dtype, name=n)

    def var_uniform_sparse(self, shape, dtype, n):
        values = tf.random_uniform(shape) * tf.cast((tf.random_uniform(shape) < 0.5), dtype=tf.float32)
        return tf.Variable(values, dtype, name=n)

    def var_fixed_m1_1(self, shape, dtype, n):
        if(len(shape) is not 1 or shape[0] is not 2):
            raise ValueError("Shape must be exactly [2]")
        return tf.Variable([-1, 1], dtype=dtype, name=n)

    def var_fixed_2_0(self, shape, dtype, n):
        if(len(shape) is not 1 or shape[0] is not 2):
            raise ValueError("Shape must be exactly [2]")
        return tf.Variable([2,0], dtype=dtype, name=n)

    def var_fixed_2_1(self, shape, dtype, n):
        if(len(shape) is not 1 or shape[0] is not 2):
            raise ValueError("Shape must be exactly [2]")
        return tf.Variable([2,1], dtype=dtype, name=n)

    def var_fixed_3_1(self, shape, dtype, n):
        if(len(shape) is not 1 or shape[0] is not 2):
            raise ValueError("Shape must be exactly [2]")
        return tf.Variable([3,1], dtype=dtype, name=n)

    def var_fixed_5_3(self, shape, dtype, n):
        if(len(shape) is not 1 or shape[0] is not 2):
            raise ValueError("Shape must be exactly [2]")
        return tf.Variable([5, 3], dtype=dtype, name=n)

    def var_fixed_2_2_4(self, shape, dtype, n):
        if(len(shape) is not 1 or shape[0] is not 3):
            raise ValueError("Shape must be exactly [3]")
        return tf.Variable([2,2,4], dtype=dtype, name=n)

    def var_fixed_0_0_3(self, shape, dtype, n):
        if(len(shape) is not 1 or shape[0] is not 3):
            raise ValueError("Shape must be exactly [3]")
        return tf.Variable([0,0,3], dtype=dtype, name=n)

    def var_fixed_2_1_4(self, shape, dtype, n):
        if(len(shape) is not 1 or shape[0] is not 3):
            raise ValueError("Shape must be exactly [3]")
        return tf.Variable([2,1,4], dtype=dtype, name=n)

    def var_fixed_3_1_4_2(self, shape, dtype, n):
        if(self.prod(shape) is not 4):
            raise ValueError("Shape must be exactly [4]")
        return tf.reshape(tf.Variable([3,1,2,4], dtype=dtype, name=n), shape)

    def var_unique_rand_5(self, shape, dtype, n):
        if(self.prod(shape) > 5):
            raise ValueError("product(shape) must be <= 5, shape is " + str(shape))
        prod = self.prod(shape)
        values = random.sample(range(0, 5), prod)
        return tf.reshape(tf.Variable(values, dtype=dtype, name=n), shape)

    def var_unique_rand_10(self, shape, dtype, n):
        if(self.prod(shape) > 10):
            raise ValueError("product(shape) must be <= 10, shape is " + str(shape))
        prod = self.prod(shape)
        values = random.sample(range(0, 10), prod)
        return tf.reshape(tf.Variable(values, dtype=dtype, name=n), shape)

    def var_segment3(self, shape, dtype, n):
        return self.var_segmentN(3, shape, dtype, n)

    def var_segment5(self, shape, dtype, n):
        return self.var_segmentN(5, shape, dtype, n)

    def var_segmentN(self, numSegments, shape, dtype, n):
        length = np.prod(shape)
        numPerSegment = length // numSegments
        segmentIds = []
        for i in range(length):
            segmentIds.append(min(numSegments-1, i//numPerSegment))
        return tf.Variable(tf.constant(value=segmentIds, dtype=dtype, shape=shape), name=n)

    def var_bernoulli(self, shape, dtype, n):
        #Random 0 or 1
        return tf.cast((tf.random_uniform(shape) < 0.5), dtype=dtype)

    def var_onehot(self, shape, dtype, n):
        if(len(shape) is not 2):
            raise ValueError("Only rank 2 input supported so far")

        # x = np.zeros(shape)
        # y = np.random.choice(shape[0], shape[1])
        # x[np.arange(shape[0]), y] = 1

        x = np.eye(shape[1])[np.random.choice(shape[1], size=shape[0])]

        return tf.Variable(x, dtype=dtype, name=n)

    def var_empty(self, shape, dtype, n):
        foundZero = False
        for v in shape:
            if(v == 0):
                foundZero = True
        print("SHAPE: ", shape)

        if(len(shape) > 0 and foundZero == False):
            raise ValueError("At least one entry in empty array must be 0 (or length 0)")

        # fill = tf.fill(shape, value=0)
        # var = tf.Variable(tf.cast(fill, dtype=dtype))
        var = tf.constant([], shape=shape, dtype=dtype)
        dummy = tf.Variable([1], dtype=tf.float32)  #workaround to TF restriction where only graphs with 1 or more variables can be exported
        return [var]

    def var_string_scalar(self, shape,  dtype, n):
        return tf.constant(u"This is a test string")

    def var_boolean(self, shape, dtype, n):
        print(dtype)
        # if(dtype is not tf.bool):
        return tf.Variable(tf.random_uniform(shape) >= 0.5, dtype=dtype)

    def var_booleanFalse(self, shape, dtype, n):
        return tf.Variable(tf.cast(tf.fill(dims=shape, value=False), dtype=dtype), name=n)

    def var_booleanTrue(self, shape, dtype, n):
        return tf.Variable(tf.cast(tf.fill(dims=shape, value=True), dtype=dtype), name=n)

    def var_positive_def_symmetric_33(self, shape, dtype, n):
        if(len(shape) is not 2 or shape[0] is not 3 or shape[1] is not 3):
            raise ValueError("Only 3x3 inputs allowed for this initializer")

        return tf.Variable(tf.constant([[1.77878559, 0.23089433, 0.50563407],[0.23089433, 1.72714126, 0.89252293],[0.50563407, 0.89252293, 1.54612088]], dtype=dtype))

    def var_positive_def_symmetric_233(self, shape, dtype, n):
        if(len(shape) is not 3 or shape[0] is not 2 or shape[1] is not 3 or shape[1] is not 3):
            raise ValueError("Only 2x3x3 inputs allowed for this initializer")

        v1 = self.var_positive_def_symmetric_33([3,3], dtype, n)
        v2 = self.var_positive_def_symmetric_33([3,3], dtype, n)
        return tf.stack([v1, v2], axis=0)

    def newPlaceholder(self, initType, shape, dtype, name):
        method_name = "placeholder_" + initType
        try:
            method = getattr(self, method_name)
        except AttributeError:
            print(method_name, "not found")
        else:
            return method(shape, dtype, name)

    def placeholder_zero(self, shape, dtype, n):
        return [tf.placeholder(dtype=dtype, shape=shape, name=n),
                np.zeros(shape, dtype.as_numpy_dtype())]

    def placeholder_one(self, shape, dtype, n):
        return [tf.placeholder(dtype=dtype, shape=shape, name=n),
                np.ones(shape, dtype.as_numpy_dtype())]

    def var_ragged2d(self, shape, dtype, n):
        values = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8], dtype=dtype, name=n)
        out = tf.RaggedTensor.from_row_splits(values=values,row_splits=[0, 4, 4, 7, 8, 8])

        if dtype is not out.values.dtype:
            return tf.cast(out, dtype=dtype)
        return out

    def var_string2(self, shape, dtype, n):
        return tf.Variable(["hello world", "a b c"], dtype=tf.string)
