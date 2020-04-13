import tensorflow as tf
import numpy as np


class ReduceOps:
    def __init__(self, vars, shapes, types, axis, extra):
        self.vars = vars
        self.shapes = shapes
        self.types = types
        self.axis = axis
        self.node_num = 0
        self.extra = extra

    def set_a(self, a):
        self.a = a

    def set_b(self, b):
        self.b = b

    def execute(self, some_op):
        self.node_num += 1
        method_name = 'execute_' + some_op
        try:
            method = getattr(self, method_name)
        except AttributeError:
            print(method_name, "not found")
        else:
            return method()

    def execute_reduce_sum(self):
        return [tf.reduce_sum(input_tensor=self.vars[0], axis=self.axis, keepdims=self.extra.get("keepdims", False), name="reduce_sum" + str(self.node_num))]

    def execute_reduce_max(self):
        return [tf.reduce_max(input_tensor=self.vars[0], axis=self.axis, keepdims=self.extra.get("keepdims", False), name="reduce_max" + str(self.node_num))]

    def execute_reduce_min(self):
        return [tf.reduce_min(input_tensor=self.vars[0], axis=self.axis, keepdims=self.extra.get("keepdims", False), name="reduce_min" + str(self.node_num))]

    def execute_reduce_mean(self):
        return [tf.reduce_mean(input_tensor=self.vars[0], axis=self.axis, keepdims=self.extra.get("keepdims", False), name="reduce_mean" + str(self.node_num))]

    def execute_reduce_prod(self):
        return [tf.reduce_prod(input_tensor=self.vars[0], axis=self.axis, keepdims=self.extra.get("keepdims", False), name="reduce_prod" + str(self.node_num))]

    def execute_is_non_decreasing(self):
        return [tf.math.is_non_decreasing(self.vars[0], name="is_non_decreasing-" + str(self.node_num))]

    def execute_argmax(self):
        return [tf.argmax(input=self.vars[0], axis=self.axis, name="argmax-" + str(self.node_num))]

    def execute_argmin(self):
        return [tf.argmin(input=self.vars[0], axis=self.axis, name="argmin-" + str(self.node_num))]

    def execute_add_n(self):
        return [tf.add_n(self.vars, name="add_n-" + str(self.node_num))]

    def execute_moments(self):
        return tf.nn.moments(x=self.vars[0], axes=self.axis, keepdims=self.extra.get("keepdims", False))

    def execute_count_nonzero(self):
        return [tf.math.count_nonzero(self.vars[0], axis=self.axis, name="count_nonzero-" + str(self.node_num))]

    def execute_normalize_moments(self):
        shift = None
        if(len(self.vars) > 3):
            shift = self.vars[3]
        return tf.nn.normalize_moments(counts=self.vars[0], mean_ss=self.vars[1], variance_ss=self.vars[2], shift=shift)

    def execute_scatter_add(self):
        # Create an intermediate variable - otherwise the scatter op will modify the variable content in-place
        # and hence we'll save the input post-modification, rather than pre-modification
        intermediate = tf.Variable(tf.zeros(self.shapes[0]), dtype=tf.float32)
        intermediate = tf.compat.v1.assign(intermediate, self.vars[0])
        return [tf.compat.v1.scatter_add(ref=intermediate, indices=self.vars[1], updates=self.vars[2], name="scatter_add-" + str(self.node_num))]

    def execute_scatter_sub(self):
        # Create an intermediate variable - otherwise the scatter op will modify the variable content in-place
        # and hence we'll save the input post-modification, rather than pre-modification
        intermediate = tf.Variable(tf.zeros(self.shapes[0]), dtype=tf.float32)
        intermediate = tf.compat.v1.assign(intermediate, self.vars[0])
        return [tf.compat.v1.scatter_sub(ref=intermediate, indices=self.vars[1], updates=self.vars[2], name="scatter_sub-" + str(self.node_num))]

    def execute_scatter_mul(self):
        # Create an intermediate variable - otherwise the scatter op will modify the variable content in-place
        # and hence we'll save the input post-modification, rather than pre-modification
        intermediate = tf.Variable(tf.zeros(self.shapes[0]), dtype=tf.float32)
        intermediate = tf.compat.v1.assign(intermediate, self.vars[0])
        return [tf.compat.v1.scatter_mul(ref=intermediate, indices=self.vars[1], updates=self.vars[2], name="scatter_mul-" + str(self.node_num))]

    def execute_scatter_div(self):
        # Create an intermediate variable - otherwise the scatter op will modify the variable content in-place
        # and hence we'll save the input post-modification, rather than pre-modification
        intermediate = tf.Variable(tf.zeros(self.shapes[0]), dtype=tf.float32)
        intermediate = tf.compat.v1.assign(intermediate, self.vars[0])
        return [tf.compat.v1.scatter_div(ref=intermediate, indices=self.vars[1], updates=self.vars[2], name="scatter_div-" + str(self.node_num))]

    def execute_scatter_update(self):
        # Create an intermediate variable - otherwise the scatter op will modify the variable content in-place
        # and hence we'll save the input post-modification, rather than pre-modification
        intermediate = tf.Variable(tf.zeros(self.shapes[0]), dtype=tf.float32)
        intermediate = tf.compat.v1.assign(intermediate, self.vars[0])
        return [tf.compat.v1.scatter_update(ref=intermediate, indices=self.vars[1], updates=self.vars[2], name="scatter_update-" + str(self.node_num))]

    def execute_scatter_max(self):
        # Create an intermediate variable - otherwise the scatter op will modify the variable content in-place
        # and hence we'll save the input post-modification, rather than pre-modification
        intermediate = tf.Variable(tf.zeros(self.shapes[0]), dtype=tf.float32)
        intermediate = tf.compat.v1.assign(intermediate, self.vars[0])
        return [tf.compat.v1.scatter_max(ref=intermediate, indices=self.vars[1], updates=self.vars[2], name="scatter_max-" + str(self.node_num))]

    def execute_scatter_min(self):
        # Create an intermediate variable - otherwise the scatter op will modify the variable content in-place
        # and hence we'll save the input post-modification, rather than pre-modification
        intermediate = tf.Variable(tf.zeros(self.shapes[0]), dtype=tf.float32)
        intermediate = tf.compat.v1.assign(intermediate, self.vars[0])
        return [tf.compat.v1.scatter_min(ref=intermediate, indices=self.vars[1], updates=self.vars[2], name="scatter_min-" + str(self.node_num))]

