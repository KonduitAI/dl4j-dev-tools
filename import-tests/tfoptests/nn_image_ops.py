from __future__ import print_function

import tensorflow as tf
import numpy as np


class NNImageOps:
    def __init__(self, img):
        self.inp = img
        self.node_num = 0
        self.kernel_size = None
        self.strides = None
        self.padding = 'SAME'
        self.filter_size = None
        self.filter = None
        self.output_shape = None
        self.rate = None
        self.pointwise_filter = None

    def set_image(self, image):
        # [batch, in_height, in_width, in_channels]
        self.inp = image

    def set_kernel_hw(self, kernel_h=5, kernel_w=5):
        # for conv: filter/kernel: [filter_height, filter_width, in_channels, out_channels]
        # for pool: should be the size in each dimension of the input  - so [1, h, w, 1]
        self.kernel_size = [1, kernel_h, kernel_w, 1]

    def set_filter_hw_inout(self, h=28, w=28, in_ch=3, out_ch=64):
        self.filter_size = [h, w, in_ch, out_ch]

    def set_stride_hw(self, stride_h=1, stride_w=1):
        # strides = [1, stride, stride, 1]
        self.strides = [1, stride_h, stride_w, 1]

    def set_padding(self, padding):
        self.padding = padding

    def set_filter_size(self, filter_size_val=None):
        if filter_size_val is not None:
            self.filter_size = filter_size_val
        # filter/kernel: [filter_height, filter_width, in_channels, out_channels]
        self.filter = tf.Variable(tf.random_normal(self.filter_size), name="filterW" + str(self.node_num),
                                  dtype=tf.float32)

    def set_stride_size(self, stride_val):
        self.strides = stride_val

    def set_output_shape(self, output_shape):
        self.output_shape = output_shape

    def set_rate(self, rate):
        self.rate = rate

    def set_pointwise_filter(self, pointwise_filter):
        self.pointwise_filter = tf.Variable(tf.random_normal(pointwise_filter), name="pointwise_filterW" + str(self.node_num),
                                  dtype=tf.float32)

    def execute(self, some_op):
        self.node_num += 1
        method_name = 'execute_' + some_op
        try:
            method = getattr(self, method_name)
        except AttributeError:
            print(method_name, "not found")
        else:
            return method()

    def execute_avg_pool(self):
        return tf.nn.avg_pool(self.inp, ksize=self.kernel_size, strides=self.strides, padding=self.padding,
                              name="avg_pool" + str(self.node_num))

    def execute_avg_pool3d(self):
        return tf.nn.avg_pool3d(self.inp, ksize=self.kernel_size, strides=self.strides, padding=self.padding,
                                name="avgpool3d" + str(self.node_num))

    def execute_conv2d(self):
        self.set_filter_size()
        return tf.nn.conv2d(self.inp, self.filter, self.strides, self.padding, name="conv2d" + str(self.node_num))

    def execute_conv3d(self):
        return tf.nn.conv3d(self.inp, self.filter, self.strides, self.padding, name="conv3d" + str(self.node_num))

    def execute_max_pool(self):
        return tf.nn.max_pool(self.inp, self.kernel_size, self.strides, self.padding,
                              name="max_pool" + str(self.node_num))

    def execute_conv2d_transpose(self):
        return tf.nn.conv2d_transpose(self.inp, self.filter, self.output_shape, self.strides, self.padding,
                                      name="conv2d_transpose" + str(self.node_num))

    def execute_atrous_conv2d_transpose(self):
        return tf.nn.atrous_conv2d_transpose(self.inp, self.filter, self.output_shape, self.rate, self.padding,
                                             name="atrous_conv2d_transpose" + str(self.node_num))

    def execute_conv1d(self):
        return tf.nn.conv1d(self.inp, self.filter, self.strides, self.padding,
                            name="conv1d" + str(self.node_num))

    def execute_depthwise_conv2d(self):
        return tf.nn.depthwise_conv2d(self.inp, self.filter, self.strides, self.padding, self.rate,
                                      name="depthwise_conv2d" + str(self.node_num))

    def execute_separable_conv2d(self):
        return tf.nn.separable_conv2d(self.inp, self.filter, self.pointwise_filter, self.strides, self.padding,
                                      self.rate, name="separable_conv2d" + str(self.node_num))

    def flatten_convolution(self, tensor_in):
        tensor_in_shape = tensor_in.get_shape()
        tensor_in_flat = tf.reshape(
            tensor_in, [tensor_in_shape[0].value or -1, np.prod(tensor_in_shape[1:]).value])
        return tensor_in_flat
