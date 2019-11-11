from tfoptests.test_graph import TestGraph
import numpy as np


# [batch, in_depth, in_height, in_width, in_channels].
# NHWC is default

class ImageInput4D(TestGraph):
    def __init__(self, batch_size=None, in_d=None, in_h=None, in_w=None, in_ch=None, *args, **kwargs):
        self.batch_size = batch_size
        self.in_d = in_d
        self.in_h = in_h
        self.in_w = in_w
        self.in_ch = in_ch
        self.batch_size = batch_size
        super(ImageInput4D, self).__init__(*args, **kwargs)
        self.images = np.random.uniform(size=(self.batch_size, self.in_d, self.in_h, self.in_w, self.in_ch))

    def list_inputs(self):
        return ["image"]

    def get_placeholder_input(self, name):
        if name == "image":
            return self.images

    def _get_placeholder_shape(self, name):
        if name == "image":
            return [None, self.in_d, self.in_h, self.in_w, self.in_ch]
