from tfoptests.test_graph import TestGraph
import numpy as np


# NHWC is default

class ImageInput(TestGraph):
    def __init__(self, batch_size=None, image_h=None, image_w=None, image_c=None, *args, **kwargs):
        self.image_w = image_w
        self.image_h = image_h
        self.image_c = image_c
        self.batch_size = batch_size
        super(ImageInput, self).__init__(*args, **kwargs)
        self.images = np.random.uniform(size=(self.batch_size, self.image_h, self.image_w, self.image_c))

    def list_inputs(self):
        return ["image"]

    def get_placeholder_input(self, name):
        if name == "image":
            return self.images

    def _get_placeholder_shape(self, name):
        if name == "image":
            return [None, self.image_h, self.image_w, self.image_c]
