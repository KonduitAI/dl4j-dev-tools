import tensorflow as tf

from tests.cnn.image_input_4d import ImageInput4D
from tfoptests.nn_image_ops import NNImageOps
from tfoptests.persistor import TensorFlowPersistor


def test_conv_1():
    # [4, 2, 28, 28, 3]
    image_input = ImageInput4D(seed=713, batch_size=4, in_d=2, in_h=28, in_w=28, in_ch=3)
    in_node = image_input.get_placeholder("image", data_type=tf.float32)
    constr = NNImageOps(in_node)
    # in_channels must match between input and filter.
    # [filter_depth, filter_height, filter_width, in_channels, out_channels].
    filter = [2, 5, 5, 3, 4]
    constr.set_filter_size(filter)
    # Must have strides[0] = strides[4] = 1.
    stride = [1, 5, 4, 3, 1]
    constr.set_stride_size(stride_val=stride)
    in1 = constr.execute("conv3d")
    constr.set_image(in1)
    in2 = constr.flatten_convolution(in1)
    out_node = tf.matmul(in2, tf.Variable(tf.random_uniform([280, 3])), name="output")  # calc required dims by hand
    placeholders = [in_node]
    predictions = [out_node]

    tfp = TensorFlowPersistor(save_dir="conv_1")
    predictions_after_freeze = tfp \
        .set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(image_input.get_test_data()) \
        .build_save_frozen_graph()
    print(predictions_after_freeze[0].shape)


if __name__ == '__main__':
    test_conv_1()
