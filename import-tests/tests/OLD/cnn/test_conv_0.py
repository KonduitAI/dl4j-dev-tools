import tensorflow as tf

from tests.cnn.image_input import ImageInput
from tfoptests.nn_image_ops import NNImageOps
from tfoptests.persistor import TensorFlowPersistor


def test_conv_0():
    image_input = ImageInput(seed=713, batch_size=4, image_h=28, image_w=28, image_c=3)
    in_node = image_input.get_placeholder("image", data_type=tf.float32)
    constr = NNImageOps(in_node)
    constr.set_filter_hw_inout(h=5, w=5, in_ch=3, out_ch=3)
    constr.set_kernel_hw(3, 3)
    constr.set_stride_hw(3, 3)

    in1 = constr.execute("conv2d")
    constr.set_image(in1)

    in2 = constr.execute("avg_pool")
    constr.set_image(in2)

    in3 = constr.execute("conv2d")
    constr.set_image(in3)

    in4 = constr.execute("max_pool")

    in5 = constr.flatten_convolution(in4)

    out_node = tf.matmul(in5, tf.Variable(tf.random_uniform([3, 2])), name="output")  # calc required dims by hand

    placeholders = [in_node]
    predictions = [out_node]

    tfp = TensorFlowPersistor(save_dir="conv_0")
    predictions_after_freeze = tfp \
        .set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(image_input.get_test_data()) \
        .build_save_frozen_graph()
    print(predictions_after_freeze[0].shape)


if __name__ == '__main__':
    test_conv_0()
