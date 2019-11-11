import tensorflow as tf

from tests.cnn.image_input import ImageInput
from tfoptests.nn_image_ops import NNImageOps
from tfoptests.persistor import TensorFlowPersistor


def test_pool_1():
    # [batch, in_height, in_width, in_channels].
    image_input = ImageInput(seed=713, batch_size=1, image_h=4, image_w=4, image_c=2)
    in_node = image_input.get_placeholder("image", data_type=tf.float32)
    dummy_var = tf.Variable(tf.random_uniform([3, 2]))  # saver barfs without a variable
    constr = NNImageOps(in_node)
    constr.set_filter_hw_inout(2, 2, 2, 2)
    constr.set_kernel_hw(2, 2)
    constr.set_stride_hw(1, 1)
    in1 = constr.execute("max_pool")
    out_node = tf.identity(in1, name="output")  # calc required dims by hand

    placeholders = [in_node]
    predictions = [out_node]

    tfp = TensorFlowPersistor(save_dir="pool_1")
    predictions_after_freeze = tfp \
        .set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(image_input.get_test_data()) \
        .build_save_frozen_graph()
    print(predictions_after_freeze[0].shape)


if __name__ == "__main__":
    test_pool_1()
