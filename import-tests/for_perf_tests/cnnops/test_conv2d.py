import tensorflow as tf

from tests.cnn.image_input import ImageInput
from tfoptests.persistor import TensorFlowPersistor


def test_conv2d():
    save_dir = "../tfProfiling/conv2d"
    image_input = ImageInput(seed=713, batch_size=16, image_h=256, image_w=256, image_c=3)
    in_node = image_input.get_placeholder("image", data_type=tf.float32)
    conv_2d_output = tf.layers.conv2d(
        inputs=in_node,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    out_node = tf.identity(conv_2d_output, name="output")

    placeholders = [in_node]
    predictions = [out_node]
    tfp = TensorFlowPersistor(save_dir=save_dir, verbose=False)
    predictions_after_freeze = tfp \
        .set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(image_input.get_test_data()) \
        .build_save_frozen_graph(skip_intermediate=True)
    print(predictions_after_freeze[0].shape)


if __name__ == "main":
    test_conv2d()
