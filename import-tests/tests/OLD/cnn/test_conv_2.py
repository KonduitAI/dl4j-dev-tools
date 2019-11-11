import tensorflow as tf

from tests.cnn.image_input import ImageInput
from tfoptests.persistor import TensorFlowPersistor


def test_conv_2():
    image_input = ImageInput(seed=713, batch_size=4, image_h=64, image_w=64, image_c=4)
    in_node = image_input.get_placeholder("image", data_type=tf.float32)
    # in_channels must match between input and filter.
    # filter shape is [filter_height, filter_width, in_channels, out_channels]
    filter_one = tf.Variable(tf.random_uniform([4, 5, image_input.image_c, 2]), name="filter1")
    atrous_one = tf.nn.atrous_conv2d(in_node, filters=filter_one, rate=8, padding='SAME', name="atrous_one")
    filter_two = tf.Variable(tf.random_uniform([31, 31, 2, 1]), name="filter2")
    atrous_two = tf.nn.atrous_conv2d(atrous_one, filters=filter_two, rate=2, padding='VALID')
    out_node = tf.identity(atrous_two, name="output")

    placeholders = [in_node]
    predictions = [out_node]

    tfp = TensorFlowPersistor(save_dir="conv_2")
    predictions_after_freeze = tfp \
        .set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(image_input.get_test_data()) \
        .build_save_frozen_graph()
    print(predictions_after_freeze[0].shape)


if __name__ == '__main__':
    test_conv_2()
