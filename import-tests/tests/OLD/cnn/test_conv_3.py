import tensorflow as tf

from tests.cnn.image_input import ImageInput
from tfoptests.persistor import TensorFlowPersistor


def test_conv_3():
    # [batch, in_height, in_width, in_channels].
    image_input = ImageInput(seed=713, batch_size=4, image_h=128, image_w=128, image_c=4)
    in_node = image_input.get_placeholder("image", data_type=tf.float32)
    # [filter_height, filter_width, in_channels, out_channels]. in_channels must match between input and filter.
    filter_one = tf.Variable(tf.random_uniform([4, 5, image_input.image_c]), name="filter1")
    dilation_one = tf.nn.dilation2d(in_node, filter=filter_one, strides=[1, 2, 3, 1], rates=[1, 5, 7, 1],
                                    padding='SAME',
                                    name="dilation_one")
    filter_two = tf.Variable(tf.random_uniform([11, 7, 4]), name="filter2")
    dilation_two = tf.nn.dilation2d(dilation_one, filter=filter_two, strides=[1, 3, 2, 1], rates=[1, 2, 3, 1],
                                    padding='VALID',
                                    name="output")
    out_node = tf.identity(dilation_two, name="output")
    placeholders = [in_node]
    predictions = [out_node]

    tfp = TensorFlowPersistor(save_dir="conv_3")
    predictions_after_freeze = tfp \
        .set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(image_input.get_test_data()) \
        .build_save_frozen_graph()
    print(predictions_after_freeze[0].shape)


if __name__ == '__main__':
    test_conv_3()
