import tensorflow as tf

from tests.cnn.image_input import ImageInput
from tfoptests.persistor import TensorFlowPersistor


def test_avg_pool():
    save_dir = "../tfProfiling/avg_pool"
    image_input = ImageInput(seed=713, batch_size=16, image_h=224, image_w=224, image_c=64)
    in_node = image_input.get_placeholder("image", data_type=tf.float32)
    dummy_var = tf.Variable(tf.zeros([2]))  # saver barfs if there are no variables in the graph??
    avg_pool_output = tf.layers.average_pooling2d(
        inputs=in_node,
        pool_size=[2, 2],
        strides=2,
        padding="same")
    out_node = tf.identity(avg_pool_output, name="output")

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
    test_avg_pool()
