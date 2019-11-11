import tensorflow as tf

from tests.cnn.image_input import ImageInput
from tfoptests.nn_image_ops import NNImageOps
from tfoptests.persistor import TensorFlowPersistor

'''
graph configuration is simply with the intent of including ops for import test...
'''
def test_conv_4():
    image_input = ImageInput(seed=713, batch_size=3, image_h=4, image_w=4, image_c=5)
    in_node = image_input.get_placeholder("image", data_type=tf.float32)
    constr = NNImageOps(in_node)
    constr.set_filter_size([7, 7, 16, 5])
    constr.set_output_shape([3, 8, 8, 16])
    constr.set_stride_size([1, 2, 2, 1])
    in1 = constr.execute("conv2d_transpose")  # size is (3, 8, 8, 16)
    constr.set_image(in1)
    constr.set_filter_size([2, 2, 32, 16])
    constr.set_output_shape([3, 8, 8, 32])
    constr.set_rate(2)
    in02 = constr.execute("atrous_conv2d_transpose")  # size is (3,8,8,32)
    constr.set_image(in02)
    constr.set_filter_size([2, 2, 32, 2])
    constr.set_pointwise_filter([1, 1, 64, 8])
    constr.set_stride_size([1, 1, 1, 1])
    constr.set_rate([1, 2])
    in2 = constr.execute("separable_conv2d")  # (3, 8, 8, 8)
    constr.set_image(in2)
    constr.set_filter_size([2, 2, 8, 4])
    constr.set_stride_size([1, 1, 1, 1])
    constr.set_rate([3, 2])
    in3 = constr.execute("depthwise_conv2d")  # size (3,8,8,32)
    in33 = tf.space_to_batch_nd(in3, [3, 3], paddings=[[1, 0], [0, 1]])  # size (27,3,3,32)
    constr.set_image(tf.reshape(in33, [27 * 2, 12, 12]))
    constr.set_filter_size([2, 12, 2])
    constr.set_stride_size(1)
    in4 = constr.execute("conv1d")  # size (54,12,2)
    out_node = tf.identity(in4, name="output")
    placeholders = [in_node]
    predictions = [out_node]

    tfp = TensorFlowPersistor(save_dir="conv_4")
    predictions_after_freeze = tfp \
        .set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(image_input.get_test_data()) \
        .build_save_frozen_graph()
    print(predictions_after_freeze[0].shape)

if __name__ == '__main__':
    test_conv_4()
