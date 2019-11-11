import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2
import numpy as np

height = 299
width = 299
channels = 3

# Create graph
X = tf.placeholder(tf.float32, shape=[None, height, width, channels])
with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    logits, end_points = resnet_v2.re(X, is_training=False)
all_saver = tf.train.Saver()

X_test = np.ones((1, 299, 299, 3))  # a fake image, you can use your own image

# Execute graph
# TODO: replace Susan's path
with tf.Session() as sess:
    all_saver.restore(sess, "/Users/susaneraly/SKYMIND/TFImport/TF_SOURCE_CODE/downloads_from_slim/inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt")
    predictions_val = logits.eval(feed_dict={X: X_test})
    tf.train.write_graph(sess.graph_def, '/Users/susaneraly/SKYMIND/TFImport/TF_SOURCE_CODE/downloads_from_slim/inception_resnet_v2', 'resnetv2.pbtxt')
