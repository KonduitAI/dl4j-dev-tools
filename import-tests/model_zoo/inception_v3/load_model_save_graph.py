import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import inception_v3
import numpy as np

from tfoptests import persistor
from model_zoo.inception_v3 import save_dir, get_input

height = 299
width = 299
channels = 3

# Create graph
X = tf.placeholder(tf.float32, shape=[None, height, width, channels],name="input")
my_feed_dict = {}
my_feed_dict[X] = get_input("input")
with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    net, end_points = inception_v3.inception_v3(X,num_classes=1001)
logits = end_points['Logits']
output = tf.nn.softmax(logits,name="output")
all_saver = tf.train.Saver()

# Execute graph
with tf.Session() as sess:
    all_saver.restore(sess, "/Users/susaneraly/SKYMIND/TFImport/TF_SOURCE_CODE/downloads_from_slim/inception_v3/inception_v3.ckpt")
    prediction = output.eval(feed_dict=my_feed_dict)
    print prediction
    print prediction.shape
    print(np.sort(prediction.ravel()))
    tf.train.write_graph(sess.graph_def, '/Users/susaneraly/SKYMIND/TFImport/TF_SOURCE_CODE/downloads_from_slim/inception_v3', 'inception_v3.pbtxt')
    persistor.save_graph(sess, all_saver, save_dir)
    persistor.save_prediction(save_dir, prediction)