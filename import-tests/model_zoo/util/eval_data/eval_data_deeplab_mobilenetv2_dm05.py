import os
from io import BytesIO
import tensorflow as tf
import numpy as np
from six.moves import urllib
from PIL import Image
from tfoptests.persistor import TensorFlowPersistor

# http://download.tensorflow.org/models/deeplabv3_mnv2_dm05_pascal_trainaug_2018_10_01.tar.gz
# https://github.com/tensorflow/models/blob/8caa269db25165fdf21e73262921aa31bc595d70/research/deeplab/g3doc/model_zoo.md
# https://github.com/tensorflow/models/blob/277a9ad5681c0534f1b079cf4bd080faa4f59695/research/deeplab/deeplab_demo.ipynb
# TODO THIS PROBABLY REQUIRES LATER VERSION OF TF - CAN'T LOAD ON 0.10?
def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="graph")
    return graph


if __name__ == '__main__':
    # file = "C:\Temp\TF_Graphs\deeplabv3_mnv2_dm05_pascal_trainaug\\frozen_inference_graph.pb"
    base_dir = "C:\\DL4J\\Git\\dl4j-test-resources\\src\\main\\resources\\tf_graphs\\zoo_models"
    graph = load_graph(file)

    # for op in graph.get_operations():
    #     print(op.name)

    url = 'https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/img/image2.jpg?raw=true'

    try:
        f = urllib.request.urlopen(url)
        jpeg_str = f.read()
        image = Image.open(BytesIO(jpeg_str))
    except IOError:
        print('Cannot retrieve image. Please check url: ' + url)

    print('running deeplab on image %s...' % url)
    #resized_im, seg_map = run(original_im)

    INPUT_SIZE = 513
    INPUT_TENSOR_NAME = 'graph/ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'graph/SemanticPredictions:0'

    width, height = image.size
    resize_ratio = 1.0 * INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    input = np.asarray(resized_image)
    with tf.Session(graph=graph) as sess:
        batch_seg_map = sess.run(
            OUTPUT_TENSOR_NAME,
            feed_dict={INPUT_TENSOR_NAME: [input]})
        seg_map = batch_seg_map[0]
        tfp = TensorFlowPersistor(base_dir=base_dir, save_dir="deeplab_mobilenetv2_dm05_coco_voc_trainaug")
        input4d = np.reshape(input, [1, input.shape[0], input.shape[1], input.shape[2]])
        tfp._save_input(input4d, "graph/ImageTensor")   #TF is weird here: placeholder is [1, -1, -1, 3] but it adds extra dimension if you pass 4d in :/
        tfp._save_predictions({"graph/SemanticPredictions":seg_map})

        #Save type info
        dtype_dict = {}
        dtype_dict[INPUT_TENSOR_NAME] = str(input.dtype)
        dtype_dict["graph/SemanticPredictions"] = str(seg_map.dtype)
        tfp._save_node_dtypes(dtype_dict)


    print(seg_map)



