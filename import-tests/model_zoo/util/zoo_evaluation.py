from io import BytesIO
import tensorflow as tf
import numpy as np
from six.moves import urllib
from PIL import Image   #pip install Pillow
from tfoptests.persistor import TensorFlowPersistor
import vgg_preprocessing
import inception_preprocessing

class ZooEvaluation(object):

    def __init__(self, name, baseDir="/dl4j-test-resources/src/main/resources/tf_graphs/zoo_models", prefix="graph"):
        tf.reset_default_graph()
        self.name = name
        self.baseDir = baseDir
        self.prefix = prefix

    def graphFile(self, graphFile):
        self.graphFile = graphFile
        return self

    def outputDir(self, outputDir):
        self.outputDir = outputDir
        return self

    def imageUrl(self, imageUrl):
        self.imageUrl = imageUrl
        return self

    def inputDims(self, h, w, c):
        self.inputH = h
        self.inputW = w
        self.inputC = c
        return self

    def inputName(self, inputName):
        self.inputName = inputName
        return self

    def outputNames(self, outputNames):
        self.outputNames = outputNames
        return self

    def preprocessingType(self, preprocessingType):
        self.preprocessingType = preprocessingType
        return self

    def getImage(self, expandDim):
        if( self.inputH is None ):
            raise ValueError("input height not set")
        if( self.inputC is not 3):
            raise ValueError("Only ch=3 implemented so far")
        if( self.imageUrl is None):
            raise ValueError("Image URL is not set")
        if( self.inputH != self.inputW):
            raise ValueError("Non-square inputs not yet implemented")

        try:
            f = urllib.request.urlopen(self.imageUrl)
            jpeg_str = f.read()
            image = Image.open(BytesIO(jpeg_str))
        except IOError:
            raise ValueError('Cannot retrieve image. Please check url: ' + self.imageUrl)

        image = np.asarray(image.convert('RGB'))
        print("image shape: ", image.shape)
        m = min(image.shape[0], image.shape[1])
        print("min: ", m)
        if(self.preprocessingType == "vgg"):
            image = vgg_preprocessing.preprocess_for_eval(image, self.inputH, self.inputW, m)
        elif(self.preprocessingType == "inception"):
            # image = image.astype("float32")
            # with tf.Session() as sess:
            #     image = image.eval(session=sess)
            image = tf.convert_to_tensor(image)
            image = inception_preprocessing.preprocess_for_eval(image, height=self.inputH, width=self.inputW)
        elif(self.preprocessingType == "resize_only"):
            image = np.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [self.inputH, self.inputW], align_corners=False)
            image = tf.squeeze(image, axis=0)
            # image = tf.convert_to_tensor(image)
        else:
            raise ValueError("Unknown preprocessing type: ", self.preprocessingType)
        # image = tf.expand_dims(image, 0)
        print("new shape: ", image.shape)
        print("new type: ", type(image))

        with tf.Session() as sess:
            image = image.eval(session=sess)
        # print("new shape2: ", image.shape)
        # print("new type2: ", type(image))

        if(len(image.shape) == 3 and expandDim):
            image = np.expand_dims(image, 0)

        print("new shape3: ", image.shape)
        print("new type3: ", type(image))

        return image

        # width, height = image.size
        # #To resize keeping aspect ratio:
        # #resize_ratio = 1.0 * self.inputH / max(width, height)
        # #target_size = (int(resize_ratio * width), int(resize_ratio * height))
        # target_size = [self.inputH, self.inputW]
        # # print("Target size: ", target_size)
        # resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        # input = np.asarray(resized_image)
        # return input

    def write(self):
        if( self.inputName is None or self.outputNames is None ):
            raise ValueError("inputName or outputName not set")

        graph = self.loadGraph()
        image = self.getImage(False)

        if(image is None):
            raise ValueError("Null image")

        print("Input name: ", self.inputName)
        print("Output names: ", self.outputNames)
        print("Input shape: ", image.shape)
        with tf.Session(graph=graph) as sess:
            outputs = sess.run(
                # self.outputName,
                self.outputNames,
                feed_dict={self.inputName: [image]})
            print(outputs)

        print("Outputs: ",outputs)

        toSave = {}
        toSave_dtype_dict = {}
        for i in range(len(outputs)):
            toSave[self.outputNames[i]] = outputs[i]
            print("Output: ", self.outputNames[i])
            print("Dtype: ", outputs[i].dtype)
            toSave_dtype_dict[self.outputNames[i]] = str(outputs[i].dtype)

        #print("Values to save: ", toSave)
        tfp = TensorFlowPersistor(base_dir=self.baseDir, save_dir=self.name, verbose=False)
        tfp._save_input(self.getImage(True), self.inputName)
        dtype_dict = {}
        dtype_dict[self.inputName] = str(image.dtype)
        tfp._save_node_dtypes(dtype_dict)
        # tfp._save_predictions({self.outputName:outputs})
        tfp._save_predictions(toSave)
        tfp._save_node_dtypes(toSave_dtype_dict)


        #Also save intermediate nodes:
        # dict = {self.inputName:image}
        # dict = {self.inputName:self.getImage(True)}
        # print("DICTIONARY: ", dict)
        # print("OUTPUT NAMES: ", self.outputNames)
        # tfp.set_output_tensors(self.outputNames)
        # tfp.set_verbose(True)
        # tfp._save_intermediate_nodes2(input_dict=dict, graph=graph)


    def loadGraph(self):
        if( self.graphFile is None ):
            raise ValueError("Graph file location not set")

        with tf.gfile.GFile(self.graphFile, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name=self.prefix)
        return graph

if __name__ == '__main__':


    #How to run this?
    #1. Download the required models, links below
    #2. For many models, just implement using the ZooEvaluation class.
    #3. Some models need special treatment - see eval_data directory for these cases
    #4. Run with: python model_zoo/util/zoo_evaluation.py
    #See also: https://gist.github.com/eraly/7d48807ed2c69233072ed06c12bf9b0a



    #DenseNet - uses vgg preprocessing according to readme
    # https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/densenet_2018_04_27.tgz
    # z = ZooEvaluation(name="densenet_2018_04_27",prefix="")
    # z.graphFile("C:\\Temp\\TF_Graphs\\densenet_2018_04_27\\densenet.pb")\
    #     .inputName("Placeholder:0")\
    #     .outputNames(["ArgMax:0", "softmax_tensor:0"])\
    #     .imageUrl("https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/img/image2.jpg?raw=true")\
    #     .inputDims(224, 224, 3)\
    #     .preprocessingType("vgg")
    # z.write()
    # # SqueezeNet: also vgg preprocessing
    # # https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz
    # z = ZooEvaluation(name="squeezenet_2018_04_27",prefix="")
    # z.graphFile("C:\\Temp\\TF_Graphs\\squeezenet_2018_04_27\\squeezenet.pb")\
    #     .inputName("Placeholder:0") \
    #     .outputNames(["ArgMax:0", "softmax_tensor:0"])\
    #     .imageUrl("https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/img/image2.jpg?raw=true")\
    #     .inputDims(224, 224, 3)\
    #     .preprocessingType("vgg")
    # z.write()
    #
    # nasnet_mobile: no preprocessing specified, going to assume inception preprocessing
    # https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz
    # z = ZooEvaluation(name="nasnet_mobile_2018_04_27",prefix="")
    # z.graphFile("C:\\Temp\\TF_Graphs\\nasnet_mobile_2018_04_27\\nasnet_mobile.pb") \
    #     .inputName("input:0") \
    #     .outputNames(["final_layer/predictions:0"]) \
    #     .imageUrl("https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/img/image2.jpg?raw=true") \
    #     .inputDims(224, 224, 3) \
    #     .preprocessingType("inception")
    # z.write()
    #
    # # https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz
    # z = ZooEvaluation(name="inception_v4_2018_04_27",prefix="")
    # z.graphFile("C:\\Temp\\TF_Graphs\\inception_v4_2018_04_27\\inception_v4.pb") \
    #     .inputName("input:0") \
    #     .outputNames(["InceptionV4/Logits/Predictions:0"]) \
    #     .imageUrl("https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/img/image2.jpg?raw=true") \
    #     .inputDims(299, 299, 3) \
    #     .preprocessingType("inception")
    # z.write()
    #
    # # https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz
    # z = ZooEvaluation(name="inception_resnet_v2_2018_04_27",prefix="")
    # z.graphFile("C:\\Temp\\TF_Graphs\\inception_resnet_v2_2018_04_27\\inception_resnet_v2.pb") \
    #     .inputName("input:0") \
    #     .outputNames(["InceptionResnetV2/AuxLogits/Logits/BiasAdd:0"]) \
    #     .imageUrl("https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/img/image2.jpg?raw=true") \
    #     .inputDims(299, 299, 3) \
    #     .preprocessingType("inception")
    # z.write()
    #
    # # http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_128.tgz
    # z = ZooEvaluation(name="mobilenet_v1_0.5_128",prefix="")   #None)  #"mobilenet_v1_0.5")
    # z.graphFile("C:\\Temp\\TF_Graphs\\mobilenet_v1_0.5_128\\mobilenet_v1_0.5_128_frozen.pb") \
    # .inputName("input:0") \
    # .outputNames(["MobilenetV1/Predictions/Reshape_1:0"]) \
    # .imageUrl("https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/img/image2.jpg?raw=true") \
    # .inputDims(128, 128, 3) \
    # .preprocessingType("inception")     #Not 100% sure on this, but more likely it's inception than vgg preprocessing...
    # z.write()
    #
    # # http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz
    # z = ZooEvaluation(name="mobilenet_v2_1.0_224",prefix="")
    # z.graphFile("C:\\Temp\\TF_Graphs\\mobilenet_v2_1.0_224\\mobilenet_v2_1.0_224_frozen.pb") \
    #     .inputName("input:0") \
    #     .outputNames(["MobilenetV2/Predictions/Reshape_1:0"]) \
    #     .imageUrl("https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/img/image2.jpg?raw=true") \
    #     .inputDims(224, 224, 3) \
    #     .preprocessingType("inception")     #Not 100% sure on this, but more likely it's inception than vgg preprocessing...
    # z.write()
    #
    # http://download.tensorflow.org/models/official/resnetv2_imagenet_frozen_graph.pb
    # https://github.com/tensorflow/models/blob/master/research/tensorrt/README.md
    # "Some ResNet models represent 1000 categories, and some represent all 1001, with the 0th category being "background". The models provided are of the latter type."
    # Runs "imagenet_preprocessing" on the images - https://github.com/tensorflow/models/blob/master/official/resnet/imagenet_preprocessing.py
    # Which seems to be merely scaling, no normalization
    # z = ZooEvaluation(name="resnetv2_imagenet_frozen_graph",prefix="")
    # z.graphFile("/TF_Graphs/resnetv2_imagenet_frozen_graph.pb") \
    #     .inputName("input_tensor:0") \
    #     .outputNames(["softmax_tensor:0"]) \
    #     .imageUrl("https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/img/image2.jpg?raw=true") \
    #     .inputDims(224, 224, 3) \
    #     .preprocessingType("resize_only")
    # z.write()


    # # http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
    # # https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
    # # Seems like this can be use on (nearly) any image size??? Docs and notebook are very vague on this point
    # # 320x320 input is arbitrary, not based on anything
    # # Outputs: detection_boxes, detection_scores, num_detections, detection_classes
    # # Preprocessing: unclear from docs, but it does have some preprocessing built into the network
    # z = ZooEvaluation(name="ssd_mobilenet_v1_coco_2018_01_28",prefix="")
    # z.graphFile("C:\\Temp\\TF_Graphs\\ssd_mobilenet_v1_coco_2018_01_28\\frozen_inference_graph.pb") \
    #     .inputName("image_tensor:0") \
    #     .outputNames(["detection_boxes:0", "detection_scores:0", "num_detections:0", "detection_classes:0"]) \
    #     .imageUrl("https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/img/image2.jpg?raw=true") \
    #     .inputDims(320, 320, 3) \
    #     .preprocessingType("resize_only")     #Not 100% sure on this, but seems most likely
    # z.write()

    # http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz
    # As above, for ssd_mobilenet_v1_coco_2018_01_28
    # z = ZooEvaluation(name="ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03",prefix="")
    # z.graphFile("C:\\Temp\\TF_Graphs\\ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03\\frozen_inference_graph.pb") \
    #     .inputName("image_tensor:0") \
    #     .outputNames(["detection_boxes:0", "detection_scores:0", "num_detections:0", "detection_classes:0"]) \
    #     .imageUrl("https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/img/image2.jpg?raw=true") \
    #     .inputDims(300, 300, 3) \
    #     .preprocessingType("resize_only")     #Not 100% sure on this, but seems most likely
    # z.write()

    # http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz
    # https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
    # Runtimes are reported for this model on 600x600 images, so use that as it's definitely supported
    # z = ZooEvaluation(name="faster_rcnn_resnet101_coco_2018_01_28",prefix="")
    # # z.graphFile("C:\\Temp\\TF_Graphs\\faster_rcnn_resnet101_coco_2018_01_28\\frozen_inference_graph.pb") \
    # z.graphFile("/TF_Graphs/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb") \
    #     .inputName("image_tensor:0") \
    #     .outputNames(["detection_boxes:0", "detection_scores:0", "num_detections:0", "detection_classes:0"]) \
    #     .imageUrl("https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/img/image2.jpg?raw=true") \
    #     .inputDims(600, 600, 3) \
    #     .preprocessingType("resize_only")     #Should be this, given it has a crop and resize op internally? Not 100% sure on normalization
    # z.write()

    # graph = z.loadGraph()
    # for op in graph.get_operations():
    #     print(op.name)



