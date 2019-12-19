

import tensorflow as tf
import numpy as np
import os
import psutil
# import pip
import pkg_resources
import platform

from tensorflow.python.client import timeline


def loadGraph(path, prefix=""):

    with tf.gfile.GFile(path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name=prefix)
    return graph

def writeSystemInfo(path, data):
    with open(path, 'w') as f:

        for k in data.keys():
            f.write(k + ": " + str(data[k]) + "\n")

        f.write("===== Python Environment =====\n")
        installed_packages = sorted(["%s==%s" % (i.project_name, i.version) for i in pkg_resources.working_set], key=str.casefold)
        for p in installed_packages:
            f.write(p + "\n")

        f.write("---------------------------------------------------------------------\n")
        line = "CPU count: " + str(psutil.cpu_count()) + " - physical: " + str(psutil.cpu_count(logical=False)) + "\n"
        f.write(line)
        f.write("CPU: " + str(platform.processor()) + "\n")
        f.write("CPU freq: " + str(psutil.cpu_freq()) + "\n")
        f.write("Platform: " + str(platform.platform()) + "\n")
        f.write("Python: " + str(platform.python_version_tuple()) + "\n")
        f.write("System: " + str(platform.system()) + "\n")
        f.write("Uname: " + str(platform.uname()) + "\n")
        f.write("Memory: " + str(psutil.virtual_memory()) + "\n")
        f.write("---------------------------------------------------------------------\n")
        d = psutil.Process().as_dict(attrs=None, ad_value=None)
        for (k,v) in d.items():
            f.write("===== " + str(k) + " =====\n")
            f.write(str(v) + "\n")




def profile():
    outputBaseDir = "/home/alex/profiling/"
    inputBaseDir = "/home/alex/TF_Graphs/"

    tfversion = tf.__version__

    for test in ["mobilenetv2", "inception_resnet_v2", "faster_rcnn_resnet101_coco"]:

        for batch in [1,32]:

            warmup = 3
            runs = 10

            # tests adapted from: zoo_evaluation.py
            if test == "mobilenetv2":
                # http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz
                testname = "mobilenet_v2_1.0_224_batch" + str(batch) + "_tf-" + tfversion
                path = inputBaseDir + "mobilenet_v2_1.0_224_frozen.pb"
                outputNames = ["MobilenetV2/Predictions/Reshape_1:0"]
                feed_dict = {"input:0": np.random.uniform(size=[batch,224,224,3])}
                data = {"testname": testname, "path": path, "outputNames": outputNames, "inputs": feed_dict.keys()}
            elif test == "inception_resnet_v2":
                # https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz
                testname = "inception_resnet_v2_batch" + str(batch) + "_tf-" + tfversion
                path = inputBaseDir + "inception_resnet_v2.pb"
                outputNames = ["InceptionResnetV2/AuxLogits/Logits/BiasAdd:0"]
                feed_dict = {"input:0": np.random.uniform(size=[batch,299,299,3])}
                data = {"testname": testname, "path": path, "outputNames": outputNames, "inputs": feed_dict.keys()}
            elif test == "faster_rcnn_resnet101_coco":
                # http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz
                testname = "faster_rcnn_resnet101_coco_batch" + str(batch) + "_tf-" + tfversion
                path = inputBaseDir + "faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb"
                outputNames = ["detection_boxes:0", "detection_scores:0", "num_detections:0", "detection_classes:0"]
                feed_dict = {"image_tensor:0": np.random.uniform(size=[batch,600,600,3])}
                data = {"testname": testname, "path": path, "outputNames": outputNames, "inputs": feed_dict.keys()}
            else:
                raise ValueError("Unknown test name: test")


            outputDir = outputBaseDir + testname + "/"
            if not os.path.exists(outputDir):
                os.mkdir(outputDir)

            # Load graph
            graph = loadGraph(path)

            print("===== Starting Test - ", outputDir, " =====")
            sysinfoFile = outputDir + "system_info.txt"
            writeSystemInfo(sysinfoFile, data)

            #Warmup:
            print("Starting warmup: ", warmup, " iterations")
            for i in range(warmup):
                with tf.Session(graph=graph) as sess:
                    outputs = sess.run(
                        outputNames,
                        feed_dict=feed_dict)
                    # print(outputs)

            #Profile:
            print("Starting profiling: ", runs, " iterations")
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            for i in range(runs):
                with tf.Session(graph=graph) as sess:
                    outputs = sess.run(
                        outputNames,
                        feed_dict=feed_dict)

                    #Unfortunately, no way to combine multiple runs into one file :(
                    # https://github.com/tensorflow/tensorflow/issues/3489
                    run_metadata = tf.RunMetadata()
                    sess.run(outputNames, feed_dict=feed_dict, options=options, run_metadata=run_metadata)

                    # Create the Timeline object, and write it to a json file
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open(outputDir + "/profile" + str(i) + ".json", 'w') as f:
                        f.write(chrome_trace)



if __name__ == '__main__':
    profile()


