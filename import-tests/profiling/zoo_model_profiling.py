

import tensorflow as tf
import numpy as np
import os

from tensorflow.python.client import timeline


def loadGraph(path, prefix=""):

    with tf.gfile.GFile(path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name=prefix)
    return graph


def profile():


    # # http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz
    # z = ZooEvaluation(name="mobilenet_v2_1.0_224",prefix="")
    # z.graphFile("C:\\Temp\\TF_Graphs\\mobilenet_v2_1.0_224\\mobilenet_v2_1.0_224_frozen.pb") \
    #     .inputName("input:0") \
    #     .outputNames(["MobilenetV2/Predictions/Reshape_1:0"]) \
    #     .imageUrl("https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/img/image2.jpg?raw=true") \
    #     .inputDims(224, 224, 3) \
    #     .preprocessingType("inception")     #Not 100% sure on this, but more likely it's inception than vgg preprocessing...
    # z.write()

    outputBaseDir = "/ImportTests/profiling/"

    for batch in [1,32]:

        testname = "mobilenet_v2_1.0_224_batch" + str(batch)
        path = "/TF_Graphs/mobilenet_v2_1.0_224_frozen.pb"
        outputNames = ["MobilenetV2/Predictions/Reshape_1:0"]
        feed_dict = {"input:0": np.random.uniform(size=[batch,224,224,3])}


        outputDir = outputBaseDir + testname + "/"
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)

        # Load graph
        graph = loadGraph(path)

        #Warmup:
        for i in range(3):
            with tf.Session(graph=graph) as sess:
                outputs = sess.run(
                    outputNames,
                    feed_dict=feed_dict)
                print(outputs)

        #Profile:
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        for i in range(5):
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


