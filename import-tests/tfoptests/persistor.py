from __future__ import print_function

try:
    from itertools import izip as zip
except:
    # just use plain zip in py3
    pass
import sys
import os
import errno
import shutil
import tensorflow as tf
import numpy as np
import traceback
from tensorflow.python.tools import freeze_graph
from google.protobuf import text_format as pbtf

isApiV2 = tf.version.VERSION.startswith("2.")
if isApiV2:
    BASE_DIR = os.environ['DL4J_TEST_RESOURCES'] + '/src/main/resources/tf_graphs/examples' + tf.version.VERSION
else:
    BASE_DIR = os.environ['DL4J_TEST_RESOURCES'] + '/src/main/resources/tf_graphs/examples'

class TensorFlowPersistor:
    '''
    TensorFlowPersistor (TFP) is the main abstraction of this module. A TFP
    has all the functionality to load and store tensorflow tests.

    TFP is an abstract base class. You need to implement `get_input`
    and `get_input_shape` for the graph data of your choice.
    '''

    def __init__(self, save_dir, base_dir=None, verbose=True):
        self.save_dir = save_dir
        self.base_dir = BASE_DIR if base_dir is None else base_dir
        self.verbose = verbose
        self._sess = None
        self._placeholders = None
        self._output_tensors = None
        self._placeholder_name_value_dict = None
        self.skipBoolean = False
        if not os.path.exists("{}/{}".format(self.base_dir, self.save_dir)):
            #print("Creating dir: " + "{}/{}".format(self.base_dir, self.save_dir))
            os.makedirs("{}/{}".format(self.base_dir, self.save_dir))

    def set_placeholders(self, graph_placeholders):
        self._placeholders = graph_placeholders
        return self

    def set_output_tensors(self, graph_output_tensors):
        '''
        TODO: Document after we decide on a framework structure
        '''
        self._output_tensors = graph_output_tensors
        return self

    def set_test_data(self, input_dict):
        self._placeholder_name_value_dict = input_dict
        return self

    def set_training_sess(self, sess):
        self._sess = sess
        return self

    def set_skip_boolean(self, skip):
        self.skipBoolean = skip
        return self

    def set_verbose(self, verbose):
        self.verbose = verbose
        return self

    def _write_to_file(self, nparray, content_file, shape_file):
        os.makedirs(os.path.dirname(content_file), exist_ok=True)
        os.makedirs(os.path.dirname(shape_file), exist_ok=True)
        if np.isscalar(nparray):
            np.savetxt(shape_file, np.asarray([0]), fmt="%i")
            f = open(content_file, 'w')
            f.write('{}'.format(nparray))
            f.close()
        else:
            np.savetxt(shape_file, np.asarray(nparray.shape), fmt="%i")
            np.savetxt(content_file, np.ndarray.flatten(nparray), fmt="%10.8f")

    def _save_content(self, nparray, varname, name):
        varnameClean = varname.replace(":0","")
        content_file = "{}/{}/{}.{}.csv".format(self.base_dir, self.save_dir, varnameClean, name)
        shape_file = "{}/{}/{}.{}.shape".format(self.base_dir, self.save_dir, varnameClean, name)
        # print("Content file: ", content_file)
        # print("shape_file: ", shape_file)
        self._write_to_file(nparray, content_file, shape_file)

    def _save_input(self, nparray, varname, name='placeholder'):
        self._save_content(nparray, varname, name)

    def _save_node_dtypes(self, dtype_dict):
        print("Saving dtypes...")
        dtype_file = "{}/{}/dtypes".format(self.base_dir, self.save_dir)
        f = open(dtype_file,"a")
        for k,v in dtype_dict.items():
            print("{} {}".format(k,v),file=f)
        f.close() 

    def _save_intermediate(self, nparray, varname, name='prediction_inbw'):
        self._save_content(nparray, varname, name)

    def _save_predictions(self, output_dict, name='prediction'):
        for output_name, output_value in output_dict.items():
            self._save_content(output_value, output_name.replace('.0',''), name)

    def _save_graph(self, sess, all_saver, data_path="data-all", model_file="model.txt"):
        all_saver.save(sess, "{}/{}/{}".format(self.base_dir, self.save_dir, data_path),
                       global_step=1000)
        tf.io.write_graph(sess.graph_def, "{}/{}".format(self.base_dir, self.save_dir),
                             model_file, True)
        print("Saved: " +  self.base_dir + "/" + self.save_dir + "/" + model_file)

    def _save_graph_v2(self, sess, data_path="data-all-v2"):
        tf.saved_model.save(self, data_path)

    def _freeze_n_save_graph(self, output_node_names="output",
                             restore_op_name="save/restore_all",
                             filename_tensor_name="save/Const:0"):
        try:
            checkpoint = tf.train.get_checkpoint_state("{}/{}/".format(self.base_dir, self.save_dir))
            input_checkpoint = checkpoint.model_checkpoint_path
        except:
            raise ValueError("Could not read checkpoint state for path {}/{}"
                             .format(self.base_dir, self.save_dir))
        if self.verbose:
            print(input_checkpoint)
        output_graph = "{}/{}/frozen_model.pb".format(self.base_dir, self.save_dir)
        input_graph = "{}/{}/model.txt".format(self.base_dir, self.save_dir)
        if isApiV2 == False:
            freeze_graph.freeze_graph(input_graph=input_graph,
                                      input_saver="",
                                      input_checkpoint=input_checkpoint,
                                      output_graph=output_graph,
                                      input_binary=False,
                                      output_node_names=output_node_names,
                                      restore_op_name=restore_op_name,
                                      filename_tensor_name=filename_tensor_name,
                                      clear_devices=True,
                                      initializer_nodes="")

    def write_frozen_graph_txt(self, model_file='frozen_model.pb'):
        graph_filename = "{}/{}/{}".format(self.base_dir, self.save_dir, model_file)
        with tf.io.gfile.GFile(graph_filename, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.io.write_graph(graph_def, "{}/{}/".format(self.base_dir, self.save_dir),
                                 'frozen_graph.pbtxt', True)

    def write_frozen_graph_txt_v2(self, model_file='model.txt'):
        graph_filename = "{}/{}/{}".format(self.base_dir, self.save_dir, model_file)
        print("Opening " + graph_filename)
        with tf.io.gfile.GFile(graph_filename, "r") as f:
            graph_def = tf.compat.v1.GraphDef()
            str_graph = f.read()
            pbtf.Parse(str_graph, graph_def)
            tf.io.write_graph(graph_def, "{}/{}/".format(self.base_dir, self.save_dir),
                                 'frozen_graph.pbtxt', True)

    def load_frozen_graph(self, model_file='frozen_model.pb'):
        graph_filename = "{}/{}/{}".format(self.base_dir, self.save_dir, model_file)
        graph = tf.Graph()
        with tf.io.gfile.GFile(graph_filename, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            if isApiV2 == False:
                graph_def.ParseFromString(f.read())
            else:
                str_graph = f.read()
                pbtf.Parse(str_graph, graph_def)
        with graph.as_default():
            # tf.import_graph_def(graph_def, name=None)
            tf.import_graph_def(graph_def, name="")
        return graph

    def _save_intermediate_nodes(self, input_dict):
        graph = self.load_frozen_graph()
        self._save_intermediate_nodes2(input_dict, graph)

    def _save_intermediate_nodes2(self, input_dict, graph):
        placeholder_dict = {}
        prediction_dict = {}
        dtype_dict = {}
        if self.verbose:
            print("-----------------------------------------------------")
            print("PLACEHOLDER LIST:")
        for op in graph.get_operations():
            if op.type != "Placeholder":
                continue
            if("/" in op.name):
                placeholder_name = "/".join(op.name.split("/")[1:])
            else:
                placeholder_name = op.name
            print("op.name: ", op.name)  # there is a prefix and a suffix - there should only be one prefix
            print("placeholder name: ", placeholder_name)
            if(placeholder_name in input_dict):
                placeholder_dict[op.name + ":0"] = input_dict[placeholder_name]
            else:
                placeholder_dict[op.name + ":0"] = input_dict[placeholder_name + ":0"]

            tensor_dtype_string = "{}".format(op.outputs[0].dtype).split("'")[1]
            dtype_dict[placeholder_name] = tensor_dtype_string
        if self.verbose:
            print("-----------------------------------------------------")

        for op in graph.get_operations():
            if op.type == "Placeholder":
                continue
            if self.verbose:
                print(op.name)
                print(op.type)
            for op_output in op.outputs:
                if self.verbose:
                    print(op_output.name)
                with tf.compat.v1.Session(graph=graph) as sess:
                    try:
                        if op_output.dtype.is_bool and self.skipBoolean is True:
                            if self.verbose:
                                print("SKIPPING bool (use skipBoolean/set_skip_boolean(False) to change")
                                print("-----------------------------------------------------")
                        else:
                            op_prediction = sess.run(op_output, feed_dict=placeholder_dict)
                            #tensor_output_name = ("/".join(op_output.name.split("/")[1:]))
                            tensor_output_name = op_output.name
                            tensor_output_name = tensor_output_name.replace(":",".")
                            namelist = self._list_output_node_names()
                            print("NAMELIST: ", namelist)
                            if tensor_output_name in namelist:
                                prediction_dict[tensor_output_name] = op_prediction
                            if self.verbose:
                                print("PREDICTIONS: ", op_prediction)
                                print("-----------------------------------------------------")
                            modified_tensor_output_name = "____".join(tensor_output_name.split("/"))
                            self._save_intermediate(op_prediction, modified_tensor_output_name)
                            tensor_dtype_string = "{}".format(op_output.dtype).split("'")[1]
                            dtype_dict[modified_tensor_output_name] = tensor_dtype_string
                    except:
                        #print("Unexpected error:", sys.exc_info()[0])
                        print("Unexpected error:", sys.exc_info())
                        traceback.print_tb(sys.exc_info()[2])
                        if self.verbose:
                            print(op_output)
                            print("SKIPPING")
                            print("-----------------------------------------------------")
        self._save_node_dtypes(dtype_dict)
        return prediction_dict

    def load_external_graph(self, model_file):
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)

        return graph

    def _get_placeholder_dict(self):
        placeholder_feed_dict = {}
        for input_tensor in self._placeholders:
            input_name = input_tensor.name.split(":")[0]
            input_value = self._placeholder_name_value_dict[input_name]
            placeholder_feed_dict[input_tensor] = input_value
            self._save_input(input_value, input_name)
        return placeholder_feed_dict

    def _check_outputs(self):
        if self._output_tensors is None:
            raise ValueError("Ouput tensor list not set")

    def _check_inputs(self):
        if self._placeholders is None:
            raise ValueError("Input tensor placeholder list not set")

    def _clean_dir(self):
        working_dir = "{}/{}".format(self.base_dir, self.save_dir)
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)
        try:
            os.makedirs(working_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    def _list_output_node_names(self):
        output_node_names = []
        # should never be nested more than two levels
        if(self._output_tensors is None):
            raise ValueError("Output tensors have not been defined!")
        for a_output in self._output_tensors:
            if isinstance(a_output,list):
                print("IS LIST: ", a_output)
                for element_a_output in a_output:
                    if(hasattr(element_a_output, 'name')):
                        output_node_names.append(element_a_output.name.replace(":",'.'))
                    elif(isinstance(a_output, tf.RaggedTensor)):
                        output_node_names.append(a_output._values.name.replace(":",'.'))
                        output_node_names.append(a_output._row_splits.name.replace(":",'.'))
                    else:
                        output_node_names.append(element_a_output.replace(":",'.'))
            else:
                if(hasattr(a_output, 'name')):
                    output_node_names.append(a_output.name.replace(":",'.'))
                elif(isinstance(a_output, tf.RaggedTensor)):
                    output_node_names.append(a_output._values.name.replace(":",'.'))
                    output_node_names.append(a_output._row_splits.name.replace(":",'.'))
                else:
                    print(type(a_output))
                    #print(a_output.name)
                    print(vars(a_output))
                    output_node_names.append(a_output.replace(":",'.'))
        return output_node_names

    def _list_output_nodes_for_freeze_graph(self):
        output_node_names = set()
        # should never be nested more than two levels
        for a_output in self._output_tensors:
            if isinstance(a_output,list):
                for element_a_output in a_output:
                    output_node_names.add(element_a_output.name.split(":")[0])
            elif(isinstance(a_output, tf.RaggedTensor)):
                output_node_names.add(a_output._values.name.split(":")[0])
                output_node_names.add(a_output._row_splits.name.split(":")[0])
            else:
                output_node_names.add(a_output.name.split(":")[0])
        return output_node_names

    def build_save_frozen_graph(self, skip_intermediate=False):
        self._check_inputs()  # make sure input placeholders are set
        self._check_outputs()  # make sure outputs are set
        self._clean_dir()  # clean contents of dir
        placeholder_feed_dict = self._get_placeholder_dict()
        if self._sess is None:
            init = tf.compat.v1.global_variables_initializer()
            with tf.compat.v1.Session() as sess:
                sess.run(init)
                tf.compat.v1.disable_eager_execution()
                predictions = sess.run(self._output_tensors, feed_dict=placeholder_feed_dict)
                self._save_graph(sess, tf.compat.v1.train.Saver())
        else:
            all_saver = tf.compat.v1.train.Saver()
            predictions = self._sess.run(self._output_tensors, feed_dict=placeholder_feed_dict)
            self._save_graph(self._sess, all_saver, tf.compat.v1.train.Saver())
            self._sess.close
        flattened_predictions = []
        # print("OUTPUT TENSORS: ", self._output_tensors)
        # print("PREDICTIONS: ", predictions)
        for a_prediction in predictions:
            # print("Prediction: ", a_prediction)
            # print("Row split: ", a_prediction.row_splits)
            # print("Row split: ", type(a_prediction.row_splits))
            if isinstance(a_prediction,list):
                for element_a_prediction in a_prediction:
                    flattened_predictions.append(element_a_prediction)
            # elif isinstance(a_prediction,tf.RaggedTensor):
            elif hasattr(a_prediction, 'row_splits'):  #ragged tensor case
                print("### RAGGED")
                flattened_predictions.append(a_prediction.values)
                flattened_predictions.append(a_prediction.row_splits)
            else:
                flattened_predictions.append(a_prediction)
        if len(self._list_output_node_names()) == len(flattened_predictions):
            first_pass_dict = dict(zip(self._list_output_node_names(), flattened_predictions))
        else:
            # print("Names: ", self._list_output_node_names())
            # print("Number of predictions: ", len(flattened_predictions))
            raise RuntimeError('Error during export - different number of outputs as names detected')
        if self.verbose:
            print("FIRST PASS DICT:")
            print(first_pass_dict)
        self._save_predictions(first_pass_dict)

        # Determine and save input and output dtypes
        dtypesToSave = {}
        for inVar in placeholder_feed_dict:
            dtypesToSave[inVar.name] = placeholder_feed_dict[inVar].dtype
        for i in range(0, len(self._list_output_node_names())):
            outName = self._list_output_node_names()[i]
            # outVal = predictions[i]
            outVal = flattened_predictions[i]
            print("outName: ", outName)
            dtypesToSave[outName] = str(outVal.dtype)
        print("dtypesToSave: ", dtypesToSave)
        self._save_node_dtypes(dtypesToSave)
        tf.compat.v1.reset_default_graph()
        self._freeze_n_save_graph(output_node_names=",".join(self._list_output_nodes_for_freeze_graph()))
        self.write_frozen_graph_txt_v2()
        if not skip_intermediate and isApiV2 == False:
            second_pass_dict = self._save_intermediate_nodes(self._placeholder_name_value_dict)
            # assert second_pass_dict.keys() == first_pass_dict.keys()
            # for a_output in second_pass_dict.keys():
            #    np.testing.assert_equal(first_pass_dict[a_output], second_pass_dict[a_output])
        return predictions
