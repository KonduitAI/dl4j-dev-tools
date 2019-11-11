import os
import argparse
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import numpy as np

from tfoptests.persistor import TensorFlowPersistor

# Enforce TF backend regardless of other configurations
os.environ["KERAS_BACKEND"]="tensorflow"


def parse_shape(string_shape):
    return [int(x) for x in string_shape.split(',')]


def get_save_dir(model_file):
    file_signature = model_file.split('/')[-1]
    return file_signature.split('.')[0]


def print_nodes():
    # Helper to find input and output nodes
    nodes = [n for n in tf.get_default_graph().as_graph_def().node]
    for node in nodes :
        if last.name in node.name:
            print(node.name)


if __name__ == '__main__':
    # python3 keras_sequential_conversion.py \ 
    # --file keras-mnist-mlp.h5 \
    # --shape "1,784"

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, required=True)
    parser.add_argument('--shape', '-s', type=str, required=True)
    parser.add_argument('--base_dir', '-b', type=str, required=False) 
    parser.add_argument('--verbose', '-v', type=bool, required=False)
    parser.add_argument('--in_name', '-i', nargs='?', type=str, default='_input')
    parser.add_argument('--out_name', '-o', nargs='?', type=str, default='/Softmax')
    args = parser.parse_args()

    model_file = args.file
    base_dir = args.base_dir
    str_shape = args.shape
    in_arg = args.in_name
    out_arg = args.out_name

    shape = parse_shape(str_shape)
    save_dir = get_save_dir(model_file)

    model = load_model(model_file)
    layers = model.layers
    first = layers[0]
    last = layers[-1]

    graph = K.get_session().graph

    if args.verbose:
        print_nodes()

    in_name = '{}{}'.format(first.name, in_arg)
    in_node = graph.get_tensor_by_name(in_name + ':0')
    out_name = '{}{}'.format(last.name, out_arg)
    out_node = graph.get_tensor_by_name(out_name + ':0')

    with tf.Session(graph=graph) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        inValue = np.random.rand(*shape)
        out = sess.run(out_node, feed_dict={
            in_node: inValue
        })

        tfp = TensorFlowPersistor(base_dir=base_dir, save_dir=save_dir)
        tfp._save_input(inValue, in_name)
        tfp._save_predictions({out_name: out})