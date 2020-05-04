'''
Create a simple imdb fasttext net for testing Keras model import. Run
Keras imdb_lstm.py example and then save that model and its
outputs to disk.
'''
from __future__ import print_function

import imp
import keras.backend as K
from util import save_model_details, save_model_output

SCRIPT_PATH = '../examples/imdb_fasttext.py'
KERAS_VERSION = '_keras_2'
PREFIX = 'imdb_fasttext_' + K.image_dim_ordering() + KERAS_VERSION
OUT_DIR = '.'

print('Entering Keras script')
example = imp.load_source('example', SCRIPT_PATH)

print('Saving model details')
save_model_details(example.model, prefix=PREFIX, out_dir=OUT_DIR)

print('Saving model outputs')
save_model_output(example.model, example.x_test, example.y_test, nb_examples=100, prefix=PREFIX, out_dir=OUT_DIR)

print('DONE!')