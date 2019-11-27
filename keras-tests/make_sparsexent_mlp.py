'''
Create a simple lstm net for testing Keras model import. Run
Keras simple_lstm.py example and then save that model and its
outputs to disk.
'''
from __future__ import print_function
from keras import Sequential, Model
from keras.layers import Dense, Activation

import imp
import keras.backend as K
import keras
import numpy as np
from util import save_model_details, save_model_output

KERAS_VERSION = '_keras_2'
PREFIX = 'simple_sparse_xent_mlp' + KERAS_VERSION
OUT_DIR = 'output'

print('Entering Keras script')
features = np.random.uniform(size=[3,4])
labels = np.array([0,1,2])

model = Sequential()
model.add(Dense(3, input_shape=[4]))
model.add(Activation('softmax'))

opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

print('Saving model details')
save_model_details(model, prefix=PREFIX, out_dir=OUT_DIR)

exp_out = model.predict(features)

print('Saving model outputs')
save_model_output(model, features, exp_out, nb_examples=None, prefix=PREFIX, out_dir=OUT_DIR, labels=labels)

print('DONE!')
