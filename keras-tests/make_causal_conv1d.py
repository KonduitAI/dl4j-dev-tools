'''
Creates a set of causal conv1d test cases for import testing


'''
from __future__ import print_function
from keras import Sequential, Model
from keras.layers import Conv1D, Activation, GlobalAveragePooling1D

import imp
import keras.backend as K
import keras
import numpy as np
from util import save_model_details, save_model_output

KERAS_VERSION = '_keras_2'
PREFIX = 'simple_sparse_xent_mlp' + KERAS_VERSION
OUT_DIR = 'output'

print('Entering Keras script')

kernels = [2, 3, 4]
strides = [1, 2, 3]
dilations = [1, 2]
formats = ["channels_last"]     #, "channels_first"]    # ValueError: When using causal padding in `Conv1D`, `data_format` must be "channels_last" (temporal data).

np.random.seed(12345)

for k in kernels:
    for s in strides:
        if s == 1:
            valid_dilations = dilations
        else:
            valid_dilations = [1]   #Keras docs: "Currently, specifying any dilation_rate value != 1 is incompatible with specifying any strides value != 1."
        for d in valid_dilations:
            for f in formats:
                if f == "channels_first":
                    input_shape = [3,16]
                    features = np.random.uniform(size=[2,3,16])
                    fShort = "cf"
                else:
                    input_shape = [16,3]
                    features = np.random.uniform(size=[2,16,3])
                    fShort = "cl"
                
                labels = np.array([[1,0,0], [0,1,0]])
                
                name = "causal_conv1d_k" + str(k) + "_s" + str(s) + "_d" + str(d) + "_" + fShort
                

                model = Sequential()
                model.add(Conv1D(name="conv0", filters=4, kernel_size=k, strides=s, padding="causal", data_format=f, dilation_rate=d, activation="tanh", use_bias=True, input_shape=input_shape))
                #model.add(Conv1D(name="conv1", filters=3, kernel_size=k, strides=s, padding="causal", data_format=f, dilation_rate=d, activation=None, use_bias=False))
                #model.add(GlobalAveragePooling1D(data_format=f))
                #model.add(Activation('softmax'))

                opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
                model.compile(loss='categorical_crossentropy',
                              optimizer=opt,
                              metrics=['accuracy'])

                print('Saving model details')
                
                
                save_model_details(model, prefix=name, out_dir=OUT_DIR)

                exp_out = model.predict(features)
                
                print("Input = ", str(features.shape), ", Out = ", str(exp_out.shape), " case - ", name)

                print('Saving model outputs')
                save_model_output(model, features, exp_out, nb_examples=None, prefix=name, out_dir=OUT_DIR, labels=None)    #labels)

                print('DONE!')
