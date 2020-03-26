'''
Creates a set of causal conv1d test cases for import testing


'''
from __future__ import print_function
from keras import Sequential, Model
from keras.layers import LeakyReLU, ELU, ThresholdedReLU, Softmax, ReLU, Dense

import imp
import keras.backend as K
import keras
import numpy as np
from util import save_model_details, save_model_output

KERAS_VERSION = '_keras_2'
OUT_DIR = 'output'

print('Entering Keras script')

kernels = [2, 3, 4]
strides = [1, 2, 3]
dilations = [1, 2]
formats = ["channels_last", "channels_first"]
padding = ["valid", "same"]

np.random.seed(12345)

for type in range(5):

    num_tests = 1
    
    if type is 4:
        num_tests = 4

    for i in range(num_tests):
        keras.backend.clear_session()
    
        features = np.random.uniform(size=[2,16])
        model = Sequential()
        model.add(Dense(name="dense", units=32, input_shape=(16,)))

        if type is 0:
            model.add(LeakyReLU(alpha=0.5))
            layername = "LeakyReLU"
        elif type is 1:
            model.add(ELU(alpha=0.7))
            layername = "ELU"
        elif type is 2:
            model.add(ThresholdedReLU(theta=0.7))
            layername = "ThresholdReLU"
        elif type is 3:
            model.add(Softmax())
            layername = "Softmax"
        elif type is 4:
            if i is 0:
                model.add(ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
            elif i is 1:
                model.add(ReLU(max_value=6.0, negative_slope=0.0, threshold=0.0))
            elif i is 2:
                model.add(ReLU(max_value=1.4, negative_slope=0.2, threshold=0.2))
            elif i is 3:
                model.add(ReLU(max_value=None, negative_slope=0.5, threshold=0.0))
            else:
                raise ValueError("Invalid test num")
            layername = "ReLU"
        else:
            raise ValueError("Unknown type")
        
        labels = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
        
        name = layername + "_" + str(i)


        opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
        model.compile(loss='mean_squared_error',
                      optimizer=opt,
                      metrics=['mae'])

        print('Saving model details')
        
        
        save_model_details(model, prefix=name, out_dir=OUT_DIR)

        exp_out = model.predict(features)
        
        print("Input = ", str(features.shape), ", Out = ", str(exp_out.shape), " case - ", name)

        print('Saving model outputs')
        save_model_output(model, features, exp_out, nb_examples=None, prefix=name, out_dir=OUT_DIR, labels=None, doEval=False)

print('DONE!')
