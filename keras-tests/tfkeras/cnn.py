import tensorflow as tf
import numpy as np
from utils import grid, save_model
import gc
from utils import tqdm
import warnings
from tensorflow.python.framework.ops import disable_eager_execution
import multiprocessing


disable_eager_execution()

np.random.seed(1337)

max_num_layers = 2

#1D
kernel_sizes = list(np.random.randint(2, 4, (max_num_layers,)))
input_shape = [(20, 32)]#, (None, 16), (3, None), (1, None)]
data_format = ['channels_first', 'channels_last']
layer_type = [tf.keras.layers.Conv1D, tf.keras.layers.SeparableConv1D, tf.keras.layers.LocallyConnected1D]
num_layers = [2]#[i + 1 for i in range(max_num_layers)]
filters = [4]
strides = [1, 2, 3]
dilation_rate = [1, 2]
activation = ['tanh']
use_bias = [True]#, False]
pooling = [tf.keras.layers.MaxPooling1D, tf.keras.layers.AveragePooling1D]
global_pooling = [tf.keras.layers.GlobalMaxPool1D, tf.keras.layers.GlobalAvgPool1D]
padding = ['causal', 'same', 'valid']
zero_padding = [3]#, None]
upsampling = [2]#, None]
spatial_dropout = [True]#, False]
cropping = [(1, 1)]#, None]
compile_args = [{'optimizer': 'rmsprop', 'loss': 'categorical_crossentropy', 'metrics': ['accuracy']}]
@grid(**globals())
def generate_cnn1ds(input_shape,
                   data_format,
                   layer_type,
                   num_layers,
                   filters,
                   strides,
                   padding,
                   zero_padding,
                   dilation_rate,
                   activation,
                   use_bias,
                   pooling,
                   global_pooling,
                   spatial_dropout,
                   upsampling,
                   cropping,
                   compile_args):
    if strides != 1 and dilation_rate != 1:
        return
    if data_format == 'channels_last':
        input_shape = input_shape[1:] + (input_shape[0],)
    inp = tf.keras.layers.Input(input_shape)
    x = inp
    for i in range(num_layers):
        try:
            args = dict(filters=filters,
                        kernel_size=(kernel_sizes[i],),
                        strides=strides,
                        padding=padding,
                        data_format=data_format,
                        use_bias=use_bias,
                        activation=activation)
            if layer_type == tf.keras.layers.Conv1D:
                args['dilation_rate'] = dilation_rate
            elif dilation_rate != 1:
                return
            if layer_type == tf.keras.layers.LocallyConnected1D and \
                padding != 'valid':
                return
            x = layer_type(**args)(x)
            if spatial_dropout:
                x = tf.keras.layers.SpatialDropout1D(0.2)(x)
            if zero_padding:
                x = tf.keras.layers.ZeroPadding1D(zero_padding)(x)
            if upsampling:
                x = tf.keras.layers.UpSampling1D(upsampling)(x)
            if cropping:
                x = tf.keras.layers.Cropping1D(cropping)(x)
        except ValueError as e:
            warnings.warn(str(e))
            return None # not all stride / input shape comnbinations are valid
        if pooling:
            try:
                x = pooling()(x)
            except:
                pass
    if global_pooling:
        x = global_pooling()(x)
    model = tf.keras.models.Model(inp, x)
    if compile_args:
        model.compile(**compile_args)
    return model


#2D
kernel_sizes = list(np.random.randint(2, 4, (max_num_layers,)))
input_shape = [(3, 10, 10)]#, (None, 16), (3, None), (1, None)]
data_format = ['channels_first', 'channels_last']
layer_type = [tf.keras.layers.Conv2D,
              tf.keras.layers.SeparableConv2D,
              tf.keras.layers.LocallyConnected2D,
              tf.keras.layers.DepthwiseConv2D,
              tf.keras.layers.Conv2DTranspose]
num_layers = [2]#[i + 1 for i in range(max_num_layers)]
filters = [4]
strides = [1]#, 2]
dilation_rate = [1]#, 2]
activation = ['tanh']
use_bias = [True]#, False]
pooling = [tf.keras.layers.MaxPooling2D, tf.keras.layers.AveragePooling2D]
global_pooling = [tf.keras.layers.GlobalMaxPool2D, tf.keras.layers.GlobalAvgPool2D]
padding = ['causal', 'same', 'valid']
zero_padding = [3]#, None]
upsampling = [2]#, None]
spatial_dropout = [True]#, False]
cropping = [(1, 1)]#, None]
compile_args = [{'optimizer': 'rmsprop', 'loss': 'categorical_crossentropy', 'metrics': ['accuracy']}]
@grid(**globals())
def generate_cnn2ds(input_shape,
                   data_format,
                   layer_type,
                   num_layers,
                   filters,
                   strides,
                   padding,
                   zero_padding,
                   dilation_rate,
                   activation,
                   use_bias,
                   pooling,
                   global_pooling,
                   spatial_dropout,
                   upsampling,
                   cropping,
                   compile_args):
    if strides != 1 and dilation_rate != 1:
        return
    if data_format == 'channels_last':
        input_shape = input_shape[1:] + (input_shape[0],)
    inp = tf.keras.layers.Input(input_shape)
    x = inp
    for i in range(num_layers):
        try:
            args = dict(
                        kernel_size=(kernel_sizes[i], kernel_sizes[i]),
                        strides=strides,
                        padding=padding,
                        data_format=data_format,
                        use_bias=use_bias,
                        activation=activation)
            if layer_type == tf.keras.layers.Conv2D:
                args['dilation_rate'] = dilation_rate
            elif dilation_rate != 1:
                return
            if layer_type != tf.keras.layers.DepthwiseConv2D:
                args['filters'] = filters
            if layer_type == tf.keras.layers.LocallyConnected2D and \
                padding != 'valid':
                return
            x = layer_type(**args)(x)
            if spatial_dropout:
                x = tf.keras.layers.SpatialDropout2D(0.2)(x)
            if zero_padding:
                x = tf.keras.layers.ZeroPadding2D(zero_padding)(x)
            if upsampling:
                x = tf.keras.layers.UpSampling2D(upsampling)(x)
            if cropping:
                x = tf.keras.layers.Cropping2D(cropping)(x)
        except ValueError as e:
            warnings.warn(str(e))
            return None # not all stride / input shape comnbinations are valid
        if pooling:
            try:
                x = pooling()(x)
            except:
                pass
    if global_pooling:
        x = global_pooling()(x)
    model = tf.keras.models.Model(inp, x)
    if compile_args:
        model.compile(**compile_args)
    return model

#3D
kernel_sizes = list(np.random.randint(2, 4, (max_num_layers,)))
input_shape = [(3, 10, 10, 10)]#, (None, 16), (3, None), (1, None)]
data_format = ['channels_first', 'channels_last']
layer_type = [tf.keras.layers.Conv3D, tf.keras.layers.Conv3DTranspose]
num_layers = [2]#[i + 1 for i in range(max_num_layers)]
filters = [4]
strides = [1]#, 2]
dilation_rate = [1]#, 2]
activation = ['tanh']
use_bias = [True]#, False]
pooling = [tf.keras.layers.MaxPooling3D, tf.keras.layers.AveragePooling3D]
global_pooling = [tf.keras.layers.GlobalMaxPool3D, tf.keras.layers.GlobalAvgPool3D]
padding = ['causal', 'same', 'valid']
zero_padding = [3]#, None]
upsampling = [2]#, None]
spatial_dropout = [True]#, False]
cropping = [(1, 1, 1)]#, None]
compile_args = [{'optimizer': 'rmsprop', 'loss': 'categorical_crossentropy', 'metrics': ['accuracy']}]
@grid(**globals())
def generate_cnn3ds(input_shape,
                   data_format,
                   layer_type,
                   num_layers,
                   filters,
                   strides,
                   padding,
                   zero_padding,
                   dilation_rate,
                   activation,
                   use_bias,
                   pooling,
                   global_pooling,
                   spatial_dropout,
                   upsampling,
                   cropping,
                   compile_args):
    if strides != 1 and dilation_rate != 1:
        return
    if data_format == 'channels_last':
        input_shape = input_shape[1:] + (input_shape[0],)
    inp = tf.keras.layers.Input(input_shape)
    x = inp
    for i in range(num_layers):
        try:
            args = dict(filters=filters,
                        kernel_size=(kernel_sizes[i],) * 3,
                        strides=strides,
                        padding=padding,
                        data_format=data_format,
                        use_bias=use_bias,
                        activation=activation)
            if layer_type == tf.keras.layers.Conv3D:
                args['dilation_rate'] = dilation_rate
            elif dilation_rate != 1:
                return
            x = layer_type(**args)(x)
            if spatial_dropout:
                x = tf.keras.layers.SpatialDropout3D(0.2)(x)
            if zero_padding:
                x = tf.keras.layers.ZeroPadding3D(zero_padding)(x)
            if upsampling:
                x = tf.keras.layers.UpSampling3D(upsampling)(x)
            if cropping:
                x = tf.keras.layers.Cropping3D(cropping)(x)
        except ValueError as e:
            warnings.warn(str(e))
            return None # not all stride / input shape comnbinations are valid
        if pooling:
            try:
                x = pooling()(x)
            except:
                pass
    if global_pooling:
        x = global_pooling()(x)
    model = tf.keras.models.Model(inp, x)
    if compile_args:
        model.compile(**compile_args)
    return model



def _start_proc(f):
    p = multiprocessing.Process(target=f)
    p.start()
    p.join()

def _run_1d():
    gen = generate_cnn1ds()
    for i, model in tqdm(enumerate(gen), total=len(gen)):
        if model:
            save_model(model, 'cnn1d_' + str(i) + '.h5')
            del model
            gc.collect()

def _run_2d():
    gen = generate_cnn2ds()
    for i, model in tqdm(enumerate(gen), total=len(gen)):
        if model:
            save_model(model, 'cnn2d_' + str(i) + '.h5')
            del model
            gc.collect()

def _run_3d():
    gen = generate_cnn3ds()
    for i, model in tqdm(enumerate(gen), total=len(gen)):
        if model:
            save_model(model, 'cnn3d_' + str(i) + '.h5')
            del model
            gc.collect()

def run():
    _start_proc(_run_1d)
    _start_proc(_run_2d)
    _start_proc(_run_3d)

if __name__ == '__main__':
    run()

    
