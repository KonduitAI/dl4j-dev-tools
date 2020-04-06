import tensorflow as tf
import numpy as np
from utils import grid, save_model
import gc
from utils import tqdm
import warnings


np.random.seed(1337)

max_num_layers = 2

kernel_sizes = list(np.random.randint(2, 4, (max_num_layers,)))
input_shape = [(3, 4, 10, 10)]#...
data_format = ['channels_first', 'channels_last']
num_layers = [2]#[i + 1 for i in range(max_num_layers)]
filters = [4]
strides = [1]#, 2]
dilation_rate = [1]#, 2]
activation = ['tanh']
padding = ['same', 'valid']
use_bias = [True]#, False]
return_sequences = [True, False]
go_backwards = [True, False]
bidirectional = [True, False]
@grid(**globals())
def generate_convlstm_models(input_shape,
                             data_format,
                             num_layers,
                             filters,
                             strides,
                             dilation_rate,
                             activation,
                             padding,
                             use_bias,
                             return_sequences,
                             go_backwards,
                             bidirectional):
    if not return_sequences and num_layers > 1:
        return
    if data_format == 'channels_last':
        input_shape = input_shape[1:] + (input_shape[0],)
    inp = tf.keras.layers.Input(input_shape)
    x = inp
    for i in range(num_layers):
        layer = tf.keras.layers.ConvLSTM2D(filters=filters,
                                           kernel_size=(kernel_sizes[i],) * 2,
                                           strides=strides,
                                           dilation_rate=dilation_rate,
                                           padding=padding,
                                           data_format=data_format,
                                           return_sequences=return_sequences,
                                           go_backwards=go_backwards
                                           )
        if bidirectional:
            layer = tf.keras.layers.Bidirectional(layer)
        x = layer(x)
    return tf.keras.models.Model(inp, x)

def run():
    gen = generate_convlstm_models()
    for i, model in tqdm(enumerate(gen), total=len(gen)):
        if model:
            save_model(model, 'convlstm2d_' + str(i) + '.h5')

if __name__ == '__main__':
    run()


