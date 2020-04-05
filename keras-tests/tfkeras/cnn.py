import tensorflow as tf
import numpy as np
from utils import grid, save_model
import gc
from utils import tqdm

np.random.seed(1337)

max_num_layers = 3
kernel_sizes = list(np.random.randint(5, 10, (max_num_layers,)))

input_shape = [(3, 16), (1, 16)]#, (None, 16), (3, None), (1, None)]
data_format = ['channels_first', 'channels_last']
num_layers = [i + 1 for i in range(max_num_layers)]
filters = [4]
strides = [1, 2, 3]
dilation_rate = [1, 2]
activation = ['tanh']
use_bias = [True, False]
pooling = [tf.keras.layers.MaxPooling1D, tf.keras.layers.AveragePooling1D]
padding = ['causal', 'same', 'valid']
compile_args = [{'optimizer': 'rmsprop', 'loss': 'categorical_crossentropy', 'metrics': ['accuracy']}]
@grid(**globals())
def generate_cnn1ds(input_shape,
                   data_format,
                   num_layers,
                   filters,
                   strides,
                   padding,
                   dilation_rate,
                   activation,
                   use_bias,
                   pooling,
                   compile_args):
    if strides != 1 and dilation_rate != 1:
        return
    if data_format == 'channels_last':
        input_shape = (input_shape[1], input_shape[0])
    inp = tf.keras.layers.Input(input_shape)
    x = inp
    for i in range(num_layers):
        try:
            x = tf.keras.layers.Conv1D(filters=filters,
                                    kernel_size=(kernel_sizes[i],),
                                    strides=strides,
                                    dilation_rate=dilation_rate,
                                    padding=padding,
                                    data_format=data_format,
                                    use_bias=use_bias,
                                    activation=activation)(x)
        except ValueError:
            return None # not all stride / input shape comnbinations are valid
        if pooling:
            try:
                x = pooling()(x)
            except:
                pass
    model = tf.keras.models.Model(inp, x)
    if compile_args:
        model.compile(**compile_args)
    return model


def run():
    gen = generate_cnn1ds()
    for i, model in tqdm(enumerate(gen), total=len(gen)):
        if model:
            save_model(model, 'cnn1d_' + str(i) + '.h5')
            del model
            gc.collect()

if __name__ == '__main__':
    run()

    
