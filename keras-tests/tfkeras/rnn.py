import tensorflow as tf
import numpy as np
from utils import grid, save_model
from utils import tqdm

np.random.seed(1337)

max_num_layers = 2
layer_output_sizes = list(np.random.randint(3, 7, (max_num_layers,)))


#grid
input_shape = [(3, 4), (1, 5), (4, 1), (None, 3), (None, 1)]
num_layers = [i + 1 for i in range(max_num_layers)]
activation_type = ['layer', 'arg', 'recurrent']
activation = ['relu']#, 'tanh']
rnn_type = [tf.keras.layers.LSTM, tf.keras.layers.GRU, tf.keras.layers.SimpleRNN]
return_sequences = [True, False]
dense_after = [False]#[True, False]
compile_args = [None]#, {'loss': 'mse', 'optimizer': 'sgd'}]
bidirectional = [None, 'concat']#, 'sum', 'mul']

@grid(**globals())
def generate_rnns(input_shape,
                  num_layers,
                  activation_type,
                  activation,
                  rnn_type,
                  return_sequences,
                  dense_after,
                  bidirectional,
                  compile_args):
    if rnn_type == tf.keras.layers.SimpleRNN and activation_type == 'recurrent':
        return
    inp = tf.keras.layers.Input(input_shape)
    x = inp
    for i in range(num_layers):
        if bidirectional:
            f = lambda *args, **kwargs: tf.keras.layers.Bidirectional(rnn_type(*args, **kwargs), merge_mode=bidirectional)
        else:
            f = rnn_type
        if activation_type == 'arg':
            x = f(layer_output_sizes[i], return_sequences=return_sequences, activation=activation)(x)
        elif activation_type == 'recurrent':
            x = f(layer_output_sizes[i], return_sequences=return_sequences, recurrent_activation=activation)(x)
        else:
            x = f(layer_output_sizes[i], return_sequences=return_sequences)(x)
            x = tf.keras.layers.Activation(activation)(x)
        if dense_after:
            x = tf.keras.layers.Dense(int(layer_output_sizes[i] / 2) + 1)(x)   
        if not return_sequences and i != num_layers - 1:
            x = tf.keras.layers.RepeatVector(5)(x) 
    model = tf.keras.models.Model(inp, x)
    if compile_args:
        model.compile(**compile_args)
    return model

def run():
    gen = generate_rnns()
    for i, model in tqdm(enumerate(gen), total=len(gen)):
        if model:
            save_model(model, 'rnn_' + str(i) + '.h5')

if __name__ == '__main__':
    run()
