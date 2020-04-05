import tensorflow as tf
import numpy as np
from utils import grid, save_model
from utils import tqdm

np.random.seed(1337)

max_num_layers = 5
layer_output_sizes = list(np.random.randint(3, 7, (max_num_layers,)))

# grid
input_shape = [(3,), (4,), (2, 3), (4, 5)]
num_layers = [i + 1 for i in range(max_num_layers)]
activation_type = ['layer', 'arg']
activation = ['sigmoid', 'relu', 'tanh']
compile_args = [None, {'loss': 'mse', 'optimizer': 'sgd'}]

@grid(**globals())
def generate_mlps(input_shape, num_layers, activation_type, activation, compile_args):
    inp = tf.keras.layers.Input(input_shape)
    x = inp
    for i in range(num_layers):
        if activation_type == 'arg':
            x = tf.keras.layers.Dense(layer_output_sizes[0], activation=activation)(x)
        else:
            x = tf.keras.layers.Dense(layer_output_sizes[0])(x)
            x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    model = tf.keras.models.Model(inp, x)
    if compile_args:
        model.compile(**compile_args)
    return model


def run():
    gen = generate_mlps()
    for i, model in tqdm(enumerate(gen), total=len(gen)):
        save_model(model, 'mlp_' + str(i) + '.h5')

if __name__ == '__main__':
    run()
