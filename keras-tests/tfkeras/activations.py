import tensorflow as tf
import numpy as np
from utils import grid, save_model
import gc
from utils import tqdm

np.random.seed(1337)


activation = [act for act in dir(tf.keras.activations)
               if not act.startswith('_') and act not in ('serialize', 'deserialize', 'get')]

activation.remove('exponential') #not supported

use_dense = [True, False]

@grid(**globals())
def generate_models(activation, use_dense):
    inp = tf.keras.layers.Input((2,))
    if use_dense:
        out = tf.keras.layers.Dense(4, activation=activation)(inp)
    else:
        out = tf.keras.layers.Activation(activation)(inp)
    model = tf.keras.models.Model(inp, out)
    return model


def run():
    gen = generate_models()
    for i, model in tqdm(enumerate(gen), total=len(gen)):
        save_model(model, 'act_' + str(i) + '.h5')

if __name__ == '__main__':
    run()
