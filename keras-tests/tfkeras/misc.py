import tensorflow as tf
import numpy as np
from utils import grid, save_model
from utils import tqdm


def get_misc_models():
    models = []
    inp = tf.keras.layers.Input((2, 3))
    x = tf.keras.layers.Dense(4)(inp)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1)(x)
    models.append(tf.keras.models.Model(inp, x))

    inp = tf.keras.layers.Input((5,))
    x = tf.keras.layers.Embedding(10, 8)(inp)
    x = tf.keras.layers.Dense(4)(x)
    models.append(tf.keras.models.Model(inp, x))

    inp = tf.keras.layers.Input((5,))
    x = tf.keras.layers.Embedding(10, 8, mask_zero=True)(inp)
    x = tf.keras.layers.Masking()(x)
    x = tf.keras.layers.Dense(4)(x)
    models.append(tf.keras.models.Model(inp, x))

    inp = tf.keras.layers.Input((2, 3))
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4))(inp)
    models.append(tf.keras.models.Model(inp, x))

    inp = tf.keras.layers.Input((2, 3))
    x = tf.keras.layers.Reshape((3, 2))(inp)
    models.append(tf.keras.models.Model(inp, x))

    inp = tf.keras.layers.Input((2, 3))
    x = tf.keras.layers.Permute((2, 1))(inp)
    models.append(tf.keras.models.Model(inp, x))

    return models

def run():
    gen = get_misc_models()
    for i, model in tqdm(enumerate(gen), total=len(gen)):
        save_model(model, 'misc_' + str(i) + '.h5')

if __name__ == '__main__':
    run()
