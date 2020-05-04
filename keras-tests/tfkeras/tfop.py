import tensorflow as tf
import numpy as np
from utils import grid, save_model
from utils import tqdm


def get_tfop_models():
    models = []

    inp = tf.keras.layers.Input((2, 3))
    reshaped = tf.reshape(inp, (-1, 6))
    out = tf.keras.layers.Dense(5)(reshaped)
    models.append(tf.keras.models.Model(inp, out))

    inp = tf.keras.layers.Input((3, 2))
    perm = tf.transpose(inp, (0, 2, 1))
    out = tf.keras.layers.Dense(5)(perm)
    models.append(tf.keras.models.Model(inp, out))

    return models

def run():
    gen = get_tfop_models()
    for i, model in tqdm(enumerate(gen), total=len(gen)):
        save_model(model, 'tfop_' + str(i) + '.h5')

if __name__ == '__main__':
    run()
