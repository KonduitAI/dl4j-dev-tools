import tensorflow as tf
import numpy as np
from utils import grid, save_model
import gc
from utils import tqdm


input_shapes = [[(2,), (2,)], [(None, 2), (None, 2)], [(None, 2, 3), (None, 2, 3)]]

merger = [
    tf.keras.layers.Add,
    tf.keras.layers.Multiply,
    tf.keras.layers.Average,
    tf.keras.layers.Subtract,
    tf.keras.layers.Maximum,
    tf.keras.layers.Minimum
]


@grid(**globals())
def generate_merge_models(input_shapes, merger):
    inputs = list(map(tf.keras.layers.Input, input_shapes))
    merged = merger()(inputs)
    model = tf.keras.models.Model(inputs, merged)

def dot_models():
    models = []
    args = [
        [(2, 3), (2, 2), (1, 1)],
        [(2, 3), (3, 3), (2, 1)]
    ]
    for arg in args:
        shapes = arg[:2]
        inputs = list(map(tf.keras.layers.Input, shapes))
        out = tf.keras.layers.Dot(axes=arg[-1])(inputs)
        models.append(tf.keras.models.Model(inputs, out))
    return models

def concat_models():
    models = []
    args = [
        [(2, 3), (4, 3), 1],
        [(2, 3), (2, 4), 2]
    ]
    for arg in args:
        shapes = arg[:2]
        inputs = list(map(tf.keras.layers.Input, shapes))
        out = tf.keras.layers.Concatenate(axis=arg[-1])(inputs)
        models.append(tf.keras.models.Model(inputs, out))
    return models
    

def run():
    gen = generate_merge_models()
    for i, model in tqdm(enumerate(gen), total=len(gen)):
        if model:
            save_model(model, 'merge_' + str(i) + '.h5')
    for i, model in tqdm(enumerate(dot_models())):
        if model:
            save_model(model, 'merge_dot_' + str(i) + '.h5')
    for i, model in tqdm(enumerate(concat_models())):
        if model:
            save_model(model, 'merge_concat_' + str(i) + '.h5')

if __name__ == '__main__':
    run()
