import tensorflow as tf
import numpy as np
from utils import grid, save_model
import gc
from utils import tqdm


input_shapes_and_axis = [[(2,), (2,), -1], [(None, 2), (None, 3), -1],
[(None, 2), (3, 4), -1], [(3, 2), (4, 2), -2], [(3, 2), (4, 2), -1],
[(2, 3), (3, 4), (1, 0)]]
merger = [
    tf.keras.layers.Dot,
    tf.keras.layers.Concatenate,
    tf.keras.layers.Add,
    tf.keras.layers.Multiply,
    tf.keras.layers.Average,
    tf.keras.layers.Subtract,
    tf.keras.layers.Maximum,
    tf.keras.layers.Minimum
]

@grid(**globals())
def generate_merge_models(input_shapes_and_axis, merger):
    input_shapes = input_shapes_and_axis[:-1]
    axis = input_shapes_and_axis[-1]
    inputs = list(map(tf.keras.layers.Input, input_shapes))
    if merger == tf.keras.layers.Concatenate:
        merge_args = {'axis' : axis}
    elif merger == tf.keras.layers.Dot:
        merge_args = {'axes': axis}
    else:
        merge_args = {}
    try:
        merged = merger(**merge_args)(inputs)
    except:
        return
    model = tf.keras.models.Model(inputs, merged)
    return model


def run():
    gen = generate_merge_models()
    for i, model in tqdm(enumerate(gen), total=len(gen)):
        if model:
            save_model(model, 'merge_' + str(i) + '.h5')

if __name__ == '__main__':
    run()
