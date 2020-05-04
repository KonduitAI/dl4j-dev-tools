import tensorflow as tf
import numpy as np
from utils import grid, save_model
import gc
from utils import tqdm
import inspect


layer = []
for x in dir(tf.keras.layers):
    l = getattr(tf.keras.layers, x)
    if inspect.isclass(l) and \
    issubclass(l, tf.keras.layers.Layer) and \
    '.layers.advanced_activations' in l.__module__:
        layer.append(l)

@grid(**globals())
def generate_models(layer):
    inp = tf.keras.layers.Input((2, 3))
    out = layer()(inp)
    return tf.keras.models.Model(inp, out)

def run():
    gen = generate_models()
    for i, model in tqdm(enumerate(gen), total=len(gen)):
        save_model(model, 'adv_act_' + str(i) + '.h5')

if __name__ == '__main__':
    run()
