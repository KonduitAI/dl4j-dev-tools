import tensorflow as tf
import mlp
import rnn
import utils
import os


mlp.run()
rnn.run()



# calculate layer coverage TODO: op coverage

all_layers = [l for l in dir(tf.keras.layers) if l[0].isupper()]

def get_layers(model):
    if hasattr(model, 'layers'):
        layers = []
        for layer in model.layers:
            layers += get_layers(layer)
        return layers
    else:
        return [model.__class__.__name__]

used_layers = set()
for r, _, fs in os.walk(utils.tfkeras_dir):
    for f in fs:
        if f.lower().endswith('.h5'):
            model = tf.keras.models.load_model(os.path.join(r, f))
            used_layers.update(get_layers(model))


unused_layers = [l for l in all_layers if l not in used_layers]
coverage = len(used_layers) / len(all_layers)

print('Layers not covered:')
for l in unused_layers:
    print(l)

print('Layer coverage: ' + str(int(coverage * 100)) + '%')
