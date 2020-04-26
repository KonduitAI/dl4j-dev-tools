import os
import itertools
import inspect
import sys
import tensorflow as tf
import json
import os
import h5py
import numpy as np
import click
try:
    from tqdm import tqdm
except ImportError:
    import warnings
    tqdm = lambda _: [_,warnings.warn("Install tqdm for fancy progress bars!")][0]


dl4j_test_resources = os.environ.get('DL4J_TEST_RESOURCES')


if dl4j_test_resources is None:
    # Try and auto detect dl4j_test_resources.
    # We are assuming that dl4j-test-resources and dl4j-dev-tools
    # are cloned in parallel (i.e they are in the same parent directory)
    # NOTE: code should be adapted in case this file is refactored
    dirname = os.path.dirname
    dl4j_test_resources = os.path.join(
        dirname(dirname(dirname(dirname(os.path.abspath(__file__))))),
        'dl4j-test-resources')
    if os.path.isdir(dl4j_test_resources):
        print('Aoto detected dl4j-test-resources: ' + dl4j_test_resources)
    else:
        raise Exception('Environment variable not set: DL4J_TEST_RESOURCES')

tfkeras_dir = os.path.join(dl4j_test_resources,
                           'src', 'main', 'resources',
                           'modelimport', 'keras', 'tfkeras')



def _get_layers(model):
    if hasattr(model, 'layers'):
        layers = []
        for layer in model.layers:
            layers += _get_layers(layer)
        return layers
    else:
        return [model.__class__.__name__]

used_layers_file = 'used_layers.json'
if os.path.isfile(used_layers_file):
    with open(used_layers_file, 'r') as f:
        _used_layers = set(json.load(f))
else:
    _used_layers = set()

def _update_used_layers(model):
    n = len(_used_layers)
    layers = _get_layers(model)
    _used_layers.update(layers)
    if len(_used_layers) > n:
        with open(used_layers_file, 'w') as f:
            json.dump(list(_used_layers), f)

def rand(shape):
    shape = [d if d is not None else 1 for d in shape]
    return np.random.random(shape)

def put_data(model, input_arrays, output_arrays, h5file):
    input_names = [i.encode('utf8') for i in model.input_names]
    output_names = [o.encode('utf8') for o in model.output_names]
    f = h5py.File(h5file)
    data = f.create_group('data')
    data.attrs['input_names'] = input_names
    data.attrs['output_names'] = output_names
    for name, arr in zip(input_names, input_arrays):
        data.create_dataset(name, shape=arr.shape, dtype='f', data=arr)
    for name, arr in zip(output_names, output_arrays):
        data.create_dataset(name, shape=arr.shape, dtype='f', data=arr)
    f.close()

def put_rand_data(model, h5file):
    input_shapes = [tuple(i.shape) for i in model.inputs]
    input_arrays = list(map(rand, input_shapes))
    output_arrays = model.predict(input_arrays)
    if not isinstance(output_arrays, list):
        output_arrays = [output_arrays]
    return put_data(model, input_arrays, output_arrays, h5file)

def _to_sequential(model):
    seq = tf.keras.models.Sequential()
    for layer in model.layers[1:]:
        seq.add(layer)
    return seq


def save_model(model, file_name, data='rand', as_sequential=False):
    try:
        if as_sequential:
            model = _to_sequential(model)
            print(model.layers[0].input_shape)
            model.predict(rand(model.layers[0].input_shape))
        path = os.path.join(tfkeras_dir, file_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            model.predict(rand(model.input_shape))
            model.save(path)
            put_rand_data(model, path)
            _update_used_layers(model)
        except:
            pass

    except Exception as e:
        print(e)

def grid(*args, **kwargs):
    if args and callable(args[0]):
        return grid()(args[0])
    def inner(f):
        def grid_call():
            class ArgIterator():
                def __init__(self, args, kwargs):
                    args = list(args)
                    fargspec = inspect.getfullargspec(f)
                    if fargspec.varargs or fargspec.kwonlyargs:
                        raise Exception("*args and and **kwargs not supported for grid call.")
                    fargs = fargspec.args
                    fdefs = fargspec.defaults
                    if fdefs is None:
                        fdefs = ()
                    for i, arg in enumerate(fargs[len(args):]):
                        if arg in kwargs:
                            args.append(kwargs[arg])
                        elif i >= len(fargs) -len(args) - len(fdefs):
                            args.append([fdefs[i - (len(fargs) -len(args) - len(fdefs))]])
                        else:
                            raise Exception('Unable to resolve value for argument ' + arg)
                    self.args = itertools.product(*args)
                    n = 1
                    for a in args:
                        n *= len(a)
                    self.n = n
                def __len__(self): return self.n
                def __iter__(self): return self
                def __next__(self): return self.next()
                def __next__(self): return f(*next(self.args))
            return ArgIterator(args, kwargs)
        return grid_call
    return inner


def get_coverage():
    ret = {}
    # calculate layer coverage TODO: op coverage
    all_layers = []
    all_layers_set = set()
    for l in dir(tf.keras.layers):
        layer = getattr(tf.keras.layers, l)
        if inspect.isclass(layer):
            l = layer.__name__
            if issubclass(layer, tf.keras.layers.Layer) \
                and l[0].isupper() and not l.startswith('Abstract') and \
                    not l.endswith('Cell') and l not in ('RNN', 'Layer', 'Wrapper'):
                        if layer not in all_layers_set:
                            all_layers_set.add(layer)
                            all_layers.append(l)
    ret['all_layers'] = all_layers
    with open(used_layers_file, 'r') as f:
        used_layers = json.load(f)
    unused_layers = [l for l in all_layers if l not in used_layers]
    coverage = len(used_layers) / len(all_layers)
    ret['covered_layers'] = used_layers
    ret['uncovered_layers'] = unused_layers
    ret['coverage'] = coverage
    return ret