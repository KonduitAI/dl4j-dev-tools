import os
import itertools
import inspect

dl4j_test_resources = os.environ.get('DL4J_TEST_RESOURCES')

if dl4j_test_resources is None:
    raise Exception('Environment variable not set: DL4J_TEST_RESOURCES')

tfkeras_dir = os.path.join(dl4j_test_resources,
                           'src', 'main', 'resources',
                           'modelimport', 'keras', 'tfkeras')


def save_model(model, file_name):
    path = os.path.join(tfkeras_dir, file_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)


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
                def __iter__(self): return self
                def __next__(self): return self.next()
                def __next__(self): return f(*next(self.args))
            return ArgIterator(args, kwargs)
        return grid_call
    return inner
