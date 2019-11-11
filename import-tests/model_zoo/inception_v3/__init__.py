from tfoptests import persistor
import numpy as np

model_name = "inception_v3_with_softmax"
save_dir = model_name

def get_input(name):
    np.random.seed(13)
    if name == "input":
        input = np.random.uniform(-1,1,size=[1,299,299,3]) #fake image
        persistor.save_input(input, name, save_dir)
        return input


def list_inputs():
    return ["input"]


def get_inputs():
    my_input_dict = {}
    for a_input in list_inputs():
        my_input_dict[a_input] = get_input(a_input)
    return my_input_dict