from tfoptests import persistor
from model_zoo.inception_v3 import save_dir, get_inputs

persistor.freeze_n_save_graph(save_dir)
persistor.write_frozen_graph_txt(save_dir)
persistor.save_intermediate_nodes(save_dir, get_inputs())
