import tensorflow as tf
import mlp
import rnn
import cnn
import merge
import activations
import advanced_activations
import utils
import os
import gc
import warnings
import json
from utils import tqdm
import sys
import subprocess

jobs = [mlp, rnn, cnn, merge, activations, advanced_activations]
jobs.remove(cnn)
jobs.remove(rnn)
def run_sequential():
    for job in jobs:
        job.run()

def run_sequential_subprocess():
    for job in jobs:
        subprocess.Popen(f'"{sys.executable}" "{job.__file__}"', stdout=subprocess.PIPE).stdout.readlines()

#run_sequential()
run_sequential_subprocess()

import coverage
