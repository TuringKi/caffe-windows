import numpy as np
import sys, os, subprocess
import argparse
                                                                 
from paths import caffe_root
sys.path.append(caffe_root)

import caffe
from config_processing import *

parser = argparse.ArgumentParser()
parser.add_argument('R', type=int, help='number of components in the decomposition')
parser.add_argument('layer', type=str, nargs='?', help='which conv layer to decompose', default='conv1')
args = parser.parse_args()

LAYER = args.layer
R = args.R
NET_PATH = 'tophand/'
NET_NAME = 'tophand'
INPUT_DIM = [1, 3, 256, 256]

prepare_models(LAYER, R, NET_PATH, NET_NAME, INPUT_DIM)
