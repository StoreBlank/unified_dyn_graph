import os
import time
import numpy as np
import cv2
import pyflex

# robot
from robot import FlexRobotHelper

pyflex.loadURDF = FlexRobotHelper.loadURDF
pyflex.resetJointState = FlexRobotHelper.resetJointState
pyflex.getRobotShapeStates = FlexRobotHelper.getRobotShapeStates

# utils
from utils import load_yaml

# load config
config = load_yaml("../config/data_gen/gnn_dyn.yaml")
data_dir = config['dataset']['folder']
n_episode = config['dataset']['n_episode']
n_timestep = config['dataset']['n_timestep']
action_dim = 4
obj = config['dataset']['obj']
wkspc_w = config['dataset']['wkspc_w']
os.system("mkdir -p %s" % data_dir)

def gen_data(info):
    start_time = time.time()

    base_epi = info["base_epi"]
    thread_idx = info["thread_idx"]
    verbose = info["verbose"]

    np.random.seed(round(time.time() * 1000 + thread_idx) % 2 ** 32)










# run 
info = {
    "base_epi": 0,
    "thread_idx": 0,
    "verbose": True
}
gen_data(info)