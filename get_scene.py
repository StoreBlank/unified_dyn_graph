import os
import cv2
import numpy as np
import time
import yaml
import trimesh
import json
import pickle
import multiprocessing as mp

from flex_env import FlexEnv

def load_yaml(filename):
    # load YAML file
    return yaml.safe_load(open(filename, 'r'))

# load config
config = load_yaml("config/data_gen/gnn_dyn.yaml")
data_dir = config['dataset']['folder']
n_worker = config['dataset']['n_worker']
n_episode = config['dataset']['n_episode']
n_timestep = config['dataset']['n_timestep']
action_dim = 4
obj = config['dataset']['obj']
wkspc_w = config['dataset']['wkspc_w']
cam_view = config['dataset']['camera_view']

os.system("mkdir -p %s" % data_dir)

def gen_scene(info):
    base_epi = info["base_epi"]
    n_epi_per_worker = info["n_epi_per_worker"]
    thread_idx = info["thread_idx"]
    verbose = info["verbose"]
    debug = info["debug"]

    # create folder
    folder_dir = os.path.join(data_dir, obj)
    os.system('mkdir -p ' + folder_dir)

    # set env 
    env = FlexEnv(config)
    np.random.seed(round(time.time() * 1000 + thread_idx) % 2 ** 32)

    idx_episode = base_epi
    while idx_episode < base_epi + n_epi_per_worker:
        
        n_steps = env.reset() 

        # time step
        img = env.render()
            
    env.close()


info = {
    "base_epi": 0,
    "n_epi_per_worker": n_episode,
    "thread_idx": 1,
    "verbose": False,
    "debug": True,
}
gen_scene(info)

