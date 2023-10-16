import os
import cv2
import numpy as np
import time
import yaml
from flex_env_dense import FlexEnv
import trimesh

def load_yaml(filename):
    # load YAML file
    return yaml.safe_load(open(filename, 'r'))

# load config
config = load_yaml("config/data_gen/gnn_dyn.yaml")
data_dir = config['dataset']['folder']
os.system("mkdir -p %s" % data_dir)

n_episode = config['dataset']['n_episode']
n_rollout = config['dataset']['n_rollout']
n_timestep = config['dataset']['n_timestep']
dt = config['dataset']['dt']

obj = config['dataset']['obj']
wkspc_w = config['dataset']['wkspc_w']

def gen_data(info):
    start_time = time.time()

    base_epi = info["base_epi"]
    thread_idx = info["thread_idx"]
    verbose = info["verbose"]

    # create folder
    folder_dir = os.path.join(data_dir, obj)
    os.system('mkdir -p ' + folder_dir)

    # set env 
    env = FlexEnv(config)
    np.random.seed(0)

    idx_episode = base_epi
    while idx_episode < base_epi + n_episode:
        start_epi_time = time.time()
        if verbose: 
            print('episode:', idx_episode)
       
        env.reset()
    


        
    env.close()
    end_time = time.time()
    print('total time: ', end_time - start_time)


info = {
    "base_epi": 0,
    "thread_idx": 0,
    "verbose": True
}
gen_data(info)

