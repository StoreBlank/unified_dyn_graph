import os
import cv2
import numpy as np
import time
import yaml
from flex_env import FlexEnv
import trimesh
import json
import pickle
import multiprocessing as mp

def load_yaml(filename):
    # load YAML file
    return yaml.safe_load(open(filename, 'r'))

# load config
config = load_yaml("config/data_gen/data_init.yaml")
data_dir = config['dataset']['folder']
init = config['dataset']['init']

os.system("mkdir -p %s" % data_dir)

def gen_init(epi):
    # read property from epi file
    epi_dir = os.path.join('/media/baoyu/sumsung/rope', "episode_%d" % epi)
    property = json.load(open(os.path.join(epi_dir, 'property.json'), 'r'))
    print('n_particle:', property['num_particles'])

    # length, thickness = property['length'] / 80., property['thickness'] / 80.
    # print(epi, 'length:', length, 'thickness:', thickness)
    
    # set env 
    env = FlexEnv(config)
    save_dir = os.path.join(data_dir, "episode_%d" % epi)
    env.reset(init, epi, property, save_dir)
    print('n_particle:', env.get_num_particles())
    
    env.close()

if __name__ == '__main__':
    epi = 2
    gen_init(epi)

