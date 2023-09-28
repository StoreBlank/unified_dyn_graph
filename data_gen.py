import os
import cv2
import numpy as np
import time
import yaml
from flex_env import FlexEnv

def load_yaml(filename):
    # load YAML file
    return yaml.safe_load(open(filename, 'r'))

# load config
config = load_yaml("config/data_gen/gnn_dyn.yaml")
data_dir = config['dataset']['folder']
n_episode = config['dataset']['n_episode']
n_timestep = config['dataset']['n_timestep']
action_dim = 4
obj = config['dataset']['obj']
wkspc_w = config['dataset']['wkspc_w']

os.system("mkdir -p %s" % data_dir)

def gen_data(info):
    base_epi = info["base_epi"]
    thread_idx = info["thread_idx"]
    verbose = info["verbose"]

    env = FlexEnv(config)
    np.random.seed(0)

    idx_episode = base_epi
    while idx_episode < base_epi + n_episode:
        if verbose:
            print("Episode %d" % idx_episode)
        env.reset()
       
        epi_dir = os.path.join(data_dir, "episode_%d" % idx_episode)
        os.system("mkdir -p %s" % epi_dir)

        # initial render
        actions = np.zeros((n_timestep, action_dim))

        img = env.render()
        cv2.imwrite(os.path.join(epi_dir, "0_color.png"), img[..., :3] / 255)
        cv2.imwrite(os.path.join(epi_dir, "0_depth.png"), img[:, :, -1].astype(np.uint16)) #TODO: check if this is correct
        with open(os.path.join(epi_dir, '0_particles.npy'), 'wb') as f:
            np.save(f, env.get_positions())
        
        # last_img = img.copy()
        # valid = True
        # for idx_timestep in range(n_timestep):
        #     color_diff = 0
        #     while color_diff < 0.001:
        #         u = None
        #         u = env.sample_action(1)
        #         u = u[0, 0] # starting and ending positions of actions

        #         # step
        #         img = env.step(u)


        idx_episode += 1




info = {
    "base_epi": 0,
    "thread_idx": 1,
    "verbose": True
}
gen_data(info)

