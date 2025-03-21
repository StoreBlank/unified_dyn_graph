import os
import numpy as np
import time
from env.flex_env import FlexEnv
import json
import multiprocessing as mp
import itertools

from utils_env import load_yaml
from utils_env import rand_float, rand_int, quatFromAxisAngle, find_min_distance

# load config
config = load_yaml("config/data_gen/gnn_dyn_grid.yaml")
data_dir = config['dataset']['folder']
n_worker = config['dataset']['n_worker']
n_episode = config['dataset']['n_episode']
n_timestep = config['dataset']['n_timestep']
action_dim = 4
obj = config['dataset']['obj']
wkspc_w = config['dataset']['wkspc_w']
cam_view = config['dataset']['camera_view']

os.system("mkdir -p %s" % data_dir)

def gen_data_grid(info):
    base_epi = info["base_epi"]
    n_epi_per_worker = info["n_epi_per_worker"]
    thread_idx = info["thread_idx"]
    verbose = info["verbose"]
    debug = info["debug"]
    combination = info["combination"]
    
    # set env 
    env = FlexEnv(config)
    np.random.seed(round(time.time() * 1000 + thread_idx) % 2 ** 32)

    # create folder
    folder_dir = os.path.join(data_dir, obj)
    os.system('mkdir -p ' + folder_dir)

    idx_episode = base_epi
    while idx_episode < base_epi + n_epi_per_worker:
        start_epi_time = time.time()
        
        length, thickness, cluster_spacing, dynamic_friction = combination
        property_params = {
            "length": length,
            "thickness": thickness,
            "cluster_spacing": cluster_spacing,
            "dynamic_friction": dynamic_friction,
        }
        print("episode: ", idx_episode, "; property_params: ", property_params)
        
        if debug:
            n_steps = env.reset(property_params=property_params) 
        else:
            epi_dir = os.path.join(folder_dir, "episode_%d" % idx_episode)
            os.system("mkdir -p %s" % epi_dir)
            n_steps = env.reset(dir=epi_dir, property_params=property_params)
            # save property
            property = env.get_property()
            with open(os.path.join(epi_dir, 'property.json'), 'w') as f:
                json.dump(property, f)
        
        actions = np.zeros((n_timestep, action_dim))
        color_threshold = 0.1

        # time step
        img = env.render()
        last_img = img.copy()
        steps_list = []
        contacts_list = []
        for idx_timestep in range(n_timestep):
            if verbose:
                print('timestep %d' % idx_timestep)
            
            color_diff = 0
            while color_diff < color_threshold:
                # u = None
                # u = env.sample_action()
                
                center_x, center_z = env.get_obj_center()
                u = [center_x, 2.0, center_x, -1.5] #-z -> +z
        
                # step
                prev_steps = n_steps
                if debug:
                    img, n_steps, contact = env.step(u)
                else: 
                    img, n_steps, contact = env.step(u, n_steps, epi_dir)
                
                # check whether action is valid 
                color_diff = np.mean(np.abs(img[:, :, :3] - last_img[:, :, :3]))
                if color_diff < color_threshold:
                    n_steps = prev_steps
                else:
                    steps_list.append(n_steps)
                    contacts_list.append(contact)
                
                if verbose:
                    print('color_diff:', color_diff)

            actions[idx_timestep] = u
            last_img = img.copy()

            if verbose:
                print('action: ', u)
                print('num particles: ', env.get_positions().shape[0] // 4)
                print('particle positions: ', env.get_positions().reshape(-1, 4))
                print('\n')
            
            # check whether the object is inside the workspace
            if not env.inside_workspace():
                print("Object outside workspace!")
                break
            
            print('episode %d timestep %d done!!! step: %d' % (idx_episode, idx_timestep, n_steps))
        
        # save actions and steps and end effector positions
        if not debug:
            np.save(os.path.join(epi_dir, 'actions.npy'), actions)
            np.save(os.path.join(epi_dir, 'steps.npy'), np.array(steps_list))
            np.save(os.path.join(epi_dir, 'contacts.npy'), np.array(contacts_list))

        end_epi_time = time.time()
        print("Finish episode %d!!!!" % idx_episode)
        print('episiode %d time: ' % idx_episode, end_epi_time - start_epi_time)
        idx_episode += 1
        
    # save camera params
    if not debug:
        cam_intrinsic_params, cam_extrinsic_matrix = env.get_camera_params()
        np.save(os.path.join(folder_dir, 'camera_intrinsic_params.npy'), cam_intrinsic_params)
        np.save(os.path.join(folder_dir, 'camera_extrinsic_matrix.npy'), cam_extrinsic_matrix)
            
    env.close()


###multiprocessing

## rope grid
length_list = [0.5, 1.0, 2.0, 2.5] 
thickness_list = [1.0, 1.5, 2.0, 2.5] 
cluster_spacing_list = [2.0, 4.0, 6.0, 8.0]
dynamic_friction_list = [0.1, 0.3, 0.5, 0.7]
# Generate all combinations of the rope properties
property_combinations = list(itertools.product(length_list, thickness_list, cluster_spacing_list, dynamic_friction_list))
# print("property_combinations: ", property_combinations)

total_episode = len(property_combinations)
print("total_episode: ", total_episode)

n_bases = total_episode // n_worker
# bases = [i*n_worker for i in range(n_bases+1)]
# print("bases: ", bases)
bases = [231]

for base in bases:
    
    # if base == bases[-1] and total_episode % n_worker != 0:
    #     n_worker = total_episode - base
    #     n_episode = n_worker
    # elif base == bases[-1] and total_episode % n_worker == 0:
    #     break
    
    infos = []
    for i in range(n_worker):
        info = {
            "base_epi": base+i*n_episode//n_worker,
            "n_epi_per_worker": n_episode//n_worker,
            "thread_idx": i,
            "combination": property_combinations[base+i],
            "verbose": False,
            "debug": False,
        }
        infos.append(info)
    pool = mp.Pool(processes=n_worker)
    pool.map(gen_data_grid, infos)




