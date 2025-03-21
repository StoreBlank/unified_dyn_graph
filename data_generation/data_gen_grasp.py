import os
import cv2
import numpy as np
import time
import trimesh
import json
import pickle
import multiprocessing as mp

from env.flex_env_rope import FlexEnv

from utils_env import load_yaml
from utils_env import rand_float, rand_int, quatFromAxisAngle, find_min_distance

# load config
config = load_yaml("config/data_gen/gnn_dyn_grasp.yaml")
data_dir = config['dataset']['folder']
n_worker = config['dataset']['n_worker']
n_episode = config['dataset']['n_episode']
n_timestep = config['dataset']['n_timestep']
action_dim = 4
obj = config['dataset']['obj']
wkspc_w = config['dataset']['wkspc_w']
cam_view = config['dataset']['camera_view']

os.system("mkdir -p %s" % data_dir)

def gen_data(info):
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
        start_epi_time = time.time()
        print('episode:', idx_episode)
        
        if debug:
            n_steps = env.reset() 
        else:
            epi_dir = os.path.join(folder_dir, "episode_%d" % idx_episode)
            os.system("mkdir -p %s" % epi_dir)
            n_steps = env.reset(dir=epi_dir)
            # save property
            property = env.get_property()
            with open(os.path.join(epi_dir, 'property.json'), 'w') as f:
                json.dump(property, f)
        
        actions = np.zeros((n_timestep, action_dim))
        color_threshold = 0.1
        
        #### actions TODO]
        ## cloth bag rigid objects
        # us = [[1.3, 0.55, -1., -0.3], 
        #       [-1., -1, 1., 0.],
        #       [-1, -1, 1., 1.]]
        
        ## stirring
        # us = [[0., -0.1, 0.5, -0.1],
        #       [0.2, -0.5, 0.2, 0.]]
        
        ## scooping
        # us = [[0., 1.3, 0.1, -0.2],]

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
                # u = [center_x, 2.0, center_x, -1.5] #-z -> +z
                
                # u = us[idx_timestep]
                
                x_min_idx, x_max_idx, z_min_idx, z_max_idx = env.get_obj_idx_range()
                u = [center_x, center_z, center_x, -0.5]
        
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
# bases = [210, 240, 270, 300, 330, 360, 390, 420, 450, 480]
# infos=[]
# for base in bases:
#     # base = 210
#     for i in range(n_worker):
#         info = {
#             "base_epi": base+i*n_episode//n_worker,
#             "n_epi_per_worker": n_episode//n_worker,
#             "thread_idx": i,
#             "verbose": False,
#             "debug": False,
#         }
#         infos.append(info)
#     pool = mp.Pool(processes=n_worker)
#     pool.map(gen_data, infos)


info = {
    "base_epi": 0,
    "n_epi_per_worker": n_episode,
    "thread_idx": 1,
    "verbose": False,
    "debug":True,
}
gen_data(info)

