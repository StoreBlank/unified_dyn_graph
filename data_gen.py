import os
import cv2
import numpy as np
import time
import yaml
from flex_env import FlexEnv
import trimesh
import json
import multiprocessing as mp

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

def gen_data(info):
    start_time = time.time()

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

    all_actions = np.array([])

    idx_episode = base_epi
    while idx_episode < base_epi + n_epi_per_worker:
        start_epi_time = time.time()
        print('episode:', idx_episode)
       
        env.reset()
       
        epi_dir = os.path.join(folder_dir, "episode_%d" % idx_episode)
        os.system("mkdir -p %s" % epi_dir)

        # save property
        property = env.get_property()
        print(property)
        with open(os.path.join(epi_dir, 'property.json'), 'w') as f:
           json.dump(property, f)
        
        actions = np.zeros((n_timestep, action_dim))
        color_threshold = 0.1

        # time step
        img = env.render()
        last_img = img.copy()
        for idx_timestep in range(n_timestep):
            if verbose:
                print('timestep %d' % idx_timestep)
            
            color_diff = 0
            while color_diff < color_threshold: #granular: 0.001 
                # if cam_view == 1:
                #     u = None
                #     u = env.sample_action()
                # else:
                #     u = actions[idx_timestep]

                # u = [0., 0., 0., 0.]
                u = [0., 1., 0., -1.]

                # step
                if debug:
                    img, steps = env.step(u)
                else: 
                    img, steps = env.step(u, epi_dir)
                
                color_diff = np.mean(np.abs(img[:, :, :3] - last_img[:, :, :3]))
                if verbose:
                    print('color_diff:', color_diff)

            actions[idx_timestep] = u
            last_img = img.copy()

            if verbose:
                print('action: ', u)
                print('num particles: ', env.get_positions().shape[0] // 4)
                print('particle positions: ', env.get_positions().reshape(-1, 4))
                print('\n')
           
        all_actions = np.append(all_actions, actions)
        
        end_epi_time = time.time()
        print('episiode %d time: ' % idx_episode, end_epi_time - start_epi_time)
        idx_episode += 1
        
    # save camera params
    # cam_intrinsic_params = env.get_camera_intrinsics()
    # cam_extrinsic_matrix = env.get_camera_extrinsics()
    # np.save(os.path.join(des_dir, 'camera_intrinsic_params.npy'), cam_intrinsic_params)
    # np.save(os.path.join(des_dir, 'camera_extrinsic_matrix.npy'), cam_extrinsic_matrix)

    # save actions
    # if cam_view in [0, 1]:
    #     all_actions = all_actions.reshape(n_episode, n_timestep, action_dim)
    #     np.save(os.path.join(folder_dir, 'actions.npy'), all_actions)
            
    env.close()
    end_time = time.time()
    print('total time: ', end_time - start_time)

# multiprocessing
infos=[]
for i in range(n_worker):
    info = {
        "base_epi": i*n_episode//n_worker,
        "n_epi_per_worker": n_episode//n_worker,
        "thread_idx": i,
        "verbose": False,
        "debug": False,
    }
    infos.append(info)

pool = mp.Pool(processes=n_worker)
pool.map(gen_data, infos)


# info = {
#     "base_epi": 0,
#     "thread_idx": 1,
#     "verbose": False,
#     "debug": True,
# }
# gen_data(info)

