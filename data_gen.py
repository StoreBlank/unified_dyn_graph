import os
import cv2
import numpy as np
import time
import yaml
from flex_env import FlexEnv
import trimesh

np.random.seed(0)

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
cam_view = config['dataset']['camera_view']

os.system("mkdir -p %s" % data_dir)

def gen_data(info):
    base_epi = info["base_epi"]
    thread_idx = info["thread_idx"]
    verbose = info["verbose"]

    # create folder
    folder_dir = os.path.join(data_dir, obj)
    os.system('mkdir -p ' + folder_dir)
    des_dir = os.path.join(folder_dir, 'camera_%d' % cam_view)
    os.system('mkdir -p ' + des_dir)

    # set env 
    env = FlexEnv(config)
    np.random.seed(0)

    idx_episode = base_epi
    while idx_episode < base_epi + n_episode:
        env.reset()
       
        epi_dir = os.path.join(des_dir, "episode_%d" % idx_episode)
        os.system("mkdir -p %s" % epi_dir)

        # initial render
        actions = np.zeros((n_timestep, action_dim))

        img = env.render()
        cv2.imwrite(os.path.join(epi_dir, "0_color.png"), img[..., :3][..., ::-1])
        cv2.imwrite(os.path.join(epi_dir, "0_depth.png"), (img[:, :, -1]*1000).astype(np.uint16))
        with open(os.path.join(epi_dir, '0_particles.npy'), 'wb') as f:
            np.save(f, env.get_positions().reshape(-1, 4))
        # save img
        with open(os.path.join(epi_dir, '0_obs.npy'), 'wb') as f:
            np.save(f, img)
        
        last_img = img.copy()
        valid = True
        for idx_timestep in range(n_timestep):
            print('timestep:', idx_timestep)
            color_diff = 0
            while color_diff < 0.1: #0.001
                u = None
                u = env.sample_action()
                # u = u[0, 0] # starting and ending positions of actions
                # u = [1, 0, -1, 0]
                # u = [0, -2, 0, 1]

                # step
                img = env.step(u)
                if img is None:
                    valid = False
                    print('rerun epsiode %d' % idx_episode)
                    break
                
                color_diff = np.mean(np.abs(img[:, :, :3] - last_img[:, :, :3]))
                print('color_diff:', color_diff)

                # # get cloth mesh
                # positions = env.get_positions().reshape((-1, 4))
                # vertices = positions[:, :3]
                # faces = env.get_faces().reshape((-1, 3))
                # cloth_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                # # save cloth mesh
                # cloth_mesh.export(os.path.join(epi_dir, "cloth.obj"))

            if valid:
                cv2.imwrite(os.path.join(epi_dir, '%d_color.png' % (idx_timestep + 1)), img[:, :, :3][..., ::-1])
                cv2.imwrite(os.path.join(epi_dir, '%d_depth.png' % (idx_timestep + 1)), (img[:, :, -1]*1000).astype(np.uint16))
                with open(os.path.join(epi_dir, '%d_particles.npy' % (idx_timestep + 1)), 'wb') as f:
                    np.save(f, env.get_positions().reshape(-1, 4))
                # save img
                with open(os.path.join(epi_dir, '%d_obs.npy' % (idx_timestep + 1)), 'wb') as f:
                    np.save(f, img)
                
                actions[idx_timestep] = u
                last_img = img.copy()

                if verbose:
                    print('timestep %d' % idx_timestep)
                    print('action: ', actions)
                    print('num particles: ', env.get_positions().shape[0] // 4)
                    print('particle positions: ', env.get_positions().reshape(-1, 4))
                    print('\n')
               
        idx_episode += 1
    
    # save camera params
    cam_intrinsic_params = env.get_camera_intrinsics()
    cam_extrinsic_matrix = env.get_camera_extrinsics()
    np.save(os.path.join(des_dir, 'camera_intrinsic_params.npy'), cam_intrinsic_params)
    np.save(os.path.join(des_dir, 'camera_extrinsic_matrix.npy'), cam_extrinsic_matrix)

    # save actions
    np.save(os.path.join(folder_dir, 'actions.npy'), actions)
    
    env.close()


info = {
    "base_epi": 0,
    "thread_idx": 1,
    "verbose": False
}
gen_data(info)

