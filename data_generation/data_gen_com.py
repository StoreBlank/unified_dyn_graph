import os
import numpy as np
import time
from env.flex_env_com import FlexEnv
import json
import multiprocessing as mp

from utils_env import load_yaml
from utils_env import rand_float, rand_int, quatFromAxisAngle, find_min_distance

# load config
config = load_yaml("config/data_gen/gnn_dyn_com.yaml")
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
    
    idx_episode = info["epi"]
    debug = info["debug"]
    
    # create folder
    folder_dir = os.path.join(data_dir, obj)
    os.system('mkdir -p ' + folder_dir)

    # set env 
    env = FlexEnv(config)
    np.random.seed(idx_episode)
    print('episode start:', idx_episode)
    
    if debug:
        particle_pos_list, eef_pos_list, step_list, contact_list = env.reset() 
    else:
        epi_dir = os.path.join(folder_dir, "episode_%d" % idx_episode)
        os.system("mkdir -p %s" % epi_dir)
        
        particle_pos_list, eef_pos_list, step_list, contact_list = env.reset(dir=epi_dir)
        
        # save property
        property_params = env.get_property()
        with open(os.path.join(epi_dir, 'property.json'), 'w') as f:
            json.dump(property_params, f)
    
    # obj_size = env.get_obj_size()
    # print("obj_size:", obj_size)
    obj_center = env.get_obj_center()
    print("obj_center before push:", obj_center)
    
    def get_com(particle_pos):
        num_particles = particle_pos.shape[0]
        center_of_mass = 0
        overall_mass = 0
        for i in range(num_particles):
            center_of_mass += particle_pos[i, :3] * (1/particle_pos[i, 3])
            overall_mass += 1/particle_pos[i, 3]
        center_of_mass /= overall_mass 
        return center_of_mass
    
    particle_pos = env.get_positions().reshape((-1, 4))
    com = get_com(particle_pos)
    print("com:", com)
    
    
    actions = np.zeros((n_timestep, action_dim))
    color_threshold = 0.01 # granular objects
    
    # n_pushes
    img = env.render()
    last_img = img.copy()
    stuck = False
    for idx_timestep in range(n_timestep):
        center_x, center_y, center_z = env.get_obj_center()
        
        color_diff = 0
        prev_particle_pos_list, prev_eef_pos_list, prev_step_list, prev_contact_list = particle_pos_list.copy(), eef_pos_list.copy(), step_list.copy(), contact_list.copy()
        for k in range(10):
            u = None
            # u = env.sample_action()
            u = [center_x, center_z+2., center_x, center_z-2.]
            # u = [0., 0., 0., 0.]
            if u is None:
                stuck = True
                print(f"Episode {idx_episode} timestep {idx_timestep}: No valid action found!")
                break
    
            # step
            if debug:
                img, particle_pos_list, eef_pos_list, step_list, contact_list = env.step(u, particle_pos_list=particle_pos_list, eef_pos_list=eef_pos_list, step_list=step_list, contact_list=contact_list)
            else: 
                img, particle_pos_list, eef_pos_list, step_list, contact_list = env.step(u, epi_dir, particle_pos_list, eef_pos_list, step_list, contact_list)
            
            # check whether action is valid 
            color_diff = np.mean(np.abs(img[:, :, :3] - last_img[:, :, :3]))
            
            
            if color_diff < color_threshold:
                particle_pos_list, eef_pos_list, step_list, contact_list = prev_particle_pos_list, prev_eef_pos_list, prev_step_list, prev_contact_list
                if k == 9:
                    stuck = True
                    print('The process is stucked on episode %d timestep %d!!!!' % (idx_episode, idx_timestep))
            else:
                break
                
        if not stuck:
            actions[idx_timestep] = u
            last_img = img.copy()
            if not debug:
                print('episode %d timestep %d done!!! step: %d' % (idx_episode, idx_timestep, step_list[-1]))
        else:
            break     
    
    obj_center = env.get_obj_center()
    print("obj_center after push:", obj_center)  
    
    # save actions and steps and end effector positions
    if not debug:
        np.save(os.path.join(epi_dir, 'actions.npy'), actions)
        np.save(os.path.join(epi_dir, 'particles_pos'), particle_pos_list)
        np.save(os.path.join(epi_dir, 'eef_pos.npy'), eef_pos_list)
        np.save(os.path.join(epi_dir, 'steps.npy'), step_list)
        np.save(os.path.join(epi_dir, 'contact.npy'), contact_list)
        
    end_time = time.time()
    # print("Finish episode %d!!!!" % idx_episode)
    print(f"Episode {idx_episode} step list: {step_list}")
    print('Episode %d time: ' % idx_episode, end_time - start_time)
    
    if not debug:
        print(f'Episode {idx_episode} physics property: {property_params}.')
        # save camera params
        cam_intrinsic_params, cam_extrinsic_matrix = env.get_camera_params()
        np.save(os.path.join(folder_dir, 'camera_intrinsic_params.npy'), cam_intrinsic_params)
        np.save(os.path.join(folder_dir, 'camera_extrinsic_matrix.npy'), cam_extrinsic_matrix)
            
    env.close()

### multiprocessing
# bases = [0 + 20*n for n in range(10)]
# # bases = [0]
# print(bases)
# for base in bases:
#     print("base:", base)
#     infos=[]
#     for i in range(n_worker):
#         info = {
#             "epi": base+i*n_episode//n_worker,
#             "debug": False,
#         }
#         infos.append(info)
#     pool = mp.Pool(processes=n_worker)
#     pool.map(gen_data, infos)


info = {
    "epi": 0,
    "debug": False,
}
gen_data(info)

