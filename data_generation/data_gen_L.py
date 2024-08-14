import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
import json
import multiprocessing as mp

from env.pymunk_L import L_Sim, parse_merging_pose, rand_float
from utils_env import load_yaml

# load config
config = load_yaml("config/data_gen/merging_L.yaml")
data_dir = config['dataset']['folder']
n_worker = config['dataset']['n_worker']
n_timestep = config['dataset']['n_timestep']
action_dim = 4

os.system("mkdir -p %s" % data_dir)

def gen_data(info):
    start_time = time.time()

    idx_episode = info["epi"]
    debug = info["debug"]

    # set env
    env = L_Sim(config['env'])
    np.random.seed(idx_episode)
    print('episode start:', idx_episode)

    # merging initial flag
    r = rand_float(0, 1)
    init_flag = True if r < 0.5 else False

    if init_flag:
        init_poses = [
            rand_float(env.width / 2 - 100, env.width / 2 + 100),
            rand_float(env.height / 2 - 100, env.height / 2 + 100),
            rand_float(-np.pi, np.pi)
        ]
        pose_mode = 0 # merging
        init_poses = parse_merging_pose(
            init_poses, config['env'], rand_float(0, 20), pose_mode, noisy=True
        )
        if debug:
            particle_pos_list, eef_states_list, step_list, contact_list = env.reset(new_poses=init_poses)
        else:
            epi_dir = os.path.join(data_dir, "episode_%d" % idx_episode)
            os.system("mkdir -p %s" % epi_dir)
            
            particle_pos_list, eef_states_list, step_list, contact_list = env.reset(new_dir=epi_dir, new_poses=init_poses)
    else:
        if debug:
            particle_pos_list, eef_states_list, step_list, contact_list = env.reset()
        else:
            epi_dir = os.path.join(data_dir, "episode_%d" % idx_episode)
            os.system("mkdir -p %s" % epi_dir)
            
            particle_pos_list, eef_states_list, step_list, contact_list = env.reset(new_dir=epi_dir)

    actions = np.zeros((n_timestep, action_dim))

    # n_pushes
    color_threshold = 0.01
    img = env.render()
    last_img = img.copy()
    stuck = False
    for idx_timestep in range(n_timestep):
        color_diff = 0
        prev_particle_pos_list, prev_eef_states_list, prev_step_list, prev_contact_list = particle_pos_list.copy(), eef_states_list.copy(), step_list.copy(), contact_list.copy()
        for k in range(10):
            u = None
            u = env.sample_action()
            if u is None:
                stuck = True
                print(f"Episode {idx_episode} timestep {idx_timestep}: No valid action found!")
                break

            # step
            img, particle_pos_list, eef_states_list, step_list, contact_list = env.step(u, particle_pos_list=particle_pos_list, eef_states_list=eef_states_list, step_list=step_list, contact_list=contact_list)

            # check if the object is stuck
            color_diff = np.mean(np.abs(img - last_img))

            if color_diff < color_threshold:
                particle_pos_list, eef_states_list, step_list, contact_list = prev_particle_pos_list, prev_eef_states_list, prev_step_list, prev_contact_list
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

    if not debug:
        np.save(os.path.join(epi_dir, 'actions.npy'), actions)
        np.save(os.path.join(epi_dir, 'particles_pos'), particle_pos_list)
        np.save(os.path.join(epi_dir, 'eef_states.npy'), eef_states_list)
        np.save(os.path.join(epi_dir, 'steps.npy'), step_list)
        np.save(os.path.join(epi_dir, 'contact.npy'), contact_list)

    end_time = time.time()
    print(f"Episode {idx_episode} step list: {step_list}")
    print('Episode %d time: ' % idx_episode, end_time - start_time)

### multiprocessing
num_episode = 2000
num_bases = num_episode // n_worker
bases = [0 + n_worker*n for n in range(num_bases)]
print(f"num_bases: {len(bases)}")
print(bases)

for base in bases:
    print("base:", base)
    infos=[]
    for i in range(n_worker):
        info = {
            "epi": base+i,
            "debug": False,
            "thres_idx": base,
        }
        infos.append(info)
    pool = mp.Pool(processes=n_worker)
    pool.map(gen_data, infos)
