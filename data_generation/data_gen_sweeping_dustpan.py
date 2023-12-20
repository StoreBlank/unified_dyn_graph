import os
import time
import numpy as np
import pyflex
import cv2
import json
import multiprocessing as mp

from utils_env import rand_float, rand_int, quatFromAxisAngle, quaternion_multuply
from data_generation.utils import add_table, set_light, set_camera
from data_generation.utils import init_multiview_camera, get_camera_intrinsics, get_camera_extrinsics
from data_generation.utils import fps_with_idx, randomize_pos, render

camera_view = 4

def data_gen_sweeping(info):
    epi_start_time = time.time()
    # info
    debug = info['debug']
    data_root_dir = info['data_root_dir']
    headless = info['headless']
    
    epi = info['epi']
    n_time_step = info['n_time_step']
    n_push = info['n_push']
    
    with_dustpan = info['with_dustpan']
    
    num_sample_points = info['num_sample_points']
    
    # create folder
    folder_dir = os.path.join(data_root_dir, 'granular_sweeping')
    os.system('mkdir -p ' + folder_dir)
    
    epi_dir = os.path.join(folder_dir, "episode_%d" % epi)
    os.system("mkdir -p %s" % epi_dir)
    
    pyflex.init(headless)
    np.random.seed(epi)
    ## set scene
    radius = 0.03
    
    num_granular_ft_x = rand_float(5, 10)
    num_granular_ft_y = np.random.choice([2, 3])
    num_granular_ft_z = rand_float(5, 10)
    num_granular_ft = [num_granular_ft_x, num_granular_ft_y, num_granular_ft_z] 
    num_granular = int(num_granular_ft_x * num_granular_ft_y * num_granular_ft_z)
    
    granular_scale = rand_float(0.1, 0.2)
    pos_granular = [rand_float(-1.5, 0), 1., rand_float(-1., 0.)]
    granular_dis = rand_float(0.1, 0.3)

    draw_mesh = 0
    
    shapeCollisionMargin = 0.01
    collisionDistance = 0.03
    
    dynamic_friction = rand_float(0.2, 0.9)
    granular_mass = rand_float(0.1, 10.)

    scene_params = np.array([radius, *num_granular_ft, granular_scale, *pos_granular, granular_dis, 
                            draw_mesh, shapeCollisionMargin, collisionDistance, dynamic_friction,
                            granular_mass])

    temp = np.array([0])
    pyflex.set_scene(35, scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0)
    
    ## set env
    ## add table
    table_height = 0.5
    table_side = 5.
    table_shape_states = add_table(table_height, table_side)
    
    ## add sponge
    sponge_choice = np.random.choice([1, 2]) # 1: sponge, 2: sponge_2, 3: chef_knife
    if sponge_choice == 1:
        sponge_scale = 0.15 #rand_float(0.1, 0.15)
        sponge_pos_y = table_height+0.3
        sponge_pos_origin = randomize_pos(sponge_pos_y)

        sponge_quat_axis = np.array([1., 0., 0.])
        angle = 90.
        sponge_quat_origin = quatFromAxisAngle(sponge_quat_axis, np.deg2rad(angle))
        sponge_quat = sponge_quat_origin.copy()
        
        sponge_color = np.array([204/255, 102/255, 0.])
        pyflex.add_mesh('assets/mesh/sponge.obj', sponge_scale, 0, sponge_color, 
                        sponge_pos_origin, sponge_quat_origin, False)
    elif sponge_choice == 2:
        sponge_scale = 10. #rand_float(8., 12.) 
        sponge_pos_y = table_height + 0.2
        sponge_pos_origin = randomize_pos(sponge_pos_y)
        
        sponge_quat_axis = np.array([0., 0., 1.])
        angle = 0.
        sponge_quat_origin = quatFromAxisAngle(sponge_quat_axis, np.deg2rad(angle))
        
        sponge_color = np.array([204/255, 102/255, 0.])
        pyflex.add_mesh('assets/mesh/sponge_2.obj', sponge_scale, 0, sponge_color, 
                        sponge_pos_origin, sponge_quat_origin, False)
    elif sponge_choice == 3:
        sponge_scale = 9.
        sponge_pos_y = table_height+0.3
        sponge_pos_origin = randomize_pos(sponge_pos_y)

        sponge_quat_axis = np.array([0., 0., 1.])
        angle = 90.
        sponge_quat_origin = quatFromAxisAngle(sponge_quat_axis, np.deg2rad(angle))
        
        sponge_color = np.array([160/255, 160/255, 160/255])
        pyflex.add_mesh('assets/mesh/chef_knife.obj', sponge_scale, 0, sponge_color, 
                        sponge_pos_origin, sponge_quat_origin, False)
    
    sponge_pos_prev = sponge_pos_origin
    sponge_quat_prev = sponge_quat_origin
        
    ## add dustpan
    if with_dustpan:
        obj_shape_states = np.zeros((1, 14))
        dustpan_scale = 1.5
        dustpan_pos = np.array([-2.0, table_height+0.45, 0.5])
        dustpan_quat = quatFromAxisAngle(np.array([0., 1., 0.]), np.deg2rad(90.))
        dustpan_color = np.array([204/255, 204/255, 1.])
        pyflex.add_mesh('assets/mesh/dustpan.obj',dustpan_scale, 0, 
                    dustpan_color,dustpan_pos,dustpan_quat, False)
        obj_shape_states[0] = np.concatenate([dustpan_pos,dustpan_pos,dustpan_quat,dustpan_quat])

    ## Light and camera setting
    screenHeight, screenWidth = 720, 720
    cam_dis, cam_height = 6., 10.
    set_light(screenHeight, screenWidth)
    set_camera(cam_dis, cam_height, camera_view)
    camPos_list, camAngle_list, cam_intrinsic_params, cam_extrinsic_matrix = init_multiview_camera(cam_dis, cam_height)
            
    pyflex.step()
    
    ## update the shape states for each time step
    count = 0
    step_list = []
    particle_pos_list = []
    tool_pos_list = []
    tool_quat_list = []
    
    n_stay_still = 100
    n_up = n_time_step - 50
    speed = 0.01
    
    for p in range(n_push):
        particle_pos = pyflex.get_positions().reshape(-1, 4)
        num_particle = particle_pos.shape[0]
        # random pick one particle
        pick_id = rand_int(0, num_particle)
        pick_pos = particle_pos[pick_id, :3]
        
        if p == 0:
            sponge_pos = sponge_pos_origin.copy()
        else:
            sponge_pos = randomize_pos(sponge_pos_y)
        
        # ending position based on the start position and pick_pos
        sponge_pos_end = sponge_pos.copy()
        sponge_pos_end[0] = pick_pos[0] 
        sponge_pos_end[2] = pick_pos[2] 
        
        # initial start orientation
        sponge_quat_axis = np.array([0., 1., 0.])
        # angle = np.rad2deg(np.arctan2(sponge_pos_z, sponge_pos_x))
        angle = np.random.randint(0, 180)
        sponge_quat_t = quatFromAxisAngle(sponge_quat_axis, np.deg2rad(angle))
        sponge_quat = quaternion_multuply(sponge_quat_t, sponge_quat_origin)
        
        # save info
        step_list.append(count)
        
        for i in range(n_time_step):
            
            # downsample the point positions
            if p == 0 and i == n_stay_still:
                particle_pos = pyflex.get_positions().reshape(-1, 4)
                sampled_particle_pos, sampled_idx = fps_with_idx(particle_pos[:, :3], num_sample_points)
            
            # move the tool
            if n_stay_still < i < n_up:
                # move sponge to the ending position
                sponge_pos = sponge_pos + (sponge_pos_end - sponge_pos) * speed
            elif i >= n_up:
                sponge_pos[1] += speed
                sponge_pos[1] = np.clip(sponge_pos[1], sponge_pos_y, table_height + 0.8)
                    
            # set shape states
            shape_states = np.zeros((3, 14))
            shape_states[0] = table_shape_states
            
            # set shape state for table
            shape_states[0] = table_shape_states
            
            # set shape state for sponge
            shape_states[1, :3] = sponge_pos
            shape_states[1, 3:6] = sponge_pos_prev
            shape_states[1, 6:10] = sponge_quat
            shape_states[1, 10:] = sponge_quat_prev
            
            sponge_pos_prev = sponge_pos
            sponge_quat_prev = sponge_quat
            
            # set shape state fordustpan
            if with_dustpan:
                shape_states[2] = obj_shape_states[0]
            
            pyflex.set_shape_states(shape_states)
            
            if not debug and i % 2 == 0 and n_stay_still < i < n_up:
                num_cam = len(camPos_list)
                for j in range(num_cam):
                    pyflex.set_camPos(camPos_list[j])
                    pyflex.set_camAngle(camAngle_list[j])
                    
                    # create dir with cameras
                    cam_dir = os.path.join(epi_dir, 'camera_%d' % (j))
                    os.system('mkdir -p ' + cam_dir)
                    
                    # save camera params
                    if p == 0 and i == 0:
                        cam_intrinsic_params[j] = get_camera_intrinsics(screenHeight, screenWidth)
                        cam_extrinsic_matrix[j] = get_camera_extrinsics()
                    
                    # save rgb images
                    img = render(screenHeight, screenWidth)
                    cv2.imwrite(os.path.join(cam_dir, '%d_color.jpg' % count), img[:, :, :3][..., ::-1])
                    cv2.imwrite(os.path.join(cam_dir, '%d_depth.png' % count), (img[:, :, -1]*1000).astype(np.uint16))
                    if j == 0:
                        # save sampled particle positions
                        particle_pos = pyflex.get_positions().reshape(-1, 4)
                        sampled_pos = particle_pos[sampled_idx]
                        particle_pos_list.append(sampled_pos)
                        # save tool pos
                        tool_pos_list.append(sponge_pos.copy())
                        tool_quat_list.append(sponge_quat.copy())
            
                count += 1
            
            pyflex.step()

    # save property params
    property_params = {
        'particle_radius': radius,
        'num_particles': len(sampled_idx),
        'granular_scale': granular_scale,
        'num_granular': num_granular,
        'distribution_r': granular_dis,
        'dynamic_friction': dynamic_friction,
        'granular_mass': granular_mass,
        'tool': float(sponge_choice),
        'dustpan': float(with_dustpan),
    }
    print(property_params)
    
    # save info
    if not debug:
        np.save(os.path.join(epi_dir, 'cam_intrinsic_params.npy'), cam_intrinsic_params)
        np.save(os.path.join(epi_dir, 'cam_extrinsic_matrix.npy'), cam_extrinsic_matrix)
        np.save(os.path.join(epi_dir, 'steps.npy'), np.array(step_list))
        np.save(os.path.join(epi_dir, 'particle_pos.npy'), np.array(particle_pos_list))
        np.save(os.path.join(epi_dir, 'tool_pos.npy'), np.array(tool_pos_list))
        np.save(os.path.join(epi_dir, 'tool_quat.npy'), np.array(tool_quat_list))
        with open(os.path.join(epi_dir, 'property_params.json'), 'w') as fp:
            json.dump(property_params, fp)
    
    pyflex.clean()
    
    print(f'done episode {epi}!!!! total time: {time.time() - epi_start_time}')

### data generation for scooping
epi_num = np.random.randint(1000)
info = {
    "epi": epi_num,
    "n_time_step": 500,
    "n_push": 5,
    "num_sample_points": 2000,
    "with_dustpan": True,
    "headless": False,
    "data_root_dir": "data_dense",
    "debug": True,
}

data_gen_sweeping(info)

## multiprocessing
# n_worker = 25
# n_episode = 25
# end_base = int(1000 / 5)
# bases = [i for i in range(0, end_base, n_episode)]
# print(bases)
# print(len(bases))
# for base in bases:
#     print("base:", base)
#     infos=[]
#     for i in range(n_worker):
#         info = {
#             "epi": base+i*n_episode//n_worker,
#             "n_time_step": 500,
#             "n_push": 5,
#             "num_sample_points": 2000,
#             "with_dustpan": False,
#             "headless": True,
#             "data_root_dir": "/media/baoyu/sumsung", #"data_dense", 
#             "debug": False,
#         }
#         infos.append(info)
#     pool = mp.Pool(processes=n_worker)
#     pool.map(data_gen_sweeping, infos)
