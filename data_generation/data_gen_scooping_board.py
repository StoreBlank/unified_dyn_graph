import os
import numpy as np
import pyflex
import time
import cv2
import json
import multiprocessing as mp

from utils_env import rand_float, rand_int, quatFromAxisAngle, quaternion_multuply
from data_generation.utils import add_table, set_light, set_camera
from data_generation.utils import init_multiview_camera, get_camera_intrinsics, get_camera_extrinsics
from data_generation.utils import fps_with_idx, randomize_pos, render

camera_view = 1

def data_gen_scooping(info):
    epi_start_time = time.time()
    # info
    debug = info['debug']
    data_root_dir = info['data_root_dir']
    headless = info['headless']
    
    epi = info['epi']
    n_time_step = info['n_time_step']
    n_scoop = info['n_scoop']
    
    num_sample_points = info['num_sample_points']
    
    # create folder
    folder_dir = os.path.join(data_root_dir, 'granular_scooping_board')
    os.system('mkdir -p ' + folder_dir)
    
    epi_dir = os.path.join(folder_dir, "episode_%d" % epi)
    os.system("mkdir -p %s" % epi_dir)
    
    pyflex.init(headless)
    np.random.seed(epi)
    ## set scene
    radius = 0.03
    
    num_granular_ft_x = rand_float(3, 8)
    num_granular_ft_y = rand_int(2, 3)
    num_granular_ft_z = rand_float(3, 8)
    num_granular_ft = [num_granular_ft_x, num_granular_ft_y, num_granular_ft_z] 
    num_granular = int(num_granular_ft_x * num_granular_ft_y * num_granular_ft_z)
    
    granular_scale = rand_float(0.1, 0.2)
    pos_granular = [-0.5, 1., 0.]
    granular_dis = 0.

    draw_mesh = 0
    
    shapeCollisionMargin = 0.05
    collisionDistance = 0.03 #granular_scale * 0.1
    
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
    
    ## add spatula
    spoon_scale = 0.4
    spoon_pos_y = table_height+0.95
    scooping_list = np.array([
        [5., spoon_pos_y, 0.5, 0.],
        [5., spoon_pos_y, -5., 45.],
        [0., spoon_pos_y, -5., 90.],
        [-5., spoon_pos_y, -5., 135.],
        [-5., spoon_pos_y, 0.5, 180.],
        [-5., spoon_pos_y, 5., -135.],
        [0., spoon_pos_y, 5., -90.],
        [5., spoon_pos_y, 5., -45.]
    ])
    l = 0 #np.random.randint(0, scooping_list.shape[0])
    spoon_pos_origin = scooping_list[l, :3]
    spoon_pos = spoon_pos_origin
    
    spoon_quat_axis_origin = np.array([0., 0., 1.])
    angle_origin = 30.
    spoon_quat_origin = quatFromAxisAngle(spoon_quat_axis_origin, np.deg2rad(angle_origin))
    
    spoon_quat_axis = np.array([0., 1., 0.])
    angle = scooping_list[l, -1] 
    spoon_quat_2 = quatFromAxisAngle(spoon_quat_axis, np.deg2rad(angle))
    spoon_quat = quaternion_multuply(spoon_quat_2, spoon_quat_origin)
    
    spoon_color = np.array([204/255, 204/255, 1.])
    pyflex.add_mesh('assets/mesh/spatula.obj', spoon_scale, 0,
                    spoon_color, spoon_pos, spoon_quat, False)
    spoon_pos_prev = spoon_pos
    spoon_quat_prev = spoon_quat
    
    ## add board
    halfEdge = np.array([0.5, 0.8, 1.0])
    center = np.array([-2.3, table_height, 0.5])
    quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
    hideShape = 0
    color = np.array([153/255, 76/255, 0.])
    pyflex.add_box(halfEdge, center, quats, hideShape, color)
    board_shape_states = np.concatenate([center, center, quats, quats])

    ## Light and camera setting
    screenHeight, screenWidth = 720, 720
    cam_dis, cam_height = 6., 10.
    set_light(screenHeight, screenWidth)
    set_camera(cam_dis, cam_height, camera_view)
    camPos_list, camAngle_list, cam_intrinsic_params, cam_extrinsic_matrix = init_multiview_camera(cam_dis, cam_height)
            
    pyflex.step()

    ## update the shape states for each time step
    count = 0
    angle_drop = 0.
    
    step_list = []
    particle_pos_list = []
    tool_pos_list = []
    tool_quat_list = []
    
    n_stay_still = 90
    n_move = 300
    n_up = 500
    speed= 0.01 #rand_float(0.03, 0.04) #rand_float(0.01, 0.02)
    
    # if np.random.randint(0, 3) == 0:
    #     spoon_quat_axis_drop = np.array([0., 0., 1.])
    #     n_drop = 800
    # elif np.random.randint(0, 3) == 1:
    #     spoon_quat_axis_drop = np.array([1., 0., 0.])
    #     n_drop = 900
    # elif np.random.randint(0, 3) == 2:
    #     spoon_quat_axis_drop = np.array([-1., 0., 0.])
    #     n_drop = 900
    
    # spoon_quat_axis_drop = np.array([1., 0., 0.])
    # n_drop = 800
    
    for p in range(n_scoop):
        particle_pos = pyflex.get_positions().reshape(-1, 4)
        num_particle = particle_pos.shape[0]
        # random pick one particle
        pick_id = rand_int(0, num_particle)
        pick_pos = particle_pos[pick_id, :3]
        
        if p == 0:
            spoon_pos = spoon_pos_origin
            spoon_quat_t = spoon_quat
        else:
            l = np.random.randint(0, scooping_list.shape[0])
            spoon_pos = scooping_list[l, :3]
            spoon_quat_axis = np.array([0., 1., 0.])
            angle = scooping_list[l, -1]  
            spoon_quat_t = quatFromAxisAngle(spoon_quat_axis, np.deg2rad(angle))
            spoon_quat_t = quaternion_multuply(spoon_quat_t, spoon_quat_origin)
        spoon_quat = spoon_quat_t
        
        # ending position based on the start position and pick position
        spoon_pos_end = spoon_pos.copy()
        spoon_pos_end[0] = center[0] + 2.2
        spoon_pos_end[2] = pick_pos[2]
        
        # save info
        step_list.append(count)
        
        for i in range(n_time_step):
            
            # downsample the point positions
            if p == 0 and i == n_stay_still:
                particle_pos = pyflex.get_positions().reshape(-1, 4)
                sampled_particle_pos, sampled_idx = fps_with_idx(particle_pos[:, :3], num_sample_points)
            
            # scooping
            if n_stay_still < i < n_move:
                spoon_pos = spoon_pos + (spoon_pos_end - spoon_pos) * speed
            if n_move < i < n_up:
                spoon_pos[1] += 0.01
                spoon_pos[1] = np.clip(spoon_pos[1], spoon_pos_y, table_height + 4.0)
            
            # set shape states
            shape_states = np.zeros((3, 14))
            shape_states[0] = table_shape_states
            
            # set shape state for table
            shape_states[0] = table_shape_states
            
            # set shape state for spoon
            shape_states[1, :3] = spoon_pos
            shape_states[1, 3:6] = spoon_pos_prev
            shape_states[1, 6:10] = spoon_quat
            shape_states[1, 10:] = spoon_quat_prev
            
            spoon_pos_prev = spoon_pos
            spoon_quat_prev = spoon_quat
            
            # set shape state for board
            shape_states[2] = board_shape_states
            
            
            pyflex.set_shape_states(shape_states)
            
            if not debug and i % 2 == 0:
                num_cam = len(camPos_list)
                for j in range(1):
                    pyflex.set_camPos(camPos_list[j])
                    pyflex.set_camAngle(camAngle_list[j])
                    
                    # create dir with cameras
                    cam_dir = os.path.join(epi_dir, 'camera_%d' % (j))
                    os.system('mkdir -p ' + cam_dir)
                    
                    # save camera params
                    
                    # save rgb images
                    img = render(screenHeight, screenWidth)
                    cv2.imwrite(os.path.join(cam_dir, '%d_color.jpg' % count), img[:, :, :3][..., ::-1])
            
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
        'speed': speed,
    }
    print(property_params)

    pyflex.clean()
    
    print(f'done episode {epi}!!!! total time: {time.time() - epi_start_time}')

### data generation for scooping
# info = {
#     'epi': 0,
#     'n_time_step': 800,
#     'n_scoop': 1,
#     'num_sample_points': 2000,
#     "headless": False,
#     "data_root_dir": "/mnt/sda",
#     "debug": True,
# }
# data_gen_scooping(info)

## multiprocessing
n_worker = 10
n_episode = 10
# end_base = int(1000 / 5)
# bases = [i for i in range(0, end_base, n_episode)]
bases = [0]
for base in bases:
    infos = []
    for i in range(n_worker):
        info = {
            'epi': base+i*n_episode//n_worker,
            'n_time_step': 800,
            'n_scoop': 1,
            'num_sample_points': 2000,
            "headless": True,
            "data_root_dir": "/mnt/sda/data",
            "debug": False,
        }
        infos.append(info)
    pool = mp.Pool(processes=n_worker)
    pool.map(data_gen_scooping, infos)