import os
import time
import numpy as np
import pyflex
import cv2
import json
import multiprocessing as mp

from utils_env import rand_float, rand_int, quatFromAxisAngle, quaternion_multuply
from data_generation.utils import init_multiview_camera, render, randomize_pos

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
    pos_granular = [-1.5, 1., -1.]
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
    halfEdge = np.array([4., table_height, 4.])
    center = np.array([0.0, 0.0, 0.0])
    quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
    hideShape = 0
    color = np.ones(3) * (160. / 255.)
    pyflex.add_box(halfEdge, center, quats, hideShape, color)
    table_shape_states = np.concatenate([center, center, quats, quats])
    # print('table_shape_states', table_shape_states.shape) # (14,)
    
    ## add sponge
    sponge_choice = np.random.choice([1, 2, 3]) # 1: sponge, 2: sponge_2, 3: chef_knife
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

    ## Light setting
    screebWidth, screenHeight = 720, 720
    pyflex.set_screenWidth(screebWidth)
    pyflex.set_screenHeight(screenHeight)
    pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
    pyflex.set_light_fov(70.)
    
    ## camera setting for view
    cam_dis = 6.
    cam_height = 10.
    if camera_view == 1:
        camPos = np.array([cam_dis, cam_height, cam_dis])
        camAngle = np.array([np.deg2rad(45.), -np.deg2rad(45.), 0.])
    elif camera_view == 2:
        camPos = np.array([cam_dis, cam_height, -cam_dis])
        camAngle = np.array([np.deg2rad(45.+90.), -np.deg2rad(45.), 0.])
    elif camera_view == 3:
        camPos = np.array([-cam_dis, cam_height, -cam_dis])
        camAngle = np.array([np.deg2rad(45.+180.), -np.deg2rad(45.), 0.])
    elif camera_view == 4:
        camPos = np.array([-cam_dis, cam_height, cam_dis])
        camAngle = np.array([np.deg2rad(45.+270.), -np.deg2rad(45.), 0.])
    
    pyflex.set_camPos(camPos)
    pyflex.set_camAngle(camAngle)
    
    camPos_list, camAngle_list, cam_intrinsic_params, cam_extrinsic_matrix = init_multiview_camera(cam_dis, cam_height)
    
    ## save property params
    particle_pos = pyflex.get_positions().reshape(-1, 4) 
    num_particle = particle_pos.shape[0]
    property_params = {
        'particle_radius': radius,
        'num_particles': num_particle,
        'granular_scale': granular_scale,
        'num_granular': num_granular,
        'distribution_r': granular_dis,
        'dynamic_friction': dynamic_friction,
        'granular_mass': granular_mass,
        'tool': float(sponge_choice),
        'dustpan': float(with_dustpan),
    }
    print(property_params)
    
    # save property params in the epi folder
    if not debug:
        with open(os.path.join(epi_dir, 'property_params.json'), 'w') as fp:
            json.dump(property_params, fp)
            
    pyflex.step()
    
    ## update the shape states for each time step
    count = 0
    step_list = []
    particle_pos_list = []
    tool_pos_list = []
    
    n_stay_still = 100
    n_up = n_time_step - 50
    speed = 0.01
    
    for p in range(n_push):
        particle_pos = pyflex.get_positions().reshape(-1, 4)
        # random pick one particle
        pick_id = rand_int(0, num_particle)
        pick_pos = particle_pos[pick_id, :3]
        
        if p == 0:
            sponge_pos = sponge_pos_origin .copy()
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
                for j in range(1):
                    pyflex.set_camPos(camPos_list[j])
                    pyflex.set_camAngle(camAngle_list[j])
                    
                    # create dir with cameras
                    cam_dir = os.path.join(epi_dir, 'camera_%d' % (j))
                    os.system('mkdir -p ' + cam_dir)
                    
                    # save rgb images
                    img = render(screenHeight, screebWidth)
                    cv2.imwrite(os.path.join(cam_dir, '%d_color.jpg' % count), img[:, :, :3][..., ::-1])
            
                count += 1
            
            pyflex.step()

    pyflex.clean()
    
    print(f'done episode {epi}!!!! total time: {time.time() - epi_start_time}')

### data generation for scooping

# info = {
#     "epi": 0,
#     "n_time_step": 500,
#     "n_push": 5,
#     "with_dustpan": False,
#     "headless": False,
#     "data_root_dir": "data_dense",
#     "debug": False,
# }

# data_gen_sweeping(info)

## multiprocessing
n_worker = 10
n_episode = 10
bases = [0]
for base in bases:
    print("base:", base)
    infos=[]
    for i in range(n_worker):
        info = {
            "epi": base+i*n_episode//n_worker,
            "n_time_step": 500,
            "n_push": 5,
            "with_dustpan": False,
            "headless": True,
            "data_root_dir": "data_dense",
            "debug": False,
        }
        infos.append(info)
    pool = mp.Pool(processes=n_worker)
    pool.map(data_gen_sweeping, infos)
