import os
import numpy as np
import pyflex
import time
import cv2

from utils_env import rand_float, rand_int, quatFromAxisAngle, quaternion_multuply
from data_generation.utils import init_multiview_camera, render

camera_view = 4

def data_gen_sweeping(info):
    # info
    debug = info['debug']
    data_root_dir = info['data_root_dir']
    headless = info['headless']
    
    n_rollout = info['n_rollout']
    n_time_step = info['n_time_step']
    n_push = info['n_push']
    
    with_dustpan = info['with_dustpan']
    
    # create folder
    folder_dir = os.path.join(data_root_dir, 'granular_sweeping')
    os.system('mkdir -p ' + folder_dir)
    
    pyflex.init(headless)
    
    ## set scene
    radius = 0.03
    
    num_granular_ft_x = rand_float(5, 10)
    num_granular_ft_y = np.random.choice([2, 3])
    num_granular_ft_z = rand_float(5, 10)
    num_granular_ft = [num_granular_ft_x, num_granular_ft_y, num_granular_ft_z] 
    granular_scale = 0.1
    pos_granular = [-1.5, 1., -1.]
    granular_dis = rand_float(0., 0.3)

    draw_mesh = 0
    
    shapeCollisionMargin = 0.01
    collisionDistance = 0.03
    dynamic_friction = 0.3

    scene_params = np.array([radius, *num_granular_ft, granular_scale, *pos_granular, granular_dis, 
                            draw_mesh, shapeCollisionMargin, collisionDistance, dynamic_friction])

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
    sponge_scale = 0.15
    sponge_pos_x = 3.0
    sponge_pos_y = table_height+0.3
    sponge_pos_z = 0.5
    sponge_pos = np.array([sponge_pos_x, sponge_pos_y, sponge_pos_z])

    sponge_quat_axis = np.array([1., 0., 0.])
    angle = 90.
    sponge_quat_origin = quatFromAxisAngle(sponge_quat_axis, np.deg2rad(angle))
    sponge_quat = sponge_quat_origin.copy()
    
    sponge_color = np.array([204/255, 102/255, 0.])
    pyflex.add_mesh('assets/mesh/sponge.obj', sponge_scale, 0,
                    sponge_color, sponge_pos, sponge_quat, False)
    sponge_pos_prev = sponge_pos
    sponge_quat_prev = sponge_quat
    
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

    pyflex.step()
    
    ## update the shape states for each time step
    count = 0
    n_stay_still = 100
    n_up = n_time_step - 50
    speed = 0.01
    for p in range(n_push):
        particle_pos = pyflex.get_positions().reshape(-1, 4) #[x, y, z, inv_mass]
        num_particle = particle_pos.shape[0]
        # random pick one particle
        pick_id = rand_int(0, num_particle)
        pick_pos = particle_pos[pick_id, :3]
        
        # initial start position
        x_range_min, x_range_max = 0., 4. # range for x if not in (2,3)
        z_range_min, z_range_max = 0., 4. # range for z if not in (2,3)
        range_min, range_max = 3., 4.
        # randomly decide whether x or z will be in the range
        if np.random.choice(['x', 'z']) == 'x':
            sponge_pos_x = rand_float(range_min, range_max) * np.random.choice([-1., 1.])
            sponge_pos_z = rand_float(z_range_min, z_range_max) * np.random.choice([-1., 1.])
        else:
            sponge_pos_x = rand_float(x_range_min, x_range_max) * np.random.choice([-1., 1.])
            sponge_pos_z = rand_float(range_min, range_max) * np.random.choice([-1., 1.])
        
        
        sponge_pos_y = table_height + 0.3
        sponge_pos = np.array([sponge_pos_x, sponge_pos_y, sponge_pos_z])
        
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
        
        for i in range(n_time_step):
            
            if n_stay_still < i < n_up:
                # move sponge to the ending position
                sponge_pos[1] = table_height + 0.3
                sponge_pos = sponge_pos + (sponge_pos_end - sponge_pos) * speed
            elif i >= n_up:
                sponge_pos[1] += speed
                sponge_pos[1] = np.clip(sponge_pos[1], table_height + 0.3, table_height + 0.8)
                    
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
            
            if not debug and i % 2 == 0:
                for j in range(1):
                    pyflex.set_camPos(camPos_list[j])
                    pyflex.set_camAngle(camAngle_list[j])
                    
                    # create dir with cameras
                    cam_dir = os.path.join(folder_dir, 'camera_%d' % (j))
                    os.system('mkdir -p ' + cam_dir)
                    
                    # save rgb images
                    img = render(screenHeight, screebWidth)
                    cv2.imwrite(os.path.join(cam_dir, '%d_color.jpg' % count), img[:, :, :3][..., ::-1])
            
                count += 1
            
            pyflex.step()

    pyflex.clean()

### data generation for scooping
info = {
    "n_rollout": 1,
    "n_time_step": 500,
    "n_push": 5,
    "with_dustpan": False,
    "headless": False,
    "data_root_dir": "data_dense",
    "debug": True,
}

data_gen_sweeping(info)