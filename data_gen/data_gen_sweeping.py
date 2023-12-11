import os
import numpy as np
import pyflex
import time
import cv2

from utils_env import rand_float, rand_int, quatFromAxisAngle

camera_view = 4

def init_multiview_camera(cam_dis = 3, cam_height = 4.5):
    camPos_list = []
    camAngle_list = []

    rad_list = np.deg2rad(np.array([0., 90., 180., 270.]) + 45.)
    cam_x_list = np.array([cam_dis, cam_dis, -cam_dis, -cam_dis])
    cam_z_list = np.array([cam_dis, -cam_dis, -cam_dis, cam_dis])

    for i in range(len(rad_list)):
        camPos_list.append(np.array([cam_x_list[i], cam_height, cam_z_list[i]]))
        camAngle_list.append(np.array([rad_list[i], -np.deg2rad(45.), 0.]))
    
    cam_intrinsic_params = np.zeros([len(camPos_list), 4]) # [fx, fy, cx, cy]
    cam_extrinsic_matrix = np.zeros([len(camPos_list), 4, 4]) # [R, t]
    
    return camPos_list, camAngle_list, cam_intrinsic_params, cam_extrinsic_matrix

def render(screenHeight, screenWidth, no_return=False):
    pyflex.step()
    if no_return:
        return
    else:
        return pyflex.render(render_depth=True).reshape(screenHeight, screenWidth, 5)

def data_gen_scooping(info):
    # info
    debug = info['debug']
    data_root_dir = info['data_root_dir']
    headless = info['headless']
    
    # create folder
    folder_dir = os.path.join(data_root_dir, 'granular_scooping')
    os.system('mkdir -p ' + folder_dir)
    
    ## set scene
    pyflex.init(headless)
     
    radius = 0.03
    
    num_granular_ft_x = 10
    num_granular_ft_y = 1
    num_granular_ft_z = 10
    num_granular_ft = [num_granular_ft_x, num_granular_ft_y, num_granular_ft_z] 
    granular_scale = 0.2
    pos_granular = [0., 1., 0.]
    granular_dis = 0.

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
    sponge_quat = quatFromAxisAngle(sponge_quat_axis, np.deg2rad(90.))
    sponge_color = np.array([204/255, 102/255, 0.])
    pyflex.add_mesh('assets/mesh/sponge.obj', sponge_scale, 0,
                    sponge_color, sponge_pos, sponge_quat, False)
    sponge_pos_prev = sponge_pos
    sponge_quat_prev = sponge_quat
    
    ## add dustpan
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
    
    ## camera setting
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
    for i in range(2000):
        angle = 30
        
        n_stay_still = 40
        n_first_move = 500
        n_first_up = 600
        
        n_second_still = 700
        n_second_move = 1200
        n_second_up = 1300
        
        n_third_still = 1400
        n_third_move = 1900
        
        if n_stay_still < i < n_first_move:
            # change sponge x position
            sponge_pos[0] -= 0.01
            sponge_pos[0] = np.clip(sponge_pos[0], -1, sponge_pos_x) 
        elif n_first_move < i < n_first_up:
            # change sponge y position
            sponge_pos[1] += 0.01
            sponge_pos[1] = np.clip(sponge_pos[1], sponge_pos_y, sponge_pos_y+0.2)
        
        elif n_first_up < i < n_second_still:
            # change sponge initial position
            sponge_pos_x = 3.0
            sponge_pos_z = -0.1
            sponge_pos = np.array([sponge_pos_x, sponge_pos_y, sponge_pos_z])
        elif n_first_move < i < n_second_move:
            # change sponge x position
            sponge_pos[0] -= 0.01
            sponge_pos[0] = np.clip(sponge_pos[0], -1, sponge_pos_x)   
        elif n_second_move < i < n_second_up:
            # change sponge y position
            sponge_pos[1] += 0.01
            sponge_pos[1] = np.clip(sponge_pos[1], sponge_pos_y, sponge_pos_y+0.2)
        
        elif n_second_up < i < n_third_still:
            # change sponge initial position
            sponge_pos_x = 3.0
            sponge_pos_z = 1.0
            sponge_pos = np.array([sponge_pos_x, sponge_pos_y, sponge_pos_z])   
        elif n_second_move < i < n_third_move:
            # change sponge x position
            sponge_pos[0] -= 0.01
            sponge_pos[0] = np.clip(sponge_pos[0], -1, sponge_pos_x)  
            
        
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
    "headless": False,
    "data_root_dir": "data_dense",
    "debug": True,
}

data_gen_scooping(info)