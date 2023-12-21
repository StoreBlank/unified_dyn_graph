import os
import numpy as np
import pyflex
import time

from utils_env import rand_float, quatFromAxisAngle, degs_to_quat
from data_generation.utils import set_camera

pyflex.init(False)

radius = 0.02
bowl_pos = [-0.3, 0.5, -0.3]
bowl_mass = 1e100
bowl_scale = 1.5

num_granular_ft = [5, 1, 5] # low 5, medium 10, high 20
granular_scale = 0.15
pos_granular = [0., 1., 0.]
granular_dis = 0.

spoon_scale = 1.
spoon_mass = 10.
spoon_rotation = 0.1

draw_mesh = 1

shapeCollisionMargin = 0.05
collisionDistance = 0.03 #granular_scale * 0.1

dynamic_friction = 0.3 
granular_mass = 0.1

scene_params = np.array([radius, *num_granular_ft, granular_scale, *pos_granular, granular_dis, 
                        draw_mesh, shapeCollisionMargin, collisionDistance, dynamic_friction, 
                        granular_mass])

temp = np.array([0])
pyflex.set_scene(35, scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0)


# add table
table_height = 0.5
table_width = 3.5
table_length = 4.5
halfEdge = np.array([table_width, table_height, table_length])
center = np.array([0.0, 0.0, 0.0])
quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
hideShape = 0
color = np.ones(3) * (160. / 255.)
pyflex.add_box(halfEdge, center, quats, hideShape, color)
table_shape_states = np.concatenate([center, center, quats, quats])

## Light setting
pyflex.set_screenWidth(720)
pyflex.set_screenHeight(720)
pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
pyflex.set_light_fov(70.)

# Camera setting
camera_view = 4
cam_dis, cam_height = 6., 10.
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

pyflex.step()

# update the shape states for each time step
for i in range(2800):
    
    # set shape states
    shape_states = np.zeros((1, 14))
    shape_states[0] = table_shape_states
    
    # set shape state for table
    shape_states[0] = table_shape_states
    
    pyflex.set_shape_states(shape_states)
    
    pyflex.step()
    

pyflex.clean()