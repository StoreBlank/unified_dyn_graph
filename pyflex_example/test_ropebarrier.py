import os
import numpy as np
import pyflex
import time
from scipy.spatial.transform import Rotation

from utils_env import rand_float, quatFromAxisAngle, degs_to_quat
from action.action_space import Picker

camera_view = 1

pyflex.init(False)

### set scene
radius = 0.04 #0.03

length = 3.0 #rand_float(0.5, 2.5)
thickness = 3.0 #rand_float(1., 2.5)
scale = np.array([length, thickness, thickness]) * 50 # length, extension, thickness
cluster_spacing = 2. #rand_float(2, 8) # change the stiffness of the rope
dynamicFriction = 0.3 #rand_float(0.1, 0.7)

trans = [0., 2., 3.]

z_rotation = 0. #rand_float(70, 80)
y_rotation = 90. #np.random.choice([0, 30, 45, 60])
rot = Rotation.from_euler('xyz', [0, y_rotation, z_rotation], degrees=True)
rotate = rot.as_quat()

cluster_radius = 0.
cluster_stiffness = 0.2

link_radius = 0. 
link_stiffness = 1.

global_stiffness = 0.

surface_sampling = 0.
volume_sampling = 4.

skinning_falloff = 5.
skinning_max_dist = 100.

cluster_plastic_threshold = 0.
cluster_plastic_creep = 0.

particleFriction = 0.25

draw_mesh = 1

relaxtion_factor = 1.
collisionDistance = radius * 0.5

scene_params = np.array([*scale, *trans, radius, 
                                cluster_spacing, cluster_radius, cluster_stiffness,
                                link_radius, link_stiffness, global_stiffness,
                                surface_sampling, volume_sampling, skinning_falloff, skinning_max_dist,
                                cluster_plastic_threshold, cluster_plastic_creep,
                                dynamicFriction, particleFriction, draw_mesh, relaxtion_factor, 
                                *rotate, collisionDistance])

temp = np.array([0])
pyflex.set_scene(26, scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0) 

num_particles = pyflex.get_n_particles()
print('num_particles', num_particles)

### set env
## add table
table_height = 0.5
halfEdge = np.array([4., table_height, 4.])
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

## camera setting
cam_dis = 6.
cam_height = 8.
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

## set picker
num_picker = 1
picker_radius = 0.025
action_tool = Picker(num_picker, picker_radius=picker_radius, picker_threshold=0.005, 
            particle_radius=0.025, picker_low=(-0.35, 0., -0.35), picker_high=(0.35, 0.3, 0.35))
action_space = action_tool.action_space
action = action_space.sample()
action_tool.reset([0., 0., 0.])
# action_tool.visualize_picker_boundary()

shape_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)
num_shape_pos = shape_pos.shape[0]
print('num_shape_pos', num_shape_pos)
print('shape_pos', shape_pos[:, :3])

# update the shape states for each time step
for i in range(500):
    
    
        
    # set shape states
    shape_states = np.zeros((1, 14))
    shape_states[0] = table_shape_states
    
    pyflex.set_shape_states(shape_states)
    
    pyflex.step()
    

pyflex.clean()