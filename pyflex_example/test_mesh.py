import os
import numpy as np
import pyflex
import time

def quatFromAxisAngle(axis, angle):
    axis /= np.linalg.norm(axis)

    half = angle * 0.5
    w = np.cos(half)

    sin_theta_over_two = np.sin(half)
    axis *= sin_theta_over_two

    quat = np.array([axis[0], axis[1], axis[2], w])

    return quat


time_step = 500 # 120
# des_dir = 'test_FluidFall'
# os.system('mkdir -p ' + des_dir)

pyflex.init(False)

global_scale = 10

scale = 0.2 * global_scale / 8.0
x = -0.9 * global_scale / 8.0
y = 8.
z = -0.9 * global_scale / 8.0
staticFriction = 1.0
dynamicFriction = 1.0
draw_skin = 1.
num_capsule = 100 # [200, 1000]
slices = 10
segments = 20
scene_params = np.array([scale, x, y, z, staticFriction, dynamicFriction, draw_skin, num_capsule, slices, segments])

temp = np.array([0])
pyflex.set_scene(21, scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0) 



# add box
wall_height = 0.5
halfEdge = np.array([4., wall_height, 4.])
center = np.array([0.0, 0.0, 0.0])
quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
hideShape = 0
color = np.ones(3) * (160. / 255.)
pyflex.add_box(halfEdge, center, quats, hideShape, color)
table_shape_states = np.concatenate([center, center, quats, quats])
# print('table_shape_states', table_shape_states.shape) # (14,)

# add mesh
# void pyflex_add_mesh(const char *s, float scaling, int hideShape, py::array_t<float> color, 
# py::array_t<float> translation, py::array_t<float> rotation, bool texture=false) 
trans = np.array([0., 1.2, 0.])
quat = quatFromAxisAngle(np.array([1., 0., 0.]), np.deg2rad(270.))
pyflex.add_mesh('/home/baoyu/2023/unified_dyn_graph/assets/mesh/bowl_2.obj', 20., 0, 
                np.ones(3), trans, quat, False)
bowl_shape_states = np.concatenate([trans, trans, quat, quat])



shape_states = np.concatenate([table_shape_states, bowl_shape_states])
pyflex.set_shape_states(shape_states)

## Light setting
pyflex.set_screenWidth(720)
pyflex.set_screenHeight(720)
pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
pyflex.set_light_fov(70.)

# folder_dir = '../ptcl_data/capsule'
# os.system('mkdir -p ' + folder_dir)

# des_dir = folder_dir + '/view_0'
# os.system('mkdir -p ' + des_dir)

camPos = np.array([0., 10, 0.])
camAngle = np.array([0., -np.deg2rad(90.), 0.])

pyflex.set_camPos(camPos)
pyflex.set_camAngle(camAngle)

for i in range(time_step):
    # pyflex.step(capture=1, path=os.path.join(des_dir, 'render_%d.tga' % i))
    pyflex.step()

obs = pyflex.render(render_depth=True).reshape(720, 720, 5)
print('obs.shape', obs.shape)

# save obs and camera_params
# np.save(os.path.join(des_dir, 'obs.npy'), obs)

pyflex.clean()