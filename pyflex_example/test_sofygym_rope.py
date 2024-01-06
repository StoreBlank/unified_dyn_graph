import pyflex
import numpy as np

pyflex.init(False)
init_pos = [-0.5, 1., 0.]
stretchstiffness = 0.1
bendingstiffness = 0.1
radius = 0.05
segment = 100
mass = 0.5
scale = 10.0
draw_mesh = 0

scene_params = np.array([*init_pos, stretchstiffness, bendingstiffness, radius, segment, mass, scale, draw_mesh])

temp = np.array([0])
pyflex.set_scene(37, scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0) 

## set light
screenWidth, screenHeight = 720, 720
pyflex.set_screenWidth(screenWidth)
pyflex.set_screenHeight(screenHeight)
pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
pyflex.set_light_fov(70.)

## set camera
camPos = np.array([0., 6., 0.])
camAngle = np.array([0., -np.deg2rad(90.), 0.])
pyflex.set_camPos(camPos)
pyflex.set_camAngle(camAngle)

for _ in range(500):
    pyflex.step()