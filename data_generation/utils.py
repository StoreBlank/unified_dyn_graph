import numpy as np
import pyflex

from utils_env import rand_float, quatFromAxisAngle
from transformations import rotation_matrix, quaternion_from_matrix

    
def randomize_pos(init_y):
    # initial start position
    x_range_min, x_range_max = 0., 4. # range for x if not in (2,3)
    z_range_min, z_range_max = 0., 4. # range for z if not in (2,3)
    range_min, range_max = 4., 4.
    # randomly decide whether x or z will be in the range
    if np.random.choice(['x', 'z']) == 'x':
        pos_x = rand_float(range_min, range_max) * np.random.choice([-1., 1.])
        pos_z = rand_float(z_range_min, z_range_max) * np.random.choice([-1., 1.])
    else:
        pos_x = rand_float(x_range_min, x_range_max) * np.random.choice([-1., 1.])
        pos_z = rand_float(range_min, range_max) * np.random.choice([-1., 1.])
    
    pos_y = init_y
    pos = np.array([pos_x, pos_y, pos_z])
    return pos

def fps_with_idx(points, N):
    """
    Input:
        points: np.array() particle positions
        N: int sample number
    Output:
        points[farthest_pts_idx]: np.array() sampled points
        farthest_pts_idx: np.array() indices of the sampled points
    """
    if N > len(points):
        return points, np.arange(len(points))
    else:
        # start with the first point
        farthest_pts_idx = [0]
        distances = np.full(len(points), np.inf)
        
        for _ in range(1, N):
            last_point = points[farthest_pts_idx[-1]]
            new_distances = np.linalg.norm(points - last_point, axis=1)
            distances = np.minimum(distances, new_distances)
            farthest_pts_idx.append(np.argmax(distances))
            
        return points[farthest_pts_idx], np.array(farthest_pts_idx)

def get_camera_intrinsics(screenHeight, screenWidth):
    projMat = pyflex.get_projMatrix().reshape(4, 4).T 
    cx = screenWidth / 2.0
    cy = screenHeight / 2.0
    fx = projMat[0, 0] * cx
    fy = projMat[1, 1] * cy
    camera_intrinsic_params = np.array([fx, fy, cx, cy])
    return camera_intrinsic_params

def get_camera_extrinsics():
    return pyflex.get_viewMatrix().reshape(4, 4).T


def render(screenHeight, screenWidth, no_return=False):
    pyflex.step()
    if no_return:
        return
    else:
        return pyflex.render(render_depth=True).reshape(screenHeight, screenWidth, 5)

def set_light(screenHeight, screenWidth):
    screenWidth, screenHeight = 720, 720
    pyflex.set_screenWidth(screenWidth)
    pyflex.set_screenHeight(screenHeight)
    pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
    pyflex.set_light_fov(70.)
    

def set_camera(camera_view, cam_dis=6., cam_height=10.):
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
    return camPos, camAngle
    

def init_multiview_camera(cam_dis = 6., cam_height = 10.):
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

def add_table(table_height, table_length=4.5, table_width=3.5):
    halfEdge = np.array([table_width, table_height, table_length])
    center = np.array([0.0, 0.0, 0.0])
    quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
    hideShape = 0
    color = np.ones(3) * (160. / 255.)
    pyflex.add_box(halfEdge, center, quats, hideShape, color)
    table_shape_states = np.concatenate([center, center, quats, quats])
    return table_shape_states

def calc_container_boxes(pos, angle, direction, size, border=0.02):
    boxes = []
    hide_shape = []

    dx, dy, dz = size
    r_mtx = rotation_matrix(angle, direction)
    quat = quaternion_from_matrix(r_mtx)

    # bottom
    halfEdge = np.array([dx / 2. + border, border / 2., dz / 2. + border])
    center = np.array([0., -(dy + border) / 2., 0., 1.])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(0)

    # left
    halfEdge = np.array([border / 2., dy / 2. + border, dz / 2. + border])
    center = np.array([-(dx + border) / 2., 0., 0., 1.])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(1)

    # right
    center = np.array([(dx + border) / 2., 0., 0., 1.])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(1)

    # back
    halfEdge = np.array([dx / 2. + border, dy / 2. + border, border / 2.])
    center = np.array([0., 0., -(dz + border) / 2., 1.])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(1)

    # front
    center = np.array([0., 0., (dz + border) / 2., 1.])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(1)

    # top bars
    halfEdge = np.array([border / 2., border / 2., dz / 2. + border])
    center = np.array([(dx + border) / 2., (dy + border) / 2., 0., 1])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(0)

    center = np.array([-(dx + border) / 2., (dy + border) / 2., 0., 1])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(0)

    halfEdge = np.array([dx / 2. + border, border / 2., border / 2.])
    center = np.array([0, (dy + border) / 2., (dz + border) / 2., 1])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(0)

    center = np.array([0, (dy + border) / 2., -(dz + border) / 2., 1])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(0)

    # side bars
    halfEdge = np.array([border / 2., dy / 2. + border, border / 2.])
    center = np.array([(dx + border) / 2., 0., (dz + border) / 2., 1])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(0)

    center = np.array([(dx + border) / 2., 0., -(dz + border) / 2., 1])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(0)

    center = np.array([-(dx + border) / 2., 0., -(dz + border) / 2., 1])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(0)

    center = np.array([-(dx + border) / 2., 0., (dz + border) / 2., 1])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(0)

    return boxes, np.array(hide_shape)

