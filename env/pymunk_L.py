import sys, os
import pymunk
from pymunk import Vec2d
import math
import numpy as np
import random
import time
import cv2

import pymunk
from pymunk import Vec2d
import math
import numpy as np
import random
import sys, os

sys.path.append(os.getcwd())
from .pymunk_base import Base_Sim

"""
main class for the L-shaped pushing task
"""


class L_Sim(Base_Sim):
    def __init__(self, param_dict, init_poses=None, target_poses=None, pusher_pos=None):
        super().__init__(param_dict)
        if target_poses is None:
            self.target_positions = None
            self.target_angles = None
        else:
            self.target_positions = [target_pose[:2] for target_pose in target_poses]
            self.target_angles = [target_pose[2] for target_pose in target_poses]  # in radians
        if init_poses is not None:
            self.obj_num = len(init_poses)
        elif "obj_num" in param_dict:
            self.obj_num = param_dict["obj_num"]
        else:
            self.obj_num = 1
        self.param_dict = param_dict
        unit_size, leg_shape, foot_shape = (
            param_dict["unit_size"],
            param_dict["leg_shape"],
            param_dict["foot_shape"],
        )
        self.unit_size = unit_size
        self.leg_size = (leg_shape[0] * unit_size, leg_shape[1] * unit_size)
        self.foot_size = (foot_shape[0] * unit_size, foot_shape[1] * unit_size)
        self.leg_shape = leg_shape
        self.foot_shape = foot_shape
        self.particles = []
        self.create_world(init_poses, pusher_pos)

    def create_object(self, id, pose=None):
        """
        Create a single L-shaped piece by defining its shapes, mass, etc., with the leg above the foot and both aligned to the left.
        """
        color = self.object_colors[id % len(self.object_colors)]
        foot_size, leg_size = self.foot_size, self.leg_size
        mass = self.obj_mass  # Define the mass of the L shape

        # Get vertices for the leg and foot
        leg_vertices, foot_vertices = self.get_l_comp_vertices()

        # Assuming equal mass distribution between the leg and the foot
        foot_square = foot_size[0] * foot_size[1]
        leg_square = leg_size[0] * leg_size[1]
        total_square = foot_square + leg_square
        leg_moment = pymunk.moment_for_poly(mass * leg_square / total_square, leg_vertices)
        foot_moment = pymunk.moment_for_poly(mass * foot_square / total_square, foot_vertices)

        # Total moment for the L-shape
        moment = leg_moment + foot_moment

        body = pymunk.Body(mass, moment)
        if pose is None:
            body.angle = random.random() * math.pi * 2
            body.position = Vec2d(random.randint(self.width / 2 - 100, self.width / 2 + 100), random.randint(self.height / 2 - 100, self.height / 2 + 100))
        else:
            body.angle = pose[2]
            body.position = Vec2d(pose[0], pose[1])

        leg = pymunk.Poly(body, leg_vertices)
        leg.color = color
        leg.elasticity = self.elasticity
        leg.friction = self.friction

        foot = pymunk.Poly(body, foot_vertices)
        foot.color = color
        foot.elasticity = self.elasticity
        foot.friction = self.friction

        # create particles, one per pixel
        center_leg, center_foot, com = get_l_comp_centers(leg_size, foot_size)
        center_leg = np.array(center_leg) - np.array(com)
        center_foot = np.array(center_foot) - np.array(com)
        x_leg, y_leg = np.meshgrid(
            np.arange(-leg_size[0] / 2, leg_size[0] / 2, 1),
            np.arange(-leg_size[1] / 2, leg_size[1] / 2, 1),
        )
        particles_leg = np.vstack((x_leg.flatten(), y_leg.flatten())).T + center_leg
        x_foot, y_foot = np.meshgrid(
            np.arange(-foot_size[0] / 2, foot_size[0] / 2, 1),
            np.arange(-foot_size[1] / 2, foot_size[1] / 2, 1),
        )
        particles_foot = np.vstack((x_foot.flatten(), y_foot.flatten())).T + center_foot
        self.particles.append(np.vstack((particles_leg, particles_foot)))

        return body, [leg, foot]
    
    def add_object(self, id, pose=None):
        """
        Create and add a single object to sim.
        """
        body, shape_components = self.create_object(id, pose)
        self.space.add(body, *shape_components)

        # add collision avoidance
        for obj in self.obj_list:
            shapes = obj[1]
            for shape in shapes:
                for self_shape in shape_components:
                    if shape.shapes_collide(self_shape).points != []:
                        # try to add object again
                        # if self.dir != None:
                        #     print("collision!")
                        self.space.remove(body, *shape_components)
                        self.particles.pop()
                        return self.add_object(id)

        self.obj_list.append([body, shape_components])  # Adjust storage to handle multiple shapes

    def create_pusher(self, position):
        """
        Create a single bar by defining its shape, mass, etc.
        """
        body = pymunk.Body(1e7, float('inf'))
        body.position = Vec2d(position[0], position[1])
        # shape = pymunk.Circle(body, radius=5)
        shape = pymunk.Poly.create_box(body, self.pusher_size)
        # pymunk.Poly.create_box(body, (50, 50))
        shape.elasticity = self.elasticity
        shape.friction = self.friction
        shape.color = (255, 0, 0, 255)
        return body, shape

    # def get_object_keypoints(self, index, target=False):
    #     return get_keypoints_from_pose(self.get_object_pose(index, target), self.param_dict, self.include_com)

    def get_object_vertices(self, index, target=False):
        return transform_polys_wrt_pose_2d(self.get_l_comp_vertices(), self.get_object_pose(index, target))

    # def gen_vertices_from_pose(self, pose):
    #     return transform_polys_wrt_pose_2d(self.get_t_comp_vertices(), pose)

    def get_object_particles(self, index):
        return transform_polys_wrt_pose_2d([self.particles[index]], self.get_object_pose(index))[0]

    def get_eef_states(self):
        if self.pusher_body is None:
            return [0., 0., 0.] # dummy
        return [self.pusher_body.position[0], self.pusher_body.position[1], self.pusher_body.angle]

    def get_current_state(self):
        obj_keypoints = self.get_all_object_keypoints()[0]
        state = np.concatenate(
            (obj_keypoints, np.array([self.pusher_body.position]), np.array([self.velocity])), axis=0
        )
        return state.flatten()

    def get_l_comp_vertices(self, flatten=False):
        """
        Get the vertices of the leg and foot of the L shape.
        """
        leg_size, foot_size = self.leg_size, self.foot_size
        center_leg, center_foot, com = get_l_comp_centers(leg_size, foot_size)
        center_leg = np.array(center_leg) - np.array(com)
        center_foot = np.array(center_foot) - np.array(com)
        leg_vertices = get_rect_vertices(*leg_size)
        leg_vertices = leg_vertices + center_leg
        if flatten:
            leg_vertices = leg_vertices.flatten()
        foot_vertices = get_rect_vertices(*foot_size)
        foot_vertices = foot_vertices + center_foot
        if flatten:
            foot_vertices = foot_vertices.flatten()
        return leg_vertices.tolist(), foot_vertices.tolist()
    
    def get_pusher_vertices(self):
        return get_rect_vertices(self.pusher_size[0], self.pusher_size[1])

    def render(self):
        if not (self.ENABLE_VIS or self.SAVE_IMG):
            return
        # start_time = time.time()

        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = self.background_color[:3]

        for draw_target in [True, False]:
            obj_list = self.get_all_object_vertices(target=draw_target)
            if obj_list is None:
                continue
            for i, obj in enumerate(obj_list):
                polys = np.array(obj, np.int32)
                color = self.object_colors[i % len(self.object_colors)][:3]
                if draw_target:
                    color = np.round(np.array(color) * 0.5).astype(np.uint8).tolist()
                cv2.fillPoly(img, polys, color)

        pusher_pos = self.get_pusher_position()

        if pusher_pos is not None:
            pusher_angle = self.pusher_body.angle
            poly = transform_polys_wrt_pose_2d(
                [self.get_pusher_vertices()],
                [pusher_pos[0], pusher_pos[1], pusher_angle],
            )[0]
            cv2.fillPoly(img, np.array([poly], np.int32), self.pusher_color[:3])

        # cv2 has BGR format
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if self.ENABLE_VIS:
            cv2.imshow("Simulator", img)
            cv2.waitKey(1)
        # print("Update Image Time: ", time.time() - start_time)
        # if self.SAVE_IMG or self.IMG_STATE:
        #     # self.image_list.append(img)
        #     self.current_image = img

        return img

    def vis_particles(self):
        if not (self.ENABLE_VIS or self.SAVE_IMG):
            return
        # start_time = time.time()

        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = self.background_color[:3]

        for draw_target in [True, False]:
            obj_list = self.get_all_object_vertices(target=draw_target)
            if obj_list is None:
                continue
            for i, obj in enumerate(obj_list):
                polys = np.array(obj, np.int32)
                color = self.object_colors[i % len(self.object_colors)][:3]
                if draw_target:
                    color = np.round(np.array(color) * 0.5).astype(np.uint8).tolist()
                cv2.fillPoly(img, polys, color)

        pusher_pos = self.get_pusher_position()

        if pusher_pos is not None:
            pusher_angle = self.pusher_body.angle
            poly = transform_polys_wrt_pose_2d(
                [self.get_pusher_vertices()],
                [pusher_pos[0], pusher_pos[1], pusher_angle],
            )[0]
            cv2.fillPoly(img, np.array([poly], np.int32), self.pusher_color[:3])

        # add particles
        obj_particles = self.get_all_particles()
        for particles in obj_particles:
            particles = np.array(particles, np.int32)
            for point in particles:
                cv2.circle(img, tuple(point), 1, (255, 255, 255), -1)

        # cv2 has BGR format
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if self.ENABLE_VIS:
            cv2.imshow("Simulator", img)
            cv2.waitKey(1)

        return img

    def sample_action(self):
        index = np.random.randint(0, self.obj_num)
        particles = self.get_object_particles(index)

        for _ in range(1000):
            # sample a no overlap start point
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            dist_threshold = self.pusher_size[0] / 2 + 5

            collision_flag = False
            for shape in self.obj_list[0][1]:
                if shape.point_query((x, y)).distance < dist_threshold:
                    collision_flag = True
                    break
            if collision_flag:
                continue

            # sample a particle on object, extend to the end
            pick_idx = random.randint(0, len(particles) - 1)
            pick_point = particles[pick_idx]
            if x == pick_point[0]:
                continue
            slope = (y - pick_point[1]) / (x - pick_point[0])
            if pick_point[0] < x:
                x_end = pick_point[0] - 100
            else:
                x_end = pick_point[0] + 100
            y_end = pick_point[1] + slope * (x_end - pick_point[0])
            end_point = np.array([x_end, y_end])
            # clip
            lower_bound = np.array([0.2 * self.width, 0.2 * self.height])
            upper_bound = np.array([0.8 * self.width, 0.8 * self.height])
            end_point = np.clip(end_point, lower_bound, upper_bound)

            return [x, y, end_point[0], end_point[1]]
        
        return None


""" 
some helper functions
"""


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo

def parse_merging_pose(merging_pose, param_dict, distance=0, pose_mode=0, noisy=False):
    unit_size, leg_shape, foot_shape = (
        param_dict["unit_size"],
        param_dict["leg_shape"],
        param_dict["foot_shape"],
    )
    assert leg_shape[1] == foot_shape[1]
    merging_pose = np.array(merging_pose)
    merging_angle = merging_pose[2]
    leg_size = np.array(leg_shape) * unit_size
    foot_size = np.array(foot_shape) * unit_size
    w_l, h_l = leg_size
    w_f, h_f = foot_size
    noise = 20
    _, _, [x_m, y_m] = get_l_comp_centers(leg_size, foot_size)
    if pose_mode == 0 or pose_mode == 1:
        x_c, y_c = (w_l + w_f) / 2, h_f
        delta_x = x_c - x_m + distance
        delta_y = y_c - y_m + distance
        offset = np.array(
            [
                [
                    -delta_x * math.cos(merging_angle) + delta_y * math.sin(merging_angle),
                    -delta_x * math.sin(merging_angle) - delta_y * math.cos(merging_angle),
                    0,
                ],
                [
                    delta_x * math.cos(merging_angle) - delta_y * math.sin(merging_angle),
                    delta_x * math.sin(merging_angle) + delta_y * math.cos(merging_angle),
                    math.pi,
                ],
            ]
        )
        if noisy:
            offset += np.array(
                [
                    [rand_float(-noise, noise), rand_float(-noise, noise), rand_float(-math.pi, math.pi) * noise / 100],
                    [rand_float(-noise, noise), rand_float(-noise, noise), rand_float(-math.pi, math.pi) * noise / 100],
                ]
            )
            offset += np.array(
                [
                    [rand_float(-noise, noise), rand_float(-noise, noise), rand_float(-math.pi, math.pi) * noise / 100],
                    [rand_float(-noise, noise), rand_float(-noise, noise), rand_float(-math.pi, math.pi) * noise / 100],
                ]
            )
        if pose_mode == 1:
            offset = offset[::-1]
    elif pose_mode == 2 or pose_mode == 3:
        x_c, y_c = (w_l + w_f) / 2, h_f
        delta_x = x_c - x_m + distance
        delta_y = y_c - y_m
        offset = np.array(
            [
                [
                    -delta_x * math.cos(merging_angle) + delta_y * math.sin(merging_angle),
                    -delta_x * math.sin(merging_angle) - delta_y * math.cos(merging_angle),
                    0,
                ],
                [
                    delta_x * math.cos(merging_angle) - delta_y * math.sin(merging_angle),
                    delta_x * math.sin(merging_angle) + delta_y * math.cos(merging_angle),
                    0.5 * math.pi,
                ],
            ]
        )
        if pose_mode == 3:
            offset = offset[::-1]
    else:
        half_height = (h_l + h_f + distance) / 2
        offset = np.array(
            [
                [
                    -half_height * math.sin(merging_angle) * rand_float(0, 1),
                    half_height * math.cos(merging_angle) * rand_float(0, 1),
                    rand_float(0, 2 * math.pi),
                ],
                [
                    half_height * math.sin(merging_angle) * rand_float(0, 1),
                    -half_height * math.cos(merging_angle) * rand_float(0, 1),
                    rand_float(0, 2 * math.pi),
                ],
            ]
        )
        # # # random exachange the order of the two L shapes
        # if random.randint(0, 1) == 0:
        #     offset = offset[::-1]
    return (merging_pose + offset).tolist()


# def generate_init_target_states(init_poses, target_poses, param_dict, include_com=False):
#     init_states = get_keypoints_from_pose(init_poses[0], param_dict, include_com)
#     target_states = get_keypoints_from_pose(target_poses[0], param_dict, include_com)
#     return init_states.flatten(), target_states.flatten()


def get_l_comp_centers(leg_size, foot_size):
    """
    Get the center of the leg, foot and the L shape.
    """
    w_l, h_l = leg_size
    w_f, h_f = foot_size
    x_l, y_l = w_l / 2, h_f + h_l / 2
    x_f, y_f = w_f / 2, h_f / 2
    m_l, m_f = w_l * h_l, w_f * h_f
    x_m, y_m = calculate_com([x_l, x_f], [y_l, y_f], [m_l, m_f])
    return [x_l, y_l], [x_f, y_f], [x_m, y_m]


# def get_keypoints_from_pose(pose, param_dict, include_com=False):
#     stem_size, bar_size = param_dict["stem_size"], param_dict["bar_size"]
#     pos = Vec2d(pose[0], pose[1])
#     angle = pose[2]
#     w_s, h_s = stem_size
#     w_b, h_b = bar_size
#     _, [x_b, y_b], [x_m, y_m] = get_t_comp_centers(stem_size, bar_size)
#     com = Vec2d(x_m, y_m)
#     # Left Center, Middle Top Center, Right Center, Bottom Center
#     # # consider the bottom center of the stem as the origin
#     offsets = [
#         Vec2d(-w_b / 2, y_b),
#         Vec2d(0, y_b),
#         Vec2d(w_b / 2, y_b),
#         Vec2d(0, 0),
#     ]
#     if include_com:
#         offsets.append(Vec2d(0, 0))
#     # Calculate the global position of each keypoint
#     keypoints = []
#     for offset in offsets:
#         offset = offset - com
#         rotated_offset = offset.rotated(angle)
#         keypoints.append(pos + rotated_offset)

#     return np.array(keypoints)


# def get_pose_from_keypoints(keypoints, param_dict):
#     """
#     Get the pose of the T shape from its keypoints.
#     """
#     stem_size, bar_size = param_dict["stem_size"], param_dict["bar_size"]
#     w_s, h_s = stem_size
#     w_b, h_b = bar_size
#     _, [x_b, y_b], [x_m, y_m] = get_t_comp_centers(stem_size, bar_size)
#     model_points = np.array(
#         [
#             [-w_b / 2, y_b],
#             [0, y_b],
#             [w_b / 2, y_b],
#             [0, 0],
#         ]
#     ) - np.array([x_m, y_m])

#     return keypoints_to_pose_2d_SVD(model_points, keypoints)


# def get_offests_w_origin(param_dict):
#     stem_size, bar_size = param_dict["stem_size"], param_dict["bar_size"]
#     _, _, [x_g, y_g] = get_t_comp_centers(stem_size, bar_size)
#     # hardcode as the top left corner of the bar, deponds on the .obj file
#     cog_w_origin = np.array([bar_size[0] / 2, y_g - (bar_size[1] + stem_size[1])])
#     offest_w_cog = get_keypoints_from_pose([0, 0, 0], param_dict)
#     return offest_w_cog + cog_w_origin


def transform_polys_wrt_pose_2d(poly_list, pose):
    # poly_list: list of 2D polygons, each represented by a list of vertices
    x, y, angle = pose
    translation_vector = np.array([x, y])
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed_poly_list = []
    for vertices in poly_list:
        transformed_vertices = np.dot(vertices, rotation_matrix.T) + translation_vector
        transformed_poly_list.append(transformed_vertices)

    return transformed_poly_list


def get_rect_vertices(w, h):
    w /= 2
    h /= 2
    return np.array([[-w, -h], [w, -h], [w, h], [-w, h]])


def calculate_com(x_i, y_i, m_i):
    """
    Calculate the center of mass (CoM) for a composite object based on
    the masses (or areas) and coordinates of individual parts.

    Parameters:
    - x_i: List or array of x-coordinates of the centers of mass of the components.
    - y_i: List or array of y-coordinates of the centers of mass of the components.
    - m_i: List or array of masses (or areas) of the components.

    Returns:
    - (C_x, C_y): A tuple representing the x and y coordinates of the composite CoG.
    """
    total_mass = sum(m_i)
    C_x = sum(m * x for m, x in zip(m_i, x_i)) / total_mass
    C_y = sum(m * y for m, y in zip(m_i, y_i)) / total_mass
    return (C_x, C_y)


if __name__ == "__main__":
    param_dict = {
        "stem_size": (10, 60),
        "bar_size": (60, 10),
        "pusher_size": 5,
        "save_img": False,
        "enable_vis": True,
    }

    # init_poses = [[[250,250,math.radians(45)], [150,150,math.radians(-45)]]]
    init_poses = [[250, 250, math.radians(0)]]
    target_poses = [[250, 250, math.radians(45)]]
    sim = L_Sim(
        param_dict=param_dict,
        init_poses=init_poses,
        target_poses=target_poses,
    )
    # [[250,250,math.radians(45)], [150,150,math.radians(-45)]]
    sim.render()
    print(sim.get_all_object_positions())
    print(sim.get_all_object_keypoints())
    print(sim.get_current_state())
    print(sim.get_all_object_keypoints(target=True))
    for i in range(5):
        sim.render()
        time.sleep(0.5)
