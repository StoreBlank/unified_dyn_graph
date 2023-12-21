import os
import numpy as np
import pyflex
import gym
import math
import cv2
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree

# robot
import pybullet as p
import pybullet_data
from bs4 import BeautifulSoup
from transformations import quaternion_from_matrix, quaternion_matrix
from env.robot_env import FlexRobotHelper

pyflex.loadURDF = FlexRobotHelper.loadURDF
pyflex.resetJointState = FlexRobotHelper.resetJointState
pyflex.getRobotShapeStates = FlexRobotHelper.getRobotShapeStates

# utils
from utils_env import load_cloth
from utils_env import rand_float, rand_int, quatFromAxisAngle, find_min_distance
from utils_env import fps_rad, recenter

class FlexEnv(gym.Env):
    def __init__(self, config=None) -> None:
        super().__init__()

        # set up pybullet
        physicsClient = p.connect(p.DIRECT)
        # physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeId = p.loadURDF("plane.urdf")

        # set up robot arm
        # xarm6
        self.flex_robot_helper = FlexRobotHelper()
        self.gripper = config['dataset']['gripper']
        self.grasp = config['dataset']['grasp']
        if self.gripper:   
            # 6(arm) + 1 base_link + 6(gripper; 9-left finger, 12-right finger)
            self.end_idx = 6 #6
            self.num_dofs = 12 
            self.gripper_state = 0
        else:
            self.end_idx = 6
            self.num_dofs = 6

        # set up pyflex
        self.screenWidth = 720
        self.screenHeight = 720

        self.wkspc_w = config['dataset']['wkspc_w']
        self.headless = config['dataset']['headless']
        self.obj = config['dataset']['obj']
        self.cont_motion = config['dataset']['cont_motion']

        pyflex.set_screenWidth(self.screenWidth)
        pyflex.set_screenHeight(self.screenHeight)
        pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
        pyflex.set_light_fov(70.)
        pyflex.init(config['dataset']['headless'])

        # set up camera
        self.camera_view = config['dataset']['camera_view']
        self.camera_radius = config['dataset']['camera_radius']

        # define action space
        self.action_dim = 4

        # define property space
        self.property = None
        self.physics = config['dataset']['physics']
        
        # others
        self.count = 0
        self.fps = config['dataset']['fps']
        self.particle_num_threshold = 0
        self.obj_shape_states = None
        
    ### shape states
    def robot_to_shape_states(self, robot_states):
        n_robot_links = robot_states.shape[0]
        n_table = self.table_shape_states.shape[0]
        
        if self.obj_shape_states == None:
            shape_states = np.zeros((n_table + n_robot_links, 14))
            shape_states[:n_table] = self.table_shape_states # set shape states for table
            shape_states[n_table:] = robot_states # set shape states for robot
        else:
            n_objs = self.obj_shape_states.shape[0]
            shape_states = np.zeros((n_table + n_objs + n_robot_links, 14))
            shape_states[:n_table] = self.table_shape_states # set shape states for table
            shape_states[n_table:n_table+n_objs] = self.obj_shape_states # set shape states for objects
            shape_states[n_table+n_objs:] = robot_states # set shape states for robot
        
        return shape_states
                        
    def reset_robot(self, jointPositions = np.zeros(13).tolist()):  
        index = 0
        for j in range(7):
            p.changeDynamics(self.robotId, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.robotId, j)

            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                pyflex.resetJointState(self.flex_robot_helper, j, jointPositions[index])
                # print(j, 'jointPositions', jointPositions[index])
                index = index + 1
                
        pyflex.set_shape_states(self.robot_to_shape_states(pyflex.getRobotShapeStates(self.flex_robot_helper)))
    
    ### cameras 
    def set_camera(self):
        cam_dis, cam_height = 6., 10.
        if self.camera_view == 0:
            self.camPos = np.array([0., cam_height+10., 0.])
            self.camAngle = np.array([0., -np.deg2rad(90.), 0.])
        elif self.camera_view == 1:
            self.camPos = np.array([cam_dis, cam_height, cam_dis])
            self.camAngle = np.array([np.deg2rad(45.), -np.deg2rad(45.), 0.])
        elif self.camera_view == 2:
            self.camPos = np.array([cam_dis, cam_height, -cam_dis])
            self.camAngle = np.array([np.deg2rad(45.+90.), -np.deg2rad(45.), 0.])
        elif self.camera_view == 3:
            self.camPos = np.array([-cam_dis, cam_height, -cam_dis])
            self.camAngle = np.array([np.deg2rad(45.+180.), -np.deg2rad(45.), 0.])
        elif self.camera_view == 4:
            self.camPos = np.array([-cam_dis, cam_height, cam_dis])
            self.camAngle = np.array([np.deg2rad(45.+270.), -np.deg2rad(45.), 0.])
        else:
            raise ValueError('camera_view not defined')
        
        pyflex.set_camPos(self.camPos)
        pyflex.set_camAngle(self.camAngle)
    
    def init_multiview_camera(self):
        self.camPos_list = []
        self.camAngle_list = []

        cam_dis, cam_height = 6., 10.
        rad_list = np.deg2rad(np.array([0., 90., 180., 270.]) + 45.)
        cam_x_list = np.array([cam_dis, cam_dis, -cam_dis, -cam_dis])
        cam_z_list = np.array([cam_dis, -cam_dis, -cam_dis, cam_dis])

        for i in range(len(rad_list)):
            self.camPos_list.append(np.array([cam_x_list[i], cam_height, cam_z_list[i]]))
            self.camAngle_list.append(np.array([rad_list[i], -np.deg2rad(45.), 0.]))
        
        self.cam_intrinsic_params = np.zeros([len(self.camPos_list), 4]) # [fx, fy, cx, cy]
        self.cam_extrinsic_matrix = np.zeros([len(self.camPos_list), 4, 4]) # [R, t]
    
    ### TODO: write the scene as a class
    def init_scene(self, obj, property_params):
        if obj == 'carrots':
            radius = 0.03
    
            num_granular_ft_x = 5 #rand_float(2, 10)
            num_granular_ft_y = 2  #np.random.choice([2, 3])
            num_granular_ft_z = 5 #rand_float(2, 10)
            num_granular_ft = [num_granular_ft_x, num_granular_ft_y, num_granular_ft_z] 
            num_granular = int(num_granular_ft_x * num_granular_ft_y * num_granular_ft_z)
            
            granular_scale = 0.25 #rand_float(0.1, 0.2)
            pos_granular = [-1.0, 1., -0.8]
            granular_dis = 0.1 #rand_float(0.1, 0.3)

            draw_mesh = 1
            
            shapeCollisionMargin = 0.01
            collisionDistance = 0.03
            
            dynamic_friction = 0.3 #rand_float(0.2, 0.9)
            granular_mass = 0.1 #rand_float(0.1, 10.)

            scene_params = np.array([radius, *num_granular_ft, granular_scale, *pos_granular, granular_dis, 
                                    draw_mesh, shapeCollisionMargin, collisionDistance, dynamic_friction,
                                    granular_mass])

            temp = np.array([0])
            pyflex.set_scene(35, scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0)
            
            property_param = {
                'particle_radius': radius,
                'num_particles': self.get_num_particles(),
                'granular_scale': granular_scale,
                'num_granular': num_granular,
                'distribution_r': granular_dis,
                'dynamic_friction': dynamic_friction,
                'granular_mass': granular_mass,
            }
            # print(property_params)
            self.property = property_param
        
        elif obj == 'coffee':
            radius = 0.03
            
            global_scale = 4
            scale = rand_float(0.2, 0.3) * global_scale / 8.0
            
            blob_r = rand_float(0.2, 0.8)
            x = - blob_r * global_scale / 8.0
            y = 0.5
            z = - blob_r * global_scale / 8.0
            
            if 0.5 <= blob_r < 0.8:
                space_scale = rand_float(1.1, 2.)
            else:
                space_scale = rand_float(1.1, 3.)
            inter_space = space_scale * scale
            
            num_x = int(abs(x/1.) / scale + 1) * 2
            num_y = np.random.randint(1, 4)
            num_z = int(abs(z/1.) / scale + 1) * 2
            num_coffee = num_x * num_z * num_y 
            
            mass = rand_float(0.1, 10.) #10g-1000g
            
            staticFriction = 0.0
            dynamicFriction = rand_float(0.1, 1.0)
            draw_skin = 1
            radius = 0.03
            
            self.scene_params = np.array([
                scale, x, y, z, staticFriction, dynamicFriction, draw_skin, radius,
                num_x, num_y, num_z, inter_space, mass])

            temp = np.array([0])
            pyflex.set_scene(20, self.scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0) 
            
            self.property = {'particle_radius': radius,
                             'num_particles': self.get_num_particles(),
                             'rand_scale': scale,
                             'blob_r': blob_r,
                             'num_granule': num_coffee,
                             'dynamic_friction': dynamicFriction,
                             'mass': mass}
        else:
            raise ValueError('obj not defined')
    
    def reset(self, count=0, dir=None, property_params=None):
        obj = self.obj
        self.init_scene(obj, property_params)
        
        ## camera setting
        self.set_camera()
        self.init_multiview_camera()
        
        ## add table board
        self.table_shape_states = np.zeros((2, 14))
        # table for workspace
        wkspace_height = 0.5
        wkspace_width = 3.5
        wkspace_length = 4.5
        halfEdge = np.array([wkspace_width, wkspace_height, wkspace_length])
        center = np.array([0.0, 0.0, 0.0])
        quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
        hideShape = 0
        color = np.ones(3) * (160. / 255.)
        pyflex.add_box(halfEdge, center, quats, hideShape, color)
        self.table_shape_states[0] = np.concatenate([center, center, quats, quats])
        
        # table for robot
        robot_table_height = 0.525
        robot_table_width = 1.0
        robot_table_length = 1.0
        halfEdge = np.array([robot_table_width, robot_table_height, robot_table_length])
        center = np.array([-wkspace_width-robot_table_width, 0.0, 0.0])
        quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
        hideShape = 0
        color = np.ones(3) * (160. / 255.)
        pyflex.add_box(halfEdge, center, quats, hideShape, color)
        self.table_shape_states[1] = np.concatenate([center, center, quats, quats])
        
        ## add robot
        robot_base_pos = [-wkspace_width-1.0, 0., wkspace_height]
        robot_base_orn = [0, 0, 0, 1]
        self.robotId = pyflex.loadURDF(self.flex_robot_helper, 'assets/xarm/xarm6_with_gripper.urdf', robot_base_pos, robot_base_orn, globalScaling=9) 
        self.rest_joints = np.zeros(8)

        pyflex.set_shape_states(self.robot_to_shape_states(pyflex.getRobotShapeStates(self.flex_robot_helper)))
        
        for _ in range(30):
            pyflex.step()
        
        self.count = count
        ### initial pose render
        if dir != None:
            for j in range(len(self.camPos_list)):
                pyflex.set_camPos(self.camPos_list[j])
                pyflex.set_camAngle(self.camAngle_list[j])

                # create dir with cameras
                cam_dir = os.path.join(dir, 'camera_%d' % (j))
                os.system('mkdir -p %s' % (cam_dir))

                img = self.render()
                # rgb and depth images
                cv2.imwrite(os.path.join(cam_dir, '%d_color.jpg' % self.count), img[:, :, :3][..., ::-1])
                cv2.imwrite(os.path.join(cam_dir, '%d_depth.png' % self.count), (img[:, :, -1]*1000).astype(np.uint16))
                
                # particles
                if j == 0:
                    particles = self.get_positions().reshape(-1, 4)
                    # if self.fps and particles.shape[0] < self.particle_num_threshold:
                    #     # sample points
                        
                    with open(os.path.join(cam_dir, '%d_particles.npy' % self.count), 'wb') as f:
                        np.save(f, particles)
            self.count += 1
            
        # update robot shape states
        for idx, joint in enumerate(self.rest_joints):
            pyflex.set_shape_states(self.robot_to_shape_states(pyflex.resetJointState(self.flex_robot_helper, idx, joint)))
        
        self.num_joints = p.getNumJoints(self.robotId)
        self.joints_lower = np.zeros(self.num_dofs)
        self.joints_upper = np.zeros(self.num_dofs)
        dof_idx = 0
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robotId, i)
            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                self.joints_lower[dof_idx] = info[8]
                self.joints_upper[dof_idx] = info[9]
                dof_idx += 1
        self.last_ee = None
        self.reset_robot()
        
        for _ in range(400):
            pyflex.step()
        
        # initial render
        if dir != None:
            for j in range(len(self.camPos_list)):
                pyflex.set_camPos(self.camPos_list[j])
                pyflex.set_camAngle(self.camAngle_list[j])

                # create dir with cameras
                cam_dir = os.path.join(dir, 'camera_%d' % (j))
                os.system('mkdir -p %s' % (cam_dir))

                img = self.render()
                cv2.imwrite(os.path.join(cam_dir, '%d_color.jpg' % self.count), img[:, :, :3][..., ::-1])
                cv2.imwrite(os.path.join(cam_dir, '%d_depth.png' % self.count), (img[:, :, -1]*1000).astype(np.uint16))
                if j == 0:
                    with open(os.path.join(cam_dir, '%d_particles.npy' % self.count), 'wb') as f:
                        np.save(f, self.get_positions().reshape(-1, 4))
                if self.cam_intrinsic_params[j].sum() == 0 or self.cam_extrinsic_matrix[j].sum() == 0:
                        self.cam_intrinsic_params[j] = self.get_camera_intrinsics()
                        self.cam_extrinsic_matrix[j] = self.get_camera_extrinsics()
            self.count += 1
        
        return self.count
        
    def step(self, action, prev_counts=0, dir=None):
        h = 0.5 + 0.9
        s_2d = np.concatenate([action[:2], [h]])
        e_2d = np.concatenate([action[2:], [h]])

        # pusher angle depending on x-axis
        if (s_2d - e_2d)[0] == 0:
            pusher_angle = np.pi/2
        else:
            pusher_angle = np.arctan((s_2d - e_2d)[1] / (s_2d - e_2d)[0])  
        # robot orientation
        orn = np.array([0.0, np.pi, pusher_angle + np.pi/2])

        # create way points
        if self.cont_motion: #TODO - strange
            if self.last_ee is None:
                self.reset_robot(self.rest_joints)
                self.last_ee = s_2d
            way_points = [self.last_ee, s_2d, e_2d]
        else:
            way_points = [s_2d + [0., 0., 0.2], s_2d, e_2d, e_2d + [0., 0., 0.2]]
            self.reset_robot(self.rest_joints)

        # set robot speed
        speed = 1.0/100.
        
        self.count = prev_counts
        self.contact = []
                
        for i_p in range(len(way_points)-1):
            s = way_points[i_p]
            e = way_points[i_p+1]
            steps = int(np.linalg.norm(e-s)/speed) + 1
            
            for i in range(steps):
                end_effector_pos = s + (e-s) * i / steps
                end_effector_orn = p.getQuaternionFromEuler(orn)
                jointPoses = p.calculateInverseKinematics(self.robotId, 
                                                        self.end_idx, 
                                                        end_effector_pos, 
                                                        end_effector_orn, 
                                                        self.joints_lower.tolist(), 
                                                        self.joints_upper.tolist(),
                                                        (self.joints_upper - self.joints_lower).tolist(),
                                                        self.rest_joints)
                    
                self.reset_robot(jointPoses)
                pyflex.step()

                # save img in each step
                obj_pos = self.get_positions().reshape(-1, 4)[:, [0, 2]]
                obj_pos[:, 1] *= -1
                robot_obj_dist = np.min(cdist(end_effector_pos[:2].reshape(1, 2), obj_pos))
                
                if dir != None:
                    if robot_obj_dist < 0.3 and i % 4 == 0: #contact
                        for j in range(len(self.camPos_list)):
                            pyflex.set_camPos(self.camPos_list[j])
                            pyflex.set_camAngle(self.camAngle_list[j])

                            # create dir with cameras
                            cam_dir = os.path.join(dir, 'camera_%d' % (j))
                            os.system('mkdir -p %s' % (cam_dir))

                            img = self.render()
                            cv2.imwrite(os.path.join(cam_dir, '%d_color.jpg' % self.count), img[:, :, :3][..., ::-1])
                            cv2.imwrite(os.path.join(cam_dir, '%d_depth.png' % self.count), (img[:, :, -1]*1000).astype(np.uint16))
                            if j == 0:
                                with open(os.path.join(cam_dir, '%d_particles.npy' % self.count), 'wb') as f:
                                    np.save(f, self.get_positions().reshape(-1, 4))
                                with open(os.path.join(cam_dir, '%d_endeffector.npy' % self.count), 'wb') as f:
                                    np.save(f, end_effector_pos)
                        self.count += 1
                        self.contact.append(self.count)
                    elif i % 20 == 0:
                        for j in range(len(self.camPos_list)):
                            pyflex.set_camPos(self.camPos_list[j])
                            pyflex.set_camAngle(self.camAngle_list[j])

                            # create dir with cameras
                            cam_dir = os.path.join(dir, 'camera_%d' % (j))
                            os.system('mkdir -p %s' % (cam_dir))

                            img = self.render()
                            cv2.imwrite(os.path.join(cam_dir, '%d_color.jpg' % self.count), img[:, :, :3][..., ::-1])
                            cv2.imwrite(os.path.join(cam_dir, '%d_depth.png' % self.count), (img[:, :, -1]*1000).astype(np.uint16))
                            if j == 0:
                                with open(os.path.join(cam_dir, '%d_particles.npy' % self.count), 'wb') as f:
                                    np.save(f, self.get_positions().reshape(-1, 4))
                                with open(os.path.join(cam_dir, '%d_endeffector.npy' % self.count), 'wb') as f:
                                    np.save(f, end_effector_pos)
                        self.count += 1
                    
                self.reset_robot()

                if math.isnan(self.get_positions().reshape(-1, 4)[:, 0].max()):
                    print('simulator exploded when action is', action)
                    return None
                        
            self.last_ee = end_effector_pos.copy()
        
        self.reset_robot()
        
        for i in range(2):
            pyflex.step()
        
        # save final rendering
        if dir != None:
            for j in range(len(self.camPos_list)):
                pyflex.set_camPos(self.camPos_list[j])
                pyflex.set_camAngle(self.camAngle_list[j])

                # create dir with cameras
                cam_dir = os.path.join(dir, 'camera_%d' % (j))
                os.system('mkdir -p %s' % (cam_dir))

                img = self.render()
                cv2.imwrite(os.path.join(cam_dir, '%d_color.jpg' % self.count), img[:, :, :3][..., ::-1])
                cv2.imwrite(os.path.join(cam_dir, '%d_depth.png' % self.count), (img[:, :, -1]*1000).astype(np.uint16))
                if j == 0:
                    with open(os.path.join(cam_dir, '%d_particles.npy' % self.count), 'wb') as f:
                        np.save(f, self.get_positions().reshape(-1, 4))
                    with open(os.path.join(cam_dir, '%d_endeffector.npy' % self.count), 'wb') as f:
                            np.save(f, np.array([-2., 0., h]))
            self.count += 1
        
        obs = self.render()
        return obs, self.count, self.contact
    
    def render(self, no_return=False):
        pyflex.step()
        if no_return:
            return
        else:
            return pyflex.render(render_depth=True).reshape(self.screenHeight, self.screenWidth, 5)
    
    def close(self):
        pyflex.clean()
    
    def sample_action(self):
        positions = self.get_positions().reshape(-1, 4)
        positions[:, 2] *= -1 # align with the coordinates
        num_points = positions.shape[0]
        pos_xz = positions[:, [0, 2]]

        # random choose a start point which can not be overlapped with the object
        while True:
            startpoint_pos = np.random.uniform(-self.wkspc_w, self.wkspc_w, size=(1, 2))
            if np.min(cdist(startpoint_pos, pos_xz)) > 0.2:
                break
        startpoint_pos = startpoint_pos.reshape(-1)

        # choose end points which is the expolation of the start point and obj point
        while True:
            pickpoint = np.random.randint(0, num_points - 1)
            obj_pos = positions[pickpoint, [0, 2]]
            slope = (obj_pos[1] - startpoint_pos[1]) / (obj_pos[0] - startpoint_pos[0])
            if obj_pos[0] < startpoint_pos[0]:
                x_end = obj_pos[0] - rand_float(0.1, 0.2)
            else:
                x_end = obj_pos[0] + rand_float(0.1, 0.2)
            y_end = slope * (x_end - startpoint_pos[0]) + startpoint_pos[1]
            endpoint_pos = np.array([x_end, y_end])
            if obj_pos[0] != startpoint_pos[0] and np.abs(x_end) < 1.5 and np.abs(y_end) < 1.5:
                break
        
        action = np.concatenate([startpoint_pos.reshape(-1), endpoint_pos.reshape(-1)], axis=0)
        return action
    
    def inside_workspace(self):
        pos = self.get_positions().reshape(-1, 4)
        if (pos[:, 0] > 3.0).any() or (pos[:, 2] > 3.0).any():
            return False
        else:
            return True
    
    def get_positions(self):
        return pyflex.get_positions()
    
    def get_faces(self):
        return pyflex.get_faces()

    def get_camera_intrinsics(self):
        projMat = pyflex.get_projMatrix().reshape(4, 4).T 
        cx = self.screenWidth / 2.0
        cy = self.screenHeight / 2.0
        fx = projMat[0, 0] * cx
        fy = projMat[1, 1] * cy
        camera_intrinsic_params = np.array([fx, fy, cx, cy])
        return camera_intrinsic_params
    
    def get_camera_extrinsics(self):
        return pyflex.get_viewMatrix().reshape(4, 4).T
    
    def get_camera_params(self):
        return self.cam_intrinsic_params, self.cam_extrinsic_matrix
    
    def get_property(self):
        return self.property
    
    def get_num_particles(self):
        return self.get_positions().reshape(-1, 4).shape[0]
    
    def get_obj_center(self):
        particle_pos = self.get_positions().reshape(-1, 4)
        particle_x = particle_pos[:, 0]
        particle_z = particle_pos[:, 2]
        center_x, center_z = np.median(particle_x), np.median(particle_z)
        return center_x, center_z

            
