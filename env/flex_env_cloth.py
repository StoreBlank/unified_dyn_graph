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
from utils_env import fps_with_idx, quaternion_multuply

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

        # define action space
        self.action_dim = 4

        # define property space
        self.property = None
        self.physics = config['dataset']['physics']
        
        # others
        self.count = 0
        self.particle_pos_list = []
        self.eef_states_list = []
        self.step_list = []
        self.contact_list = []
        
        self.fps = config['dataset']['fps']
        self.fps_number = config['dataset']['fps_number']
        self.obj_shape_states = None
        
        # carrots: 180mm others: 100mm
        if self.obj in ['carrots']:
            self.stick_len = 1.3
        else:
            self.stick_len = 1.0
        
    
    ###TODO: action class
    def _set_pos(self, picker_pos, particle_pos):
        """For gripper and grasp task."""
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[:, 3:6] = shape_states[:, :3] #picker_pos
        shape_states[:, :3] = picker_pos
        pyflex.set_shape_states(shape_states)
        pyflex.set_positions(particle_pos)
    
    def _reset_pos(self, particle_pos):
        """For gripper and grasp task."""
        pyflex.set_positions(particle_pos)
    
    def robot_close_gripper(self, close, jointPoses=None):
        """For gripper and grasp task."""
        for j in range(8, self.num_joints):
            pyflex.resetJointState(self.flex_robot_helper, j, close)
        
        pyflex.set_shape_states(self.robot_to_shape_states(pyflex.getRobotShapeStates(self.flex_robot_helper)))            
    
    def robot_open_gripper(self):
        """For gripper and grasp task."""
        for j in range(8, self.num_joints):
            pyflex.resetJointState(self.flex_robot_helper, j, 0.0)
    
    ### shape states
    def robot_to_shape_states(self, robot_states):
        n_robot_links = robot_states.shape[0]
        n_table = self.table_shape_states.shape[0]
        
        if self.obj_shape_states == None: #TODO
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
        if obj == 'cloth':
            
            # particle bigger, cloth more stiff
            particle_r = 0.03 #0.03
            cloth_pos = [-0.5, 1., 0.0]
            # cloth_size = np.array([rand_float(1.0, 1.8), rand_float(1.0, 1.8)]) * 50 
            cloth_size = np.array([1., 1.]) * 70.
            
            """
            stretch stiffness: resistance to lengthwise stretching
            bend stiffness: resistance to bending
            shear stiffness: resistance to forces that cause sliding or twisting deformation
            """
            # stretch_stiffness = 1.0  # ~elasticity remain the same
            # bend_stiffness = 1.0 #rand_float(0.1, 2.0)
            # shear_stiffness = 1.0 #rand_float(0.1, 2.0)
            # # (max:2.0)
            # stiffness = np.array([stretch_stiffness, bend_stiffness, shear_stiffness]) * 0.1
            # # stiffness = np.array([stretch_stiffness, bend_stiffness, shear_stiffness]) * property_params[1]  # [stretch, bend, shear] 
            # stiffness[0] = np.clip(stiffness[0], 1.0, 1.6)
            # stiffness[2] = np.clip(stiffness[2], 0.1, 1.6)
            
            # dynamicFriction = property_params[0] #rand_float(0.1, 1.) 
            # dynamicFriction = 3.0 # (0.1, 1.0)
            
            # TODO: margnify the differences
            sf = np.random.rand()
            stiffness_factor = sf * 1.4 + 0.1
            stiffness = np.array([1.0, 1.0, 1.0]) * stiffness_factor 
            stiffness[0] = np.clip(stiffness[0], 1.0, 1.5)
            dynamicFriction = -sf * 0.9 + 1.0
            
            cloth_mass = 0.1 
            
            render_mode = 1 # 1: particles; 2: mesh
            flip_mesh = 0
            
            staticFriction = 0.0 
            particleFriction = 0.0
            
            self.scene_params = np.array([
                *cloth_pos,
                *cloth_size,
                *stiffness,
                cloth_mass,
                particle_r,
                render_mode,
                flip_mesh, 
                dynamicFriction, staticFriction, particleFriction])
            
            # cloth
            temp = np.array([0])
            pyflex.set_scene(29, self.scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0) 
            
            self.property = {'particle_radius': particle_r,
                             'num_particles': self.get_num_particles(),
                            #  'cloth_mass': cloth_mass,
                             'stretch_stiffness': stiffness[0],
                             'bend_stiffness': stiffness[1],
                             'shear_stiffness': stiffness[2],
                             'dynamic_friction': dynamicFriction,
                            #  'static_friction': staticFriction,
                            #  'particle_friction': particleFriction,
                            'sf': sf,
                            }
            
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
        wkspace_width = 3.5 # 3.5*2=7 grid = 700mm
        wkspace_length = 4.5 # 4.5*2=9 grid = 900mm
        halfEdge = np.array([wkspace_width, wkspace_height, wkspace_length])
        center = np.array([0.0, 0.0, 0.0])
        quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
        hideShape = 0
        color = np.ones(3) * (160. / 255.)
        pyflex.add_box(halfEdge, center, quats, hideShape, color)
        self.table_shape_states[0] = np.concatenate([center, center, quats, quats])
        
        # table for robot
        robot_table_height = 0.5+1.0
        robot_table_width = 126 / 200 # 126mm
        robot_table_length = 126 / 200 # 126mm
        halfEdge = np.array([robot_table_width, robot_table_height, robot_table_length])
        center = np.array([-wkspace_width-robot_table_width, 0.0, 0.0])
        quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
        hideShape = 0
        color = np.ones(3) * (160. / 255.)
        pyflex.add_box(halfEdge, center, quats, hideShape, color)
        self.table_shape_states[1] = np.concatenate([center, center, quats, quats])
        
        ## add robot
        if self.gripper:
            robot_base_pos = [-wkspace_width-0.6, 0., wkspace_height+1.0]
            robot_base_orn = [0, 0, 0, 1]
            self.robotId = pyflex.loadURDF(self.flex_robot_helper, 'assets/xarm/xarm6_with_gripper_2.urdf', robot_base_pos, robot_base_orn, globalScaling=10.0) 
            self.rest_joints = np.zeros(13)
        
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
                
                if self.cam_intrinsic_params[j].sum() == 0 or self.cam_extrinsic_matrix[j].sum() == 0:
                        self.cam_intrinsic_params[j] = self.get_camera_intrinsics()
                        self.cam_extrinsic_matrix[j] = self.get_camera_extrinsics()

                img = self.render()
                # rgb and depth images
                cv2.imwrite(os.path.join(cam_dir, '%d_color.jpg' % self.count), img[:, :, :3][..., ::-1])
                cv2.imwrite(os.path.join(cam_dir, '%d_depth.png' % self.count), (img[:, :, -1]*1000).astype(np.uint16))
                if j == 0:
                    # save particle pos
                    particles = self.get_positions().reshape(-1, 4)
                    particles_pos = particles[:, :3]
                    if self.fps:
                        _, self.sampled_idx = fps_with_idx(particles_pos, self.fps_number)
                        particles_pos = particles_pos[self.sampled_idx]
                    self.particle_pos_list.append(particles_pos)
                    # save eef pos
                    robot_shape_states = pyflex.getRobotShapeStates(self.flex_robot_helper)
                    eef_states = np.zeros((2, 14))
                    eef_states[0] = robot_shape_states[9] # left finger
                    eef_states[1] = robot_shape_states[12] # right finger
                    self.eef_states_list.append(eef_states)
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
                    # save particle pos
                    particles = self.get_positions().reshape(-1, 4)
                    particles_pos = particles[:, :3]
                    if self.fps:
                        particles_pos = particles_pos[self.sampled_idx]
                    self.particle_pos_list.append(particles_pos)
                    # save eef pos
                    robot_shape_states = pyflex.getRobotShapeStates(self.flex_robot_helper)
                    eef_states = np.zeros((2, 14))
                    eef_states[0] = robot_shape_states[9] # left finger
                    eef_states[1] = robot_shape_states[12] # right finger
                    self.eef_states_list.append(eef_states)
            self.count += 1
            self.step_list.append(self.count)
        
        return self.particle_pos_list, self.eef_states_list, self.step_list, self.contact_list
        
    def step(self, action, dir=None, particle_pos_list = None, eef_states_list = None, step_list = None, contact_list = None):
        if dir != None:
            self.particle_pos_list = particle_pos_list
            self.eef_states_list = eef_states_list
            self.step_list = step_list
            self.contact_list = contact_list
            self.count = self.step_list[-1]
        
        if self.gripper:
            h = 0.5 + 1.8
        else:
            h = 0.5 + self.stick_len
        s_2d = np.concatenate([action[:2], [h]])
        e_2d = np.concatenate([action[2:], [h]])

        # pusher angle depending on x-axis
        if (s_2d - e_2d)[0] == 0:
            pusher_angle = np.pi/2
        else:
            # pusher_angle = np.arctan((s_2d - e_2d)[1] / (s_2d - e_2d)[0]) #TODO
            pusher_angle = -np.arctan((s_2d - e_2d)[1] / (s_2d - e_2d)[0])
            # pusher_angle = -np.pi/4
        # pusher_angle = np.pi/2
        
        # robot orientation
        orn = np.array([0.0, np.pi, pusher_angle + np.pi/2])

        # create way points
        if self.cont_motion: #TODO - strange
            if self.last_ee is None:
                self.reset_robot(self.rest_joints)
                self.last_ee = s_2d
            way_points = [self.last_ee, s_2d, e_2d]
        else:
            if self.grasp:
                way_points = [s_2d + [0., 0., 0.1], s_2d, s_2d, e_2d + [0., 0., 0.1], e_2d]
            else:
                way_points = [s_2d + [0., 0., 0.2], s_2d, e_2d, e_2d + [0., 0., 0.2]]
            self.reset_robot(self.rest_joints)

        # set robot speed
        speed = 1.0/100.
        
        # set up gripper
        if self.gripper:
            if self.grasp:
                self.robot_open_gripper()
            else:
                self.robot_close_gripper(0.7)
                
        for i_p in range(len(way_points)-1):
            s = way_points[i_p]
            e = way_points[i_p+1]
            steps = int(np.linalg.norm(e-s)/speed) + 1
            
            for i in range(steps):
                end_effector_pos = s + (e-s) * i / steps # expected eef position
                end_effector_orn = p.getQuaternionFromEuler(orn)
                jointPoses = p.calculateInverseKinematics(self.robotId, 
                                                        self.end_idx, 
                                                        end_effector_pos, 
                                                        end_effector_orn, 
                                                        self.joints_lower.tolist(), 
                                                        self.joints_upper.tolist(),
                                                        (self.joints_upper - self.joints_lower).tolist(),
                                                        self.rest_joints)
                # print('jointPoses:', jointPoses)
                self.reset_robot(jointPoses)
                pyflex.step()
                
                ## gripper control
                if self.gripper and self.grasp and i_p >= 1:
                    grasp_thresd = 0.1 #0.1
                    obj_pos = self.get_positions().reshape(-1, 4)[:, :3]
                    new_particle_pos = self.get_positions().reshape(-1, 4).copy()
                    
                    ### grasping 
                    if i_p == 1:
                        close = 0
                        start = 0
                        end = 1.0
                        close_steps = 50 #500
                        finger_y = 0.5
                        for j in range(close_steps):
                            robot_shape_states = pyflex.getRobotShapeStates(self.flex_robot_helper) # 9: left finger; 12: right finger
                            left_finger_pos, right_finger_pos = robot_shape_states[9][:3], robot_shape_states[12][:3]
                            left_finger_pos[1], right_finger_pos[1] = left_finger_pos[1] - finger_y, right_finger_pos[1] - finger_y 
                            new_finger_pos = (left_finger_pos + right_finger_pos) / 2
                            
                            if j == 0:
                                # fine the k pick point
                                pick_k = 10
                                left_min_dist, left_pick_index = find_min_distance(left_finger_pos, obj_pos, pick_k)
                                right_min_dist, right_pick_index = find_min_distance(right_finger_pos, obj_pos, pick_k)
                                min_dist, pick_index = find_min_distance(new_finger_pos, obj_pos, pick_k)
                                # save the original setting for restoring
                                pick_origin = new_particle_pos[pick_index]
                            
                            if left_min_dist <= grasp_thresd or right_min_dist <= grasp_thresd:
                                    new_particle_pos[left_pick_index, :3] = left_finger_pos
                                    new_particle_pos[left_pick_index, 3] = 0
                                    new_particle_pos[right_pick_index, :3] = right_finger_pos
                                    new_particle_pos[right_pick_index, 3] = 0
                            self._set_pos(new_finger_pos, new_particle_pos)
                            
                            # close the gripper slowly 
                            close += (end - start) / close_steps
                            self.robot_close_gripper(close)
                            pyflex.step()
                    
                    # find finger positions
                    robot_shape_states = pyflex.getRobotShapeStates(self.flex_robot_helper) # 9: left finger; 12: right finger
                    left_finger_pos, right_finger_pos = robot_shape_states[9][:3], robot_shape_states[12][:3]
                    left_finger_pos[1], right_finger_pos[1] = left_finger_pos[1] - finger_y, right_finger_pos[1] - finger_y
                    new_finger_pos = (left_finger_pos + right_finger_pos) / 2
                    # connect pick pick point to the finger
                    new_particle_pos[pick_index, :3] = new_finger_pos
                    new_particle_pos[pick_index, 3] = 0
                    self._set_pos(new_finger_pos, new_particle_pos)
                
                # reset robot
                self.reset_robot(jointPoses)
                pyflex.step()

                # save img in each step
                obj_pos = self.get_positions().reshape(-1, 4)[:, [0, 2]]
                obj_pos[:, 1] *= -1
                robot_obj_dist = np.min(cdist(end_effector_pos[:2].reshape(1, 2), obj_pos))
                
                if dir != None:
                    if robot_obj_dist < 0.2 and i % 10 == 0: #contact
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
                                # save particle pos
                                particles = self.get_positions().reshape(-1, 4)
                                particles_pos = particles[:, :3]
                                if self.fps:
                                    particles_pos = particles_pos[self.sampled_idx]
                                self.particle_pos_list.append(particles_pos)
                                # save eef pos
                                robot_shape_states = pyflex.getRobotShapeStates(self.flex_robot_helper)
                                eef_states = np.zeros((2, 14))
                                eef_states[0] = robot_shape_states[9] # left finger
                                eef_states[1] = robot_shape_states[12] # right finger
                                self.eef_states_list.append(eef_states)  
                        self.count += 1
                        self.contact_list.append(self.count)
                        
                    elif i % 30 == 0:
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
                                # save particle pos
                                particles = self.get_positions().reshape(-1, 4)
                                particles_pos = particles[:, :3]
                                if self.fps:
                                    particles_pos = particles_pos[self.sampled_idx]
                                self.particle_pos_list.append(particles_pos)
                                # save eef pos
                                robot_shape_states = pyflex.getRobotShapeStates(self.flex_robot_helper)
                                eef_states = np.zeros((2, 14))
                                eef_states[0] = robot_shape_states[9] # left finger
                                eef_states[1] = robot_shape_states[12] # right finger
                                self.eef_states_list.append(eef_states)
                        self.count += 1
                    
                self.reset_robot()

                if math.isnan(self.get_positions().reshape(-1, 4)[:, 0].max()):
                    print('simulator exploded when action is', action)
                    return None
                        
            self.last_ee = end_effector_pos.copy()
        
        # set up gripper
        if self.gripper:
            if self.grasp:
                self.robot_open_gripper()
            else:
                self.robot_close_gripper(1.0)
        
        # reset the mass for the pick points
        if self.gripper and self.grasp:
            new_particle_pos[pick_index, 3] = pick_origin[:, 3]
            self._reset_pos(new_particle_pos)
        
        
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
                    # save particle pos
                    particles = self.get_positions().reshape(-1, 4)
                    particles_pos = particles[:, :3]
                    if self.fps:
                        particles_pos = particles_pos[self.sampled_idx]
                    self.particle_pos_list.append(particles_pos)
                    # save eef pos
                    robot_shape_states = pyflex.getRobotShapeStates(self.flex_robot_helper)
                    eef_states = np.zeros((2, 14))
                    eef_states[0] = robot_shape_states[9] # left finger
                    eef_states[1] = robot_shape_states[12] # right finger
                    self.eef_states_list.append(eef_states)
            self.count += 1
            self.step_list.append(self.count)
        
        obs = self.render()
        return obs, self.particle_pos_list, self.eef_states_list, self.step_list, self.contact_list
    
    def render(self, no_return=False):
        pyflex.step()
        if no_return:
            return
        else:
            return pyflex.render(render_depth=True).reshape(self.screenHeight, self.screenWidth, 5)
    
    def close(self):
        pyflex.clean()
    
    def sample_action(self, init=False, boundary_points=None):
        # action = self.sample_grasp_actions()
        action = self.sample_grasp_actions_corner(init, boundary_points)
        return action
    
    def sample_grasp_actions(self):
        positions = self.get_positions().reshape(-1, 4)
        positions[:, 2] *= -1 # align with the coordinates
        num_points = positions.shape[0]
        
        # random pick a point as start point
        startpoint_pos = positions[np.random.choice(num_points), [0, 2]]
        # choose end points which is outside the obj
        valid = False
        for _ in range(1000):
            endpoint_pos = np.random.uniform(-self.wkspc_w, self.wkspc_w, size=(1, 2))
            if np.min(cdist(endpoint_pos, positions[:, [0, 2]])) > 0.2:
                valid = True
                break
        
        if valid:
            action = np.concatenate([startpoint_pos.reshape(-1), endpoint_pos.reshape(-1)], axis=0)
        else:
            action = None
        return action

    def sample_grasp_actions_corner(self, init=False, boundary_points=None, boundary=None):
        positions = self.get_positions().reshape(-1, 4)
        positions[:, 2] *= -1
        particle_x, particle_y, particle_z = positions[:, 0], positions[:, 1], positions[:, 2]
        x_min, y_min, z_min = np.min(particle_x), np.min(particle_y), np.min(particle_z)
        x_max, y_max, z_max = np.max(particle_x), np.max(particle_y), np.max(particle_z)
        
        # choose the starting point at the boundary of the object
        if init: # record boundary points
            boundary_points = []
            boundary = []
            for idx, point in enumerate(positions):
                if point[0] == x_max:
                    boundary_points.append(idx)
                    boundary.append(1)
                elif point[0] == x_min:
                    boundary_points.append(idx)
                    boundary.append(2)
                elif point[2] == z_max:
                    boundary_points.append(idx)
                    boundary.append(3)
                elif point[2] == z_min:
                    boundary_points.append(idx)
                    boundary.append(4)
        assert len(boundary_points) == len(boundary)
        
        
        # random pick a point as start point
        valid = False
        for _ in range(1000):
            pick_idx = np.random.choice(len(boundary_points))
            startpoint_pos = positions[boundary_points[pick_idx], [0, 2]]
            endpoint_pos = startpoint_pos.copy()
            # choose end points which is outside the obj
            move_distance = rand_float(1.0, 1.5)
            # if startpoint_pos[0] >= x_max:
            #     endpoint_pos[0] += move_distance
            # elif startpoint_pos[0] <= x_min:
            #     endpoint_pos[0] -= move_distance
            # elif startpoint_pos[1] >= z_max:
            #     endpoint_pos[1] += move_distance
            # elif startpoint_pos[1] <= z_min:
            #     endpoint_pos[1] -= move_distance
            
            if boundary[pick_idx] == 1:
                endpoint_pos[0] += move_distance 
            elif boundary[pick_idx] == 2:
                endpoint_pos[0] -= move_distance
            elif boundary[pick_idx] == 3:
                endpoint_pos[1] += move_distance
            elif boundary[pick_idx] == 4:
                endpoint_pos[1] -= move_distance
            
            if np.abs(endpoint_pos[0]) < 3.5 and np.abs(endpoint_pos[1]) < 2.5:
                valid = True
                break
        
        
        if valid:
            action = np.concatenate([startpoint_pos.reshape(-1), endpoint_pos.reshape(-1)], axis=0)
        else:
            action = None
        
        return action, boundary_points, boundary
        
    
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
        particle_x, particle_y, particle_z = particle_pos[:, 0], particle_pos[:, 1], particle_pos[:, 2]
        center_x, center_y, center_z = np.mean(particle_x), np.mean(particle_y), np.mean(particle_z)
        return center_x, center_y, center_z
    
    def get_obj_size(self):
        particle_pos = self.get_positions().reshape(-1, 4)
        particle_x, particle_y, particle_z = particle_pos[:, 0], particle_pos[:, 1], particle_pos[:, 2]
        size_x, size_y, size_z = np.max(particle_x) - np.min(particle_x), np.max(particle_y) - np.min(particle_y), np.max(particle_z) - np.min(particle_z)
        return size_x, size_y, size_z

    def get_obj_corners(self):
        particle_pos = self.get_positions().reshape(-1, 4)
        particle_x, particle_y, particle_z = particle_pos[:, 0], particle_pos[:, 1], particle_pos[:, 2]
        x_min, y_min, z_min = np.min(particle_x), np.min(particle_y), np.min(particle_z)
        x_max, y_max, z_max = np.max(particle_x), np.max(particle_y), np.max(particle_z)
        corner_0 = np.array([x_min, y_max, z_min])
        corner_1 = np.array([x_min, y_max, z_max])
        corner_2 = np.array([x_max, y_max, z_max])
        corner_3 = np.array([x_max, y_max, z_min])
        
        return corner_0, corner_1, corner_2, corner_3
            



