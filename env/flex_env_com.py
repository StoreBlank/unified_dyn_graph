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
        self.eef_pos_list = []
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
        if obj == "rigid_object":
            
            """
            1: box
            2: cube_mesh
            3: cracker_box
            4: sugar_box
            5: tomato_soup_can
            6: mustard_bottle
            7: tuna_fish_can
            8: pudding_box
            9: gelatin_box
            10: strawberry
            11: apple
            12: lemon
            13: peach
            14: pear
            15: orange
            16: pitcher_base
            17: bleach_cleanser
            18: bowl
            19: power_drill
            20: wood_block
            """
            obj_types = range(3, 21)
            obj_sizes = [1.0, 1.0, 0.7, 0.8, 0.6, 0.6, 0.6, 0.2, #3-10
                         0.3, 0.3, 0.4, 0.4, 0.35, 0.8, 0.8, 0.8, 0.8, 0.8] #11-20
            
            index = 3-3 #np.random.randint(0, len(obj_types))
            
            x = 0.
            y = 1. #3.5
            z = 0. #-3.3
            obj_type = 2 #obj_types[index]
            size = 0.1 #obj_sizes[index]
            draw_mesh = 0

            radius = 0.1
            
            mass = 1 #rand_float(0.1, 10.) #10g-1000g
            mass_weight = 1e50
            
            rigidStiffness = 1.
            dynamicFriction = 0.3 #rand_float(0.1, 0.7)
            staticFriction = 0.
            viscosity = 2.
            
            rotation = 0. #rand_float(0., 360.)
            springStiffness = 1.0

            self.scene_params = np.array([x, y, z, size, obj_type, draw_mesh,
                                          radius, mass, rigidStiffness, dynamicFriction, staticFriction, 
                                          viscosity, rotation, springStiffness])
            
            temp = np.array([0])
            pyflex.set_scene(25, self.scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0) 

            self.property = {'object_type': obj_type,
                            'particle_radius': radius,
                            'num_particles': self.get_num_particles(),
                            'mass': mass,
                            'dynamic_friction': dynamicFriction}
            
            num_particles = self.get_num_particles()
            print('num_particles:', num_particles)
            
            # rest_positions = pyflex.get_restPositions().reshape(-1, 4)
            # print('rest_positions:', rest_positions.shape)
            # print('rest_positions:', rest_positions)
            
            particle_pos = self.get_positions().reshape(-1, 4)
            # print('particle_pos:', particle_pos.shape)
            # print('particle_pos:', particle_pos)
            
            ### change center of mass
            n_weight = 1
            if n_weight > 0:
                new_particle_pos = self.get_positions().reshape(-1, 4).copy()
                particle_x, particle_y, particle_z = new_particle_pos[:, 0], new_particle_pos[:, 1], new_particle_pos[:, 2]
                center_x, center_y, center_z = np.mean(particle_x), np.mean(particle_y), np.mean(particle_z)
                min_x, min_y, min_z = np.min(particle_x), np.min(particle_y), np.min(particle_z)
                max_x, max_y, max_z = np.max(particle_x), np.max(particle_y), np.max(particle_z)
                # print('center:', center_x, center_y, center_z)
                # print('min:', min_x, min_y, min_z)
                # print('max:', max_x, max_y, max_z)
                p0 = np.array([min_x, min_y, min_z]) #np.array([(min_x+center_x)/2, min_y, (min_z+center_z)/2])
                p1 = np.array([max_x, min_y, min_z]) #np.array([(max_x+center_x)/2, min_y, (min_z+center_z)/2])
                p2 = np.array([max_x, min_y, max_z]) #np.array([(max_x+center_x)/2, min_y, (max_z+center_z)/2])
                p3 = np.array([min_x, min_y, max_z]) #np.array([(min_x+center_x)/2, min_y, (max_z+center_z)/2])
                ps = [p0, p1, p2, p3]
                # print('ps:', ps)
                # print(ps)

                if n_weight == 1:
                    choose_pos = 0 #np.random.randint(0, 4)
                    
                    def choose_one_pos(choose_pos, ps, particle_pos):
                        out_particle_pos = particle_pos.copy()
                        com_index = []
                        for i, particle in enumerate(particle_pos):
                            if np.linalg.norm(particle[:3] - ps[choose_pos]) < 1.0:
                                out_particle_pos[i, 3] = 1/mass_weight #invMass
                                com_index.append(i)
                        return out_particle_pos, com_index
                            
                    new_particle_pos, com_index = choose_one_pos(choose_pos, ps, new_particle_pos)
                    # import ipdb; ipdb.set_trace()
                    # print(new_particle_pos)
                    # print(len(com_index))
                    # print(com_index)
                    pyflex.set_positions(new_particle_pos)

        
        elif obj == "softbox":
            # x = 8.0
            # y = 8.0
            # z = 8.0
            # clusterStiffness = 0.2
            # clusterPlasticThreshold = 0.0
            # clusterPlasticCreep = 0.0
            # scene_params = np.array([x, y, z, clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep])
            # temp = np.array([0])
            # pyflex.set_scene(5, scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0)
            
            radius = 0.1
            scale = np.array([1., 1., 1.]) * 20
            cluster_spacing = 10 #rand_float(2, 8) # change the stiffness of the rope
            dynamicFriction = 0.3 #rand_float(0.1, 0.45)
            
            trans = [-0.0, 2., 0.0]
            
            z_rotation = 0. #rand_float(60, 70)
            y_rotation = 0. 
            rot_1 = Rotation.from_euler('xyz', [0, y_rotation, 0.], degrees=True)
            rotate_1 = rot_1.as_quat()
            rot_2 = Rotation.from_euler('xyz', [0, 0, z_rotation], degrees=True)
            rotate_2 = rot_2.as_quat()
            rotate = quaternion_multuply(rotate_1, rotate_2)
            
            cluster_radius = 0.
            cluster_stiffness = 0.2

            link_radius = 0. 
            link_stiffness = 1.

            global_stiffness = 1.

            surface_sampling = 0.
            volume_sampling = 4.

            skinning_falloff = 5.
            skinning_max_dist = 100.

            cluster_plastic_threshold = 0.
            cluster_plastic_creep = 0.

            particleFriction = 0.
            
            draw_mesh = 0

            relaxtion_factor = 1.
            collisionDistance = radius * 0.5
            
            self.scene_params = np.array([*scale, *trans, radius, 
                                            cluster_spacing, cluster_radius, cluster_stiffness,
                                            link_radius, link_stiffness, global_stiffness,
                                            surface_sampling, volume_sampling, skinning_falloff, skinning_max_dist,
                                            cluster_plastic_threshold, cluster_plastic_creep,
                                            dynamicFriction, particleFriction, draw_mesh, relaxtion_factor, 
                                            *rotate, collisionDistance])
            
            temp = np.array([0])
            pyflex.set_scene(38, self.scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0) 
            
            ### change center of mass
            mass_weight = 1e30
            n_weight = 1
            if n_weight > 0:
                new_particle_pos = self.get_positions().reshape(-1, 4).copy()
                particle_x, particle_y, particle_z = new_particle_pos[:, 0], new_particle_pos[:, 1], new_particle_pos[:, 2]
                center_x, center_y, center_z = np.mean(particle_x), np.mean(particle_y), np.mean(particle_z)
                min_x, min_y, min_z = np.min(particle_x), np.min(particle_y), np.min(particle_z)
                max_x, max_y, max_z = np.max(particle_x), np.max(particle_y), np.max(particle_z)
                # print('center:', center_x, center_y, center_z)
                # print('min:', min_x, min_y, min_z)
                # print('max:', max_x, max_y, max_z)
                p0 = np.array([min_x, min_y, min_z]) #np.array([(min_x+center_x)/2, min_y, (min_z+center_z)/2])
                p1 = np.array([max_x, min_y, min_z]) #np.array([(max_x+center_x)/2, min_y, (min_z+center_z)/2])
                p2 = np.array([max_x, min_y, max_z]) #np.array([(max_x+center_x)/2, min_y, (max_z+center_z)/2])
                p3 = np.array([min_x, min_y, max_z]) #np.array([(min_x+center_x)/2, min_y, (max_z+center_z)/2])
                ps = [p0, p1, p2, p3]
                # print('ps:', ps)
                # print(ps)

                if n_weight == 1:
                    choose_pos = 0 #np.random.randint(0, 4)
                    
                    def choose_one_pos(choose_pos, ps, particle_pos):
                        out_particle_pos = particle_pos.copy()
                        com_index = []
                        for i, particle in enumerate(particle_pos):
                            if np.linalg.norm(particle[:3] - ps[choose_pos]) < 1.0:
                                out_particle_pos[i, 3] = 1/mass_weight #invMass
                                com_index.append(i)
                        return out_particle_pos, com_index
                            
                    new_particle_pos, com_index = choose_one_pos(choose_pos, ps, new_particle_pos)
                    # import ipdb; ipdb.set_trace()
                    # print(new_particle_pos)
                    # print(len(com_index))
                    # print(com_index)
                    pyflex.set_positions(new_particle_pos)
        
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
        robot_table_height = 0.5
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
        robot_base_pos = [-wkspace_width-0.6, 0., wkspace_height]
        robot_base_orn = [0, 0, 0, 1]
        self.robotId = pyflex.loadURDF(self.flex_robot_helper, 'assets/xarm/xarm6_with_gripper.urdf', robot_base_pos, robot_base_orn, globalScaling=10.0) 
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
                
                if self.cam_intrinsic_params[j].sum() == 0 or self.cam_extrinsic_matrix[j].sum() == 0:
                        self.cam_intrinsic_params[j] = self.get_camera_intrinsics()
                        self.cam_extrinsic_matrix[j] = self.get_camera_extrinsics()

                img = self.render()
                # rgb and depth images
                cv2.imwrite(os.path.join(cam_dir, '%d_color.jpg' % self.count), img[:, :, :3][..., ::-1])
                # cv2.imwrite(os.path.join(cam_dir, '%d_depth.png' % self.count), (img[:, :, -1]*1000).astype(np.uint16))
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
                    eef_pos = robot_shape_states[-1][:3] # actual eef position
                    eef_pos[1] -= self.stick_len
                    self.eef_pos_list.append(eef_pos)
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
                # cv2.imwrite(os.path.join(cam_dir, '%d_depth.png' % self.count), (img[:, :, -1]*1000).astype(np.uint16))
                if j == 0:
                    # save particle pos
                    particles = self.get_positions().reshape(-1, 4)
                    particles_pos = particles[:, :3]
                    if self.fps:
                        particles_pos = particles_pos[self.sampled_idx]
                    self.particle_pos_list.append(particles_pos)
                    # save eef pos
                    robot_shape_states = pyflex.getRobotShapeStates(self.flex_robot_helper)
                    eef_pos = robot_shape_states[-1][:3] # actual eef position
                    eef_pos[1] -= self.stick_len
                    self.eef_pos_list.append(eef_pos)
            self.count += 1
            self.step_list.append(self.count)
        
        return self.particle_pos_list, self.eef_pos_list, self.step_list, self.contact_list
        
    def step(self, action, dir=None, particle_pos_list = None, eef_pos_list = None, step_list = None, contact_list = None):
        if dir != None:
            self.particle_pos_list = particle_pos_list
            self.eef_pos_list = eef_pos_list
            self.step_list = step_list
            self.contact_list = contact_list
            self.count = self.step_list[-1]
        
        if self.gripper:
            h = 1.35 #TODO change
        else:
            h = 0.5 + self.stick_len
        s_2d = np.concatenate([action[:2], [h]])
        e_2d = np.concatenate([action[2:], [h]])

        # pusher angle depending on x-axis
        if (s_2d - e_2d)[0] == 0:
            pusher_angle = np.pi/2
        else:
            pusher_angle = np.arctan((s_2d - e_2d)[1] / (s_2d - e_2d)[0])
            # pusher_angle = -np.arctan((s_2d - e_2d)[1] / (s_2d - e_2d)[0])
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
            way_points = [s_2d + [0., 0., 0.2], s_2d, e_2d, e_2d + [0., 0., 0.2]]
            self.reset_robot(self.rest_joints)

        # set robot speed
        speed = 1.0/300.
        
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
                            # cv2.imwrite(os.path.join(cam_dir, '%d_depth.png' % self.count), (img[:, :, -1]*1000).astype(np.uint16))
                            if j == 0:
                                # save particle pos
                                particles = self.get_positions().reshape(-1, 4)
                                particles_pos = particles[:, :3]
                                if self.fps:
                                    particles_pos = particles_pos[self.sampled_idx]
                                self.particle_pos_list.append(particles_pos)
                                # save eef pos
                                robot_shape_states = pyflex.getRobotShapeStates(self.flex_robot_helper)
                                eef_pos = robot_shape_states[-1][:3] # actual eef position
                                eef_pos[1] -= self.stick_len
                                self.eef_pos_list.append(eef_pos)  
                        self.count += 1
                        self.contact_list.append(self.count)
                        
                    elif i % 20 == 0:
                        for j in range(len(self.camPos_list)):
                            pyflex.set_camPos(self.camPos_list[j])
                            pyflex.set_camAngle(self.camAngle_list[j])

                            # create dir with cameras
                            cam_dir = os.path.join(dir, 'camera_%d' % (j))
                            os.system('mkdir -p %s' % (cam_dir))

                            img = self.render()
                            cv2.imwrite(os.path.join(cam_dir, '%d_color.jpg' % self.count), img[:, :, :3][..., ::-1])
                            # cv2.imwrite(os.path.join(cam_dir, '%d_depth.png' % self.count), (img[:, :, -1]*1000).astype(np.uint16))
                            if j == 0:
                                # save particle pos
                                particles = self.get_positions().reshape(-1, 4)
                                particles_pos = particles[:, :3]
                                if self.fps:
                                    particles_pos = particles_pos[self.sampled_idx]
                                self.particle_pos_list.append(particles_pos)
                                # save eef pos
                                robot_shape_states = pyflex.getRobotShapeStates(self.flex_robot_helper)
                                eef_pos = robot_shape_states[-1][:3] # actual eef position
                                eef_pos[1] -= self.stick_len
                                self.eef_pos_list.append(eef_pos)
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
                # cv2.imwrite(os.path.join(cam_dir, '%d_depth.png' % self.count), (img[:, :, -1]*1000).astype(np.uint16))
                if j == 0:
                    # save particle pos
                    particles = self.get_positions().reshape(-1, 4)
                    particles_pos = particles[:, :3]
                    if self.fps:
                        particles_pos = particles_pos[self.sampled_idx]
                    self.particle_pos_list.append(particles_pos)
                    # save eef pos
                    robot_shape_states = pyflex.getRobotShapeStates(self.flex_robot_helper)
                    eef_pos = robot_shape_states[-1][:3] # actual eef position
                    eef_pos[1] -= self.stick_len
                    self.eef_pos_list.append(eef_pos)
            self.count += 1
            self.step_list.append(self.count)
        
        obs = self.render()
        return obs, self.particle_pos_list, self.eef_pos_list, self.step_list, self.contact_list
    
    def render(self, no_return=False):
        pyflex.step()
        if no_return:
            return
        else:
            return pyflex.render(render_depth=True).reshape(self.screenHeight, self.screenWidth, 5)
    
    def close(self):
        pyflex.clean()
    
    def sample_action(self):
        if self.obj in ['mustard_bottle', 'power_drill', 'rigid_object']:
            action = self.sample_rigid_actions()
        elif self.obj in ['Tshirt', 'carrots', 'coffee', 'rope']:
            action = self.sample_deform_actions()
        else:
            raise ValueError('action not defined')
        return action
    
    def sample_rigid_actions(self):
        positions = self.get_positions().reshape(-1, 4)
        positions[:, 2] *= -1 # align with the coordinates
        num_points = positions.shape[0]
        pos_xz = positions[:, [0, 2]]
        pos_x, pos_z = positions[:, 0], positions[:, 2]

        # choose end points within the limited region of workspace
        pickpoint = np.random.randint(0, num_points - 1)
        obj_pos = positions[pickpoint, [0, 2]]

        # check if the objects is close to the table edge
        table_edge = self.wkspc_w / 2
        action_thres = 0.1
        if np.min((pos_x-table_edge)**2) < action_thres:
            endpoint_pos = np.array([0., 0.])
            startpoint_pos = obj_pos + np.array([1., 0.])
        elif np.min((pos_x+table_edge)**2) < action_thres:
            endpoint_pos = np.array([0., 0.])
            startpoint_pos = obj_pos + np.array([-1., 0.])
        elif np.min((pos_z-table_edge)**2) < action_thres:
            endpoint_pos = np.array([0., 0.])
            startpoint_pos = obj_pos + np.array([0., 1.])
        elif np.min((pos_z+table_edge)**2) < action_thres:
            endpoint_pos = np.array([0., 0.])
            startpoint_pos = obj_pos + np.array([0., -1.])
        else:
            endpoint_pos = obj_pos 
            while True:
                np.random.uniform(-table_edge + 0.5, table_edge - 0.5, size=(1, 2))
                startpoint_pos = np.random.uniform(-self.wkspc_w // 2 + 0.5, self.wkspc_w // 2 - 0.5, size=(1, 2))
                if np.min(cdist(startpoint_pos, pos_xz)) > 0.2:
                    break
        
        action = np.concatenate([startpoint_pos.reshape(-1), endpoint_pos.reshape(-1)], axis=0)
        return action
    
    def sample_deform_actions(self):
        positions = self.get_positions().reshape(-1, 4)
        positions[:, 2] *= -1 # align with the coordinates
        num_points = positions.shape[0]
        pos_xz = positions[:, [0, 2]]

        # random choose a start point which can not be overlapped with the object
        valid = False
        for _ in range(100):
            startpoint_pos_origin = np.random.uniform(-self.wkspc_w // 2 - 1, self.wkspc_w // 2 + 1., size=(1, 2))
            startpoint_pos = startpoint_pos_origin.copy()
            startpoint_pos = startpoint_pos.reshape(-1)

            # choose end points which is the expolation of the start point and obj point
            pickpoint = np.random.randint(0, num_points)
            obj_pos = positions[pickpoint, [0, 2]]
            slope = (obj_pos[1] - startpoint_pos[1]) / (obj_pos[0] - startpoint_pos[0])
            if obj_pos[0] < startpoint_pos[0]:
                x_end = obj_pos[0] - rand_float(0.5, 1.0)
            else:
                x_end = obj_pos[0] + rand_float(0.5, 1.0)
            y_end = slope * (x_end - startpoint_pos[0]) + startpoint_pos[1]
            endpoint_pos = np.array([x_end, y_end])
            if obj_pos[0] != startpoint_pos[0] and np.abs(x_end) < 1.5 and np.abs(y_end) < 1.5 \
                and np.min(cdist(startpoint_pos_origin, pos_xz)) > 0.2:
                valid = True
                break
        
        if valid:
            action = np.concatenate([startpoint_pos.reshape(-1), endpoint_pos.reshape(-1)], axis=0)
        else:
            action = None
        
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
        particle_x, particle_y, particle_z = particle_pos[:, 0], particle_pos[:, 1], particle_pos[:, 2]
        center_x, center_y, center_z = np.median(particle_x), np.median(particle_y), np.median(particle_z)
        return center_x, center_y, center_z
    
    def get_obj_size(self):
        particle_pos = self.get_positions().reshape(-1, 4)
        particle_x, particle_y, particle_z = particle_pos[:, 0], particle_pos[:, 1], particle_pos[:, 2]
        size_x, size_y, size_z = np.max(particle_x) - np.min(particle_x), np.max(particle_y) - np.min(particle_y), np.max(particle_z) - np.min(particle_z)
        return size_x, size_y, size_z
            



