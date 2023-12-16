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
        self.obj_shape_states = np.zeros((1, 14))
        
    
    
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
    
    
    ### shape state
    def robot_to_shape_states(self, robot_states):
        n_robot_links = robot_states.shape[0]
        n_table = 1
        
        if (self.obj_shape_states == 0).all():
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
    
    def add_table(self, table_side = 4., table_height = 0.5):
        halfEdge = np.array([table_side, table_height, table_side])
        center = np.array([0.0, 0.0, 0.0])
        quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
        hideShape = 0
        color = np.ones(3) * (160. / 255.)
        pyflex.add_box(halfEdge, center, quats, hideShape, color)
        self.table_shape_states = np.concatenate([center, center, quats, quats])
        return table_height
    
    def add_cable_holder(self, num_holder = 2, box_side = 0.05):
        particle_pos = self.get_positions().reshape(-1, 4)
        particle_pos_x, particle_pos_y, particle_pos_z = particle_pos[:, 0], particle_pos[:, 1], particle_pos[:, 2]
        particle_pos_z_min, particle_pos_z_max = np.min(particle_pos_z), np.max(particle_pos_z)
        
        self.obj_shape_states = np.zeros((num_holder+1, 14))
        
        # fixed rope box
        halfEdge = np.ones(3) * 0.1
        center = np.array([np.median(particle_pos_x), self.table_height+0.1, np.max(particle_pos_z)])
        quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
        hideShape = 0
        color = np.ones(3) * (160. / 255.)
        pyflex.add_box(halfEdge, center, quats, hideShape, color)
        self.obj_shape_states[0] = np.concatenate([center, center, quats, quats])
        
        # set holder
        self.holder_center_pos = np.zeros((num_holder, 3))
        halfEdge = np.ones(3) * box_side
        quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
        hideShape = 0
        color = np.array([1., 0., 0.])
        # for i in range(1, num_holder+1):
        #     halfEdge = np.ones(3) * box_side
        
        # box 1
        box_x_1, box_z_1 = -0.3, -0.5
        box_1_center = np.array([box_x_1, self.table_height+box_side, box_z_1])
        self.holder_center_pos[0] = box_1_center
        pyflex.add_box(halfEdge, box_1_center, quats, hideShape, color)
        self.obj_shape_states[1] = np.concatenate([box_1_center, box_1_center, quats, quats])
        # box 2
        box_x_2, box_z_2 = 0.3, 0.5
        box_2_center = np.array([box_x_2, self.table_height+box_side, box_z_2])
        self.holder_center_pos[1] = box_2_center
        pyflex.add_box(halfEdge, box_2_center, quats, hideShape, color)
        self.obj_shape_states[2] = np.concatenate([box_2_center, box_2_center, quats, quats])
            

    
    ### cameras 
    def set_camera(self):
        cam_dis = 3.
        cam_height = 4.5
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

        cam_dis = 3.
        cam_height = 4.5

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
        if obj == 'rope':
            
            self.particle_num_threshold = 500 # for fps
            
            radius = 0.03
            
            if self.physics == "random":
                length = 2.0 #rand_float(0.5, 2.5)
                thickness = 2.0 #rand_float(1., 2.5)
                scale = np.array([length, thickness, thickness]) * 50 # length, extension, thickness
                cluster_spacing = 8. #rand_float(2, 8) # change the stiffness of the rope
                dynamicFriction = 0.3 #rand_float(0.1, 0.7)
            elif self.physics == "grid":
                length = property_params['length']
                thickness = property_params['thickness']
                scale = np.array([length, thickness, thickness]) * 50 # length, extension, thickness
                cluster_spacing = property_params['cluster_spacing']
                dynamicFriction = property_params['dynamic_friction']
            
            trans = [0., 0.5, 1.5]
            
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
            
            draw_mesh = 0

            relaxtion_factor = 1.
            collisionDistance = 0.05 #radius * 0.5
            
            self.scene_params = np.array([*scale, *trans, radius, 
                                            cluster_spacing, cluster_radius, cluster_stiffness,
                                            link_radius, link_stiffness, global_stiffness,
                                            surface_sampling, volume_sampling, skinning_falloff, skinning_max_dist,
                                            cluster_plastic_threshold, cluster_plastic_creep,
                                            dynamicFriction, particleFriction, draw_mesh, relaxtion_factor, 
                                            *rotate, collisionDistance])
            
            temp = np.array([0])
            pyflex.set_scene(26, self.scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0) 

            self.property = {'particle_radius': radius,
                             'num_particles': self.get_num_particles(),
                             'length': scale[0],
                             'thickness': scale[2],
                             'dynamic_friction': dynamicFriction,
                             'cluster_spacing': cluster_spacing,
                             'global_stiffness': global_stiffness,}
            
        else:
            raise ValueError('obj not defined')
    
    def reset(self, count=0, dir=None, property_params=None):
        obj = self.obj
        self.init_scene(obj, property_params)
        
        ## camera setting
        self.set_camera()
        self.init_multiview_camera()
        
        ## add table board and holders
        self.table_height = self.add_table()
        self.add_cable_holder()
        
        ## add robot
        robot_base_pos = [-3., 0., 1.]
        robot_base_orn = [0, 0, 0, 1]
        self.robotId = pyflex.loadURDF(self.flex_robot_helper, 'assets/xarm/xarm6_with_gripper_2.urdf', robot_base_pos, robot_base_orn, globalScaling=5) 
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
        
        for _ in range(100):
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
        if self.gripper:
            h = 1.35
        elif self.obj == 'bowl_granular':
            h = 1.4
        else:
            # h = 0.5 + 0.5 # table + pusher
            h = 1.2
        s_2d = np.concatenate([action[:2], [h]])
        e_2d = np.concatenate([action[2:], [h]])

        # pusher angle depending on x-axis
        if (s_2d - e_2d)[0] == 0:
            pusher_angle = np.pi
        else:
            # pusher_angle = np.arctan((s_2d - e_2d)[1] / (s_2d - e_2d)[0])
            # pusher_angle = -np.arctan((s_2d - e_2d)[1] / (s_2d - e_2d)[0])
            pusher_angle = -np.pi/2
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
                # way_points = [s_2d + [0., 0., 0.5], s_2d, s_2d, s_2d + [0., 0., 0.7], e_2d + [0., 0., 0.7], e_2d + [0., 0., 0.2]]
                way_points = [s_2d + [0., 0., 0.5], s_2d, s_2d, s_2d + [0., 0., 1.]]
                # self.holder_center_pos
                way_points.append(np.array([2., s_2d[1], h]))
                way_points.append(np.array([-1., 0., h]))
                
                
       
                
            else:
                way_points = [s_2d + [0., 0., 0.2], s_2d, e_2d, e_2d + [0., 0., 0.2]]
            self.reset_robot(self.rest_joints)

        # set robot speed
        if self.obj in ["Tshirt"]:
            speed = 1.0/300.
        else:
            speed = 1.0/100.
        
        self.count = prev_counts
        self.contact = []
        
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
                
                ## gripper control
                if self.gripper and self.grasp and i_p >= 1:
                    grasp_thresd = 0.1 #0.1
                    obj_pos = self.get_positions().reshape(-1, 4)[:, :3]
                    new_particle_pos = self.get_positions().reshape(-1, 4).copy()
                    
                    ### grasping 
                    if i_p == 1:
                        close = 0
                        start = 0
                        end = 0.7 #wood:0.35 #0.7
                        close_steps = 1 #500
                        for j in range(close_steps):
                            robot_shape_states = pyflex.getRobotShapeStates(self.flex_robot_helper) # 9: left finger; 12: right finger
                            left_finger_pos, right_finger_pos = robot_shape_states[9][:3], robot_shape_states[12][:3]
                            #print(left_finger_pos, right_finger_pos)
                            left_finger_pos[1], right_finger_pos[1] = left_finger_pos[1] - 0.2, right_finger_pos[1] - 0.2 #0.2
                            new_finger_pos = (left_finger_pos + right_finger_pos) / 2
                            
                            if j == 0:
                                # fine the k pick point
                                pick_k = 10 #wood:100 #rope:5 #cloth:80
                                left_min_dist, left_pick_index = find_min_distance(left_finger_pos, obj_pos, pick_k)
                                right_min_dist, right_pick_index = find_min_distance(right_finger_pos, obj_pos, pick_k)
                                min_dist, pick_index = find_min_distance(new_finger_pos, obj_pos, pick_k)
                                # save the original setting for restoring
                                pick_origin = new_particle_pos[pick_index]
                            
                            # connect pick pick point to the finger
                            if min_dist <= grasp_thresd:
                                new_particle_pos[pick_index, :3] = new_finger_pos
                                new_particle_pos[pick_index, 3] = 0
                            self._set_pos(new_finger_pos, new_particle_pos)
                            
                            # close the gripper slowly 
                            close += (end - start) / close_steps
                            self.robot_close_gripper(close)
                            pyflex.step()
                    
                    # find finger positions
                    robot_shape_states = pyflex.getRobotShapeStates(self.flex_robot_helper) # 9: left finger; 12: right finger
                    left_finger_pos, right_finger_pos = robot_shape_states[9][:3], robot_shape_states[12][:3]
                    left_finger_pos[1], right_finger_pos[1] = left_finger_pos[1] - 0.2, right_finger_pos[1] - 0.2
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
                    if robot_obj_dist < 0.2 and i % 3 == 0: #contact
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
                    elif i % 10 == 0:
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
        
        # set up gripper
        if self.gripper:
            if self.grasp:
                self.robot_open_gripper()
            else:
                self.robot_close_gripper(0.7)
        
        # reset the mass for the pick points
        if self.gripper and self.grasp:
            new_particle_pos[pick_index, 3] = pick_origin[:, 3]
            self._reset_pos(new_particle_pos)
        
        
        self.reset_robot()
        for i in range(400):
            pyflex.step()
        
        ### save final rendering
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
        if self.obj in ['mustard_bottle', 'power_drill', 'rigid_objects']:
            action = self.sample_rigid_actions()
        elif self.obj in ['rope']:
            action = self.sample_rope_actions()
        elif self.obj in ['Tshirt', 'carrots', 'coffee']:
            action = self.sample_cloth_actions()
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
    
    def sample_rope_actions(self):
        positions = self.get_positions().reshape(-1, 4)
        positions[:, 2] *= -1 # align with the coordinates
        num_points = positions.shape[0]
        pos_xz = positions[:, [0, 2]]

        # random choose a start point which can not be overlapped with the object
        while True:
            startpoint_pos = np.random.uniform(-self.wkspc_w // 2 - 1, self.wkspc_w // 2 + 1., size=(1, 2))
            if np.min(cdist(startpoint_pos, pos_xz)) > 0.2 and np.max(cdist(startpoint_pos, pos_xz)) < 1.5:
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
    
    def sample_cloth_actions(self):
        positions = self.get_positions().reshape(-1, 4)
        positions[:, 2] *= -1 # align with the coordinates
        num_points = positions.shape[0]
        pos_xz = positions[:, [0, 2]]

        # random choose a start point which can not be overlapped with the object
        while True:
            startpoint_pos = np.random.uniform(-self.wkspc_w // 2 - 1, self.wkspc_w // 2 + 1., size=(1, 2))
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
        if (pos[:, 1] < 0.4).any():
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
    
    def get_obj_idx_range(self):
        particle_pos = self.get_positions().reshape(-1, 4)
        particle_pos_x, particle_pos_z = particle_pos[:, 0], particle_pos[:, 2]
        x_min_idx, x_max_idx = np.argmin(particle_pos_x), np.argmax(particle_pos_x)
        z_min_idx, z_max_idx = np.argmin(particle_pos_z), np.argmax(particle_pos_z)
        
        x_min, x_max = particle_pos_x[x_min_idx], particle_pos_x[x_max_idx]
        z_min, z_max = particle_pos_z[z_min_idx], particle_pos_z[z_max_idx]
        return x_min, x_max, z_min, z_max
    
    
            



