import os
import numpy as np
import pyflex
import gym
import math

# robot
import pybullet as p
import pybullet_data
from bs4 import BeautifulSoup
from transformations import quaternion_from_matrix, quaternion_matrix

# utils
from utils_env import load_cloth

def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo

def rand_int(lo, hi):
    return np.random.randint(lo, hi)

def quatFromAxisAngle(axis, angle):
    axis /= np.linalg.norm(axis)

    half = angle * 0.5
    w = np.cos(half)

    sin_theta_over_two = np.sin(half)
    axis *= sin_theta_over_two

    quat = np.array([axis[0], axis[1], axis[2], w])

    return quat

# Robot Arm
class FlexRobotHelper:
    def __init__(self):
        self.transform_bullet_to_flex = np.array([
            [1, 0, 0, 0], 
            [0, 0, 1, 0], 
            [0, -1, 0, 0], 
            [0, 0, 0, 1]])
        self.robotId = None

    def loadURDF(self, fileName, basePosition, baseOrientation, useFixedBase = True, globalScaling = 1.0):
        if self.robotId is None:
            self.robotId = p.loadURDF(fileName, basePosition, baseOrientation, useFixedBase = useFixedBase, globalScaling = globalScaling)
        p.resetBasePositionAndOrientation(self.robotId, basePosition, baseOrientation)
        
        robot_path = fileName # changed the urdf file
        robot_path_par = os.path.abspath(os.path.join(robot_path, os.pardir))
        with open(robot_path, 'r') as f:
            robot = f.read()
        robot_data = BeautifulSoup(robot, 'xml')
        links = robot_data.find_all('link')
        
        # add the mesh to pyflex
        self.num_meshes = 0
        self.has_mesh = np.ones(len(links), dtype=bool)
        
        for i in range(len(links)):
            link = links[i]
            if link.find_all('geometry'):
                mesh_name = link.find_all('geometry')[0].find_all('mesh')[0].get('filename')
                pyflex.add_mesh(os.path.join(robot_path_par, mesh_name), globalScaling, 0, np.ones(3), False)
                self.num_meshes += 1
            else:
                self.has_mesh[i] = False
        
        self.num_link = len(links)
        self.state_pre = None

        return self.robotId

    def resetJointState(self, i, pose):
        p.resetJointState(self.robotId, i, pose)
        return self.getRobotShapeStates()
    
    def getRobotShapeStates(self):
        # convert pybullet link state to pyflex link state
        state_cur = []
        base_com_pos, base_com_orn = p.getBasePositionAndOrientation(self.robotId)
        di = p.getDynamicsInfo(self.robotId, -1)
        local_inertial_pos, local_inertial_orn = di[3], di[4]
        
        pos_inv, orn_inv = p.invertTransform(local_inertial_pos, local_inertial_orn)
        pos, orn = p.multiplyTransforms(base_com_pos, base_com_orn, pos_inv, orn_inv)
    
        state_cur.append(list(pos) + [1] + list(orn))

        for l in range(self.num_link-1):
            ls = p.getLinkState(self.robotId, l)
            pos = ls[4]
            orn = ls[5]
            state_cur.append(list(pos) + [1] + list(orn))
        
        state_cur = np.array(state_cur)
        
        shape_states = np.zeros((self.num_meshes, 14))
        if self.state_pre is None:
            self.state_pre = state_cur.copy()

        mesh_idx = 0
        for i in range(self.num_link):
            if self.has_mesh[i]:
                # pos + [1]
                shape_states[mesh_idx, 0:3] = np.matmul(
                    self.transform_bullet_to_flex, state_cur[i, :4])[:3]
                shape_states[mesh_idx, 3:6] = np.matmul(
                    self.transform_bullet_to_flex, self.state_pre[i, :4])[:3]
                # orientation
                shape_states[mesh_idx, 6:10] = quaternion_from_matrix(
                    np.matmul(self.transform_bullet_to_flex,
                            quaternion_matrix(state_cur[i, 4:])))
                shape_states[mesh_idx, 10:14] = quaternion_from_matrix(
                    np.matmul(self.transform_bullet_to_flex,
                            quaternion_matrix(self.state_pre[i, 4:])))
                mesh_idx += 1
        
        self.state_pre = state_cur
        return shape_states

pyflex.loadURDF = FlexRobotHelper.loadURDF
pyflex.resetJointState = FlexRobotHelper.resetJointState
pyflex.getRobotShapeStates = FlexRobotHelper.getRobotShapeStates

# PyFlex Environment
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
        self.end_idx = 6
        self.num_dofs = 6

        # set up pyflex
        self.screenWidth = 720
        self.screenHeight = 720

        self.wkspc_w = config['dataset']['wkspc_w']
        self.headless = config['dataset']['headless']
        self.obj = config['dataset']['obj']

        self.global_scale = config['dataset']['global_scale']

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

    def robot_to_shape_states(self, robot_states):
        return np.concatenate([self.wall_shape_states, robot_states], axis=0)

    def reset_robot(self, jointPositions = np.zeros(6).tolist()):
        index = 0
        for j in range(self.num_joints):
            p.changeDynamics(self.robotId, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.robotId, j)

            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                pyflex.resetJointState(self.flex_robot_helper, j, jointPositions[index])
                index = index + 1
            
        pyflex.set_shape_states(self.robot_to_shape_states(pyflex.getRobotShapeStates(self.flex_robot_helper)))
    
    def set_camera(self):
        self.move_x = 0.
        self.move_z = 0.
        cam_height = np.sqrt(2)/2 * self.camera_radius
        cam_dis = np.sqrt(2)/2 * self.camera_radius
        cam_dis = 2.5
        if self.camera_view == 0:
            self.camPos = np.array([self.move_x, cam_height+5., self.move_z])
            self.camAngle = np.array([0., -np.deg2rad(90.), 0.])
        elif self.camera_view == 1:
            # self.camPos = np.array([2. + self.move_x, cam_height, 0. + cam_dis + self.move_z])
            self.camPos = np.array([cam_dis + self.move_x, cam_height, cam_dis + self.move_z])
            self.camAngle = np.array([np.deg2rad(45.), -np.deg2rad(45.), 0.])
        elif self.camera_view == 2:
            self.camPos = np.array([cam_dis + self.move_x, cam_height, -cam_dis + self.move_z])
            self.camAngle = np.array([np.deg2rad(45.+90.), -np.deg2rad(45.), 0.])
        elif self.camera_view == 3:
            self.camPos = np.array([-cam_dis + self.move_x, cam_height, -cam_dis + self.move_z])
            self.camAngle = np.array([np.deg2rad(45.+180.), -np.deg2rad(45.), 0.])
        elif self.camera_view == 4:
            self.camPos = np.array([-cam_dis + self.move_x, cam_height, cam_dis + self.move_z])
            self.camAngle = np.array([np.deg2rad(45.+270.), -np.deg2rad(45.), 0.])
        else:
            raise ValueError('camera_view not defined')
        
        pyflex.set_camPos(self.camPos)
        pyflex.set_camAngle(self.camAngle)
    
    def init_scene(self):
        if self.obj == 'cloth':
            cloth_pos = [-0.6, 0, -0.6]
            cloth_size = [100, 100]
            # [0.85, 0.90, 0.90]
            stiffness = [1.5, 1.5, 1.5] # [stretch, bend, shear]
            cloth_mass = 1.5
            particle_r = 0.01
            render_mode = 1
            flip_mesh = 0
            self.scene_params = np.array([
                *cloth_pos,
                *cloth_size,
                *stiffness,
                cloth_mass,
                particle_r,
                render_mode,
                flip_mesh])
            zeros = np.array([0])
            pyflex.set_scene(29, self.scene_params, zeros.astype(np.float64), zeros, zeros, zeros, zeros, 0)
        elif self.obj == 'shirt':
            path = "cloth3d/Tshirt2.obj"
            retval = load_cloth(path)
            mesh_verts = retval[0]
            mesh_faces = retval[1]
            mesh_stretch_edges, mesh_bend_edges, mesh_shear_edges = retval[2:]
            num_particle = mesh_verts.shape[0]//3
            
            cloth_pos = [0, 0, 0]
            cloth_size = [100, 100]
            stiffness = [0.85, 0.90, 0.90] # [stretch, bend, shear]
            cloth_mass = 1.5
            particle_r = 0.01
            render_mode = 1
            flip_mesh = 0
            self.scene_params = np.array([
                *cloth_pos,
                *cloth_size,
                *stiffness,
                cloth_mass,
                particle_r,
                render_mode,
                flip_mesh])
            
            pyflex.set_scene(
                    29,
                    self.scene_params,
                    mesh_verts.reshape(-1),
                    mesh_stretch_edges.reshape(-1),
                    mesh_bend_edges.reshape(-1),
                    mesh_shear_edges.reshape(-1),
                    mesh_faces.reshape(-1),
                    0)
        elif self.obj == 'mustard_bottle':
            x = 0.
            y = 1. 
            z = -0.5
            size = 1.
            obj_type = 6
            self.scene_params = np.array([x, y, z, size, obj_type])
            temp = np.array([0])
            pyflex.set_scene(25, self.scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0) 


        else:
            raise ValueError('obj not defined')
    
    def reset(self):
        self.init_scene()
        self.set_camera()
        
        # add "table"
        wall_height = 0.5
        halfEdge = np.array([2., wall_height, 2.])
        # center = np.array([self.global_scale/2.0-3., 0.0, 0.0])
        center = np.array([0.0, 0.0, 0.0])
        quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
        hideShape = 0
        color = np.ones(3) * (160. / 255.)
        self.wall_shape_states = np.zeros((1, 14))
        pyflex.add_box(halfEdge, center, quats, hideShape, color)
        self.wall_shape_states[0] = np.concatenate([center, center, quats, quats])

        # add robot
        # robot_base_pos = [-6.0 * self.global_scale / 8.0, 0., 0.]
        robot_base_pos = [-3.5, 0., 0.]
        robot_base_orn = [0, 0, 0, 1]
        self.robotId = pyflex.loadURDF(self.flex_robot_helper, 'xarm/xarm6_with_gripper.urdf', robot_base_pos, robot_base_orn, globalScaling=self.global_scale) 
        self.rest_joints = np.zeros(8) 

        pyflex.set_shape_states(self.robot_to_shape_states(pyflex.getRobotShapeStates(self.flex_robot_helper)))
        
        for i in range(300):
            pyflex.step()
        
        # update robot actions
        for idx, joint in enumerate(self.rest_joints):
            pyflex.set_shape_states(self.robot_to_shape_states(pyflex.resetJointState(self.flex_robot_helper, idx, joint)))
        
        self.num_joints = p.getNumJoints(self.robotId)
        self.joints_lower = np.zeros(self.num_dofs)
        self.joints_upper = np.zeros(self.num_dofs)
        dof_idx = 0
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robotId, i)
            # print(f"Joint {i}:", info)
            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                self.joints_lower[dof_idx] = info[8]
                self.joints_upper[dof_idx] = info[9]
                dof_idx += 1
        self.last_ee = None
        self.reset_robot()
    
    def step(self, action):
        h = 0.5 + self.global_scale / 8.0 #TODO
        # h = 0
        s_2d = np.concatenate([action[:2], [h]])
        e_2d = np.concatenate([action[2:], [h]])
        # print('start action:', s_2d)
        # print('end action:', e_2d)

        # pusher angle depending on x-axis
        if (s_2d - e_2d)[0] == 0:
            pusher_angle = np.pi/2
        else:
            pusher_angle = np.arctan((s_2d - e_2d)[1] / (s_2d - e_2d)[0])
        
        # robot orientation
        orn = np.array([0.0, np.pi, pusher_angle + np.pi/2])

        # create way points
        # way_points = [s_2d + np.array([0., 0., self.global_scale / 24.0]), s_2d, e_2d, e_2d + np.array([0., 0., self.global_scale / 24.0])]
        way_points = [s_2d + np.array([0., 0., 1.]), s_2d, e_2d, e_2d + np.array([0., 0., 1.])]
        self.reset_robot(self.rest_joints)
        
        # steps from waypoints
        speed = 1.0/50.
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
                self.reset_robot()

                if math.isnan(self.get_positions().reshape(-1, 4)[:, 0].max()):
                    print('simulator exploded when action is', action)
                    return None
        
            self.last_ee = end_effector_pos.copy()
        
        self.reset_robot()
        for i in range(200):
            pyflex.step()
        
        obs = self.render()
        return obs
    
    def render(self, no_return=False):
        pyflex.step()
        if no_return:
            return
        else:
            return pyflex.render(render_depth=True).reshape(self.screenHeight, self.screenWidth, 5)
    
    def close(self):
        pyflex.clean()
    
    def sample_action(self, n):
        # sample one action within feasible space and with corresponding convex region label
        action = -self.wkspc_w + 2 * self.wkspc_w * np.random.rand(n, 1, 4)
        return action
    
    def get_positions(self):
        return pyflex.get_positions()

            
            



