import os
import numpy as np
import pyflex
import gym

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

        self.global_scale = 1.0 #TODO

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
        # ? self.wall_shape_states
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
        self.move_x = 0
        self.move_z = 0
        cam_height = np.sqrt(2)/2 * self.camera_radius
        cam_dis = np.sqrt(2)/2 * self.camera_radius
        if self.camera_view == 0:
            self.camPos = np.array([0.+self.move_x, cam_height, 0.+self.move_z])
            self.camAngle = np.array([0., -np.deg2rad(90.), 0.])
        elif self.camera_view == 1:
            self.camPos = np.array([0.+self.move_x, cam_height, cam_dis+self.move_z])
            self.camAngle = np.array([0., -np.deg2rad(45.), 0.])
        elif self.camera_view == 2:
            self.camPos = np.array([cam_dis+self.move_x, cam_height, 0.+self.move_z])
            self.camAngle = np.array([np.deg2rad(90.), -np.deg2rad(45.), 0.])
        elif self.camera_view == 3:
            self.camPos = np.array([0.+self.move_x, cam_height, -cam_dis+self.move_z])
            self.camAngle = np.array([np.deg2rad(180.), -np.deg2rad(45.), 0.])
        elif self.camera_view == 4:
            self.camPos = np.array([-cam_dis+self.move_x, cam_height, 0.+self.move_z])
            self.camAngle = np.array([np.deg2rad(270.), -np.deg2rad(45.), 0.])
        else:
            raise ValueError('camera_view not defined')
        
        pyflex.set_camPos(self.camPos)
        pyflex.set_camAngle(self.camAngle)
    
    def init_scene(self):
        if self.obj == 'cloth':
            cloth_pos = [-0.6, 0, -0.6]
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
            zeros = np.array([0])
            pyflex.set_scene(29, self.scene_params, zeros.astype(np.float64), zeros, zeros, zeros, zeros, 0)
        else:
            raise ValueError('obj not defined')
    
    def reset(self):
        self.init_scene()
        self.set_camera()

        for i in range(100):
            pyflex.step()
        
        # add wall
        halfEdge = np.array([0.05, 1.0, self.global_scale/2.0])
        centers = [np.array([self.global_scale/2.0, 1.0, 0.0]),
                   np.array([0.0, 1.0, -self.global_scale/2.0]),
                   np.array([-self.global_scale/2.0, 1.0, 0.0]),
                   np.array([0.0, 1.0, self.global_scale/2.0])]
        quats = [quatFromAxisAngle(axis=np.array([0., 1., 0.]),
                                 angle=0.),
                 quatFromAxisAngle(axis=np.array([0., 1., 0.]),
                                 angle=np.pi/2.),
                 quatFromAxisAngle(axis=np.array([0., 1., 0.]),
                                 angle=0.),
                 quatFromAxisAngle(axis=np.array([0., 1., 0.]),
                                 angle=np.pi/2.)]
        hideShape = 0
        color = np.ones(3) * 0.9
        self.wall_shape_states = np.zeros((4, 14))
        for i, center in enumerate(centers):
            pyflex.add_box(halfEdge, center, quats[i], hideShape, color)
            self.wall_shape_states[i] = np.concatenate([center, center, quats[i], quats[i]])
        
        # add robot
        self.robotId = pyflex.loadURDF(self.flex_robot_helper, 'xarm/xarm6_with_gripper.urdf', [-5.0 * self.global_scale / 8.0, 0, 0], [0, 0, 0, 1], globalScaling=self.global_scale) 
        self.rest_joints = np.zeros(8)

        pyflex.set_shape_states(self.robot_to_shape_states(pyflex.getRobotShapeStates(self.flex_robot_helper)))
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
        pass
    
    def render(self, no_return=False):
        pyflex.step()
        if no_return:
            return
        else:
            return pyflex.render(render_depth=True).reshape(self.screenHeight, self.screenWidth, 5)
    
    def sample_action(self, n):
        # sample one action within feasible space and with corresponding convex region label
        action = -self.wkspc_w + 2 * self.wkspc_w * np.random.rand(n, 1, 4)
        return action
    
    def get_positions(self):
        return pyflex.get_positions()

            
            



