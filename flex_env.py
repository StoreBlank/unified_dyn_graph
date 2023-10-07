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
        if self.obj == 'shirt':
            path = "cloth3d/Tshirt2.obj"
            retval = load_cloth(path)
            mesh_verts = retval[0]
            mesh_faces = retval[1]
            mesh_stretch_edges, mesh_bend_edges, mesh_shear_edges = retval[2:]

            mesh_verts = mesh_verts * 3.5
            
            cloth_pos = [-1., 1., 0.]
            cloth_size = [20, 20]
            # stiffness = [0.85, 0.90, 0.90] # [stretch, bend, shear]
            stiffness = rand_float(0.4, 1.0)
            stiffness = [stiffness, stiffness, stiffness] # [stretch, bend, shear]
            cloth_mass = 1.0
            particle_r = 0.00625
            render_mode = 2
            flip_mesh = 0
            
            # 0.6, 1.0, 0.6
            dynamicFriction = 0.5
            staticFriction = 1.0
            particleFriction = 0.5
            
            self.scene_params = np.array([
                *cloth_pos,
                *cloth_size,
                *stiffness,
                cloth_mass,
                particle_r,
                render_mode,
                flip_mesh, 
                dynamicFriction, staticFriction, particleFriction])
            
            pyflex.set_scene(
                    29,
                    self.scene_params,
                    mesh_verts.reshape(-1),
                    mesh_stretch_edges.reshape(-1),
                    mesh_bend_edges.reshape(-1),
                    mesh_shear_edges.reshape(-1),
                    mesh_faces.reshape(-1),
                    0)
        
        elif self.obj == '2d_cloth':
            radius = 0.05
            offset_x = -1.
            offset_y = 5.
            offset_z = -1.
            fabric_type = 1 # 0: cloth, 1: shirt, 2: pants

            if fabric_type == 0:
                # parameters of the shape
                dimx = rand_int(25, 35)    # dimx, width
                dimy = rand_int(25, 35)    # dimy, height
                dimz = 0
                # the actuated points
                ctrl_idx = np.array([
                    0, dimx // 2, dimx - 1,
                    dimy // 2 * dimx,
                    dimy // 2 * dimx + dimx - 1,
                    (dimy - 1) * dimx,
                    (dimy - 1) * dimx + dimx // 2,
                    (dimy - 1) * dimx + dimx - 1])

                offset_x = -dimx * radius / 2.
                offset_y = 0.06
                offset_z = -dimy * radius / 2.

            elif fabric_type == 1:
                # parameters of the shape
                dimx = rand_int(16, 25)     # width of the body
                dimy = rand_int(30, 35)     # height of the body
                dimz = 7                    # size of the sleeves
                # the actuated points
                ctrl_idx = np.array([
                    dimx * dimy,
                    dimx * dimy + dimz * (dimz + dimz // 2) + (1 + dimz) * (dimz + 1) // 4,
                    dimx * dimy + (1 + dimz) * dimz // 2 + dimz * (dimz - 1),
                    dimx * dimy + dimz * (dimz + dimz // 2) + (1 + dimz) * (dimz + 1) // 4 + \
                        (1 + dimz) * dimz // 2 + dimz * dimz - 1,
                    dimy // 2 * dimx,
                    dimy // 2 * dimx + dimx - 1,
                    (dimy - 1) * dimx,
                    dimy * dimx - 1])

                offset_x = -(dimx + dimz * 4) * radius / 2.
                offset_y = 0.06
                offset_z = -dimy * radius / 2.

            elif fabric_type == 2:
                # parameters of the shape
                dimx = rand_int(9, 13) * 2 # width of the pants
                dimy = rand_int(6, 11)      # height of the top part
                dimz = rand_int(24, 31)     # height of the leg
                # the actuated points
                ctrl_idx = np.array([
                    0, dimx - 1,
                    (dimy - 1) * dimx,
                    (dimy - 1) * dimx + dimx - 1,
                    dimx * dimy + dimz // 2 * (dimx - 4) // 2,
                    dimx * dimy + (dimz - 1) * (dimx - 4) // 2,
                    dimx * dimy + dimz * (dimx - 4) // 2 + 3 + \
                        dimz // 2 * (dimx - 4) // 2 + (dimx - 4) // 2 - 1,
                    dimx * dimy + dimz * (dimx - 4) // 2 + 3 + \
                        dimz * (dimx - 4) // 2 - 1])

                offset_x = -dimx * radius / 2.
                offset_y = 0.06
                offset_z = -(dimy + dimz) * radius / 2.

            # physical param
            stiffness = rand_float(0.4, 1.0)
            stretchStiffness = stiffness
            bendStiffness = stiffness
            shearStiffness = stiffness

            dynamicFriction = 0.6
            staticFriction = 1.0
            particleFriction = 0.6

            invMass = 1.0

            # other parameters
            windStrength = 0.0
            draw_mesh = 1.

            # set up environment
            self.scene_params = np.array([
                offset_x, offset_y, offset_z,
                fabric_type, dimx, dimy, dimz,
                ctrl_idx[0], ctrl_idx[1], ctrl_idx[2], ctrl_idx[3],
                ctrl_idx[4], ctrl_idx[5], ctrl_idx[6], ctrl_idx[7],
                stretchStiffness, bendStiffness, shearStiffness,
                dynamicFriction, staticFriction, particleFriction,
                invMass, windStrength, draw_mesh])
            zeros = np.array([0])
            pyflex.set_scene(27, self.scene_params, zeros.astype(np.float64), zeros, zeros, zeros, zeros, 0)
        
        elif self.obj == 'rope':
            scale = [30., 30., 30.]       # x, y, z
            trans = [-1.5, 1., 0.]       # x, y, z
            # stiffness = 0.03 + (y - 4) * 0.04
            cluster = [1.3, 0., 0.55]    # spacing, radius, stiffness
            draw_mesh = 1
            scene_params = np.array(scale + trans + cluster + [draw_mesh])
            temp = np.array([0])
            pyflex.set_scene(26, scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0) 
        
        elif self.obj == 'carrots':
            global_scale = 5
            np.random.seed(0)
            # rand_scale = np.random.uniform(0.09, 0.12) * global_scale / 8.0
            rand_scale = 0.08
            max_scale = rand_scale
            min_scale = rand_scale
            # blob_r = np.random.uniform(0.7, 1.0)
            blob_r = 0.7
            x = - blob_r * global_scale / 8.0
            y = 0.5
            z = - blob_r * global_scale / 8.0
            inter_space = 1.5 * max_scale
            num_x = int(abs(x/1.5) / max_scale + 1) * 2
            num_y = 1
            num_z = int(abs(z/1.5) / max_scale + 1) * 2
            x_off = np.random.uniform(-1./24., 1./24.)
            z_off = np.random.uniform(-1./24., 1./24.)
            x += x_off
            z += z_off
            num_carrots = (num_x * num_z - 1) * 3
            add_singular = 0.0
            add_sing_x = -1
            add_sing_y = -1
            add_sing_z = -1
            add_noise = 0.0

            staticFriction = 1.0
            dynamicFriction = 1.0
            draw_skin = 1.0
            min_dist = 5.0
            max_dist = 10.0

            self.scene_params = np.array([max_scale,
                        min_scale,
                        x,
                        y,
                        z,
                        staticFriction,
                        dynamicFriction,
                        draw_skin,
                        num_carrots,
                        min_dist,
                        max_dist,
                        num_x,
                        num_y,
                        num_z,
                        inter_space,
                        add_singular,
                        add_sing_x,
                        add_sing_y,
                        add_sing_z,
                        add_noise,])

            temp = np.array([0])
            pyflex.set_scene(22, self.scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0) 
        
        elif self.obj == 'coffee':
            global_scale = 4
            scale = 0.2 * global_scale / 8.0
            x = -0.9 * global_scale / 8.0
            y = 0.5
            z = -0.9 * global_scale / 8.0
            staticFriction = 0.0
            dynamicFriction = 1.0
            draw_skin = 1.0
            num_coffee = 71 # [200, 1000]
            scene_params = np.array([
                scale, x, y, z, staticFriction, dynamicFriction, draw_skin, num_coffee])

            temp = np.array([0])
            pyflex.set_scene(20, scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0) 

        elif self.obj == 'mustard_bottle':
            x = 0.
            y = 1. 
            z = -0.5
            size = 1.
            obj_type = 6
            self.scene_params = np.array([x, y, z, size, obj_type])
            temp = np.array([0])
            pyflex.set_scene(25, self.scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0) 
        
        elif self.obj == 'power_drill':
            x = 0.
            y = 1. 
            z = -0.5
            size = 1.
            obj_type = 35
            self.scene_params = np.array([x, y, z, size, obj_type])
            temp = np.array([0])
            pyflex.set_scene(25, self.scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0) 
        
        elif self.obj == 'multi_ycb':
            x = 0.
            y = 0.
            z = 0.
            size = 1.
            self.scene_params = np.array([x, y, z, size])
            temp = np.array([0])
            pyflex.set_scene(28, self.scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0) 


        else:
            raise ValueError('obj not defined')
    
    def reset(self):
        self.init_scene()
        self.set_camera()
        
        # add table board
        wall_height = 0.5
        if self.obj == 'multi_ycb':
            halfEdge = np.array([self.wkspc_w, wall_height, self.wkspc_w])
        else:
            halfEdge = np.array([2., wall_height, 2.])
        center = np.array([0.0, 0.0, 0.0])
        quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
        hideShape = 0
        color = np.ones(3) * (160. / 255.)
        self.wall_shape_states = np.zeros((1, 14))
        pyflex.add_box(halfEdge, center, quats, hideShape, color)
        self.wall_shape_states[0] = np.concatenate([center, center, quats, quats])

        # add robot
        robot_base_pos = [-3., 0., 0.5]
        robot_base_orn = [0, 0, 0, 1]
        self.robotId = pyflex.loadURDF(self.flex_robot_helper, 'xarm/xarm6_with_gripper.urdf', robot_base_pos, robot_base_orn, globalScaling=5) 
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
        # h = 0
        h = 0.5 + 0.5
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
        way_points = [s_2d + np.array([0., 0., 0.]), s_2d, e_2d, e_2d + np.array([0., 0., 0.])]

        self.reset_robot(self.rest_joints)
        
        # steps from waypoints
        speed = 1.0/200.
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
    
    def sample_action(self):
        # sample one action within feasible space and with corresponding convex region label
        positions = self.get_positions().reshape(-1, 4)
        num_points = positions.shape[0]
        pickpoint = np.random.randint(0, num_points - 1)
        pickpoint_pos = positions[pickpoint, :2]
        # print('start pos:', pickpoint_pos)

        endpoint_pos = pickpoint_pos + np.random.uniform(-self.wkspc_w // 2, self.wkspc_w // 2, size=(1, 2))
        endpoint_pos = endpoint_pos.reshape(-1)
        # print('end pos:', endpoint_pos)
        
        action = np.concatenate([pickpoint_pos, endpoint_pos], axis=0)
        return action
    
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
            
            



