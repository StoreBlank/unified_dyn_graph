import os
import time
import numpy as np
import cv2
import pyflex

# robot
from robot import FlexRobotHelper
import pybullet as p
import pybullet_data

pyflex.loadURDF = FlexRobotHelper.loadURDF
pyflex.resetJointState = FlexRobotHelper.resetJointState
pyflex.getRobotShapeStates = FlexRobotHelper.getRobotShapeStates

# utils
from utils import load_yaml
from utils import set_scene, set_table

# load config
config = load_yaml("../config/data_gen/gnn_dyn.yaml")
data_dir = config['dataset']['folder']
os.system("mkdir -p %s" % data_dir)

n_rollout = config['dataset']['n_rollout']
n_timestep = config['dataset']['n_timestep']
obj = config['dataset']['obj']
camera_radius = config['dataset']['camera_radius']
wkspc_w = config['dataset']['wkspc_w']
headless = config['dataset']['headless']   
dt = config['dataset']['dt'] 

def gen_data(info):
    start_time = time.time()

    base_epi = info["base_epi"]
    thread_idx = info["thread_idx"]
    verbose = info["verbose"]

    np.random.seed(round(time.time() * 1000 + thread_idx) % 2 ** 32)

    # set up pybullet
    physicsClient = p.connect(p.DIRECT)
    # physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    planeId = p.loadURDF("plane.urdf")

    # set up robot arm
    # xarm6
    flex_robot_helper = FlexRobotHelper()
    end_idx = 6
    num_dofs = 6

    # step up pyflex
    screenWidth = 720
    screenHeight = 720
    pyflex.set_screenWidth(screenWidth)
    pyflex.set_screenHeight(screenHeight)
    pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
    pyflex.set_light_fov(70.)
    pyflex.init(headless)

    # define action space
    action_dim = 4

    for i in range(n_rollout):

        if i % 1 == 0:
            print("%d / %d" % (i, n_rollout))

        rollout_idx = thread_idx * n_rollout + i
        rollout_dir = os.path.join(data_dir, str(rollout_idx))
        os.system('mkdir -p ' + rollout_dir)

        ### Reset
        # init multi-view cameras 
        camPos_list = []
        camAngle_list = []

        cam_height = np.sqrt(2)/2 * camera_radius #TODO
        cam_dis = wkspc_w // 2 + 0.5
        
        rad_list = np.deg2rad(np.array([0., 90., 180., 270.]) + 45.)
        cam_x_list = np.array([cam_dis, cam_dis, -cam_dis, -cam_dis])
        cam_z_list = np.array([cam_dis, -cam_dis, -cam_dis, cam_dis])

        for i in range(len(rad_list)):
            camPos_list.append(np.array([cam_x_list[i], cam_height, cam_z_list[i]]))
            camAngle_list.append(np.array([rad_list[i], -np.deg2rad(45.), 0.]))
        
        cam_intrinsic_params = np.zeros([len(camPos_list), 4]) # [fx, fy, cx, cy]
        cam_extrinsic_matrix = np.zeros([len(camPos_list), 4, 4]) # [R, t]

        # set pyflex scene
        scene_params = set_scene(obj)

        # set table
        table_size = 4.
        table_height = 0.5
        # table_shape_states = set_table(table_size, table_height)

        # set robot arm
        robot_base_pos = [-3., 0., 0.5]
        robot_base_orn = [0, 0, 0, 1]
        robotId = pyflex.loadURDF(flex_robot_helper, '../assets/xarm/xarm6_with_gripper.urdf', robot_base_pos, robot_base_orn, globalScaling=5) 
        rest_joints = np.zeros(8) 

        num_joints = p.getNumJoints(robotId)
        joints_lower = np.zeros(num_dofs)
        joints_upper = np.zeros(num_dofs)
        dof_idx = 0
        for j in range(num_joints):
            info = p.getJointInfo(robotId, j)
            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                joints_lower[dof_idx] = info[8]
                joints_upper[dof_idx] = info[9]
                dof_idx += 1
        last_ee = None
        # reset_robot()

        pyflex.step()

        for t in range(n_timestep):
            




        



    
    










# run 
info = {
    "base_epi": 0,
    "thread_idx": 0,
    "verbose": True
}
gen_data(info)