from xarm6 import XARM6
from d455 import RS_D455
import cv2
import time
import numpy as np
import pickle


def rpy_to_rotation_matrix(roll, pitch, yaw):
    # Assume the input in in degree
    roll = roll / 180 * np.pi
    pitch = pitch / 180 * np.pi
    yaw = yaw / 180 * np.pi

    # Define the rotation matrices
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )
    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    Rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )

    # Combine the rotations
    R = Rz @ Ry @ Rx

    return R


class RoboCalibrate:
    def __init__(self):
        self.robot = XARM6()
        self.camera = RS_D455(WH=[640, 480], depth_threshold=[0, 2])
        # Initialize the calibration board, this should corresponds to the real board
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.board = cv2.aruco.CharucoBoard(
            (6, 9),
            squareLength=0.03,
            markerLength=0.022,
            dictionary=self.dictionary,
        )
        # Get the camera intrinsic parameters
        self.intrinsic_matrix, self.dist_coef = (
            self.camera.intrinsic_matrix,
            self.camera.dist_coef,
        )
        # Specify the poses used for the hand-eye calirbation
        self.poses = [
            # The first column
            [-275, 612, 300, 175, 2.6, -35],
            [-167, 612, 300, 177.5, 1.3, -62.5],
            [-60, 612, 300, 180, 0, -90],
            [54, 612, 300, 180, 5, -106],
            [148, 612, 300, -179, 3.6, -121.6],
            # The second column
            [280, 370, 250, -175, 0.9, 175.1],
            [110, 370, 250, -176.2, 15, 175.1],
            [-60, 370, 250, 180, 30, -20],
            [-230, 370, 250, 180, 1.9, 0],
            [-400, 370, 250, 180, -26.2, 0],
            # The third column
            [-280, 172.8, 250, -171.1, -4.8, 37.1],
            [-175, 172.8, 250, -167.5, 1.2, 65.4],
            [0, 172.8, 310, -172, 13, 110.9],
            [175.1, 213.1, 281.4, -173.9, 4.3, 154.8],
            [280, 128, 400, -168, 3.8, 155.4],
        ]

        # self.poses = [
        #     [196.2, -1.6, 434, 179.2, 0, 0.3],
        #     [169.7, 119.9, 351, 179.2, 0, -29.8],
        #     [257.1, 176.8, 351, -161.5, -1.4, -47.8],
        #     [430.7, 334.6, 351, 179.5, -0.8, -73.5],
        #     [501.8, 115.4, 429.4, -178.1, 14.3, -72.9],
        #     [622.2, -111.2, 429.4, -176.6, 22.5, -165.5],
        #     [467.1, -392.9, 404.7, 154.8, 26.5, 152.5],
        #     [354.2, -392.9, 404.7, 161, 16.8, 106]
        # ]

    def set_calibration_poses(self, poses):
        self.poses = poses

    def calibrate(self, visualize=True):
        R_gripper2base = []
        t_gripper2base = []
        R_board2cam = []
        t_board2cam = []
        rgbs = []
        depths = []
        for pose in self.poses:
            # Move to the pose and wait for 5s to make it stable
            self.robot.move_to_pose(pose=pose, wait=True, ignore_error=True)
            time.sleep(5)

            # Calculate the markers
            calibration_img, depth_img = self.camera.get_observations(only_raw=True)

            cv2.imwrite('test.jpg', calibration_img)

            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
                image=calibration_img,
                dictionary=self.dictionary,
                parameters=None,
            )

            # Calculate the charuco corners
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=calibration_img,
                board=self.board,
                cameraMatrix=self.intrinsic_matrix,
                distCoeffs=self.dist_coef,
            )

            print("number of corners: ", len(charuco_corners))

            if visualize:
                cv2.aruco.drawDetectedCornersCharuco(
                    image=calibration_img,
                    charucoCorners=charuco_corners,
                    charucoIds=charuco_ids,
                )
                cv2.imshow("cablibration", calibration_img)
                cv2.waitKey(1)

            rvec = None
            tvec = None
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners,
                charuco_ids,
                self.board,
                self.intrinsic_matrix,
                self.dist_coef,
                rvec,
                tvec,
            )
            if not retval:
                raise ValueError("pose estimation failed")
            # Save the transformation of board2cam
            R_board2cam.append(cv2.Rodrigues(rvec)[0])
            t_board2cam.append(tvec[:, 0])
            # Save the transformation of the gripper2base
            current_pose = self.robot.get_current_pose()
            print("Current pose: ", current_pose)
            R_gripper2base.append(
                rpy_to_rotation_matrix(
                    current_pose[3], current_pose[4], current_pose[5]
                )
            )
            t_gripper2base.append(np.array(current_pose[:3]) / 1000)
            # Save the rgb and depth images
            rgbs.append(calibration_img)
            depths.append(depth_img)


        # Do the hand-eye calibration
        R_cam2gripper = None
        t_cam2gripper = None
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base=R_gripper2base,
            t_gripper2base=t_gripper2base,
            R_target2cam=R_board2cam,
            t_target2cam=t_board2cam,
            R_cam2gripper=R_cam2gripper,
            t_cam2gripper=t_cam2gripper,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )

        results = {}
        results["R_cam2gripper"] = R_cam2gripper
        results["t_cam2gripper"] = t_cam2gripper[:, 0]
        results["R_gripper2base"] = R_gripper2base
        results["t_gripper2base"] = t_gripper2base
        results["R_board2cam"] = R_board2cam
        results["t_board2cam"] = t_board2cam
        results["rgbs"] = rgbs
        results["depths"] = depths

        self._save_results(results, "calibrate.pkl")

        print(R_cam2gripper)
        print(t_cam2gripper)

    def _save_results(self, results, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(results, f)

if __name__ == "__main__":
    robo_calirbate = RoboCalibrate()
    robo_calirbate.calibrate()