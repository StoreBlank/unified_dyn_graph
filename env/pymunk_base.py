import sys, os
import pymunk
from pymunk import Vec2d
import math
import numpy as np
import random
import time

import pymunk
from pymunk import Vec2d
import math
import numpy as np
import random
import cv2
import PIL
from PIL import Image, ImageSequence


class Base_Sim(object):
    def __init__(self, param_dict):
        self.SAVE_IMG, self.ENABLE_VIS = (
            param_dict["save_img"],
            param_dict["enable_vis"],
        )
        self.task_name = param_dict["task_name"]
        self.dir = param_dict["dir"]
        if self.dir is not None and not os.path.exists(self.dir):
            os.makedirs(self.dir)
        # if self.IMG_STATE:
        #     img_size = param_dict["img_size"]
        #     self.img_size = img_size
        # self.include_com = param_dict["include_com"]
        # Sim window parameters. These also define the resolution of the image
        self.width = self.height = param_dict["window_size"]
        self.elasticity = 0.1
        self.friction = 0.1
        self.obj_mass = 0.5
        self.velocity = np.array([0, 0])
        self.target_positions = None
        self.target_angles = None
        self.pusher_body = None
        self.pusher_shape = None
        self.pusher_size = param_dict["pusher_size"]
        self.global_time = 0.0
        self.obj_num = 0
        self.obj_list = []
        # self.image_list = []
        self.current_image = None
        self.object_colors = [(0, 0, 255, 255), (255, 255, 0, 255)]
        self.pusher_color = (255, 0, 0, 255)
        self.background_color = (0, 0, 0, 255)

        # others
        self.count = 0
        self.particle_pos_list = []
        self.eef_states_list = []
        self.step_list = []
        self.contact_list = []

        self.space = pymunk.Space()

    def create_world(self, init_poses, pusher_pos):
        self.space.gravity = Vec2d(0, 0)  # planar setting
        self.space.damping = 0.0001  # quasi-static. low value is higher damping.
        self.space.iterations = 5  # TODO(terry-suh): re-check. what does this do?
        self.add_objects(self.obj_num, init_poses)
        if pusher_pos is not None:
            self.add_pusher(pusher_pos)
        self.wait(1.0)
        # self.render()
        # self.image_list = []
        # self.current_image = None

    def add_objects(self, obj_num, poses=None):
        """
        Create and add multiple object to sim.
        """
        if poses is None:
            for i in range(obj_num):
                self.add_object(i)
        else:
            for i in range(obj_num):
                self.add_object(i, poses[i])

    def add_object(self, id, pose=None):
        """
        Create and add a single object to sim.
        """
        body, shape_components = self.create_object(id, pose)
        self.space.add(body, *shape_components)
        self.obj_list.append([body, shape_components])  # Adjust storage to handle multiple shapes

    def create_object(self, id, poses=None):
        """
        Create a single object by defining its shape, mass, etc.
        """
        raise NotImplementedError

    def remove_all_objects(self):
        """
        Remove all objects from sim.
        """
        for i in range(len(self.obj_list)):
            body = self.obj_list[i][0]
            shapes = self.obj_list[i][1]
            self.space.remove(body, *shapes)
        self.obj_list = []

    def get_object_pose(self, index, target=False):
        """
        Return the pose of an object in sim.
        """
        if target:
            pos = self.target_positions[index]
            angle = self.target_angles[index]
            pose = [pos[0], pos[1], angle]
        else:
            body: pymunk.Body = self.obj_list[index][0]
            pos = body.position
            angle = body.angle
            pose = [pos.x, pos.y, angle]
        return pose

    def get_all_object_poses(self, target=False):
        """
        Return the poses of all objects in sim.
        """
        if target and self.target_positions is None:
            return None
        all_poses = []
        for i in range(len(self.obj_list)):
            all_poses.append(self.get_object_pose(i, target))
        return all_poses

    def update_object_pose(self, index, new_pose):
        """
        Update the pose of an object in sim.
        """
        body = self.obj_list[index][0]
        body.angle = new_pose[2]
        body.position = pymunk.Vec2d(new_pose[0], new_pose[1])
        self.wait(1.0)  # Give some time for collision pieces to stabilize.
        return

    def get_all_object_positions(self):
        """
        Return the positions of all objects in sim.
        """
        return [body.position for body, _ in self.obj_list]

    def get_all_object_angles(self):
        """
        Return the angles of all objects in sim.
        """
        return [body.angle for body, _ in self.obj_list]

    def get_object_keypoints(self, index):
        """
        Return the keypoints of an object in sim.
        """
        raise NotImplementedError

    def get_all_object_keypoints(self):
        """
        Return the keypoints of all objects in sim.
        """
        all_keypoints = []
        for i in range(len(self.obj_list)):
            all_keypoints.append(self.get_object_keypoints(i))

        return all_keypoints

    def get_object_particles(self, index):
        """
        Return the particles of an object in sim.
        """
        raise NotImplementedError

    def get_all_particles(self):
        """
        Return the particles of all objects in sim.
        """
        all_particles = []
        for i in range(len(self.obj_list)):
            all_particles.append(self.get_object_particles(i))
        # all_particles = np.concatenate(all_particles, axis=0)

        return all_particles

    def get_object_vertices(self, index, target=False, **kwargs):
        """
        Return the vertices of an object in sim.
        """
        raise NotImplementedError

    def get_all_object_vertices(self, target=False, **kwargs):
        """
        Return the vertices of all objects in sim.
        """
        if target and self.target_positions is None:
            return None
        all_vertices = []
        for i in range(len(self.obj_list)):
            all_vertices.append(self.get_object_vertices(i, target, **kwargs))

        return all_vertices

    def gen_vertices_from_pose(self, pose, **kwargs):
        """
        Generate vertices from a pose.
        """
        raise NotImplementedError

    def get_particle_state(self):
        """
        Return the particle state of all objects in sim.
        """
        return np.array(self.get_all_particles()).flatten()

    # def get_current_state(self):
    #     raise NotImplementedError

    def create_pusher(self, position):
        """
        Create a single pusher by defining its shape, mass, etc.
        """
        body = pymunk.Body(1e7, float("inf"))
        if position is None:
            body.position = Vec2d(
                random.randint(int(self.width * 0.25), int(self.width * 0.75)),
                random.randint(int(self.height * 0.25), int(self.height * 0.75)),
            )
        else:
            body.position = Vec2d(position[0], position[1])
        shape = pymunk.Circle(body, radius=self.pusher_size)
        shape.elasticity = 0.1
        shape.friction = 0.6
        shape.color = self.pusher_color
        return body, shape

    def add_pusher(self, position):
        """
        Create and add a single pusher to the sim.
        """
        self.pusher_body, self.pusher_shape = self.create_pusher(position)
        self.space.add(self.pusher_body, self.pusher_shape)

    def remove_pusher(self):
        """
        Remove pusher from simulation.
        """
        self.space.remove(self.pusher_body, self.pusher_shape)

    def get_pusher_position(self):
        """
        Return the position of the pusher.
        """
        if self.pusher_body is None:
            return None
        return self.pusher_body.position

    def get_eef_states(self):
        raise NotImplementedError

    def detect_collision(self):
        if self.pusher_shape is None:
            return False
        for obj in self.obj_list:
            shape_components = obj[1]
            for shape in shape_components:
                if self.pusher_shape.shapes_collide(shape).points != []:
                    return True
        return False

    def step(self, action, particle_pos_list = None, eef_states_list = None, step_list = None, contact_list = None):
        """
        Once given a control action, run the simulation forward and return.
        action: x_start, y_start, x_end, y_end.
        return keypoints instead of all particles
        """
        if self.dir != None:
            self.particle_pos_list = particle_pos_list
            self.eef_states_list = eef_states_list
            self.step_list = step_list
            self.contact_list = contact_list
            self.count = self.step_list[-1]

        # Parse into integer coordinates
        uxi, uyi, uxf, uyf = [int(x) for x in action]
        theta = np.arctan2(uyf - uyi, uxf - uxi)
        length = np.linalg.norm(np.array([uxf - uxi, uyf - uyi]), ord=2)

        # add the pusher if not added
        if self.pusher_body is None:
            self.add_pusher((uxi, uyi))
            self.pusher_body.angle = theta - np.pi / 2
            # self.render()
        else:
            self.pusher_body.position = Vec2d(uxi, uyi)
            self.pusher_body.angle = theta - np.pi / 2
            self.pusher_body.velocity = Vec2d(0, 0)

        self.velocity = np.array([np.cos(theta), np.sin(theta)]) * 10 # 1cm per second
        step_dt = 1.0 / 60 # 60Hz
        n_sim_step = int(length / 10 / step_dt)
        self.pusher_body.velocity = self.velocity.tolist()

        for i in range(n_sim_step):
            self.pusher_body.velocity = self.velocity.tolist()
            self.space.step(step_dt)
            self.global_time += step_dt

            # save infos
            if self.dir != None:
                if (self.detect_collision() and i % 30 == 0) or \
                    (i % 60 == 0):
                    img = self.render()
                    cv2.imwrite(os.path.join(self.dir, '%d_color.jpg' % self.count), img)
                    particles = self.get_all_object_keypoints()[0]
                    self.particle_pos_list.append(particles)
                    eef_states = self.get_eef_states()
                    self.eef_states_list.append(eef_states)
                    self.count += 1

        self.wait(1.0)
        if self.dir != None:
            self.step_list.append(self.count)

        img = self.render()
        return img, self.particle_pos_list, self.eef_states_list, self.step_list, self.contact_list


    # def get_env_state(self, rel=True):
    #     """
    #     Return the environment state.
    #     """
    #     env_dict = {
    #         "state": self.get_kp_state(),
    #         "pusher_pos": self.get_pusher_position(),
    #         "action": self.velocity,
    #     }
    #     if rel:
    #         env_dict["state"][0::2] -= env_dict["pusher_pos"][0]
    #         env_dict["state"][1::2] -= env_dict["pusher_pos"][1]
    #     if self.IMG_STATE:
    #         env_dict["image"] = self.get_img_state()
    #     env_state = np.concatenate([env_dict["state"], env_dict["pusher_pos"], env_dict["action"]], axis=0)
    #     return env_state, env_dict

    def wait(self, time):
        """
        Wait for some time in the simulation. Gives some time to stabilize bodies in collision.
        """
        t = 0
        step_dt = 1 / 60.0
        while t < time:
            self.space.step(step_dt)
            t += step_dt

    """
    2. Methods related to rendering and image publishing
    """

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
            pusher_pos = np.array(pusher_pos, dtype=np.int32)
            cv2.circle(img, pusher_pos, self.pusher_size, self.pusher_color[:3], -1)
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

    # def gen_img_from_poses(self, poses, pusher_pos, img_file=None):
    #     """
    #     Generate an image from a list of object poses and pusher position.
    #     """
    #     assert len(poses) == self.obj_num, "Number of poses does not match the number of objects!"
    #     img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
    #     img[:] = self.background_color[:3]
    #     for i, pose in enumerate(poses):
    #         if "insert" in self.task_name:
    #             obj = self.gen_vertices_from_pose(pose, "peg" if i % 2 else "hole")
    #         else:
    #             obj = self.gen_vertices_from_pose(pose)
    #         polys = np.array(obj, np.int32)
    #         cv2.fillPoly(img, polys, self.object_colors[i % len(self.object_colors)][:3])

    #     pusher_pos = np.array(pusher_pos, dtype=np.int32)
    #     cv2.circle(img, pusher_pos, self.pusher_size, self.pusher_color[:3], -1)

    #     # cv2 has BGR format, and flipped y-axis
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     img = cv2.flip(img, 0)
    #     img = cv2.resize(img, (self.img_size, self.img_size))
    #     if img_file is not None:
    #         cv2.imwrite(img_file, img)
    #         print(f"img saved to {img_file}")
    #     # cv2.imwrite("test.png", img)
    #     # print("img saved to test.png")
    #     # cv2.imshow('image', img)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     img = img / 255.0
    #     return img

    def close(self):
        """
        Close the simulation.
        """
        # if self.window is not None:
        #     self.window.close()
        cv2.destroyAllWindows()

    # def get_img_state(self):
    #     img = self.current_image
    #     assert img is not None, "Image is not initialized!"
    #     img = cv2.resize(img, (self.img_size, self.img_size))
    #     # cv2.imshow('image', img)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     img = img / 255.0
    #     return img

    # def save_mp4(self, filename="output_video.mp4", fps=10):
    #     """
    #     Save the list of images as a video.

    #     Parameters:
    #         filename (str): Video filename
    #         fps (int): Frames per second
    #     """
    #     if not self.SAVE_IMG:
    #         print("no save")
    #         return
    #     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #     out = cv2.VideoWriter(filename, fourcc, fps, (self.width, self.height))

    #     try:
    #         for frame in self.image_list:
    #             out.write(frame)
    #         out.release()
    #         print(f"Video saved as {filename}")
    #     except Exception as e:
    #         out.release()  # Make sure to release the video writer object
    #         print(f"Failed to save video. Error: {e}")
    #     # self.image_list = []

    def play_mp4(self, filename="output_video.mp4"):
        """
        Play the video using OpenCV (or do other processing)

        Parameters:
            filename (str): Video filename
        """
        if not os.path.exists(filename):
            print(f"Error: File {filename} does not exist.")
            return
        cap = cv2.VideoCapture(filename)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imshow(filename, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

    # def save_gif(self, filename="output_video.gif", fps=5):
    #     """
    #     Save the list of images as a gif.

    #     Parameters:
    #         filename (str): Gif filename
    #         fps (int): Frames per second
    #     """
    #     if not self.SAVE_IMG:
    #         print("no save")
    #         return
    #     images = []
    #     for frame in self.image_list:
    #         images.append(Image.fromarray(frame[:, :, ::-1]))
    #     images[0].save(filename, save_all=True, append_images=images[1:], optimize=False, duration=2000 / len(self.image_list), loop=0)
    #     print(f"Gif saved as {filename}")
    #     # self.image_list = []

    def play_gif(self, filename="output_video.gif"):
        """
        Play the gif using OpenCV (or do other processing)

        Parameters:
            filename (str): Gif filename
        """
        if not os.path.exists(filename):
            print(f"Error: File {filename} does not exist.")
            return
        # Read the gif from the file
        img = Image.open(filename)
        frames = [frame.copy() for frame in ImageSequence.Iterator(img)]

        for frame in frames:
            # Convert the PIL image to an OpenCV frame
            # opencv_image = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            opencv_image = np.array(frame)

            # Display the frame
            cv2.imshow(filename, opencv_image)

            if cv2.waitKey(100) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    """
    3. Methods for External Commands
    """

    def reset(self, new_dir=None, new_poses=None):
        self.dir = new_dir
        self.remove_all_objects()
        if self.pusher_body is not None:
            self.remove_pusher()
        self.pusher_body = None
        self.pusher_shape = None
        self.add_objects(self.obj_num, new_poses)
        if self.dir is not None and not os.path.exists(self.dir):
            os.makedirs(self.dir)

        self.wait(1.0)  # Give some time for collision pieces to stabilize.
        if self.dir != None:
            img = self.render()
            cv2.imwrite(os.path.join(self.dir, '%d_color.jpg' % self.count), img)
            particles = self.get_all_object_keypoints()[0]
            self.particle_pos_list.append(particles)
            eef_states = self.get_eef_states()
            self.eef_states_list.append(eef_states)
        self.count += 1
        self.step_list.append(self.count)

        return self.particle_pos_list, self.eef_states_list, self.step_list, self.contact_list

    def sample_action(self):
        raise NotImplementedError
