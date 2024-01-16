import os
import cv2
import time
import numpy as np
import argparse
import multiprocessing as mp

import pygame
from box_sim import BoxSim

def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo

def convert_coordinates(point, screen_height):
        return point[0], screen_height - point[1]

# data type: "custom" or "random"
def main(args, info):
    
    # load info
    epi_idx = info["epi_idx"]
    out_root = info["out_root"]
    data_type = info["data_type"]
    debug = info["debug"]
    
    start_time = time.time()
    
    # create folder
    if not debug:
        out_dir = os.path.join(out_root, f"episode_{epi_idx:03d}")
        os.makedirs(out_dir, exist_ok=True)
    
    # set env
    screen_width, screen_height = 720, 720
    sim = BoxSim(screen_width, screen_height)

    # center of mass and friction
    box_size = sim.get_obj_size()
    if data_type == "custom":
        center_of_mass = (args.com_x, args.com_y)
        friction = args.friction
    elif data_type == "random":
        np.random.seed(epi_idx)
        center_of_mass = (rand_float(0, box_size[0]), rand_float(0, box_size[1]))
        friction = 0.5
    sim.add_box(center_of_mass, friction)
    print(f"Episode {epi_idx}, center of mass: {center_of_mass}, friction: {friction}")

    # init pos for pusher
    box_pos = sim.get_obj_state()[:2]
    # print("box init pos: ", box_pos)
    pusher_choice = np.random.choice([0, 1, 2, 3])
    if pusher_choice == 0: # top to bottom
        pusher_pos = (box_pos[0] - center_of_mass[0], box_pos[1] + 200)
    elif pusher_choice == 1: # bottom to top
        pusher_pos = (box_pos[0] - center_of_mass[0], box_pos[1] - 200)
    elif pusher_choice == 2: # left to right
        pusher_pos = (box_pos[0] - center_of_mass[0] - 200, box_pos[1] - center_of_mass[1])
    elif pusher_choice == 3: # right to left
        pusher_pos = (box_pos[0] - center_of_mass[0] + 200, box_pos[1] - center_of_mass[1])

    n_iter_rest = 100
    for i in range(n_iter_rest):
        sim.update(pusher_pos)

    n_sim_step = 50
    box_states = []
    eef_states = []
    for i in range(n_sim_step):
        
        # print("%d/%d" % (i, n_sim_step))
        
        pusher_x, pusher_y = pusher_pos
        
        if pusher_choice == 0: # top to bottom
            pusher_y -= 10
        elif pusher_choice == 1: # bottom to top
            pusher_y += 10
        elif pusher_choice == 2: # left to right
            pusher_x += 10
        elif pusher_choice == 3: # right to left
            pusher_x -= 10
            
        pusher_pos = (pusher_x, pusher_y)
        sim.update(pusher_pos)
        
        # save image
        if not debug:
            img_out_dir = os.path.join(out_dir, "images")
            os.makedirs(img_out_dir, exist_ok=True)
            out_path = os.path.join(img_out_dir, f"{i:03d}.png")
            sim.save_image(out_path)
            
            # save info
            box_init_state = sim.get_obj_state()[:3] # (x, y, theta)
            box_state = np.array([box_init_state[0] - center_of_mass[0], box_init_state[1] - center_of_mass[1], box_init_state[2]])
            eef_state = np.array(pusher_pos)
            box_states.append(box_state)
            eef_states.append(eef_state)
        
        time.sleep(0.1)
    
    if not debug:
        np.save(os.path.join(out_dir, "box_states.npy"), np.array(box_states))
        np.save(os.path.join(out_dir, "eef_states.npy"), np.array(eef_states))
        np.save(os.path.join(out_dir, "center_of_mass.npy"), np.array(center_of_mass))
    
    end_time = time.time()
    print(f"Episode {epi_idx} finshed!!! Time: {end_time - start_time}")

    sim.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--com_x", type=int, default=0)
    parser.add_argument("--com_y", type=int, default=0)
    parser.add_argument("--friction", type=float, default=0.5)
    args = parser.parse_args()

    out_root = "/mnt/sda/data/box"
    
    # epi_idx = np.random.randint(0, 1000)
    for epi_idx in range(10):
        info = {
            "epi_idx": epi_idx,
            "out_root": out_root,
            "data_type": "random",
            "debug": False,
        }
        main(args, info)
    
    ## multi-processing
    # n_worker = 10
    # bases = [0]
    # for base in bases:
    #     print(f"base: {base}")
    #     infos = []
    #     for i in range(n_worker):
    #         info = {
    #             "epi_idx": base + i,
    #             "out_root": out_root,
    #             "data_type": "random",
    #             "debug": False
    #         }
    #         infos.append(info)
    #     pool = mp.Pool(processes=n_worker)
    #     pool.map(main, infos)
    
