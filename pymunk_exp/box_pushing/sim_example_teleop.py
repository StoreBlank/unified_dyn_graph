import numpy as np
import cv2
import argparse

import pygame
from box_sim import BoxSim

parser = argparse.ArgumentParser()
parser.add_argument("--com_x", type=int, default=0)
parser.add_argument("--com_y", type=int, default=0)
parser.add_argument("--friction", type=float, default=0.5)
args = parser.parse_args()

pygame.init()
screen_width, screen_height = 720, 720
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

sim = BoxSim(screen_width, screen_height)
sim.add_box()

def convert_coordinates(point):
    return point[0], screen_height - point[1]

while(True):

    for event in pygame.event.get():
        # compute random actions.
        # u = -0.5 + 1.0 * np.random.rand(4)
        u = pygame.mouse.get_pos()
        if (u == (0, 0)):
            continue
        
        u = convert_coordinates(u)
        sim.update(u)

        # save screenshot
        # image = sim.get_current_image()
        # cv2.imwrite("screenshot.png", sim.get_current_image())
        # count = count + 1

        '''
        # refresh rollout every 10 timesteps.
        if (count % 10 == 0):
            print(count // 10)
            sim.refresh()
        '''

    clock.tick(30)