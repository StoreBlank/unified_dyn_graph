import numpy as np
import cv2

import pygame

"""
Minimal example for pile simulation.
"""


pygame.init()
screen = pygame.display.set_mode((500, 500))
clock = pygame.time.Clock()

from carrot_sim import CarrotSim

sim = CarrotSim('plain') # initialize sim, choose between 'bin' or 'plain'
count = 0

def convert_coordinates(point):
    return point[0], 500 - point[1]

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