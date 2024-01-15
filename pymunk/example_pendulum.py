import pygame
import pymunk
import random

pygame.init()

screenWidth, screenHeight = 600, 600
display = pygame.display.set_mode((screenWidth, screenHeight))
clock = pygame.time.Clock()
space = pymunk.Space()
space.gravity = 0, -1000
fps = 50

def convert_coordinates(point):
    return point[0], screenHeight - point[1]

class Ball():
    def __init__(self, x, y):
        self.ball_radius = 10
        
        self.body = pymunk.Body()
        self.body.position = x, y
        # self.body.velocity = random.uniform(-100, 100), random.uniform(-100, 100)
        
        self.shape = pymunk.Circle(self.body, self.ball_radius)
        self.shape.density = 1
        self.shape.elasticity = 1
        
        space.add(self.body, self.shape)
    
    def draw(self):
        x, y = convert_coordinates(self.body.position)
        pygame.draw.circle(display, (255,0,0), (int(x), int(y)), self.ball_radius) #red

class String():
    def __init__(self, body1, attachment, identifier="body"):
        self.body1 = body1
        
        if identifier == "body":
            self.body2 = attachment
        elif identifier == "position":
            self.body2 = pymunk.Body(body_type=pymunk.Body.STATIC)
            self.body2.position = attachment
        
        joint = pymunk.PinJoint(self.body1, self.body2)
        space.add(joint)
    
    def draw(self):
        x1, y1 = convert_coordinates(self.body1.position)
        x2, y2 = convert_coordinates(self.body2.position)
        pygame.draw.line(display, (0,0,0), (int(x1), int(y1)), (int(x2), int(y2)), 2)

def game():
    ball_1 = Ball(200, 450)
    ball_2 = Ball(100, 150)
    
    string_1 = String(ball_1.body, (300, 550), "position")
    string_2 = String(ball_1.body, ball_2.body, "body")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                return
        display.fill((255,255,255)) # white background
        
        ball_1.draw()
        ball_2.draw()
        string_1.draw()
        string_2.draw()
        
        pygame.display.update()
        clock.tick(fps)
        space.step(1/fps)

game()
pygame.quit()