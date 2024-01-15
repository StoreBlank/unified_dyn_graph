import pygame
import pymunk
import random

pygame.init()

screenWidth, screenHeight = 600, 600
display = pygame.display.set_mode((screenWidth, screenHeight))
clock = pygame.time.Clock()
space = pymunk.Space()
fps = 50

def convert_coordinates(point):
    return point[0], screenHeight - point[1]

class Ball():
    def __init__(self, x, y, collision_type, up=1):
        self.ball_radius = 10
        
        self.body = pymunk.Body()
        self.body.position = x, y
        self.body.velocity = random.uniform(-100, 100), random.uniform(-100, 100)
        
        self.shape = pymunk.Circle(self.body, self.ball_radius)
        self.shape.density = 1
        self.shape.elasticity = 1
        self.shape.collision_type = collision_type  
        
        space.add(self.body, self.shape)
    
    def draw(self):
        if self.shape.collision_type != 2:
            x, y = convert_coordinates(self.body.position)
            pygame.draw.circle(display, (255,0,0), (int(x), int(y)), self.ball_radius) #red
        else:
            x, y = convert_coordinates(self.body.position)
            pygame.draw.circle(display, (0,0,255), (int(x), int(y)), self.ball_radius) #blue
    
    def change_to_blue(self, arbiter, space, data):
        self.shape.collision_type = 2

def game():
    balls = [Ball(random.randint(0, screenWidth), random.randint(0, screenHeight), i+3) for i in range(100)]
    balls.append(Ball(400, 400, 2))
    # ball = Ball(100, 100, 1)
    # ball_2 = Ball(100, 500, 2, -1)
    
    # handler = space.add_collision_handler(1, 2)
    # handler.begin = collide
    
    handlers = [space.add_collision_handler(2, i+3) for i in range(100)]
    for i, handler in enumerate(handlers):
        handler.separate = balls[i].change_to_blue
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                return
        display.fill((255,255,255)) # white background
        
        # ball.draw()
        # ball_2.draw()
        
        [ball.draw() for ball in balls]
        
        pygame.display.update()
        clock.tick(fps)
        space.step(1/fps)

game()
pygame.quit()