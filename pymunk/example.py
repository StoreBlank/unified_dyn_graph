import pymunk
import pygame

## PYMUNK EXAMPLE
# space = pymunk.Space() # Create a Space which contain the simulation
# space.gravity = 0,-981 # Set its gravity

# body = pymunk.Body() # Create a Body with default mass and moment
# body.position = 50,100 # Set the position of the body

# poly = pymunk.Poly.create_box(body) # Create a box shape and attach to body
# poly.mass = 10 # Set the mass of the body
# space.add(body, poly) # Add both body and shape to the simulation

# print_options = pymunk.SpaceDebugDrawOptions() # For easy printing

# for _ in range(100): # Run the simulation for 100 steps in total
#     space.step(0.02) # Step the simulation one step forward
#     space.debug_draw(print_options) # Print the current state of the simulation


## PYGAME & PYMUNK EXAMPLE
"""

"""
pygame.init()
screehWidth, screenHeight = 800, 800
display = pygame.display.set_mode((800, 800))
clock = pygame.time.Clock()
fps = 50

space = pymunk.Space()
space.gravity = 0, -1000

def convert_coordinates(point):
    return point[0], screenHeight - point[1]

class Ball():
    def __init__(self):
        self.ball_radius = 30
        
        self.body = pymunk.Body()
        self.body.position = 400, 600
        
        self.shape = pymunk.Circle(self.body, self.ball_radius)
        self.shape.density = 1
        self.shape.elasticity = 1
        
        space.add(self.body, self.shape)
    
    def draw(self):
        x, y = convert_coordinates(self.body.position)
        pygame.draw.circle(display, (255,0,0), (int(x), int(y)), self.ball_radius)

class Floor():
    def __init__(self):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        # a, b: end points of the segment; 
        # radius: thickness of the segment
        self.shape = pymunk.Segment(self.body, (0, 50), (800, 50), 10)
        self.shape.elasticity = 1
        space.add(self.body, self.shape)
    
    def draw(self):
        pygame.draw.line(display, (0,0,0), (0, 550), (800, 750), 10) # convert coordinates


def game():
    ball = Ball()
    floor = Floor()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                return
        
        display.fill((255,255,255)) # white background
        ball.draw()
        floor.draw()
        
        pygame.display.update()
        clock.tick(fps)
        space.step(1/fps)
        
game()    
pygame.quit()













