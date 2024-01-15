import pygame
import pymunk
import pymunk.pygame_util

# Define collision types
BOX_COLLISION_TYPE = 1
WALL_COLLISION_TYPE = 2

# Initialize Pygame
pygame.init()
width, height = 600, 600
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(screen)

# Create a new space for the simulation
space = pymunk.Space()

# Create walls
walls = [
    pymunk.Segment(space.static_body, (0, 0), (width, 0), 1),  # Top
    pymunk.Segment(space.static_body, (0, height), (width, height), 1),  # Bottom
    pymunk.Segment(space.static_body, (0, 0), (0, height), 1),  # Left
    pymunk.Segment(space.static_body, (width, 0), (width, height), 1)  # Right
]
for wall in walls:
    wall.friction = 1.0
    wall.collision_type = WALL_COLLISION_TYPE
    space.add(wall)
    
    
# Create a rectangle box in the middle with an offset center of mass
box_body = pymunk.Body(1, 1666)
box_body.position = (width / 2, height / 2)

box_shape = pymunk.Poly.create_box(box_body, (100, 50))
box_body.center_of_gravity = (-50, 0)

box_shape.friction = 1.0
box_shape.collision_type = BOX_COLLISION_TYPE
space.add(box_body, box_shape)

# Create a pusher as a line segment
pusher_body = pymunk.Body(1, 100, pymunk.Body.DYNAMIC)
pusher_body.position = (width / 2, 100)  # Starting position of the pusher
line_start = (-10, 0)
line_end = (10, 0)
thickness = 2
pusher_shape = pymunk.Segment(pusher_body, line_start, line_end, thickness)
pusher_shape.friction = 1.0
space.add(pusher_body, pusher_shape)

# Function to move pusher towards the box
def move_pusher_towards_box(pusher, box, max_speed=100):
    dx = box.position.x - pusher.position.x
    dy = box.position.y - pusher.position.y
    distance = (dx**2 + dy**2)**0.5
    if distance > 10:  # To avoid division by zero and jittering near the target
        velocity = (max_speed * dx / distance, max_speed * dy / distance)
        pusher.velocity = velocity
    else:
        pusher.velocity = (0, 0)  # Stop when close to the box

def stop_box(arbiter, space, data):
    box_body = arbiter.shapes[0].body
    box_body.velocity = pymunk.Vec2d(0, 0)
    return True

def convert_coordinates(point):
    return point[0], height - point[1]

# Simulation loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((192, 192, 192))  # Fill the screen with white

    # Move the pusher towards the box
    move_pusher_towards_box(pusher_body, box_body)
    
    # Add collision handler
    handler = space.add_collision_handler(BOX_COLLISION_TYPE, WALL_COLLISION_TYPE)
    handler.begin = stop_box
    
    print("center of mass={0}\nposition ={1}\n".format(
    box_body.center_of_gravity, box_body.position))

    # Update the space and draw everything
    space.step(1/50.0)
    space.debug_draw(draw_options)

    pygame.display.flip()  # Update the full display surface to the screen
    clock.tick(50)  # Limit the frame rate
    
    

pygame.quit()
