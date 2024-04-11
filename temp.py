import matplotlib.pyplot as plt
import numpy as np

# Function to check if a point is inside the given polygon
def is_inside_polygon(points, p):
    n = len(points)
    inside = False
    
    # Function to calculate x intercept for horizontal line intersecting a polygon edge
    x_intercept = lambda p1, p2, y: p1[0] + (p2[0] - p1[0]) * (y - p1[1]) / (p2[1] - p1[1])
    
    p1 = points[0]
    for i in range(n + 1):
        p2 = points[i % n]
        if p[1] > min(p1[1], p2[1]):
            if p[1] <= max(p1[1], p2[1]):
                if p[0] <= max(p1[0], p2[0]):
                    if p1[1] != p2[1]:
                        xints = x_intercept(p1, p2, p[1])
                        if p1[0] == p2[0] or p[0] <= xints:
                            inside = not inside
        p1 = p2
    
    return inside

# Define the corners of the polygon
polygon = [(0.1, 0), (0.1, 1), (0.5, 1), (0.7, 0.6), (0.7, 0)]

# Define the number of points to sample
num_samples = 1000

# Initialize a list to hold the sampled points
sampled_points = []

# Sample points until we have the desired number
while len(sampled_points) < num_samples:
    # Generate a random point within the bounds of the polygon's x and y
    friction = np.random.uniform(0.1, 0.7)
    stiffness = np.random.uniform(0, 1)
    random_point = (friction, stiffness)
    
    # Check if the point is inside the polygon
    if is_inside_polygon(polygon, random_point):
        sampled_points.append(random_point)

# Plotting code (for visualization purposes, can be removed if not needed)
polygon.append(polygon[0])  # Repeat the first point to create a 'closed loop'
xs, ys = zip(*polygon)
plt.plot(xs, ys, marker='o')

sampled_x, sampled_y = zip(*sampled_points)
plt.scatter(sampled_x, sampled_y, alpha=0.5)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('Friction')
plt.ylabel('Stiffness')
plt.title('Random Sampled Points inside the Polygon')
plt.show()

# Return the sampled points
sampled_points
