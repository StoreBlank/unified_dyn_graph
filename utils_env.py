import numpy as np
import scipy
import yaml

def load_yaml(filename):
    # load YAML file
    return yaml.safe_load(open(filename, 'r'))

def load_cloth(path):
    """Load .obj of cloth mesh. Only quad-mesh is acceptable!
    Return:
        - vertices: ndarray, (N, 3)
        - triangle_faces: ndarray, (S, 3)
        - stretch_edges: ndarray, (M1, 2)
        - bend_edges: ndarray, (M2, 2)
        - shear_edges: ndarray, (M3, 2)
    This function was written by Zhenjia Xu
    email: xuzhenjia [at] cs (dot) columbia (dot) edu
    website: https://www.zhenjiaxu.com/
    """
    vertices, faces = [], []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        # 3D vertex
        if line.startswith('v '):
            vertices.append([float(n)
                             for n in line.replace('v ', '').split(' ')])
        # Face
        elif line.startswith('f '):
            idx = [n.split('/') for n in line.replace('f ', '').split(' ')]
            face = [int(n[0]) - 1 for n in idx]
            assert(len(face) == 4)
            faces.append(face)

    triangle_faces = []
    for face in faces:
        triangle_faces.append([face[0], face[1], face[2]])
        triangle_faces.append([face[0], face[2], face[3]])

    stretch_edges, shear_edges, bend_edges = set(), set(), set()

    # Stretch & Shear
    for face in faces:
        stretch_edges.add(tuple(sorted([face[0], face[1]])))
        stretch_edges.add(tuple(sorted([face[1], face[2]])))
        stretch_edges.add(tuple(sorted([face[2], face[3]])))
        stretch_edges.add(tuple(sorted([face[3], face[0]])))

        shear_edges.add(tuple(sorted([face[0], face[2]])))
        shear_edges.add(tuple(sorted([face[1], face[3]])))

    # Bend
    neighbours = dict()
    for vid in range(len(vertices)):
        neighbours[vid] = set()
    for edge in stretch_edges:
        neighbours[edge[0]].add(edge[1])
        neighbours[edge[1]].add(edge[0])
    for vid in range(len(vertices)):
        neighbour_list = list(neighbours[vid])
        N = len(neighbour_list)
        for i in range(N - 1):
            for j in range(i+1, N):
                bend_edge = tuple(
                    sorted([neighbour_list[i], neighbour_list[j]]))
                if bend_edge not in shear_edges:
                    bend_edges.add(bend_edge)

    return np.array(vertices), np.array(triangle_faces),\
        np.array(list(stretch_edges)), np.array(
            list(bend_edges)), np.array(list(shear_edges))

def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo

def rand_int(lo, hi):
    return np.random.randint(lo, hi)

def quatFromAxisAngle(axis, angle):
    axis /= np.linalg.norm(axis)

    half = angle * 0.5
    w = np.cos(half)

    sin_theta_over_two = np.sin(half)
    axis *= sin_theta_over_two

    quat = np.array([axis[0], axis[1], axis[2], w])

    return quat

def quaternion_multuply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])

def rotation_to_quaternion(rot):
    # Ensure the rotation matrix is in the correct shape (3x3)
    if rot.shape != (3, 3):
        raise ValueError("Rotation matrix must be a 3x3 matrix")

    # Allocate space for the quaternion
    q = np.zeros(4)

    # Calculate each component of the quaternion
    q[0] = np.sqrt(max(0, 1 + rot[0, 0] + rot[1, 1] + rot[2, 2])) / 2
    q[1] = np.sqrt(max(0, 1 + rot[0, 0] - rot[1, 1] - rot[2, 2])) / 2
    q[2] = np.sqrt(max(0, 1 - rot[0, 0] + rot[1, 1] - rot[2, 2])) / 2
    q[3] = np.sqrt(max(0, 1 - rot[0, 0] - rot[1, 1] + rot[2, 2])) / 2

    # Determine the sign of each quaternion component
    q[1] *= np.sign(rot[2, 1] - rot[1, 2])
    q[2] *= np.sign(rot[0, 2] - rot[2, 0])
    q[3] *= np.sign(rot[1, 0] - rot[0, 1])

    return q

def degs_to_quat(deg_xyz, init_rot):
    deg_x, deg_y, deg_z = deg_xyz
    rad_x, rad_y, rad_z = np.deg2rad(deg_x), np.deg2rad(deg_y), np.deg2rad(deg_z)
    
    rot = init_rot
    rot_y = np.array([[np.cos(rad_x), 0., np.sin(rad_x)], [0., 1., 0.], [-np.sin(rad_x), 0., np.cos(rad_x)]])
    rot_x = np.array([[1., 0., 0.], [0., np.cos(rad_y), -np.sin(rad_y)], [0., np.sin(rad_y), np.cos(rad_y)]])
    rot_z = np.array([[np.cos(rad_z), -np.sin(rad_z), 0.], [np.sin(rad_z), np.cos(rad_z), 0.], [0., 0., 1.]])
    rot = rot @ rot_z @ rot_y @ rot_x
    
    quat = rotation_to_quaternion(rot)
    return quat
    

def find_min_distance(X, Z, k):
    """Find the top k minimum distance between point X and set of points Z using numpy."""
    Z_array = np.array(Z)
    distances = np.linalg.norm(Z_array - X, axis=1)
    # find k minimum distance
    index = np.argsort(distances)[:k]
    min_distances = distances[index[0]]
    return min_distances, index

def fps_with_idx(points, N):
    """
    Input:
        points: np.array() particle positions
        N: int sample number
    Output:
        points[farthest_pts_idx]: np.array() sampled points
        farthest_pts_idx: np.array() indices of the sampled points
    """
    if N > len(points):
        return points, np.arange(len(points))
    else:
        # start with the first point
        farthest_pts_idx = [0]
        distances = np.full(len(points), np.inf)
        
        for _ in range(1, N):
            last_point = points[farthest_pts_idx[-1]]
            new_distances = np.linalg.norm(points - last_point, axis=1)
            distances = np.minimum(distances, new_distances)
            farthest_pts_idx.append(np.argmax(distances))
            
        return points[farthest_pts_idx], np.array(farthest_pts_idx)