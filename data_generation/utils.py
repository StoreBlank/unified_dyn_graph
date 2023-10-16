import yaml
import numpy as np
import pyflex

def load_yaml(filename):
    # load YAML file
    return yaml.safe_load(open(filename, 'r'))

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

def set_scene(obj):
    if obj == 'Tshirt':
        data_path = '../assets/cloth3d/Tshirt2.obj'
        retval = load_cloth(data_path)
        mesh_verts = retval[0]
        mesh_faces = retval[1]
        mesh_stretch_edges, mesh_bend_edges, mesh_shear_edges = retval[2:]

        mesh_verts = mesh_verts * 3.5
        
        cloth_pos = [-1., 1., 0.]
        cloth_size = [20, 20]
        stiffness = [1.0, 0.85, 0.85] # [stretch, bend, shear]
        cloth_mass = 1.0
        particle_r = 0.00625
        render_mode = 2
        flip_mesh = 0
        
        # 0.6, 1.0, 0.6
        dynamicFriction = 0.5
        staticFriction = 1.0
        particleFriction = 0.5
        
        scene_params = np.array([
            *cloth_pos,
            *cloth_size,
            *stiffness,
            cloth_mass,
            particle_r,
            render_mode,
            flip_mesh, 
            dynamicFriction, staticFriction, particleFriction])
        
        pyflex.set_scene(
                29,
                scene_params,
                mesh_verts.reshape(-1),
                mesh_stretch_edges.reshape(-1),
                mesh_bend_edges.reshape(-1),
                mesh_shear_edges.reshape(-1),
                mesh_faces.reshape(-1),
                0)
    
    else:
        raise ValueError("Unknown object: %s" % obj)
    
    return scene_params

def set_table(table_size, table_height):
        halfEdge = np.array([table_size/2., table_height, table_size/2.])
        center = np.array([0., 0., 0.])
        quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
        hideShape = 0
        color = np.ones(3) * (160. / 255.)
        table_shape_states = np.zeros((1, 14))
        pyflex.add_box(halfEdge, center, quats, hideShape, color)
        table_shape_states[0] = np.concatenate([center, center, quats, quats])
        return table_shape_states
