import numpy as np

def farthest_point_sampling(mesh, n, gdist_mat):
    vertices = mesh.v
    vertices_num = vertices.len()
    faces = mesh.f
    nodes_inds = np.range(vertices_num)
    S_inds = [] # Initialize compact set for vertices S inds
    s1 = np.random.randint(vertices_num)
    S_inds.append(s1)
    while not len(S_inds) == n:
        v_S_dist = gdist_mat[nodes_inds, S_inds]
        v_f = np.argmax(np.min(v_S_dist, axis=0),axis=1)
        S_inds.append(v_f)

