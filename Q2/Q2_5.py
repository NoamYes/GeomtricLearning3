import numpy as np
import pyvista as pv
from utils.mesh_tools import Mesh
import scipy.sparse.linalg as linalg

## Load the FAUST object from PLY and create Mesh

ply_name = 'tr_reg_000.ply'
path_read = 'HW3_Resources/' + ply_name
obj_mesh = Mesh('ply', path_read)

cls_s = ['uniform', 'half_cotangent']

# def euc_dist_func(vertices):
#     return np.linalg.norm(vertices, axis=1)

## Compute the Laplacian eigen Decomposition

## Euclidean Distance - up to k bandwidth

# k = 5
# k_list = [2, 12, 30, 100]

# for i, cls_ in enumerate(cls_s):
#     plotter = pv.Plotter(shape=(1, k))
#     eig_vals_lap, eig_vecs_lap = obj_mesh.laplacian_spectrum(k=k, cls=cls_)
#     for j in range(k):
#         plotter.subplot(0, j)
#         plotter.add_text('k = ' + str(j+1))
#         func = np.linalg.norm(obj_mesh.v, axis=1)
#         M = obj_mesh.barycenter_vertex_mass_matrix()
#         scalar_func = eig_vecs_lap[:,:j+1] @ eig_vecs_lap[:,:j+1].T @ M @ func
#         obj_mesh.render_surface(scalar_func, cmap_name='winter', plotter=plotter)
#         plotter.add_scalar_bar()
#     plotter.show()

## Euclidean Distance - Full bandwidth 

for i, cls_ in enumerate(cls_s):
    plotter = pv.Plotter()
    L = obj_mesh.laplacian(cls=cls_)
    M = obj_mesh.barycenter_vertex_mass_matrix()
    func = np.linalg.norm(obj_mesh.v, axis=1)
    M_inv = linalg.inv(M)
    scalar_func = M_inv @ L @ func
    obj_mesh.render_surface(scalar_func, cmap_name='winter', plotter=plotter)
    plotter.add_scalar_bar()
    plotter.show()