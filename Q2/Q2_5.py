import numpy as np
import pyvista as pv
import scipy.sparse.linalg as linalg

from utils.create_from_ply import create_from_ply
from utils.mesh_tools import Mesh

## Load the FAUST object from PLY and create Mesh

ply_name = 'tr_reg_000.ply'
path_read = 'HW3_Resources/' + ply_name
obj_mesh = Mesh('ply', path_read)
ply_obj, obj_gdist = create_from_ply(ply_name='tr_reg_000', write=False, read=True)
gdist_dist = obj_gdist[:,900]
cls_s = ['half_cotangent']

# def euc_dist_func(vertices):
#     return np.linalg.norm(vertices, axis=1)

## Compute the Laplacian eigen Decomposition

## Euclidean Distance - up to k bandwidth

# k = 5
# k_list = [2, 12, 30, 100]

# for i, cls_ in enumerate(cls_s):
#     plotter = pv.Plotter(shape=(1, len(k_list)))
#     eig_vals_lap, eig_vecs_lap = obj_mesh.laplacian_spectrum(k=max(k_list), cls=cls_)
#     for j, k in enumerate(k_list):
#         plotter.subplot(0, j)
#         plotter.add_text('k = ' + str(k))
#         # func = np.linalg.norm(obj_mesh.v, axis=1) ## Euclidean
#         # func = obj_mesh.va_map
#         func = gdist_dist
#         M = obj_mesh.barycenter_vertex_mass_matrix()
#         scalar_func = eig_vecs_lap[:,:k] @ eig_vecs_lap[:,:k].T @ M @ func
#         obj_mesh.render_surface(scalar_func, cmap_name='winter', plotter=plotter)
#         plotter.add_scalar_bar()
#     plotter.show(auto_close=False)

## Euclidean Distance - Full bandwidth 

# func = np.linalg.norm(obj_mesh.v, axis=1) ## Euclidean
# func = obj_mesh.va_map
# func = gdist_dist.toarray()
# plotter = pv.Plotter()    
# obj_mesh.render_surface(func, cmap_name='winter', plotter=plotter)
# plotter.add_scalar_bar()
# plotter.add_text('Full Bandwidth Euclidean Distance')
# plotter.show()

 
## Function Laplacian Euclidean Distance 

for i, cls_ in enumerate(cls_s):
    L = obj_mesh.laplacian(cls=cls_)
    M = obj_mesh.barycenter_vertex_mass_matrix()
    # func = np.linalg.norm(obj_mesh.v, axis=1) ## Euclidean
    # func = obj_mesh.va_map
    func = gdist_dist.toarray()
    M_inv = linalg.inv(M)
    scalar_func = M_inv @ L @ func
    thresh_max = np.mean(scalar_func) + np.std(scalar_func)
    thresh_min = np.mean(scalar_func) - np.std(scalar_func)
    scalar_func = np.clip(scalar_func, thresh_min, thresh_max)
    plotter = pv.Plotter()
    obj_mesh.render_surface(scalar_func, cmap_name='winter', plotter=plotter)
    plotter.add_scalar_bar()
    plotter.show(auto_close=False)

print('ya')