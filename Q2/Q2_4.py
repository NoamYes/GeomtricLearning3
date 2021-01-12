import numpy as np
import pyvista as pv
from utils.mesh_tools import Mesh

## Load the FAUST object from PLY and create Mesh
# ply_name = 'tr_reg_001.ply'
# path_read = 'HW3_Resources/' + ply_name
# obj_mesh = Mesh('ply', path_read)

# H = obj_mesh.mean_curvature(cls='half_cotangent')

k = 5 
ply_objs = ['tr_reg_000.ply', 'tr_reg_001.ply']
for i, ply_name in enumerate(ply_objs):
    path_read = 'HW3_Resources/' + ply_name
    obj_mesh = Mesh('ply', path_read)
    # plotters.append(pv.Plotter(i, shape=(1, k)))
    H = obj_mesh.mean_curvature(cls='half_cotangent')
    plotter = pv.Plotter()
    thresh_max = np.mean(H) + 0.5*np.std(H)
    thresh_min = np.mean(H) - 0.5*np.std(H)
    scalars = np.clip(H, thresh_min, thresh_max)
    obj_mesh.render_surface(scalars, cmap_name='winter', plotter=plotter)
    plotter.add_scalar_bar()
    plotter.show(auto_close=False)
