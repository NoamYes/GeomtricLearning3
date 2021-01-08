import numpy as np
import pyvista as pv
from utils.mesh_tools import Mesh

## Load the FAUST object from PLY and create Mesh
ply_name = 'tr_reg_001.ply'
path_read = 'HW3_Resources/' + ply_name
obj_mesh = Mesh('ply', path_read)

eig_vals_lap, eig_vecs_lap = obj_mesh.laplacian_spectrum(k=5, cls='half_cotangent')

cls_s = ['half_cotangent']
k = 5 
plotters = []
for i, cls_ in enumerate(cls_s):
    plotters.append(pv.Plotter(i, shape=(1, k)))
    plotter = plotters[i]
    eig_vals_lap, eig_vecs_lap = obj_mesh.laplacian_spectrum(k=k, cls=cls_)
    for j in range(k):
        plotter.subplot(0, j)
        plotter.add_text('k = ' + str(j+1))
        obj_mesh.render_surface(eig_vecs_lap[:,j], cmap_name='winter', plotter=plotter)
        plotter.add_scalar_bar()
    
## Plot uniform

# plotters[0].show()

## Plot Cotangent

plotter = plotters[0]
# plotter.add_text('Uniform')
plotter.show()

print('ya')