import numpy as np
import pyvista as pv
from matplotlib import cm
import pickle

from utils.create_from_ply import create_from_ply
from utils.mesh_tools import Mesh

def farthest_point_sampling(mesh_vertices, n, gdist_mat):
    gdist_mat = gdist_mat.toarray()
    vertices = mesh_vertices
    vertices_num = vertices.shape[0]
    s1 = np.random.randint(vertices_num)
    S_inds = [s1] # Initialize compact set for vertices S inds
    while not len(S_inds) == n:
        v_S_dist = gdist_mat[:, S_inds]
        v_f = np.argmax(np.min(v_S_dist, axis=1),axis=0)
        S_inds.append(int(v_f))

    return S_inds, mesh_vertices[S_inds]

def plot_mesh_sampling_pointcloud(mesh, n, gdist=None, plotter=None):
        if gdist == None:
            mesh_gdist = gdist.local_gdist_matrix(mesh.v.astype(np.float64), mesh.f.astype(np.int32))
        else:
            mesh_gdist = gdist
        fps_inds, fps_vals = farthest_point_sampling(mesh.v, n, mesh_gdist)
        if plotter == None:
            plotter = pv.Plotter()
        mesh.render_surface(plotter=plotter, scalar_func=mesh.v)
        fps_pointcloud = pv.PolyData(fps_vals)
        return fps_pointcloud


# Create from ply and read gdist from file
ply_obj, obj_gdist = create_from_ply(ply_name='tr_reg_000', write=False, read=True)


## Create obj mesh
mesh_vertices = ply_obj.points
mesh_faces = ply_obj.cells_dict['triangle']
obj_mesh = Mesh(mesh_vertices, np.c_[3*np.ones(np.shape(mesh_faces)[0]), mesh_faces])

# Visualize for different n the farthest_sampling

n_list = [300, 700, 1300, 2000, 4000]
num_n = len(n_list)

plotter = pv.Plotter(shape=(1, num_n))

for i, n in enumerate(n_list):
    plotter.subplot(0, i)
    plotter.add_text("n = " + str(n))
    obj_mesh.render_surface(scalar_func=obj_mesh.v, plotter=plotter)
    fps_pointcloud = plot_mesh_sampling_pointcloud(obj_mesh, n=n, gdist=obj_gdist)
    plotter.add_mesh(fps_pointcloud, render_points_as_spheres=True, cmap=cm.get_cmap('winter'), point_size=8)


plotter.show()