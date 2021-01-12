import meshio
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pickle
import gdist

from utils.mesh_tools import Mesh



def create_from_ply(ply_name=None, write=False, read=False):
    path = './HW3_Resources/' + ply_name + '.ply'
    path_write = './obj_cache/'
    ply_obj, obj_mesh, obj_gdist = [], [], []
    ply_obj = meshio.read(path)
    if write == True:
        np.save(path_write + ply_name + '.obj', ply_obj)
        mesh_vertices = ply_obj.points
        mesh_faces = ply_obj.cells_dict['triangle']
        obj_mesh = Mesh(mesh_vertices, np.c_[3*np.ones(np.shape(mesh_faces)[0]), mesh_faces])
        np.save(path_write + ply_name + '_mesh.obj', obj_mesh)
        obj_gdist = gdist.local_gdist_matrix(mesh_vertices.astype(np.float64), mesh_faces.astype(np.int32))
        np.save(path_write + ply_name + '_gdist', obj_gdist)
    else:
        try:
            obj_gdist = np.load(path_write + ply_name + '_gdist.npy', allow_pickle=True)[()]
        except  EOFError:
            print('EOFError')
    return ply_obj, obj_gdist