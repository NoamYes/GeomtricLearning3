import meshio
import matplotlib.pyplot as plt
import numpy as np
import pickle
import gdist
from sklearn.manifold import LocallyLinearEmbedding

from MDS import Locally_Linear_Embedding
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
        np.save(path_write + ply_name + '_gdist.obj', obj_gdist)
    else:
        try:
            # file_obj = open(path_write + ply_name + '.obj', 'rb') 
            # ply_obj = pickle.load(file_obj, encoding='utf8')
            # file_obj = open(path_write + ply_name + '_mesh.obj', 'rb') 
            # obj_mesh = pickle.load(file_obj, encoding='utf8')
            file_obj = open(path_write + ply_name + '_gdist.obj', 'rb') 
            obj_gdist = pickle.load(file_obj, encoding='utf8')
        except  EOFError:
            print('EOFError')
    return ply_obj, obj_mesh, obj_gdist

## Import man1, man2

# tr_reg_000 = meshio.read("./HW3_Resources/tr_reg_000.ply")
# tr_reg_001 = meshio.read("./HW3_Resources/tr_reg_001.ply")
# man1_ply = meshio.read("./HW3_Resources/man1.ply")
# man2_ply = meshio.read("./HW3_Resources/man2.ply")

## Save them to file
# filehandler = open('tr_reg_000.obj', 'wb') 
# pickle.dump(tr_reg_000, filehandler)
# filehandler = open('tr_reg_001.obj', 'wb') 
# pickle.dump(tr_reg_001, filehandler)

ply_obj, obj_mesh, obj_gdist = create_from_ply(ply_name='tr_reg_000', write=False, read=True)

## Read from file

# try:
#     file_tr_reg_000 = open('tr_reg_000.obj', 'rb') 
#     tr_reg_000_ply = pickle.load(file_tr_reg_000)
#     file_tr_reg_001 = open('tr_reg_001.obj', 'rb') 
#     tr_reg_001_ply = pickle.load(file_tr_reg_001)
# except  EOFError:
#     print('EOFError')


## Plot original mesh

# obj_vertices = ply_obj.points
# obj_faces = ply_obj.cells_dict['triangle']
# obj_mesh = Mesh(obj_vertices, np.c_[3*np.ones(np.shape(obj_faces)[0]), obj_faces])
# plt.figure(1)

obj_mesh.render_surface(ply_obj.points, cmap_name='winter')
plt.show()

# tr_reg_001_gdist = gdist.local_gdist_matrix(man1_vertices.astype(np.float64), man1_faces.astype(np.int32))


## Read them from file 

#  Plot man1 LLE with various n_neighbors
neighbors_list = [25, 100, 200]
# fig2 = plt.figure(figsize=(15, 6))
for i, n_neighbor in enumerate(neighbors_list):
    # obj_reduced_LLE = Locally_Linear_Embedding(obj_mesh.v, n_neighbor=n_neighbor, epsilon=1, d=3)
    embedding = LocallyLinearEmbedding(n_neighbors=n_neighbor,n_components=3)
    obj_reduced_LLE = embedding.fit_transform(obj_gdist)
    obj_reduced_mesh = Mesh(obj_reduced_LLE, obj_mesh.f)
    # plt.figure(2+i)
    obj_reduced_mesh.render_surface(obj_reduced_mesh.v, cmap_name='winter')
    # axs = fig2.add_subplot(1, len(neighbors_list), i+1, projection='3d')
    # scat = axs.scatter(man1_reduced_LLE[:, 0], man1_reduced_LLE[:, 1], man1_reduced_LLE[:, 2], s=5)
    # axs.set_title('n_neighbors=' + str(n_neighbor))

# fig2.suptitle('man1 Local Linear Embedding (LLE) MAP for multiple n_neighbors')
plt.show()

print('ya')