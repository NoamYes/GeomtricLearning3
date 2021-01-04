import meshio
import matplotlib.pyplot as plt
import numpy as np
import pickle
import gdist
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS

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
            file_gdist = open(path_write + ply_name + '_gdist.obj', 'rb') 
            obj_gdist = pickle.load(file_gdist, encoding='utf8')
            # obj_gdist = np.load(path_write + ply_name + '_gdist_np_DATA.obj.npy', allow_pickle=True) 
        except  EOFError:
            print('EOFError')
    return ply_obj, obj_gdist


# Create from ply and read gdist from file
ply_obj, obj_gdist = create_from_ply(ply_name='tr_reg_000', write=False, read=True)


## Plot original mesh

mesh_vertices = ply_obj.points
mesh_faces = ply_obj.cells_dict['triangle']
obj_mesh = Mesh(mesh_vertices, np.c_[3*np.ones(np.shape(mesh_faces)[0]), mesh_faces])
# plt.figure(1)

obj_mesh.render_surface(ply_obj.points, cmap_name='winter')
plt.show()

#  Plot man1 LLE with various n_neighbors
neighbors_list = [100]
# fig2 = plt.figure(figsize=(15, 6))
for i, n_neighbor in enumerate(neighbors_list):

    ## Compute standard MDS embedding

    # obj_reduced_LLE = Locally_Linear_Embedding(obj_mesh.v, n_neighbor=n_neighbor, epsilon=1, d=3, aff_mat=obj_gdist.toarray() > 0)
    # embedding = LocallyLinearEmbedding(n_neighbors=n_neighbor,n_components=3)
    # obj_reduced_LLE = embedding.fit_transform(obj_gdist)
    # embedding = MDS(n_components=3, dissimilarity='precomputed')
    # obj_reduced_MDS = embedding.fit_transform(obj_gdist.toarray())
    # obj_reduced_mesh = Mesh(obj_reduced_MDS, obj_mesh.f)
    # # plt.figure(2+i)
    # obj_reduced_mesh.render_surface(obj_reduced_mesh.v.real, cmap_name='winter')

    ## COmpute 2 method standatd MDS

    d = 3
    D_mat = obj_gdist
    n = D_mat.shape[0]
    J = np.eye(n) - (1/n)*np.ones((n,n))
    MDS_mat = -(1/2)*J*D_mat*D_mat*J
    MDS_eigVals, MDS_eigVecs = np.linalg.eig(MDS_mat)
    idx_pn = np.abs(MDS_eigVals).argsort()[::-1]
    MDS_eigVals = MDS_eigVals[idx_pn]
    MDS_eigVecs = MDS_eigVecs[:, idx_pn]
    p_reduced_MDS = MDS_eigVecs[:,:d]*np.expand_dims(MDS_eigVals[:d]**0.5,0)

    p_MDS_mesh = Mesh(p_reduced_MDS, obj_mesh.f)

    ## Compute spherical embedding

    d = 3
    D_mat = obj_gdist
    d_mat_sphere = np.cos(D_mat)
    D_eigVals, D_eigVecs = np.linalg.eig(d_mat_sphere)
    idx_pn = np.abs(D_eigVals).argsort()[::-1]
    D_eigVals = D_eigVecs[idx_pn]
    D_eigVecs = D_eigVecs[:, idx_pn]
    p_reduced_sphere = D_eigVecs[:,:d]*np.expand_dims(D_eigVals[:d]**0.5,0)

    # plt.figure(2+i)
    p_sphere_mesh = Mesh(p_reduced_sphere, obj_mesh.f)



    # axs = fig2.add_subplot(1, len(neighbors_list), i+1, projection='3d')
    # scat = axs.scatter(man1_reduced_LLE[:, 0], man1_reduced_LLE[:, 1], man1_reduced_LLE[:, 2], s=5)
    # axs.set_title('n_neighbors=' + str(n_neighbor))

# fig2.suptitle('man1 Local Linear Embedding (LLE) MAP for multiple n_neighbors')
plt.show()



print('ya')