import meshio
import matplotlib.pyplot as plt
import numpy as np
import pickle
import gdist
import scipy
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS


from MDS import Locally_Linear_Embedding
from utils.mesh_tools import Mesh
from utils.create_from_ply import create_from_ply


# Create from ply and read gdist from file

ply_obj, obj_gdist = create_from_ply(ply_name='tr_reg_000', write=False, read=True)


## Plot original mesh

ply_name = 'tr_reg_001.ply'
path_read = 'HW3_Resources/' + ply_name
mesh_vertices = ply_obj.points
mesh_faces = ply_obj.cells_dict['triangle']
obj_mesh = Mesh('ply', path_read)
# plt.figure(1)

# obj_mesh.render_surface(ply_obj.points, cmap_name='winter')
# plt.show()

#  Plot man1 LLE with various n_neighbors
neighbors_list = [300]
# fig2 = plt.figure(figsize=(15, 6))
for i, n_neighbor in enumerate(neighbors_list):

    ## Compute standard MDS embedding

    # embedding = MDS(n_components=3, dissimilarity='precomputed')
    # obj_reduced_MDS = embedding.fit_transform(obj_gdist.toarray())
    # obj_reduced_mesh = Mesh(obj_reduced_MDS, obj_mesh.f)
    # # plt.figure(2+i)
    # obj_reduced_mesh.render_surface(obj_reduced_mesh.v.real, cmap_name='winter')

    ## Compute 2 method standard MDS

    d = 3
    D_mat = obj_gdist.toarray()
    n = D_mat.shape[0]
    J = np.eye(n) - (1/n)*np.ones((n,n))
    MDS_mat = -(1/2) * J @ D_mat**2 @ J
    MDS_mat = MDS_mat.T @ MDS_mat
    MDS_eigVals, MDS_eigVecs = scipy.linalg.eigh(MDS_mat, subset_by_index=[n-d,n-1])
    MDS_eigVals = np.real(MDS_eigVals)
    MDS_eigVecs = np.real(MDS_eigVecs)
    p_reduced_MDS = MDS_eigVecs[:,:d]*np.expand_dims(MDS_eigVals[:d]**0.5,0)

    ## Save the standard MDS
    np.save('./obj_cache/p_reduced_MDS' +ply_name + '.npy', p_reduced_MDS, allow_pickle=True)

    ## Load the standard MDS
    # p_reduced_MDS = np.load('./obj_cache/p_reduced_MDS.npy')

    p_MDS_mesh = Mesh('vf', p_reduced_MDS.real, obj_mesh.f)

    p_MDS_mesh.render_surface(p_MDS_mesh.v.real, cmap_name='winter')
    # plt.show()

    ## Compute the geodesic distance matrix on the standard 
    
    g_dist_MDS = gdist.local_gdist_matrix(p_reduced_MDS, mesh_faces.astype(np.int32))
    np.save('./obj_cache/g_dist_MDS' +ply_name + '.npy', g_dist_MDS, allow_pickle=True)
    error_MDS = np.linalg.norm(g_dist_MDS.toarray() - obj_gdist.toarray(), 'fro')

    ## Compute spherical embedding

    d = 3
    D_mat = obj_gdist
    d_mat_sphere = np.cos(D_mat.toarray())
    D_eigVals, D_eigVecs = np.linalg.eig(d_mat_sphere)
    idx_pn = np.abs(D_eigVals).argsort()[::-1]
    D_eigVals = D_eigVals[idx_pn]
    D_eigVecs = D_eigVecs[:, idx_pn]
    p_reduced_sphere = D_eigVecs[:,:d]*np.expand_dims(D_eigVals[:d]**0.5,0)
    np.save('./obj_cache/p_reduced_sphere' +ply_name + '.npy', p_reduced_sphere, allow_pickle=True)

    ## Compute the geodesic distance on spherical MDS embedding
    
    g_dist_sphere_MDS = gdist.local_gdist_matrix(p_reduced_sphere, mesh_faces.astype(np.int32))
    np.save('./obj_cache/g_dist_sphere_MDS' +ply_name + '.npy', g_dist_sphere_MDS, allow_pickle=True)
    error_sphere_MDS = np.linalg.norm(g_dist_sphere_MDS.toarray() - obj_gdist.toarray(), 'fro')

    # plt.figure(2+i)
    # p_sphere_mesh = Mesh('vf', p_reduced_sphere, obj_mesh.f)

    # p_sphere_mesh.render_surface(p_sphere_mesh.v.real, cmap_name='winter')

    # axs = fig2.add_subplot(1, len(neighbors_list), i+1, projection='3d')
    # scat = axs.scatter(man1_reduced_LLE[:, 0], man1_reduced_LLE[:, 1], man1_reduced_LLE[:, 2], s=5)
    # axs.set_title('n_neighbors=' + str(n_neighbor))

# fig2.suptitle('Classic and Spherical MDS embedding for ' +  ply_name)
plt.show()
