import os
import numpy as np
import scipy
from scipy.spatial.distance import cdist as cdist
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, smacof
import json

from utils.mesh_tools import Mesh

faust_dir = 'MPI-FAUST/training/registrations/'

subjectIDs = []
poses = []

DNAs, GPSs, HKSs, Hs = [], [], [], []
D_DNA, D_GPS, D_HKS, D_H = [], [], [], []

# k = np.shape(obj_mesh.v)[0]
k = 5

## Calculate for each FAUST obj the relevent descriptors

def expspace(start, stop, n): 
    return np.exp(np.linspace(np.log(start), np.log(stop), n)) 

for filename in np.sort(os.listdir(faust_dir)):
    if filename.endswith(".ply"): 
        path_read = faust_dir + filename
        obj_mesh = Mesh('ply', path_read)

        eigVals, eigVecs = obj_mesh.laplacian_spectrum(k, 'uniform')
        
        # Calculate the shape DNA
        obj_DNA = eigVals[1:]
        DNAs.append(obj_DNA)

        # Calculate the GPS
        inv_sqrt_eigVals = np.diag(1/np.sqrt(eigVals[1:]))
        obj_GPS_ =  eigVecs[:,1:] @ inv_sqrt_eigVals 
        obj_GPS = obj_GPS_.T
        GPSs.append(obj_GPS.flatten())

        # Calclate the HKS
        t = expspace(0.01, 10, n=10)
        eig_t = np.expand_dims(t,axis=1) @ np.expand_dims(eigVals[1:], axis=1).T
        exp_t_eigVals = np.exp(-eig_t)
        eigVecsSqr = (eigVecs[:,1:].T)**2
        obj_HKS_ = exp_t_eigVals @ eigVecsSqr
        HKSs.append(obj_HKS_.flatten())

        # Calculate the mean curvature H
        H = obj_mesh.mean_curvature('uniform')
        Hs.append(H)

        print(filename)
        
## Reshape the results to matrix corresponding to subjectID (row) and pose (col)

DNAs = np.array(DNAs)
GPSs = np.array(GPSs)
HKSs = np.array(HKSs)
Hs = np.array(Hs)

path_write = './Q2/cache/'

## Save to file
# MATS_DICT = {'DNAs': DNAs, 'GPSs': GPSs, 'HKSs': HKSs, 'Hs': Hs}
# np.save(path_write + 'MATS_DICT_k=' + str(k), MATS_DICT)
np.savez(path_write + 'MATS_DICT_k=' + str(k) + '.npz', DNAs=DNAs, GPSs=GPSs, HKSs=HKSs, Hs=Hs)

## Load Dict from file

# MATS_DICT = np.load(path_write + 'MATS_DICT_k=' + str(k) + '.npy', allow_pickle=True)
# MATS_DICT = MATS_DICT[()]
MATS_DICT = np.load(path_write + 'MATS_DICT_k=' + str(k) + '.npz')

## Load Matrices from dict
DNAs = MATS_DICT['DNAs']
GPSs = MATS_DICT['GPSs']
HKSs = MATS_DICT['HKSs']
Hs = MATS_DICT['Hs']

D_DNA = cdist(DNAs, DNAs, 'euclidean')
D_GPS = cdist(GPSs, GPSs, 'euclidean')
D_HKS = cdist(HKSs, HKSs, 'euclidean')
D_H = cdist(Hs, Hs, 'euclidean')

def compute_MDS(D_mat, d):
    n = D_mat.shape[0]
    J = np.eye(n) - (1/n)*np.ones((n,n))
    MDS_mat = -(1/2) * J @ D_mat**2 @ J
    MDS_mat = MDS_mat.T @ MDS_mat
    MDS_eigVals, MDS_eigVecs = scipy.linalg.eigh(MDS_mat, subset_by_index=[n-d-1,n-d+1])
    MDS_eigVals = np.real(MDS_eigVals)
    MDS_eigVecs = np.real(MDS_eigVecs)
    p_reduced_MDS = MDS_eigVecs[:,:d]*np.expand_dims(MDS_eigVals[:d]**0.5,0)
    return p_reduced_MDS

subjects_id = (np.arange(100)/10).astype('int')
subjects_poses = np.arange(100)%10

D_MATS_DICT = {'D_DNA': D_DNA, 'D_GPS': D_GPS, 'D_HKS': D_HKS, 'D_H': D_H}
np.save(path_write + 'D_MATS_DICT_k=' + str(k) + '.npy', MATS_DICT)
## Load Distance Dict from file
# D_MATS_DICT = np.load(path_write + 'D_MATS_DICT_k=' + str(k), allow_pickle=True)

fig, axs = plt.subplots(2, 4, figsize=(15, 6))
for g, D_mat_name in enumerate(D_MATS_DICT.keys()):
    d = 2
    D_mat = D_MATS_DICT[D_mat_name]
    # embedding = MDS(n_components=2, dissimilarity='precomputed')
    # p_MDS = embedding.fit_transform(D_mat)
    p_MDS = smacof(D_mat, n_components=2)[0]
    # p_MDS = compute_MDS(D_mat, d)
    axs[0,g].scatter(p_MDS[:, 0], p_MDS[:, 1], c=subjects_id)
    axs[0,g].set_title('MDS on ' + D_mat_name + ' for subjects ID')
    axs[1,g].scatter(p_MDS[:, 0], p_MDS[:, 1], c=subjects_poses)
    axs[1,g].set_title('MDS on ' + D_mat_name + ' for subjects pose')
    
    
plt.show()