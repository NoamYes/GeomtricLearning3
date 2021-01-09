import os
import numpy as np
from utils.mesh_tools import Mesh

faust_dir = 'MPI-FAUST/training/registrations'

subjectIDs = []
poses = []

D_DNA, D_GPS, D_HKS, D_H = [], [], [], []

for filename in os.listdir(faust_dir):
    if filename.endswith(".ply"): 
        subjectID = int(filename[-6])
        pose = int(filename[-5])
        path_read = faust_dir + filename
        obj_mesh = Mesh('ply', path_read)

        k = np.shape(obj_mesh.v)[0]
        eigVals, eigVecs = obj_mesh.laplacian_spectrum(k, 'uniform')
        
        # Calculate the shape DNA
        obj_DNA = eigVals
        D_DNA.append(obj_DNA)

        # Calculate the GPS
        inv_sqrt_eigVals = np.diag(1/np.sqrt(eigVals))
        print('ya')
        
