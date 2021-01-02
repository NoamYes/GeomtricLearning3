import numpy as np
import eikonalfm
import matplotlib.pyplot as plt
from scipy.io import loadmat
from Q1.shortest_paths import eikonal_path_grad

# Prepare the image
thresh = 180
pool_mat = loadmat('../HW3_Resources/pool.mat')
ref_mat = pool_mat['n']
plt.figure(1)
plt.imshow(ref_mat, cmap='jet')
# plt.show()

(dimY, dimX) = np.shape(ref_mat)
x_s = (0, 0)
x_t = (499, 399)
dx = (1, 1)
order = 2

tau_fm = eikonalfm.fast_marching(1/ref_mat**4, x_s, dx, order)
plt.figure(2)
plt.imshow(tau_fm, cmap='jet')
# plt.show()
plt.figure(3)
eikonal_path_grad(tau_fm, x_s, x_t, ref_mat)
plt.show()
print('ya')