import numpy as np
import eikonalfm
import matplotlib.pyplot as plt

def eikonal_path_grad(tau, source, target, org_mat):

    path_points = np.array([target], ndmin=2)
    path_indices = np.array([target], ndmin=2)
    gy, gx = np.gradient(tau)
    while not (path_indices[-1, 0], path_indices[-1, 1]) == source:
        i = path_indices[-1, 0]
        j = path_indices[-1, 1]
        x = path_points[-1, 0]
        y = path_points[-1, 1]
        curr_point = np.array([x, y])
        grad_point = np.array([gy[i, j], gx[i, j]]).reshape([1, 2])
        next_point = curr_point - grad_point/(np.linalg.norm(grad_point))
        next_indices = next_point.astype('int').reshape([1, 2])

        path_indices = np.append(path_indices, next_indices, axis=0)
        path_points = np.append(path_points, next_point, axis=0)


    image_file_path = np.copy(org_mat)
    path_array = np.array(path_indices)
    plt.imshow(image_file_path,  cmap='jet')
    plt.scatter(path_array[:, 1], path_array[:, 0], s=3, c='green')
    plt.title('Negative Gradient - Shortest path from ' + str(source) + ' to ' + str(target))

def eikonal_path(tau, source, target):

    path = [target]
    while not path[-1] == source:
        i = path[-1][0]
        j = path[-1][1]
        T_north = tau[i - 1, j] if i > 0 else float('inf')
        T_south = tau[i + 1, j] if i < dimY - 1 else float('inf')
        T_east = tau[i, j - 1] if j > 0 else float('inf')
        T_west = tau[i, j + 1] if j < dimX - 1 else float('inf')
        tuples = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
        ind_min = np.argmin([T_north, T_south, T_east, T_west])
        next_node = tuples[ind_min]
        path.append(next_node)

    image_file_path = np.copy(image_file_original)
    path_array = np.array(path)
    plt.imshow(image_file_path)
    plt.scatter(path_array[:, 1], path_array[:, 0], s=3, c='red')
    plt.title('Negative Gradient - Shortest path from ' + str(source) + ' to ' + str(target))