import numpy as np
import eikonalfm
import matplotlib.pyplot as plt
from PIL import Image

thresh = 180  # TODO What value should it be?
image_file_original = Image.open("../HW3_Resources/maze.png")  # open colour image
image_file = image_file_original.convert("L")
maze_img = np.asarray(image_file).astype('int')
# maze_img = maze_img >= thresh
# maze_img = maze_img.astype('int')

# Remove all 1 lines and columns from maze
maze_img_cpy = np.copy(maze_img)
# maze_img[np.where(np.all(maze_img_cpy > thresh, axis=1)), :] = 0
# maze_img[:, np.where(np.all(maze_img_cpy > thresh, axis=0))] = 0

plt.imshow(maze_img, cmap='gray', vmin=0, vmax=255)
# plt.show()

maze_img[maze_img < thresh] = 0
maze_img[maze_img > thresh] = 1

c = 1 + 1e6 * maze_img
# c = np.ones(np.shape(maze_img))*255
dx = (1, 1)
order = 2


def eikonal_path(maze, x_s, x_t):

    tau_fm = eikonalfm.fast_marching(c, x_s, dx, order)

    # plt.contourf(tau_fm)
    # plt.show()

    dimY = np.shape(maze)[0]
    dimX = np.shape(maze)[1]

    path = [x_t]
    while not path[-1] == x_s:
        i = path[-1][0]
        j = path[-1][1]
        T_north = tau_fm[i - 1, j] if i > 0 else float('inf')
        T_south = tau_fm[i + 1, j] if i < dimY - 1 else float('inf')
        T_east = tau_fm[i, j - 1] if j > 0 else float('inf')
        T_west = tau_fm[i, j + 1] if j < dimX - 1 else float('inf')
        tuples = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
        ind_min = np.argmin([T_north, T_south, T_east, T_west])
        next_node = tuples[ind_min]
        path.append(next_node)

    image_file_path = np.copy(image_file_original)
    path_array = np.array(path)
    plt.imshow(image_file_path)
    plt.scatter(path_array[:, 1], path_array[:, 0], s=3, c='red')
    plt.title('Shortest path from ' + str(x_s) + ' to ' + str(x_t))



x_s = (383, 814)
x_t = (233, 8)
plt.figure(1)
eikonal_path(maze_img,x_s,x_t)
# plt.show()
plt.figure(2)
eikonal_path(maze_img,x_t,x_s)
plt.show()

print('ya')
