import numpy as np
import eikonalfm
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from Q1.shortest_paths import eikonal_path_grad
from Q1.shortest_paths import networkx_path

# Prepare the image
thresh = 180
image_file_original = Image.open("../HW3_Resources/maze.png")  # open colour image
image_file = image_file_original.convert("L")
maze_img = np.asarray(image_file).astype('int')

plt.imshow(maze_img, cmap='gray', vmin=0, vmax=255)
# plt.show()

dimY = np.shape(maze_img)[0]
dimX = np.shape(maze_img)[1]

# Section 1.1.a
mask = np.copy(maze_img)
mask[maze_img <= thresh] = 0
mask[maze_img > thresh] = 1
c = 1 + 1e6*mask/255
dx = (1, 1)
order = 2
x_s = (383, 814)
tau_fm = eikonalfm.fast_marching(c, x_s, dx, order)


visualize_tau = np.copy(tau_fm)
visualize_tau[mask == 0] = -0.2
visualize_tau[mask == 1] = visualize_tau[mask == 1]/np.max(visualize_tau)
plt.figure(1)
plt.title('Eikonal FMM - Maze distance image with source = ' + str(x_s))
plt.imshow(visualize_tau, cmap='jet')
# plt.show()


# Section 1.1.b

x_t = (233, 8)
plt.figure(2)
eikonal_path_grad(tau_fm, x_s, x_t, visualize_tau)
# plt.show()
plt.figure(3)
tau_fm = eikonalfm.fast_marching(c, x_t, dx, order)
eikonal_path_grad(tau_fm, x_t, x_s, visualize_tau)
# plt.show()

# Section 1.1.c
# Create nodes and edges for maze graph
nodes = np.arange(dimX*dimY)
edges = []
for i in range(dimY):
    for j in range(dimX):
        isWall = mask[i, j] == 0
        isWallSouth = mask[i+1, j] == 0 if i < dimY-1 else True
        isWallEast = mask[i, j+1] == 0 if j < dimX-1 else True
        if not isWall:
            if i < dimY - 1 and not isWallSouth:
                edge_south = (i*dimX+j, (i+1)*dimX+j)
                edges.append(edge_south)
            if j < dimX - 1 and not isWallEast:
                edge_east = (i*dimX+j, i*dimX+j+1)
                edges.append(edge_east)

# Create maze graph
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

source_node = x_s[0] * dimX + x_s[1]
target_node = x_t[0] * dimX + x_t[1]

shortest_path = nx.shortest_paths.weighted.dijkstra_path(G, source_node, target_node)
plt.figure(4)
networkx_path(shortest_path, x_s, x_t, visualize_tau)
# plt.show()

shortest_path = nx.shortest_paths.weighted.dijkstra_path(G, target_node, source_node)
plt.figure(5)
networkx_path(shortest_path, x_t, x_s, visualize_tau)
plt.show()

print('ya')
