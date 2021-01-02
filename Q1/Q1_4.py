import meshio
import matplotlib.pyplot as plt
import numpy as np
from Q1.MDS import Locally_Linear_Embedding
from utils.mesh_tools import Mesh
import gdist


# Import man1, man2

man1_ply = meshio.read("../HW3_Resources/man1.ply")
man2_ply = meshio.read("../HW3_Resources/man2.ply")

# Plot original mesh

man1_vertices = man1_ply.points
man1_faces = man1_ply.cells_dict['triangle']
man1_mesh = Mesh(man1_vertices, np.c_[3*np.ones(np.shape(man1_faces)[0]), man1_faces])
plt.figure(1)
man1_mesh.render_surface(man1_vertices, cmap_name='winter')
# plt.show()

man1_gdist = gdist.local_gdist_matrix(man1_vertices.astype('float'), man1_faces.astype('int'))

#  Plot man1 LLE with various n_neighbors
neighbors_list = [3, 7, 10, 25, 100]
# fig2 = plt.figure(figsize=(15, 6))
for i, n_neighbor in enumerate(neighbors_list):
    man1_reduced_LLE = Locally_Linear_Embedding(man1_vertices, n_neighbor=n_neighbor, epsilon=1, d=3)
    man1_mesh = Mesh(man1_reduced_LLE, np.c_[3 * np.ones(np.shape(man1_faces)[0]), man1_faces])
    # plt.figure(2+i)
    man1_mesh.render_surface(man1_vertices, cmap_name='winter')
    # axs = fig2.add_subplot(1, len(neighbors_list), i+1, projection='3d')
    # scat = axs.scatter(man1_reduced_LLE[:, 0], man1_reduced_LLE[:, 1], man1_reduced_LLE[:, 2], s=5)
    # axs.set_title('n_neighbors=' + str(n_neighbor))

# fig2.suptitle('man1 Local Linear Embedding (LLE) MAP for multiple n_neighbors')
plt.show()

print('ya')