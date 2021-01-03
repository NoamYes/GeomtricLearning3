import numpy as np
import scipy.sparse as sparse
import pyvista as pv
from matplotlib import cm


def read_off(path):
    # The function receives a path of a file, and given an .off file, it returns a tuple of two ndarrays,
    # one of the vertices, and one of the faces of the given mesh.

    with open(path, 'r') as file:
        f_lines = file.readlines()
        if f_lines[0] != 'OFF\n':
            print('Invalid OFF file - missing keyword "OFF"!')
            return
        [vertices_num, faces_num, _] = [int(n) for n in (f_lines[1].split())]
        v = [[float(part) for part in (line.split())] for line in f_lines[2:2+vertices_num]]
        f = [[int(part) for part in (line.split())] for line in f_lines[2+vertices_num:2+vertices_num+faces_num]]
        v = np.array(v, dtype=float)
        f = np.array(f, dtype=int)

    return v, f


def write_off(path, v, f):
    # The function receives a path to a file (creates one, if such does not exist), an ndarray of vertices ,
    # and an ndarray of faces, and writes the file into a proper .off file of the given mesh defined by the two arrays.

    with open(path, "w") as file:
        lines = ['OFF\n']
        vertices_num = len(v)
        faces_num = len(f)
        lines.append(" ".join([str(elem) for elem in [vertices_num, faces_num, 0]]) + '\n')
        v_lines = [" ".join([str(part) for part in vertex]) + '\n' for vertex in v]
        f_lines = [" ".join([str(part) for part in face])+'\n' for face in f]
        lines.extend(v_lines)
        lines.extend(f_lines)
        file.writelines(lines)


class Mesh:

    def __init__(self, v, f): #off_path
        # (self.v, self.f) = read_off(off_path)  # Basic definition of the mesh. Read off the given .off file
        (self.v, self.f) = (v, f.astype('int'))
        self.vf_adj_mat = self.vertex_face_adjacency()  # Define the Vertex-Face adjacency matrix for the mesh
        self.vv_adj_mat = self.vertex_vertex_adjacency()  # Define the Vertex-Vertex adjacency matrix for the mesh
        self.v_deg_vec = self.vertex_degree()  # Define the Vertex degree vector for the mesh.
        self.fv_map = self.face_vertex_map()  # Define the Face-Vertex map - being a form of self.f with explicit vertex values (used for calculations)
        self.fn_map = self.face_normals()  #
        self.fbc_map = self.face_barycenters()
        self.fa_map = self.face_areas()
        self.va_map = self.barycentric_vertex_areas()
        self.vn_map = self.vertex_normals()
        # self.gc_map = self.gaussian_curvature()

    def vertex_face_adjacency(self):
        # The function returns the Vertex-Face adjacency matrix of the mesh, as a boolean sparse matrix of size
        # (vertices x faces). A cell (vertex,face) in the matrix gets the value 'True' if the vertex (row), and face
        # (column) are adjacent.

        vertices = np.arange(len(self.v))
        vf_adj_mat = [np.isin(vertices, f) for f in self.f[:, 1:]]
        vf_adj_mat = sparse.csc_matrix(vf_adj_mat).transpose()
        return vf_adj_mat

    def vertex_vertex_adjacency(self):
        # The function returns the Vertex-Vertex adjacency matrix of the mesh, as a boolean sparse matrix of size
        # (vertices x Vertices). A cell (vertex1,Vertex2) in the matrix gets the value 'True' if the vertex1 (row),
        # and face (column1) are adjacent.

        vv_adj_mat = np.dot(self.vf_adj_mat, self.vf_adj_mat.transpose()).astype(bool)
        vv_adj_mat.setdiag(False)
        return vv_adj_mat

    def vertex_degree(self):
        # The function returns the Vertex degree vector of the mesh, as an ndarray of size (vertices).
        # A cell (vertex) gets the number of adjacent vertices the vertex has (excluding itself, of course).

        vv_adj_mat = self.vv_adj_mat.toarray()
        return sum(vv_adj_mat)

    def face_vertex_map(self):
        # The function returns the Face-Vertex map of the mesh, as an ndarray of size (faces,3,3)
        # being a form of self.f with explicit vertex values, instead of vertex indexes (used for easier calculations).

        fv_map = np.array([[self.v[vertex].tolist() for vertex in face] for face in self.f[:, 1:]])
        return fv_map

    def render_wireframe(self, plotter=None):
        # The function renders the mesh in a wireform view.

        # Added optional parameter, plotter: gets pv.Plotter from outer scope to present this rendering as part of
        # a subplot, in which case, the plotter.show() function should be called on the outer scope. The default value
        # plotter==None, is for the case that the rendering is a standalone, and the function will plot it from within
        # it.

        standalone = 0
        if plotter == None:
            plotter = pv.Plotter()
            standalone = 1
        wifeframe = pv.PolyData(self.v, self.f)
        plotter.add_mesh(wifeframe, style='wireframe')
        if standalone == 1:
            plotter.show()

    def render_pointcloud(self, scalar_func, cmap_name="viridis", plotter=None):
        # The function renders the mesh in a pointcloud view.

        # scalar_func should be ndarray of shape [size(self.v),1]

        # Added optional parameter, cmap_name: gets one of the possible "matplotlib.cm.get_cmap()"
        # string values to define the colormap

        # Added optional parameter, plotter: gets pv.Plotter from outer scope to present this rendering as part of
        # a subplot, in which case, the plotter.show() function should be called on the outer scope. The default value
        # plotter=None, is for the case that the rendering is a standalone, and the function will plot it from within
        # it.

        standalone = 0
        if plotter == None:
            plotter = pv.Plotter()
            standalone = 1
        pointcloud = pv.PolyData(self.v)
        plotter.add_mesh(pointcloud, render_points_as_spheres=True, scalars=scalar_func, cmap=cm.get_cmap(cmap_name))
        if standalone == 1:
            plotter.show()

    def render_surface(self, scalar_func, cmap_name="viridis", plotter=None):
        # The function renders the mesh in a surface view.

        # scalar_func should be ndarray of shape [size(self.v),1] to color the vertices,
        # of shape [size(self.f),1] to color the faces.

        # Added optional parameter, cmap_name: gets one of the possible "matplotlib.cm.get_cmap()"
        # string values to define the colormap

        # Added optional parameter, plotter: gets pv.Plotter from outer scope to present this rendering as part of
        # a subplot, in which case, the plotter.show() function should be called on the outer scope. The default value
        # plotter==None, is for the case that the rendering is a standalone, and the function will plot it from within
        # it.
        standalone = 0
        pointcloud = pv.PolyData(self.v)
        if plotter == None:
            plotter = pv.Plotter()
            standalone = 1
        surf = pv.PolyData(self.v, self.f)
        plotter.add_mesh(surf, scalars=scalar_func, show_edges=True, cmap=cm.get_cmap(cmap_name))
        if standalone == 1:
            plotter.show()

    def face_normals(self, normalized=True):
        # Compute the face normals matrix for the mesh. If normalized = True (default), also normalize the normals.

        fn_map = np.array([np.cross(face[1]-face[0],face[2]-face[1]).tolist() for face in self.fv_map])
        if normalized:
            fn_map = np.array([fn/np.linalg.norm(fn).tolist() for fn in fn_map])
        return fn_map

    def face_barycenters(self):
        # Compute the face berycenters matrix for the mesh.

        fbc_map = np.array([np.mean(face, axis=0).tolist() for face in self.fv_map])
        return fbc_map

    def face_areas(self):
        # Compute the face areas vector for the mesh.

        fn_map = self.face_normals(normalized=False)
        fa_map = 0.5 * np.array([np.linalg.norm(fn).tolist() for fn in fn_map])
        return fa_map

    def barycentric_vertex_areas(self):
        # Compute the barycentric vertex areas vector for the mesh.

        vf_adj_mat = self.vf_adj_mat
        fa_map = self.fa_map
        va_map = (1/3)*np.dot(vf_adj_mat.toarray(), fa_map)
        return va_map

    def vertex_normals(self, normalized=True):
        # Compute the vertex normals matrix for the mesh. If normalized = True (default), also normalize the normals.

        vf_adj_mat = self.vf_adj_mat
        fn_map = self.face_normals(normalized=False)
        vn_map = np.dot(vf_adj_mat.toarray(), fn_map)
        if normalized:
            vn_map = np.array([vn/(np.linalg.norm(vn)+1e-6).tolist() for vn in vn_map])
        return vn_map

    def gaussian_curvature(self):
        # Compute the gaussian curvature vector for the mesh at each vertex.

        gc_map = np.zeros((self.v.shape[0],))
        for v in range(self.v.shape[0]):
            adj_faces = self.f[self.vf_adj_mat.toarray()[v]][:, 1:]
            adj_vertices = adj_faces[np.where(adj_faces != v)]
            adj_vertices = adj_vertices.reshape(int(adj_vertices.shape[0] / 2), 2)
            adj_edges = self.v[v] - self.v[adj_vertices]
            adj_angles = np.zeros(adj_vertices.shape[0], )
            for f in range(adj_edges.shape[0]):
                edge0 = adj_edges[f, 0]
                edge1 = adj_edges[f, 1]
                adj_angles[f] = np.arccos(np.dot(edge0/np.linalg.norm(edge0), edge1/np.linalg.norm(edge1)))
            gc_map[v] = (2*np.pi - sum(adj_angles)) / self.va_map[v]
        return gc_map

    def visualize_normals(self, plotter, normalized=True, v_flag=True, f_flag=True, mag=0.2):
        # Visualize the normals of the mesh via arrows.
        # If normalized = True (default), also normalize the normals.
        # If v_flag=True (default), show the vertex normals.
        # If f_flag=True (default), show the face normals.
        # You are able to tune the magnitude of the arrows via 'mag'.

        # Added parameter, plotter: gets pv.Plotter from outer scope.
        # The plotter.show() function should be called on the outer scope.

        if v_flag:
            v_centers = self.v
            v_normals = self.vertex_normals(normalized)
            plotter.add_arrows(v_centers, v_normals, mag=mag)
        if f_flag:
            f_centers = self.fbc_map
            f_normals = self.face_normals(normalized)
            plotter.add_arrows(f_centers, f_normals, mag=mag)

    def visualize_centroid(self, cmap_name="viridis", plotter=None):
        # The function renders the mesh in a pointcloud view, and presents the mesh centroid. The scalar function used
        # is the Euclidean distance from said centroid.

        # Added optional parameter, cmap_name: gets one of the possible "matplotlib.cm.get_cmap()"
        # string values to define the colormap

        # Added optional parameter, plotter: gets pv.Plotter from outer scope to present this rendering as part of
        # a subplot, in which case, the plotter.show() function should be called on the outer scope. The default value
        # plotter=None, is for the case that the rendering is a standalone, and the function will plot it from within
        # it.

        centroid = sum(self.v)/len(self.v)
        centroid_distance = np.array([(np.linalg.norm(centroid-v)).tolist() for v in M.v])

        standalone = 0
        if plotter == None:
            plotter = pv.Plotter()
            standalone = 1
        pointcloud = pv.PolyData(self.v)
        plotter.add_mesh(pointcloud, render_points_as_spheres=True, scalars=centroid_distance, cmap=cm.get_cmap(cmap_name))
        centroid_mesh = pv.Sphere(radius=max(centroid_distance)/10, center=centroid)
        plotter.add_mesh(centroid_mesh, color='black')
        if standalone == 1:
            plotter.show()


if __name__ == '__main__':

    # Section 4:
    # M = Mesh('example_off_files/sphere_s0.off')
    # M = Mesh('example_off_files/disk.off')
    # M = Mesh('example_off_files/teddy171.off')
    # plotter = pv.Plotter(shape=(2, 2))
    # plotter.subplot(0, 0)
    # M.render_wireframe(plotter=plotter)
    # plotter.subplot(0, 1)
    # M.render_pointcloud(scalar_func=M.v, cmap_name="viridis", plotter=plotter)
    # plotter.subplot(1, 0)
    # M.render_surface(scalar_func=np.random.random([len(M.f), 1]), cmap_name="Blues", plotter=plotter)
    # plotter.subplot(1, 1)
    # M.render_surface(scalar_func=np.random.random([len(M.v), 1]), cmap_name="Blues", plotter=plotter)
    # plotter.show()

    # Section 6b
    # M = Mesh('example_off_files/sphere_s0.off')
    # M = Mesh('example_off_files/disk.off')
    # M = Mesh('example_off_files/teddy171.off')
    # plotter = pv.Plotter(shape=(1, 2))
    # plotter.subplot(0, 0)
    # M.render_surface(scalar_func=M.va_map, cmap_name="Blues", plotter=plotter)
    # plotter.subplot(0, 1)
    # M.render_surface(scalar_func=M.fa_map, cmap_name="Blues", plotter=plotter)
    # plotter.show()

    # Section 6c
    # M = Mesh('example_off_files/sphere_s0.off')
    # M = Mesh('example_off_files/disk.off')
    M = Mesh('example_off_files/teddy171.off')
    plotter = pv.Plotter(shape=(2, 2))
    plotter.subplot(0, 0)
    M.render_surface(scalar_func=M.vn_map, cmap_name="Blues", plotter=plotter)
    plotter.subplot(0, 1)
    M.render_surface(scalar_func=M.fn_map, cmap_name="Blues", plotter=plotter)
    plotter.subplot(1, 0)
    M.render_surface(scalar_func=M.vertex_normals(False), cmap_name="Blues", plotter=plotter)
    M.visualize_normals(normalized=False, plotter=plotter, f_flag=False, mag=30)
    plotter.subplot(1, 1)
    M.render_surface(scalar_func=M.face_normals(False), cmap_name="Blues", plotter=plotter)
    M.visualize_normals(normalized=False, plotter=plotter,  v_flag=False, mag=30)
    plotter.show()

    # Section 6d
    # M = Mesh('example_off_files/sphere_s0.off')
    # M = Mesh('example_off_files/disk.off')
    # M = Mesh('example_off_files/teddy171.off')
    # M.visualize_centroid(cmap_name="jet")



