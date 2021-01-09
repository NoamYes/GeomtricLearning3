import numpy as np
import pyvista as pv
from matplotlib import cm
import meshio
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

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

    def __init__(self, loadType, *args): #off_path
        if loadType == 'off':
            off_path = args[0]
            (self.v, self.f) = read_off(off_path)  # Basic definition of the mesh. Read off the given .off file
        elif loadType == 'ply':
            ply_path = args[0]
            ply_obj = meshio.read(ply_path)  # Read from ply via Meshio
            self.v = ply_obj.points.astype(np.float)
            f = ply_obj.cells_dict['triangle']
            self.f = np.c_[3*np.ones(np.shape(f)[0]), f].astype(np.int32)
        elif loadType == 'vf':
            self.v, self.f = args[0].astype(np.float), args[1].astype(np.int32)
        else:
            raise(NameError('Wrong loadType'))

        self.Laps = {}

        self.vf_adj_mat = self.vertex_face_adjacency()  # Define the Vertex-Face adjacency matrix for the mesh
        self.vv_adj_mat = self.vertex_vertex_adjacency()  # Define the Vertex-Vertex adjacency matrix for the mesh
        self.v_deg_vec = self.vertex_degree()  # Define the Vertex degree vector for the mesh.
        self.fv_map = self.face_vertex_map()  # Define the Face-Vertex map - being a form of self.f with explicit vertex values (used for calculations)
        self.fn_map = self.face_normals()  #
        self.fbc_map = self.face_barycenters()
        self.fa_map = self.face_areas()
        self.va_map = self.barycentric_vertex_areas()
        self.vn_map = self.vertex_normals()
        # self.w_adj = self.weighted_adjacency(cls='half_cotangent')
        # self.lap = self.laplacian(cls='half_cotangent')
        # self.bc_v_mass_mat = self.barycenter_vertex_mass_matrix()

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

    def weighted_adjacency(self, cls='half_cotangent'):
        vv_adj_mat = self.vv_adj_mat
        if cls == 'half_cotangent':
            cotangent_mat = np.zeros(np.shape(vv_adj_mat))
            adj_ones = sparse.find(vv_adj_mat)[0:2]
            for i, j in zip(*adj_ones):
                if j > i:
                    vi_faces = np.squeeze(self.vf_adj_mat[i].toarray())
                    vj_faces = np.squeeze(self.vf_adj_mat[j].toarray())
                    comm_faces = np.where(np.logical_and(vi_faces, vj_faces))[0]
                    sum_ang = 0
                    for f_ind in comm_faces:
                        k_idx = np.where(np.logical_not(np.logical_or(self.f[f_ind,1:] == i, self.f[f_ind,1:]== j)))[0]
                        k = int(self.f[f_ind, k_idx+1])
                        edge0 = self.v[i]-self.v[k]
                        edge1 = self.v[j]-self.v[k]
                        angle = np.arccos(np.dot(edge0/np.linalg.norm(edge0), edge1/np.linalg.norm(edge1)))
                        sum_ang = sum_ang + 1/np.tan(angle)
                    A_i = self.va_map[i]
                    # cotangent_mat[i,j] = (1/(2*A_i))*sum_ang
                    cotangent_mat[i,j] = sum_ang
            W = cotangent_mat + cotangent_mat.T
        else:
            W = vv_adj_mat.astype(np.float64)
        return sparse.csc_matrix(W)

    def laplacian(self, cls='half_cotangent'):
        self.w_adj = self.weighted_adjacency(cls=cls)
        w_deg_mat = np.diag(np.squeeze(np.asarray(self.w_adj.sum(axis=0))))
        self.w_deg_mat = sparse.csc_matrix(w_deg_mat)
        self.Laps[cls] = self.w_deg_mat - self.w_adj
        return self.Laps[cls]

    def barycenter_vertex_mass_matrix(self):
        M = np.diag(self.va_map)
        return sparse.csc_matrix(M)

    def laplacian_spectrum(self, k, cls):
        L = self.laplacian(cls=cls)
        M = self.barycenter_vertex_mass_matrix()
        eig_val, eig_vec = linalg.eigsh(L, k, M, which='LM', sigma=0, tol=1e-7)
        idx_pn = eig_val.argsort()[::1] 
        eig_val = eig_val[idx_pn]  # Rounds up to 9 digits?
        eig_vec = eig_vec[:, idx_pn]

        eig_val = np.round(eig_val, decimals=12)
        eig_vec = np.round(eig_vec, decimals=12)
        return eig_val, eig_vec
    
    def mean_curvature(self, cls='half_cotangent'):
        M = self.barycenter_vertex_mass_matrix()
        L = self.laplacian(cls=cls)
        M_inv = linalg.inv(M)
        H_n = M_inv @ L @ self.v
        vn_map = self.vertex_normals(normalized=False)
        H_abs = np.linalg.norm(H_n, axis=1) / np.linalg.norm(vn_map, axis=1)
        MUL = vn_map @ H_n.T
        H = H_abs * np.sign(MUL.diagonal())
        return H
 
