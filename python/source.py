import enum
import numpy as np
import igl
import gpytoolbox as gpy
import robust_laplacian as rl

import sampler

class MassMode(enum.Enum):
    UNIFORM = 1
    WN = 2

class Source:
    def __init__(self, dim):
        self.dim = dim

    # Can use kwargs to support general mass processing, should probably have a base class method to do that?
    def sample(self, num_points, mass_mode):
        pass

    def generate_masses(self, mass_mode, points, **kwargs):
        num_points = points.shape[0]
        if mass_mode == MassMode.UNIFORM:
            masses = 1/num_points*np.ones(num_points)
        elif mass_mode == MassMode.WN:
            masses = self.generate_wn_weights(points, **kwargs)
        else:
            print(f'{mass_mode} not implemented yet!')
            masses = None
        return masses


class UniformCircle(Source):
    def __init__(self, r):
        super().__init__(2)
        self.r = r

    def sample(self, num_points, mass_mode):
        points, normals = sampler.uniform_circle_particles(num_points, self.r)
        masses = self.generate_masses(mass_mode, points)
        return points, normals, masses

    def generate_wn_weights(self, points, **kwargs):
        # Unused kwargs
        return sampler.uniform_circle_wn_weights(points.shape[0], self.r)


class MeshContour(Source):
    def __init__(self, mesh_file, scale=1):
        super().__init__(2)
        self.mesh_file = mesh_file
        self.scale = scale

    def sample(self, num_points, mass_mode):
        if self.mesh_file[-3:] == 'obj':
            V, F = igl.read_triangle_mesh(self.mesh_file)
            bidxs = igl.boundary_loop(F)
            V = V[bidxs, :2]
        elif self.mesh_file[-3:] == 'png':
            Vs = gpy.png2poly(self.mesh_file)
            V = np.concatenate(Vs, axis=0)
        else:
            print(f'format for {self.mesh_file} not implemented yet!')
            V = None

        # Rescale V to be centered at 0 and have dimension [-scale/2, scale/2] in largest dim
        Vmin = np.min(V, axis=0)
        Vmax = np.max(V, axis=0)
        center = (Vmax + Vmin)/2
        width = np.max(Vmax - Vmin)
        V = (V - center)/width * self.scale

        points, normals = sampler.uniform_sample_boundary_curve(num_points, V)
        if self.mesh_file[-3:] == 'png':
            normals = -normals # flipped in output due to opposite point order
        masses = self.generate_masses(mass_mode, points)

        return points, normals, masses

    def generate_wn_weights(self, points):
        # Unused kwargs
        return sampler.boundary_particle_weights(points)


class MeshVertices(Source):
    def __init__(self, mesh_file, scale=1):
        super().__init__(3)
        self.mesh_file = mesh_file
        self.scale = scale

    def sample(self, num_points, mass_mode):
        if self.mesh_file[-3:] == 'obj':
            V, F = igl.read_triangle_mesh(self.mesh_file)
        else:
            print(f'format for {self.mesh_file} not implemented yet!')
            V = None
            F = None

        # Rescale V to be centered at 0 and have dimension [-scale/2, scale/2] in largest dim
        Vmin = np.min(V, axis=0)
        Vmax = np.max(V, axis=0)
        center = (Vmax + Vmin)/2
        width = np.max(Vmax - Vmin)
        V = (V - center)/width * self.scale

        normals = igl.per_vertex_normals(V, F, 1)
        masses = self.generate_masses(mass_mode, V, F=F)

        return V, normals, masses

    def generate_wn_weights(self, points, **kwargs):
        F = kwargs['F']
        M = igl.massmatrix(points, F)
        masses = M.diagonal()
        return masses


class MeshSurface(Source):
    def __init__(self, mesh_file, scale=1):
        super().__init__(3)
        self.mesh_file = mesh_file
        self.scale = scale
        self.V = None
        self.F = None
        self.NF = None

    def sample(self, num_points, mass_mode):
        if self.V is None:
            if self.mesh_file[-3:] == 'obj':
                V, F = igl.read_triangle_mesh(self.mesh_file)
            else:
                print(f'format for {self.mesh_file} not implemented yet!')
                return None
            # Rescale V to be centered at 0 and have dimension [-scale/2, scale/2] in largest dim
            Vmin = np.min(V, axis=0)
            Vmax = np.max(V, axis=0)
            center = (Vmax + Vmin)/2
            width = np.max(Vmax - Vmin)
            V = (V - center)/width * self.scale
            print(f'effective scale: {self.scale/width}')
            NF = gpy.per_face_normals(V, F)
            self.V = V
            self.F = F
            self.NF = NF

        points, normals = sampler.sample_points_on_mesh(num_points, self.V, self.F, self.NF)
        masses = self.generate_masses(mass_mode, points)

        return points, normals, masses

    def generate_wn_weights(self, points, **kwargs):
        # Unused kwargs
        _, M = rl.point_cloud_laplacian(points)
        masses = M.diagonal()
        return masses


# Some example sources
source_dict = {
    # 2D
    'circle': UniformCircle(0.4),
    # 3D
    'bunny_verts': MeshVertices('../data/bunny.obj'),
    'bunny': MeshSurface('../data/bunny.obj'),
}
