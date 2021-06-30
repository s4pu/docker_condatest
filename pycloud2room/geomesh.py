from typing import Type, TypeVar, Tuple, Sequence

#import kaolin as kal
#import torch
import numpy as np
import open3d as o3d
from . import open3d_tutorial as o3dtut
import trimesh

T = TypeVar('T', bound='TrivialClass')

class Geo3dMeshData:
    def __init__(self):
        self.vertices = None
        self.faces = None


class GeoMesh:

    @classmethod
    def from_kalmesh(cls: Type[T], kalmesh) -> T:
        mesh = GeoMesh(trimesh.Trimesh(kalmesh.vertices.numpy(), kalmesh.faces.numpy()))
        mesh.cached_kalmesh = kalmesh
        return mesh

    @classmethod
    def from_o3d_mesh(cls: Type[T], o3dmesh) -> T:
        tmesh = trimesh.Trimesh(vertices=o3dmesh.vertices, faces=o3dmesh.triangles)
        mesh = cls(tmesh)
        return mesh

    @classmethod
    def from_trimesh(cls: Type[T], trimesh: trimesh.Trimesh) -> T:
        return cls(trimesh)

    @classmethod
    def from_vertices_faces(cls: Type[T], vertices: np.ndarray, faces: np.ndarray):
        tmesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return cls(tmesh)

    @classmethod
    def from_others(cls: Type[T], others: Sequence['GeoMesh']) -> T:
        vertices = []
        faces = []
        count = 0
        for other in others:
            vertices.append(other.vertices)
            faces.append(other.faces + count)
            count += other.vertices.shape[0]
        vertices = np.concatenate(vertices, axis=0)
        faces = np.concatenate(faces, axis=0)
        return cls.from_vertices_faces(vertices, faces)

    def __init__(self, trimesh):
        self.tmesh = trimesh

        self.cached_kalmesh = None

    def __getstate__(self):
        data = Geo3dMeshData()
        data.vertices = self.vertices()
        data.faces = self.faces()
        return data

    def __setstate__(self, state : Geo3dMeshData):
        tmesh = trimesh.Trimesh(vertices=state.vertices, faces=state.faces)
        self.__init__(tmesh)

    def center_and_scale(self, center: Tuple[float, float, float] = (0,0,0), max_bbx_side: float = 1):
        self.tmesh = self.tmesh.apply_transform(
            trimesh.transformations.scale_matrix(max_bbx_side / self.tmesh.extents.max()))
        center = np.array(center)
        shift_mat = trimesh.transformations.scale_and_translate(
            scale=1, translate=center - 1/2 * self.tmesh.bounds.sum(axis=0))
        self.tmesh = self.tmesh.apply_transform(shift_mat)
        self._invalidate_caches()

    def to_trimesh(self):
        return self.tmesh

    def to_o3d(self):
        vertices = o3d.utility.Vector3dVector(self.vertices)
        triangles = o3d.utility.Vector3iVector(self.faces)
        return o3d.geometry.TriangleMesh(vertices, triangles)

    # def as_kmesh(self):
    #    if not self.cached_kalmesh:
    #        self.cached_kalmesh = kal.rep.TriangleMesh(vertices=torch.from_numpy(self.vertices()),
    #                                                   faces=torch.from_numpy(self.faces()))
    #    return self.cached_kalmesh

    def plot_o3d(self, *args, **kwargs):
        mesh = self.to_o3d()
        mesh.compute_triangle_normals()
        o3d.visualization.draw_geometries([mesh], *args, **kwargs)

    @property
    def vertices(self):
        return self.tmesh.vertices

    @vertices.setter
    def vertices(self, vertices: np.ndarray):
        assert vertices.shape == self.vertices.shape
        self.tmesh.vertices = vertices

    @property
    def faces(self):
        return self.tmesh.faces

    @property
    def vertex_normals(self):
        return self.tmesh.vertex_normals

    def _invalidate_caches(self):
        self.cached_kalmesh = None

    @classmethod
    def get_armadillo_mesh(cls: Type[T]) -> T:
        o3dmesh = o3dtut.get_armadillo_mesh()
        return cls.from_o3d_mesh(o3dmesh)

    @classmethod
    def get_box_mesh(cls: Type[T]) -> T:
        o3dmesh = o3d.geometry.TriangleMesh.create_box()
        return cls.from_o3d_mesh(o3dmesh)

    @classmethod
    def get_open_box_mesh(cls: Type[T]) -> T:
        o3dmesh = o3dtut.get_open_box_mesh()
        return cls.from_o3d_mesh(o3dmesh)

    @classmethod
    def get_intersecting_boxes_mesh(cls: Type[T]) -> T:
        o3dmesh = o3dtut.get_intersecting_boxes_mesh()
        return cls.from_o3d_mesh(o3dmesh)

    @classmethod
    def get_bunny_mesh(cls: Type[T]) -> T:
        o3dmesh = o3dtut.get_bunny_mesh()
        return cls.from_o3d_mesh(o3dmesh)

    @classmethod
    def get_knot_mesh(cls: Type[T]) -> T:
        o3dmesh = o3dtut.get_knot_mesh()
        return cls.from_o3d_mesh(o3dmesh)

