from typing import List, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import open3d as o3d
from open3d.visualization import draw_geometries

from pycloud2room import GeoPointCloud
import pycloud2room.voxio as voxio


class GeoVoxelGrid:

    def __init__(self, occupied: np.ndarray, voxel_size: float = 1,
                 origin: Union[List[float], np.ndarray] = [0, 0, 0],
                 data: Optional[pd.DataFrame] = None, rgb_max=1.0):
        if origin is None:
            origin = [0, 0, 0]
        self.occupied = occupied
        self.voxel_size = voxel_size
        self.origin = np.array(origin)
        self.data = data
        self.rgb_max = rgb_max

    @classmethod
    def from_vox_file(cls, path) -> 'GeoVoxelGrid':
        return voxio.read_vox_file(path)

    @classmethod
    def from_geopointcloud(cls, cloud: GeoPointCloud, voxel_size: float) -> 'GeoVoxelGrid':
        cloud = cloud.thin_by_grid(voxel_size, inplace=False)
        xyz_min = cloud.xyz.min().to_numpy()
        xyz_max = cloud.xyz.max().to_numpy()
        dimensions = ((xyz_max - xyz_min) // voxel_size).astype(int) + 1
        xyz = cloud.xyz.to_numpy() - xyz_min + voxel_size / 2
        rgb = cloud.rgb
        indices = (xyz // voxel_size).astype(int)
        occupied = np.zeros(dimensions, dtype=bool)
        occupied[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
        md = pd.MultiIndex.from_arrays(indices.transpose(), names=['x', 'y', 'z'])
        data = pd.DataFrame(rgb.to_numpy(), index=md, columns=['r', 'g', 'b'])
        return GeoVoxelGrid(occupied=occupied, voxel_size=voxel_size, origin=xyz_min, data=data,
                            rgb_max=cloud.rgb_max)

    @classmethod
    def from_indices(cls, indices: np.ndarray, voxel_size: float = 1, origin: List[float] = [0, 0, 0],
                          rgb: Optional[np.ndarray] = None) -> 'GeoVoxelGrid':
        dimensions = indices.max(axis=0) + 1
        occupied = np.zeros(dimensions, dtype=bool)
        occupied[indices[0, :], indices[1, :], indices[2, :]] = 1
        rgb_max = 1.0
        data = None
        if rgb is not None:
            md = pd.MultiIndex.from_arrays(indices.transpose(), names=['x', 'y', 'z'])
            data = pd.DataFrame(rgb, index=md, columns=['r', 'g', 'b'])
            rgb_max = np.iinfo(rgb.dtype).max
        return GeoVoxelGrid(occupied=occupied, voxel_size=voxel_size, origin=origin, data=data,
                            rgb_max=rgb_max)

    @classmethod
    def from_o3d(cls, o3d_voxel_grid: o3d.geometry.VoxelGrid) -> 'GeoVoxelGrid':
        origin = o3d_voxel_grid.origin
        voxels = o3d_voxel_grid.get_voxels()
        indices = np.asarray([pt.grid_index for pt in voxels]).transpose()
        rgb = np.asarray([pt.color for pt in voxels])
        return GeoVoxelGrid.from_indices(indices, o3d_voxel_grid.voxel_size, origin, rgb)

    def has_rgb(self) -> bool:
        return self.data is not None and 'r' in self.data and 'g' in self.data and 'b' in self.data

    def indices(self) -> np.ndarray:
        return np.stack(np.where(self.occupied == 1), axis=0)

    def voxel_coordinates(self) -> np.ndarray:
        return (self.indices() * self.voxel_size + self.origin[:, np.newaxis]).transpose()

    def to_o3d_pointcloud(self) -> o3d.geometry.PointCloud:
        xyz_o3d = o3d.utility.Vector3dVector(self.voxel_coordinates())
        pc3d = o3d.geometry.PointCloud()
        pc3d.points = xyz_o3d
        if self.has_rgb():
            rgb = self.data[['r', 'g', 'b']].to_numpy().astype(float) / self.rgb_max
            rgb_o3d = o3d.utility.Vector3dVector(rgb)
            pc3d.colors = rgb_o3d
        return pc3d

    def to_geopointcloud(self) -> GeoPointCloud:
        return GeoPointCloud.from_o3d(self.to_o3d_pointcloud())

    def to_o3d(self):
        return o3d.geometry.VoxelGrid.create_from_point_cloud(self.to_o3d_pointcloud(), voxel_size=self.voxel_size)

    def plot_o3d(self):
        draw_geometries([self.to_o3d()])

    def to_vox_file(self, path: Union[str, Path], vox_color: int = 10):
        voxio.write_vox_file(self, path, vox_color)
