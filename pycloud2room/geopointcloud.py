import json
from typing import Union, Tuple, List, Optional
from pathlib import Path
from xml.dom import minidom

import numpy as np
import pandas as pd
import pylas
import pdal
#import pptk
import open3d as o3d
from sklearn.decomposition import PCA


class GeoPointCloud:

    def __init__(self):
        self.data = pd.DataFrame(columns=['x','y','z'])
        self.rgb_max = 1.0
        self._reset_cached()

    @classmethod
    def from_numpy_arrays(self, xyz: np.ndarray, rgb: Union[np.ndarray, None] = None, rgb_max=1.0,
                          normals_xyz: Union[np.ndarray, None] = None, curvature: Union[np.ndarray, None] = None,
                          data_axis_is_first=False):
        cloud = GeoPointCloud()
        if data_axis_is_first:
            xyz = xyz.transpose()
        cloud.data['x'] = xyz[:,0]
        cloud.data[['y', 'z']] = xyz[:,1:]
        if rgb is not None:
            if data_axis_is_first:
                rgb = rgb.transpose()
            cloud.data[['r', 'g', 'b']] = rgb
        cloud.rgb_max = rgb_max
        if normals_xyz is not None:
            if data_axis_is_first:
                normals_xyz = normals_xyz.transpose()
            cloud.data[['n_x', 'n_y', 'n_z']] = normals_xyz
        if curvature is not None:
            if data_axis_is_first:
                curvature = curvature.transpose()
            cloud.data['curvature'] = curvature
        return cloud

    @classmethod
    def create_empty(cls):
        return GeoPointCloud()

    @classmethod
    def from_pandas(cls, data_frame: pd.DataFrame, rgb_max=1.0):
        if 'x' not in data_frame or 'y' not in data_frame or 'z' not in data_frame:
            raise KeyError("DataFrame needs at least columns x,y,z.")
        cloud = GeoPointCloud()
        cloud.data = data_frame
        cloud.rgb_max = rgb_max
        return cloud

    @classmethod
    def from_structured_array(cls, structured_array: np.ndarray, rgb_max=1.0):
        x = structured_array['X']
        y = structured_array['Y']
        z = structured_array['Z']
        xyz = np.stack((x, y, z))
        fields = structured_array.dtype.fields
        rgb = None
        if 'Red' in fields and 'Green' in fields and 'Blue' in fields:
            rgb = np.stack((structured_array['Red'], structured_array['Green'], structured_array['Blue']))
        normals_xyz = None
        if 'NormalX' in fields and 'NormalY' in fields and 'NormalZ' in fields:
            normals_xyz = np.stack((structured_array['NormalX'], structured_array['NormalY'],
                                    structured_array['NormalZ']))
        curvature = None
        if 'Curvature' in fields:
            curvature = structured_array['Curvature']
        return GeoPointCloud.from_numpy_arrays(xyz, rgb=rgb, rgb_max=rgb_max, normals_xyz=normals_xyz,
                                               curvature=curvature, data_axis_is_first=True)

    @classmethod
    def from_o3d(cls, o3d_point_cloud: o3d.geometry.PointCloud):
        if(not o3d_point_cloud.has_points()):
            raise ValueError("Open3d point cloud needs to have points")
        xyz = np.array(o3d_point_cloud.points)
        normals = None
        if(o3d_point_cloud.has_normals()):
            normals = np.array(o3d_point_cloud.normals)
            if(normals.shape[0] != xyz.shape[0]):
                normals = None
        rgb = None
        if(o3d_point_cloud.has_colors()):
            rgb = np.array(o3d_point_cloud.colors)
            if (rgb.shape[0] != xyz.shape[0]):
                rgb = None
        return GeoPointCloud.from_numpy_arrays(xyz, rgb=rgb, normals_xyz=normals, rgb_max=255)

    @classmethod
    def from_las(cls, file_path):
        pipeline_description = [str(file_path)]
        pipeline = pdal.Pipeline(json.dumps(pipeline_description))
        count = pipeline.execute()
        arrays = pipeline.arrays
        metadata = json.loads(pipeline.metadata)
        return GeoPointCloud.from_structured_array(arrays[0], rgb_max=np.iinfo(np.uint16).max)

    @classmethod
    def from_xyz_file(cls, file_path: Union[str, Path], read_normals=False):
        file_path = Path(file_path)
        names = ('x', 'y', 'z')
        if read_normals:
            names = ('x', 'y', 'z', 'n_x', 'n_y', 'n_z')
        points = pd.read_csv(file_path, sep=' ', header=0, names=names)
        xyz = np.stack([points['x'].to_numpy(), points['y'].to_numpy(), points['z'].to_numpy()], axis=0)
        normals = None
        if read_normals:
            normals = np.stack([points['n_x'].to_numpy(), points['n_y'].to_numpy(), points['n_z'].to_numpy()], axis=0)
        return GeoPointCloud.from_numpy_arrays(xyz, normals_xyz=normals, data_axis_is_first=True)

    @classmethod
    def from_others(cls, others: List['GeoPointCloud'], keep_others=True) -> 'GeoPointCloud':
        if keep_others:
            result = others[0].copy()
        else:
            result = others[0]
        if len(others) == 1:
            return result
        for other in others[1:]:
            result.extend(other, keep_others)
        return result

    @property
    def xyz(self) -> pd.DataFrame:
        return self.data[['x', 'y', 'z']]

    @xyz.setter
    def xyz(self, xyz: np.ndarray) -> pd.DataFrame:
        assert xyz.shape == self.xyz.shape
        self.data[['x', 'y', 'z']] = xyz

    @property
    def x(self) -> pd.Series:
        return self.data['x']

    @property
    def y(self) -> pd.Series:
        return self.data['z']

    @property
    def z(self) -> pd.Series:
        return self.data['z']

    @property
    def rgb(self) -> Optional[pd.DataFrame]:
        if 'r' in self.data and 'g' in self.data and 'b' in self.data:
            return self.data[['r', 'g', 'b']]
        else:
            return None

    @property
    def normals_xyz(self) -> Optional[pd.DataFrame]:
        if 'n_x' in self.data and 'n_y' in self.data and 'n_z' in self.data:
            return self.data[['n_x', 'n_y', 'n_z']]
        else:
            return None

    @property
    def curvature(self) -> Optional[pd.Series]:
        if 'curvature' in self.data:
            return self.data['curvature']
        else:
            return None

    @property
    def confidence(self) -> Optional[pd.Series]:
        if 'confidence' in self.data:
            return self.data['confidence']
        else:
            return None

    @property
    def shape(self) -> Tuple[float, ...]:
        return self.data.shape

    def _reset_cached(self):
        self._bounds = None
        self._principle_components = None
        self._cov = None
        self._aab_dims = None
        self._center = None
        if 'n_x' in self.data:
            del self.data['n_x']
        if 'n_y' in self.data:
            del self.data['n_y']
        if 'n_z' in self.data:
            del self.data['n_z']

    def _dtypes(self):
        dtypes = {}
        for key, value in self.data.items():
            dtypes[key] = value.dtype

    def to_o3d(self):
        xyz_o3d = o3d.utility.Vector3dVector(self.xyz.to_numpy())
        pc3d = o3d.geometry.PointCloud()
        pc3d.points = xyz_o3d
        if self.normals_xyz is not None:
            normals_xyz_o3d = o3d.utility.Vector3dVector(self.normals_xyz.to_numpy())
            pc3d.normals = normals_xyz_o3d
        if self.rgb is not None:
            rgb = self.rgb.to_numpy().astype(float) / self.rgb_max
            rgb_o3d = o3d.utility.Vector3dVector(rgb)
            pc3d.colors = rgb_o3d
        return pc3d

    def to_structured_array(self) -> np.ndarray:
        sub_arrays = []
        dtypes = []
        dt = self.xyz.dtype
        sub_arrays.append(('X', self.xyz[0]))
        sub_arrays.append(('Y', self.xyz[1]))
        sub_arrays.append(('Z', self.xyz[2]))
        dtypes.extend([dt, dt, dt])
        if self.rgb is not None:
            dt = self.rgb.dtype
            sub_arrays.append(('Red', self.rgb[0]))
            sub_arrays.append(('Green', self.rgb[1]))
            sub_arrays.append(('Blue', self.rgb[2]))
        dtypes.extend([dt, dt, dt])
        if self.normals_xyz is not None:
            dt = self.normals_xyz.dtype
            sub_arrays.append(('NormalX', self.normals_xyz[0]))
            sub_arrays.append(('NormalY', self.normals_xyz[1]))
            sub_arrays.append(('NormalZ', self.normals_xyz[2]))
        dtypes.extend([dt, dt, dt])
        if self.curvature is not None:
            sub_arrays.append(('Curvature', self.curvature))
            dtypes.append(self.curvature.dtype)
        result = np.empty(self.size, np.dtype([(a[0], dtypes[i]) for i, a in enumerate(sub_arrays)]))
        for a in sub_arrays:
            result[a[0]] = a[1]
        return result

    def to_las(self, file_path: Union[str, Path]):
        file_path = Path(file_path)
        point_format_id = 0
        if self.rgb is not None:
            point_format_id = 3
        las_data = pylas.create(point_format_id=point_format_id, file_version="1.3")
        if self.size:
            las_data.x = self.xyz['x'].to_numpy()
            las_data.y = self.xyz['y'].to_numpy()
            las_data.z = self.xyz['z'].to_numpy()
        if self.rgb is not None:
            rgb = self.rgb
            if self.rgb_max != np.iinfo(np.uint16).max:
                rgb = rgb / self.rgb_max * np.iinfo(np.uint16).max
            las_data.red = rgb['r'].to_numpy().astype(np.uint16)
            las_data.green = rgb['g'].to_numpy().astype(np.uint16)
            las_data.blue = rgb['b'].to_numpy().astype(np.uint16)
        las_data.update_header()
        file_path.parents[0].mkdir(parents=True, exist_ok=True)
        las_data.write(str(file_path))

    def apply_pdal_pipeline(self, pipeline_json: str, return_raw_output=False):
        pipeline = pdal.Pipeline(pipeline_json, [self.to_structured_array()])
        pipeline.validate()
        count = pipeline.execute()
        if return_raw_output:
            return pipeline.arrays, pipeline.metadata, pipeline.log
        else:
            return [GeoPointCloud.from_structured_array(a) for a in pipeline.arrays]

    def __getitem__(self, item):
        if isinstance(item, np.ndarray):
            return self.copy(indices=item)

    def copy(self, indices: Union[np.ndarray, None] = None):
        if indices is not None:
            result = GeoPointCloud.from_pandas(self.data[indices])
        else:
            result = GeoPointCloud.from_pandas(self.data.copy())
        result._bounds = self._bounds
        result._principle_components = self._principle_components
        result._cov = self._cov
        result._aab_dims = self._aab_dims
        result._center = self._center
        result.rgb_max = self.rgb_max
        return result

    def column_descriptor(self):
        return dict([(name, column.dtype) for name, column in self.data.items()])

    def extend(self, other: 'GeoPointCloud', keep_other=True):
        if self.column_descriptor() != other.column_descriptor():
            raise ValueError("The point clouds have different attribute names or types")
        self.data = pd.concat([self.data, other.data])
        if not keep_other:
            del other

    @property
    def size(self):
        return self.xyz.shape[1]

    def __len__(self):
        return self.size

    @property
    def cov(self):
        if self._cov is None:
            self._cov = np.cov((self.xyz - self.xyz.mean(axis=1, keepdims=True)) /
                               self.xyz.std(axis=1, keepdims=True))
        return self._cov

    @property
    def principle_components(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns: (eigenvalues, eigenvectors)

        """
        if self._principle_components is None:
            pca = PCA(n_components=3)
            pca.fit(self.xyz.T)
            self._principle_components = (pca.singular_values_, pca.components_)
        return self._principle_components

    def thin_by_grid(self, grid_size: float, inplace=True):
        if inplace:
            cloud = self
        else:
            cloud = self.copy()
        cloud.data[['x', 'y', 'z']] //= grid_size
        #cloud.data[['x', 'y', 'z']] = cloud.data[['x', 'y', 'z']].round()
        data = cloud.data.groupby(['x', 'y', 'z'], as_index=False).mean()
        data[['x', 'y', 'z']] *= grid_size
        cloud.data = data
        return cloud
        #self._reset_cached() TODO: think about whether we should be super accurate here and reset the cache

    @property
    def center(self) -> np.ndarray:
        if self._center is None:
            self._center = self.xyz.mean(axis=1)
        return self._center

    def make_axis_aligned(self):
        transform = self.principle_components[1]
        self.xyz = transform @ (self.xyz - self.xyz.mean(axis=1, keepdims=True))

    def estimate_normals(self, knn_k=30, fast_normal_computation=True):
        o3d_pc = self.to_o3d()
        o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn_k), fast_normal_computation)
        normals = np.array(o3d_pc.normals)
        self.data[['n_x', 'n_y', 'n_z']] = normals

    def subsample_random(self, num_points) -> 'GeoPointCloud':
        indices = np.random.choice(self.shape[0], num_points, replace=False)
        return GeoPointCloud.from_pandas(self.data.iloc[indices])

    @property
    def aab_dims(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns: (aab_dim_lengths, aab_dim_vectors) sorted big to small (largest component first)

        """
        if self._aab_dims is None:
            transform = self.principle_components[1]
            xyz = transform @ (self.xyz - self.xyz.mean(axis=1, keepdims=True))
            self._aab_dims = (xyz.max(axis=1) - xyz.min(axis=1), self.principle_components[1])
        return self._aab_dims

    @property
    def bounds(self):
        if self._bounds is None:
            mins = self.xyz.min(axis=1)
            maxs = self.xyz.max(axis=1)
            self._bounds = [*mins[:3], *maxs[:3]]
        return self._bounds

    #TODO: Deprecated, remove!
    def visualize_pptk(self):
        import warnings
        warnings.warn(".visualize_pptk got renamed use .plot_pptk() instead", DeprecationWarning)
        self.plot_pptk()

    def plot_pptk(self):
        if self.rgb is None:
            v = pptk.viewer(self.xyz.to_numpy())
        else:
            print("rendering with RGB")
            rgb = self.rgb.to_numpy()
            if self.rgb_max != 255:
                rgb = rgb.astype(float) / self.rgb_max
            v = pptk.viewer(self.xyz.to_numpy(), rgb)
        return v

    def plot_o3d(self, *args, **kwargs):
        o3d.visualization.draw_geometries([self.to_o3d()], *args, **kwargs)
