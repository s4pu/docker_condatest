from typing import Union, Tuple, Optional
from pathlib import Path
import glob
import zlib
import json

import numpy as np
import pandas as pd
import ffmpeg
from tqdm.auto import tqdm

from .geopointcloud import GeoPointCloud

FLIP_YZ_MAT = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]])

class QuaryKitImporter:

    def __init__(self, folder_path: Union[str, Path], confidence_thresh: Optional[float] = None,
                 depth_thresh: Optional[float] = None, grid_size: Optional[float] = None):
        folder_path = Path(folder_path)
        self.metadata_files = glob.glob(str(folder_path / "*.json"))
        self.confidence_thresh = confidence_thresh
        self.depth_thresh = depth_thresh
        self.grid_size = grid_size

    def read(self, make_progress_bar=False):
        if make_progress_bar:
            it = tqdm(self.metadata_files)
        else:
            it = self.metadata_files
        result_clouds = []
        for file in it:
            result_clouds.append(self.read_package(file, make_progress_bar=make_progress_bar))
        result = GeoPointCloud.from_others(result_clouds, keep_others=False)
        if self.grid_size is not None:
            result.thin_by_grid(self.grid_size)
        return result


    def read_package(self, meta_info_file: Union[str, Path], make_progress_bar=False):
        path = Path(meta_info_file)
        with open(path) as f:
            print("reading " + str(path))
            metadata = json.load(f)
            metadata['root_dir'] = path.parent
            frames = metadata['frames']
            result_clouds = []
            depth_data = self.read_depth_package(metadata)
            confidence_data = self.read_confidence_package(metadata)
            rgb_data = self.read_rgb_package(metadata)
            if make_progress_bar:
                it = tqdm(range(0, len(frames)))
            else:
                it = range(0, len(frames))
            for frame in it:
                result_clouds.append(self._get_cloud_for_frame(metadata, frame,
                                                               depth_data=depth_data, rgb_data=rgb_data,
                                                               confidence_data=confidence_data))
            df = pd.DataFrame(np.concatenate(result_clouds, axis=0),
                              columns=['x', 'y', 'z', 'r', 'g', 'b', 'confidence'])
            result = GeoPointCloud.from_pandas(df)
            if self.grid_size is not None:
                result.thin_by_grid(self.grid_size)
            return result

    def read_depth_package(self, package_metadata: dict):
        frames = package_metadata['frames']
        num_frames = len(frames)
        depth_file = package_metadata['root_dir'] / frames[0]['depthVideoFrame']['fileName']
        depth_resolution = frames[0]['depthVideoFrame']['resolution']
        shape = (num_frames, depth_resolution[1], depth_resolution[0])
        return self.read_compressed_buffer(depth_file, np.float32, shape)

    def read_confidence_package(self, package_metadata: dict):
        frames = package_metadata['frames']
        num_frames = len(frames)
        file = package_metadata['root_dir'] / frames[0]['confidenceVideoFrame']['fileName']
        print(file)
        res = frames[0]['confidenceVideoFrame']['resolution']
        shape = (num_frames, res[1], res[0])
        return self.read_compressed_buffer(file, np.uint8, shape)

    def read_rgb_package(self, package_metadata: dict, scale_to_depth_resolution = True):
        frames = package_metadata['frames']
        rgb_file = package_metadata['root_dir'] / frames[0]['rgbVideoFrame']['fileName']
        if scale_to_depth_resolution:
            resolution = frames[0]['depthVideoFrame']['resolution']
        else:
            resolution = frames[0]['rgbVideoFrame']['resolution']
        return self.read_movie_data(rgb_file, resolution)

    def read_movie_data(self, movie_file_name: str, output_resolution: Tuple[int, int]):
        out, _ = (
            ffmpeg
                .input(movie_file_name)
                .filter('scale', output_resolution[0], output_resolution[1])
                .output('pipe:', format='rawvideo', pix_fmt='rgba')
                .run(capture_stdout=True, )
        )
        video = (
            np
                .frombuffer(out, np.uint8)
                .reshape([-1, output_resolution[1], output_resolution[0], 4])
        )
        return video

    def read_compressed_buffer(self, file_path: Union[str, Path], dtype: np.dtype, shape: Tuple[float, ...]) -> np.ndarray:
        data = np.fromfile(file_path, dtype=np.uint8)
        data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
        data = np.frombuffer(data, dtype=dtype).reshape(shape)
        return data

    def _get_cloud_for_frame(self, package_metadata:dict, frame: int, depth_data, rgb_data, confidence_data):
        frames = package_metadata['frames']
        depth_resolution = frames[0]['depthVideoFrame']['resolution']
        rgb_resolution = frames[0]['rgbVideoFrame']['resolution']
        x, y = np.meshgrid(np.arange(depth_resolution[0]), np.arange(depth_resolution[1]))
        x = x.flatten()
        y = y.flatten()
        depth_frame = int(frames[frame]['depthVideoFrame']['frame'])
        rgb_frame = int(frames[frame]['rgbVideoFrame']['frame'])
        confidence_frame = int(frames[frame]['confidenceVideoFrame']['frame'])
        #TODO: we might be a tiny bit more efficient by changing the oder in which we apply the filters
        if self.confidence_thresh is not None:
            mask = confidence_data[confidence_frame, y, x] >= self.confidence_thresh
            x = x[mask]
            y = y[mask]
        if self.depth_thresh is not None:
            mask = depth_data[depth_frame, y, x] <= self.depth_thresh
            x = x[mask]
            y = y[mask]
        depths_flat = depth_data[depth_frame, y, x]
        rgba_flat = rgb_data[rgb_frame, y, x]
        confidence_flat = confidence_data[confidence_frame, y, x]
        img_coords = np.stack([x, y, np.ones(x.shape[0])], axis=0)
        img_coords = img_coords.astype(float)
        img_coords[0] *= rgb_resolution[0] / depth_resolution[0]
        img_coords[1] *= rgb_resolution[1] / depth_resolution[1]
        camera_intrinsics = np.array(frames[frame]['cameraIntrinsics'][0]).transpose()
        camera_intrinsics_inv = np.linalg.inv(camera_intrinsics)
        local_coords = np.matmul(camera_intrinsics_inv, img_coords * depths_flat)
        # local_coords = local_coords * depths_flat
        rgb = rgba_flat[:, :3].astype(float) / 255
        # viewer = pptk.viewer(local_coords.transpose(), rgb)
        local_to_world = np.linalg.inv(np.array(frames[frame]['cameraViewMatrix'][0]).transpose())
        xyzw = np.concatenate([local_coords, np.expand_dims(np.ones(local_coords.shape[1]), axis=0)], axis=0)
        xyzw = np.matmul(np.matmul(local_to_world, FLIP_YZ_MAT), xyzw)
        xyz = xyzw[:3] / xyzw[3]
        res =  np.concatenate([xyz.transpose(), rgb, confidence_flat[:, np.newaxis]], axis=1)
        return res
