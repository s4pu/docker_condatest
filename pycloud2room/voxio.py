from typing import List, Union
from pathlib import Path

import numpy as np

import pycloud2room.geovoxelgrid as geovoxelgrid

def read_vox_file( path) -> 'geovoxelgrid.GeoVoxelGrid':
    data = np.fromfile(path, np.uint8)
    header = data[:8]
    main = data[8:]
    chunks = split_chunks(main)[1]
    if chunks[0][0] != 'SIZE':
        raise Exception("We only support simple .vox files with one model (first chunk of type SIZE)")
    size_content = chunks[0][2]
    size_x = size_content[:4].view(np.uint32).item()
    size_y = size_content[4:8].view(np.uint32).item()
    size_z = size_content[8:12].view(np.uint32).item()
    if chunks[1][0] != 'XYZI':
        raise Exception("Something went wrong, expecting XYZI chunk after SIZE chunk")
    xyzi_content = chunks[1][2]
    num_voxels = xyzi_content[:4].view(np.uint32).item()
    voxels = np.frombuffer(xyzi_content[4:4 + 4 * num_voxels], dtype=np.uint8).reshape(num_voxels, 4)
    voxel_grid = np.zeros((size_x, size_y, size_z), dtype=bool)
    voxel_grid[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = 1
    return geovoxelgrid.GeoVoxelGrid(voxel_grid)

def write_vox_file(voxel_grid: 'geovoxelgrid.GeoVoxelGrid', path: Union[str,Path], default_color=55):
    voxel_positions = voxel_grid.indices().transpose()
    voxel_chunks = create_voxel_chunks(voxel_positions, default_color)
    main_chunk = ('MAIN', np.array([], dtype=np.uint8), voxel_chunks)
    vox_header = np.frombuffer('VOX '.encode('utf8'), dtype=np.uint8)
    version_number = np.array([150], dtype=np.uint32).view(np.uint8)
    header = np.concatenate([vox_header, version_number])
    file_data = encode_chunks([main_chunk])
    file_data = np.concatenate([header, file_data])
    with open(Path(path), 'wb') as file:
        file.write(bytes(file_data))


def parse_chunk(chunk):
    chunk_id = chunk[:4]
    content_size = chunk[4:8].view(np.uint32).item()
    children_size = chunk[8:12].view(np.uint32).item()
    content = chunk[12: 12 + content_size]
    children = chunk[12 + content_size:12 + content_size + children_size]
    return chunk_id, content_size, children_size, content, children

def split_chunks(chunk: np.ndarray) -> List[np.ndarray]:
    result = []
    while True:
        chunk_id, content_size, children_size, content, children = parse_chunk(chunk)
        result.append((bytes(chunk_id).decode('utf8'), content_size, content))
        if children_size != 0:
            result.append(split_chunks(children))
        _, chunk = np.split(chunk, [12 + content_size,])
        if chunk.shape[0] == 0:
            break
    return result

def encode_chunks(chunk_informations: List = None):
    result_data = []
    for chunk_name, chunk_data, children in chunk_informations:
        if len(children) > 0:
            child_data = encode_chunks(children)
        else:
            child_data = np.array([], dtype=np.uint8)
        data_size = np.array([chunk_data.shape[0]], dtype=np.uint32).view(np.uint8)
        child_size = np.array([child_data.shape[0]], dtype=np.uint32).view(np.uint8)
        chunk_name = np.frombuffer(chunk_name.encode("utf8"), dtype=np.uint8)
        data = [chunk_name, data_size, child_size, chunk_data, child_data]
        result_data.append(np.concatenate(data, axis=0))
    return np.concatenate(result_data, axis=0)

def create_voxel_chunks(voxel_positions: np.ndarray, default_color):
    min_xyz = voxel_positions.min(axis=0)
    max_xyz = voxel_positions.max(axis=0)
    max_xyz -= min_xyz
    voxel_positions -= min_xyz
    for item in max_xyz:
        assert item < 256
    size_chunk_data = (max_xyz + 1).astype(np.uint32).view(np.uint8)
    size_chunk = ('SIZE', size_chunk_data, [])
    num_voxels = np.array([voxel_positions.shape[0]], dtype=np.uint32).view(np.uint8)
    voxel_data = voxel_positions.astype(np.uint8)
    colors = np.ones(voxel_data.shape[0], dtype=np.uint8)[:, np.newaxis] * int(default_color)
    voxel_data = np.concatenate([voxel_data, colors], axis=1)
    voxel_data = voxel_data.flatten()
    xyzi_chunk_data = np.concatenate([num_voxels, voxel_data])
    xyzi_chunk = ('XYZI', xyzi_chunk_data, [])
    return [size_chunk, xyzi_chunk]