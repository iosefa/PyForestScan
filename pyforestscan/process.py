import os

import numpy as np

from pyforestscan.calculate import calculate_fhd, calculate_pad, calculate_pai, assign_voxels
from pyforestscan.handlers import create_geotiff


def process_with_tiles(arrays, tile_size, output_path, metric, voxel_size, srs=None):
    """
    Generates tiles from a point cloud array and writes them to the specified output path.

    :param voxel_size:
    :param metric:
    :param arrays: The point cloud data, containing 'X' and 'Y' coordinates.
    :type arrays: numpy.ndarray
    :param tile_size: The dimensions of each tile as (width, height).
    :type tile_size: tuple
    :param output_path: The directory path where the generated tiles will be saved.
    :type output_path: str
    :param srs: The spatial reference system for the tiles. Optional.
    :type srs: object, optional
    :return: None
    :rtype: None
    :raises OSError: If unable to create the output directory.
    """
    x_coords = arrays['X']
    y_coords = arrays['Y']

    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    num_tiles_x = int(np.ceil((max_x - min_x) / tile_size[0]))
    num_tiles_y = int(np.ceil((max_y - min_y) / tile_size[1]))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            tile_min_x = min_x + i * tile_size[0]
            tile_max_x = min_x + (i + 1) * tile_size[0]
            tile_min_y = min_y + j * tile_size[1]
            tile_max_y = min_y + (j + 1) * tile_size[1]

            spatial_extent = (tile_min_x, tile_max_x, tile_min_y, tile_max_y)

            tile_indices = np.where(
                (x_coords >= tile_min_x) & (x_coords < tile_max_x) &
                (y_coords >= tile_min_y) & (y_coords < tile_max_y)
            )

            tile_points = arrays[tile_indices]

            if tile_points.size == 0:
                empty_tile_shape = (
                    int(tile_size[0] / voxel_size[0]),
                    int(tile_size[1] / voxel_size[1]),
                )
                empty_tile = np.zeros(empty_tile_shape)

                result_file = os.path.join(output_path, f"tile_{i}_{j}_{metric}.tif")
                create_geotiff(empty_tile, result_file, srs, spatial_extent)
            else:
                voxels, spatial_extent = assign_voxels(tile_points, voxel_size)

                if metric == "fhd":
                    result = calculate_fhd(voxels)
                    result_file = os.path.join(output_path, f"tile_{i}_{j}_fhd.tif")
                    create_geotiff(result, result_file, srs, spatial_extent)
                elif metric == "pad":
                    result = calculate_pad(voxels, voxel_size[-1])
                    # todo: need to specify layer! or do all layers!
                    # result_file = os.path.join(output_path, f"{os.path.splitext(os.path.basename(lidar_file))[0]}_pad.tif")
                    # create_geotiff(result, result_file, srs, spatial_extent)
                elif metric == "pai":
                    pad = calculate_pad(voxels, voxel_size[-1])
                    result = calculate_pai(pad)
                    result_file = os.path.join(output_path, f"tile_{i}_{j}_pai.tif")
                    create_geotiff(result, result_file, srs, spatial_extent)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
                # todo; add logic to also write tiled las if requested
