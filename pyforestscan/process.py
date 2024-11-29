import json

from pyforestscan.calculate import calculate_fhd, calculate_pad, calculate_pai, assign_voxels, calculate_chm
from pyforestscan.handlers import create_geotiff

import numpy as np
import os
import pdal
from tqdm import tqdm

from pyforestscan.pipeline import _hag_raster, _hag_delaunay


def get_bounds(ept_file):
    """
    Extracts the spatial bounds of a point cloud from an ept file using PDAL.

    :param ept_file: Path to the ept file containing the point cloud data.
    :return: A tuple with bounds in the format (min_x, max_x, min_y, max_y).
    """
    pipeline_json = f"""
    {{
        "pipeline": [
            "{ept_file}",
            {{
                "type": "filters.info"
            }}
        ]
    }}
    """

    pipeline = pdal.Pipeline(pipeline_json)
    pipeline.execute()
    metadata = pipeline.metadata['metadata']
    try:
        min_x = metadata['filters.info']['bbox']['minx']
        max_x = metadata['filters.info']['bbox']['maxx']
        min_y = metadata['filters.info']['bbox']['miny']
        max_y = metadata['filters.info']['bbox']['maxy']
        return min_x, max_x, min_y, max_y
    except KeyError:
        raise KeyError("Bounds information is not available in the metadata.")


def process_with_tiles(ept_file, tile_size, output_path, metric, voxel_size, buffer_size=0.1, srs=None, hag=False,
                       hag_dtm=False, dtm=None):
    """
    Processes a large EPT point cloud by tiling, calculates CHM or other metrics for each tile,
    and writes the results to the specified output path.

    :param ept_file: Path to the EPT file containing the point cloud data.
    :param tile_size: Tuple (tile_width, tile_height) specifying the size of each tile.
    :param output_path: Directory where the output files will be saved.
    :param metric: Metric to compute ("chm", "fhd", or "pai").
    :param voxel_size: Tuple specifying the voxel resolution (x_resolution, y_resolution, z_resolution).
    :param buffer_size: Fractional buffer size relative to tile size (e.g., 0.1 for 10% buffer).
    :param srs: Spatial reference system (optional).
    :param hag: Boolean indicating whether to compute Height Above Ground using Delaunay triangulation.
    :param hag_dtm: Boolean indicating whether to compute Height Above Ground using a provided DTM raster.
    :param dtm: Path to the DTM raster file (required if hag_dtm is True).
    """
    min_x, max_x, min_y, max_y = get_bounds(ept_file)
    num_tiles_x = int(np.ceil((max_x - min_x) / tile_size[0]))
    num_tiles_y = int(np.ceil((max_y - min_y) / tile_size[1]))
    total_tiles = num_tiles_x * num_tiles_y

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with tqdm(total=total_tiles, desc="Processing tiles") as pbar:
        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                if metric == "chm":
                    current_buffer_size = buffer_size
                else:
                    current_buffer_size = 0.0

                buffer_x = current_buffer_size * tile_size[0]
                buffer_y = current_buffer_size * tile_size[1]
                tile_min_x = min_x + i * tile_size[0] - buffer_x
                tile_max_x = min_x + (i + 1) * tile_size[0] + buffer_x
                tile_min_y = min_y + j * tile_size[1] - buffer_y
                tile_max_y = min_y + (j + 1) * tile_size[1] + buffer_y

                tile_min_x = max(min_x, tile_min_x)
                tile_max_x = min(max_x, tile_max_x)
                tile_min_y = max(min_y, tile_min_y)
                tile_max_y = min(max_y, tile_max_y)

                if tile_max_x <= tile_min_x or tile_max_y <= tile_min_y:
                    print(f"Warning: Skipping tile ({i}, {j}) due to invalid spatial extent.")
                    pbar.update(1)
                    continue

                tile_pipeline_stages = [
                    {
                        "type": "filters.crop",
                        "bounds": f"([{tile_min_x},{tile_max_x}], [{tile_min_y},{tile_max_y}])"
                    }
                ]

                if hag:
                    tile_pipeline_stages.append(_hag_delaunay())
                elif hag_dtm:
                    if not dtm or not os.path.isfile(dtm):
                        raise FileNotFoundError(f"DTM file is required for HAG calculation using DTM: {dtm}")
                    tile_pipeline_stages.append(_hag_raster(dtm))

                tile_pipeline_json = {
                    "pipeline": [
                                    {"type": "readers.ept", "filename": ept_file}
                                ] + tile_pipeline_stages
                }

                tile_pipeline = pdal.Pipeline(json.dumps(tile_pipeline_json))
                tile_pipeline.execute()
                tile_points = tile_pipeline.arrays[0]

                if tile_points.size == 0:
                    print(f"Warning: No data in tile ({i}, {j}). Skipping.")
                    pbar.update(1)
                    continue

                buffer_pixels_x = int(np.ceil(buffer_x / voxel_size[0]))
                buffer_pixels_y = int(np.ceil(buffer_y / voxel_size[1]))

                if metric == "chm":
                    chm, extent = calculate_chm(tile_points, voxel_size)

                    if buffer_pixels_x * 2 >= chm.shape[1] or buffer_pixels_y * 2 >= chm.shape[0]:
                        print(
                            f"Warning: Buffer size exceeds CHM dimensions for tile ({i}, {j}). Adjusting buffer size.")
                        buffer_pixels_x = max(0, chm.shape[1] // 2 - 1)
                        buffer_pixels_y = max(0, chm.shape[0] // 2 - 1)

                    chm = chm[buffer_pixels_y:-buffer_pixels_y, buffer_pixels_x:-buffer_pixels_x]

                    core_extent = (
                        tile_min_x + buffer_x,
                        tile_max_x - buffer_x,
                        tile_min_y + buffer_y,
                        tile_max_y - buffer_y,
                    )

                    result_file = os.path.join(output_path, f"tile_{i}_{j}_chm.tif")
                    create_geotiff(chm, result_file, srs, core_extent)
                elif metric in ["fhd", "pai"]:
                    voxels, spatial_extent = assign_voxels(tile_points, voxel_size)

                    if metric == "fhd":
                        result = calculate_fhd(voxels)
                    elif metric == "pai":
                        pad = calculate_pad(voxels, voxel_size[-1])

                        if np.all(pad == 0):
                            result = np.zeros((pad.shape[0], pad.shape[1]))
                        else:
                            result = calculate_pai(pad)
                        result = np.where(np.isfinite(result), result, 0)

                    if current_buffer_size > 0:
                        if buffer_pixels_x * 2 >= result.shape[1] or buffer_pixels_y * 2 >= result.shape[0]:
                            print(
                                f"Warning: Buffer size exceeds {metric.upper()} dimensions for tile ({i}, {j}). Adjusting buffer size.")
                            buffer_pixels_x = max(0, result.shape[1] // 2 - 1)
                            buffer_pixels_y = max(0, result.shape[0] // 2 - 1)

                        result = result[buffer_pixels_y:-buffer_pixels_y, buffer_pixels_x:-buffer_pixels_x]

                    core_extent = (
                        tile_min_x + buffer_x,
                        tile_max_x - buffer_x,
                        tile_min_y + buffer_y,
                        tile_max_y - buffer_y,
                    )

                    if core_extent[1] <= core_extent[0] or core_extent[3] <= core_extent[2]:
                        print(f"Warning: Invalid core extent for tile ({i}, {j}): {core_extent}. Skipping.")
                        pbar.update(1)
                        continue

                    result_file = os.path.join(output_path, f"tile_{i}_{j}_{metric}.tif")
                    create_geotiff(result, result_file, srs, core_extent)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")

                pbar.update(1)
