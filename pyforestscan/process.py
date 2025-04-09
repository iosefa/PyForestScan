import json
import pdal
import numpy as np
import os

from tqdm import tqdm

from pyforestscan.calculate import calculate_fhd, calculate_pad, calculate_pai, assign_voxels, calculate_chm
from pyforestscan.filters import remove_outliers_and_clean
from pyforestscan.handlers import create_geotiff
from pyforestscan.pipeline import _hag_raster, _hag_delaunay
from pyforestscan.utils import get_bounds_from_ept, get_srs_from_ept


import tempfile
import os
import rasterio
from rasterio.windows import from_bounds


def _crop_dtm(dtm_path, tile_min_x, tile_min_y, tile_max_x, tile_max_y):
    """
    Crops the input DTM TIFF to the given bounding box (in the same CRS),
    saves it to a temporary file, and returns the path to that temp file.
    """
    with rasterio.open(dtm_path) as src:
        window = from_bounds(
            left=tile_min_x, bottom=tile_min_y,
            right=tile_max_x, top=tile_max_y,
            transform=src.transform
        )
        data = src.read(1, window=window)
        new_transform = src.window_transform(window)

        fd, cropped_path = tempfile.mkstemp(suffix=".tif")
        os.close(fd)

        new_profile = src.profile.copy()
        new_profile.update({
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": new_transform
        })

        with rasterio.open(cropped_path, "w", **new_profile) as dst:
            dst.write(data, 1)

    return cropped_path


def process_with_tiles(ept_file, tile_size, output_path, metric, voxel_size, buffer_size=0.1, srs=None, hag=False,
                       hag_dtm=False, dtm=None, bounds=None, interpolation=None, remove_outliers=False):
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
    :param interpolation: Interpolation method to use for CHM calculation (e.g., "linear", "cubic", "nearest", or None).
    :param bounds: Bounds within which to crop the data. Must be of the form: ([xmin, xmax], [ymin, ymax], [zmin, zmax]) or ([xmin, xmax], [ymin, ymax]). If none is given, tiling will happen on the entire dataset.
    :param remove_outliers: Boolean indicating whether to remove statistical outliers before calculating metrics.
    """
    if metric not in ["chm", "fhd", "pai"]:
        raise ValueError(f"Unsupported metric: {metric}")

    (min_z, max_z) = (None, None)
    if bounds:
        if len(bounds) == 2:
            (min_x, max_x), (min_y, max_y) = bounds
        else:
            (min_x, max_x), (min_y, max_y), (min_z, max_z) = bounds
    else:
        min_x, max_x, min_y, max_y, min_z, max_z = get_bounds_from_ept(ept_file)

    if not srs:
        srs = get_srs_from_ept(ept_file)

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

                if min_z and max_z:
                    tile_bounds = ([tile_min_x, tile_max_x], [tile_min_y, tile_max_y], [min_z, max_z])
                else:
                    tile_bounds = ([tile_min_x, tile_max_x], [tile_min_y, tile_max_y])
                tile_pipeline_stages = []

                if hag:
                    tile_pipeline_stages.append(_hag_delaunay())
                elif hag_dtm:
                    if not dtm or not os.path.isfile(dtm):
                        raise FileNotFoundError(f"DTM file is required for HAG calculation using DTM: {dtm}")
                    cropped_dtm_path = _crop_dtm(
                        dtm,
                        tile_min_x, tile_min_y,
                        tile_max_x, tile_max_y
                    )
                    tile_pipeline_stages.append(_hag_raster(cropped_dtm_path))
                base_pipeline = {
                    "type": "readers.ept",
                    "filename": ept_file,
                    "bounds": f"{tile_bounds}",
                }
                tile_pipeline_json = {
                    "pipeline": [base_pipeline] + tile_pipeline_stages
                }

                tile_pipeline = pdal.Pipeline(json.dumps(tile_pipeline_json))
                tile_pipeline.execute()
                if remove_outliers:
                    tile_points = remove_outliers_and_clean(tile_pipeline.arrays)[0]
                else:
                    tile_points = tile_pipeline.arrays[0]

                if tile_points.size == 0:
                    print(f"Warning: No data in tile ({i}, {j}). Skipping.")
                    pbar.update(1)
                    continue

                buffer_pixels_x = int(np.ceil(buffer_x / voxel_size[0]))
                buffer_pixels_y = int(np.ceil(buffer_y / voxel_size[1]))

                if metric == "chm":
                    chm, extent = calculate_chm(tile_points, voxel_size, interpolation=interpolation)

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
                                f"Warning: Buffer size exceeds {metric.upper()} dimensions for tile ({i}, {j}). "
                                f"Adjusting buffer size."
                            )
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
