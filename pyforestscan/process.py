import json
import pdal
import numpy as np
import os

from tqdm import tqdm

from pyforestscan.calculate import calculate_fhd, calculate_pad, calculate_pai, assign_voxels, calculate_chm, calculate_canopy_cover
from pyforestscan.filters import remove_outliers_and_clean, downsample_poisson
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


def process_with_tiles(ept_file, tile_size, output_path, metric, voxel_size,
                       voxel_height=1, buffer_size=0.1, srs=None, hag=False,
                       hag_dtm=False, dtm=None, bounds=None, interpolation=None, remove_outliers=False,
                       cover_min_height: float = 2.0, cover_k: float = 0.5,
                       skip_existing: bool = False, verbose: bool = False,
                       thin_radius: float | None = None) -> None:
    """
    Process a large EPT point cloud by tiling, compute CHM or other metrics for each tile,
    and write the results to the specified output directory.

    Args:
        ept_file (str): Path to the EPT file containing the point cloud data.
        tile_size (tuple): Size of each tile as (tile_width, tile_height).
        output_path (str): Directory where the output files will be saved.
        metric (str): Metric to compute for each tile ("chm", "fhd", "pai", or "cover").
        voxel_size (tuple): Voxel resolution as (x_resolution, y_resolution, z_resolution).
        voxel_height (float, optional): Height of each voxel in meters. Required if metric is "pai".
        buffer_size (float, optional): Fractional buffer size relative to tile size (e.g., 0.1 for 10% buffer). Defaults to 0.1.
        srs (str, optional): Spatial Reference System for the output. If None, uses SRS from the EPT file.
        hag (bool, optional): If True, compute Height Above Ground using Delaunay triangulation. Defaults to False.
        hag_dtm (bool, optional): If True, compute Height Above Ground using a provided DTM raster. Defaults to False.
        dtm (str, optional): Path to the DTM raster file. Required if hag_dtm is True.
        bounds (tuple, optional): Spatial bounds to crop the data. Must be of the form
            ([xmin, xmax], [ymin, ymax], [zmin, zmax]) or ([xmin, xmax], [ymin, ymax]).
            If None, tiling is done over the entire dataset.
        interpolation (str or None, optional): Interpolation method for CHM calculation ("linear", "cubic", "nearest", or None).
        remove_outliers (bool, optional): Whether to remove statistical outliers before calculating metrics. Defaults to False.
        cover_min_height (float, optional): Height threshold (in meters) for canopy cover (used when metric == "cover"). Defaults to 2.0.
        cover_k (float, optional): Beer–Lambert extinction coefficient for canopy cover. Defaults to 0.5.
        skip_existing (bool, optional): If True, skip tiles whose output file already exists. Defaults to False.
        verbose (bool, optional): If True, print warnings for empty/invalid tiles and buffer adjustments. Defaults to False.
        thin_radius (float or None, optional): If provided (> 0), apply Poisson radius-based thinning per tile before metrics.
            Units are in the same CRS as the data (e.g., meters). Defaults to None.

    Returns:
        None

    Raises:
        ValueError: If an unsupported metric is requested, if buffer or voxel sizes are invalid, or required arguments are missing.
        FileNotFoundError: If the EPT or DTM file does not exist, or a required file for processing is missing.
    """
    if metric not in ["chm", "fhd", "pai", "cover"]:
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
                # Apply buffer+crop for CHM and for PAI/COVER to avoid seam artifacts.
                if metric in ["chm", "pai", "cover"]:
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
                    if verbose:
                        print(f"Warning: Skipping tile ({i}, {j}) due to invalid spatial extent.")
                    pbar.update(1)
                    continue

                if metric == "chm":
                    result_file = os.path.join(output_path, f"tile_{i}_{j}_chm.tif")
                else:
                    result_file = os.path.join(output_path, f"tile_{i}_{j}_{metric}.tif")

                if skip_existing and os.path.isfile(result_file):
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

                # Extract points from pipeline output safely
                arrays = tile_pipeline.arrays if hasattr(tile_pipeline, "arrays") else []
                if not arrays or arrays[0].size == 0:
                    if verbose:
                        print(f"Warning: No data in tile ({i}, {j}). Skipping.")
                    pbar.update(1)
                    continue

                if remove_outliers:
                    tile_points = remove_outliers_and_clean(arrays)[0]
                else:
                    tile_points = arrays[0]

                # Optional radius-based thinning before metrics
                if thin_radius is not None and thin_radius > 0:
                    thinned = downsample_poisson([tile_points], thin_radius=thin_radius)
                    tile_points = thinned[0] if thinned else tile_points

                    if tile_points.size == 0:
                        if verbose:
                            print(f"Warning: Tile ({i}, {j}) empty after thinning. Skipping.")
                        pbar.update(1)
                        continue

                buffer_pixels_x = int(np.ceil(buffer_x / voxel_size[0]))
                buffer_pixels_y = int(np.ceil(buffer_y / voxel_size[1]))

                if metric == "chm":
                    chm, extent = calculate_chm(tile_points, voxel_size, interpolation=interpolation)

                    if buffer_pixels_x * 2 >= chm.shape[1] or buffer_pixels_y * 2 >= chm.shape[0]:
                        if verbose:
                            print(
                                f"Warning: Buffer size exceeds CHM dimensions for tile ({i}, {j}). Adjusting buffer size.")
                        buffer_pixels_x = max(0, chm.shape[1] // 2 - 1)
                        buffer_pixels_y = max(0, chm.shape[0] // 2 - 1)

                    # Safe crop: avoid Python's -0 slice behavior producing empty arrays.
                    if buffer_pixels_x < 0 or buffer_pixels_y < 0:
                        raise ValueError("Computed negative buffer pixels; check voxel and buffer sizes.")

                    start_x = buffer_pixels_x
                    end_x = chm.shape[1] - buffer_pixels_x if buffer_pixels_x > 0 else chm.shape[1]
                    start_y = buffer_pixels_y
                    end_y = chm.shape[0] - buffer_pixels_y if buffer_pixels_y > 0 else chm.shape[0]

                    # If cropping would yield an empty array due to small tiles, skip cropping
                    if end_x <= start_x or end_y <= start_y:
                        # Fall back to no buffer crop for this tile
                        core = chm
                    else:
                        core = chm[start_y:end_y, start_x:end_x]

                    chm = core

                    core_extent = (
                        tile_min_x + buffer_x,
                        tile_max_x - buffer_x,
                        tile_min_y + buffer_y,
                        tile_max_y - buffer_y,
                    )

                    create_geotiff(chm, result_file, srs, core_extent)
                elif metric in ["fhd", "pai", "cover"]:
                    voxels, spatial_extent = assign_voxels(tile_points, voxel_size)

                    if metric == "fhd":
                        result = calculate_fhd(voxels)
                    elif metric == "pai":
                        if not voxel_height:
                            raise ValueError(f"voxel_height is required for metric {metric}")

                        pad = calculate_pad(voxels, voxel_size[-1])

                        if np.all(pad == 0):
                            result = np.zeros((pad.shape[0], pad.shape[1]))
                        else:
                            # Guard against empty integration range when top height < default min_height
                            effective_max_height = pad.shape[2] * voxel_size[-1]
                            default_min_height = 1.0
                            if default_min_height >= effective_max_height:
                                result = np.zeros((pad.shape[0], pad.shape[1]))
                            else:
                                result = calculate_pai(pad, voxel_height)
                    elif metric == "cover":
                        if not voxel_height:
                            raise ValueError(f"voxel_height is required for metric {metric}")

                        pad = calculate_pad(voxels, voxel_size[-1])
                        if np.all(pad == 0):
                            result = np.zeros((pad.shape[0], pad.shape[1]))
                        else:
                            result = calculate_canopy_cover(
                                pad,
                                voxel_height=voxel_height,
                                min_height=cover_min_height,
                                max_height=None,
                                k=cover_k,
                            )

                    if current_buffer_size > 0:
                        if buffer_pixels_x * 2 >= result.shape[1] or buffer_pixels_y * 2 >= result.shape[0]:
                            if verbose:
                                print(
                                    f"Warning: Buffer size exceeds {metric.upper()} dimensions for tile ({i}, {j}). "
                                    f"Adjusting buffer size."
                                )
                            buffer_pixels_x = max(0, result.shape[1] // 2 - 1)
                            buffer_pixels_y = max(0, result.shape[0] // 2 - 1)

                        # Safe crop (avoid -0 slicing)
                        start_x = buffer_pixels_x
                        end_x = result.shape[1] - buffer_pixels_x if buffer_pixels_x > 0 else result.shape[1]
                        start_y = buffer_pixels_y
                        end_y = result.shape[0] - buffer_pixels_y if buffer_pixels_y > 0 else result.shape[0]

                        if end_x > start_x and end_y > start_y:
                            result = result[start_y:end_y, start_x:end_x]

                    core_extent = (
                        tile_min_x + buffer_x,
                        tile_max_x - buffer_x,
                        tile_min_y + buffer_y,
                        tile_max_y - buffer_y,
                    )

                    if core_extent[1] <= core_extent[0] or core_extent[3] <= core_extent[2]:
                        if verbose:
                            print(f"Warning: Invalid core extent for tile ({i}, {j}): {core_extent}. Skipping.")
                        pbar.update(1)
                        continue

                    create_geotiff(result, result_file, srs, core_extent)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")

                pbar.update(1)
