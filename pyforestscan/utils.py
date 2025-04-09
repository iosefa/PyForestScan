import json
import os
import math
import numpy as np
import requests

from pyforestscan.handlers import read_lidar, write_las



def _read_ept_json(ept_source):
    if any(ept_source.lower().startswith(proto) for proto in ["http://", "https://", "s3://"]):
        r = requests.get(ept_source)
        r.raise_for_status()
        ept_json = r.json()
    else:
        if not os.path.isfile(ept_source):
            raise FileNotFoundError(f"EPT JSON file not found at {ept_source}")
        with open(ept_source, "r") as f:
            ept_json = json.load(f)
    return ept_json


def get_srs_from_ept(ept_file):
    """
    Extracts the Spatial Reference System (SRS) from an EPT (Entwine Point Tile) file.

    This function reads the EPT JSON file and retrieves the SRS information, if available.
    The SRS is returned in the format '{authority}:{horizontal}'.

    :param ept_file: Path to the ept file containing the point cloud data.
    :return: The SRS string in the format '{authority}:{horizontal}' if available,
            otherwise None.
    """
    ept_json = _read_ept_json(ept_file)

    srs_obj = ept_json.get("srs", {})
    authority = srs_obj.get("authority", "")
    horizontal = srs_obj.get("horizontal", "")
    if authority and horizontal:
        return f"{authority}:{horizontal}"
    else:
        return None


def get_bounds_from_ept(ept_file):
    """
    Extracts the spatial bounds of a point cloud from an ept file using PDAL.

    :param ept_file: Path to the ept file containing the point cloud data.
    :return: A tuple with bounds in the format (min_x, max_x, min_y, max_y, min_z, max_z).
    """
    ept_json = _read_ept_json(ept_file)

    try:
        bounds = ept_json["bounds"]
        return bounds
    except KeyError:
        raise KeyError("Bounds information is not available in the ept metadata.")


def tile_las_in_memory(
        las_file,
        tile_width,
        tile_height,
        overlap,
        output_dir,
        srs=None
):
    """
    Reads the entire LAS file into memory, then subdivides it into tiles
    with a specified overlap on the right and bottom edges.

    :param las_file: Path to the source .las/.laz/.copc/.copc.laz file
    :param tile_width: Tile width in map units (meters if in UTM)
    :param tile_height: Tile height in map units
    :param overlap: Overlap (meters) to apply on the right & bottom edges
    :param output_dir: Directory where the tiled outputs will be written
    :param srs: Spatial reference for the input data (e.g., 'EPSG:32610').
                Pass None if not needed or if the file already has it.
    """
    arrays = read_lidar(
        input_file=las_file,
        srs=srs,
        bounds=None,
        thin_radius=None,
        hag=False,
        hag_dtm=False,
        dtm=None,
        crop_poly=False,
        poly=None
    )
    if not arrays or len(arrays) == 0 or arrays[0].size == 0:
        print(f"No data found in {las_file}. Exiting.")
        return

    big_cloud = arrays[0]

    min_x = np.min(big_cloud['X'])
    max_x = np.max(big_cloud['X'])
    min_y = np.min(big_cloud['Y'])
    max_y = np.max(big_cloud['Y'])

    total_width = max_x - min_x
    total_height = max_y - min_y

    step_x = tile_width - overlap
    step_y = tile_height - overlap
    num_tiles_x = max(1, math.ceil(total_width / step_x))
    num_tiles_y = max(1, math.ceil(total_height / step_y))

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    tile_index = 0
    for i in range(num_tiles_x):
        tile_min_x = min_x + i * step_x
        tile_max_x = tile_min_x + tile_width
        if tile_max_x > max_x:
            tile_max_x = max_x
        if tile_min_x >= max_x:
            break
        for j in range(num_tiles_y):
            tile_min_y = min_y + j * step_y
            tile_max_y = tile_min_y + tile_height
            if tile_max_y > max_y:
                tile_max_y = max_y
            if tile_min_y >= max_y:
                break
            in_tile = (
                    (big_cloud['X'] >= tile_min_x) & (big_cloud['X'] < tile_max_x) &
                    (big_cloud['Y'] >= tile_min_y) & (big_cloud['Y'] < tile_max_y)
            )
            tile_points = big_cloud[in_tile]
            if tile_points.size == 0:
                continue

            tile_index += 1
            out_path = os.path.join(output_dir, f"tile_{tile_index}.las")

            write_las([tile_points], out_path, srs=srs, compress=False)
            print(f"Created tile: {out_path}")
