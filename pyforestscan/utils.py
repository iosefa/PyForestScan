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


def get_srs_from_ept(ept_file) -> str or None:
    """
    Extract the Spatial Reference System (SRS) from an EPT (Entwine Point Tile) file.

    This function reads the EPT JSON file and retrieves the SRS information, if available.
    The SRS is returned as a string in the format '{authority}:{horizontal}'.

    Args:
        ept_file (str): Path to the EPT file containing the point cloud data.

    Returns:
        str or None: The SRS string in the format '{authority}:{horizontal}' if available,
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


def get_bounds_from_ept(ept_file) -> tuple[float, float, float, float, float, float]:
    """
    Extract dataset bounds from an EPT (Entwine Point Tile) source.

    Args:
        ept_file (str): Path or URL to the EPT JSON.

    Returns:
        tuple: (min_x, max_x, min_y, max_y, min_z, max_z)

    Raises:
        KeyError: If bounds information is not available in the EPT metadata.
        ValueError: If the EPT bounds are malformed.
    """
    ept_json = _read_ept_json(ept_file)

    # Prefer "bounds"; some datasets may also include "boundsConforming".
    raw_bounds = ept_json.get("bounds")
    if raw_bounds is None:
        raw_bounds = ept_json.get("boundsConforming")
    if raw_bounds is None:
        raise KeyError("Bounds information is not available in the ept metadata.")

    # Handle common representations: list/tuple of 6 numbers or a dict with keys.
    if isinstance(raw_bounds, (list, tuple)) and len(raw_bounds) == 6:
        min_x, min_y, min_z, max_x, max_y, max_z = raw_bounds
    elif isinstance(raw_bounds, dict):
        try:
            min_x = raw_bounds["minx"]; min_y = raw_bounds["miny"]; min_z = raw_bounds["minz"]
            max_x = raw_bounds["maxx"]; max_y = raw_bounds["maxy"]; max_z = raw_bounds["maxz"]
        except Exception as e:
            raise ValueError(f"Malformed EPT bounds dictionary: {e}") from None
    else:
        raise ValueError(f"Unexpected EPT bounds format: {type(raw_bounds)}")

    return (min_x, max_x, min_y, max_y, min_z, max_z)


def tile_las_in_memory(
        las_file,
        tile_width,
        tile_height,
        overlap,
        output_dir,
        srs=None
) -> None:
    """
    Read an entire LAS/LAZ/COPC file into memory and subdivide it into tiles
    with a specified overlap on the right and bottom edges. Writes each tile as a new LAS file.

    Args:
        las_file (str): Path to the source .las, .laz, .copc, or .copc.laz file.
        tile_width (float): Tile width in map units (e.g., meters if in UTM).
        tile_height (float): Tile height in map units.
        overlap (float): Overlap (in map units) to apply to the right and bottom edges of each tile.
        output_dir (str): Directory where the tiled output files will be written.
        srs (str, optional): Spatial reference for the input data (e.g., 'EPSG:32610'). Pass None if not needed or if the file already has SRS.

    Returns:
        None

    Notes:
        - Tiles are written in uncompressed LAS format (.las).
        - Only non-empty tiles are written.
        - The function creates output_dir if it does not exist.
        - The input file is fully loaded into memory; not recommended for extremely large files.
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
