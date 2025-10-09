import os
import pytest

from pyforestscan.utils import (
    get_srs_from_ept,
    get_bounds_from_ept,
    tile_las_in_memory
)


def get_test_data_path(filename):
    return os.path.join(os.path.dirname(__file__), '..', 'test_data', filename)


def test_get_srs_from_usgs_hawaii_ept():
    """
    Tests that we can retrieve the EPSG code ("EPSG:3857") from the Hawaii EPT metadata.
    This requires internet access to fetch the remote JSON.
    """
    ept_url = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/HI_Hawaii_Island_2017/ept.json"
    srs = get_srs_from_ept(ept_url)
    assert srs == "EPSG:3857"


def test_get_bounds_from_usgs_hawaii_ept():
    """
    Tests retrieving the EPT bounds from a real EPT JSON.
    get_bounds_from_ept returns (min_x, max_x, min_y, max_y, min_z, max_z).
    """
    ept_url = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/HI_Hawaii_Island_2017/ept.json"
    bounds = get_bounds_from_ept(ept_url)
    assert len(bounds) == 6, "Bounds should be (minX, maxX, minY, maxY, minZ, maxZ)."
    expected = (-17405012, -17229390, 2136822, 2312444, -5677, 169945)
    assert tuple(bounds) == expected

    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    assert min_x < max_x and min_y < max_y and min_z < max_z


def test_tile_las_in_memory():
    """
    Tests the tiling function using a real COPC LAZ file.
    This will read the entire file, tile it, and create multiple .las tiles in ../test_data/tiles.
    """
    input_las = get_test_data_path("20191210_5QKB020880_DS05.copc.laz")
    output_dir = "../test_data/tiles"

    if not os.path.exists(input_las):
        pytest.skip(f"Input LAS file not found at {input_las}. Test skipped.")

    os.makedirs(output_dir, exist_ok=True)

    tile_las_in_memory(
        las_file=input_las,
        tile_width=50,
        tile_height=50,
        overlap=10,
        output_dir=output_dir,
        srs="EPSG:32605"
    )

    created_tiles = [
        f for f in os.listdir(output_dir)
        if f.endswith(".las") or f.endswith(".laz")
    ]
    assert len(created_tiles) > 0, "No tiles were created. Possibly no points or an error occurred."
