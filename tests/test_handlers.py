import os
import pytest
from pyproj.exceptions import CRSError

from pyforestscan.handlers import (
    simplify_crs,
    load_polygon_from_file,
    get_raster_epsg,
    validate_extensions,
    _read_point_cloud,
    _build_pdal_pipeline,
    validate_crs,
    read_lidar,
    write_las,
    create_geotiff
)
from pyproj import CRS
from shapely.geometry import Polygon, MultiPolygon
import rasterio
import pdal


def get_test_data_path(filename):
    return os.path.join(os.path.dirname(__file__), '..', 'test_data', filename)


# ----------------------------
# Tests for simplify_crs
# ----------------------------

def test_simplify_crs_valid_inputs():
    crs_list = ["EPSG:4326", "EPSG:3857"]
    expected = [4326, 3857]
    assert simplify_crs(crs_list) == expected


def test_simplify_crs_with_crs_objects():
    crs_list = [CRS.from_epsg(4326), CRS.from_epsg(3857)]
    expected = [4326, 3857]
    assert simplify_crs(crs_list) == expected


def test_simplify_crs_invalid_crs():
    crs_list = ["INVALID_CRS", "EPSG:999999"]
    with pytest.raises(CRSError):
        simplify_crs(crs_list)


# ----------------------------
# Tests for load_polygon_from_file
# ----------------------------

def test_load_polygon_from_file_success():
    vector_file = get_test_data_path("valid_polygon.gpkg")
    polygon_wkt, crs = load_polygon_from_file(vector_file)
    assert isinstance(polygon_wkt, str)
    assert crs == "EPSG:32605"


def test_load_polygon_from_file_file_not_found():
    vector_file = get_test_data_path("nonexistent.shp")
    with pytest.raises(FileNotFoundError):
        load_polygon_from_file(vector_file)


# ----------------------------
# Tests for get_raster_epsg
# ----------------------------

def test_get_raster_epsg_success():
    raster_file = get_test_data_path("20191210_5QKB020880_DS05_dtm.tif")
    epsg = get_raster_epsg(raster_file)
    assert isinstance(epsg, str)
    assert epsg == "EPSG:32605"


def test_get_raster_epsg_file_not_found():
    raster_file = get_test_data_path("nonexistent.tif")
    with pytest.raises(FileNotFoundError):
        get_raster_epsg(raster_file)


# ----------------------------
# Tests for validate_extensions
# ----------------------------

def test_validate_extensions_valid():
    las_file = get_test_data_path("20191210_5QKB020880_DS05.copc.laz")
    dtm_file = get_test_data_path("20191210_5QKB020880_DS05_dtm.tif")
    validate_extensions(las_file, dtm_file)


def test_validate_extensions_invalid_las():
    las_file = get_test_data_path("input.txt")
    dtm_file = get_test_data_path("20191210_5QKB020880_DS05_dtm.tif")
    with pytest.raises(ValueError, match="The point cloud file must be a .las or .laz file."):
        validate_extensions(las_file, dtm_file)


def test_validate_extensions_invalid_dtm():
    las_file = get_test_data_path("20191210_5QKB020880_DS05.copc.laz")
    dtm_file = get_test_data_path("input.txt")
    with pytest.raises(ValueError, match="The DTM file must be a .tif file."):
        validate_extensions(las_file, dtm_file)


# ----------------------------
# Tests for validate_crs
# ----------------------------

def test_validate_crs_all_match():
    crs_list = ["EPSG:4326", "EPSG:4326", "EPSG:4326"]
    assert validate_crs(crs_list) is True


def test_validate_crs_do_not_match():
    crs_list = ["EPSG:4326", "EPSG:3857", "EPSG:4326"]
    with pytest.raises(ValueError, match="The CRS of the inputs do not match."):
        validate_crs(crs_list)


def test_validate_crs_empty_list():
    crs_list = []
    assert validate_crs(crs_list) is True


# ----------------------------
# Tests for read_lidar
# ----------------------------

def test_read_lidar_success():
    input_file = get_test_data_path("20191210_5QKB020880_DS05.copc.laz")
    srs = "EPSG:32605"
    point_cloud = read_lidar(
        input_file=input_file,
        srs=srs
    )
    assert point_cloud is not None
    assert isinstance(point_cloud, list)
    assert len(point_cloud) > 0


def test_read_lidar_with_thinning():
    input_file = get_test_data_path("20191210_5QKB020880_DS05.copc.laz")
    srs = "EPSG:32605"
    thin_radius = 1.0
    point_cloud = read_lidar(
        input_file=input_file,
        srs=srs,
        thin_radius=thin_radius
    )
    assert point_cloud is not None
    assert isinstance(point_cloud, list)


def test_read_lidar_with_hag():
    input_file = get_test_data_path("20191210_5QKB020880_DS05.copc.laz")
    srs = "EPSG:32605"
    point_cloud = read_lidar(
        input_file=input_file,
        srs=srs,
        hag=True
    )
    assert point_cloud is not None
    assert isinstance(point_cloud, list)


def test_read_lidar_with_hag_dtm():
    input_file = get_test_data_path("20191210_5QKB020880_DS05.copc.laz")
    srs = "EPSG:32605"
    dtm_file = get_test_data_path("20191210_5QKB020880_DS05_dtm.tif")
    point_cloud = read_lidar(
        input_file=input_file,
        srs=srs,
        hag_dtm=True,
        dtm=dtm_file
    )
    assert point_cloud is not None
    assert isinstance(point_cloud, list)


def test_read_lidar_with_cropping():
    input_file = get_test_data_path("20191210_5QKB020880_DS05.copc.laz")
    srs = "EPSG:32605"
    poly_file = get_test_data_path("valid_polygon.gpkg")
    point_cloud = read_lidar(
        input_file=input_file,
        srs=srs,
        crop_poly=True,
        poly=poly_file
    )
    assert point_cloud is not None
    assert isinstance(point_cloud, list)


def test_read_lidar_hag_and_hag_dtm_simultaneously():
    input_file = get_test_data_path("20191210_5QKB020880_DS05.copc.laz")
    srs = "EPSG:32605"
    with pytest.raises(ValueError, match="Cannot use both 'hag' and 'hag_dtm' options at the same time."):
        read_lidar(
            input_file=input_file,
            srs=srs,
            hag=True,
            hag_dtm=True
        )


def test_read_lidar_missing_dtm_file():
    input_file = get_test_data_path("20191210_5QKB020880_DS05.copc.laz")
    srs = "EPSG:32605"
    dtm_file = get_test_data_path("nonexistent.tif")
    with pytest.raises(FileNotFoundError, match=f"No such DTM file: '{dtm_file}'"):
        read_lidar(
            input_file=input_file,
            srs=srs,
            hag_dtm=True,
            dtm=dtm_file
        )


def test_read_lidar_unsupported_input_extension():
    input_file = get_test_data_path("input.txt")
    srs = "EPSG:4326"
    with pytest.raises(ValueError, match="The input file must be a .las, .laz, .copc, .copc.laz file, or an ept.json file."):
        read_lidar(
            input_file=input_file,
            srs=srs
        )


# ----------------------------
# Tests for create_geotiff
# ----------------------------

def test_create_geotiff():
    output_file = get_test_data_path("output.tif")
    srs = "EPSG:32605"
    spatial_extent = (0, 10, 0, 10)
    import numpy as np
    layer = np.random.rand(100, 100).astype('float32')

    create_geotiff(
        layer=layer,
        output_file=output_file,
        crs=srs,
        spatial_extent=spatial_extent
    )
    assert os.path.isfile(output_file)
    with rasterio.open(output_file) as src:
        assert src.crs.to_string() == srs
        assert src.width == 100
        assert src.height == 100
        assert src.transform is not None
