import os

import geopandas as gpd
import json
import pdal
import rasterio
import numpy as np

from pyproj import CRS
from rasterio.transform import from_bounds
from shapely.geometry import MultiPolygon
from typing import List, Tuple
from urllib.parse import urlparse

from pyforestscan.pipeline import _crop_polygon, _filter_radius, _hag_delaunay, _hag_raster
from pyproj.exceptions import CRSError


def _is_url(input_str):
    try:
        result = urlparse(input_str)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def simplify_crs(crs_list) -> List:
    """
    Convert a list of coordinate reference system (CRS) representations to their corresponding EPSG codes.

    Args:
        crs_list (list): List of CRS definitions to be simplified. Each element may be any format accepted by pyproj.CRS (e.g., WKT string, PROJ string, EPSG code, etc.).

    Returns:
        list: List of EPSG codes corresponding to the input CRS definitions.

    Raises:
        CRSError: If any of the CRS definitions cannot be converted to an EPSG code.
    """
    epsg_codes = []
    for crs in crs_list:
        try:
            crs_obj = CRS(crs)
            epsg = crs_obj.to_epsg()
            if epsg is None:
                raise CRSError(f"Cannot convert CRS '{crs}' to an EPSG code.")
            epsg_codes.append(epsg)
        except CRSError as e:
            raise CRSError(f"Error converting CRS '{crs}': {e}") from None
    return epsg_codes


def load_polygon_from_file(vector_file_path, index=0) -> Tuple[str, str]:
    """
    Load a polygon geometry and its CRS from a given vector file.

    Args:
        vector_file_path (str): Path to the vector file containing the polygon geometry.
        index (int, optional): Index of the polygon to load from the file. Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - str: Well-Known Text (WKT) representation of the selected polygon.
            - str: Coordinate reference system (CRS) of the vector file as a string.

    Raises:
        FileNotFoundError: If the specified vector file does not exist.
        ValueError: If the file cannot be read or is not a valid vector file format.
    """
    if not os.path.isfile(vector_file_path):
        raise FileNotFoundError(f"No such file: '{vector_file_path}'")

    try:
        gdf = gpd.read_file(vector_file_path)
    except Exception as e:
        raise ValueError(f"Unable to read file: {vector_file_path}. Ensure it is a valid vector file format.") from e

    polygon = gdf.loc[index, 'geometry']
    if isinstance(polygon, MultiPolygon):
        polygon = list(polygon.geoms)[0]
    return polygon.wkt, gdf.crs.to_string()


def get_raster_epsg(dtm_path) -> str:
    """
    Retrieve the EPSG code from a raster file.

    Args:
        dtm_path (str): File path to the raster file.

    Returns:
        str: The EPSG code or CRS string of the raster file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if not os.path.isfile(dtm_path):
        raise FileNotFoundError(f"No such file: '{dtm_path}'")
    with rasterio.open(dtm_path) as dtm:
        return dtm.crs.to_string()


def validate_extensions(las_file_path, dtm_file_path):
    """
    Validate that input file paths have the correct extensions for point cloud and DTM files.

    Args:
        las_file_path (str): File path of the point cloud file. Supported extensions are .las and .laz.
        dtm_file_path (str): File path of the DTM (Digital Terrain Model) file. Supported extension is .tif.

    Raises:
        ValueError: If the point cloud file does not have a .las or .laz extension.
        ValueError: If the DTM file does not have a .tif extension.
    """
    if not las_file_path.lower().endswith(('.las', '.laz')):
        raise ValueError("The point cloud file must be a .las or .laz file.")
    if not dtm_file_path.lower().endswith('.tif'):
        raise ValueError("The DTM file must be a .tif file.")


def _read_point_cloud(input_file, pipeline_stages=None):
    """
    Read a point cloud file using a PDAL pipeline.

    :param input_file: The path to the input file.
    :type input_file: str
    :param pipeline_stages: A list of pipeline stages to be applied. Defaults to None.
    :type pipeline_stages: list, optional
    :return: A PDAL pipeline object after execution.
    :rtype: pdal.Pipeline
    :raises FileNotFoundError: If the input file does not exist.
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"No such file: '{input_file}'")

    pipeline_def = {
        "pipeline": [input_file] + pipeline_stages
    }

    pipeline_json = json.dumps(pipeline_def)
    pipeline = pdal.Pipeline(pipeline_json)
    pipeline.execute()
    return pipeline


def _build_pdal_pipeline(arrays, pipeline_stages):
    """
    Builds and executes a PDAL pipeline with the given arrays and pipeline stages.

    :param arrays: list
        Data arrays to be processed by the pipeline.
    :param pipeline_stages: list
        List of stages defining the pipeline's operations and transformations.
    :return: pdal.Pipeline
        The executed PDAL pipeline.
    :raises pdal.PipelineException:
        If there's an error during the pipeline execution.
    """
    pipeline_def = {
        "pipeline": pipeline_stages
    }

    pipeline_json = json.dumps(pipeline_def)
    pipeline = pdal.Pipeline(pipeline_json, arrays=arrays)
    pipeline.execute()
    return pipeline


def validate_crs(crs_list) -> bool:
    """
    Validate that all CRS (Coordinate Reference System) representations in the list are identical.

    Args:
        crs_list (list): List of coordinate reference system definitions to validate.

    Returns:
        bool: True if all CRSs match.

    Raises:
        ValueError: If the CRSs do not match.
    """
    simplified_crs_list = simplify_crs(crs_list)
    if not all(crs == simplified_crs_list[0] for crs in simplified_crs_list[1:]):
        raise ValueError("The CRS of the inputs do not match.")
    return True


def read_lidar(input_file, srs, bounds=None, thin_radius=None,
               hag=False, hag_dtm=False, dtm=None, crop_poly=False,
               poly=None) -> np.ndarray or None:
    """
    Read and process a LiDAR point cloud file using PDAL with various options.

    Args:
        input_file (str): Path to the input LiDAR file. Supported formats: .las, .laz, .copc, .copc.laz, or ept.json.
        srs (str): Spatial Reference System (SRS) of the point cloud.
        bounds (tuple or list, optional): Bounds for cropping data (only applies to EPT format).
            Format: ([xmin, xmax], [ymin, ymax], [zmin, zmax]).
        thin_radius (float, optional): Radius for thinning the point cloud. Must be positive.
        hag (bool, optional): If True, calculate Height Above Ground (HAG) using Delaunay triangulation.
            Defaults to False.
        hag_dtm (bool, optional): If True, calculate Height Above Ground (HAG) using a DTM file.
            Defaults to False.
        dtm (str, optional): Path to the DTM (.tif) file, required if hag_dtm is True.
        crop_poly (bool, optional): If True, crop the point cloud using a polygon. Defaults to False.
        poly (str, optional): Path to a polygon file or the WKT string of the polygon geometry.

    Returns:
        np.ndarray or None: Processed point cloud data as a NumPy array, or None if no data is retrieved.

    Raises:
        FileNotFoundError: If the input file, DTM file, or polygon file does not exist.
        ValueError: If the input file extension is unsupported; thinning radius is non-positive;
            both 'hag' and 'hag_dtm' are True at the same time; or a required parameter (e.g., DTM for hag_dtm)
            is missing or invalid.
    """
    if not _is_url(input_file) and not os.path.isfile(input_file):
        raise FileNotFoundError(f"No such file: '{input_file}'")

    las_extensions = ('.las', '.laz')
    copc_extensions = ('.copc', '.copc.laz')
    ept_file = ('ept.json')

    file_lower = input_file.lower()
    if file_lower.endswith(las_extensions):
        reader = 'readers.las'
    elif file_lower.endswith(copc_extensions):
        reader = 'readers.copc'
    elif file_lower.endswith(ept_file):
        reader = 'readers.ept'
    else:
        raise ValueError(
            "The input file must be a .las, .laz, .copc, .copc.laz file, or an ept.json file."
        )

    if hag and hag_dtm:
        raise ValueError("Cannot use both 'hag' and 'hag_dtm' options at the same time.")

    pipeline_stages = []
    crs_list = []

    if crop_poly:
        if not poly:
            raise ValueError(f"Must provide a polygon or polygon wkt if cropping to a polygon.")
        if poly.strip().startswith(('POLYGON', 'MULTIPOLYGON')):
            polygon_wkt = poly
        else:
            if not poly or not os.path.isfile(poly):
                raise FileNotFoundError(f"No such polygon file: '{poly}'")
            polygon_wkt, crs_vector = load_polygon_from_file(poly)
        pipeline_stages.append(_crop_polygon(polygon_wkt))

    if thin_radius is not None:
        if thin_radius <= 0:
            raise ValueError("Thinning radius must be a positive number.")
        pipeline_stages.append(_filter_radius(thin_radius))

    if hag:
        pipeline_stages.append(_hag_delaunay())

    if hag_dtm:
        if not dtm:
            raise ValueError("DTM file path must be provided when 'hag_dtm' is True.")
        if not os.path.isfile(dtm):
            raise FileNotFoundError(f"No such DTM file: '{dtm}'")
        if not dtm.lower().endswith('.tif'):
            raise ValueError("The DTM file must be a .tif file.")
        crs_raster = get_raster_epsg(dtm)
        crs_list.append(crs_raster)
        pipeline_stages.append(_hag_raster(dtm))

    base_pipeline = {
        "type": reader,
        "spatialreference": srs,
        "filename": input_file
    }
    if bounds and reader == 'readers.ept':
        base_pipeline["bounds"] = f"{bounds}"
    main_pipeline_json = {
        "pipeline": [
            base_pipeline
        ] + pipeline_stages
    }

    main_pipeline = pdal.Pipeline(json.dumps(main_pipeline_json))
    main_pipeline.execute()

    point_cloud = main_pipeline.arrays
    return point_cloud if point_cloud else None


def write_las(arrays, output_file, srs=None, compress=True) -> None:
    """
    Write point cloud data to a LAS or LAZ file.

    Args:
        arrays (list or np.ndarray): Point cloud data arrays to write.
        output_file (str): Path for the output file. Must end with .las (if uncompressed) or .laz (if compressed).
        srs (str, optional): Spatial Reference System to reproject the data. If provided, reprojection is applied.
        compress (bool, optional): If True, write a compressed LAZ file (.laz). If False, write an uncompressed LAS file (.las). Defaults to True.

    Raises:
        ValueError: If 'compress' is True and the output file extension is not .laz.
        ValueError: If 'compress' is False and the output file extension is not .las.

    Returns:
        None
    """
    output_extension = os.path.splitext(output_file)[1].lower()

    if compress:
        if output_extension != '.laz':
            raise ValueError("If 'compress' is True, output file must have a .laz extension.")
        output_format = "writers.las"
    else:
        if output_extension != '.las':
            raise ValueError("If 'compress' is False, output file must have a .las extension.")
        output_format = "writers.las"

    pipeline_steps = []

    if srs:
        pipeline_steps.append({
            "type": "filters.reprojection",
            "in_srs": srs,
            "out_srs": srs
        })

    pipeline_steps.append({
        "type": output_format,
        "filename": output_file,
        "minor_version": "4",
        "extra_dims": "all"
    })

    pipeline_def = {
        "pipeline": pipeline_steps
    }

    pipeline_json = json.dumps(pipeline_def)

    if not isinstance(arrays, list):
        arrays = [arrays]

    pipeline = pdal.Pipeline(pipeline_json, arrays=arrays)
    pipeline.execute()


def create_geotiff(layer, output_file, crs, spatial_extent, nodata=-9999) -> None:
    """
    Create a GeoTIFF file from the given data layer.

    The function transposes the input layer before saving and writes it as a single-band GeoTIFF.

    Args:
        layer (np.ndarray): The data layer to write, assumed to have shape (X, Y).
        output_file (str): Path where the GeoTIFF file will be saved.
        crs (str): Coordinate Reference System for the GeoTIFF.
        spatial_extent (tuple): The spatial extent as (x_min, x_max, y_min, y_max).
        nodata (float or int, optional): Value to use for NoData areas. Defaults to -9999.

    Returns:
        None

    Raises:
        rasterio.errors.RasterioError: If there is an error creating the GeoTIFF.
        ValueError: If the layer has invalid dimensions or the spatial extent is invalid.
    """
    layer = np.nan_to_num(layer, nan=-9999)

    x_min, x_max, y_min, y_max = spatial_extent

    if layer.size == 0 or layer.shape[0] == 0 or layer.shape[1] == 0:
        raise ValueError(f"Invalid layer dimensions: {layer.shape}. Cannot create GeoTIFF.")

    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Invalid spatial extent: {spatial_extent}.")

    layer = layer.T

    transform = from_bounds(
        x_min, y_min, x_max, y_max,
        layer.shape[1], layer.shape[0]
    )

    with rasterio.open(
            output_file, 'w',
            driver='GTiff',
            height=layer.shape[0],
            width=layer.shape[1],
            count=1,
            dtype=layer.dtype.name,
            crs=crs,
            transform=transform,
            nodata=nodata
    ) as new_dataset:
        new_dataset.write(layer, 1)
