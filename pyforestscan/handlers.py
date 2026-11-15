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

from pyforestscan.pipeline import _crop_polygon, _filter_radius, _hag_delaunay, _hag_raster, _reproject
from pyproj.exceptions import CRSError
from pyforestscan.tiledb_io import (
    _tiledb_reader, convert_to_tiledb, get_tiledb_info, write_tiledb,
)


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


def _resolve_source_type(input_file, source_type=None):
    """Resolve explicit input formats or the existing filename conventions."""
    if source_type is not None:
        if not isinstance(source_type, str) or source_type.lower() not in ("las", "copc", "ept", "tiledb"):
            raise ValueError("source_type must be 'las', 'copc', 'ept', or 'tiledb'")
        return source_type.lower()
    filename = os.fspath(input_file).lower()
    # Retain the existing LAS reader for .laz names, including .copc.laz.
    # An explicit source_type="copc" selects its indexed reader and bounds.
    if filename.endswith(('.las', '.laz')):
        return "las"
    if filename.endswith('.copc'):
        return "copc"
    if filename.endswith('ept.json'):
        return "ept"
    raise ValueError(
        "The input file must be a .las, .laz, .copc, .copc.laz file, or an ept.json file. "
        "For a TileDB array, pass source_type='tiledb'."
    )


def _reader_stage(input_file, *, source_type=None, bounds=None, srs=None, tiledb_config=None):
    """Build a native-coordinate reader; TileDB queries may have no overlap."""
    input_file = os.fspath(input_file)
    source_type = _resolve_source_type(input_file, source_type)
    if source_type == "tiledb":
        reader = _tiledb_reader(input_file, bounds=bounds, config_file=tiledb_config)
        if reader is None:
            return None
    else:
        if not _is_url(input_file) and not os.path.isfile(input_file):
            raise FileNotFoundError(f"No such file: '{input_file}'")
        reader = {"type": f"readers.{source_type}", "filename": input_file}
        if bounds is not None and source_type in ("ept", "copc"):
            reader["bounds"] = str(bounds)
    if srs is not None:
        reader.pop("default_srs", None)
        reader["spatialreference"] = srs
    return reader


def read_lidar(input_file, srs=None, bounds=None, thin_radius=None,
               hag=False, hag_dtm=False, dtm=None, crop_poly=False,
               poly=None, reproject=False, *, source_type=None,
               tiledb_config=None) -> list[np.ndarray] | None:
    """
    Read and process a LiDAR point cloud file using PDAL with various options.

    Args:
        input_file (str): Point-cloud filename or TileDB array URI. File formats
            are inferred from .las, .laz, .copc, .copc.laz, or ept.json. TileDB
            arrays require source_type="tiledb" and may be local directories or
            remote URIs supported by TileDB, such as s3://bucket/array.
        srs (str, optional): Override the input CRS without changing coordinates,
            or the target CRS when reproject=True. Defaults to the source CRS,
            including CRS metadata stored in a TileDB array.
        bounds (tuple or list, optional): Native-coordinate bounds for EPT,
            COPC, or TileDB reads, applied before any reprojection. Format:
            ([xmin, xmax], [ymin, ymax]) with an optional [zmin, zmax].
            LAS/LAZ reads ignore this option.
            For indexed .copc.laz reads, explicitly pass source_type="copc".
        thin_radius (float, optional): Radius for thinning the point cloud. Must be positive.
        hag (bool, optional): If True, calculate Height Above Ground (HAG) using Delaunay triangulation.
            Defaults to False.
        hag_dtm (bool, optional): If True, calculate Height Above Ground (HAG) using a DTM file.
            Defaults to False.
        dtm (str, optional): Path to the DTM (.tif) file, required if hag_dtm is True.
        crop_poly (bool, optional): If True, crop the point cloud using a polygon. Defaults to False.
        poly (str, optional): Path to a polygon file or the WKT string of the polygon geometry.
        reproject (bool, optional): If True, reproject the point cloud to the CRS
            defined by srs, which is required in this case. Defaults to False.
        source_type (str, optional): Explicit reader: "las", "copc", "ept", or
            "tiledb". Defaults to filename inference for the file formats;
            .copc.laz uses the LAS reader by default.
        tiledb_config (str, optional): TileDB configuration file, passed to
            metadata inspection and PDAL reads. Useful for S3 settings and
            credentials; credentials are not written to array metadata.

    Returns:
        list[np.ndarray] or None: PDAL point-view arrays, or None if no points
            remain after the query and filters.

    Raises:
        FileNotFoundError: If the input file, DTM file, or polygon file does not exist.
        ValueError: If the input file extension is unsupported; thinning radius is non-positive;
            both 'hag' and 'hag_dtm' are True at the same time; or a required parameter (e.g., DTM for hag_dtm)
            is missing or invalid.
    """
    if reproject and not srs:
        raise ValueError("srs is required when reproject=True")
    base_pipeline = _reader_stage(
        input_file, source_type=source_type, bounds=bounds,
        srs=None if reproject else srs, tiledb_config=tiledb_config,
    )

    if hag and hag_dtm:
        raise ValueError("Cannot use both 'hag' and 'hag_dtm' options at the same time.")

    pipeline_stages = []
    tiledb_source = _resolve_source_type(input_file, source_type) == "tiledb"
    native_srs = None if base_pipeline is None else (
        base_pipeline.get("spatialreference") or base_pipeline.get("default_srs")
    )

    if crop_poly:
        if not poly:
            raise ValueError(f"Must provide a polygon or polygon wkt if cropping to a polygon.")
        if poly.strip().startswith(('POLYGON', 'MULTIPOLYGON')):
            polygon_wkt = poly
        else:
            if not poly or not os.path.isfile(poly):
                raise FileNotFoundError(f"No such polygon file: '{poly}'")
            polygon_wkt, crs_vector = load_polygon_from_file(poly)
            if tiledb_source and base_pipeline is not None:
                if not native_srs:
                    raise ValueError("TileDB source CRS is required to crop with a polygon file")
                if CRS(native_srs) != CRS(crs_vector):
                    raise ValueError("The polygon CRS must match the TileDB source CRS")
        pipeline_stages.append(_crop_polygon(polygon_wkt))

    if thin_radius is not None:
        if thin_radius <= 0:
            raise ValueError("Thinning radius must be a positive number.")
        pipeline_stages.append(_filter_radius(thin_radius))

    if reproject:
        pipeline_stages.append(_reproject(srs))

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
        if tiledb_source and base_pipeline is not None:
            current_srs = srs if reproject else native_srs
            if not current_srs:
                raise ValueError("TileDB source CRS is required to calculate HAG from a DTM")
            if CRS(current_srs) != CRS(crs_raster):
                raise ValueError("The DTM CRS must match the point-cloud CRS after reprojection")
        pipeline_stages.append(_hag_raster(dtm))

    if base_pipeline is None:
        return None
    main_pipeline_json = {
        "pipeline": [
            base_pipeline
        ] + pipeline_stages
    }

    main_pipeline = pdal.Pipeline(json.dumps(main_pipeline_json))
    main_pipeline.execute()

    point_cloud = [array for array in main_pipeline.arrays if array.size]
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
