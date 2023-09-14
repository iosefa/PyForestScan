import os
import json
import rasterio
import pdal
import geopandas as gpd

from pyproj import CRS
from shapely.geometry import MultiPolygon

from pyforestscan.pipeline import _crop_polygon, _filter_radius, _hag_delaunay, _hag_raster


def simplify_crs(crs_list):
    """
    Converts a list of CRS representations to their corresponding EPSG codes.

    Args:
        crs_list (list): List of Coordinate Reference Systems to be simplified.

    Returns:
        list: A list of simplified EPSG codes.

    Example:
        >>> simplify_crs(["EPSG:4326", "WGS84"])
        [4326, 4326]
    """
    return [CRS(crs).to_epsg() for crs in crs_list]


def load_polygon_from_file(vector_file_path, index=0):
    """
    Load a polygon geometry and its CRS from a given vector file.

    Args:
        vector_file_path (str): Path to the vector file.
        index (int, optional): The index of the geometry to be extracted. Defaults to 0.

    Returns:
        tuple: A tuple containing the WKT representation of the geometry and the CRS.

    Raises:
        FileNotFoundError: If the given vector file does not exist.
        ValueError: If the file format is not supported.

    Example:
        >>> load_polygon_from_file("path/to/file.shp")
        ('POLYGON((...))', 'EPSG:4326')
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


def get_raster_epsg(dtm_path):
    """
    Retrieve the EPSG code from a raster file.

    Args:
        dtm_path (str): Path to the raster file.

    Returns:
        str: The EPSG code of the raster file.

    Raises:
        FileNotFoundError: If the given raster file does not exist.

    Example:
        >>> get_raster_epsg("path/to/dtm.tif")
        'EPSG:4326'
    """
    if not os.path.isfile(dtm_path):
        raise FileNotFoundError(f"No such file: '{dtm_path}'")
    with rasterio.open(dtm_path) as dtm:
        return dtm.crs.to_string()


def validate_extensions(las_file_path, dtm_file_path):
    """
    Validate the file extensions for a point cloud and a DTM file.

    Args:
        las_file_path (str): Path to the point cloud file.
        dtm_file_path (str): Path to the DTM file.

    Raises:
        ValueError: If the file extension is not valid.

    Example:
        >>> validate_extensions("pointcloud.las", "dtm.tif")
    """
    if not las_file_path.lower().endswith(('.las', '.laz')):
        raise ValueError("The point cloud file must be a .las or .laz file.")
    if not dtm_file_path.lower().endswith('.tif'):
        raise ValueError("The DTM file must be a .tif file.")


def _read_point_cloud(input_file, pipeline_stages=None):
    """
    Read a point cloud file using a PDAL pipeline.

    Args:
        input_file (str): Path to the point cloud file.
        pipeline_stages (list, optional): Additional PDAL pipeline stages to be appended.

    Returns:
        pdal.Pipeline: PDAL Pipeline object containing the point cloud data.

    Raises:
        FileNotFoundError: If the given point cloud file does not exist.

    Example:
        >>> _read_point_cloud("path/to/pointcloud.las", [{"type": "filters.sort", "dimension": "Z"}])
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
    Build a PDAL pipeline for array data.

    Args:
        arrays (list): List of NumPy arrays containing point cloud data.
        pipeline_stages (list): List of PDAL pipeline stages to be appended.

    Returns:
        pdal.Pipeline: PDAL Pipeline object containing the point cloud data.

    Example:
        >>> _build_pdal_pipeline([array1, array2], [{"type": "filters.merge"}])
    """
    pipeline_def = {
        "pipeline": pipeline_stages
    }

    pipeline_json = json.dumps(pipeline_def)
    pipeline = pdal.Pipeline(pipeline_json, arrays=arrays)
    pipeline.execute()
    return pipeline


def validate_crs(crs_list):
    """
    Validate that all CRS representations in the list are identical.

    Args:
        crs_list (list): List of Coordinate Reference Systems to be compared.

    Returns:
        bool: True if all CRS are identical, False otherwise.

    Raises:
        ValueError: If the CRS do not match.

    Example:
        >>> validate_crs(["EPSG:4326", "WGS84"])
        True
    """
    simplified_crs_list = simplify_crs(crs_list)
    if not all(crs == simplified_crs_list[0] for crs in simplified_crs_list[1:]):
        raise ValueError("The CRS of the inputs do not match.")
    return True


def read_lidar(input_file, thin_radius=None, hag=False, hag_dtm=False, dtm=None, crop_poly=False, poly=None):
    """
    Read LIDAR data and perform various preprocessing operations.

    Args:
        input_file (str): Path to the LIDAR file.
        thin_radius (float, optional): Radius for thinning filter. Defaults to None.
        hag (bool, optional): Whether to calculate Height Above Ground (HAG) using Delaunay triangulation. Defaults to False.
        hag_dtm (bool, optional): Whether to calculate HAG using a raster DTM. Defaults to False.
        dtm (str, optional): Path to the DTM file for HAG calculation. Defaults to None.
        crop_poly (bool, optional): Whether to crop the point cloud using a polygon. Defaults to False.
        poly (str, optional): Path to the polygon file for cropping. Defaults to None.

    Returns:
        list: List of NumPy arrays containing the processed point cloud data.

    Raises:
        FileNotFoundError: If the given file does not exist.
        ValueError: For various types of invalid input.

    Example:
        >>> read_lidar("path/to/lidar.las", thin_radius=1.5, hag=True)
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"No such file: '{input_file}'")

    if not input_file.lower().endswith(('.las', '.laz')):
        raise ValueError("The input file must be a .las or .laz file.")

    if hag and hag_dtm:
        raise ValueError("Cannot use both 'hag' and 'hag_dtm' options at the same time.")

    pipeline_stages = []
    crs_list = []

    if crop_poly:
        if not os.path.isfile(poly):
            raise FileNotFoundError(f"No such file: '{poly}'")
        polygon_wkt, crs_vector = load_polygon_from_file(poly)
        crs_list.append(crs_vector)
        pipeline_stages.append(_crop_polygon(polygon_wkt))

    if thin_radius is not None:
        if thin_radius <= 0:
            raise ValueError("Thinning radius must be a positive number.")
        pipeline_stages.append(_filter_radius(thin_radius))

    if hag:
        pipeline_stages.append(_hag_delaunay)

    if hag_dtm:
        if not os.path.isfile(dtm):
            raise FileNotFoundError(f"No such file: '{dtm}'")
        if not dtm.lower().endswith('.tif'):
            raise ValueError("The DTM file must be a .tif file.")
        crs_raster = get_raster_epsg(dtm)
        crs_list.append(crs_raster)
        pipeline_stages.append(_hag_raster(dtm))

    pipeline = _read_point_cloud(input_file, [{"type": "filters.info"}])
    crs_pointcloud = pipeline.metadata["metadata"]["readers.las"]["comp_spatialreference"]
    crs_list.append(crs_pointcloud)
    # Validate CRS
    validate_crs(crs_list)

    pipeline = _read_point_cloud(input_file, pipeline_stages)

    return pipeline.arrays if pipeline.arrays else None


def write_las(arrays, output_file, compress=True):
    """
    Write point cloud data to a LAS or LAZ file.

    Args:
        arrays (list): List of NumPy arrays containing point cloud data.
        output_file (str): Path to the output file.
        compress (bool, optional): Whether to compress the output (LAZ format). Defaults to True.

    Raises:
        ValueError: If the file extension does not match the compression option.

    Example:
        >>> write_las([array1, array2], "path/to/output.las", compress=False)
    """
    output_extension = os.path.splitext(output_file)[1].lower()

    if compress:
        if output_extension != '.laz':
            raise ValueError("If 'compress' is True, output file must have a .laz extension.")
        output_format = "writers.laz"
    else:
        if output_extension != '.las':
            raise ValueError("If 'compress' is False, output file must have a .las extension.")
        output_format = "writers.las"

    pipeline_def = {
        "pipeline": [
            {
                "type": output_format,
                "filename": output_file
            }
        ]
    }

    pipeline_json = json.dumps(pipeline_def)
    pipeline = pdal.Pipeline(pipeline_json, arrays=arrays)
    pipeline.execute()


def create_geotiff(layer, output_file, crs, spatial_extent):
    """
    Create a GeoTIFF file from a 2D NumPy array.

    Args:
        layer (numpy.ndarray): 2D NumPy array containing raster data.
        output_file (str): Path to the output GeoTIFF file.
        crs (str): Coordinate Reference System for the output file.
        spatial_extent (tuple): Tuple containing the spatial extents (minx, miny, maxx, maxy).

    Example:
        >>> create_geotiff(np.array([[1, 2], [3, 4]]), "path/to/output.tif", "EPSG:4326", (0, 0, 1, 1))
    """
    transform = rasterio.transform.from_bounds(spatial_extent[0], spatial_extent[2],
                                               spatial_extent[1], spatial_extent[3],
                                               layer.shape[1], layer.shape[0])

    new_dataset = rasterio.open(output_file, 'w', driver='GTiff',
                                height=layer.shape[0], width=layer.shape[1],
                                count=1, dtype=str(layer.dtype),
                                crs=crs,
                                transform=transform)

    new_dataset.write(layer, 1)
    new_dataset.close()
