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

    :param crs_list: List of CRS definitions to be simplified.
    :type crs_list: list
    :return: List of EPSG codes corresponding to the input CRS definitions.
    :rtype: list
    :raises ValueError: If any of the CRS definitions cannot be converted to an EPSG code.
    """
    return [CRS(crs).to_epsg() for crs in crs_list]


def load_polygon_from_file(vector_file_path, index=0):
    """
    Load a polygon geometry and its CRS from a given vector file.

    :param vector_file_path: str, Path to the vector file containing the polygon.
    :param index: int, optional, Index of the polygon to be loaded (default is 0).
    :return: tuple, containing the Well-Known Text (WKT) representation of the polygon and the coordinate reference system (CRS) as a string.
    :raises FileNotFoundError: If the vector file does not exist.
    :raises ValueError: If the file cannot be read or is not a valid vector file format.
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

    :param dtm_path: str
        The file path to the raster file.
    :return: str
        The EPSG code as a string.
    :raises FileNotFoundError:
        If the specified file does not exist.
    """
    if not os.path.isfile(dtm_path):
        raise FileNotFoundError(f"No such file: '{dtm_path}'")
    with rasterio.open(dtm_path) as dtm:
        return dtm.crs.to_string()


def validate_extensions(las_file_path, dtm_file_path):
    """
    Validates the extensions of the provided file paths to check if they match
    the required .las/.laz for point cloud files and .tif for DTM files.

    :param las_file_path: The file path of the point cloud file.
                            Supported extensions are .las and .laz.
    :type las_file_path: str
    :param dtm_file_path: The file path of the DTM (Digital Terrain Model) file.
                            Supported extension is .tif.
    :type dtm_file_path: str
    :raises ValueError: If the point cloud file does not have a .las or .laz extension.
    :raises ValueError: If the DTM file does not have a .tif extension.
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


def validate_crs(crs_list):
    """
    Validate that all CRS representations in the list are identical.

    :param crs_list: List of coordinate reference systems to validate.
    :type crs_list: list
    :return: True if all CRSs match.
    :rtype: bool
    :raises ValueError: If the CRSs do not match.
    """
    simplified_crs_list = simplify_crs(crs_list)
    if not all(crs == simplified_crs_list[0] for crs in simplified_crs_list[1:]):
        raise ValueError("The CRS of the inputs do not match.")
    return True


def read_lidar(input_file, srs, thin_radius=None, hag=False, hag_dtm=False, dtm=None, crop_poly=False, poly=None):
    """
    Reads and processes a LiDAR point cloud file using PDAL based on specified options.

    :param srs:
    :param input_file: str, The path to the input LiDAR file. Supported formats are .las, .laz, .copc, and .copc.laz.
    :param thin_radius: float, optional, The radius for thinning the point cloud. Must be a positive number.
    :param hag: bool, optional, If True, calculate Height Above Ground (HAG) using Delaunay triangulation.
    :param hag_dtm: bool, optional, If True, calculate Height Above Ground (HAG) using a DTM file.
    :param dtm: str, optional, The path to the DTM file used when hag_dtm is True. Must be a .tif file.
    :param crop_poly: bool, optional, If True, crop the point cloud using the polygon defined in the poly file.
    :param poly: str, optional, The path to the polygon file used for cropping.

    :return: numpy.ndarray, The processed point cloud data or None if no data is retrieved.

    :raises FileNotFoundError: If the input file, polygon file, or DTM file does not exist.
    :raises ValueError: If the input file extension is unsupported, thinning radius is non-positive, or
                        both 'hag' and 'hag_dtm' are True simultaneously, or the DTM file path is not
                        provided when 'hag_dtm' is True.
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"No such file: '{input_file}'")

    las_extensions = ('.las', '.laz')
    copc_extensions = ('.copc', '.copc.laz')

    file_lower = input_file.lower()
    if file_lower.endswith(las_extensions):
        reader = 'readers.las'
    elif file_lower.endswith(copc_extensions):
        reader = 'readers.copc'
    else:
        raise ValueError("The input file must be a .las, .laz, .copc, or .copc.laz file.")

    if hag and hag_dtm:
        raise ValueError("Cannot use both 'hag' and 'hag_dtm' options at the same time.")

    pipeline_stages = []
    crs_list = []

    if crop_poly:
        if not poly or not os.path.isfile(poly):
            raise FileNotFoundError(f"No such polygon file: '{poly}'")
        polygon_wkt, crs_vector = load_polygon_from_file(poly)
        crs_list.append(crs_vector)
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

    metadata_pipeline = pdal.Pipeline(
        json.dumps({
            "pipeline": [
                {
                    "type": reader,
                    "spatialreference": srs, #todo: make srs optional
                    "filename": input_file
                },
                {
                    "type": "filters.info"
                }
            ]
        })
    )
    metadata_pipeline.execute()

    try:
        crs_pointcloud = metadata_pipeline.metadata['metadata']['readers.las']['spatialreference']
    except KeyError:
        try:
            crs_pointcloud = metadata_pipeline.metadata['metadata']['readers.copc']['spatialreference']
        except KeyError:
            raise ValueError("Unable to retrieve spatial reference from the point cloud metadata.")
    crs_list.append(crs_pointcloud)

    validate_crs(crs_list)

    main_pipeline_json = {
        "pipeline": [
            {
                "type": reader,
                "filename": input_file
            }
        ] + pipeline_stages
    }

    main_pipeline = pdal.Pipeline(json.dumps(main_pipeline_json))
    main_pipeline.execute()

    point_cloud = main_pipeline.arrays
    return point_cloud if point_cloud else None


def write_las(arrays, output_file, srs=None, compress=True):
    """
    Writes point cloud data to a LAS or LAZ file.

    :param arrays: The point cloud data arrays.
    :param output_file: The path of the output file.
    :param srs: Optional; Spatial Reference System to reproject the data.
    :param compress: Optional; Boolean flag to compress the output. Defaults to True.
    :raises ValueError: If 'compress' is True and output file extension is not .laz.
    :raises ValueError: If 'compress' is False and output file extension is not .las.

    :return: None
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
            "out_srs": srs
        })

    pipeline_steps.append({
        "type": output_format,
        "filename": output_file,
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


def create_geotiff(layer, output_file, crs, spatial_extent):
    """
    Creates a GeoTIFF file from the given data layer. Note, it performs a transpose on the layer.

    :param layer: The data layer to be written into the GeoTIFF file. Assumes (X, Y) shape.
    :type layer: numpy.ndarray
    :param output_file: The path where the GeoTIFF file will be saved.
    :type output_file: str
    :param crs: The coordinate reference system for the GeoTIFF.
    :type crs: str
    :param spatial_extent: The spatial extent of the data, defined as (x_min, x_max, y_min, y_max).
    :type spatial_extent: tuple
    :return: None
    :raises rasterio.errors.RasterioError: If there is an error in creating the GeoTIFF.
    """
    import rasterio
    from rasterio.transform import from_bounds

    x_min, x_max, y_min, y_max = spatial_extent

    layer = layer.T  # assumes (X, Y) shape

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
            transform=transform
    ) as new_dataset:
        new_dataset.write(layer, 1)
