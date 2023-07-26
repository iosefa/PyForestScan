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
    Convert and unify the list of CRS to EPSG codes.
    """
    return [CRS(crs).to_epsg() for crs in crs_list]


def load_polygon_from_file(vector_file_path, index=0):
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
    if not os.path.isfile(dtm_path):
        raise FileNotFoundError(f"No such file: '{dtm_path}'")
    with rasterio.open(dtm_path) as dtm:
        return dtm.crs.to_string()


def validate_extensions(las_file_path, dtm_file_path):
    if not las_file_path.lower().endswith(('.las', '.laz')):
        raise ValueError("The point cloud file must be a .las or .laz file.")
    if not dtm_file_path.lower().endswith('.tif'):
        raise ValueError("The DTM file must be a .tif file.")


def _read_point_cloud(input_file, pipeline_stages=None):
    """
    Function to build and execute a PDAL pipeline.

    :param input_file: Path to the input file.
    :param pipeline_stages: List of dictionaries, each representing a pipeline stage.
    :param arrays: Optional. Numpy array(s) of point data to process.

    :return: The PDAL pipeline object.
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
    pipeline_def = {
        "pipeline": pipeline_stages
    }

    pipeline_json = json.dumps(pipeline_def)
    pipeline = pdal.Pipeline(pipeline_json, arrays=arrays)
    pipeline.execute()
    return pipeline


def validate_crs(crs_list):
    """
    Function to validate the Coordinate Reference System (CRS) of different sources of spatial data.
    This function checks if all CRS in the list match.
    If they do not match, it raises a ValueError.
    """
    simplified_crs_list = simplify_crs(crs_list)
    if not all(crs == simplified_crs_list[0] for crs in simplified_crs_list[1:]):
        raise ValueError("The CRS of the inputs do not match.")
    return True


def read_lidar(input_file, thin_radius=None, hag=False, hag_dtm=False, dtm=None, crop_poly=False, poly=None):
    """
    Function to load a point cloud file (.las or .laz) and perform optional processing steps.

    :param input_file: Path to the input .las or .laz file.
    :param thin_radius: Optional thinning radius. If specified, points will be sampled within this radius.
    :param hag: Optional flag to perform height above ground (HAG) filtering.
    :param hag_dtm: Optional flag to perform HAG filtering using a user-supplied DTM.
    :param dtm: Path to a user-supplied DTM file.
    :param crop_poly: Optional flag to crop the point cloud data using a user-supplied polygon.
    :param poly: Path to a user-supplied polygon file.

    :return: The first numpy array in the PDAL pipeline arrays.

    :raises ValueError: If more than one point cloud view is present, or if hag and hag_dtm are both True.
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"No such file: '{input_file}'")

    if not input_file.lower().endswith(('.las', '.laz')):
        raise ValueError("The input file must be a .las or .laz file.")

    if hag and hag_dtm:
        raise ValueError("Cannot use both 'hag' and 'hag_dtm' options at the same time.")

    pipeline_stages = []
    crs_list = []

    # Add cropping stage
    if crop_poly:
        if not os.path.isfile(poly):
            raise FileNotFoundError(f"No such file: '{poly}'")
        polygon_wkt, crs_vector = load_polygon_from_file(poly)
        crs_list.append(crs_vector)
        pipeline_stages.append(_crop_polygon(polygon_wkt))

    # Add thinning stage
    if thin_radius is not None:
        if thin_radius <= 0:
            raise ValueError("Thinning radius must be a positive number.")
        pipeline_stages.append(_filter_radius(thin_radius))

    # Add HAG filtering stage
    if hag:
        pipeline_stages.append(_hag_delaunay)

    # Add HAG_DTM filtering stage
    if hag_dtm:
        if not os.path.isfile(dtm):
            raise FileNotFoundError(f"No such file: '{dtm}'")
        if not dtm.lower().endswith('.tif'):
            raise ValueError("The DTM file must be a .tif file.")
        crs_raster = get_raster_epsg(dtm)
        crs_list.append(crs_raster)
        pipeline_stages.append(_hag_raster(dtm))

    # Get CRS from the point cloud data
    pipeline = _read_point_cloud(input_file, [{"type": "filters.info"}])
    crs_pointcloud = pipeline.metadata["metadata"]["readers.las"]["comp_spatialreference"]
    crs_list.append(crs_pointcloud)
    # Validate CRS
    validate_crs(crs_list)

    # Execute the actual processing pipeline
    pipeline = _read_point_cloud(input_file, pipeline_stages)

    return pipeline.arrays if pipeline.arrays else None


def write_las(arrays, output_file, compress=True):
    """
    Function to write the processed point clouds to a .las or .laz file.
    The output file will contain all the points from the input arrays.

    :param arrays: List of numpy arrays representing point clouds.
    :param output_file: Path to the output .las or .laz file.
    :param compress: If True, output will be written in the .laz format. If False, the .las format will be used.

    :raises ValueError: If the output file extension is not .las or .laz.
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

    # Initialize and execute the pipeline
    pipeline_json = json.dumps(pipeline_def)
    pipeline = pdal.Pipeline(pipeline_json, arrays=arrays)
    pipeline.execute()


def create_geotiff(layer, output_file, crs, spatial_extent):
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
