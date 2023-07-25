import os
import json
import rasterio
import pdal
import geopandas as gpd

from shapely.geometry import MultiPolygon


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


def process_lidar_data(las_file_path, vector_file_path, dtm_file_path, index=0, radius=1):
    validate_extensions(las_file_path, dtm_file_path)

    polygon_wkt, vector_epsg = load_polygon_from_file(vector_file_path, index=index)
    dtm_epsg = get_raster_epsg(dtm_file_path)

    if vector_epsg != dtm_epsg:
        raise ValueError("The coordinate systems of the input files do not match.")
    return run_pdal_pipeline(las_file_path, polygon_wkt, dtm_file_path, radius)


def run_pdal_pipeline(file_path, polygon_wkt, dtm_path, radius):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")

    pipeline_def = {
        "pipeline": [
            file_path,
            {
                "type": "filters.crop",
                "polygon": polygon_wkt
            },
            {
                "type": "filters.sample",
                "radius": radius
            },
            {
                "type": "filters.hag_dem",
                "raster": dtm_path
            },
            {
                "type": "filters.range",
                "limits": "HeightAboveGround[0:]"
            }
        ]
    }

    pipeline_json = json.dumps(pipeline_def)
    pipeline = pdal.Pipeline(pipeline_json)
    pipeline.execute()
    return pipeline.arrays


def write_pdal_pipeline(arrays, output_file):
    """
    Function to write the processed point clouds to a .laz file.
    The output .laz file will contain all the points from the input arrays.
    """
    pipeline_def = {
        "pipeline": [
            {
                "type": "writers.las",
                "filename": output_file
            }
        ]
    }

    # Initialize and execute the pipeline
    pipeline_json = json.dumps(pipeline_def)
    pipeline = pdal.Pipeline(pipeline_json, arrays=arrays)
    pipeline.execute()


def create_geotiff(layer, output_file, crs, spatial_extent):
    transform = rasterio.transform.from_bounds(spatial_extent['xmin'], spatial_extent['ymin'],
                                               spatial_extent['xmax'], spatial_extent['ymax'],
                                               layer.shape[1], layer.shape[0])

    new_dataset = rasterio.open(output_file, 'w', driver='GTiff',
                                height=layer.shape[0], width=layer.shape[1],
                                count=1, dtype=str(layer.dtype),
                                crs=crs,
                                transform=transform)

    new_dataset.write(layer, 1)
    new_dataset.close()
