def _crop_polygon(polygon_wkt):
    """
    Generate a PDAL crop filter configuration for a given polygon.

    This function returns a dictionary that can be used as a stage in a PDAL pipeline
    to crop a point cloud using a polygon.

    Args:
        polygon_wkt (str): Polygon geometry in WKT format.

    Returns:
        dict: PDAL crop filter configuration dictionary.

    Example:
        >>> _crop_polygon("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
        {'type': 'filters.crop', 'polygon': 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'}
    """
    return {
        "type": "filters.crop",
        "polygon": polygon_wkt
    }


def _hag_delaunay():
    """
    Generate a PDAL Height Above Ground (HAG) Delaunay filter configuration.

    This function returns a dictionary that can be used as a stage in a PDAL pipeline
    to calculate Height Above Ground (HAG) using Delaunay triangulation.

    Returns:
        dict: PDAL HAG Delaunay filter configuration dictionary.

    Example:
        >>> _hag_delaunay()
        {'type': 'filters.hag_delaunay'}
    """
    return {
        "type": "filters.hag_delaunay"
    }


def _hag_raster(raster):
    """
    Generate a PDAL Height Above Ground (HAG) raster filter configuration.

    This function returns a dictionary that can be used as a stage in a PDAL pipeline
    to calculate Height Above Ground (HAG) using a given raster file.

    Args:
        raster (str): Path to the raster file for HAG calculation.

    Returns:
        dict: PDAL HAG raster filter configuration dictionary.

    Example:
        >>> _hag_raster("path/to/raster.tif")
        {'type': 'filters.hag_dem', 'raster': 'path/to/raster.tif'}
    """
    return {
        "type": "filters.hag_dem",
        "raster": raster
    }


def _filter_hag(lower_limit=0, upper_limit=None):
    """
    Generate a PDAL range filter configuration based on Height Above Ground (HAG) limits.

    Args:
        lower_limit (float, optional): Lower limit for HAG. Defaults to 0.
        upper_limit (float, optional): Upper limit for HAG. If None, there is no upper limit. Defaults to None.

    Returns:
        dict: PDAL range filter configuration dictionary.

    Example:
        >>> _filter_hag(1, 5)
        {'type': 'filters.range', 'limits': 'HeightAboveGround[1:5]'}
    """
    if upper_limit is None:
        limits = f"HeightAboveGround[{lower_limit}:]"
    else:
        limits = f"HeightAboveGround[{lower_limit}:{upper_limit}]"

    return {
        "type": "filters.range",
        "limits": limits
    }


def _filter_radius(radius):
    """
    Generate a PDAL sample filter configuration based on a given radius.

    Args:
        radius (float): The radius within which to sample points.

    Returns:
        dict: PDAL sample filter configuration dictionary.

    Example:
        >>> _filter_radius(1.5)
        {'type': 'filters.sample', 'radius': 1.5}
    """
    return {
        "type": "filters.sample",
        "radius": radius
    }
