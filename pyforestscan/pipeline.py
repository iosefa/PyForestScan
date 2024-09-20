def _crop_polygon(polygon_wkt):
    """
    Generate a PDAL crop filter configuration for a given polygon.

    This function returns a dictionary that can be used as a stage in a PDAL pipeline
    to crop a point cloud using a polygon.

    :param polygon_wkt: str. A Well-Known Text (WKT) string representing the polygon to be used for cropping.
    :return: dict. A dictionary with keys "type" and "polygon" where "type" is "filters.crop" and "polygon" is the input WKT string.
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
        dict: A dictionary with a single key "type" having the value
              "filters.hag_delaunay".
    """
    return {
        "type": "filters.hag_delaunay"
    }


def _hag_raster(raster):
    """
    Generate a PDAL Height Above Ground (HAG) raster filter configuration.

    This function returns a dictionary that can be used as a stage in a PDAL pipeline
    to calculate Height Above Ground (HAG) using a given raster file.

    :param raster: Path to the raster file.
    :type raster: str
    :return: A dictionary with the type of transformation and the raster file path.
    :rtype: dict
    """
    return {
        "type": "filters.hag_dem",
        "raster": raster
    }


def _filter_hag(lower_limit=0, upper_limit=None):
    """
    Generate a PDAL range filter configuration based on Height Above Ground (HAG) limits.

    :param lower_limit: int or float, optional
        The lower limit for the HAG filter. Defaults to 0.
    :param upper_limit: int or float, optional
        The upper limit for the HAG filter. Defaults to None.

    :return: dict
        A dictionary containing the filter type and limits string for HAG filtering.

    :raises ValueError:
        If lower_limit is greater than upper_limit when both are specified.
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
    todo: update: this should be described as random down sample. Uses Poisson sampling.
    Generates a filter configuration with a specified radius.

    :param radius:
        The radius to be used in the filter configuration.
    :type radius: int
    :return:
        A dictionary containing the filter type and radius.
    :rtype: dict
    """
    return {
        "type": "filters.sample",
        "radius": radius
    }


def _filter_ground():
    """
    Generate a PDAL classification filter to remove ground points (classification 2).

    :return: dict
        A dictionary containing the filter type and limits string for removing ground points.
    """
    return {
        "type": "filters.expression",
        "expression": "!(Classification == 2)"
    }


def _select_ground():
    """
    Generate a PDAL classification filter to select ground points (classification 2).

    :return: dict
        A dictionary containing the filter type and limits string for selecting ground points.
    """
    return {
        "type": "filters.range",
        "limits": "Classification[2:2]"
    }


def _filter_expression(expression="Classification == 2"):
    """
    Generate a PDAL expression filter configuration.

    :param expression: str, the logical expression to apply.
    :return: dict representing the PDAL expression filter.
    """
    return {
        "type": "filters.expression",
        "expression": expression
    }


def _filter_statistical_outlier(mean_k=8, multiplier=3.0):
    """
    Generate a PDAL statistical outlier filter configuration.

    :param mean_k: int, number of nearest neighbors to use for mean distance estimation.
    :param multiplier: float, standard deviation multiplier for identifying outliers.
    :return: dict representing the PDAL outlier filter.
    """
    return {
        "type": "filters.outlier",
        "method": "statistical",
        "mean_k": mean_k,
        "multiplier": multiplier
    }


def _filter_smrf(
        cell=1.0,
        cut=0.0,
        ignore="Classification[7:7]",
        returns="last,only",
        scalar=1.25,
        slope=0.15,
        threshold=0.5,
        window=18.0
):
    """
    Generate a PDAL SMRF filter configuration.

    :param cell: float, cell size in meters.
    :param cut: float, cut net size (0 skips net cutting).
    :param ignore: str, a range of values of a dimension to ignore.
    :param returns: str, return types to include in output ("first", "last", "intermediate", "only").
    :param scalar: float, elevation scalar.
    :param slope: float, slope threshold for ground classification.
    :param threshold: float, elevation threshold.
    :param window: float, max window size in meters.
    :return: dict representing the PDAL SMRF filter.
    """
    smrf_filter = {
        "type": "filters.smrf",
        "cell": cell,
        "cut": cut,
        "ignore": ignore,
        "returns": returns,
        "scalar": scalar,
        "slope": slope,
        "threshold": threshold,
        "window": window
    }

    return smrf_filter
