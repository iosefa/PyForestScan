from pyforestscan.handlers import _build_pdal_pipeline
from pyforestscan.pipeline import _filter_hag, _filter_ground


def filter_hag(arrays, lower_limit=0, upper_limit=None):
    """
    Applies a Height Above Ground (HAG) filter to the input arrays.

    :param arrays: List of point cloud arrays to be processed.
    :type arrays: list
    :param lower_limit: The minimum value for the height filter, defaults to 0.
    :type lower_limit: int, optional
    :param upper_limit: The maximum value for the height filter, defaults to None.
    :type upper_limit: int, optional
    :return: Processed point cloud arrays after applying the HAG filter.
    :rtype: list
    :raises ValueError: If upper_limit is less than lower_limit.
    """
    pipeline = _build_pdal_pipeline(arrays, [_filter_hag(lower_limit, upper_limit)])
    return pipeline.arrays


def filter_ground(arrays):
    """
    Applies a filter to remove ground points (classification 2) from the input arrays.

    :param arrays: List of point cloud arrays to be processed.
    :type arrays: list
    :return: Processed point cloud arrays after removing ground points.
    :rtype: list
    """
    # Build and execute the PDAL pipeline with the ground filter
    pipeline = _build_pdal_pipeline(arrays, [_filter_ground()])
    return pipeline.arrays
