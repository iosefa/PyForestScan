from pyforestscan.handlers import _build_pdal_pipeline
from pyforestscan.pipeline import _filter_hag

def filter_hag(arrays, lower_limit=0, upper_limit=None):
    """Filter a point cloud based on Height Above Ground (HAG) limits.

    This function uses PDAL to filter a point cloud using the given lower and upper HAG limits.
    The function leverages a pipeline for efficient filtering.

    Args:
        arrays (list): A list of NumPy structured arrays representing the point cloud.
        lower_limit (float, optional): Lower limit for Height Above Ground. Defaults to 0.
        upper_limit (float, optional): Upper limit for Height Above Ground. If None, there is no upper limit. Defaults to None.

    Returns:
        list: A list of NumPy structured arrays representing the filtered point cloud.

    Example:
        >>> # Assume `original_arrays` is a list of NumPy structured arrays representing a point cloud.
        >>> filtered_arrays = filter_hag(original_arrays, lower_limit=1, upper_limit=5)

    """
    pipeline = _build_pdal_pipeline(arrays, [_filter_hag(lower_limit, upper_limit)])
    return pipeline.arrays
