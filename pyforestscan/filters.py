from pyforestscan.handlers import _build_pdal_pipeline
from pyforestscan.pipeline import _filter_hag, _filter_ground, _filter_statistical_outlier, _filter_smrf, \
    _filter_radius, _select_ground


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
    pipeline = _build_pdal_pipeline(arrays, [_filter_ground()])
    return pipeline.arrays


def filter_select_ground(arrays):
    """
    Applies a filter to select ground points (classification 2) from the input arrays.

    :param arrays: List of point cloud arrays to be processed.
    :type arrays: list
    :return: Processed point cloud arrays after removing ground points.
    :rtype: list
    """
    pipeline = _build_pdal_pipeline(arrays, [_select_ground()])
    return pipeline.arrays


def remove_outliers_and_clean(arrays, mean_k=8, multiplier=3.0, remove=False):
    """
    Processes input arrays by removing statistical outliers and optionally cleans
    the data, filtering out specific classifications.
    By default, this function only labels outliers as "7" (outlier).

    # todo: this function will be renamed in a future release.

    Args:
        arrays: The input arrays to process, typically a list of structured arrays or
            similar data types.
        mean_k (int, optional): The number of neighbors to analyze for each point for
            statistical outlier removal. Default is 8.
        multiplier (float, optional): The multiplication factor for the standard deviation
            threshold to classify an outlier. Default is 3.0.
        remove (bool, optional): If True, additional cleaning is performed to remove points
            specifically classified as outliers. Defaults to False.

    Returns:
        list: A list of processed arrays after outlier removal, and optionally after
        cleaning, if specified.

    Raises:
        RuntimeError: If the outlier removal pipeline fails during execution.
        ValueError: If no data is returned after applying outlier removal.
    """
    pipeline_stages = [
        _filter_statistical_outlier(mean_k=mean_k, multiplier=multiplier)
    ]

    try:
        pipeline = _build_pdal_pipeline(arrays, pipeline_stages)
    except RuntimeError as e:
        raise RuntimeError(f"Outlier Removal Pipeline Failed: {e}")

    if remove:
        clean_arrays = []
        for arr in pipeline.arrays:
            mask = (arr["Classification"] != 7)
            cleaned = arr[mask]
            clean_arrays.append(cleaned)
        processed_arrays = clean_arrays
    else:
        processed_arrays = pipeline.arrays

    if not processed_arrays:
        raise ValueError("No data returned after outlier removal.")

    return processed_arrays


def classify_ground_points(
        arrays,
        ignore_class="Classification[7:7]",
        cell=1.0,
        cut=0.0,
        returns="last,only",
        scalar=1.25,
        slope=0.15,
        threshold=0.5,
        window=18.0
):
    """
    Applies the SMRF filter to classify ground points in the point cloud.

    :param arrays: list
        Cleaned point cloud arrays after outlier removal.
    :type arrays: list
    :param ignore_class: str, classification codes to ignore during filtering.
    :param cell: float, cell size in meters.
    :param cut: float, cut net size (0 skips net cutting).
    :param returns: str, return types to include in output ("first", "last", "intermediate", "only").
    :param scalar: float, elevation scalar.
    :param slope: float, slope threshold for ground classification.
    :param threshold: float, elevation threshold.
    :param window: float, max window size in meters.
    :return: list
        Point cloud arrays with classified ground points.
    :rtype: list
    :raises RuntimeError:
        If there's an error during the pipeline execution.
    :raises ValueError:
        If no data is returned after SMRF classification.
    """
    pipeline_stages = [
        _filter_smrf(
            cell=cell,
            cut=cut,
            ignore=ignore_class,
            returns=returns,
            scalar=scalar,
            slope=slope,
            threshold=threshold,
            window=window
        )
    ]

    try:
        pipeline = _build_pdal_pipeline(arrays, pipeline_stages)
    except RuntimeError as e:
        raise RuntimeError(f"SMRF Classification Pipeline Failed: {e}")

    processed_arrays = pipeline.arrays

    if not processed_arrays:
        raise ValueError("No data returned after SMRF classification.")

    return processed_arrays


def downsample_poisson(arrays, thin_radius):
    pipeline = _build_pdal_pipeline(arrays, [_filter_radius(thin_radius)])
    return pipeline.arrays
