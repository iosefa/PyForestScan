from typing import List

from pyforestscan.handlers import _build_pdal_pipeline
from pyforestscan.pipeline import _filter_hag, _filter_ground, _filter_statistical_outlier, _filter_smrf, \
    _filter_radius, _select_ground


def filter_hag(arrays, lower_limit=0, upper_limit=None) -> List:
    """
    Apply a Height Above Ground (HAG) filter to a list of point cloud arrays.

    Filters each input array to include only points where Height Above Ground is
    within the specified range.

    Args:
        arrays (list): List of point cloud arrays to be processed.
        lower_limit (int or float, optional): Minimum Height Above Ground value to keep.
            Defaults to 0.
        upper_limit (int or float, optional): Maximum Height Above Ground value to keep.
            If None, no upper limit is applied. Defaults to None.

    Returns:
        list: List of processed point cloud arrays after applying the HAG filter.

    Raises:
        ValueError: If upper_limit is specified and is less than lower_limit.
    """
    pipeline = _build_pdal_pipeline(arrays, [_filter_hag(lower_limit, upper_limit)])
    return pipeline.arrays


def filter_ground(arrays) -> List:
    """
    Remove ground points (classification 2) from a list of point cloud arrays.

    Args:
        arrays (list): List of point cloud arrays to be processed.

    Returns:
        list: List of point cloud arrays after removing points classified as ground (classification 2).
    """
    pipeline = _build_pdal_pipeline(arrays, [_filter_ground()])
    return pipeline.arrays


def filter_select_ground(arrays) -> List:
    """
    Select only ground points (classification 2) from a list of point cloud arrays.

    Args:
        arrays (list): List of point cloud arrays to be processed.

    Returns:
        list: List of point cloud arrays containing only points classified as ground (classification 2).
    """
    pipeline = _build_pdal_pipeline(arrays, [_select_ground()])
    return pipeline.arrays


def remove_outliers_and_clean(arrays, mean_k=8, multiplier=3.0, remove=False) -> List:
    """
    Processes input arrays by removing statistical outliers and optionally cleans
    the data, filtering out specific classifications.
    By default, this function only labels outliers as "7" (outlier).

    # todo: this function will be renamed in a future release.

    Args:
        arrays (list): Input arrays to process, typically a list of structured arrays or
            similar data types (e.g., numpy structured arrays with a 'Classification' field).
        mean_k (int, optional): Number of neighbors to analyze for each point during
            statistical outlier removal. Defaults to 8.
        multiplier (float, optional): Multiplication factor for the standard deviation
            threshold to classify a point as an outlier. Defaults to 3.0.
        remove (bool, optional): If True, remove points classified as outliers (Classification == 7)
            after labeling. If False, only label outliers. Defaults to False.

    Returns:
        list: List of processed arrays after outlier removal and, if specified, cleaning.

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
) -> List:
    """
    Apply the SMRF (Simple Morphological Filter) to classify ground points in the point cloud arrays.

    Args:
        arrays (list): Cleaned point cloud arrays after outlier removal, typically as a list
            of structured arrays.
        ignore_class (str, optional): Classification codes to ignore during filtering,
            e.g., "Classification[7:7]". Defaults to "Classification[7:7]".
        cell (float, optional): Cell size in meters for the morphological filter grid. Defaults to 1.0.
        cut (float, optional): Cut net size; if set to 0, net cutting is skipped. Defaults to 0.0.
        returns (str, optional): Return types to include in output. Supported values include
            "first", "last", "intermediate", "only". Defaults to "last,only".
        scalar (float, optional): Elevation scalar for filter sensitivity. Defaults to 1.25.
        slope (float, optional): Slope threshold for ground classification. Defaults to 0.15.
        threshold (float, optional): Elevation threshold for morphological operations. Defaults to 0.5.
        window (float, optional): Maximum window size in meters for filter application. Defaults to 18.0.

    Returns:
        list: Point cloud arrays with classified ground points.

    Raises:
        RuntimeError: If there is an error during pipeline execution.
        ValueError: If no data is returned after SMRF classification.
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


def downsample_poisson(arrays, thin_radius) -> List:
    """
    Downsample point cloud arrays using Poisson (radius-based) thinning.

    This function applies a radius-based filter to the input point cloud arrays,
    reducing the point density so that no two points are closer than the specified radius.

    Args:
        arrays (list): List of point cloud arrays to be downsampled.
        thin_radius (float): Minimum allowed distance (radius) between retained points, in the same units as the point cloud coordinates.

    Returns:
        list: List of downsampled point cloud arrays.
    """
    pipeline = _build_pdal_pipeline(arrays, [_filter_radius(thin_radius)])
    return pipeline.arrays
