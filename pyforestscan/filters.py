from typing import List
import math
import os
import numpy as np

from pyforestscan.handlers import _build_pdal_pipeline
from pyforestscan.pipeline import (
    _filter_hag,
    _filter_ground,
    _filter_pointsourceid,
    _filter_radius,
    _filter_smrf,
    _filter_statistical_outlier,
    _filter_voxeldownsize,
    _hag_delaunay,
    _hag_raster,
    _select_ground,
)


def add_height_above_ground(existing_points, method=None, dtm=None) -> List:
    """
    Add HeightAboveGround values to in-memory point cloud arrays.

    This applies a PDAL Height Above Ground filter directly to existing
    NumPy structured arrays. By default, the Delaunay method uses points
    classified as ground (Classification == 2) as the terrain surface. When
    ``dtm`` is provided, the DTM raster method is used unless ``method`` is
    explicitly set.

    Args:
        existing_points (np.ndarray or list): A single structured point cloud
            array, or a list/tuple of arrays, containing at least 'X', 'Y', 'Z',
            and, for the Delaunay method, 'Classification' fields.
        method (str, optional): Height Above Ground method. Supported values
            are 'delaunay' and 'dtm' (also accepts 'raster'). If None, this
            defaults to 'dtm' when a ``dtm`` path is provided, otherwise
            'delaunay'.
        dtm (str or path-like, optional): Path to a DTM GeoTIFF for the DTM
            method.

    Returns:
        list: Point cloud arrays with a 'HeightAboveGround' dimension.

    Raises:
        TypeError: If the input is not a structured NumPy array or iterable of
            structured NumPy arrays.
        ValueError: If required dimensions are missing, no points are supplied,
            an unsupported method is requested, or no ground points
            (Classification == 2) are present for the Delaunay method.
        FileNotFoundError: If the requested DTM file does not exist.
        RuntimeError: If the PDAL HAG pipeline fails.
    """
    if method is None:
        method = "dtm" if dtm is not None else "delaunay"
    elif not isinstance(method, str):
        raise TypeError("method must be 'delaunay', 'dtm', or 'raster'.")

    method_lc = method.lower()
    if method_lc == "raster":
        method_lc = "dtm"
    if method_lc not in {"delaunay", "dtm"}:
        raise ValueError("method must be one of: 'delaunay', 'dtm', or 'raster'.")

    if method_lc == "dtm":
        if dtm is None:
            raise ValueError("dtm must be provided when method is 'dtm' or 'raster'.")
        dtm_path = os.fspath(dtm)
        if not os.path.isfile(dtm_path):
            raise FileNotFoundError(f"No such DTM file: '{dtm_path}'")
        if not dtm_path.lower().endswith((".tif", ".tiff")):
            raise ValueError("The DTM file must be a .tif or .tiff file.")
    else:
        dtm_path = None

    if isinstance(existing_points, np.ndarray):
        arrays = [existing_points]
    else:
        try:
            arrays = list(existing_points)
        except TypeError as exc:
            raise TypeError(
                "existing_points must be a structured NumPy array or an iterable "
                "of structured NumPy arrays."
            ) from exc

    if not arrays:
        raise ValueError("At least one point cloud array is required.")

    required_fields = {"X", "Y", "Z"}
    if method_lc == "delaunay":
        required_fields.add("Classification")

    total_points = 0
    ground_points = 0

    for idx, arr in enumerate(arrays):
        dtype_names = getattr(getattr(arr, "dtype", None), "names", None)
        if dtype_names is None:
            raise TypeError(
                f"Point cloud array at index {idx} must be a structured NumPy array."
            )

        missing_fields = sorted(required_fields.difference(dtype_names))
        if missing_fields:
            raise ValueError(
                f"Point cloud array at index {idx} is missing required fields: "
                f"{', '.join(missing_fields)}."
            )

        total_points += len(arr)
        if len(arr) and method_lc == "delaunay":
            ground_points += int(np.count_nonzero(arr["Classification"] == 2))

    if total_points == 0:
        raise ValueError("At least one point is required to calculate HeightAboveGround.")
    if method_lc == "delaunay" and ground_points == 0:
        raise ValueError(
            "At least one ground point with Classification == 2 is required "
            "to calculate HeightAboveGround."
        )

    pipeline_stages = [_hag_raster(dtm_path)] if method_lc == "dtm" else [_hag_delaunay()]

    try:
        pipeline = _build_pdal_pipeline(arrays, pipeline_stages)
    except RuntimeError as e:
        raise RuntimeError(f"Height Above Ground Pipeline Failed: {e}")

    if not pipeline.arrays:
        raise ValueError("No data returned after HeightAboveGround calculation.")

    return pipeline.arrays


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


def filter_pointsourceid(arrays, pointsource_ids) -> List:
    """
    Filter point cloud arrays to include only the requested PointSourceId values.

    Args:
        arrays (list): Point cloud arrays to be processed.
        pointsource_ids (int or iterable of int): PointSourceId identifiers to keep.

    Returns:
        list: Point cloud arrays containing only points with the specified PointSourceId values.
    """
    pipeline = _build_pdal_pipeline(arrays, [_filter_pointsourceid(pointsource_ids)])
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


def downsample_voxel(arrays, cell, mode) -> List:
    """
    Downsample point cloud arrays using PDAL's voxel-based thinning.

    Uses PDAL ``filters.voxeldownsize`` to partition space into cubic voxels of
    edge length ``cell`` and keep one representative point per occupied voxel.
    The representative is chosen by ``mode``:

      - "center": keeps the first point encountered in each voxel and overwrites
        its X/Y/Z with the voxel center coordinates (shifts positions).
      - "first": keeps the first point encountered in each voxel unchanged.

    Args:
        arrays (list): Point cloud arrays to be downsampled (as expected by your
            pipeline builder).
        cell (float): Voxel edge length in the same units as the coordinates.
            Must be a positive, finite number (> 0).
        mode (str): One of {"center", "first"} (case-insensitive).

    Returns:
        list: Downsampled point cloud arrays from the executed pipeline.

    Raises:
        ValueError: If ``cell`` is not positive/finite or if ``mode`` is invalid.
        TypeError: If ``mode`` is not a string.

    Notes:
        - "center" modifies point coordinates to voxel centers; use "first" if
          you must preserve original XY/Z.
    """
    if not isinstance(cell, (int, float)) or not math.isfinite(cell) or cell <= 0:
        raise ValueError(f"'cell' must be a positive, finite number (got {cell!r}).")

    if not isinstance(mode, str):
        raise TypeError("'mode' must be a string: 'center' or 'first'.")

    mode_lc = mode.lower()
    allowed_modes = {"center", "first"}
    if mode_lc not in allowed_modes:
        raise ValueError(f"'mode' must be one of {sorted(allowed_modes)} (got {mode!r}).")

    pipeline = _build_pdal_pipeline(arrays, [_filter_voxeldownsize(float(cell), mode_lc)])
    return pipeline.arrays
