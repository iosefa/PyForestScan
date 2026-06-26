import numpy as np
import pytest

import pyforestscan.filters as filter_module
from pyforestscan.filters import (
    add_height_above_ground, filter_hag, filter_ground, filter_select_ground
)


def create_synthetic_point_cloud(num_points=100, include_ground=True, classification_probs=None):
    """
    Creates a synthetic point cloud for testing purposes with 'X', 'Y', 'Z', and 'HeightAboveGround'.

    :param num_points: int, number of points to generate
    :param include_ground: bool, whether to include ground points (Classification == 2)
    :param classification_probs: list of floats, probabilities for classification codes
    :return: structured NumPy array representing the point cloud
    """
    if classification_probs is None:
        classification_probs = [0.7, 0.3]
    x = np.random.uniform(0, 100, num_points)
    y = np.random.uniform(0, 100, num_points)
    z = np.random.uniform(0, 50, num_points)
    height_above_ground = z - np.random.uniform(0, 10, num_points)
    if include_ground:
        classification = np.random.choice([2, 0], size=num_points, p=classification_probs)
    else:
        classification = np.full(num_points, 0)

    dtype = [('X', 'f4'), ('Y', 'f4'), ('Z', 'f4'), ('HeightAboveGround', 'f4'), ('Classification', 'i4')]
    point_cloud = np.array(list(zip(x, y, z, height_above_ground, classification)), dtype=dtype)
    return point_cloud


def create_classified_point_cloud():
    """
    Creates a small structured point cloud with classified ground points.
    """
    dtype = [('X', 'f8'), ('Y', 'f8'), ('Z', 'f8'), ('Classification', 'u1')]
    return np.array([
        (0.0, 0.0, 10.0, 2),
        (1.0, 0.0, 10.5, 2),
        (0.0, 1.0, 9.5, 2),
        (1.0, 1.0, 18.0, 1),
    ], dtype=dtype)


def create_xyz_point_cloud():
    """
    Creates a small structured point cloud without classification.
    """
    dtype = [('X', 'f8'), ('Y', 'f8'), ('Z', 'f8')]
    return np.array([
        (0.0, 0.0, 10.0),
        (1.0, 0.0, 10.5),
        (0.0, 1.0, 9.5),
        (1.0, 1.0, 18.0),
    ], dtype=dtype)


def test_add_height_above_ground(monkeypatch):
    """
    Test that add_height_above_ground runs an in-memory HAG Delaunay pipeline.
    """
    point_cloud = create_classified_point_cloud()
    output_dtype = point_cloud.dtype.descr + [('HeightAboveGround', 'f8')]
    output = np.empty(point_cloud.shape, dtype=output_dtype)
    for name in point_cloud.dtype.names:
        output[name] = point_cloud[name]
    output['HeightAboveGround'] = [0.0, 0.0, 0.0, 8.0]

    captured = {}

    class FakePipeline:
        arrays = [output]

    def fake_build_pdal_pipeline(arrays, pipeline_stages):
        captured["arrays"] = arrays
        captured["pipeline_stages"] = pipeline_stages
        return FakePipeline()

    monkeypatch.setattr(filter_module, "_build_pdal_pipeline", fake_build_pdal_pipeline)

    hag_arrays = add_height_above_ground(point_cloud)

    assert isinstance(hag_arrays, list)
    assert hag_arrays[0] is output
    assert captured["arrays"][0] is point_cloud
    assert captured["pipeline_stages"] == [{"type": "filters.hag_delaunay"}]
    assert "HeightAboveGround" in hag_arrays[0].dtype.names


def test_add_height_above_ground_with_dtm(monkeypatch, tmp_path):
    """
    Test that add_height_above_ground can use an in-memory DTM HAG pipeline.
    """
    point_cloud = create_xyz_point_cloud()
    dtm = tmp_path / "dtm.tif"
    dtm.write_bytes(b"")

    output_dtype = point_cloud.dtype.descr + [('HeightAboveGround', 'f8')]
    output = np.empty(point_cloud.shape, dtype=output_dtype)
    for name in point_cloud.dtype.names:
        output[name] = point_cloud[name]
    output['HeightAboveGround'] = [1.0, 1.5, 0.5, 9.0]

    captured = {}

    class FakePipeline:
        arrays = [output]

    def fake_build_pdal_pipeline(arrays, pipeline_stages):
        captured["arrays"] = arrays
        captured["pipeline_stages"] = pipeline_stages
        return FakePipeline()

    monkeypatch.setattr(filter_module, "_build_pdal_pipeline", fake_build_pdal_pipeline)

    hag_arrays = add_height_above_ground(point_cloud, dtm=dtm)

    assert isinstance(hag_arrays, list)
    assert hag_arrays[0] is output
    assert captured["arrays"][0] is point_cloud
    assert captured["pipeline_stages"] == [
        {"type": "filters.hag_dem", "raster": str(dtm)}
    ]
    assert "HeightAboveGround" in hag_arrays[0].dtype.names


def test_add_height_above_ground_dtm_method_requires_dtm():
    """
    Test that the DTM method requires a DTM path.
    """
    point_cloud = create_xyz_point_cloud()

    with pytest.raises(ValueError, match="dtm must be provided"):
        add_height_above_ground(point_cloud, method="dtm")


def test_add_height_above_ground_rejects_unknown_method():
    """
    Test that only supported Height Above Ground methods are accepted.
    """
    point_cloud = create_classified_point_cloud()

    with pytest.raises(ValueError, match="method must be one of"):
        add_height_above_ground(point_cloud, method="nearest")


def test_add_height_above_ground_missing_required_field():
    """
    Test that add_height_above_ground requires the dimensions needed by PDAL.
    """
    point_cloud = np.array(
        [(0.0, 0.0, 10.0)],
        dtype=[('X', 'f8'), ('Y', 'f8'), ('Z', 'f8')]
    )

    with pytest.raises(ValueError, match="Classification"):
        add_height_above_ground(point_cloud)


def test_add_height_above_ground_requires_ground_points():
    """
    Test that add_height_above_ground validates classified ground points.
    """
    point_cloud = create_classified_point_cloud()
    point_cloud['Classification'] = 1

    with pytest.raises(ValueError, match="ground point"):
        add_height_above_ground(point_cloud)


def test_filter_hag():
    """
    Test that filter_hag correctly filters points within the specified HAG limits.
    """
    point_cloud = create_synthetic_point_cloud(num_points=100)
    lower_limit = 0
    upper_limit = 30
    filtered_arrays = filter_hag([point_cloud], lower_limit, upper_limit)

    assert isinstance(filtered_arrays, list), "Filtered data should be returned as a list."
    assert len(filtered_arrays) == 1, "Filtered data list should contain one array."

    filtered_points = filtered_arrays[0]
    assert isinstance(filtered_points, np.ndarray), "Filtered data should be a NumPy array."
    assert np.all(filtered_points['HeightAboveGround'] >= lower_limit), "Some points are below the lower HAG limit."
    if upper_limit is not None:
        assert np.all(filtered_points['HeightAboveGround'] <= upper_limit), "Some points are above the upper HAG limit."


def test_filter_ground():
    """
    Test that filter_ground removes all ground points (Classification == 2).
    """
    point_cloud = create_synthetic_point_cloud(num_points=100, include_ground=True)
    filtered_arrays = filter_ground([point_cloud])

    assert isinstance(filtered_arrays, list), "Filtered data should be returned as a list."
    assert len(filtered_arrays) == 1, "Filtered data list should contain one array."

    filtered_points = filtered_arrays[0]
    assert isinstance(filtered_points, np.ndarray), "Filtered data should be a NumPy array."
    assert not np.any(filtered_points['Classification'] == 2), "Ground points were not removed."


def test_filter_select_ground():
    """
    Test that filter_select_ground retains only ground points (Classification == 2).
    """
    point_cloud = create_synthetic_point_cloud(num_points=100, include_ground=True)
    filtered_arrays = filter_select_ground([point_cloud])

    assert isinstance(filtered_arrays, list), "Filtered data should be returned as a list."
    assert len(filtered_arrays) == 1, "Filtered data list should contain one array."

    filtered_points = filtered_arrays[0]
    assert isinstance(filtered_points, np.ndarray), "Filtered data should be a NumPy array."
    assert np.all(filtered_points['Classification'] == 2), "Non-ground points were not removed."
