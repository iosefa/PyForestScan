import numpy as np

from pyforestscan.filters import (
    filter_hag, filter_ground, filter_select_ground
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
