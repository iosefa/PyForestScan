import pytest
import numpy as np
from scipy.stats import entropy
from pyforestscan.calculate import (
    generate_dtm,
    assign_voxels,
    calculate_pad,
    calculate_pai,
    calculate_fhd,
    calculate_chm,
    calculate_canopy_cover,
    calculate_rumple
)
import math


# ----------------------------
# Helper Functions for Tests
# ----------------------------

def create_point_cloud(num_points=100, missing_fields=False):
    """
    Creates a synthetic point cloud array with 'X', 'Y', 'Z', and 'HeightAboveGround' fields.

    :param num_points: int, number of points in the point cloud.
    :param missing_fields: bool, if True, omit some fields to test error handling.
    :return: numpy.ndarray
    """
    if missing_fields:
        dtype = [('X', 'f4'), ('Y', 'f4')]
    else:
        dtype = [('X', 'f4'), ('Y', 'f4'), ('Z', 'f4'), ('HeightAboveGround', 'f4')]
    data = np.zeros(num_points, dtype=dtype)
    data['X'] = np.random.uniform(0, 100, num_points)
    data['Y'] = np.random.uniform(0, 100, num_points)
    if not missing_fields:
        data['Z'] = np.random.uniform(0, 50, num_points)
        data['HeightAboveGround'] = np.random.uniform(0, 30, num_points)
    return data


# ----------------------------
# Tests for generate_dtm
# ----------------------------

def test_generate_dtm_success():
    ground_points = [create_point_cloud(num_points=100)]
    dtm, extent = generate_dtm(ground_points, resolution=10.0)
    assert isinstance(dtm, np.ndarray)
    assert dtm.ndim == 2
    assert isinstance(extent, list)
    assert len(extent) == 4
    # Calculate expected number of bins using ceiling
    x_range = ground_points[0]['X'].max() - ground_points[0]['X'].min()
    y_range = ground_points[0]['Y'].max() - ground_points[0]['Y'].min()
    expected_x_bins = math.ceil(x_range / 10.0)
    expected_y_bins = math.ceil(y_range / 10.0)
    assert dtm.shape == (expected_x_bins, expected_y_bins)


def test_generate_dtm_no_ground_points():
    ground_points = []
    with pytest.raises(ValueError):
        generate_dtm(ground_points, resolution=10.0)


def test_generate_dtm_missing_fields():
    ground_points = [create_point_cloud(num_points=100, missing_fields=True)]
    with pytest.raises(ValueError):
        generate_dtm(ground_points, resolution=10.0)


def test_generate_dtm_empty_arrays():
    ground_points = [np.array([], dtype=[('X', 'f4'), ('Y', 'f4'), ('Z', 'f4')])]
    with pytest.raises(ValueError):
        generate_dtm(ground_points, resolution=10.0)


# ----------------------------
# Tests for assign_voxels
# ----------------------------

def test_assign_voxels_success():
    arr = create_point_cloud(num_points=100)
    voxel_resolution = (10.0, 10.0, 5.0)
    histogram, extent = assign_voxels(arr, voxel_resolution)
    assert isinstance(histogram, np.ndarray)
    assert histogram.ndim == 3
    assert isinstance(extent, list)
    assert len(extent) == 4
    # Verify histogram shape
    expected_x_bins = math.ceil((arr['X'].max() - arr['X'].min()) / voxel_resolution[0])
    expected_y_bins = math.ceil((arr['Y'].max() - arr['Y'].min()) / voxel_resolution[1])
    expected_z_bins = math.ceil((arr['HeightAboveGround'].max() - arr['HeightAboveGround'].min()) / voxel_resolution[2])
    assert histogram.shape == (expected_x_bins, expected_y_bins, expected_z_bins)


def test_assign_voxels_missing_fields():
    arr = create_point_cloud(num_points=100, missing_fields=True)
    voxel_resolution = (10.0, 10.0, 5.0)
    with pytest.raises(ValueError):
        assign_voxels(arr, voxel_resolution)


def test_assign_voxels_empty_array():
    arr = create_point_cloud(num_points=0)
    voxel_resolution = (10.0, 10.0, 5.0)
    with pytest.raises(ValueError):
        assign_voxels(arr, voxel_resolution)


def test_assign_voxels_non_uniform_resolution():
    arr = create_point_cloud(num_points=100)
    voxel_resolution = (5.0, 10.0, 2.5)
    histogram, extent = assign_voxels(arr, voxel_resolution)
    assert isinstance(histogram, np.ndarray)
    assert histogram.ndim == 3
    # Verify histogram shape
    x_bins = np.arange(
        np.floor(arr['X'].min() / voxel_resolution[0]) * voxel_resolution[0],
        arr['X'].max() + voxel_resolution[0],
        voxel_resolution[0]
    )
    expected_x_bins = len(x_bins) - 1

    y_bins = np.arange(
        np.floor(arr['Y'].min() / voxel_resolution[1]) * voxel_resolution[1],
        arr['Y'].max() + voxel_resolution[1],
        voxel_resolution[1]
    )
    expected_y_bins = len(y_bins) - 1

    z_bins = np.arange(
        np.floor(arr['HeightAboveGround'].min() / voxel_resolution[2]) * voxel_resolution[2],
        arr['HeightAboveGround'].max() + voxel_resolution[2],
        voxel_resolution[2]
    )
    expected_z_bins = len(z_bins) - 1
    assert histogram.shape == (expected_x_bins, expected_y_bins, expected_z_bins)


# ----------------------------
# Tests for calculate_pad
# ----------------------------

def test_calculate_pad_success():
    voxel_returns = np.random.randint(0, 10, size=(10, 10, 5))
    voxel_height = 1.0
    pad = calculate_pad(voxel_returns, voxel_height)
    assert isinstance(pad, np.ndarray)
    assert pad.shape == voxel_returns.shape
    # Check that PAD values are non-negative
    assert np.nanmin(pad) >= 0


def test_calculate_pad_with_zero_shots():
    voxel_returns = np.zeros((5, 5, 5))
    voxel_height = 1.0
    pad = calculate_pad(voxel_returns, voxel_height)
    assert isinstance(pad, np.ndarray)
    assert pad.shape == voxel_returns.shape
    # PAD should be nan where shots_through == 0
    assert np.all(np.isnan(pad))


def test_calculate_pad_with_inf_nan_values():
    voxel_returns = np.random.randint(0, 10, size=(5, 5, 5))
    voxel_returns[0, 0, 0] = 0  # This will cause division by zero
    voxel_height = 1.0
    pad = calculate_pad(voxel_returns, voxel_height)
    assert isinstance(pad, np.ndarray)
    assert pad.shape == voxel_returns.shape
    # PAD at (0,0,0) should be nan due to division by zero
    assert np.isnan(pad[0, 0, 0])


def test_calculate_pad_with_custom_lambert_constant():
    voxel_returns = np.random.randint(1, 10, size=(5, 5, 5))
    voxel_height = 1.0
    beer_lambert_constant = 1.5
    pad = calculate_pad(voxel_returns, voxel_height, beer_lambert_constant=beer_lambert_constant)
    assert isinstance(pad, np.ndarray)
    assert pad.shape == voxel_returns.shape
    # Verify that PAD values are scaled correctly
    # Since k=1.5 and dz=1.0, PAD = log(shots_in / shots_through) * (1 / 1.5)
    # Specific value checks can be added based on expected calculations


def test_calculate_pad_large_dataset():
    voxel_returns = np.random.randint(0, 100, size=(100, 100, 50))
    voxel_height = 1.0
    pad = calculate_pad(voxel_returns, voxel_height)
    assert isinstance(pad, np.ndarray)
    assert pad.shape == voxel_returns.shape
    # Verify that PAD values are computed
    assert np.any(~np.isnan(pad))
    assert np.all(pad[~np.isnan(pad)] >= 0)


def test_calculate_pad_negative_voxel_height():
    voxel_returns = np.random.randint(1, 10, size=(5, 5, 5))

    with pytest.raises(ValueError):
        calculate_pad(voxel_returns, voxel_height=-1.0)


def test_calculate_pad_non_integer_returns():
    voxel_returns = np.random.uniform(0, 10, size=(5, 5, 5))
    voxel_height = 1.0
    pad = calculate_pad(voxel_returns, voxel_height)
    assert isinstance(pad, np.ndarray)
    assert pad.shape == voxel_returns.shape
    # PAD values should be correctly calculated with floating point returns


def test_calculate_pad_mixed_data_types():
    voxel_returns = np.array([
        [[1, 2.5, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]]
    ], dtype=float)
    voxel_height = 1.0
    pad = calculate_pad(voxel_returns, voxel_height)
    assert isinstance(pad, np.ndarray)
    assert pad.shape == voxel_returns.shape

    assert np.isnan(pad[:, :, 0]).all()

    finite_mask = np.isfinite(pad)
    assert np.all(pad[finite_mask] >= 0)


def test_calculate_pad_non_finite_returns():
    voxel_returns = np.array([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [np.inf, 11, 12]]
    ])
    voxel_height = 1.0
    pad = calculate_pad(voxel_returns, voxel_height)
    assert isinstance(pad, np.ndarray)
    assert pad.shape == voxel_returns.shape
    assert np.isnan(pad[1, 1, 0])
    finite_mask = np.isfinite(pad)
    assert np.nanmin(pad[finite_mask]) >= 0


# ----------------------------
# Tests for calculate_pai
# ----------------------------

def test_calculate_pai_success():
    pad = np.random.rand(10, 10, 5)
    pai = calculate_pai(pad, 1, min_height=1, max_height=4)
    assert isinstance(pai, np.ndarray)
    assert pai.shape == (10, 10)
    # PAI should be the sum across specified height axis
    expected_pai = np.sum(pad[:, :, 1:4], axis=2)
    assert np.array_equal(pai, expected_pai)


def test_calculate_pai_max_height_none():
    pad = np.random.rand(10, 10, 5)
    pai = calculate_pai(pad, 1, min_height=2)
    assert isinstance(pai, np.ndarray)
    assert pai.shape == (10, 10)
    # PAI should be the sum from min_height to end
    expected_pai = np.sum(pad[:, :, 2:], axis=2)
    assert np.array_equal(pai, expected_pai)


def test_calculate_pai_min_height_greater_than_max():
    pad = np.random.rand(10, 10, 5)
    # Empty integration range -> zeros (no canopy above threshold)
    pai = calculate_pai(pad, 5, min_height=6, max_height=4)
    assert isinstance(pai, np.ndarray)
    assert pai.shape == (10, 10)
    assert np.all(pai == 0)


def test_calculate_pai_min_height_equals_max():
    pad = np.random.rand(10, 10, 5)
    # Empty integration range -> zeros
    pai = calculate_pai(pad, 5, min_height=3, max_height=3)
    assert isinstance(pai, np.ndarray)
    assert pai.shape == (10, 10)
    assert np.all(pai == 0)


def test_calculate_pai_all_heights():
    pad = np.random.rand(10, 10, 5)
    pai = calculate_pai(pad, 1, min_height=0)
    assert isinstance(pai, np.ndarray), "PAI should be a NumPy array."
    assert pai.shape == (10, 10), f"PAI shape should be (10, 10), got {pai.shape}."
    expected_pai = np.sum(pad, axis=2)
    assert np.array_equal(pai, expected_pai), "PAI values do not match the expected sum across all heights."


def test_calculate_pai_invalid_pad_dimensions():
    pad = np.random.rand(10, 10)  # 2D array instead of 3D
    with pytest.raises(IndexError):
        calculate_pai(pad, 1, min_height=1, max_height=4)


def test_calculate_pai_varying_height_ranges():
    pad = np.ones((10, 10, 10))
    voxel_height = 1.0
    pai = calculate_pai(pad, voxel_height, min_height=2, max_height=5)
    expected_pai = np.sum(pad[:, :, 2:5], axis=2) * voxel_height
    assert np.array_equal(pai, expected_pai)
    # Expected PAI should be 3 for each (x, y) voxel
    assert np.all(pai == 3)


def test_calculate_pai_all_zero_pad():
    pad = np.zeros((10, 10, 5))
    pai = calculate_pai(pad, 5)
    assert isinstance(pai, np.ndarray)
    assert pai.shape == (10, 10)
    assert np.all(pai == 0)


# ----------------------------
# Tests for calculate_fhd
# ----------------------------

def test_calculate_fhd_success():
    voxel_returns = np.random.randint(0, 10, size=(5, 5, 5))
    fhd = calculate_fhd(voxel_returns, voxel_height=1, min_height=0)
    assert isinstance(fhd, np.ndarray)
    assert fhd.shape == (5, 5)
    # Entropy should be non-negative
    assert np.all(fhd >= 0)


def test_calculate_fhd_zero_shots():
    voxel_returns = np.zeros((5, 5, 5))
    fhd = calculate_fhd(voxel_returns, voxel_height=1, min_height=0)
    assert isinstance(fhd, np.ndarray)
    assert fhd.shape == (5, 5)
    # FHD should be NaN where shots_in == 0
    assert np.all(np.isnan(fhd))


def test_calculate_fhd_single_class():
    voxel_returns = np.zeros((5, 5, 5))
    voxel_returns[:, :, 2] = 10  # All returns in z=2
    fhd = calculate_fhd(voxel_returns, voxel_height=1, min_height=0)
    assert isinstance(fhd, np.ndarray)
    assert fhd.shape == (5, 5)
    # Entropy should be 0 since only one class
    assert np.all(fhd == 0)


def test_calculate_fhd_uniform_distribution():
    voxel_returns = np.ones((3, 3, 3))
    fhd = calculate_fhd(voxel_returns, voxel_height=1, min_height=0)
    expected_entropy = entropy([1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(fhd, expected_entropy)


def test_calculate_fhd_non_uniform():
    voxel_returns = np.array([
        [[5, 0, 0], [1, 1, 1]],
        [[2, 2, 2], [3, 0, 0]],
        [[0, 0, 0], [4, 4, 4]]
    ])
    fhd = calculate_fhd(voxel_returns, voxel_height=1, min_height=0)

    # Define the expected entropy for each voxel column
    # [5, 0, 0] => entropy([1, 0, 0]) = 0
    # [1, 1, 1] => entropy([1/3, 1/3, 1/3]) = ln(3)
    # [2, 2, 2] => entropy([1/3, 1/3, 1/3]) = ln(3)
    # [3, 0, 0] => entropy([1, 0, 0]) = 0
    # [0, 0, 0] => entropy undefined, set to NaN
    # [4, 4, 4] => entropy([1/3, 1/3, 1/3]) = ln(3)

    expected_fhd = np.array([
        [0.0, np.log(3)],
        [np.log(3), 0.0],
        [np.nan, np.log(3)]
    ])

    assert isinstance(fhd, np.ndarray), "FHD should be a NumPy array."
    assert fhd.shape == (3, 2), f"FHD shape should be (3, 2), got {fhd.shape}."

    for i in range(fhd.shape[0]):
        for j in range(fhd.shape[1]):
            if np.isnan(expected_fhd[i, j]):
                assert np.isnan(fhd[i, j]), f"FHD[{i}, {j}] should be NaN."
            else:
                assert np.isclose(
                    fhd[i, j],
                    expected_fhd[i, j],
                    atol=1e-6
                ), f"FHD[{i}, {j}] should be close to {expected_fhd[i, j]}, got {fhd[i, j]}."


def test_calculate_fhd_high_diversity():
    voxel_returns = np.array([
        [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
        [[2, 3, 4, 5, 6], [6, 5, 4, 3, 2]],
        [[3, 4, 5, 6, 7], [7, 6, 5, 4, 3]],
        [[4, 5, 6, 7, 8], [8, 7, 6, 5, 4]],
        [[5, 6, 7, 8, 9], [9, 8, 7, 6, 5]]
    ])
    fhd = calculate_fhd(voxel_returns, voxel_height=1, min_height=0)
    assert isinstance(fhd, np.ndarray)
    assert fhd.shape == (5, 2)
    # Entropy should be consistent with distribution
    for i in range(5):
        for j in range(2):
            counts = voxel_returns[i, j, :]
            proportions = counts / counts.sum()
            expected_entropy = entropy(proportions)
            assert np.isclose(fhd[i, j], expected_entropy)


def test_calculate_fhd_partial_zero_distribution():
    # Construct deterministic columns to avoid flaky equality due to randomness.
    voxel_returns = np.zeros((5, 5, 5), dtype=int)
    # Column with all mass in a single bin -> entropy 0
    voxel_returns[0, 0, :] = [1, 0, 0, 0, 0]
    # Column with uniform distribution across bins -> maximal entropy
    voxel_returns[1, 1, :] = [1, 1, 1, 1, 1]

    fhd = calculate_fhd(voxel_returns, voxel_height=1, min_height=0)
    assert isinstance(fhd, np.ndarray)
    assert fhd.shape == (5, 5)
    assert fhd[0, 0] < fhd[1, 1]


def test_calculate_fhd_all_zero_returns():
    voxel_returns = np.zeros((5, 5, 5))
    fhd = calculate_fhd(voxel_returns, voxel_height=1, min_height=0)
    assert isinstance(fhd, np.ndarray)
    assert fhd.shape == (5, 5)
    assert np.all(np.isnan(fhd))


def test_calculate_fhd_respects_min_height():
    voxel_returns = np.array([[[4, 1, 1]]], dtype=float)
    full_range = calculate_fhd(voxel_returns, voxel_height=1, min_height=0)
    above_1m = calculate_fhd(voxel_returns, voxel_height=1, min_height=1)

    assert full_range.shape == (1, 1)
    assert above_1m.shape == (1, 1)

    expected_above_1m = entropy([0.5, 0.5])
    assert not np.isclose(full_range[0, 0], above_1m[0, 0])
    assert np.isclose(above_1m[0, 0], expected_above_1m)


# ----------------------------
# Tests for calculate_chm
# ----------------------------

def test_calculate_chm_success():
    arr = create_point_cloud(num_points=100)
    voxel_resolution = (10.0, 10.0)
    chm, extent = calculate_chm(arr, voxel_resolution)
    assert isinstance(chm, np.ndarray), "CHM should be a NumPy array."
    assert chm.ndim == 2, f"CHM should be 2-dimensional, got {chm.ndim} dimensions."
    assert isinstance(extent, list), "Extent should be a list."
    assert len(extent) == 4, f"Extent should have 4 elements, got {len(extent)}."
    valid_values = ~np.isnan(chm)
    assert np.all(chm[valid_values] >= 0), "CHM contains values less than 0 that are not NaN."


def test_calculate_chm_missing_fields():
    arr = create_point_cloud(num_points=100, missing_fields=True)
    voxel_resolution = (10.0, 10.0)
    with pytest.raises(ValueError):
        calculate_chm(arr, voxel_resolution)


def test_calculate_chm_empty_array():
    arr = create_point_cloud(num_points=0)
    voxel_resolution = (10.0, 10.0)
    with pytest.raises(ValueError):
        calculate_chm(arr, voxel_resolution)


def test_calculate_chm_varying_heights():
    """
    Test that calculate_chm correctly assigns maximum heights to each voxel.
    For voxels with no points, CHM should be either NaN or 0.
    """
    arr = create_point_cloud(num_points=100)
    arr['HeightAboveGround'] = np.random.uniform(0, 50, 100)
    voxel_resolution = (10.0, 10.0)
    chm, extent = calculate_chm(arr, voxel_resolution, interpolation=None)

    assert isinstance(chm, np.ndarray), "CHM should be a NumPy array."
    assert chm.ndim == 2, f"CHM should be 2-dimensional, got {chm.ndim} dimensions."
    assert isinstance(extent, list), "Extent should be a list."
    assert len(extent) == 4, f"Extent should have 4 elements, got {len(extent)}."

    for i in range(chm.shape[0]):
        for j in range(chm.shape[1]):
            x_min = extent[0] + i * voxel_resolution[0]
            x_max = extent[0] + (i + 1) * voxel_resolution[0]
            y_index = chm.shape[1] - 1 - j
            y_min = extent[2] + y_index * voxel_resolution[1]
            y_max = extent[2] + (y_index + 1) * voxel_resolution[1]

            points_in_voxel = arr[
                (arr['X'] >= x_min) & (arr['X'] < x_max) &
                (arr['Y'] >= y_min) & (arr['Y'] < y_max)
            ]

            if len(points_in_voxel) > 0:
                expected_max = points_in_voxel['HeightAboveGround'].max()
                actual_value = chm[i, j]
                assert np.isclose(actual_value, expected_max, atol=1e-6), (
                    f"CHM[{i}, {j}] should be {expected_max}, got {actual_value}."
                )
            else:
                actual_value = chm[i, j]
                assert np.isnan(actual_value) or actual_value == 0, (
                    f"CHM[{i}, {j}] should be NaN or 0, got {actual_value}."
                )

def test_calculate_chm_negative_heights():
    """
    Test that calculate_chm correctly assigns maximum heights to each voxel,
    including negative heights. For voxels with no points, CHM should be either NaN or 0.
    """
    np.random.seed(42)
    arr = create_point_cloud(num_points=100)
    arr['HeightAboveGround'] = np.random.uniform(-20, 50, 100)
    voxel_resolution = (10.0, 10.0)
    chm, extent = calculate_chm(arr, voxel_resolution, interpolation=None)

    assert isinstance(chm, np.ndarray), "CHM should be a NumPy array."
    assert chm.ndim == 2, f"CHM should be 2-dimensional, got {chm.ndim} dimensions."
    assert isinstance(extent, list), "Extent should be a list."
    assert len(extent) == 4, f"Extent should have 4 elements, got {len(extent)}."

    for i in range(chm.shape[0]):
        for j in range(chm.shape[1]):
            x_min = extent[0] + i * voxel_resolution[0]
            x_max = extent[0] + (i + 1) * voxel_resolution[0]
            y_index = chm.shape[1] - 1 - j
            y_min = extent[2] + y_index * voxel_resolution[1]
            y_max = extent[2] + (y_index + 1) * voxel_resolution[1]

            points_in_voxel = arr[
                (arr['X'] >= x_min) & (arr['X'] < x_max) &
                (arr['Y'] >= y_min) & (arr['Y'] < y_max)
            ]

            if len(points_in_voxel) > 0:
                expected_max = points_in_voxel['HeightAboveGround'].max()
                actual_value = chm[i, j]
                assert np.isclose(actual_value, expected_max, atol=1e-6), (
                    f"CHM[{i}, {j}] should be {expected_max}, got {actual_value}."
                )
            else:
                actual_value = chm[i, j]
                assert np.isnan(actual_value) or actual_value == 0, (
                    f"CHM[{i}, {j}] should be NaN or 0, got {actual_value}."
                )

def test_calculate_chm_floating_point_resolution():
    """
    Test that calculate_chm correctly assigns maximum heights to each voxel
    when using floating-point voxel resolutions. For voxels with no points,
    CHM should be either NaN or 0.
    """
    np.random.seed(42)
    arr = create_point_cloud(num_points=100)
    arr['HeightAboveGround'] = np.random.uniform(-20, 50, 100)
    voxel_resolution = (7.5, 12.3)
    chm, extent = calculate_chm(arr, voxel_resolution, interpolation=None)

    assert isinstance(chm, np.ndarray), "CHM should be a NumPy array."
    assert chm.ndim == 2, f"CHM should be 2-dimensional, got {chm.ndim} dimensions."
    assert isinstance(extent, list), "Extent should be a list."
    assert len(extent) == 4, f"Extent should have 4 elements, got {len(extent)}."

    x_min, x_max, y_min, y_max = extent
    x_bins = np.arange(x_min, x_max + voxel_resolution[0], voxel_resolution[0])
    y_bins = np.arange(y_min, y_max + voxel_resolution[1], voxel_resolution[1])
    expected_shape = (len(x_bins) - 1, len(y_bins) - 1)
    assert chm.shape == expected_shape, (
        f"CHM shape should be {expected_shape}, got {chm.shape}."
    )

    for i in range(chm.shape[0]):
        for j in range(chm.shape[1]):
            x_min_voxel = x_bins[i]
            x_max_voxel = x_bins[i + 1]
            y_index = chm.shape[1] - 1 - j
            y_min_voxel = y_bins[y_index]
            y_max_voxel = y_bins[y_index + 1]

            points_in_voxel = arr[
                (arr['X'] >= x_min_voxel) & (arr['X'] < x_max_voxel) &
                (arr['Y'] >= y_min_voxel) & (arr['Y'] < y_max_voxel)
            ]

            if len(points_in_voxel) > 0:
                expected_max = points_in_voxel['HeightAboveGround'].max()
                actual_value = chm[i, j]
                assert np.isclose(actual_value, expected_max, atol=1e-6), (
                    f"CHM[{i}, {j}] should be {expected_max}, got {actual_value}."
                )
            else:
                actual_value = chm[i, j]
                assert np.isnan(actual_value) or actual_value == 0, (
                    f"CHM[{i}, {j}] should be NaN or 0, got {actual_value}."
                )


def test_calculate_chm_non_numeric_data():
    dtype = [('X', 'U10'), ('Y', 'U10'), ('HeightAboveGround', 'U10')]
    arr = np.array([('a', 'b', 'c')], dtype=dtype)
    voxel_resolution = (10.0, 10.0)
    with pytest.raises(TypeError):
        calculate_chm(arr, voxel_resolution)


def test_calculate_chm_large_heights():
    arr = create_point_cloud(num_points=1000)
    arr['HeightAboveGround'] = np.random.uniform(1000, 2000, 1000)
    voxel_resolution = (50.0, 50.0)
    chm, extent = calculate_chm(arr, voxel_resolution)
    assert isinstance(chm, np.ndarray)
    assert chm.ndim == 2
    assert np.all(chm >= 1000)


# ----------------------------
# Tests for calculate_rumple
# ----------------------------

def rumple_points(heights, resolution=(1.0, 1.0, 1.0)):
    """Place canopy samples at cell centers, ordered west-east/north-south."""
    heights = np.asarray(heights)
    dx, dy, _ = resolution
    x, y = np.meshgrid((np.arange(heights.shape[0]) + 0.5) * dx,
                       (heights.shape[1] - np.arange(heights.shape[1]) - 0.5) * dy,
                       indexing='ij')
    points = np.zeros(heights.size, dtype=[('X', 'f8'), ('Y', 'f8'), ('HeightAboveGround', 'f8')])
    points['X'], points['Y'] = x.ravel(), y.ravel()
    points['HeightAboveGround'] = heights.ravel()
    return points


def test_calculate_rumple_flat_surface_matches_voxel_grid():
    resolution = (2.0, 3.0, 1.0)
    points = rumple_points(np.full((5, 7), 12.0), resolution)
    # Off-center samples must still align with the voxel grid.
    points['X'] += 0.2
    points['Y'] -= 0.3
    rumple, extent = calculate_rumple(points, resolution)
    voxels, voxel_extent = assign_voxels(points, resolution)

    assert rumple.shape == voxels.shape[:2] == (5, 7)
    assert extent == voxel_extent == [0.0, 10.0, 0.0, 21.0]
    np.testing.assert_allclose(rumple[1:-1, 1:-1], 1.0)
    assert np.isnan(rumple[[0, -1], :]).all()
    assert np.isnan(rumple[:, [0, -1]]).all()


def test_calculate_rumple_planar_slope_matches_analytical_ratio():
    dx, dy = 2.0, 3.0
    a, b = 0.5, -0.25
    points = rumple_points(np.full((5, 7), 12.0), (dx, dy, 1.0))
    points['HeightAboveGround'] += a * points['X'] + b * points['Y']

    rumple, _ = calculate_rumple(points, (dx, dy, 1.0))
    expected = math.sqrt(1.0 + a ** 2 + b ** 2)
    np.testing.assert_allclose(rumple[1:-1, 1:-1], expected)


def test_calculate_rumple_peak_area_and_spatial_variation():
    dx, dy, height = 2.0, 3.0, 4.0
    canopy = np.full((9, 7), 10.0)
    canopy[3, 2] += height
    points = rumple_points(canopy, (dx, dy, 1.0))

    rumple, _ = calculate_rumple(points, (dx, dy, 1.0))

    # Eight triangles form a rectangular pyramid around the summit.
    expected = (math.sqrt(1 + (height / dx) ** 2) +
                math.sqrt(1 + (height / dy) ** 2)) / 2
    assert rumple[3, 2] == pytest.approx(expected)
    assert rumple[6, 4] == pytest.approx(1.0)
    assert np.nanmin(rumple) >= 1.0


@pytest.mark.parametrize('missing_height', [1.0, np.nan, np.inf])
def test_calculate_rumple_masks_missing_or_low_neighborhoods(missing_height):
    canopy = np.full((7, 7), 3.0)
    canopy[3, 3] = missing_height
    points = rumple_points(canopy)
    rumple, extent = calculate_rumple(points, (1, 1, 1), min_height=3.0)

    assert extent == [0, 7, 0, 7]
    assert np.isnan(rumple[2:5, 2:5]).all()
    assert rumple[1, 1] == pytest.approx(1.0)
    assert np.isfinite(rumple).sum() == 16


@pytest.mark.parametrize('interpolation', ['linear', 'cubic'])
def test_calculate_rumple_interpolates_planar_gaps(interpolation):
    resolution = (2.0, 3.0, 1.0)
    points = rumple_points(np.full((7, 9), 12.0), resolution)
    points['HeightAboveGround'] += 0.5 * points['X'] - 0.25 * points['Y']
    heights = points['HeightAboveGround'].reshape(7, 9)
    heights[2:4, 3:5] = np.nan
    unfilled, unfilled_extent = calculate_rumple(points, resolution)

    filled, extent = calculate_rumple(points, resolution, interpolation=interpolation)

    assert extent == unfilled_extent
    assert np.isfinite(unfilled).sum() < np.isfinite(filled).sum()
    np.testing.assert_allclose(filled[1:-1, 1:-1], math.sqrt(1 + 0.5**2 + 0.25**2), atol=1e-6)
    assert np.isnan(filled[[0, -1], :]).all()
    assert np.isnan(filled[:, [0, -1]]).all()


def test_calculate_rumple_nearest_uses_distances_and_recomputes_surface():
    resolution = (8.0, 2.0, 1.0)
    canopy = np.full((7, 9), 12.0)
    # Both closest samples are 2 m away; the lower side samples are 8 m away.
    canopy[3, 3:6] = 20.0
    expected, expected_extent = calculate_rumple(rumple_points(canopy, resolution), resolution)
    canopy[3, 4] = np.nan

    filled, extent = calculate_rumple(
        rumple_points(canopy, resolution), resolution, interpolation='nearest',
    )

    assert extent == expected_extent
    # Check the surface area of the filled canopy, including its observed peaks.
    np.testing.assert_allclose(filled, expected, equal_nan=True)


@pytest.mark.parametrize('interpolation', ['linear', 'cubic', 'nearest'])
def test_calculate_rumple_interpolation_preserves_height_mask(interpolation):
    canopy = np.full((9, 9), 5.0)
    canopy[2, 2] = 1.0
    canopy[6, 6] = np.nan
    rumple, _ = calculate_rumple(
        rumple_points(canopy), (1, 1, 1), min_height=2.0, interpolation=interpolation,
    )

    assert np.isnan(rumple[1:4, 1:4]).all()
    assert np.isfinite(rumple[5:8, 5:8]).all()
    assert np.isfinite(rumple).sum() == 40


@pytest.mark.parametrize('interpolation', ['linear', 'cubic', 'nearest'])
def test_calculate_rumple_interpolation_outside_convex_hull(interpolation):
    x, y = np.indices((7, 7))
    canopy = np.where(x + y >= 5, 10.0, np.nan)
    canopy[4, 3] = np.nan
    rumple, _ = calculate_rumple(
        rumple_points(canopy), (1, 1, 1), interpolation=interpolation,
    )

    assert rumple[4, 3] == pytest.approx(1.0)
    if interpolation == 'nearest':
        np.testing.assert_allclose(rumple[1:-1, 1:-1], 1.0)
    else:
        assert np.isnan(rumple[1, 1])


@pytest.mark.parametrize('interpolation', ['linear', 'cubic'])
@pytest.mark.parametrize('sample_count', [2, 3])
def test_calculate_rumple_interpolation_without_2d_support(interpolation, sample_count):
    canopy = np.full((7, 7), np.nan)
    diagonal = np.linspace(0, 6, sample_count, dtype=int)
    canopy[diagonal, diagonal] = 10.0
    rumple, extent = calculate_rumple(
        rumple_points(canopy), (1, 1, 1), interpolation=interpolation,
    )
    assert extent == [0, 7, 0, 7]
    assert np.isnan(rumple).all()


@pytest.mark.parametrize('shape', [(1, 1), (1, 5), (2, 5), (5, 2)])
def test_calculate_rumple_small_grids_return_nodata(shape):
    rumple, _ = calculate_rumple(rumple_points(np.full(shape, 10.0)), (1, 1, 1))
    assert rumple.shape == shape
    assert np.isnan(rumple).all()


@pytest.mark.parametrize('interpolation', [None, 'linear', 'cubic', 'nearest'])
def test_calculate_rumple_threshold_preserves_extent_when_all_cells_masked(interpolation):
    canopy = np.full((5, 4), 1.0)
    canopy[2, 2] = np.nan
    rumple, extent = calculate_rumple(
        rumple_points(canopy), (1, 1, 1), min_height=2.0, interpolation=interpolation,
    )
    assert rumple.shape == (5, 4)
    assert extent == [0, 5, 0, 4]
    assert np.isnan(rumple).all()


def test_calculate_rumple_uses_canopy_maxima_and_ignores_vertical_bin_size():
    heights = np.full((6, 5), 10.25)
    heights[3, 2] = 12.75
    canopy = rumple_points(heights)
    understory = canopy.copy()
    understory['HeightAboveGround'] = 1.0
    expected, expected_extent = calculate_rumple(canopy, (1, 1, 1))

    actual, extent = calculate_rumple(np.concatenate([understory, canopy]), (1, 1, 5))

    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert extent == expected_extent


def test_calculate_rumple_resolution_controls_output_shape():
    points = rumple_points(np.full((8, 6), 10.0))
    fine, fine_extent = calculate_rumple(points, (1, 1, 1))
    coarse, coarse_extent = calculate_rumple(points, (2, 2, 1))
    assert fine.shape == (8, 6)
    assert coarse.shape == (4, 3)
    assert fine_extent == coarse_extent
    np.testing.assert_allclose(coarse[1:-1, 1:-1], 1.0)


def test_calculate_rumple_ignores_invalid_points_without_changing_grid():
    points = rumple_points(np.full((5, 5), 10.0))
    invalid = np.array([(100, 100, -1), (np.nan, 2, 10), (2, np.inf, 10),
                        (100, 100, np.nan)], dtype=points.dtype)
    expected, expected_extent = calculate_rumple(points, (1, 1, 1))
    actual, extent = calculate_rumple(np.concatenate([points, invalid]), (1, 1, 1))
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert extent == expected_extent


@pytest.mark.parametrize('resolution', [(1, 1), (1,), (0, 1, 1), (-1, 1, 1),
                                        (1, np.nan, 1), (1, 1, np.inf)])
def test_calculate_rumple_invalid_resolution(resolution):
    with pytest.raises(ValueError, match='voxel_resolution'):
        calculate_rumple(rumple_points(np.ones((3, 3))), resolution)


@pytest.mark.parametrize('threshold', [np.nan, np.inf, -np.inf])
def test_calculate_rumple_invalid_threshold(threshold):
    with pytest.raises(ValueError, match='min_height'):
        calculate_rumple(rumple_points(np.ones((3, 3))), (1, 1, 1), min_height=threshold)


@pytest.mark.parametrize('interpolation', ['', 'bilinear', True])
def test_calculate_rumple_invalid_interpolation(interpolation):
    with pytest.raises(ValueError, match='interpolation'):
        calculate_rumple(rumple_points(np.ones((3, 3))), (1, 1, 1), interpolation=interpolation)


def test_calculate_rumple_missing_dimensions():
    with pytest.raises(KeyError, match='HeightAboveGround'):
        calculate_rumple(np.zeros(3, dtype=[('X', 'f8'), ('Y', 'f8')]), (1, 1, 1))


def test_calculate_rumple_non_one_dimensional_points():
    points = rumple_points(np.ones((3, 3))).reshape(3, 3)
    with pytest.raises(ValueError, match='Input point array must be one-dimensional'):
        calculate_rumple(points, (1, 1, 1))


@pytest.mark.parametrize('height', [-1.0, np.nan, np.inf])
def test_calculate_rumple_no_valid_points(height):
    with pytest.raises(ValueError, match='No finite points'):
        calculate_rumple(rumple_points(np.full((3, 3), height)), (1, 1, 1))


def test_calculate_rumple_empty_input():
    with pytest.raises(ValueError, match='No finite points'):
        calculate_rumple(rumple_points(np.ones((3, 3)))[:0], (1, 1, 1))


# ----------------------------
# Tests for calculate_canopy_cover
# ----------------------------

def test_calculate_canopy_cover_basic():
    pad = np.ones((4, 3, 10), dtype=float)
    voxel_height = 1.0
    z = 2.0
    k = 0.5
    # PAI above z=2 with dz=1 and 10 layers => sum from idx 2..9 = 8
    expected_cover = 1.0 - np.exp(-k * 8.0)
    cov = calculate_canopy_cover(pad, voxel_height, min_height=z, k=k)
    assert cov.shape == (4, 3)
    assert np.allclose(cov, expected_cover)


def test_calculate_canopy_cover_zero_pad():
    pad = np.zeros((2, 2, 5), dtype=float)
    cov = calculate_canopy_cover(pad, voxel_height=1.0, min_height=2.0, k=0.5)
    assert np.all(cov == 0.0)


def test_calculate_canopy_cover_nans_propagate():
    pad = np.ones((2, 2, 6), dtype=float)
    pad[:, :, 3:] = np.nan  # everything above z=3 m is NaN
    cov = calculate_canopy_cover(pad, voxel_height=1.0, min_height=3.0, k=0.5)
    # All NaN in integration range -> NaN in cover
    assert np.all(np.isnan(cov))


def test_calculate_canopy_cover_monotonic_with_height():
    pad = np.ones((3, 3, 10), dtype=float)
    cov0 = calculate_canopy_cover(pad, voxel_height=1.0, min_height=0.0, k=0.5)
    cov2 = calculate_canopy_cover(pad, voxel_height=1.0, min_height=2.0, k=0.5)
    cov5 = calculate_canopy_cover(pad, voxel_height=1.0, min_height=5.0, k=0.5)
    assert np.all(cov0 >= cov2)
    assert np.all(cov2 >= cov5)


def test_calculate_canopy_cover_empty_range_above_top_returns_zero():
    # Z extent is 0..1 m (2 layers with dz=1); threshold at 2 m leaves no range
    pad = np.random.rand(4, 4, 2)
    cov = calculate_canopy_cover(pad, voxel_height=1.0, min_height=2.0, k=0.5)
    assert cov.shape == (4, 4)
    assert np.allclose(cov, 0.0)


def test_calculate_canopy_cover_min_ge_max_returns_zero():
    pad = np.random.rand(3, 3, 5)
    # Explicitly set max_height below min_height so integration is empty
    cov = calculate_canopy_cover(pad, voxel_height=1.0, min_height=3.0, max_height=2.0, k=0.5)
    assert cov.shape == (3, 3)
    assert np.allclose(cov, 0.0)
