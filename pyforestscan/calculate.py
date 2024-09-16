import pandas as pd
import numpy as np
from scipy.stats import entropy


def assign_voxels(arr, voxel_resolution):
    """
    Assigns voxel grids to spatial data points based on the specified resolutions.

    :param arr:
        Input array-like object containing point cloud data with 'X', 'Y', and 'HeightAboveGround' fields.
    :type arr: array-like
    :param voxel_resolution:
        The resolution for x, y, and z dimensions of the voxel grid.
    :type voxel_resolution: tuple of floats

    :return:
        A tuple containing the histogram of the voxel grid and the extent of the point cloud.
    :rtype: tuple of (numpy.ndarray, list)
    :raises ValueError:
        If input data does not have required fields.
    """
    laz_df = pd.DataFrame(arr)
    x_resolution, y_resolution, z_resolution = voxel_resolution
    xy_resolution = (x_resolution, y_resolution,)
    x_bin = np.arange(laz_df['X'].min(), laz_df['X'].max() + xy_resolution, xy_resolution)
    y_bin = np.arange(laz_df['Y'].min(), laz_df['Y'].max() + xy_resolution, xy_resolution)
    z_bin = np.arange(laz_df['HeightAboveGround'].min(), laz_df['HeightAboveGround'].max() + z_resolution, z_resolution)

    histogram, edges = np.histogramdd(
        np.vstack((arr['X'], arr['Y'], arr['HeightAboveGround'])).transpose(),
        bins=(x_bin, y_bin, z_bin)
    )
    extent = [arr['X'].min(), arr['X'].max(), arr['Y'].min(), arr['Y'].max()]

    return histogram, extent


def calculate_fhd(arr, voxel_resolution):
    """
    Calculate the Foliage Height Diversity (FHD) for the given array based on voxel resolution and z resolution.

    Assigns the elements of the input array to a voxel grid, computes proportions of occupancy per voxel, and
    calculates the entropy across each voxel to measure foliage height diversity. Elements that do not fall into any
    voxel are assigned a NaN value.

    :param arr: numpy.ndarray
        The input array for which the FHD is to be calculated.
    :param voxel_resolution:
        The resolution for x, y, and z dimensions of the voxel grid.
    :type voxel_resolution: tuple of floats

    :return: numpy.ndarray
        An array containing the computed FHD values.

    :raises ValueError: If the input array is not a numpy array or if voxel resolutions are not valid.
    """
    histogram, _ = assign_voxels(arr, voxel_resolution)

    sum_counts = np.sum(histogram, axis=2, keepdims=True)

    with np.errstate(divide='ignore', invalid='ignore'):
        proportions = np.divide(
            histogram,
            sum_counts,
            out=np.zeros_like(histogram, dtype=float),
            where=sum_counts != 0
        )

    fhd = entropy(proportions, axis=2)

    fhd[sum_counts.squeeze() == 0] = np.nan

    return fhd


def calculate_pad(voxel_returns, voxel_height, beer_lambert_constant=None):
    """
    Calculate the Plant Area Density (PAD) from voxel return data using the Beer-Lambert Law.

    This function computes the PAD by performing a cumulative sum of the voxel returns, adjusting
    for shots passing through each voxel, and applying the Beer-Lambert Law.

    :param voxel_returns:
        A numpy array representing the returns from each voxel with shape (x, y, z) as calculated by assign_voxels.
    :param voxel_height:
        A float indicating the height of each voxel.
    :param beer_lambert_constant:
        (Optional) A float representing the Beer-Lambert constant. Defaults to 1 if not provided.

    :return:
        A numpy array of the same shape as voxel_returns containing the PAD values.

    :raises ValueError:
        If voxel_returns and voxel_height have incompatible shapes.
    :raises TypeError:
        If voxel_returns is not a numpy array or voxel_height is not a float.
    """
    return_accum = np.cumsum(voxel_returns[::-1], axis=2)[::-1]
    shots_in = return_accum
    shots_through = return_accum - voxel_returns

    division_result = np.divide(
        shots_in,
        shots_through,
        out=np.full(shots_in.shape, np.nan),
        where=~np.isnan(shots_through)
    )

    k = beer_lambert_constant if beer_lambert_constant else 1
    dz = voxel_height

    pad = np.log(division_result) * (1 / (k * dz))

    pad = np.where(np.isinf(pad) | np.isnan(pad), np.nan, pad)

    return pad


def calculate_pai(pad):
    """
    Calculates Plant Area Index (PAI) from the provided Plant Area Density (PAD) data.

    :param pad:
        The input array. It is expected to be a NumPy array with at least 3 dimensions.
    :type pad: numpy.ndarray

    :return:
        The sum of the input array over its third axis, with NaNs ignored.
    :rtype: numpy.ndarray

    :raises ValueError:
        If 'pad' does not have at least 3 dimensions.
    """
    return np.nansum(pad, axis=2)
