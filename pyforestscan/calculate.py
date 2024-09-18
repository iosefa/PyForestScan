import numpy as np
from scipy.stats import entropy


def assign_voxels(arr, voxel_resolution):
    """
    Assigns voxel grids to spatial data points based on the specified resolutions.

    :param arr:
        Input array-like object containing point cloud data with 'X', 'Y', and 'HeightAboveGround' fields.
    :type arr: numpy.ndarray
    :param voxel_resolution:
        The resolution for x, y, and z dimensions of the voxel grid.
    :type voxel_resolution: tuple of floats

    :return:
        A tuple containing the histogram of the voxel grid (with corrected orientation) and the extent of the point cloud.
    :rtype: tuple of (numpy.ndarray, list)
    """
    x_resolution, y_resolution, z_resolution = voxel_resolution

    x = arr['X']
    y = arr['Y']
    z = arr['HeightAboveGround']

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()

    x_bins = np.arange(x_min, x_max + x_resolution, x_resolution)
    y_bins = np.arange(y_min, y_max + y_resolution, y_resolution)
    z_bins = np.arange(z_min, z_max + z_resolution, z_resolution)

    histogram, edges = np.histogramdd(
        np.stack((x, y, z), axis=-1),
        bins=(x_bins, y_bins, z_bins)
    )
    histogram = histogram[:, ::-1, :]

    extent = [x_min, x_max, y_min, y_max]

    return histogram, extent


def calculate_pad(voxel_returns, voxel_height, beer_lambert_constant=None):
    """
    Calculate the Plant Area Density (PAD) using the Beer-Lambert Law.

    :param voxel_returns: numpy array representing the returns from each voxel (x, y, z).
    :param voxel_height: float, height of each voxel.
    :param beer_lambert_constant: Optional Beer-Lambert constant, defaults to 1 if not provided.

    :return: numpy array containing PAD values for each voxel.
    """
    shots_in = np.cumsum(voxel_returns[::-1], axis=2)[::-1]
    shots_through = shots_in - voxel_returns

    division_result = np.divide(
        shots_in,
        shots_through,
        out=np.ones(shots_in.shape),
        where=shots_through != 0
    )

    k = beer_lambert_constant if beer_lambert_constant else 1
    dz = voxel_height

    pad = np.log(division_result) * (1 / (k * dz))

    pad = np.where(np.isinf(pad) | np.isnan(pad) | (pad < 0), 0, pad)

    return pad


def calculate_pai(pad, min_height=1, max_height=None):
    """
    Calculate Plant Area Index (PAI) from PAD data by summing LAD values across the Z (height) axis.

    :param pad: A 3D numpy array representing the Plant Area Density (PAD) values.
    :param min_height: Minimum height index for summing PAD values (optional).
    :param max_height: Maximum height index for summing PAD values (optional).

    :return: A 2D numpy array with PAI values for each (x, y) voxel column.
    """
    if max_height is None:
        max_height = pad.shape[2]
    pai = np.sum(pad[:, :, min_height:max_height], axis=2)

    return pai


def calculate_fhd(voxel_returns):
    """
    Calculate the Frequency Histogram Diversity (FHD) for a given set of voxel returns.

    This function computes the Frequency Histogram Diversity by calculating the entropy
    of the voxel return proportions along the z-axis.

    :param voxel_returns:
        A numpy array of shape (x, y, z) representing voxel returns.

    :return:
        A numpy array of shape (x, y"""
    sum_counts = np.sum(voxel_returns, axis=2, keepdims=True)

    with np.errstate(divide='ignore', invalid='ignore'):
        proportions = np.divide(
            voxel_returns,
            sum_counts,
            out=np.zeros_like(voxel_returns, dtype=float),
            where=sum_counts != 0
        )

    fhd = entropy(proportions, axis=2)

    fhd[sum_counts.squeeze() == 0] = np.nan

    return fhd
