import pandas as pd
import numpy as np


def assign_voxels(arr, voxel_resolution, z_resolution):
    """
    Assigns voxel bins to 3D point cloud data.

    Args:
        arr (np.ndarray): The input array containing 3D point cloud data with fields 'X', 'Y', and 'HeightAboveGround'.
        voxel_resolution (float): The spatial resolution of the voxel grid in the x and y dimensions.
        z_resolution (float): The spatial resolution of the voxel grid in the z dimension.

    Returns:
        tuple: A tuple containing the following:
            - histogram (np.ndarray): 3D histogram array showing the density of points in each voxel.
            - extent (list): The minimum and maximum coordinates for x and y in the format [xmin, xmax, ymin, ymax].

    Example:
        >>> arr = np.array([(1, 2, 3), (2, 3, 4), (3, 4, 5)], dtype=[('X', 'f4'), ('Y', 'f4'), ('HeightAboveGround', 'f4')])
        >>> assign_voxels(arr, 1, 1)
        (array([[[0., 1.],
                 [0., 0.]],
                [[0., 0.],
                 [0., 0.]]]), [1.0, 3.0, 2.0, 4.0])
    """
    laz_df = pd.DataFrame(arr)

    x_bin = np.arange(laz_df['X'].min(), laz_df['X'].max() + voxel_resolution, voxel_resolution)
    y_bin = np.arange(laz_df['Y'].min(), laz_df['Y'].max() + voxel_resolution, voxel_resolution)
    z_bin = np.arange(laz_df['HeightAboveGround'].min(), laz_df['HeightAboveGround'].max() + z_resolution, z_resolution)

    histogram, edges = np.histogramdd(np.vstack((arr['X'], arr['Y'], arr['HeightAboveGround'])).transpose(),
                                      bins=(x_bin, y_bin, z_bin))
    extent = [arr['X'].min(), arr['X'].max(), arr['Y'].min(), arr['Y'].max()]

    return histogram, extent


def calculate_lad(voxel_returns, voxel_height, beer_lambert_constant=None):
    """
    Calculates the Leaf Area Density (LAD) using voxelized return data.

    Args:
        voxel_returns (np.ndarray): 3D array where each element indicates the number of returns for that voxel.
        voxel_height (float): The height of each voxel.
        beer_lambert_constant (float, optional): The Beer-Lambert extinction coefficient. Defaults to 1.

    Returns:
        np.ndarray: 3D array containing the Leaf Area Density (LAD) for each voxel.

    Example:
        >>> voxel_returns = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        >>> calculate_lad(voxel_returns, 1)
        array([[[...], [...]],
               [[...], [...]]])
    """
    return_accum = np.cumsum(voxel_returns[::-1], axis=2)[::-1]
    shots_in = return_accum
    shots_through = return_accum - voxel_returns

    division_result = np.divide(shots_in, shots_through, out=np.full(shots_in.shape, np.nan),
                                where=~np.isnan(shots_through))

    k = beer_lambert_constant if beer_lambert_constant else 1
    dz = voxel_height

    lad = np.log(division_result) * (1 / (k * dz))

    lad = np.where(np.isinf(lad) | np.isnan(lad), np.nan, lad)

    return lad


def calculate_lai(lad):
    """
    Calculates the Leaf Area Index (LAI) from the Leaf Area Density (LAD).

    Args:
        lad (np.ndarray): 3D array containing the Leaf Area Density (LAD) for each voxel.

    Returns:
        np.ndarray: 2D array containing the Leaf Area Index (LAI) for each x, y coordinate.

    Example:
        >>> lad = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        >>> calculate_lai(lad)
        array([[6, 8],
               [12, 16]])
    """
    return np.nansum(lad, axis=2)
