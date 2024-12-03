import numpy as np
from scipy.interpolate import griddata
from scipy.stats import entropy


def generate_dtm(ground_points, resolution=2.0):
    """
    Generates a Digital Terrain Model (DTM) raster from classified ground points.

    :param ground_points: list
        Point cloud arrays of classified ground points.
    :type ground_points: list
    :param resolution: float, spatial resolution of the DTM in meters.
    :return: tuple
        A tuple containing the DTM as a 2D NumPy array and the spatial extent [x_min, x_max, y_min, y_max].
    :rtype: tuple (numpy.ndarray, list)
    :raises ValueError:
        If no ground points are found for DTM generation.
    :raises KeyError:
        If point cloud data is missing 'X', 'Y', or 'Z' fields.
    """
    #todo: add parameter to allow interpolation of NA values.
    try:
        x = np.array([pt['X'] for array in ground_points for pt in array])
        y = np.array([pt['Y'] for array in ground_points for pt in array])
        z = np.array([pt['Z'] for array in ground_points for pt in array])
    except ValueError:
        raise ValueError("Ground point cloud data missing 'X', 'Y', or 'Z' fields.")

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_bins = np.arange(x_min, x_max + resolution, resolution)
    y_bins = np.arange(y_min, y_max + resolution, resolution)

    x_indices = np.digitize(x, x_bins) - 1
    y_indices = np.digitize(y, y_bins) - 1

    dtm = np.full((len(x_bins) - 1, len(y_bins) - 1), np.nan)

    for xi, yi, zi in zip(x_indices, y_indices, z):
        if 0 <= xi < dtm.shape[0] and 0 <= yi < dtm.shape[1]:
            if np.isnan(dtm[xi, yi]) or zi < dtm[xi, yi]:
                dtm[xi, yi] = zi

    dtm = np.fliplr(dtm)

    extent = [x_min, x_max, y_min, y_max]

    return dtm, extent


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

    try:
        x = arr['X']
        y = arr['Y']
        z = arr['HeightAboveGround']
    except ValueError:
        raise ValueError("Point cloud data missing 'X', 'Y', or 'HeightAboveGround' fields.")

    if x.size == 0 or y.size == 0 or z.size == 0:
        raise ValueError("Point cloud data contains no points.")

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

    with np.errstate(divide='ignore', invalid='ignore'):
        division_result = np.true_divide(shots_in, shots_through)
        division_result = np.where((shots_in == 0) & (shots_through == 0), 1, division_result)
        division_result = np.where((shots_through == 0) & (shots_in != 0), np.nan, division_result)

        pad = np.log(division_result) * (1 / (beer_lambert_constant or 1) / voxel_height)

    pad = np.where(np.isfinite(pad) & (pad > 0), pad, 0)

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
    if min_height >= max_height:
        raise ValueError("Minimum height index must be less than maximum height index.")

    pai = np.nansum(pad[:, :, min_height:max_height], axis=2)

    return pai


def calculate_fhd(voxel_returns):
    """
    Calculate the Foliage Height Diversity (FHD) for a given set of voxel returns.

    This function computes Foliage Height Diversity by calculating the entropy
    of the voxel return proportions along the z-axis, which represents vertical structure
    in the canopy.

    :param voxel_returns:
        A numpy array of shape (x, y, z) representing voxel returns, where x and y are spatial
        dimensions, and z represents height bins (or layers along the vertical axis).

    :return:
        A numpy array of shape (x, y) representing the FHD values for each (x, y) location.
        Areas with no voxel returns will have NaN values.
    """
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


def calculate_chm(arr, voxel_resolution, interpolation="linear"):
    """
    Calculate Canopy Height Model (CHM) for a given voxel size.
    The height is the highest HeightAboveGround value in each (x, y) voxel.

    :param arr: Input array-like object containing point cloud data with 'X', 'Y', and 'HeightAboveGround' fields.
    :type arr: numpy.ndarray
    :param voxel_resolution:
        The resolution for x and y dimensions of the voxel grid.
    :param interpolation:
        method for interpolating pixel caps in the CHM. Supported methods are: "nearest", "linear", and "cubic".
    :type voxel_resolution: tuple of floats (x_resolution, y_resolution)

    :return:
        A tuple containing the CHM as a 2D numpy array and the spatial extent.
    :rtype: tuple of (numpy.ndarray, list)
    """
    x_resolution, y_resolution = voxel_resolution[:2]
    x = arr['X']
    y = arr['Y']
    z = arr['HeightAboveGround']

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_bins = np.arange(x_min, x_max + x_resolution, x_resolution)
    y_bins = np.arange(y_min, y_max + y_resolution, y_resolution)

    x_indices = np.digitize(x, x_bins) - 1
    y_indices = np.digitize(y, y_bins) - 1

    chm = np.full((len(x_bins)-1, len(y_bins)-1), np.nan)

    for xi, yi, zi in zip(x_indices, y_indices, z):
        if 0 <= xi < chm.shape[0] and 0 <= yi < chm.shape[1]:
            if np.isnan(chm[xi, yi]) or zi > chm[xi, yi]:
                chm[xi, yi] = zi

    mask = np.isnan(chm)

    x_grid, y_grid = np.meshgrid(
        (x_bins[:-1] + x_bins[1:]) / 2,
        (y_bins[:-1] + y_bins[1:]) / 2
    )

    valid_mask = ~mask.flatten()
    valid_x = x_grid.flatten()[valid_mask]
    valid_y = y_grid.flatten()[valid_mask]
    valid_values = chm.flatten()[valid_mask]

    interp_x = x_grid.flatten()[mask.flatten()]
    interp_y = y_grid.flatten()[mask.flatten()]

    chm[mask] = griddata(
        points=(valid_x, valid_y),
        values=valid_values,
        xi=(interp_x, interp_y),
        method=interpolation
    )
    chm = np.flip(chm, axis=1)
    extent = [x_min, x_max, y_min, y_max]

    return chm, extent
