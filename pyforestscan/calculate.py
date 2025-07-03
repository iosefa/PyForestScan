import numpy as np
from scipy.interpolate import griddata
from scipy.stats import entropy
from scipy import ndimage


def generate_dtm(ground_points, resolution=2.0):
    """
    Generates a Digital Terrain Model (DTM) raster from classified ground points.

    Args:
        ground_points (list): Point cloud arrays of classified ground points.
        resolution (float): Spatial resolution of the DTM in meters.

    Returns:
        tuple: A tuple containing the DTM as a 2D NumPy array and the spatial extent [x_min, x_max, y_min, y_max].

    Raises:
        ValueError: If no ground points are found for DTM generation.
        KeyError: If point cloud data is missing 'X', 'Y', or 'Z' fields.
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

    Args:
        arr (numpy.ndarray): Input array-like object containing point cloud data with 'X', 'Y', and 'HeightAboveGround' fields.
        voxel_resolution (tuple of floats): The resolution for x, y, and z dimensions of the voxel grid.

    Returns:
        tuple of (numpy.ndarray, list): A tuple containing the histogram of the voxel grid (with corrected orientation) and the extent of the point cloud.
    """
    dx, dy, dz = voxel_resolution

    pts = arr[arr['HeightAboveGround'] >= 0]

    x0 = np.floor(pts['X'].min() / dx) * dx
    y0 = np.ceil (pts['Y'].max() / dy) * dy

    x_bins = np.arange(x0, pts['X'].max() + dx, dx)
    y_bins = np.arange(y0, pts['Y'].min() - dy, -dy)
    z_bins = np.arange(0.0, pts['HeightAboveGround'].max() + dz, dz)

    hist, _ = np.histogramdd(
        np.column_stack((pts['X'], pts['Y'], pts['HeightAboveGround'])),
        bins=(x_bins, y_bins[::-1], z_bins)
    )
    hist = hist[:, ::-1, :]

    extent = [x_bins[0], x_bins[-1], y_bins[-1], y_bins[0]]
    return hist, extent


def calculate_pad(voxel_returns,
                  voxel_height=1.0,
                  beer_lambert_constant=1.0,
                  drop_ground=True
                  ):
    """
    Calculate the Plant Area Density (PAD) using the Beer-Lambert Law.

    Args:
        voxel_returns (np.ndarray): 3D numpy array of shape (X, Y, Z) representing
            the LiDAR returns in each voxel column.
        voxel_height (float, optional): Height of each voxel. Defaults to 1.0.
        beer_lambert_constant (float, optional): The Beer-Lambert constant used
            in the calculation. Defaults to 1.0.
        drop_ground (bool, optional): If True, sets PAD values in the ground (lowest)
            voxel layer to NaN in the output. Defaults to True.

    Returns:
        np.ndarray: 3D numpy array containing PAD values for each voxel, same shape as `voxel_returns`.
            Columns that have zero returns across all Z are set to NaN.
    """
    if voxel_height <= 0:
        raise ValueError(
            f"voxel_height must be > 0 metres (got {voxel_height})"
        )
    reversed_cols = voxel_returns[:, :, ::-1]

    total = np.sum(reversed_cols, axis=2, keepdims=True)

    csum = np.cumsum(reversed_cols, axis=2)

    shots_out = total - csum

    shots_in = np.concatenate(
        (total, shots_out[:, :, :-1]), axis=2
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        pad_sky = np.log(shots_in / shots_out) / (beer_lambert_constant * voxel_height)
    pad_sky[~np.isfinite(pad_sky)] = np.nan

    pad = pad_sky[:, :, ::-1]

    if drop_ground:
        pad[:, :, 0] = np.nan

    pad[voxel_returns[:, :, 0] == 0, :] = np.nan

    return pad


def calculate_pai(pad,
                  voxel_height,
                  min_height=1.0,
                  max_height=None):
    """
    Calculate Plant Area Index (PAI) from Plant Area Density (PAD) data by summing PAD values along the height (Z) axis.

    Args:
        pad (np.ndarray): 3D numpy array representing Plant Area Density (PAD) values, shape (X, Y, Z).
        voxel_height (float): Height of each voxel in meters.
        min_height (float, optional): Minimum height in meters for summing PAD values. Defaults to 1.0.
        max_height (float, optional): Maximum height in meters for summing PAD values. If None, uses the full height of the input array. Defaults to None.

    Returns:
        np.ndarray: 2D numpy array of shape (X, Y) with PAI values for each (x, y) voxel column.

    Raises:
        ValueError: If min_height is greater than or equal to max_height.
    """
    if max_height is None:
        max_height = pad.shape[2] * voxel_height

    if min_height >= max_height:
        raise ValueError("Minimum height index must be less than maximum height index.")

    start_idx = int(np.ceil(min_height / voxel_height))
    end_idx   = int(np.floor(max_height / voxel_height))

    core = pad[:, :, start_idx:end_idx]
    pai  = np.nansum(core, axis=2) * voxel_height
    return pai


def calculate_fhd(voxel_returns):
    """
    Calculate the Foliage Height Diversity (FHD) for a given set of voxel returns.

    This function computes FHD by calculating the entropy of the voxel return proportions
    along the Z (height) axis, which represents the vertical diversity of canopy structure.

    Args:
        voxel_returns (np.ndarray): 3D numpy array of shape (X, Y, Z) representing voxel returns,
            where X and Y are spatial dimensions and Z represents height bins (vertical layers).

    Returns:
        np.ndarray: 2D numpy array of shape (X, Y) with FHD values for each (X, Y) location.
            Areas with no voxel returns will have NaN values.
    """
    sum_counts = np.sum(voxel_returns, axis=2)

    with np.errstate(divide='ignore', invalid='ignore'):
        proportions = np.divide(
            voxel_returns,
            sum_counts[..., None],
            out=np.zeros_like(voxel_returns, dtype=float),
            where=sum_counts[..., None] != 0
        )

    fhd = entropy(proportions, axis=2)
    fhd[sum_counts == 0] = np.nan
    return fhd


def calculate_chm(arr, voxel_resolution, interpolation="linear",
                  interp_valid_region=False, interp_clean_edges=False):
    """
    Calculate the Canopy Height Model (CHM) for a given voxel grid.

    The CHM is computed as the maximum 'HeightAboveGround' value within each (X, Y) voxel.
    Optionally, gaps in the CHM can be filled using interpolation.

    Args:
        arr (np.ndarray): Input structured numpy array containing point cloud data
            with fields 'X', 'Y', and 'HeightAboveGround'.
        voxel_resolution (tuple of float): The resolution for the X and Y dimensions
            of the voxel grid, specified as (x_resolution, y_resolution).
        interpolation (str or None, optional): Method for interpolating gaps in the CHM.
            Supported methods are "nearest", "linear", "cubic", or None. If None, no interpolation
            is performed. Defaults to "linear".
        interp_valid_region (bool): Whether to calculate a valid region mask using morphological operations for
            interpolation. If True, interpolation is only applied within the valid data region. If False (default),
            interpolation is applied to all NaN values. Ignored if `interpolation` is None.
        interp_clean_edges (bool): Whether to clean edge fringes of the interpolated CHM. Default is False.
            Ignored if `interpolation` is None.

    Returns:
        tuple:
            - np.ndarray: 2D numpy array representing the CHM, with each value corresponding to the maximum
                height in that (X, Y) voxel.
            - list: The spatial extent as [x_min, x_max, y_min, y_max].

    Raises:
        ValueError: If input array does not contain the required fields.
        ValueError: If `interpolation` is specified but not one of the supported methods.

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

    chm = np.full((len(x_bins) - 1, len(y_bins) - 1), np.nan)

    for xi, yi, zi in zip(x_indices, y_indices, z):
        if 0 <= xi < chm.shape[0] and 0 <= yi < chm.shape[1]:
            if np.isnan(chm[xi, yi]) or zi > chm[xi, yi]:
                chm[xi, yi] = zi

    if interpolation is not None:
        if interp_valid_region is True:
            valid_region_mask = _calc_valid_region_mask(chm)
            interp_mask = np.isnan(chm) & valid_region_mask
        else:
            interp_mask = np.isnan(chm)

        if np.any(interp_mask):
            x_grid, y_grid = np.meshgrid(
                (x_bins[:-1] + x_bins[1:]) / 2,
                (y_bins[:-1] + y_bins[1:]) / 2
            )

            valid_mask = ~np.isnan(chm)
            valid_x = x_grid.flatten()[valid_mask.flatten()]
            valid_y = y_grid.flatten()[valid_mask.flatten()]
            valid_values = chm.flatten()[valid_mask.flatten()]

            interp_coords = np.column_stack([
                x_grid.flatten()[interp_mask.flatten()],
                y_grid.flatten()[interp_mask.flatten()]
            ])

            if len(interp_coords) > 0 and len(valid_values) > 0:
                chm[interp_mask] = griddata(
                    points=np.column_stack([valid_x, valid_y]),
                    values=valid_values,
                    xi=interp_coords,
                    method=interpolation
                )
            if interp_clean_edges:
                chm = _clean_edges(chm)

    chm = np.flip(chm, axis=1)
    extent = [x_min, x_max, y_min, y_max]

    return chm, extent


def _calc_valid_region_mask(arr):
    """Create valid region mask using morphological operations."""
    kernel = ndimage.generate_binary_structure(2, 1)
    valid_cells = ~np.isnan(arr)
    valid_mask = ndimage.binary_dilation(valid_cells, structure=kernel)
    valid_mask = ndimage.binary_fill_holes(valid_mask, structure=kernel)
    return valid_mask


def _clean_edges(arr, distance=5):
    """Clean edges by excluding cells within a certain distance from the edge."""
    kernel = ndimage.generate_binary_structure(2, 1)
    valid_cells = ~np.isnan(arr)
    valid_mask = ndimage.binary_erosion(valid_cells, structure=kernel, iterations=2)
    distance_from_edge = ndimage.distance_transform_edt(valid_mask)
    valid_mask = (distance_from_edge >= distance) & valid_mask
    arr_cleaned = np.where(valid_mask, arr, np.nan)
    return arr_cleaned
