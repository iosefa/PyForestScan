import matplotlib.pyplot as plt
import numpy as np

from mayavi import mlab


def plot_2d(points, x_dim='X', y_dim='Z', color_map='viridis', alpha=1.0, point_size=1, fig_size=None):
    """
    Plots a 2D scatter plot from a set of points with customizable dimensions, colors, and sizes.

    :param points: A DataFrame containing point data with dimensions and 'HeightAboveGround'.
    :type points: pandas.DataFrame
    :param x_dim: The dimension for the X-axis ('X', 'Y', 'Z', or 'HeightAboveGround').
    :type x_dim: str
    :param y_dim: The dimension for the Y-axis ('X', 'Y', 'Z', or 'HeightAboveGround').
    :type y_dim: str
    :param color_map: The colormap used to color the points according to 'HeightAboveGround'.
    :type color_map: str
    :param alpha: The transparency level of the points.
    :type alpha: float
    :param point_size: The size of each point in the scatter plot.
    :type point_size: float
    :param fig_size: The size of the figure as a tuple (width, height). If None, it is computed based on the aspect ratio.
    :type fig_size: tuple or None
    :return: None
    :rtype: None
    :raises ValueError: If the provided dimensions are not valid.
    """
    valid_dims = ['X', 'Y', 'Z', 'HeightAboveGround']
    if x_dim not in valid_dims or y_dim not in valid_dims:
        raise ValueError(f"Invalid dimensions. Choose from: {valid_dims}")

    x = points[x_dim]
    y = points[y_dim]
    colors = points['HeightAboveGround']

    if fig_size is None:
        aspect_ratio = (np.max(x) - np.min(x)) / (np.max(y) - np.min(y))
        fig_size = (10 * aspect_ratio, 10)

        max_fig_size = 20  # inches
        if max(fig_size) > max_fig_size:
            scale_factor = max_fig_size / max(fig_size)
            fig_size = (fig_size[0] * scale_factor, fig_size[1] * scale_factor)

    plt.figure(figsize=fig_size)

    plt.scatter(x, y, c=colors, cmap=color_map, alpha=alpha, s=point_size)
    plt.xlabel(x_dim)
    plt.ylabel(y_dim)
    plt.title(f'{x_dim} vs {y_dim} Colored by Height Above Ground')
    plt.colorbar(label='Height Above Ground (m)')
    plt.show()


def plot_3d(arrays, z_dim='Z', fig_size=None):
    """
    Plots a 3D visualization of the given array data.

    :param arrays: A list of dictionaries containing point data with keys 'X', 'Y', and specified z_dim.
    :type arrays: list
    :param z_dim: Dimension to be used for the Z-axis. Must be 'Z' or 'HeightAboveGround'. Defaults to 'Z'.
    :type z_dim: str, optional
    :param fig_size: Size of the figure to be plotted. If None, the size will be computed automatically based on data.
    :type fig_size: tuple, optional
    :return: None
    :rtype: None
    :raises ValueError: If the provided z_dim is not in the list of valid dimensions.
    """
    valid_dims = ['Z', 'HeightAboveGround']
    if z_dim not in valid_dims:
        raise ValueError(f"Invalid dimensions. Choose from: {valid_dims}")

    points = arrays[0]
    x = points['X']
    y = points['Y']
    z = points[z_dim]
    colors = points['HeightAboveGround']

    if fig_size is None:
        aspect_ratio = (np.max(x) - np.min(x)) / (np.max(y) - np.min(y))
        fig_size = (800 * aspect_ratio, 800)

        max_fig_size = 1600  # pixels
        if max(fig_size) > max_fig_size:
            scale_factor = max_fig_size / max(fig_size)
            fig_size = (fig_size[0] * scale_factor, fig_size[1] * scale_factor)

    fig = mlab.figure(size=fig_size)

    pts = mlab.points3d(x, y, z, colors, colormap='viridis', scale_mode='none', scale_factor=0.5)
    mlab.axes()
    mlab.title('3D Point Cloud Colored by Height Above Ground')
    mlab.show()


def plot_pai(pai, extent, cmap='viridis', fig_size=None):
    """
    Plots the Plant Area Index (PAI) using the provided data and configuration.

    :param pai: ndarray
        2D array representing the PAI values.
    :param extent: list
        List of four elements [xmin, xmax, ymin, ymax] defining the extent of the plot.
    :param cmap: str, optional
        Colormap to be used for the plot. Default is 'viridis'.
    :param fig_size: tuple, optional
        Tuple specifying the size of the figure (width, height). Default is calculated based on the extent.
    :return: None
    :rtype: None
    """
    if fig_size is None:
        x_range = extent[1] - extent[0]
        y_range = extent[3] - extent[2]
        aspect_ratio = x_range / y_range
        fig_size = (10 * aspect_ratio, 10)

        max_fig_size = 20  # inches
        if max(fig_size) > max_fig_size:
            scale_factor = max_fig_size / max(fig_size)
            fig_size = (fig_size[0] * scale_factor, fig_size[1] * scale_factor)

    plt.figure(figsize=fig_size)

    plt.imshow(pai, extent=extent, cmap=cmap)
    plt.colorbar(label='PAI')
    plt.title('Plant Area Index (PAI)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def plot_pad_2d(pad, slice_index, axis='x', cmap='viridis'):
    """
    Plots a 2D slice of Plant Area Density (PAD) data.

    :param pad: numpy.ndarray, The 3D Plant Area Density data.
    :param slice_index: int, The index of the slice to be plotted.
    :param axis: str, optional, The axis along which to slice the data. Default is 'x'. Choose from 'x', 'y'.
    :param cmap: str, optional, The colormap to be used for plotting. Default is 'viridis'.
    :raises ValueError: If an invalid axis is provided.
    :return: None
    :rtype: None
    """
    if axis == 'x':
        pad_2d = pad[slice_index, :, :]
    elif axis == 'y':
        pad_2d = pad[:, slice_index, :]
    else:
        raise ValueError(f"Invalid axis: {axis}. Choose from 'x', 'y'.")

    plt.imshow(pad_2d, cmap=cmap)
    plt.colorbar(label='PAD')
    plt.title(f'Plant Area Density (PAD) - {axis.upper()} vs HAG slice')
    plt.xlabel('HAG' if axis == 'x' else 'X')
    plt.ylabel('HAG' if axis == 'y' else 'Y')
    plt.show()
