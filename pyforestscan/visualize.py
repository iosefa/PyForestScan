import matplotlib.pyplot as plt
import numpy as np

from mayavi import mlab


def plot_2d(points, x_dim='X', y_dim='Z', color_map='viridis', alpha=1.0, point_size=1, fig_size=None):
    """
    Generate a 2D scatter plot of point cloud data.

    Args:
        points (pandas.DataFrame): Data frame containing point cloud coordinates and attributes.
        x_dim (str, optional): The dimension to plot along the x-axis. Defaults to 'X'.
        y_dim (str, optional): The dimension to plot along the y-axis. Defaults to 'Z'.
        color_map (str, optional): The colormap used for coloring points. Defaults to 'viridis'.
        alpha (float, optional): Opacity of points. Defaults to 1.0.
        point_size (int, optional): Size of points. Defaults to 1.
        fig_size (tuple, optional): Size of the figure. Calculated based on data if None. Defaults to None.

    Raises:
        ValueError: If invalid dimensions are provided.

    Example:
        >>> plot_2d(df, x_dim='X', y_dim='Z')
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
    Generate a 3D scatter plot of point cloud data using Mayavi.

    Args:
        arrays (list): List of NumPy structured arrays containing point cloud data.
        z_dim (str, optional): The dimension to plot along the z-axis. Defaults to 'Z'.
        fig_size (tuple, optional): Size of the Mayavi window. Calculated based on data if None. Defaults to None.

    Raises:
        ValueError: If invalid dimensions are provided.

    Example:
        >>> plot_3d([array1, array2], z_dim='Z')
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
    mlab.show()


def plot_lai(lai, extent, cmap='viridis', fig_size=None):
    """
    Plot the Leaf Area Index (LAI) on a 2D grid.

    Args:
        lai (numpy.ndarray): 2D array of LAI values.
        extent (tuple): (xmin, xmax, ymin, ymax) defining the spatial extent of the data.
        cmap (str, optional): The colormap used for coloring the LAI. Defaults to 'viridis'.
        fig_size (tuple, optional): Size of the figure. Calculated based on data if None. Defaults to None.

    Example:
        >>> plot_lai(lai_array, (0, 10, 0, 10))
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

    plt.imshow(lai, extent=extent, cmap=cmap)
    plt.colorbar(label='LAI')
    plt.title('Leaf Area Index (LAI)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def plot_lad_2d(lad, slice_index, axis='x', cmap='viridis'):
    """
    Plot a 2D slice of the Leaf Area Density (LAD) on a grid.

    Args:
        lad (numpy.ndarray): 3D array of LAD values.
        slice_index (int): Index at which to slice the 3D LAD array.
        axis (str, optional): Axis along which to slice ('x' or 'y'). Defaults to 'x'.
        cmap (str, optional): The colormap used for coloring the LAD. Defaults to 'viridis'.

    Raises:
        ValueError: If invalid axis is provided.

    Example:
        >>> plot_lad_2d(lad_array, 5, axis='x')
    """
    if axis == 'x':
        lad_2d = lad[slice_index, :, :]
    elif axis == 'y':
        lad_2d = lad[:, slice_index, :]
    else:
        raise ValueError(f"Invalid axis: {axis}. Choose from 'x', 'y'.")

    plt.imshow(lad_2d, cmap=cmap)
    plt.colorbar(label='LAD')
    plt.title(f'Leaf Area Density (LAD) - {axis.upper()} vs HAG slice')
    plt.xlabel('HAG' if axis == 'x' else 'X')
    plt.ylabel('HAG' if axis == 'y' else 'Y')
    plt.show()
