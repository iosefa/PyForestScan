import matplotlib.pyplot as plt
import numpy as np


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

    plt.imshow(pai.T, extent=extent, cmap=cmap)
    plt.colorbar(label='PAI')
    plt.title('Plant Area Index (PAI)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def plot_metric(title, metric, extent, cmap='viridis', fig_size=None):
    """
    Plots a given metric using the provided data and configuration.

    :param title: string
        The name of the metric to be plotted
    :param metric: ndarray
        2D array representing the metric's values.
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

    plt.imshow(metric.T, extent=extent, cmap=cmap)
    plt.colorbar(label=title)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def plot_pad(pad, slice_index, axis='x', cmap='viridis', hag_values=None, horizontal_values=None):
    """
    Plots a 2D slice of Plant Area Density (PAD) data with dZ HAG on the Y-axis.

    :param pad: numpy.ndarray
        The 3D Plant Area Density data with shape (X, Y, HAG).
    :param slice_index: int
        The index of the slice to be plotted along the specified axis.
    :param axis: str, optional
        The axis along which to slice the data. Default is 'x'. Choose from 'x', 'y'.
    :param cmap: str, optional
        The colormap to be used for plotting. Default is 'viridis'.
    :param hag_values: numpy.ndarray, optional
        Array of HAG (Height Above Ground) values corresponding to the third dimension.
        If None, will use indices.
    :param horizontal_values: numpy.ndarray, optional
        Array of horizontal (X or Y) values corresponding to the first or second dimension.
        If None, will use indices.
    :raises ValueError: If an invalid axis is provided or slice_index is out of bounds.
    :return: None
    :rtype: None
    """
    # Validate the axis parameter
    if axis not in ['x', 'y']:
        raise ValueError(f"Invalid axis: '{axis}'. Choose from 'x' or 'y'.")

    if axis == 'x':
        if slice_index < 0 or slice_index >= pad.shape[0]:
            raise ValueError(f"slice_index {slice_index} out of range for axis 'x' with size {pad.shape[0]}")
        pad_2d = pad[slice_index, :, :]
        horizontal_axis_label = 'Y'
        horizontal_axis_values = horizontal_values if horizontal_values is not None else np.arange(pad.shape[1])
    else:  # axis == 'y'
        if slice_index < 0 or slice_index >= pad.shape[1]:
            raise ValueError(f"slice_index {slice_index} out of range for axis 'y' with size {pad.shape[1]}")
        pad_2d = pad[:, slice_index, :]
        horizontal_axis_label = 'X'
        horizontal_axis_values = horizontal_values if horizontal_values is not None else np.arange(pad.shape[0])

    hag_values = hag_values if hag_values is not None else np.arange(pad.shape[2])

    pad_2d = pad_2d.T

    if horizontal_values is not None:
        if axis == 'x' and len(horizontal_values) != pad.shape[1]:
            raise ValueError("Length of horizontal_values does not match the Y dimension of pad.")
        if axis == 'y' and len(horizontal_values) != pad.shape[0]:
            raise ValueError("Length of horizontal_values does not match the X dimension of pad.")
    else:
        horizontal_axis_values = np.arange(pad.shape[1]) if axis == 'x' else np.arange(pad.shape[0])

    plt.figure(figsize=(10, 6))
    img = plt.imshow(
        pad_2d,
        cmap=cmap,
        origin='lower',
        extent=(
            horizontal_axis_values.min(),
            horizontal_axis_values.max(),
            hag_values.min(),
            hag_values.max()
        ),
        aspect='auto'
    )
    plt.colorbar(img, label='PAD')
    plt.title(f'Plant Area Density (PAD) - {horizontal_axis_label} vs dZ at Slice {slice_index}')
    plt.xlabel(horizontal_axis_label)
    plt.ylabel('dZ')
    plt.tight_layout()
    plt.show()