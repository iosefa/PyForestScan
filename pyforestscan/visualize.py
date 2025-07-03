import matplotlib.pyplot as plt
import numpy as np


def plot_2d(points, x_dim='X',
            y_dim='Z', color_by='HeightAboveGround',
            color_map='viridis', colorbar_label=None,
            alpha=1.0, point_size=1,
            fig_size=None, fig_title=None,
            slice_dim=None, slice_val=0.0,
            slice_tolerance=5, save_fname=None
):
    """
    Plots a 2D scatter plot of data points with customizable axes, coloring, and display settings.

    Args:
        points (np.ndarray): A structured array with named columns like 'X','Y','Z','Classification', etc.
        x_dim (str): The column name for the horizontal axis. Defaults to 'X'.
        y_dim (str): The column name for the vertical axis. Defaults to 'Z'.
        color_by (str): The column name used to color the points. Defaults to 'HeightAboveGround'.
                       If 'HeightAboveGround' doesn't exist, pick e.g. 'Classification' or another column.
        color_map (str): A valid matplotlib colormap name. Defaults to 'viridis'.
        colorbar_label (str): Label for the colorbar. If None, uses color_by.
        alpha (float): Transparency of the points (0 to 1). Defaults to 1.0 (opaque).
        point_size (int): Size of the scatter points. Defaults to 1.
        fig_size (tuple): (width, height) in inches. If None, auto-derive from data aspect ratio.
        fig_title (str): Title of the plot. If None, auto-generated.
        slice_dim (str): If provided, dimension to "fix" at a slice_val (e.g. 'Y').
        slice_val (float): The coordinate value at which to slice the slice_dim dimension.
        slice_tolerance (float): Allowed absolute difference when matching slice_val.
                                 (Helps with floating-point comparisons.)
        save_fname (str): If provided, will be forwarded to `plt.savefig` to save the figure.

    Returns:
        None

    Raises:
        ValueError: If the `x_dim` or `y_dim` parameter is not one of ['X', 'Y', 'Z',
            'HeightAboveGround'].
    """
    valid_dims = ['X', 'Y', 'Z', 'HeightAboveGround']
    if x_dim not in valid_dims or y_dim not in valid_dims:
        raise ValueError(f"Invalid dimensions. Choose from: {valid_dims}")

    required_dims = [x_dim, y_dim, color_by]
    if slice_dim is not None:
        required_dims.append(slice_dim)

    for dim in required_dims:
        if dim not in points.dtype.names:
            raise ValueError(f"'{dim}' not found in array dtype names: {points.dtype.names}")

    if slice_dim and slice_dim in points.dtype.names:
        mask = np.isclose(points[slice_dim], slice_val, atol=slice_tolerance, rtol=0)
        points = points[mask]

    if colorbar_label is None:
        colorbar_label = color_by

    x = points[x_dim]
    y = points[y_dim]
    colors = points[color_by]

    if fig_size is None:
        aspect_ratio = (np.max(x) - np.min(x)) / (np.max(y) - np.min(y))
        fig_size = (10 * aspect_ratio, 10)

        max_fig_size = 20
        if max(fig_size) > max_fig_size:
            scale_factor = max_fig_size / max(fig_size)
            fig_size = (fig_size[0] * scale_factor, fig_size[1] * scale_factor)

    if fig_title is None:
        fig_title = f'{x_dim} vs {y_dim} Colored by {color_by}'

    plt.figure(figsize=fig_size)

    plt.scatter(x, y, c=colors, cmap=color_map, alpha=alpha, s=point_size)
    plt.xlabel(x_dim)
    plt.ylabel(y_dim)
    plt.title(fig_title)
    plt.colorbar(label=colorbar_label)
    if save_fname is not None:
        plt.savefig(save_fname, dpi=300, bbox_inches='tight')
    plt.show()


def plot_metric(title, metric, extent, metric_name=None, cmap='viridis', fig_size=None,
                save_fname=None):
    """
    Plots a given metric using the provided data and configuration.

    :param title: string
        The name of the metric to be plotted
    :param metric: ndarray
        2D array representing the metric's values.
    :param extent: list
        List of four elements [xmin, xmax, ymin, ymax] defining the extent of the plot.
    :param metric_name: string, optional
        Label to be used for the colorbar. If None, the title is used as the
        metric name. This is useful for specifying units or a more detailed
        description of the metric.
    :param cmap: str, optional
        Colormap to be used for the plot. Default is 'viridis'.
    :param fig_size: tuple, optional
        Tuple specifying the size of the figure (width, height). Default is calculated based on the extent.
    :param save_fname: str, optional
        If provided, will be forwarded to `plt.savefig` to save the figure.
    :return: None
    :rtype: None
    """
    if metric_name is None:
        metric_name = title
    if fig_size is None:
        x_range = extent[1] - extent[0]
        y_range = extent[3] - extent[2]
        aspect_ratio = x_range / y_range
        fig_size = (10 * aspect_ratio, 10)

        max_fig_size = 20
        if max(fig_size) > max_fig_size:
            scale_factor = max_fig_size / max(fig_size)
            fig_size = (fig_size[0] * scale_factor, fig_size[1] * scale_factor)

    plt.figure(figsize=fig_size)

    plt.imshow(metric.T, extent=extent, cmap=cmap)
    plt.colorbar(label=metric_name)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    if save_fname is not None:
        plt.savefig(save_fname, dpi=300, bbox_inches='tight')
    plt.show()


def plot_pad(pad, slice_index=None, axis='x', cmap='viridis',
             hag_values=None, horizontal_values=None, title=None, save_fname=None):
    """
    Plots the plant area density (PAD) data as a 2D image visualization.

    This function projects or slices 3D PAD data based on the specified axis and
    optional slice index. It visualizes the density using a colormap and displays
    the data with respect to the specified coordinates and labels.

    Args:
        pad: A 3D numpy array representing PAD values (shape: dZ x Y x X).
        slice_index: Optional; An integer specifying the index of the 2D slice
            along the given axis. If None, the PAD data will be collapsed along
            the axis using the maximum value.
        axis: A string specifying the axis ('x' or 'y') to use for slicing
            or projection. Defaults to 'x'.
        cmap: A string specifying the colormap to use for visualization.
            Defaults to 'viridis'.
        hag_values: Optional; A 1D numpy array representing height above ground
            (dZ) values. If None, an array from 0 to the size of the dZ axis
            will be used.
        horizontal_values: Optional; A 1D numpy array representing horizontal
            axis values (X or Y, depending on `axis`). If None, an array from
            0 to the corresponding size of the horizontal axis will be used.
        title: Optional; A string specifying the title of the plot. If None, an
            appropriate default title will be generated based on the input
            parameters.
        save_fname: Optional; A string that will be forwarded to `plt.savefig` to save
            the figure.

    Returns:
        None. The function visualizes the PAD data using matplotlib.

    Raises:
        ValueError: If `axis` is not 'x' or 'y'.
        ValueError: If `slice_index` is out of the valid range for the specified
            axis.
        ValueError: If the length of `horizontal_values` does not match the
            dimension of the specified axis.
    """
    # Validate axis
    if axis not in ['x', 'y']:
        raise ValueError(f"Invalid axis: '{axis}'. Choose from 'x' or 'y'.")

    hag_values = hag_values if hag_values is not None else np.arange(pad.shape[2])

    if axis == 'x':
        if slice_index is None:
            pad_2d = pad.max(axis=0)
            horizontal_axis_label = 'Y'
            horizontal_count = pad.shape[1]
            if horizontal_values is not None and len(horizontal_values) != horizontal_count:
                raise ValueError("Length of horizontal_values does not match the Y dimension of pad.")
            horizontal_axis_values = horizontal_values if horizontal_values is not None else np.arange(horizontal_count)

        else:
            if slice_index < 0 or slice_index >= pad.shape[0]:
                raise ValueError(f"slice_index {slice_index} out of range for axis 'x' with size {pad.shape[0]}")
            pad_2d = pad[slice_index, :, :]
            horizontal_axis_label = 'Y'
            horizontal_axis_values = horizontal_values if horizontal_values is not None else np.arange(pad.shape[1])

    else:  # axis == 'y'
        if slice_index is None:
            pad_2d = np.nanmax(pad, axis=1)
            horizontal_axis_label = 'X'
            horizontal_count = pad.shape[0]
            if horizontal_values is not None and len(horizontal_values) != horizontal_count:
                raise ValueError("Length of horizontal_values does not match the X dimension of pad.")
            horizontal_axis_values = horizontal_values if horizontal_values is not None else np.arange(horizontal_count)

        else:
            if slice_index < 0 or slice_index >= pad.shape[1]:
                raise ValueError(f"slice_index {slice_index} out of range for axis 'y' with size {pad.shape[1]}")
            pad_2d = pad[:, slice_index, :]
            horizontal_axis_label = 'X'
            horizontal_axis_values = horizontal_values if horizontal_values is not None else np.arange(pad.shape[0])

    pad_2d = pad_2d.T

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
    if title is None:
        if slice_index is None:
            title = f'Plant Area Density (PAD) - Collapsed by max along axis {axis}'
        else:
            title = f'Plant Area Density (PAD) - {horizontal_axis_label} vs dZ at slice {slice_index}'
    plt.title(title)
    plt.xlabel(horizontal_axis_label)
    plt.ylabel('dZ')
    plt.tight_layout()
    if save_fname is not None:
        plt.savefig(save_fname, dpi=300, bbox_inches='tight')
    plt.show()