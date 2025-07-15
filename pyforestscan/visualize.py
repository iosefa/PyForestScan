import matplotlib.pyplot as plt
import numpy as np


def plot_2d(points, x_dim='X',
            y_dim='Z', color_by='HeightAboveGround',
            color_map='viridis', colorbar_label=None,
            alpha=1.0, point_size=1,
            fig_size=None, fig_title=None,
            slice_dim=None, slice_val=0.0,
            slice_tolerance=5, save_fname=None
) -> None:
    """
    Plot a 2D scatter plot of point cloud data with customizable axes, coloring, slicing, and figure settings.

    Args:
        points (np.ndarray): Structured numpy array with named columns (e.g., 'X', 'Y', 'Z', 'Classification', 'HeightAboveGround', etc.).
        x_dim (str, optional): Column name for the horizontal axis. Defaults to 'X'.
        y_dim (str, optional): Column name for the vertical axis. Defaults to 'Z'.
        color_by (str, optional): Column name to use for point color values. Defaults to 'HeightAboveGround'.
        color_map (str, optional): Name of the matplotlib colormap to use. Defaults to 'viridis'.
        colorbar_label (str, optional): Label for the colorbar. If None, uses `color_by`. Defaults to None.
        alpha (float, optional): Transparency of scatter points (0–1). Defaults to 1.0.
        point_size (int, optional): Size of the points in the scatter plot. Defaults to 1.
        fig_size (tuple, optional): Figure size as (width, height) in inches. If None, auto-derive from aspect ratio. Defaults to None.
        fig_title (str, optional): Title for the plot. If None, auto-generated. Defaults to None.
        slice_dim (str, optional): Name of the dimension to "slice" (fix) at a specific value. Defaults to None.
        slice_val (float, optional): Value to use for slicing the `slice_dim` dimension. Defaults to 0.0.
        slice_tolerance (float, optional): Allowed difference for slice matching. Defaults to 5.
        save_fname (str, optional): If provided, will be forwarded to `plt.savefig` to save the figure.

    Returns:
        None

    Raises:
        ValueError: If `x_dim` or `y_dim` is not in ['X', 'Y', 'Z', 'HeightAboveGround'].
        ValueError: If required dimension names are not present in `points.dtype.names`.
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
                save_fname=None) -> None:
    """
    Plot a 2D metric array as an image with geospatial extent, colorbar, and customizable settings.

    Args:
        title (str): Title of the plot.
        metric (np.ndarray): 2D array representing the metric values to plot.
        extent (list): List of four elements [xmin, xmax, ymin, ymax] defining the spatial extent of the plot.
        metric_name (str, optional): Label for the colorbar. If None, uses `title`. Useful for specifying units or a more detailed description.
        cmap (str, optional): Matplotlib colormap name for the plot. Defaults to 'viridis'.
        fig_size (tuple, optional): Figure size as (width, height) in inches. If None, computed from aspect ratio and extent.
        save_fname (str, optional): If provided, will be forwarded to `plt.savefig` to save the figure.

    Returns:
        None
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
             hag_values=None, horizontal_values=None, title=None,
             save_fname=None) -> None:
    """
    Visualize 3D Plant Area Density (PAD) data as a 2D image, using projection or slicing.

    Projects or slices a 3D PAD array along a specified axis and displays the result as an image with a colormap.

    Args:
        pad (np.ndarray): 3D numpy array of PAD values, shape (X, Y, dZ) or (dZ, Y, X).
        slice_index (int, optional): Index at which to take a 2D slice along the specified axis. If None, the PAD data is collapsed along the axis using the maximum value. Defaults to None.
        axis (str, optional): Axis along which to slice or project ('x' or 'y'). Defaults to 'x'.
        cmap (str, optional): Name of the matplotlib colormap for visualization. Defaults to 'viridis'.
        hag_values (np.ndarray, optional): 1D array of height-above-ground (dZ) values. If None, uses a range based on the PAD array shape.
        horizontal_values (np.ndarray, optional): 1D array of horizontal axis values (X or Y, depending on `axis`). If None, uses a range based on PAD array shape.
        title (str, optional): Title for the plot. If None, generates an appropriate title.
        save_fname (str, optional): A string that will be forwarded to `plt.savefig` to save
            the figure.

    Returns:
        None

    Raises:
        ValueError: If `axis` is not 'x' or 'y'.
        ValueError: If `slice_index` is out of range for the specified axis.
        ValueError: If the length of `horizontal_values` does not match the dimension of the specified axis.
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