import matplotlib.pyplot as plt
import numpy as np

from mayavi import mlab


def plot_2d(points, x_dim='X', y_dim='Z', color_map='viridis', alpha=1.0, point_size=1, fig_size=None):
    """
    Creates a 2D scatter plot with colormap, adjustable opacity, point size and figure dimensions.

    Parameters:
        points (np.ndarray): A pdal points.
        x_dim (str): The x dimension to be plotted. Default is 'X'.
        y_dim (str): The y dimension to be plotted. Default is 'Z'.
        color_map (str): The color map to use for color values. Default is 'viridis'.
        alpha (float): Opacity of points. Default value is 1.0.
        point_size (float): Size of points. Default value is 1.
        fig_size (tuple): Figure size in inches. Default value is None, and in this case it will be determined by the data's aspect ratio.
    """
    # Check that the requested dimensions are valid
    valid_dims = ['X', 'Y', 'Z', 'HeightAboveGround']
    if x_dim not in valid_dims or y_dim not in valid_dims:
        raise ValueError(f"Invalid dimensions. Choose from: {valid_dims}")

    # Prepare data
    x = points[x_dim]
    y = points[y_dim]
    colors = points['HeightAboveGround']

    # Determine figure size based on aspect ratio if not provided
    if fig_size is None:
        aspect_ratio = (np.max(x) - np.min(x)) / (np.max(y) - np.min(y))
        fig_size = (10 * aspect_ratio, 10)

        # Check the figure size and adjust it if necessary
        max_fig_size = 20  # inches
        if max(fig_size) > max_fig_size:
            scale_factor = max_fig_size / max(fig_size)
            fig_size = (fig_size[0] * scale_factor, fig_size[1] * scale_factor)

    # Define figure size
    plt.figure(figsize=fig_size)

    # Plot
    plt.scatter(x, y, c=colors, cmap=color_map, alpha=alpha, s=point_size)
    plt.xlabel(x_dim)
    plt.ylabel(y_dim)
    plt.title(f'{x_dim} vs {y_dim} Colored by Height Above Ground')
    plt.colorbar(label='Height Above Ground (m)')
    plt.show()


def plot_3d(arrays, z_dim='Z', fig_size=None):
    """
    Creates a 3D scatter plot with colormap and figure dimensions.

    Parameters:
        arrays (list): A list containing pdal arrays.
        z_dim (str): The z dimension to be plotted. Default is 'Z', can also be 'HeightAboveGround'.
        fig_size (tuple): Figure size in pixels. Default value is None, and in this case it will be determined by the data's aspect ratio.
    """
    # Check that the requested dimensions are valid
    valid_dims = ['Z', 'HeightAboveGround']
    if z_dim not in valid_dims:
        raise ValueError(f"Invalid dimensions. Choose from: {valid_dims}")

    # Prepare data
    points = arrays[0]
    x = points['X']
    y = points['Y']
    z = points[z_dim]
    colors = points['HeightAboveGround']

    # Determine figure size based on aspect ratio if not provided
    if fig_size is None:
        aspect_ratio = (np.max(x) - np.min(x)) / (np.max(y) - np.min(y))
        fig_size = (800 * aspect_ratio, 800)

        # Check the figure size and adjust it if necessary
        max_fig_size = 1600  # pixels
        if max(fig_size) > max_fig_size:
            scale_factor = max_fig_size / max(fig_size)
            fig_size = (fig_size[0] * scale_factor, fig_size[1] * scale_factor)

    # Define figure size
    fig = mlab.figure(size=fig_size)

    # Plot
    pts = mlab.points3d(x, y, z, colors, colormap='viridis', scale_mode='none', scale_factor=0.5)
    mlab.axes()
    mlab.show()


def plot_lai(lai, extent, cmap='viridis', fig_size=None):
    """
    Plots the Leaf Area Index (LAI) map using imshow.

    Parameters:
        lai (numpy array): The 2D LAI values to plot.
        extent (list): The extent of the plot in the form of [xmin, xmax, ymin, ymax].
        cmap (str): The color map to use for color values. Default is 'viridis'.
        fig_size (tuple): Figure size in inches. Default value is None, and in this case it will be determined by the data's aspect ratio.
    """

    # Calculate aspect ratio and figure size if not provided
    if fig_size is None:
        x_range = extent[1] - extent[0]
        y_range = extent[3] - extent[2]
        aspect_ratio = x_range / y_range
        fig_size = (10 * aspect_ratio, 10)

        # Check the figure size and adjust it if necessary
        max_fig_size = 20  # inches
        if max(fig_size) > max_fig_size:
            scale_factor = max_fig_size / max(fig_size)
            fig_size = (fig_size[0] * scale_factor, fig_size[1] * scale_factor)

    # Create figure and axes with the calculated size
    plt.figure(figsize=fig_size)

    # Plot the LAI
    plt.imshow(lai, extent=extent, cmap=cmap)
    plt.colorbar(label='LAI')
    plt.title('Leaf Area Index (LAI)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def plot_lad_2d(lad, slice_index, axis='x', cmap='viridis'):
    """
    Plots a 2D slice of the Leaf Area Density (LAD) 3D array along the specified axis.
    The other axis is chosen by slice_index.

    Parameters:
        lad (numpy array): The 3D LAD values.
        slice_index (int): The index at which to slice the 3D array to get a 2D cross-section.
        axis (str): The axis along which to slice ('x', 'y', or 'z'). Default is 'x'.
        cmap (str): The color map to use for color values. Default is 'viridis'.
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
