import matplotlib.pyplot as plt

from mayavi import mlab


def plot_2d(arrays, x_dim='X', y_dim='Z', color_map='viridis', alpha=0.5, point_size=1, fig_size=(10, 10)):
    """
    Creates a 2D scatter plot with colormap, adjustable opacity, point size and figure dimensions.

    Parameters:
        arrays (list): A list containing pdal arrays.
        x_dim (str): The x dimension to be plotted. Default is 'X'.
        y_dim (str): The y dimension to be plotted. Default is 'Z'.
        color_map (str): The color map to use for color values. Default is 'viridis'.
        alpha (float): Opacity of points. Default value is 0.5.
        point_size (float): Size of points. Default value is 1.
        fig_size (tuple): Figure size in inches. Default value is (10,10).
    """
    # Check that the requested dimensions are valid
    valid_dims = ['X', 'Y', 'Z', 'HeightAboveGround']
    if x_dim not in valid_dims or y_dim not in valid_dims:
        raise ValueError(f"Invalid dimensions. Choose from: {valid_dims}")

    # Prepare data
    points = arrays[0]
    x = points[x_dim]
    y = points[y_dim]
    colors = points['HeightAboveGround']

    # Define figure size
    plt.figure(figsize=fig_size)

    # Plot
    plt.scatter(x, y, c=colors, cmap=color_map, alpha=alpha, s=point_size)
    plt.xlabel(x_dim)
    plt.ylabel(y_dim)
    plt.title(f'{x_dim} vs {y_dim} Colored by Height Above Ground')
    plt.colorbar(label='Height Above Ground (m)')
    plt.show()


def plot_3d(arrays):
    # Prepare data
    points = arrays[0]
    x = points['X']
    y = points['Y']
    z = points['Z']
    colors = points['HeightAboveGround']

    # Plot
    fig = mlab.figure(size=(800, 800))
    pts = mlab.points3d(x, y, z, colors, colormap='viridis', scale_mode='none', scale_factor=0.5)
    mlab.axes()
    mlab.show()
