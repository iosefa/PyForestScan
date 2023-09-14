# PyForestScan

__This library is currently experimental and is under heavy development.__

![Height Above Ground](https://github.com/iosefa/PyForestScan/screenshots/hag.png)

## Overview

PyForestScan is a Python library designed to process, analyze, and visualize 3D point cloud data of forests.
It aims to simplify complex point cloud processing pipelines, offering functionalities such as filtering by
height above ground, 2D and 3D visualizations, and various density metrics. Whether you are a researcher, developer,
or data scientist, PyForestScan is designed to make your point cloud data processing tasks easier.

## Features

- **Height Above Ground Filtering**: Easily filter points based on their height above ground.
- **2D and 3D Visualization**: Plot your data in 2D or 3D space with customizable settings.
- **Density Metrics**: Calculate and visualize metrics such as Leaf Area Density (LAD) and Leaf Area Index (LAI).
- **Extensibility**: Designed to be easily extensible for custom filters and visualization techniques.

## Installation

To install PyForestScan, run:

```bash
pip install pyforestscan
```

## Dependencies

- NumPy
- Matplotlib
- Mayavi

## Usage Examples

Here are some quick examples to get you started:

### Filtering Height Above Ground

```python
from pyforestscan.pipeline import filter_hag
filtered_arrays = filter_hag(arrays, lower_limit=5, upper_limit=15)
```

### 2D Plotting

```python
from pyforestscan.visualization import plot_2d
plot_2d(df, x_dim='X', y_dim='Z')
```

### 3D Plotting

```python
from pyforestscan.visualization import plot_3d
plot_3d([array1, array2], z_dim='Z')
```

## Documentation

Complete documentation is available [here](https://pyforestscan.readthedocs.io/).

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

For more information, check out the [documentation](https://pyforestscan.readthedocs.io/) or feel free to open an issue.
