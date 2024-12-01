# PyForestScan: Airborne Point Cloud Analysis for Forest Structure

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/iosefa/PyForestScan/HEAD?labpath=docs%2Fexamples%2Fgetting-started.ipynb)
[![PyPI](https://img.shields.io/pypi/v/PyForestScan.svg)](https://pypi.org/project/PyForestScan/)
[![Tests](https://img.shields.io/github/actions/workflow/status/iosefa/PyForestScan/main.yml?branch=main)](https://github.com/iosefa/PyForestScan/actions/workflows/main.yml)
[![Coverage](https://img.shields.io/codecov/c/github/iosefa/PyForestScan/main)](https://codecov.io/gh/iosefa/PyForestScan)

**Calculate Forest Structural Metrics from lidar point clouds in Python**

![Height Above Ground](./screenshots/hag.png)

## Overview

PyForestScan is a Python library designed for analyzing and visualizing forest structure using airborne 3D point cloud data. The library helps derive important forest metrics such as Canopy Height, Plant Area Index (PAI), Canopy Cover, Plant Area Density (PAD), and Foliage Height Diversity (FHD).

## Features

- **Forest Metrics**: Calculate and visualize key metrics like Canopy Height, PAI, PAD, and FHD.
- **Large Point Cloud Support**: Utilizes efficient data formats such as EPT for large point cloud processing.
- **Visualization**: Create 2D and 3D visualizations of forest structure and structural metrics
- **Extensibility**: Easily add custom filters and visualization techniques to suit your needs.

## Installation

Install PyForestScan using pip:

```bash
pip install pyforestscan
```

### Dependencies

> [!IMPORTANT]
> You MUST have installed both PDAL and GDAL to use PyForestScan. If you use conda to install PDAL, make sure you install pyforestscan in the conda environment with PDAL (and GDAL if using conda). See https://pdal.io/en/latest/ for more information on PDAL and https://gdal.org/en/stable/.

- PDAL >= 2.7
- GDAL >= 3.5
- Python >= 3.10

## Quick Start

### Calculate, Export, and Plot Plant Area Index

The following snippet shows how you can load a las file, create 5m by 5m by 1m voxels with points assigned to them, and generate plant area density at 1m layers and plant area index for each 5m grid cell before writing the resulting PAI layer to a geotiff and plotting. 

```python
from pyforestscan.handlers import read_lidar, create_geotiff
from pyforestscan.calculate import assign_voxels, calculate_pad, calculate_pai
from pyforestscan.visualize import plot_pai

arrays = read_lidar("example_data/20191210_5QKB020880.laz", "EPSG:32605", hag=True)
voxel_resolution = (5, 5, 1)
voxels, extent = assign_voxels(arrays[0], voxel_resolution)
pad = calculate_pad(voxels, voxel_resolution[-1])
pai = calculate_pai(pad)
create_geotiff(pai, "output_pai.tiff", "EPSG:32605", extent)
plot_pai(pai, extent, cmap='viridis', fig_size=None)
```

![Plant Area Index](./screenshots/pai.png)

## Documentation

For detailed instructions and examples, visit our [documentation](https://pyforestscan.readthedocs.io/).

## Developer Guides

To build locally and contribute to PyForestScan, you will need the following dependencies:

- PDAL and Python PDAL bindings
- GDAL
- Python
- Python requirements (requirements.txt and requirements-dev.txt)

## Testing

PyForestScan uses `pytest` for running tests. To run the tests, you can follow the steps below:

### Running Tests

1. Install the development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. Run the tests using `pytest`:
   ```bash
   pytest
   ```

This will run all the test cases under the `tests/` directory. The tests include functionality checks for filters, forest metrics, and other core components of the PyForestScan library.

You can also run specific tests by passing the test file or function name:
```bash
pytest tests/test_calculate.py
```

## Contributing

We welcome contributions! Please check our [Contributing Guidelines](docs/contributing.md) to get started.

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.
