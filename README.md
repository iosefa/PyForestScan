# PyForestScan: Airborne Point Cloud Analysis for Forest Structure

![Height Above Ground](./screenshots/hag.png)

## Overview

PyForestScan is a Python library designed for analyzing and visualizing forest structure using airborne 
3D point cloud data. The library helps derive important forest metrics such as Canopy Height, 
Plant Area Index (PAI), Canopy Cover, Plant Area Density (PAD), and Foliage Height Diversity (FHD).

## Features

- **Forest Metrics**: Calculate and visualize key metrics like Canopy Height, PAI, PAD, and FHD.
- **Airborne Data Compatibility**: Supports LiDAR and Structure from Motion (SfM) data from drones and UAVs.
- **Visualization**: Create 2D and 3D visualizations of forest structures.
- **Extensibility**: Easily add custom filters and visualization techniques to suit your needs.

## Installation

Install PyForestScan using pip:

```bash
pip install pyforestscan
```

### Dependencies

> [!IMPORTANT]
> You MUST have installed PDAL to use PyForestScan. If you use conda to install PDAL, make sure you install pyforestscan in the conda environment with PDAL. See https://pdal.io/en/latest/ for more information.

- PDAL >= 2.7
- Python >= 3.8

## Quick Start

### Derive Forest Metrics from Airborne Data

The following snipped shows how you can load a las file, create 25m by 25m by 5m voxels with points assigned to them, and generate plant area density at 5m layers and plant area index for each 25m grid cell before writing the resulting PAI layer to a geotiff. 
```python
from pyforestscan.handlers import read_lidar, create_geotiff
from pyforestscan.calculate import assign_voxels, calculate_pad, calculate_pai

arrays = read_lidar("path/to/lidar/file.las", "EPSG:32605", hag=True)
voxels, extent = assign_voxels(arrays[0], (25, 25, 5))
pad = calculate_pad(voxels, 5)
pai = calculate_pai(pad)
create_geotiff(pai, "output_pai.tiff", "EPSG:32605", extent)
```

## Documentation

For detailed instructions and examples, visit our [documentation](https://pyforestscan.readthedocs.io/).

## Developer Guides

To build locally and contribute to PyForestScan, you will need the following dependencies:

- PDAL and Python PDAL bindings
- GDAL
- Python
- Python requirements (requirements.txt)

## Contributing

We welcome contributions! Please check our [Contributing Guidelines](CONTRIBUTING.md) to get started.

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.
