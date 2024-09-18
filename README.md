# PyForestScan: Airborne Point Cloud Analysis for Forest Structure

![Height Above Ground](./screenshots/hag.png)

## Overview

PyForestScan is a Python library designed for analyzing and visualizing forest structure using airborne 
3D point cloud data. The library helps derive important forest metrics such as Canopy Height, 
Plant Area Index (PAI), Canopy Cover, Plant Area Density (PAD), and Foliage Height Diversity (FHD).

## Features

- **Forest Metrics**: Calculate and visualize key metrics like Canopy Height, PAI, Canopy Cover, PAD, and FHD.
- **Airborne Data Compatibility**: Supports LiDAR and Structure from Motion (SfM) data from drones and UAVs.
- **Visualization**: Create 2D and 3D visualizations of forest structures.
- **Extensibility**: Easily add custom filters and visualization techniques to suit your needs.

## Installation

Install PyForestScan using pip:

```bash
pip install pyforestscan
```
## Developer Dependencies
- PDAL and Python PDAL bindings
- GDAL
- Python
- Python requirements (requirements.txt)

## Quick Start

### Derive Forest Metrics from Airborne Data

```python
from pyforestscan.handlers import read_lidar
from pyforestscan.calculate import assign_voxels, calculate_pad, calculate_pai

arrays = read_lidar("path/to/lidar/file.las", "EPSG:32605", hag=True)
voxels = assign_voxels(arrays[0], (25, 25, 5))
pad = calculate_pad(voxels, 5)
pai = calculate_pai(pad)
```

### 2D Visualization of Forest Metrics

```python
from pyforestscan.visualize import plot_pai

# Define the spatial extent of your data
extent = [0, 100, 0, 100]

# Plot Plant Area Index (PAI)
plot_pai(pai, extent=extent)
```

### 3D Visualization of Airborne Canopy Structure

```python
from pyforestscan.visualize import plot_3d

plot_3d([array1, array2], z_dim='HeightAboveGround')
```

## Documentation

For detailed instructions and examples, visit our [documentation](https://pyforestscan.readthedocs.io/).

## Contributing

We welcome contributions! Please check our [Contributing Guidelines](CONTRIBUTING.md) to get started.

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.
