# PyForestScan: Airborne Point Cloud Analysis for Forest Structure

_This library is currently experimental and is under heavy development._

![Height Above Ground](./screenshots/hag.png)

## Overview

PyForestScan is a Python library optimized for the analysis and visualization of forest structure through airborne 3D point cloud data. Engineered to work seamlessly with point clouds derived from drones and other UAV (Unmanned Aerial Vehicle) devices, the library is a go-to solution for researchers, ecologists, and forestry professionals. It specializes in deriving key forest metrics such as Height Above Ground (HAG), Leaf Area Index (LAI), Canopy Cover, and Leaf Area Density (LAD).

## Core Focus Areas

### 🌳 Forest Metrics
Gain unprecedented insights into forest structure with metrics like HAG, LAI, Canopy Cover, and LAD, specially tuned for airborne data.

### 🚁 Airborne Data Compatibility
Designed for compatibility with LiDAR and Structure from Motion (SfM) data acquired through drones and UAVs, ensuring high-resolution and accurate analysis.

### 🌱 Intuitive Visualization
Unlock actionable insights with sophisticated 2D and 3D visualizations that are tailored to represent airborne point cloud data effectively.

## Features

- **Specialized Forest Metrics**: Calculate and visualize crucial airborne forest metrics such as LAI and LAD.
- **Height Above Ground Filtering**: Customize your data filtering based on the height above ground.
- **2D and 3D Visualization**: Get interactive and focus on forest-centric metrics in both 2D and 3D visualizations.
- **Extensibility**: Adapt PyForestScan to your specific needs with its built-in support for custom filters and visualization techniques.

## Installation

To jumpstart your forest analysis with airborne data, run:

```bash
pip install pyforestscan
```

## Dependencies

- NumPy
- Matplotlib
- Mayavi

## Quick Start

### Deriving Forest Metrics from Airborne Data

```python
from pyforestscan.pipeline import calculate_lai
lai = calculate_lai(arrays)
```

### 2D Visualization of Forest Metrics

```python
from pyforestscan.visualization import plot_lai
plot_lai(lai, extent=[0, 100, 0, 100])
```

### 3D Visualization of Airborne Canopy Structure

```python
from pyforestscan.visualization import plot_3d
plot_3d([array1, array2], z_dim='HeightAboveGround')
```

## Documentation

Explore our [complete documentation](https://pyforestscan.readthedocs.io/) to fully leverage the capabilities of PyForestScan.

## Contributing

Contribute to this airborne-focused forest analysis toolkit! Check our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

For further information, dive into our [documentation](https://pyforestscan.readthedocs.io/) or open an issue.