# Installation

This guide will help you get started with [PyForestScan]{.title-ref}, a
Python package for processing and analyzing LiDAR point cloud data.
Follow these steps to install dependencies, including PDAL, and learn
how to run a basic example using forest structure metrics.

## Prerequisites

To use **PyForestScan**, you need to have PDAL 
(Point Data Abstraction Library) installed. PDAL is a powerful library
for working with point cloud data, and PyForestScan relies
on it for many core functions.

### Installing PDAL

For a quick start, it is recommended to use [conda]{.title-ref} to
install PDAL, as it handles dependencies and installation paths
efficiently. Once PDAL is installed, you can then use [pip]{.title-ref}
to install [PyForestScan]{.title-ref} within the same environment.

Steps to install PDAL and PyForestScan:

1.  First, install [PDAL]{.title-ref} using \`conda\`:

``` bash
conda create -n pyforestscan_env pdal
conda activate pyforestscan_env
```

2.  Next, install [PyForestScan]{.title-ref} using \`pip\`:

``` bash
pip install pyforestscan
```

3.  Verify your installation by running the following to check PDAL
    version:

``` bash
pdal --version
```

For more information on PDAL and its capabilities, visit the official
PDAL documentation: <https://pdal.io/en/latest/>

## Quick Start Example

The following example demonstrates how to use [PyForestScan]{.title-ref}
to load LiDAR data, calculate plant area density (PAD) and plant area
index (PAI), and write the result to a GeoTIFF file.

1.  **Load LiDAR Data** Use the [read_lidar]{.title-ref} function to
    load a [.las]{.title-ref} file and ensure the point cloud is
    projected using the correct CRS.

``` python
from pyforestscan.handlers import read_lidar
arrays = read_lidar("path/to/lidar/file.las", "EPSG:32605", hag=True)
```

2.  **Assign Voxels** Divide the point cloud data into voxel grids of a
    given resolution (e.g., 25m by 25m by 5m).

``` python
from pyforestscan.calculate import assign_voxels
voxels, extent = assign_voxels(arrays[0], (25, 25, 5))
```

3.  **Calculate PAD** Generate Plant Area Density (PAD) for each voxel
    layer.

``` python
from pyforestscan.calculate import calculate_pad
pad = calculate_pad(voxels, 5)
```

4.  **Calculate PAI** Calculate the Plant Area Index (PAI) by summing
    the PAD values for each voxel column.

``` python
from pyforestscan.calculate import calculate_pai
pai = calculate_pai(pad)
```

5.  **Write Output to GeoTIFF** Finally, write the calculated PAI data
    to a GeoTIFF file.

``` python
from pyforestscan.handlers import create_geotiff
create_geotiff(pai, "output_pai.tif", "EPSG:32605", extent)
```