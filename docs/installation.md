# Installation

## Prerequisites

To use **PyForestScan**, you need to have both PDAL (Point Data Abstraction Library) and GDAL (Geospatial Data Abstraction Library) installed. PDAL is a powerful library for working with point cloud data, and PyForestScan relies on it for many core functions.  GDAL contains the necessary raster processing capabilities and is used for writing geospatial raster data.

### GDAL and PDAL

For complete installation guides, please visit the [GDAL official documentation](https://gdal.org/en/stable/) and [PDAL official documentation](https://pdal.io/en/stable/) sites. However, for a quick start, it is recommended to use conda to install both PDAL and GDAL, as it handles dependencies and installation paths efficiently. Once these dependencies are installed, you can then use pip to install pyforestscan within the same environment.

Steps to install PDAL and GDAL:

1.  Install PDAL and GDAL using `conda`:

``` bash
conda create -n pyforestscan_env -c conda-forge pdal gdal
```

2. Activate the conda env:

```bash
conda activate pyforestscan_env
```

For more information on PDAL and its capabilities, visit the official PDAL documentation: <https://pdal.io/en/latest/>.


## Install from PyPI
Once dependencies are installed, PyForestScan can be installed from PyPI:

``` bash
pip install pyforestscan
```

## Install from GitHub

It is also possible to install the latest development version from GitHub using Git:

pip install git+https://github.com/iosefa/pyforestscan


At any time, you can verify your installation:

``` bash
pip show pyforestscan
```

## Docker

A docker environment with `pyforestscan` is also available and includes jupyter with example notebooks and data. To use the docker environment:

```bash
docker run -it --rm -p 8888:8888 iosefa/pyforestscan:latest
```

This will launch a jupyter notebook. 
