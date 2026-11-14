# Installation

## Prerequisites

To use **PyForestScan**, you need to have both PDAL (Point Data Abstraction Library) and GDAL (Geospatial Data Abstraction Library) installed. PDAL is a powerful library for working with point cloud data, and PyForestScan relies on it for many core functions.  GDAL contains the necessary raster processing capabilities and is used for writing geospatial raster data.

### GDAL and PDAL

For complete installation guides, please visit the [GDAL official documentation](https://gdal.org/en/stable/) and [PDAL official documentation](https://pdal.io/en/stable/) sites. However, for a quick start, it is recommended to use conda to install both PDAL and GDAL, as it handles dependencies and installation paths efficiently. Once these dependencies are installed, you can then use pip to install pyforestscan within the same environment.

Steps to install PDAL and GDAL:

1.  Install PDAL and GDAL using `conda`:

``` bash
conda create -n pyforestscan_env -c conda-forge python=3.12 pdal python-pdal libpdal-tiledb gdal pip
```

2. Activate the conda env:

```bash
conda activate pyforestscan_env
```

The environment includes the Python PDAL bindings and the native TileDB
plugin. The conda-forge package
[`libpdal-tiledb`](https://anaconda.org/conda-forge/libpdal-tiledb) supplies
PDAL's TileDB reader and writer.

For more information on PDAL and its capabilities, visit the official PDAL documentation: <https://pdal.io/en/latest/>.


## Install from PyPI
Once dependencies are installed, PyForestScan can be installed from PyPI:

``` bash
pip install pyforestscan
```

The Python `tiledb` package is a required dependency and is installed with
PyForestScan. It provides array metadata access. TileDB point-cloud reads and
writes also require the native PDAL plugin installed above; installing the
Python package alone does not provide it.

Verify that the environment has both TileDB drivers:

```bash
pdal --drivers
```

Look for `readers.tiledb` and `writers.tiledb` in the output. If either is
missing, install `libpdal-tiledb` into the active conda environment. See the
[TileDB guide](usage/tiledb.md) for conversion, spatial queries, and tiled
forest-metric examples.

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
