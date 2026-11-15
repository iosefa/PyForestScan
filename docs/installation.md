# Installation

## Prerequisites

PyForestScan uses [PDAL](https://pdal.io/en/stable/) for point-cloud processing
and [GDAL](https://gdal.org/en/stable/) for raster data. Use Python 3.11 or later
and install these libraries with Conda before installing PyForestScan in the
same environment.

## Install from PyPI

```bash
conda create -n pyforestscan_env -c conda-forge python=3.12 pdal python-pdal libpdal-tiledb gdal pip
conda activate pyforestscan_env
pip install pyforestscan
```

The Conda environment includes PDAL's Python bindings and TileDB drivers
(`libpdal-tiledb`), along with GDAL. Pip installs PyForestScan and its Python
dependencies.

## Install from GitHub

To install the development version in the same environment:

```bash
pip install git+https://github.com/iosefa/pyforestscan
```

Check the installed version with:

```bash
pip show pyforestscan
```

## Docker

The Docker image includes PyForestScan, Jupyter, and the example notebooks
and data:

```bash
docker run -it --rm -p 8888:8888 iosefa/pyforestscan:latest
```

Open the URL printed by Jupyter to use the notebooks.
