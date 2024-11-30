# PyForestScan Documentation

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/iosefa/PyForestScan/HEAD?labpath=docs%2Fexamples%2Fgetting-started.ipynb)
[![PyPI](https://img.shields.io/pypi/v/PyForestScan.svg)](https://pypi.org/project/PyForestScan/)
[![Docker Pulls](https://badgen.net/docker/pulls/iosefa/pyforestscan?icon=docker&label=pulls&cache=0)](https://hub.docker.com/r/iosefa/pyforestscan)
[![Tests](https://img.shields.io/github/actions/workflow/status/iosefa/PyForestScan/main.yml?branch=main)](https://github.com/iosefa/PyForestScan/actions/workflows/main.yml)
[![Coverage](https://img.shields.io/codecov/c/github/iosefa/PyForestScan/main)](https://codecov.io/gh/iosefa/PyForestScan)

**Calculate Forest Structural Metrics from lidar point clouds in Python**

![Height Above Ground visualization of lidar point cloud data](images/hag.png)
*Height Above Ground visualization of lidar point cloud data.*

**pyforestscan** is a Python package designed for processing
and analyzing lidar point cloud data. It simplifies complex lidar
workflows, making it easier to extract and visualize forest structure
metrics such as canopy height, plant area density (PAD), plant area index (PAI), foliage height diversity (FHD), and more.

## Features

- **Forest Metrics**: Calculate and visualize key metrics like Canopy Height, PAI, PAD, and FHD.
- **Large Point Cloud Support**: Utilizes efficient data formats such as EPT for large point cloud processing.
- **Visualization**: Create 2D and 3D visualizations of forest structure and structural metrics
- **Extensibility**: Easily add custom filters and visualization techniques to suit your needs.

## Examples

- jupyter 1
- 2

## Core Modules

-   **calculate**: Includes methods for calculating CHM, PAD, PAI, and
    other vegetation metrics from voxel-based data.
-   **filters**: Provides tools for classifying ground points, applying
    height filters, and cleaning point clouds.
-   **handlers**: Functions to create GeoTIFF files, load and validate
    point cloud data, and simplify CRS transformations.
-   **visualize**: Visualization utilities to plot 2D scatter plots, PAD
    slices, and PAI data.
