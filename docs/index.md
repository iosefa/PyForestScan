# PyForestScan Documentation

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/iosefa/PyForestScan/HEAD?labpath=docs%2Fexamples%2Fgetting-started.ipynb)
[![PyPI](https://img.shields.io/pypi/v/PyForestScan.svg)](https://pypi.org/project/PyForestScan/)
[![Tests](https://img.shields.io/github/actions/workflow/status/iosefa/PyForestScan/main.yml?branch=main)](https://github.com/iosefa/PyForestScan/actions/workflows/main.yml)
[![Coverage](https://img.shields.io/codecov/c/github/iosefa/PyForestScan/main)](https://codecov.io/gh/iosefa/PyForestScan)

![Height Above Ground visualization of LiDAR point cloud data](images/hag.png)
*Height Above Ground visualization of LiDAR point cloud data.*

**pyforestscan** is a Python package designed for processing
and analyzing LiDAR point cloud data. It simplifies complex LiDAR
workflows, making it easier to extract and visualize forest structure
metrics such as canopy height, plant area density, and more.

## Key Features

-   **Voxel-based Canopy Height Model (CHM) Calculation**: Generate 2D
    CHMs using voxel grids.
-   **Plant Area Density (PAD), Plant Area Index (PAI), and Foliage
    Height Diversity (FHD)**: Calculate detailed vegetation metrics with
    voxel-based methods.
-   **Digital Terrain Model (DTM) Generation**: Create terrain models
    from ground-classified points.
-   **Ground Point Classification**: Apply filters like SMRF to classify
    ground and non-ground points.
-   **Outlier Removal and Point Cloud Cleaning**: Efficiently clean data
    and remove statistical outliers.
-   **Visualization Tools**: Plot 2D and 3D representations of point
    clouds and vegetation metrics.

<div style="display: flex; justify-content: space-around;">
    <figure style="margin: 0 10px;">
        <img src="images/chm.png" alt="Canopy Height Model" width="300" />
        <figcaption style="text-align: center;">Canopy Height Model</figcaption>
    </figure>
    <figure style="margin: 0 10px;">
        <img src="images/pai.png" alt="Plant Area Index" width="300" />
        <figcaption style="text-align: center;">Plant Area Index</figcaption>
    </figure>
</div>

## Core Modules

-   **calculate**: Includes methods for calculating CHM, PAD, PAI, and
    other vegetation metrics from voxel-based data.
-   **filters**: Provides tools for classifying ground points, applying
    height filters, and cleaning point clouds.
-   **handlers**: Functions to create GeoTIFF files, load and validate
    point cloud data, and simplify CRS transformations.
-   **visualize**: Visualization utilities to plot 2D scatter plots, PAD
    slices, and PAI data.

## Next Steps

For more details on how to get started, see the