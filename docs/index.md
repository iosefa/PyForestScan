# PyForestScan Documentation

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/iosefa/PyForestScan/HEAD?labpath=docs%2Fexamples%2Fgetting-started-importing-preprocessing-dtm-chm.ipynb)
[![PyPI](https://img.shields.io/pypi/v/PyForestScan.svg)](https://pypi.org/project/PyForestScan/)
[![Docker Pulls](https://img.shields.io/docker/pulls/iosefa/pyforestscan?logo=docker&label=pulls)](https://hub.docker.com/r/iosefa/pyforestscan)
[![Contributors](https://img.shields.io/github/contributors/iosefa/PyForestScan.svg?label=contributors)](https://github.com/iosefa/PyForestScan/graphs/contributors)
[![Tests](https://img.shields.io/github/actions/workflow/status/iosefa/PyForestScan/main.yml?branch=main)](https://github.com/iosefa/PyForestScan/actions/workflows/main.yml)
[![Coverage](https://img.shields.io/codecov/c/github/iosefa/PyForestScan/main)](https://codecov.io/gh/iosefa/PyForestScan)

**Calculate Forest Structural Metrics from lidar point clouds in Python**

![Height Above Ground visualization of lidar point cloud data](images/hag.png)

## Overview

PyForestScan is a Python library designed for analyzing and visualizing forest structure using airborne 3D point cloud data. The library helps derive important forest metrics such as Canopy Height, Plant Area Index (PAI), Canopy Cover, Plant Area Density (PAD), and Foliage Height Diversity (FHD).

## Features

- **Forest Metrics**: Calculate and visualize key metrics like Canopy Height, PAI, PAD, and FHD.
- **Large Point Cloud Support**: Utilizes efficient data formats such as EPT for large point cloud processing.
- **Visualization**: Create 2D and 3D visualizations of forest structure and structural metrics
- **Extensibility**: Easily add custom filters and visualization techniques to suit your needs.

## Examples

- [Getting Started: DTM and CHM](examples/getting-started.ipynb)
- [Calculating Forest Metrics](examples/calculate-forest-metrics.ipynb)
- [Working with Large Point Clouds](examples/working-with-large-point-clouds.ipynb)
