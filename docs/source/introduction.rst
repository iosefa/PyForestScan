Introduction
============

pyforestscan: Forest Structure Analysis Using Point Cloud Data
--------------------------------------------------------------
``pyforestscan`` is a specialized Python library aimed at enabling comprehensive analysis and visualization of forest structure using point cloud data. Designed to work seamlessly with LiDAR and Structure from Motion (SfM) data, this library simplifies the derivation of key forest metrics like Height Above Ground (HAG), Leaf Area Index (LAI), Canopy Cover, and Leaf Area Density (LAD).

Whether you are an ecologist, a forester, or a geospatial scientist, ``pyforestscan`` provides you with the tools to make data-driven decisions and to deepen your understanding of forest ecosystems.

Modules
-------

- **handlers**: Utilities to configure PDAL pipelines for specialized tasks like data filtering and cropping.
- **pipeline**: Core module offering a rich set of features for processing point cloud data through PDAL pipelines.
- **visualization**: Advanced 2D and 3D plotting functions for insightful visualizations of derived forest metrics.

Key Features
------------

1. **Forest Metrics Calculation**: Obtain critical metrics such as HAG, LAI, Canopy Cover, and LAD with ease.
2. **Data Source Flexibility**: Compatible with LiDAR and Structure from Motion (SfM) datasets.
3. **Intuitive Visualization**: Comprehensive visualization tools to study forest structure and metrics.
4. **Advanced Filtering**: Customizable filters to perform precise data manipulation.
5. **Efficient Data Handling**: Optimized for both small-scale and large-scale forest datasets.

Getting Started
---------------
To begin your journey with ``pyforestscan``, install the package using pip:

.. code-block:: bash

   pip install pyforestscan

Here's a quick example to calculate and visualize Leaf Area Index (LAI) from a point cloud dataset:

.. code-block:: python

   from pyforestscan.pipeline import calculate_lai
   from pyforestscan.visualization import plot_lai

   arrays = [...]  # Your point cloud data
   lai = calculate_lai(arrays)
   plot_lai(lai)

Feel free to explore the detailed examples and full API documentation in the respective modules. With ``pyforestscan``, diving into the intricate world of forest structure analysis is now more accessible than ever.