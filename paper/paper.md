
---
title: 'PyForestScan: A Python Library for Calculating Forest Structural Metrics from LiDAR Point Cloud Data'
tags:
  - Python
  - forest structure
  - ecology
  - lidar
  - point clouds
authors:
  - name: Joseph Emile Honour Percival
    orcid: 0000-0001-5941-4601
    corresponding: true
    affiliation: "1,2"
  - name: Benjamin Palsa Leamon
    orcid: 0009-0002-4614-2322
    affiliation: "3"
  - name: Ryan Perroy
    orcid: 0000-0002-4210-3281
    affiliation: "1,2"
affiliations:
 - name: Department of Geography & Environmental Science, University of Hawai‘i at Hilo, Hilo, HI 96720, USA
   index: 1
 - name: Spatial Data Analysis & Visualization Laboratory, University of Hawai‘i at Hilo, Hilo, HI 96720, USA
   index: 2
 - name: Independent Researcher, Japan
   index: 3
date: 15 September 2024
bibliography: paper.bib

---

# Summary

PyForestScan is an open-source Python library designed for calculating forest structural metrics from lidar point cloud data at scale. The software supports input formats including .las, .laz, and .copc files, efficiently handles large-scale lidar datasets, and calculates key ecological metrics such as foliage height diversity (FHD), plant area density (PAD), canopy height, and plant area index (PAI). In addition to metrics computation, the library supports the import and generation of digital terrain models, the generation of GeoTIFF outputs, and integrates with geospatial libraries like PDAL, making it a valuable tool for forest monitoring, carbon accounting, and ecological research.

# Statement of Need

Remote sensing data, particularly Light Detection and Ranging (lidar) data from airborne sensors, are becoming increasingly accessible, offering a detailed understanding of forest ecosystems at fine spatial resolutions. This data is critical for calculating forest structural metrics such as canopy height, canopy cover, Plant Area Index (PAI), Plant Area Density (PAD), Foliage Height Diversity (FHD), as well as digital terrain models (DTM), which are essential for forest management, biodiversity conservation, and carbon accounting [@mcelhinnyForestWoodlandStand:2005; @drakeEstimationTropicalForest:2002]. However, working with large-scale lidar datasets remains a challenge due to the complexity and size of the data. Additionally, despite Python being a powerful language widely used for geospatial and ecological analysis, there is a notable lack of dedicated, open-source tools within the Python ecosystem specifically designed for calculating these forest structural metrics from lidar data. Calculating these metrics is non-trivial, and several steps are often required to process the point clouds in order to generate these metrics. Existing open source solutions are primarily in the R programming language [@rousselLidRPackageAnalysis:2020a;  @rousselAirborneLiDARData:2024; @dealmeidaLeafRCalculatesLeaf:2021] or are proprietary, computationally intensive, or not flexible enough for the variety of ecological contexts in which these metrics are needed [@lastools; @fusion; @globalmapper]. This gap makes it difficult for researchers, ecologists, and forest managers to integrate lidar-based analysis into their workflows efficiently.

PyForestScan was developed to fill this gap by providing an open-source, Python-based solution that can handle the complexities of lidar data while remaining accessible and efficient. Designed for point clouds captured by airborne lidar and points generated from structure from motion (SfM), it supports commonly used formats such as .las, .laz, and .copc, and integrates with well-established geospatial frameworks for point clouds like Point Cloud Data Abstraction Library (PDAL). The more mathematically intensive calculations of PAD, PAI, and FHD are calculated following established methods by [@kamoskeLeafAreaDensity:2019] and [@hurlbertNonconceptSpeciesDiversity:1971], and are given by equations (1) - (3). PyForestScan provides tiling mechanisms to calculate metrics across large landscapes, IO support across multiple formats, point cloud processing tools to filter points and create ground surfaces, as well as simple visualization functions for core metrics. PyForestScan brings this functionality to Python, while also introducing capabilities not found in any single existing open-source software. These include canopy height, PAD, PAI, FHD, DTM, and advanced tiling functionality that efficiently handles large datasets enabling analysis of forest metrics across large landscapes. By focusing on forest structural metrics, PyForestScan provides an essential tool for the growing need to analyze forest structure at scale in the context of environmental monitoring, conservation, and climate-related research.


$$
  PAD_{i-1,i} = \ln\left(\frac{S_e}{S_t}\right) \frac{1}{k \Delta z}
  \tag{1}
$$

$$
  PAI = \sum_{i=1}^{n} PAD_{i-1,i}
  \tag{2}
$$

$$
  FHD = - \sum_{i=1}^{n} p_i \ln(p_i)
  \tag{3}
$$

In equation (1), $PAD_{i-1,i}$ represents the PAD between two adjacent voxels in the canopy, denoted by the indices $i-1$ and $i$. It quantifies the density of plant material within this vertical slice of the forest. Here, $\ln\left(\frac{S_e}{S_t}\right)$ calculates the natural logarithm of the ratio between the number of lidar pulses entering a voxel ($S_e$) and the number of pulses exiting the voxel ($S_t$), and $\frac{1}{k \Delta z}$ is the inverse of the extinction coefficient ($k$) as derived from the Beer-Lambert Law,  multiplied by the height of each voxel ($\Delta z$).

Equation (2) represents PAI as the vertical summation of PAD across all layers $i$ through $n$, as derived in equation (1).

In equation (3), $FHD$ is calculated as the Shannon entropy of the vertical distribution of plant material across all layers of the canopy. $p_i$ is the proportion of total plant material in each voxel $i$, relative to the entire vertical column, with $n$ representing the number of vertical layers. 

# Acknowledgements

We would like to express our gratitude to Juan Guerra-Hernandez and Adrian Pascual for providing a noise-free classified point cloud dataset, which was instrumental in testing and validating the PyForestScan library. 

# References



