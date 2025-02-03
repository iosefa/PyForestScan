
---
title: 'PyForestScan: A Python library for calculating forest structural metrics from lidar point cloud data'
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

`PyForestScan` is an open-source Python library designed to compute forest structural metrics from Light Detection and Ranging (lidar) point cloud data at scale. The software calculates key ecological metrics such as foliage height diversity (FHD), plant area density (PAD), canopy height, plant area index (PAI), and digital terrain models (DTMs), efficiently handles large-scale lidar datasets, and supports input formats including the Entwine Point Tile (EPT) format [@manning_entwine], .las, .laz, and .copc files. In addition to metrics computation, the library supports the generation of GeoTIFF outputs and integrates with geospatial libraries like the Point Cloud Data Abstraction Library (`PDAL`) [@howard_butler_2024_13993879; @BUTLER2021104680], making it a valuable tool for forest monitoring, carbon accounting, and ecological research.

# Statement of Need

Remote sensing data, particularly point cloud data from airborne lidar sensors, are becoming increasingly accessible, offering a detailed understanding of forest ecosystems at fine spatial resolutions over large areas. This data is useful for calculating forest structural metrics such as canopy height, canopy cover, PAI, PAD, FHD, as well as DTMs, which are essential for forest management, biodiversity conservation, and carbon accounting [@mcelhinnyForestWoodlandStand2005; @drakeEstimationTropicalForest2002; @pascualRoleImprovedGround2020; @guerra-hernandezUsingBitemporalALS2024; @pascualIntegratedAssessmentCarbon2023; @pascualNewRemoteSensingbased2021a]. 

Despite Python's prominence as a powerful language for geospatial and ecological analysis, there is a notable lack of dedicated, open-source tools within the Python ecosystem specifically designed for calculating comprehensive forest structural metrics from airborne lidar point-cloud data. This gap is significant given Python's extensive libraries for data science and its increasingly important role in ecology and deep learning [@doi:10.1111/2041-210X.13901]. Existing open-source solutions that offer some of these metrics are primarily available in the R programming language. For instance, `lidR` [@rousselLidRPackageAnalysis2020a; @rousselAirborneLiDARData2024] provides functions for point cloud manipulation, metric computation, and visualization but lacks native calculations for FHD and PAI. Another tool, `leafR` [@dealmeidaLeafRCalculatesLeaf2021], calculates FHD, leaf area index (LAI), and leaf area density (LAD) - both of which are very similar to PAI and PAD - but is limited in processing large datasets due to the absence of tiling functionality. Moreover, the importance of scale in lidar-based analyses of forest structure is well-documented [@doi:10.1111/2041-210X.14040], and `leafR` does not allow users to modify voxel depth, which can be important for accurate estimation of structural metrics across different forest types and scales. Similarly, `canopyLazR` [@kamoskeLeafAreaDensity2019] provides tools to calculate LAD and LAI from point cloud lidar data but only allows the calculation of these metrics and lacks support for tiling mechanisms, limiting its applicability to large datasets. Proprietary solutions like LAStools [@lastools], FUSION [@fusion], and Global Mapper [@globalmapper] offer tools to calculate some of these metrics -mostly canopy height- but may not provide the flexibility required for diverse ecological contexts and are often inaccessible due to licensing costs. This lack of a comprehensive, scalable Python-based solution makes it challenging for researchers, ecologists, and forest managers to integrate point-cloud-based analysis into their Python workflows efficiently. This is particularly problematic when working with large datasets or when integrating analyses with other Python-based tools, such as those used for processing space-based waveform lidar data from the Global Ecosystem Dynamics Investigation (GEDI) mission [@tangAlgorithmTheoreticalBasis2019; @DUBAYAH2020100002], which also provides data on PAI, plant area volume density (PAVD), and FHD.

`PyForestScan` was developed to fill this gap by providing an open-source, Python-based solution to calculate forest structural metrics that can handle large-scale point-cloud data while remaining accessible and efficient. By leveraging IO capabilities of `PDAL`, it handles large-scale analyses by allowing users to work with more efficient point-cloud data structure, such as spatially indexed hierarchical octree formats like EPT or COPC. `PyForestScan` supports commonly used formats such as .las, .laz, as well as more efficient formats such as COPC and EPT, and integrates with well-established geospatial frameworks for point clouds like `PDAL` [@howard_butler_2024_13993879; @BUTLER2021104680]. The more mathematically intensive calculations of PAD, PAI, and FHD are calculated following established methods by @kamoskeLeafAreaDensity2019 and @hurlbertNonconceptSpeciesDiversity1971, and details are provided in the documentation. `PyForestScan` provides native tiling mechanisms to calculate metrics across large landscapes, IO support across multiple formats, point cloud processing tools to filter points and create ground surfaces, as well as simple visualization functions for core metrics. `PyForestScan` brings this functionality to Python, while also introducing capabilities not found in any single existing open-source software. By focusing on forest structural metrics, `PyForestScan` provides an essential tool for the growing need to analyze forest structure at scale in the context of environmental monitoring, conservation, and climate-related research.

# Usage

To facilitate usage of the software, we have included [Jupyter notebooks](https://github.com/iosefa/PyForestScan/tree/main/docs/examples) in the [GitHub repository](https://github.com/iosefa/PyForestScan) detailing how to get started using `PyForestScan` as well as how to calculate forest metrics. The Jupyter notebooks include an example data set of a point cloud with a nominal pulse spacing of 0.35 meters and was captured over a dry forest environment. This example dataset is a one-square-kilometer tile derived from a 2018-2020 aerial lidar survey of the Big Island of Hawaii [@NOAA_HI_Lidar_2019]. The data has been preprocessed to classify ground and vegetation points [@guerra-hernandezHighresolutionCanopyHeight2024]. More details are available in the documentation. 

# Contributions
JEHP developed the concept with input from BPL; JEHP wrote the initial versions of the software and automatic tests with contributions from BPL; BPL and JEHP wrote the software documentation and created Jupyter notebooks for example usage; and both authors wrote the manuscript.

# Acknowledgements

We would like to express our gratitude to Juan Guerra-Hernandez and Adrian Pascual for providing a noise-free classified point cloud dataset [@guerra-hernandezHighresolutionCanopyHeight2024], which was instrumental in testing and validating the PyForestScan library. We would also like to thank Ryan Perroy for his guidance, feedback, and help in revising this manuscript. This work was enabled in part by funding from the National Science Foundation award: 2149133. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
 

# References



