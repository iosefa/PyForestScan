# Importing, Preprocessing, and Writing Data

To use pyforestscan, first import it in your Python project:

```python
import pyforestscan
```

Then, you can use it to load point cloud data and extract forest structure metrics. 

The following sections will provide an overview of usage of the major functions of pyforestscan. For a complete reference of all functions in pyforestscan, please check the [API documentation](/api/calculate/). For comprehensive examples of these functions, please see the [example jupyter notebooks](/examples/getting-started-importing-preprocessing-dtm-chm/). 

## Importing Point Cloud Data

pyforestscan supports reading from the following point cloud data formats:

* las
* laz
* copc
* ept

and reading point clouds is done using the `read_lidar` function:

```python
from pyforestscan.handlers import read_lidar

file_path = "../example_data/20191210_5QKB020880.laz"
arrays = read_lidar(file_path, "EPSG:32605", hag=True)
pointcloud = arrays[0]
```

## Preprocessing Point Cloud Data

pyforestscan provides some basic functionality to help preprocess point cloud data. Many of these functions are wrapped PDAL routines. For example, to remove outliers and classify ground points:

```python
from pyforestscan.filters import remove_outliers_and_clean, classify_ground_points

cleaned_arrays = remove_outliers_and_clean(arrays, mean_k=8, multiplier=3.0)
classified_arrays = classify_ground_points(cleaned_arrays)
```

### Adding Height Above Ground

Forest structure metrics require `HeightAboveGround`; raw elevation (`Z`) is not a substitute. If `HeightAboveGround` is not already present in your point cloud arrays, add it either while reading the data or afterward with `add_height_above_ground`. If you are reading directly from a point cloud file, you can calculate Height Above Ground during import:

```python
from pyforestscan.handlers import read_lidar

file_path = "../example_data/20191210_5QKB020880.laz"
arrays = read_lidar(file_path, "EPSG:32605", hag=True)
points = arrays[0]
```

If your points are already in memory, use `add_height_above_ground`. The default method uses PDAL's Delaunay Height Above Ground filter and requires ground points classified as `Classification == 2`:

```python
from pyforestscan.filters import add_height_above_ground

hag_arrays = add_height_above_ground(classified_arrays)
points = hag_arrays[0]
```

You can also calculate Height Above Ground from a DTM raster. This method only requires `X`, `Y`, and `Z` fields in the point array, and the DTM must be in the same coordinate reference system as the points:

```python
from pyforestscan.filters import add_height_above_ground

dtm_path = "../example_data/20191210_5QKB020880_DS05_dtm.tif"
hag_arrays = add_height_above_ground(arrays, dtm=dtm_path)
points = hag_arrays[0]
```

The DTM method can also be selected explicitly:

```python
hag_arrays = add_height_above_ground(arrays, method="dtm", dtm=dtm_path)
```

## Exporting Point Clouds

pyforestscan supports exporting processed point clouds to las and laz formats. To export a point cloud as a LAZ file:

```python
from pyforestscan.handlers import write_las

write_las(classified_arrays, "/path/to/exported_file.las", srs="EPSG:32605", compress=True)
```
