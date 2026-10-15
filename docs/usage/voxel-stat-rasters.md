# Custom Voxel Statistic Rasters

PyForestScan can build rasters from arbitrary point dimensions by combining:

- `calculate_voxel_stat` to aggregate values per XY cell.
- `create_geotiff` to write the result as a GeoTIFF.

This is useful for products like mean intensity, return count, median height-above-ground, and more.

## Example: Mean Intensity at 1 m Resolution

```python
from pyforestscan.handlers import read_lidar, create_geotiff
from pyforestscan.calculate import calculate_voxel_stat

file_path = "../example_data/20191210_5QKB020880.laz"
arrays = read_lidar(file_path, "EPSG:32605", hag=True)

# 1 m XY cells, 1 m vertical bins
voxel_resolution = (1.0, 1.0, 1.0)

mean_intensity, extent = calculate_voxel_stat(
    arrays[0],
    voxel_resolution=voxel_resolution,
    dimension="Intensity",
    stat="mean"
)

create_geotiff(
    mean_intensity,
    "../example_data/mean_intensity_1m.tif",
    "EPSG:32605",
    extent
)
```

## Supported Statistics

`calculate_voxel_stat` supports:

- `mean`
- `sum`
- `count`
- `min`
- `max`
- `median`
- `std`

## Restricting by Height Range

Use `z_index_range=(start, stop)` to include only specific vertical bins (inclusive-exclusive index range).

```python
subset_mean_intensity, extent = calculate_voxel_stat(
    arrays[0],
    voxel_resolution=(1.0, 1.0, 1.0),
    dimension="Intensity",
    stat="mean",
    z_index_range=(2, 10)  # use bins 2..9
)
```

