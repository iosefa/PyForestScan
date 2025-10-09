# Canopy Cover (GEDI-style)

## Definition

Canopy cover at height z is the fraction of ground area occluded by vegetation above z meters (Height Above Ground).
Following GEDI, we model the gap probability using the Beer–Lambert law:

Cover(z) = 1 − exp(−k · PAI_above(z))

Where:
- `k` is the extinction coefficient (default 0.5 for spherical leaf-angle assumption).
- `PAI_above(z)` is the vertical integral of Plant Area Density (PAD) above height `z`.

Common choices:
- Tree canopy cover (GEDI-like): z = 2.0 m.
- Total vegetative cover: z = 0.0 m.

## Usage

```python
from pyforestscan.handlers import read_lidar
from pyforestscan.filters import filter_hag
from pyforestscan.calculate import assign_voxels, calculate_pad, calculate_canopy_cover
from pyforestscan.visualize import plot_metric

file_path = "../example_data/20191210_5QKB020880.laz"

# Ensure HAG is present for PAD/PAI calculations
arrays = read_lidar(file_path, "EPSG:32605", hag=True)
arrays = filter_hag(arrays)
points = arrays[0]

voxel_resolution = (5, 5, 1)  # dx, dy, dz in meters
voxels, extent = assign_voxels(points, voxel_resolution)

pad = calculate_pad(voxels, voxel_resolution[-1])

# GEDI-like canopy cover at z=2 m
cover = calculate_canopy_cover(pad, voxel_height=voxel_resolution[-1], min_height=2.0, k=0.5)

plot_metric('Canopy Cover (z=2 m)', cover, extent, metric_name='Cover', cmap='viridis')
```

## Notes

- As z increases, canopy cover decreases because less foliage lies above higher thresholds.
- `calculate_canopy_cover` returns NaN where PAD is entirely missing in the integration range.
- When tiling large EPT datasets, use `process_with_tiles(..., metric="cover", voxel_height=..., cover_min_height=2.0, cover_k=0.5)` to write GeoTIFF tiles.

