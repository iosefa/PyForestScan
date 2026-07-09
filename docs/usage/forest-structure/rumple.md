# Rumple Index

## Theory

Rumple is a measure of canopy surface complexity. It is defined as the ratio
of canopy surface area to projected ground area:

$$
\text{Rumple} = \frac{A_{\text{surface}}}{A_{\text{planar}}}
$$

Where:

- \( A_{\text{surface}} \) is the area of the canopy surface.
- \( A_{\text{planar}} \) is the projected ground area beneath that surface.

A flat canopy has a rumple value of 1.0. More structurally complex or
corrugated canopies have values greater than 1.0.

PyForestScan computes rumple as a gridded point-cloud metric. For each output
cell, points inside the cell are treated as a local triangulated canopy surface.
Rumple is then calculated as the 3D surface area of those triangles divided by
their projected planar ground area.

## Calculating Rumple

To calculate rumple:

```python
from pyforestscan.handlers import read_lidar
from pyforestscan.calculate import calculate_rumple
from pyforestscan.visualize import plot_metric

file_path = "../example_data/20191210_5QKB020880.laz"
arrays = read_lidar(file_path, "EPSG:32605", hag=True)
points = arrays[0]

voxel_resolution = (10.0, 10.0)
rumple, extent = calculate_rumple(points, voxel_resolution, min_height=2.0)

plot_metric(
    "Rumple Index",
    rumple,
    extent,
    metric_name="Rumple",
    cmap="viridis",
)
```

## Notes

- `calculate_rumple` returns a 2D raster and extent, matching the pattern used
  by gridded metrics such as CHM.
- `min_height` can be used to exclude low vegetation before calculating
  canopy surface complexity.
- Cells with fewer than three unique point locations cannot form a triangulated
  surface and are returned as `NaN`.
- The projected ground area is the planar area covered by valid triangles
  inside each output cell, not the full rectangular cell footprint.

## References

McElhinny, Chris, Phillip Gibbons, Cris Brack, and Juergen Bauhus. 2005.
"Forest and woodland stand structural complexity: Its definition and
measurement." Forest Ecology and Management 218 (1-3): 1-24.
<https://doi.org/10.1016/j.foreco.2005.08.034>.

Kane, Van R., Jonathan D. Bakker, Robert J. McGaughey, James A. Lutz,
Rolf F. Gersonde, and Jerry F. Franklin. 2010. "Examining conifer canopy
structural complexity across forest ages and elevations with LiDAR data."
Canadian Journal of Forest Research 40 (4): 774-787.
<https://doi.org/10.1139/X10-064>.
