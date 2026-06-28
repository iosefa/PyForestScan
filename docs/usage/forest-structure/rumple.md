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

PyForestScan computes rumple from a Canopy Height Model (CHM) by treating the
CHM as a triangulated surface over the raster grid and summing surface area
over valid 2x2 CHM patches.

## Calculating Rumple

To calculate rumple:

```python
from pyforestscan.handlers import read_lidar
from pyforestscan.calculate import calculate_chm, calculate_rumple

file_path = "../example_data/20191210_5QKB020880.laz"
arrays = read_lidar(file_path, "EPSG:32605", hag=True)
points = arrays[0]

cell_resolution = (5.0, 5.0)
chm, extent = calculate_chm(points, cell_resolution, interpolation="linear")
rumple = calculate_rumple(chm, cell_resolution, min_height=2.0)

print(f"Rumple: {rumple:.3f}")
```

## Notes

- `calculate_rumple` returns a single scalar value, not a raster.
- `min_height` can be used to exclude low vegetation before calculating
  canopy surface complexity.
- Interpolating the CHM before calculating rumple may fill gaps, but it can
  also smooth the canopy surface and reduce rumple slightly.

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
