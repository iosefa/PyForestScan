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

PyForestScan returns one rumple value per XY voxel cell. It first takes the
maximum height above ground in each column to form a canopy surface on the
same grid as `assign_voxels`. It then uses the cell and its eight neighbors
to estimate surface area following [Jenness (2004)](https://www.jennessent.com/downloads/WSB_32_3_Jenness.pdf).
The eight triangles are clipped to the central cell, and their combined
area is divided by that cell's ground area, `dx * dy`.

The result has the same XY resolution and grid alignment as canopy cover,
PAI, and FHD calculated with the same points and voxel resolution. The
neighboring cells describe the surface across each cell; this is a local
surface-area ratio, not a single average for the point cloud.

## Calculating Rumple

Pass the point array and a `(dx, dy, dz)` voxel resolution. The function
returns a 2D array and its spatial extent, ready to plot or save as a GeoTIFF:

```python
from pyforestscan.handlers import read_lidar, create_geotiff
from pyforestscan.calculate import calculate_rumple
from pyforestscan.visualize import plot_metric

file_path = "../example_data/20191210_5QKB020880.laz"
arrays = read_lidar(file_path, "EPSG:32605", hag=True)
points = arrays[0]

voxel_resolution = (5.0, 5.0, 1.0)
rumple, extent = calculate_rumple(points, voxel_resolution, min_height=2.0)

plot_metric("Rumple Index", rumple, extent, metric_name="Rumple", cmap="viridis")
create_geotiff(rumple, "rumple.tif", "EPSG:32605", extent)
```

## Filling Gaps

Use `interpolation="linear"` to fill missing canopy heights before calculating
rumple. Only empty cells are filled; measured heights are preserved.

```python
rumple, extent = calculate_rumple(
    points, voxel_resolution, min_height=2.0, interpolation="linear",
)
```

The default, `interpolation=None`, leaves gaps unfilled. The available methods
match the CHM options:

- `"linear"`: interpolate heights on triangles between observed cell centers.
- `"cubic"`: use a smooth cubic surface between observed cell centers.
- `"nearest"`: copy the closest observed height, using XY distances.

Linear and cubic interpolation leave cells outside the observed cell centers'
convex hull as NoData. They also leave gaps unfilled if there are too few
noncollinear samples to form a surface. Nearest can fill outside that hull,
but does not expand the raster extent. A complete 3x3 neighborhood is still
needed for rumple, so the outermost row and column remain NoData.

Interpolation happens before `min_height` is applied. Observed low or ground
cells remain subject to the height mask; they are not replaced with higher
canopy values. Interpolated heights below the threshold are masked too.

This follows the same sequence as lidR: its
[`p2r(na.fill = tin())`](https://search.r-project.org/CRAN/refmans/lidR/html/dsm_point2raster.html)
option fills the canopy height model before `rumple_index()` is calculated.
Here interpolation uses the gridded canopy maxima. For a numerical comparison,
use the same filled height grid in both implementations. Filling gaps changes
the estimated canopy surface and can affect roughness, so use consistent
resolution and interpolation settings when comparing results.

## Tiled GeoTIFF Output

For large EPT point clouds, use `metric="rumple"`:

```python
from pyforestscan.process import process_with_tiles

process_with_tiles(
    ept_file="../example_data/ept/ept.json",
    tile_size=(1000, 1000),
    output_path="rumple_tiles",
    metric="rumple",
    voxel_size=(5.0, 5.0, 1.0),
    srs="EPSG:32605",
    hag=True,
    rumple_min_height=2.0,
    interpolation="linear",  # Optional: fill canopy gaps before calculating rumple.
)
```

This writes `tile_<i>_<j>_rumple.tif` files. Tile dimensions must be multiples
of the XY voxel sizes. Output bounds expand to whole voxel cells, and the
last tile can be smaller than `tile_size` without changing pixel size.
Each tile reads at least one extra cell on every available side before
calculating rumple, then crops to its output grid. This also applies when
`buffer_size=0`, so adjacent tiles retain the neighboring canopy data needed
at their shared edges. `skip_existing` and `tile_indices` work as for the
other tiled metrics.

When interpolation is enabled, it uses the points in each buffered tile.
Increase `buffer_size` enough to include observed canopy around gaps near tile
edges. The one-cell minimum buffer supports the surface-area calculation, but
may not be enough for interpolation across larger gaps. Filled values can
differ from a whole-cloud calculation because fewer samples are available.
The `interpolation` option applies to CHM and rumple only.

## Notes

- The array is shaped `(X, Y)`, with Y ordered north to south, matching
  `assign_voxels` and `create_geotiff`.
- `dx` and `dy` determine both the surface sampling and output pixel size.
  Changing them changes the scale of canopy roughness being measured. Use
  the same resolution when comparing sites.
- `dz` is accepted as part of the shared voxel-resolution tuple. Actual
  maximum heights are used without rounding them to vertical bins, so
  changing `dz` alone does not change rumple.
- XY coordinates and heights must use the same linear units, such as meters
  in a projected CRS.
- A complete 3x3 canopy neighborhood is required. The outermost row/column,
  unfilled cells, and cells adjacent to missing or height-masked canopy
  remain NaN, written as NoData in the GeoTIFF. Grids smaller than three
  cells in either direction contain only NoData.
- `min_height` masks canopy maxima strictly below the threshold; it does
  not change the grid extent. Interpolation is optional and defaults to None.
- This replaces the earlier scalar API. Update
  `calculate_rumple(chm, cell_resolution)` calls to
  `rumple, extent = calculate_rumple(points, voxel_resolution)`.

## References

Jenness, Jeff S. 2004. "Calculating landscape surface area from digital
elevation models." Wildlife Society Bulletin 32 (3): 829-839.
<https://www.jennessent.com/downloads/WSB_32_3_Jenness.pdf>.

McElhinny, Chris, Phillip Gibbons, Cris Brack, and Juergen Bauhus. 2005.
"Forest and woodland stand structural complexity: Its definition and
measurement." Forest Ecology and Management 218 (1-3): 1-24.
<https://doi.org/10.1016/j.foreco.2005.08.034>.

Kane, Van R., Jonathan D. Bakker, Robert J. McGaughey, James A. Lutz,
Rolf F. Gersonde, and Jerry F. Franklin. 2010. "Examining conifer canopy
structural complexity across forest ages and elevations with LiDAR data."
Canadian Journal of Forest Research 40 (4): 774-787.
<https://doi.org/10.1139/X10-064>.
