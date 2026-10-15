# Rumple Index

## Theory

Rumple measures canopy surface complexity as the ratio of canopy surface
area to its projected ground area:

$$
\text{Rumple} = \frac{A_{\text{surface}}}{A_{\text{planar}}}
$$

A flat, horizontal canopy has a rumple value of 1.0. Slopes and height
variation increase its surface area and therefore its rumple value.

`calculate_rumple()` returns one value for a canopy height model (CHM).
`calculate_rumple_grid()` divides that surface into larger output cells and
returns one value per cell. Both functions offer the same two methods for
estimating surface area.

## Surface Resolution and Output Resolution

These resolutions answer different questions:

- **CHM resolution** determines how finely the canopy surface is sampled.
- **Rumple grid resolution** determines the area summarized by each result.

For example, a 1 m CHM can produce a 20 m rumple raster. Each output cell
summarizes surface area within a 20 m by 20 m footprint, using the
1 m canopy heights. The CHM is not first resampled to 20 m.

Output resolution must be a whole multiple of CHM resolution in each
direction. Equal resolutions are also allowed: this gives a local surface
area ratio at each CHM cell. Use the same CHM resolution, output resolution,
grid alignment, method, and treatment of gaps when comparing maps or sites.

## Calculating Rumple

Create the CHM first, then use it for either a single value or a raster:

```python
from pyforestscan.handlers import read_lidar, create_geotiff
from pyforestscan.calculate import (
    calculate_chm,
    calculate_rumple,
    calculate_rumple_grid,
)
from pyforestscan.visualize import plot_metric

file_path = "../example_data/20191210_5QKB020880.laz"
points = read_lidar(file_path, "EPSG:32605", hag=True)[0]

cell_resolution = (1.0, 1.0)
chm, chm_extent = calculate_chm(
    points, cell_resolution, interpolation="linear",
)

# One value for the whole CHM.
rumple_index = calculate_rumple(chm, cell_resolution)
print(f"Whole-CHM rumple: {rumple_index:.3f}")

# A 20 m rumple raster, calculated from the same 1 m canopy surface.
rumple, extent = calculate_rumple_grid(
    chm,
    cell_resolution,
    grid_resolution=20.0,
    extent=chm_extent,
)

plot_metric("Rumple Index", rumple, extent, metric_name="Rumple", cmap="viridis")
create_geotiff(rumple, "rumple_20m.tif", "EPSG:32605", extent)
```

`cell_resolution` is `(dx, dy)`. `grid_resolution` accepts either one number
for square pixels or `(dx, dy)` for rectangular pixels. XY coordinates and
heights must use the same linear units, such as meters in a projected CRS.

Pass the CHM extent as `[x_min, x_max, y_min, y_max]` to preserve its location.
The output starts at the CHM's northwest corner. If necessary, the last
column and row extend east and south to make whole output pixels. The added
space contributes neither surface area nor ground area. Without `extent`,
the CHM is placed at `[0, nx * dx, 0, ny * dy]`.

Input and output arrays have shape `(X, Y)`, with Y ordered north to south,
matching `calculate_chm()` and `create_geotiff()`.

## Choosing a Method

The `method` argument is available on both rumple functions:

| Method | Surface estimate | Missing heights |
| --- | --- | --- |
| `"two_triangle"` (default) | Split each patch of four neighboring CHM cell centers into two triangles along the northeast–southwest diagonal. | Use a patch only when all four heights are valid. |
| `"jenness"` | Use the central height and its eight neighbors to estimate the surface within each CHM cell. | Omit missing centers; substitute the central height for missing neighbors. Repeat edge heights at the outer raster boundary. |

Fixed-diagonal triangulation is described by
[FUSION's GridSurfaceStats](https://bpb-us-e1.wpmucdn.com/sites.uw.edu/dist/a/122/files/2017/12/FUSION_manual-1glx79l.pdf#page=96)
and by [Geng et al. (2019)](https://doi.org/10.3390/jmse7030076).

The [Jenness (2004)](https://www.jennessent.com/downloads/WSB_32_3_Jenness.pdf)
method forms eight triangles around each center and sums the portions inside
that center's cell. Its handling of missing heights and outer edges follows
[`sp::surfaceArea()`](https://search.r-project.org/CRAN/refmans/sp/html/surfaceArea.html),
which lidR uses for raster inputs to `rumple_index()`.

```python
rumple, extent = calculate_rumple_grid(
    chm,
    cell_resolution,
    grid_resolution=20.0,
    extent=chm_extent,
    method="jenness",
)
```

Both methods sum surface area and corresponding ground area within each
output cell, then divide those totals. They do not take an unweighted mean
of fine-cell rumple values. With `"two_triangle"`, triangle portions crossing
an output boundary contribute to each side according to their area. The
outer half-cell margin beyond the CHM centers has no triangle support and
is excluded. With `"jenness"`, each valid center contributes its full CHM
cell footprint.

The methods can give different values on the same CHM. Also, lidR's
[point-cloud rumple map example](https://r-lidar.github.io/lidRbook/outbox.html#rumple-index)
triangulates selected surface points within each output cell. That is a
different surface estimate from either CHM method here.

## Missing Heights

The `method` argument determines how missing heights are handled; there is
no separate missing-data setting. Both functions treat `NaN` and infinite
heights as missing. Finite heights, including zero, are used unless excluded
by `min_height`.

- **`"two_triangle"`:** Each surface patch needs four heights from a 2x2 group
  of CHM cells. If any height is missing, the entire patch is excluded from
  both surface area and ground area. For example, one missing CHM cell
  removes every patch that uses it, while complete patches still contribute.
- **`"jenness"`:** A cell with a missing center height contributes neither
  surface area nor ground area. A cell with a valid center still contributes
  when neighbors are missing: each missing neighbor is assigned the center
  height only for that cell's area calculation. For example, a valid 10 m
  center uses 10 m for a missing neighbor. The missing cell itself stays
  missing. At the outer raster boundary, edge heights are repeated to supply
  neighbors outside the raster.

Jenness's local height substitution follows the raster calculation in
[`sp::surfaceArea()`](https://search.r-project.org/CRAN/refmans/sp/html/surfaceArea.html),
which is used by lidR. It assumes a level surface toward a missing neighbor
and can reduce estimated roughness near gaps.

Each rumple value divides the included surface area by its corresponding
ground area. A partially observed output cell can therefore have a value;
it describes only the portion included by the selected method. If no area
is included, the result is `NaN` (NoData in the GeoTIFF).

## Optional Height Threshold

`min_height=None` is the default and uses all finite CHM heights. Set a
threshold when the analysis should exclude low vegetation or ground heights:

```python
rumple_index = calculate_rumple(chm, cell_resolution, min_height=2.0)
```

This treats heights strictly below 2 m as missing in a temporary copy used
for calculation. The input CHM is not modified, and excluded heights are
not replaced with zero. With `"two_triangle"`, patches using those heights
are omitted. With `"jenness"`, those center cells are omitted and valid
neighboring centers use their own heights in their place. The same threshold
is available in `calculate_rumple_grid()`.

The threshold is applied after CHM creation, including any interpolation.
Low areas and canopy gaps can be meaningful parts of the canopy surface;
leave the threshold as `None` to include them.

## Filling CHM Gaps

For either method, fill missing heights during CHM creation if interpolation
is appropriate for the data. `calculate_chm()` supports `"linear"` (its
default), `"cubic"`, `"nearest"`, or `None` to leave gaps unfilled:

```python
chm, chm_extent = calculate_chm(
    points, cell_resolution, interpolation="linear",
)
```

The resulting CHM can be inspected and reused for either rumple method.
The rumple functions themselves have no interpolation argument. Any gaps
remaining in the CHM follow the missing-height rules above. Interpolation
changes the estimated surface, so use consistent settings when comparing
rumple results.

## Tiled GeoTIFF Output

For large EPT point clouds, use `metric="rumple"` and specify both the fine
CHM resolution and the rumple output resolution:

```python
from pyforestscan.process import process_with_tiles

process_with_tiles(
    ept_file="../example_data/ept/ept.json",
    tile_size=(1000, 1000),
    output_path="rumple_tiles",
    metric="rumple",
    voxel_size=(1.0, 1.0, 1.0),  # XY resolution of the canopy surface.
    rumple_resolution=20.0,     # XY resolution of the output raster.
    rumple_method="two_triangle",
    srs="EPSG:32605",
    hag=True,
    interpolation="linear",
)
```

This writes `tile_<i>_<j>_rumple.tif` files. `rumple_resolution` is required
for this metric; `rumple_method` defaults to `"two_triangle"`. The vertical
voxel size does not round the canopy heights or affect rumple.

Tile dimensions must be whole multiples of the rumple output resolution.
Requested bounds expand to fine-grid cells aligned to absolute XY multiples
of the voxel sizes, and the output begins at their northwest corner. The
final east and south pixels can extend beyond the fine grid; the added space
is excluded from the area calculation. Rumple tile indices start in the
northwest, with `i` increasing
east and `j` increasing south.

`calculate_chm()` anchors its grid at the point minima, which can differ from
the tiled grid origin. To compare tiled and direct calculations numerically,
use the same canopy heights, fine-grid alignment, extent, and resolutions.

Each tile reads at least one extra fine-grid cell on every available side,
including when `buffer_size=0`. This supplies the neighboring heights needed
at internal tile boundaries. `skip_existing` and `tile_indices` work as for
the other tiled metrics.

The tiled `interpolation` option fills missing fine-grid canopy heights
before calculating rumple and applying `rumple_min_height`. Increase
`buffer_size` to include observed canopy around larger gaps near tile
boundaries. Interpolated values can differ from a whole-cloud calculation
when a buffered tile has fewer surrounding observations. This option
applies to CHM and rumple only. Use `rumple_min_height` to apply an optional
height threshold with the rules described above.

## References

Jenness, Jeff S. 2004. "Calculating landscape surface area from digital
elevation models." Wildlife Society Bulletin 32 (3): 829–839.
<https://www.jennessent.com/downloads/WSB_32_3_Jenness.pdf>.

Geng, Yuyang, et al. 2019. "Marine Coastline Polygonal Ridges and Surface
Roughness Development on a Salt-Crusted Playa: Recognition by
Structure-from-Motion Photogrammetry." Journal of Marine Science and
Engineering 7 (3): 76. Section 2.4, Figure 3 and equations 14–18.
<https://doi.org/10.3390/jmse7030076>.

McGaughey, Robert J. FUSION/LDV software manual. "GridSurfaceStats" and
"SurfaceStats."
<https://bpb-us-e1.wpmucdn.com/sites.uw.edu/dist/a/122/files/2017/12/FUSION_manual-1glx79l.pdf>.

Kane, Van R., Jonathan D. Bakker, Robert J. McGaughey, James A. Lutz,
Rolf F. Gersonde, and Jerry F. Franklin. 2010. "Examining conifer canopy
structural complexity across forest ages and elevations with LiDAR data."
Canadian Journal of Forest Research 40 (4): 774–787.
<https://doi.org/10.1139/X10-064>.
