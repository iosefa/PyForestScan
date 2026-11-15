# TileDB Point Clouds

TileDB stores a point cloud as a sparse array that can be queried by location.
PyForestScan can create these arrays, read a selected area into NumPy, and
process large arrays into forest-metric GeoTIFF tiles.

Points retain numeric attributes such as `Classification`, `Intensity`, and
`HeightAboveGround`. You choose the voxel size and raster resolution when
calculating metrics, independently of how TileDB stores the points.

See the [installation guide](../installation.md) for setup.

## Create an Array

For a LAS, LAZ, COPC, or EPT source, `convert_to_tiledb()` streams points into
a new array without loading the entire point cloud into NumPy:

```python
from pyforestscan.handlers import convert_to_tiledb

point_count = convert_to_tiledb(
    "forest.laz",
    "forest.tiledb",
    chunk_size=1_000_000,
)
print(f"Stored {point_count:,} points")
```

The source CRS is retained. If the source has no CRS metadata, supply its
coordinate reference system with `srs="EPSG:32605"`. This describes the
existing coordinates; it does not reproject them. `chunk_size` is a point
count for streaming batches, not a spatial resolution.

If the data is already in memory, use `write_tiledb()` with one structured
NumPy array or a list of arrays with the same fields and dtypes:

```python
from pyforestscan.handlers import read_lidar, write_tiledb

arrays = read_lidar("forest.laz", "EPSG:32605", hag=True)
write_tiledb(arrays, "forest_with_heights.tiledb", "EPSG:32605")
```

Each input must be a nonempty, one-dimensional structured NumPy array.
`X`, `Y`, and `Z` must contain finite `float64` values. Other fields must be
scalar signed or unsigned integers (8, 16, 32, or 64 bits), or `float32` or
`float64`, with native byte order. Supply LAS flags as unpacked PDAL fields,
such as `Classification` and `ReturnNumber`, rather than a packed `BitFields`
field. The writer retains coordinate precision and duplicate points. Reading
may return points in a different order.

Both creation functions require a new destination: they do not overwrite an
existing array or append to it. A local TileDB array is a directory; the
`.tiledb` suffix is a naming convention rather than a required extension.

## Inspect and Read an Array

`get_tiledb_info()` returns the bounds and CRS without loading the points:

```python
from pyforestscan.tiledb_io import get_tiledb_info

info = get_tiledb_info("forest_with_heights.tiledb")
print(info["bounds"])
print(info["srs"])
```

`bounds` is `(xmin, xmax, ymin, ymax, zmin, zmax)` for the points actually
stored in the array. It is `None` for an empty array. `srs` is the stored CRS
in WKT form, or `None` when CRS metadata is absent.

Use `source_type="tiledb"` when reading an array. Supply XY bounds to select
an area, or add a third pair for a Z range. The bounds use the stored
coordinate system; Z refers to point elevation, not `HeightAboveGround`.

```python
from pyforestscan.handlers import read_lidar

arrays = read_lidar(
    "forest_with_heights.tiledb",
    source_type="tiledb",
    bounds=([500000, 500100], [2200000, 2200100]),
)
if arrays is not None:
    points = arrays[0]
    print(points.dtype.names)
```

The result is a list of structured NumPy arrays, or `None` when the query
returns no points. Omitting `srs` uses the array's stored CRS. If the metadata
has no CRS, pass the known source CRS explicitly.

`read_lidar()` loads all selected points into memory. Omitting `bounds`
selects the entire array. Use a bounded query or tiled processing when a
whole-array read would be too large.

## Calculate Forest Metrics

Read a manageable area and use the same calculation functions as for other
point-cloud sources. For example, an array containing `HeightAboveGround`
can provide a canopy height model:

```python
from pyforestscan.calculate import calculate_chm
from pyforestscan.handlers import create_geotiff

chm, extent = calculate_chm(points, (1.0, 1.0), interpolation=None)
create_geotiff(chm, "canopy_height.tif", info["srs"], extent)
```

If heights have not been calculated, use `hag=True` with ground-classified
points, or `hag_dtm=True` with `dtm=...`, when reading or processing. Include
surrounding ground observations when selecting bounds for height
normalization.

For large arrays, `process_with_tiles()` reads each area with its buffer and
writes the resulting metric as GeoTIFF tiles:

```python
from pyforestscan.process import process_with_tiles

process_with_tiles(
    "forest_with_heights.tiledb",
    tile_size=(100, 100),
    output_path="chm_tiles",
    metric="chm",
    voxel_size=(1.0, 1.0, 1.0),
    source_type="tiledb",
    interpolation=None,
)
```

The first argument identifies the source array. Its parameter name is
`ept_file`, so pass the array URI there if using a keyword argument. With
`source_type="tiledb"`, the processor obtains its default bounds and CRS
from the array's occupied extent and metadata. Supply `srs` if the array has
no CRS metadata.

TileDB inputs support CHM, canopy cover, PAI, FHD, and rumple. Their metric
arguments, buffers, and GeoTIFF outputs work the same way as for EPT inputs.
For rumple, use the XY components of `voxel_size` for the fine canopy surface
and set `rumple_resolution` for the output grid. See the
[rumple guide](forest-structure/rumple.md) for that distinction.

## Local Storage and S3

Supported inputs are PDAL-compatible sparse arrays with exactly three
`float64` dimensions named `X`, `Y`, and `Z`, stored in local directories or
at `s3://` URIs. Attributes must be fixed-size, non-nullable numeric scalars
using the integer or floating-point types described above. These checks also
apply to arrays created outside PyForestScan. A native PDAL `BitFields`
attribute, if present, must be `uint16`.

These APIs do not read dense raster arrays or managed TileDB Cloud
`tiledb://` URIs.

For S3, pass a TileDB configuration file to both metadata access and point
reads. For example, a file named `tiledb.config` can specify the bucket's
region:

```text
vfs.s3.region us-west-2
```

Use the AWS credentials configured for your environment. See TileDB's
[S3 authentication and configuration](https://documentation.cloud.tiledb.com/academy/structure/arrays/tutorials/advanced/backends/amazon-s3/)
and [configuration file options](https://documentation.cloud.tiledb.com/academy/structure/arrays/tutorials/basics/configuration/).

```python
uri = "s3://your-bucket/point-clouds/forest"
config = "tiledb.config"

info = get_tiledb_info(uri, config_file=config)
arrays = read_lidar(
    uri,
    source_type="tiledb",
    bounds=([500000, 500100], [2200000, 2200100]),
    tiledb_config=config,
)
```

Use `config_file` with `write_tiledb()`, `convert_to_tiledb()`, and
`get_tiledb_info()`. Use `tiledb_config` with `read_lidar()` and
`process_with_tiles()`. The configuration applies to TileDB access, including
metadata reads.

Try the [TileDB notebook](../examples/working-with-tiledb.ipynb) for a complete
local example with a small generated point cloud, bounded reads, streaming
conversion, and tiled GeoTIFF output.

See the [TileDB IO API reference](../api/tiledb_io.md) for complete arguments
and return values for metadata inspection, writing, and streaming conversion.
