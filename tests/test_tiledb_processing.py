"""Native TileDB inputs share the existing reading and raster metric workflows."""
import ast
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pdal
import pytest
import rasterio
from pyproj import CRS, Transformer
from rasterio.transform import from_origin

from pyforestscan.calculate import calculate_rumple_grid, calculate_voxel_stat
from pyforestscan.handlers import _reader_stage, read_lidar, write_tiledb
from pyforestscan.process import process_with_tiles


@pytest.fixture
def cloud_points():
    x, y = np.meshgrid(np.arange(11) + 0.5, np.arange(9) + 0.5, indexing="ij")
    canopy = 8.0 + 0.03 * x.ravel() ** 2 + 0.1 * y.ravel() ** 2
    dtype = [("X", "f8"), ("Y", "f8"), ("Z", "f8"),
             ("HeightAboveGround", "f8"), ("Classification", "u1"), ("Intensity", "u2")]
    points = np.zeros(2 * x.size, dtype=dtype)
    points["X"], points["Y"] = np.tile(x.ravel(), 2), np.tile(y.ravel(), 2)
    points["HeightAboveGround"][x.size:] = canopy
    points["Z"] = 100 + points["HeightAboveGround"]
    points["Classification"][:x.size] = 2
    points["Classification"][x.size:] = 5
    points["Intensity"] = np.arange(points.size)
    # A truly missing CHM cell near an internal tile boundary.
    return points[~((points["X"] == 4.5) & (points["Y"] == 4.5))]


@pytest.fixture
def tiledb_config(tmp_path, monkeypatch):
    if not hasattr(pdal.Reader, "tiledb") or not hasattr(pdal.Writer, "tiledb"):
        pytest.skip("Native PDAL TileDB reader and writer are not installed")
    monkeypatch.setenv("TILEDB_SM_COMPUTE_CONCURRENCY_LEVEL", "2")
    monkeypatch.setenv("TILEDB_SM_IO_CONCURRENCY_LEVEL", "2")
    path = tmp_path / "tiledb.conf"
    path.write_text("sm.compute_concurrency_level 2\nsm.io_concurrency_level 2\n")
    return str(path)


@pytest.fixture
def tiledb_cloud(tmp_path, cloud_points, tiledb_config):
    uri = str(tmp_path / "points")
    write_tiledb(cloud_points, uri, "EPSG:32605", config_file=tiledb_config)
    return uri, cloud_points, tiledb_config


@pytest.fixture
def tiledb_cloud_without_hag(tmp_path, cloud_points, tiledb_config):
    uri = str(tmp_path / "raw-points")
    fields = [name for name in cloud_points.dtype.names if name != "HeightAboveGround"]
    write_tiledb(cloud_points[fields], uri, "EPSG:32605", config_file=tiledb_config)
    return uri, cloud_points, tiledb_config


def _sort_points(arrays):
    points = np.concatenate(arrays)
    return points[np.argsort(points["Intensity"])]


def test_read_tiledb_directory_preserves_dimensions_and_metadata(tiledb_cloud):
    uri, points, config = tiledb_cloud
    reader = _reader_stage(uri, source_type="tiledb", tiledb_config=config)
    assert CRS(reader["default_srs"]).to_epsg() == 32605
    actual = _sort_points(read_lidar(uri, source_type="tiledb", tiledb_config=config))
    assert set(points.dtype.names) <= set(actual.dtype.names)
    for field in points.dtype.names:
        np.testing.assert_array_equal(actual[field], points[field])


@pytest.mark.parametrize("bounds", [([2, 7], [1, 5]), ([2, 7], [1, 5], [105, 120])])
def test_read_tiledb_native_bounds(tiledb_cloud, bounds):
    uri, points, config = tiledb_cloud
    keep = np.ones(points.size, dtype=bool)
    for dimension, limits in zip(("X", "Y", "Z"), bounds):
        keep &= (points[dimension] >= limits[0]) & (points[dimension] <= limits[1])
    actual = _sort_points(read_lidar(uri, source_type="tiledb", bounds=bounds, tiledb_config=config))
    np.testing.assert_array_equal(actual["Intensity"], points["Intensity"][keep])


@pytest.mark.parametrize("bounds", [([100, 101], [100, 101]), ([4.4, 4.6], [4.4, 4.6])])
def test_read_tiledb_empty_query_returns_none(tiledb_cloud, bounds):
    uri, _, config = tiledb_cloud
    assert read_lidar(uri, source_type="tiledb", bounds=bounds, tiledb_config=config) is None


def test_read_tiledb_crs_override_keeps_coordinates(tiledb_cloud, monkeypatch):
    uri, points, config = tiledb_cloud
    pipeline = pdal.Pipeline
    recorded = []

    def capture(definition, *args, **kwargs):
        recorded.append(json.loads(definition))
        return pipeline(definition, *args, **kwargs)

    monkeypatch.setattr("pyforestscan.handlers.pdal.Pipeline", capture)
    actual = _sort_points(read_lidar(uri, srs="EPSG:32606", source_type="tiledb", tiledb_config=config))
    assert recorded[0]["pipeline"][0]["spatialreference"] == "EPSG:32606"
    assert len(recorded[0]["pipeline"]) == 1
    for field in ("X", "Y", "Z"):
        np.testing.assert_array_equal(actual[field], points[field])


def test_read_tiledb_reprojects_after_native_query(tiledb_cloud):
    uri, points, config = tiledb_cloud
    bounds = ([2, 7], [1, 5], [105, 120])
    keep = ((points["X"] >= 2) & (points["X"] <= 7) &
            (points["Y"] >= 1) & (points["Y"] <= 5) & (points["Z"] >= 105))
    expected = points[keep]
    actual = _sort_points(read_lidar(uri, srs="EPSG:4326", source_type="tiledb",
                                    bounds=bounds, reproject=True, tiledb_config=config))
    x, y = Transformer.from_crs(32605, 4326, always_xy=True).transform(expected["X"], expected["Y"])
    np.testing.assert_array_equal(actual["Intensity"], expected["Intensity"])
    np.testing.assert_allclose(actual["X"], x)
    np.testing.assert_allclose(actual["Y"], y)


def test_read_reproject_requires_target_before_io():
    with pytest.raises(ValueError, match="srs is required"):
        read_lidar("unused", source_type="tiledb", reproject=True)


def test_read_tiledb_hag_polygon_and_thinning(tiledb_cloud_without_hag):
    uri, points, config = tiledb_cloud_without_hag
    arrays = read_lidar(uri, source_type="tiledb", hag=True, tiledb_config=config)
    actual = _sort_points(arrays)
    np.testing.assert_allclose(actual["HeightAboveGround"], points["HeightAboveGround"])
    polygon = "POLYGON ((1 1, 7 1, 7 7, 1 7, 1 1))"
    cropped = _sort_points(read_lidar(uri, source_type="tiledb", crop_poly=True, poly=polygon,
                                     thin_radius=1.5, tiledb_config=config))
    assert 0 < cropped.size < points.size
    assert np.all((cropped["X"] >= 1) & (cropped["X"] <= 7))
    assert np.all((cropped["Y"] >= 1) & (cropped["Y"] <= 7))


def _write_dtm(path, crs="EPSG:32605"):
    with rasterio.open(path, "w", driver="GTiff", width=16, height=16,
                       count=1, dtype="float32", crs=crs,
                       transform=from_origin(-2, 12, 1, 1)) as dst:
        dst.write(np.full((16, 16), 100, dtype="float32"), 1)


def test_read_tiledb_dtm_uses_metadata_crs(tiledb_cloud_without_hag, tmp_path):
    uri, points, config = tiledb_cloud_without_hag
    dtm = tmp_path / "dtm.tif"
    _write_dtm(dtm)
    actual = _sort_points(read_lidar(uri, source_type="tiledb", hag_dtm=True,
                                    dtm=str(dtm), tiledb_config=config))
    np.testing.assert_allclose(actual["HeightAboveGround"], points["HeightAboveGround"])
    _write_dtm(dtm, "EPSG:32606")
    with pytest.raises(ValueError, match="DTM CRS"):
        read_lidar(uri, source_type="tiledb", hag_dtm=True, dtm=str(dtm), tiledb_config=config)


def test_read_tiledb_polygon_file_uses_native_crs(tiledb_cloud, monkeypatch, tmp_path):
    uri, _, config = tiledb_cloud
    poly = tmp_path / "polygon.gpkg"
    poly.touch()
    polygon = "POLYGON ((1 1, 7 1, 7 7, 1 7, 1 1))"
    monkeypatch.setattr("pyforestscan.handlers.load_polygon_from_file",
                        lambda path: (polygon, "EPSG:32605"))
    # Polygon cropping precedes reprojection, so its CRS is the native one.
    arrays = read_lidar(uri, source_type="tiledb", crop_poly=True, poly=str(poly),
                        srs="EPSG:4326", reproject=True, tiledb_config=config)
    assert sum(array.size for array in arrays) > 0
    monkeypatch.setattr("pyforestscan.handlers.load_polygon_from_file",
                        lambda path: (polygon, "EPSG:32606"))
    with pytest.raises(ValueError, match="polygon CRS"):
        read_lidar(uri, source_type="tiledb", crop_poly=True, poly=str(poly), tiledb_config=config)


def test_explicit_copc_source_uses_indexed_reader(tmp_path):
    path = tmp_path / "points.copc.laz"
    path.touch()
    reader = _reader_stage(path, source_type="copc", bounds=([0, 1], [0, 1]))
    assert reader["type"] == "readers.copc"
    assert reader["bounds"] == "([0, 1], [0, 1])"


@pytest.mark.parametrize("filename", ["points.las", "points.copc.laz"])
def test_legacy_las_bounds_remain_ignored(tmp_path, filename):
    path = tmp_path / filename
    path.touch()
    reader = _reader_stage(path, bounds=([0, 1], [0, 1]))
    assert reader["type"] == "readers.las"
    assert "bounds" not in reader


@pytest.mark.parametrize("method", ["two_triangle", "jenness"])
def test_tiledb_rumple_tiles_match_whole_grid(tiledb_cloud, tmp_path, method):
    uri, points, config = tiledb_cloud
    chm, extent = calculate_voxel_stat(points, (1, 1, 1), "HeightAboveGround", "max")
    expected, extent = calculate_rumple_grid(chm, (1, 1), (2, 3), extent=extent, method=method)
    output = tmp_path / "rumple"
    process_with_tiles(uri, (4, 6), str(output), "rumple", (1, 1, 1),
                       source_type="tiledb", tiledb_config=config,
                       rumple_resolution=(2, 3), rumple_method=method, buffer_size=0)
    mosaic = np.full(expected.shape, np.nan)
    covered = np.zeros(expected.shape, dtype=int)
    for path in output.glob("*.tif"):
        with rasterio.open(path) as src:
            assert src.crs.to_epsg() == 32605
            assert src.res == (2, 3)
            x0 = int(round((src.bounds.left - extent[0]) / 2))
            y0 = int(round((extent[3] - src.bounds.top) / 3))
            xs, ys = slice(x0, x0 + src.width), slice(y0, y0 + src.height)
            mosaic[xs, ys] = src.read(1, masked=True).filled(np.nan).T
            covered[xs, ys] += 1
    np.testing.assert_array_equal(covered, 1)
    np.testing.assert_allclose(mosaic, expected, equal_nan=True)


@pytest.mark.parametrize("metric", ["chm", "pai", "fhd", "cover"])
def test_tiledb_other_metrics_match_ept_inputs(tiledb_cloud, tmp_path, monkeypatch, metric):
    uri, points, config = tiledb_cloud
    tiledb_output, ept_output = tmp_path / "tiledb", tmp_path / "ept"
    kwargs = dict(tile_size=(6, 6), metric=metric, voxel_size=(1, 1, 1),
                  bounds=([0, 12], [0, 9]), srs="EPSG:32605", buffer_size=0)
    process_with_tiles(uri, output_path=str(tiledb_output), source_type="tiledb",
                       tiledb_config=config, **kwargs)

    def fake_ept(definition):
        stage = json.loads(definition)["pipeline"][0]
        assert stage["type"] == "readers.ept"
        bounds = ast.literal_eval(stage["bounds"])
        keep = np.ones(points.size, dtype=bool)
        for dimension, limits in zip(("X", "Y", "Z"), bounds):
            keep &= (points[dimension] >= limits[0]) & (points[dimension] <= limits[1])
        return SimpleNamespace(arrays=[points[keep]], execute=lambda: int(keep.sum()))

    monkeypatch.setattr("pyforestscan.process.pdal.Pipeline", fake_ept)
    process_with_tiles("fake_ept", output_path=str(ept_output), **kwargs)
    files = sorted(path.name for path in tiledb_output.glob("*.tif"))
    assert files == sorted(path.name for path in ept_output.glob("*.tif"))
    assert files
    for name in files:
        with rasterio.open(tiledb_output / name) as actual, rasterio.open(ept_output / name) as expected:
            assert actual.transform == expected.transform
            assert actual.crs == expected.crs
            np.testing.assert_allclose(actual.read(), expected.read(), equal_nan=True)


def test_tiledb_tiling_hag_and_crs_override(tiledb_cloud_without_hag, tmp_path):
    uri, _, config = tiledb_cloud_without_hag
    output = tmp_path / "hag"
    process_with_tiles(uri, (12, 9), str(output), "rumple", (1, 1, 1),
                       source_type="tiledb", tiledb_config=config, srs="EPSG:32606",
                       rumple_resolution=(2, 3), hag=True, bounds=([0, 12], [0, 9]))
    with rasterio.open(output / "tile_0_0_rumple.tif") as src:
        assert src.crs.to_epsg() == 32606
        assert src.bounds.left == 0
        assert src.bounds.top == 9
        assert np.isfinite(src.read(1, masked=True).compressed()).all()


def test_tiledb_s3_config_forwarded_for_reads_and_tiling(monkeypatch, tmp_path):
    uri, config = "s3://example-bucket/points", "/tmp/example-tiledb.conf"
    reader = MagicMock(return_value=None)
    monkeypatch.setattr("pyforestscan.handlers._tiledb_reader", reader)
    assert read_lidar(uri, source_type="tiledb", tiledb_config=config) is None
    reader.assert_called_once_with(uri, bounds=None, config_file=config)

    info = {"bounds": (0, 4, 0, 4, 0, 10), "srs": "EPSG:32605"}
    metadata = MagicMock(return_value=info)
    monkeypatch.setattr("pyforestscan.process.get_tiledb_info", metadata)
    reader = MagicMock(return_value=None)
    monkeypatch.setattr("pyforestscan.process._tiledb_reader", reader)
    process_with_tiles(uri, (2, 2), str(tmp_path / "output"), "chm", (1, 1, 1),
                       source_type="tiledb", tiledb_config=config)
    metadata.assert_called_once_with(uri, config_file=config)
    assert reader.call_count == 4
    for call in reader.call_args_list:
        assert call.args == (uri,)
        assert call.kwargs["config_file"] == config
        assert call.kwargs["info"] is info
        assert len(call.kwargs["bounds"]) == 3


def test_tiledb_empty_array_tiling_returns_cleanly(monkeypatch, tmp_path):
    monkeypatch.setattr("pyforestscan.process.get_tiledb_info",
                        lambda *args, **kwargs: {"bounds": None, "srs": None})
    pipeline = MagicMock()
    monkeypatch.setattr("pyforestscan.process.pdal.Pipeline", pipeline)
    output = tmp_path / "empty"
    process_with_tiles("empty-array", (4, 4), str(output), "chm", (1, 1, 1), source_type="tiledb")
    assert not output.exists()
    pipeline.assert_not_called()


def test_tiledb_tiling_requires_crs_when_metadata_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("pyforestscan.process.get_tiledb_info",
                        lambda *args, **kwargs: {"bounds": (0, 4, 0, 4, 0, 10), "srs": None})
    with pytest.raises(ValueError, match="provide srs"):
        process_with_tiles("array", (4, 4), str(tmp_path), "chm", (1, 1, 1), source_type="tiledb")


def test_tiling_rejects_unsupported_source_before_io(tmp_path):
    with pytest.raises(ValueError, match="source_type"):
        process_with_tiles("array", (4, 4), str(tmp_path), "chm", (1, 1, 1), source_type="las")


@pytest.mark.parametrize("bounds", [([100, 101], [100, 101]), ([4.4, 4.6], [4.4, 4.6])])
def test_tiledb_tiling_empty_spatial_query_writes_no_rasters(tiledb_cloud, tmp_path, bounds):
    uri, _, config = tiledb_cloud
    output = tmp_path / "empty-query"
    process_with_tiles(uri, (1, 1), str(output), "chm", (0.1, 0.1, 1),
                       source_type="tiledb", tiledb_config=config,
                       bounds=bounds, buffer_size=0)
    assert not list(output.glob("*.tif"))


def test_read_tiledb_dtm_matches_reprojected_crs(tiledb_cloud_without_hag, tmp_path):
    uri, points, config = tiledb_cloud_without_hag
    x, y = Transformer.from_crs(32605, 32606, always_xy=True).transform(points["X"], points["Y"])
    left, top = np.floor(x.min()) - 2, np.ceil(y.max()) + 2
    width = int(np.ceil(x.max()) - left + 2)
    height = int(top - np.floor(y.min()) + 2)
    dtm = tmp_path / "target-dtm.tif"
    with rasterio.open(dtm, "w", driver="GTiff", width=width, height=height,
                       count=1, dtype="float32", crs="EPSG:32606",
                       transform=from_origin(left, top, 1, 1)) as dst:
        dst.write(np.full((height, width), 100, dtype="float32"), 1)
    actual = _sort_points(read_lidar(uri, source_type="tiledb", srs="EPSG:32606",
                                    reproject=True, hag_dtm=True, dtm=str(dtm), tiledb_config=config))
    np.testing.assert_allclose(actual["X"], x)
    np.testing.assert_allclose(actual["Y"], y)
    np.testing.assert_allclose(actual["HeightAboveGround"], points["HeightAboveGround"])


@pytest.mark.parametrize("metric", ["chm", "pai", "fhd", "cover", "rumple"])
@pytest.mark.parametrize("axis, offset", [("x", 0.5), ("x", 0.0), ("y", 0.0), ("both", 0.0)])
def test_tiledb_single_axis_cloud_creates_raster(tmp_path, tiledb_config, metric, axis, offset):
    points = np.zeros(3, dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("HeightAboveGround", "f8")])
    points["X"] = 500_000 + (offset if axis in ("x", "both") else np.arange(3) + 0.5)
    points["Y"] = 2_000_000 + (offset if axis in ("y", "both") else np.arange(3) + 0.5)
    points["HeightAboveGround"] = [2, 3, 4]
    points["Z"] = 100 + points["HeightAboveGround"]
    uri, output = str(tmp_path / "line-points"), tmp_path / "line-metric"
    write_tiledb(points, uri, "EPSG:32605", config_file=tiledb_config)
    process_with_tiles(uri, (4, 4), str(output), metric, (1, 1, 1),
                       source_type="tiledb", tiledb_config=tiledb_config,
                       rumple_resolution=2 if metric == "rumple" else None,
                       buffer_size=0)
    paths = list(output.glob("*.tif"))
    assert len(paths) == 1
    with rasterio.open(paths[0]) as src:
        assert src.width >= 1 and src.height >= 1
        assert src.res == ((2, 2) if metric == "rumple" else (1, 1))
        if metric == "chm":
            assert src.read(1, masked=True).max() == 4


@pytest.mark.parametrize("bounds", [([0.5, 0.5], [0, 5]), ([0, 5], [1, 1])])
def test_tiledb_explicit_bounds_require_positive_spans(tiledb_cloud, tmp_path, bounds):
    uri, _, config = tiledb_cloud
    with pytest.raises(ValueError, match="positive width and height"):
        process_with_tiles(uri, (4, 4), str(tmp_path / "invalid"), "chm", (1, 1, 1),
                           source_type="tiledb", tiledb_config=config, bounds=bounds)


@pytest.mark.parametrize("coordinate", [0.0, 0.5])
def test_single_point_gridline_keeps_one_voxel_and_chm_cell(coordinate):
    from pyforestscan.calculate import assign_voxels, calculate_chm

    points = np.array([(coordinate, coordinate, 0.0)],
                      dtype=[("X", "f8"), ("Y", "f8"), ("HeightAboveGround", "f8")])
    voxels, _ = assign_voxels(points, (1, 1, 1))
    assert voxels.shape == (1, 1, 1)
    assert voxels.sum() == 1
    chm, _ = calculate_chm(points, (1, 1, 1))
    np.testing.assert_array_equal(chm, [[0]])


@pytest.mark.parametrize("metric", ["chm", "pai", "cover", "fhd"])
@pytest.mark.parametrize("constant_axis", ["X", "Y"])
def test_tiledb_long_single_axis_cloud_with_default_buffer(tmp_path, tiledb_config, metric, constant_axis):
    points = np.zeros(101, dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("HeightAboveGround", "f8")])
    points["X"] = 500_000.5
    points["Y"] = 2_000_000.5
    varying_axis = "Y" if constant_axis == "X" else "X"
    points[varying_axis] += np.arange(101)
    points["Z"], points["HeightAboveGround"] = 105, 5
    uri, output = str(tmp_path / "long-line"), tmp_path / "long-line-metric"
    write_tiledb(points, uri, "EPSG:32605", config_file=tiledb_config)
    process_with_tiles(uri, (100, 100), str(output), metric, (1, 1, 1),
                       source_type="tiledb", tiledb_config=tiledb_config)
    paths = list(output.glob("*.tif"))
    assert len(paths) == 1
    with rasterio.open(paths[0]) as src:
        assert src.res == (1, 1)
        assert src.crs.to_epsg() == 32605
        if constant_axis == "X":
            assert src.width == 1 and src.height >= 80
        else:
            assert src.height == 1 and src.width >= 80
        assert src.bounds.right > src.bounds.left
        assert src.bounds.top > src.bounds.bottom
