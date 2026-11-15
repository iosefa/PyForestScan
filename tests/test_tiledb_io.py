"""TileDB metadata and lossless point IO against the real native plugin."""

import json

import numpy as np
import pdal
import pytest
import tiledb
from pyproj import CRS

from pyforestscan import tiledb_io as io


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("TILEDB_SM_COMPUTE_CONCURRENCY_LEVEL", "2")
    monkeypatch.setenv("TILEDB_SM_IO_CONCURRENCY_LEVEL", "2")
    path = tmp_path / "tiledb.conf"
    path.write_text("sm.compute_concurrency_level 2\nsm.io_concurrency_level 2\nsm.memory_budget 10000000\n")
    return str(path)


@pytest.fixture
def native(config):
    if not hasattr(pdal.Reader, "tiledb") or not hasattr(pdal.Writer, "tiledb"):
        pytest.skip("Native PDAL TileDB reader and writer are not installed")
    return config


@pytest.fixture
def points():
    points = np.zeros(4, dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8"),
                               ("HeightAboveGround", "f4"), ("Classification", "u1"),
                               ("ReturnNumber", "u1"), ("Intensity", "u8"),
                               ("CustomSigned", "i8")])
    points["X"] = [400000.123456789, 400000.123456789, 400010.987654321, 400010.987654321]
    points["Y"] = [2200000.987654321, 2200000.987654321, 2200010.123456789, 2200010.123456789]
    points["Z"] = [3.456789123, 3.456789123, 115.2362238732553, 115.2362238732553]
    points["HeightAboveGround"] = [1.234567, 2.345678, 5.678912, 5.678912]
    points["Classification"] = [2, 5, 7, 7]
    points["ReturnNumber"] = [255, 33, 1, 1]
    points["Intensity"] = [2**53 + 1, 2**53 + 2, 65536, 65536]
    points["CustomSigned"] = [-2**53 - 1, 0, 17, 17]
    return points


def _native_read(uri, config, pipeline_type=None):
    factory = pdal.Pipeline if pipeline_type is None else pipeline_type
    pipeline = factory(json.dumps([{"type": "readers.tiledb", "filename": str(uri),
                                   "config_file": config}]))
    pipeline.execute()
    return np.concatenate(pipeline.arrays), pipeline


def _assert_same_points(actual, expected):
    assert set(actual.dtype.names) == set(expected.dtype.names)
    # Compare rows as multisets: TileDB intentionally does not promise order.
    names = expected.dtype.names
    actual_rows = sorted(tuple(actual[name][i].item() for name in names) for i in range(len(actual)))
    expected_rows = sorted(tuple(expected[name][i].item() for name in names) for i in range(len(expected)))
    assert actual_rows == expected_rows


def _create_array(uri, config, *, names=("X", "Y", "Z"), coordinate_dtype="f8",
                  attribute_dtype="f8", nullable=False, var=False, sparse=True):
    ctx = io._context(config)
    domain = tiledb.Domain(*[tiledb.Dim(name=name, domain=(-100, 100),
                                      dtype=coordinate_dtype, ctx=ctx) for name in names], ctx=ctx)
    attr = tiledb.Attr(name="Value", dtype=attribute_dtype, nullable=nullable, var=var, ctx=ctx)
    schema = tiledb.ArraySchema(domain=domain, attrs=[attr], sparse=sparse,
                                allows_duplicates=sparse, ctx=ctx)
    tiledb.Array.create(str(uri), schema, ctx=ctx)
    return ctx


def test_config_file_values_are_actually_loaded(config):
    assert io._context(config).config()["sm.memory_budget"] == "10000000"


def test_lossless_native_roundtrip_precision_fields_duplicates_and_crs(tmp_path, points, native):
    uri = tmp_path / "points"
    before = points.copy()
    io.write_tiledb([points[:2], points[2:]], uri, "EPSG:32605", config_file=native, chunk_size=2)
    actual, _ = _native_read(uri, native)
    _assert_same_points(actual, points)
    np.testing.assert_array_equal(points, before)
    info = io.get_tiledb_info(uri, config_file=native)
    assert info["bounds"] == tuple(v for name in ("X", "Y", "Z")
                                   for v in (points[name].min(), points[name].max()))
    assert CRS(info["srs"]).to_epsg() == 32605
    with tiledb.open(str(uri), ctx=io._context(native)) as array:
        assert array.schema.allows_duplicates
        metadata = json.loads(bytes(array.meta["_pdal"]).decode().rstrip("\x00"))
        assert CRS(metadata["root"]["writers.tiledb"]["spatialreference"]).to_epsg() == 32605
        assert "BitFields" not in [attr.name for attr in array.schema]
        for dim in array.schema.domain:
            assert all(not isinstance(f, tiledb.FloatScaleFilter) for f in dim.filters)


@pytest.mark.parametrize("kind", ["file", "directory", "array", "symlink"])
def test_existing_destinations_are_never_overwritten(tmp_path, points, native, kind):
    uri = tmp_path / "existing"
    if kind == "file":
        uri.write_text("keep me")
    elif kind == "directory":
        uri.mkdir()
    elif kind == "symlink":
        uri.symlink_to(tmp_path / "missing")
    else:
        io.write_tiledb(points, uri, "EPSG:32605", config_file=native)
    with pytest.raises(FileExistsError):
        io.write_tiledb(points, uri, "EPSG:32605", config_file=native)
    if kind == "file":
        assert uri.read_text() == "keep me"
    elif kind == "array":
        _assert_same_points(_native_read(uri, native)[0], points)


@pytest.mark.parametrize("bad", [None, [], np.empty(0, dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8")]),
                                np.zeros(2), np.zeros((1, 2), dtype=[("X", "f8")]),
                                np.zeros(1, dtype=[("X", "f8"), ("Y", "f8")]),
                                np.zeros(1, dtype=[("X", "f4"), ("Y", "f8"), ("Z", "f8")])])
def test_invalid_point_arrays_rejected_before_creation(tmp_path, config, bad):
    with pytest.raises(ValueError):
        io.write_tiledb(bad, tmp_path / "output", "EPSG:32605", config_file=config)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("dtype", ["bool", "complex128", "object", "float16", ("f8", (2,))])
def test_unsupported_point_fields_rejected(tmp_path, config, dtype):
    points = np.zeros(1, dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("Bad", dtype)])
    with pytest.raises(ValueError, match="numeric scalar"):
        io.write_tiledb(points, tmp_path / "output", "EPSG:32605", config_file=config)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_coordinates_rejected(tmp_path, points, config, value):
    points["Z"][0] = value
    with pytest.raises(ValueError, match="finite"):
        io.write_tiledb(points, tmp_path / "output", "EPSG:32605", config_file=config)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_chunk_size(tmp_path, points, config, value):
    with pytest.raises(ValueError, match="positive integer"):
        io.write_tiledb(points, tmp_path / "output", "EPSG:32605", config_file=config, chunk_size=value)


def test_missing_crs_and_mixed_dtypes_rejected(tmp_path, points, config):
    with pytest.raises(ValueError, match="srs is required"):
        io.write_tiledb(points, tmp_path / "output", None, config_file=config)
    with pytest.raises(ValueError, match="same dtype"):
        io.write_tiledb([points, points[["X", "Y", "Z"]]], tmp_path / "output",
                        "EPSG:32605", config_file=config)


def test_missing_native_plugin_message_and_independent_metadata(tmp_path, config, monkeypatch):
    uri = tmp_path / "empty"
    _create_array(uri, config)
    monkeypatch.delattr(pdal.Reader, "tiledb", raising=False)
    assert io.get_tiledb_info(uri, config_file=config) == {"bounds": None, "srs": None}
    with pytest.raises(RuntimeError, match="libpdal-tiledb"):
        io._tiledb_reader(uri, config_file=config)


@pytest.mark.parametrize("options", [{"names": ("X", "Y")}, {"coordinate_dtype": "f4"},
                                      {"nullable": True}, {"var": True},
                                      {"attribute_dtype": "S1"},
                                      {"sparse": False, "coordinate_dtype": "i4"}])
def test_unsupported_external_schema(tmp_path, config, options):
    uri = tmp_path / "unsupported"
    _create_array(uri, config, **options)
    with pytest.raises(ValueError):
        io.get_tiledb_info(uri, config_file=config)


def test_external_schema_actual_bounds_and_dimension_order(tmp_path, native):
    uri = tmp_path / "external"
    ctx = _create_array(uri, native, names=("Z", "Y", "X"))
    assert io.get_tiledb_info(uri, config_file=native)["bounds"] is None
    with tiledb.open(str(uri), "w", ctx=ctx) as array:
        array[np.array([3.0, 6.0]), np.array([2.0, 5.0]), np.array([1.0, 4.0])] = {"Value": [10.0, 20.0]}
    assert io.get_tiledb_info(uri, config_file=native)["bounds"] == (1, 4, 2, 5, 3, 6)
    reader = io._tiledb_reader(uri, bounds=([3.5, 4], [4.5, 5]), config_file=native)
    pipeline = pdal.Pipeline(json.dumps([reader]))
    assert pipeline.execute() == 1
    assert pipeline.arrays[0]["X"].tolist() == [4.0]
    with tiledb.open(str(uri), "w", ctx=ctx) as array:
        array[np.array([9.0]), np.array([8.0]), np.array([7.0])] = {"Value": [30.0]}
    assert io.get_tiledb_info(uri, config_file=native)["bounds"] == (1, 7, 2, 8, 3, 9)


def test_malformed_metadata_and_missing_paths(tmp_path, config):
    with pytest.raises(FileNotFoundError):
        io.get_tiledb_info(tmp_path / "missing", config_file=config)
    with pytest.raises(FileNotFoundError):
        io.get_tiledb_info(tmp_path / "missing", config_file=tmp_path / "missing.conf")
    uri = tmp_path / "invalid"
    ctx = _create_array(uri, config)
    with tiledb.open(str(uri), "w", ctx=ctx) as array:
        array.meta["_pdal"] = "not JSON"
    with pytest.raises(ValueError, match="_pdal metadata"):
        io.get_tiledb_info(uri, config_file=config)


@pytest.mark.parametrize("with_attribute", [False, True])
@pytest.mark.parametrize("with_origin", [False, True])
def test_empty_interior_query_does_not_fabricate_origin(tmp_path, native, with_attribute, with_origin):
    dtype = [(name, "f8") for name in ("X", "Y", "Z")]
    if with_attribute:
        dtype.append(("Intensity", "u2"))
    points = np.zeros(3 if with_origin else 2, dtype=dtype)
    for name in ("X", "Y", "Z"):
        points[name][:2] = [-1, 1]
    uri = tmp_path / "points"
    io.write_tiledb(points, uri, "EPSG:32605", config_file=native)
    reader = io._tiledb_reader(uri, bounds=([-.5, .5], [-.5, .5], [-.5, .5]), config_file=native)
    if not with_origin:
        assert reader is None
    else:
        pipeline = pdal.Pipeline(json.dumps([reader]))
        assert pipeline.execute() == 1
        assert all(pipeline.arrays[0][name][0] == 0 for name in ("X", "Y", "Z"))
    assert io._tiledb_reader(uri, bounds=([10, 11], [10, 11]), config_file=native) is None


def _write_las(path, points, srs="EPSG:32605"):
    writer = {"type": "writers.las", "filename": str(path), "scale_x": .000001,
              "scale_y": .000001, "scale_z": .000001, "offset_x": "auto",
              "offset_y": "auto", "offset_z": "auto", "extra_dims": "all"}
    if srs:
        writer["a_srs"] = srs
    # LAS imposes fixed standard-dimension limits; use coordinate-only input.
    pipeline = pdal.Pipeline(json.dumps([writer]), arrays=[points[["X", "Y", "Z"]]])
    pipeline.execute()


def test_streaming_las_conversion_uses_no_numpy_materialization(tmp_path, points, native, monkeypatch):
    source, destination = tmp_path / "source.las", tmp_path / "output"
    _write_las(source, points)
    original_pipeline = pdal.Pipeline
    source_reader = original_pipeline(json.dumps([{"type": "readers.las", "filename": str(source)}]))
    source_reader.execute()
    expected = np.concatenate(source_reader.arrays)
    chunks = []

    class StreamingOnly:
        def __init__(self, *args, **kwargs):
            self.pipeline = original_pipeline(*args, **kwargs)

        @property
        def quickinfo(self):
            return self.pipeline.quickinfo

        def execute_streaming(self, *, chunk_size):
            chunks.append(chunk_size)
            return self.pipeline.execute_streaming(chunk_size=chunk_size)

        def execute(self):
            pytest.fail("Conversion must stream, not execute a materialized pipeline")

        @property
        def arrays(self):
            pytest.fail("Conversion must not materialize NumPy arrays")

    monkeypatch.setattr(pdal, "Pipeline", StreamingOnly)
    assert io.convert_to_tiledb(source, destination, srs="EPSG:4326", config_file=native, chunk_size=2) == 4
    assert chunks == [2]
    _assert_same_points(_native_read(destination, native, pipeline_type=original_pipeline)[0], expected)
    assert CRS(io.get_tiledb_info(destination, config_file=native)["srs"]).to_epsg() == 32605


def test_streaming_conversion_bounds_empty_and_missing_crs(tmp_path, points, native):
    source = tmp_path / "source.las"
    _write_las(source, points, srs=None)
    with pytest.raises(ValueError, match="no stored CRS"):
        io.convert_to_tiledb(source, tmp_path / "no-crs", config_file=native)
    assert not (tmp_path / "no-crs").exists()
    destination = tmp_path / "selected"
    bounds = ([400000, 400001], [2200000, 2200001])
    assert io.convert_to_tiledb(source, destination, bounds=bounds, srs="EPSG:32605", config_file=native) == 2
    assert CRS(io.get_tiledb_info(destination, config_file=native)["srs"]).to_epsg() == 32605
    empty = tmp_path / "empty"
    assert io.convert_to_tiledb(source, empty, bounds=([0, 1], [0, 1]),
                               srs="EPSG:32605", config_file=native) == 0
    assert io.get_tiledb_info(empty, config_file=native)["bounds"] is None
    assert io.convert_to_tiledb(empty, tmp_path / "empty-copy", source_type="tiledb", config_file=native) == 0
    assert not (tmp_path / "empty-copy").exists()


def test_streaming_tiledb_conversion_preserves_duplicates(tmp_path, points, native):
    source, destination = tmp_path / "source", tmp_path / "copy"
    io.write_tiledb(points, source, "EPSG:32605", config_file=native)
    assert io.convert_to_tiledb(source, destination, source_type="tiledb", config_file=native, chunk_size=2) == 4
    _assert_same_points(_native_read(destination, native)[0], points)


@pytest.mark.parametrize("kind", ["copc", "ept"])
@pytest.mark.parametrize("subset", [False, True])
def test_streaming_indexed_sources(tmp_path, points, native, kind, subset):
    if kind == "copc":
        source = tmp_path / "source.copc.laz"
        writer = {"type": "writers.copc", "filename": str(source), "a_srs": "EPSG:32605",
                  "scale_x": .000001, "scale_y": .000001, "scale_z": .000001,
                  "offset_x": "auto", "offset_y": "auto", "offset_z": "auto"}
        pdal.Pipeline(json.dumps([writer]), arrays=[points[["X", "Y", "Z"]]]).execute()
    else:
        source = tmp_path / "source" / "ept.json"
        (source.parent / "ept-data").mkdir(parents=True)
        (source.parent / "ept-hierarchy").mkdir()
        # A complete small EPT root node, with no remote data or Entwine binary.
        ept_points = np.empty(len(points), dtype=[(name, "<f8") for name in ("X", "Y", "Z")])
        for name in ept_points.dtype.names:
            ept_points[name] = points[name]
        (source.parent / "ept-data" / "0-0-0-0.bin").write_bytes(ept_points.tobytes())
        (source.parent / "ept-hierarchy" / "0-0-0-0.json").write_text(json.dumps({"0-0-0-0": len(points)}))
        bounds = [float(points[name].min()) for name in ("X", "Y", "Z")]
        bounds += [float(points[name].max()) for name in ("X", "Y", "Z")]
        source.write_text(json.dumps({"version": "1.0.0", "dataType": "binary", "hierarchyType": "json",
                                      "bounds": bounds, "boundsConforming": bounds,
                                      "points": len(points), "span": 128,
                                      "srs": {"wkt": CRS(32605).to_wkt()},
                                      "schema": [{"name": name, "type": "floating", "size": 8}
                                                 for name in ("X", "Y", "Z")]}))
    reader = pdal.Pipeline(json.dumps([{"type": f"readers.{kind}", "filename": str(source)}]))
    reader.execute()
    expected = np.concatenate(reader.arrays)
    selection = ([400000, 400001], [2200000, 2200001]) if subset else None
    if subset:
        expected = expected[(expected["X"] <= 400001) & (expected["Y"] <= 2200001)]
    destination = tmp_path / "converted"
    assert io.convert_to_tiledb(source, destination, source_type=kind, bounds=selection,
                               config_file=native, chunk_size=2) == len(expected)
    _assert_same_points(_native_read(destination, native)[0], expected)
    assert CRS(io.get_tiledb_info(destination, config_file=native)["srs"]).to_epsg() == 32605
