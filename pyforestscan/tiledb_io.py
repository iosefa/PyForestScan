"""Point-cloud IO for PDAL-compatible TileDB sparse arrays."""

import json
import os
from urllib.parse import unquote, urlparse

import numpy as np
import pdal
import tiledb
from pyproj import CRS


def _uri(value):
    value = os.fspath(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A nonempty TileDB array path or URI is required")
    if urlparse(value).scheme == "tiledb":
        raise ValueError("Use a local path or object-store URI; TileDB Cloud registration is not supported")
    return value


def _context(config_file=None):
    if config_file is None:
        return tiledb.Ctx()
    config_file = os.fspath(config_file)
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"No TileDB configuration file: '{config_file}'")
    # TileDB-Py 0.36.1's Config(path=...) constructor discards the loaded
    # configuration. Ctx accepts the explicit loader's native Config result.
    return tiledb.Ctx(tiledb.Config.load(config_file))


def _require_tiledb_driver(stage):
    """Fail clearly when Python PDAL cannot load its native TileDB plugin."""
    kind, _, name = stage.partition(".")
    factory = {"readers": pdal.Reader, "writers": pdal.Writer}.get(kind)
    if factory is None or not hasattr(factory, name):
        raise RuntimeError(
            f"PDAL stage '{stage}' is unavailable. Install the native TileDB "
            "plugin in the same environment as PDAL, for example with "
            "'conda install -c conda-forge libpdal-tiledb', then restart Python. "
            "Installing the Python tiledb package alone does not install this plugin."
        )


def _numeric_dtype(dtype):
    dtype = np.dtype(dtype)
    return (dtype.subdtype is None and dtype.fields is None and dtype.isnative
            and ((dtype.kind in "iu" and dtype.itemsize in (1, 2, 4, 8))
                 or (dtype.kind == "f" and dtype.itemsize in (4, 8))))


def _validate_schema(schema):
    if not schema.sparse:
        raise ValueError("TileDB point clouds must use a sparse array")
    dimensions = list(schema.domain)
    if len(dimensions) != 3 or {dim.name for dim in dimensions} != {"X", "Y", "Z"}:
        raise ValueError("TileDB point clouds must have exactly the X, Y, and Z dimensions")
    if any(dim.dtype != np.dtype("float64") for dim in dimensions):
        raise ValueError("TileDB X, Y, and Z dimensions must use float64 coordinates")
    for attr in schema:
        if (not _numeric_dtype(attr.dtype) or attr.isvar or attr.isnullable
                or attr.ncells != 1):
            raise ValueError(
                f"TileDB attribute '{attr.name}' must be a non-nullable, fixed-size numeric scalar"
            )
        if attr.name == "BitFields" and attr.dtype != np.dtype("uint16"):
            raise ValueError("The PDAL BitFields attribute must have dtype uint16")


def _read_pdal_metadata(array):
    if "_pdal" not in array.meta:
        return {}
    value = array.meta["_pdal"]
    try:
        if not isinstance(value, str):
            value = bytes(value).decode("utf-8")
        metadata = json.loads(value.rstrip("\x00"))
        if not isinstance(metadata, dict):
            raise ValueError("expected a JSON object")
        root = metadata.get("root", {})
        if not isinstance(root, dict) or not isinstance(root.get("writers.tiledb", {}), dict):
            raise ValueError("expected PDAL metadata objects")
        return metadata
    except (ValueError, TypeError, UnicodeError) as error:
        raise ValueError("Invalid TileDB _pdal metadata") from error


def get_tiledb_info(uri, *, config_file=None) -> dict:
    """Inspect a point-cloud array without reading its points.

    Args:
        uri (str or os.PathLike): Local array directory or object-store URI.
        config_file (str or os.PathLike or None): TileDB configuration file,
            shared with native PDAL operations when supplied.

    Returns:
        dict: ``bounds`` is (xmin, xmax, ymin, ymax, zmin, zmax), or None for
            an empty array. These are actual point bounds, not schema limits.
            ``srs`` is the stored CRS as WKT, or None when it is absent.

    Raises:
        FileNotFoundError: If the array or configuration file does not exist.
        ValueError: If the schema is not a supported sparse XYZ point cloud,
            or the PDAL metadata is malformed.
    """
    uri = _uri(uri)
    ctx = _context(config_file)
    if tiledb.object_type(uri, ctx=ctx) != "array":
        raise FileNotFoundError(f"No TileDB array found at '{uri}'")
    with tiledb.open(uri, "r", ctx=ctx) as array:
        _validate_schema(array.schema)
        domain = array.nonempty_domain()
        bounds = None
        if domain is not None:
            by_name = {array.schema.domain.dim(i).name: pair for i, pair in enumerate(domain)}
            bounds = tuple(float(value) for name in ("X", "Y", "Z") for value in by_name[name])
        metadata = _read_pdal_metadata(array)
    stored_srs = metadata.get("root", {}).get("writers.tiledb", {}).get("spatialreference")
    srs = CRS.from_user_input(stored_srs).to_wkt() if stored_srs else None
    return {"bounds": bounds, "srs": srs}


def _bounds_pairs(bounds):
    pairs = np.asarray(bounds, dtype=float)
    if (pairs.shape not in ((2, 2), (3, 2)) or not np.all(np.isfinite(pairs))
            or np.any(pairs[:, 0] > pairs[:, 1])):
        raise ValueError("bounds must contain finite ordered [min, max] pairs for XY or XYZ")
    return pairs


def _tiledb_reader(uri, bounds=None, config_file=None, info=None):
    """Build a native spatial query, or return None for an empty intersection."""
    _require_tiledb_driver("readers.tiledb")
    uri = _uri(uri)
    requested = _bounds_pairs(bounds) if bounds is not None else None
    if info is None:
        info = get_tiledb_info(uri, config_file=config_file)
    if info["bounds"] is None:
        return None
    extent = np.asarray(info["bounds"], dtype=float).reshape(3, 2)
    query = extent.copy()
    if requested is not None:
        query[:len(requested), 0] = np.maximum(query[:len(requested), 0], requested[:, 0])
        query[:len(requested), 1] = np.minimum(query[:len(requested), 1], requested[:, 1])
        if np.any(query[:, 0] > query[:, 1]):
            return None
        if not np.array_equal(query, extent) and not _query_has_points(uri, query, config_file):
            return None
    reader = {"type": "readers.tiledb", "filename": uri,
              "bbox3d": "(" + ", ".join(str(pair) for pair in query.tolist()) + ")"}
    if config_file is not None:
        reader["config_file"] = os.fspath(config_file)
    if info.get("srs"):
        reader["default_srs"] = info["srs"]
    return reader


def _query_has_points(uri, bounds, config_file):
    # PDAL 2.10's TileDB reader can emit a fabricated zero point for an empty
    # spatial query. Check occupancy independently, preserving real zero points.
    with tiledb.open(uri, "r", ctx=_context(config_file)) as array:
        by_name = dict(zip(("X", "Y", "Z"), bounds))
        selection = tuple(slice(*by_name[dim.name]) for dim in array.schema.domain)
        if array.schema.nattr:
            attribute = array.schema.attr(0).name
            # multi_index interprets negative values as coordinates, whereas
            # ordinary indexing interprets them relative to the domain end.
            return bool(array.query().agg({attribute: "count"}).multi_index[selection])
        # Coordinate-only arrays have no attribute to aggregate. Read just
        # the first bounded query chunk, rather than materializing the cloud.
        chunks = array.query(attrs=[], dims=["X"], return_incomplete=True).multi_index[selection]
        first = next(iter(chunks), {})
        return bool(first.get("X", np.empty(0)).size)


def _new_destination(uri, config_file):
    uri = _uri(uri)
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        local_path = unquote(parsed.path) if parsed.scheme else uri
        if os.path.lexists(local_path):
            raise FileExistsError(f"Destination already exists: '{uri}'")
    ctx = _context(config_file)
    vfs = tiledb.VFS(ctx=ctx)
    if (tiledb.object_type(uri, ctx=ctx) is not None
            or vfs.is_dir(uri) or vfs.is_file(uri)):
        raise FileExistsError(f"Destination already exists: '{uri}'")
    return uri, ctx


def _chunk_size(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError("chunk_size must be a positive integer")
    return int(value)


def _writer_stage(uri, config_file, chunk_size, bounds=None):
    writer = {"type": "writers.tiledb", "filename": uri, "append": False,
              "filter_profile": "none", "compression": "zstd", "allow_dups": True,
              "combine_bit_fields": False, "chunk_size": chunk_size}
    if config_file is not None:
        writer["config_file"] = os.fspath(config_file)
    if bounds is not None:
        for name, (low, high) in zip(("x", "y", "z"), np.asarray(bounds).reshape(3, 2)):
            # Native PDAL option parsing can round domain limits to fewer
            # significant digits. Pad outward so exact endpoint coordinates
            # remain inside the schema; nonempty_domain still reports them.
            padding = max(1.0, abs(float(low)), abs(float(high))) * 1e-12
            if low == high:
                padding = max(1.0, padding)
            low, high = low - padding, high + padding
            if np.isfinite(low) and np.isfinite(high) and low < high:
                writer[f"{name}_domain_st"] = float(low)
                writer[f"{name}_domain_end"] = float(high)
    return writer


def _store_srs(uri, srs, ctx):
    # The native writer has no CRS-assignment option for NumPy inputs. Update
    # its standard PDAL metadata without passing points through a transform.
    with tiledb.open(uri, "r", ctx=ctx) as array:
        metadata = _read_pdal_metadata(array)
    metadata.setdefault("root", {}).setdefault("writers.tiledb", {})["spatialreference"] = srs
    encoded = (json.dumps(metadata) + "\x00").encode("utf-8")
    with tiledb.open(uri, "w", ctx=ctx) as array:
        array.meta["_pdal"] = np.frombuffer(encoded, dtype=np.uint8)


def _point_arrays(arrays):
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]
    elif not isinstance(arrays, (list, tuple)):
        raise ValueError("Supply a structured NumPy array or a list of point arrays")
    if not arrays:
        raise ValueError("At least one nonempty point array is required")
    dtype = None
    for array in arrays:
        if not isinstance(array, np.ndarray) or array.ndim != 1 or not array.dtype.names:
            raise ValueError("Point arrays must be one-dimensional structured NumPy arrays")
        if array.size == 0:
            raise ValueError("Point arrays must be nonempty")
        if not {"X", "Y", "Z"}.issubset(array.dtype.names):
            raise ValueError("Point arrays require X, Y, and Z fields")
        if dtype is not None and array.dtype != dtype:
            raise ValueError("All point arrays must have the same dtype")
        dtype = array.dtype
        for name in dtype.names:
            field_dtype = dtype.fields[name][0]
            if not _numeric_dtype(field_dtype):
                raise ValueError(f"Point field '{name}' must be a native-endian numeric scalar")
        for name in ("X", "Y", "Z"):
            if dtype.fields[name][0] != np.dtype("float64"):
                raise ValueError("X, Y, and Z point fields must use float64 coordinates")
            if not np.all(np.isfinite(array[name])):
                raise ValueError("X, Y, and Z coordinates must be finite")
        if "BitFields" in dtype.names:
            raise ValueError("Use unpacked PDAL fields instead of a packed BitFields input field")
    return arrays


def write_tiledb(arrays, output_uri, srs, *, config_file=None, chunk_size=1_000_000) -> None:
    """Write loaded point arrays to a new TileDB point-cloud array.

    XYZ must be finite float64 fields; other fields must be numeric scalars.
    Coordinates are stored losslessly, including coincident points. Input
    arrays are not modified. Read order may differ from input order.

    Args:
        arrays (np.ndarray or list[np.ndarray]): One structured point array,
            or nonempty arrays with identical dtypes.
        output_uri (str or os.PathLike): New local directory or object-store URI.
        srs (str): CRS of the supplied coordinates. This assigns metadata;
            it does not reproject the points.
        config_file (str or os.PathLike or None): TileDB configuration file.
        chunk_size (int): Maximum points buffered by the native writer.

    Raises:
        FileExistsError: If the destination exists. Appending and overwriting
            are deliberately not supported by this function.
        ValueError: If points, CRS, or chunk size are invalid.
        RuntimeError: If the native PDAL TileDB writer is unavailable.
    """
    arrays = _point_arrays(arrays)
    chunk_size = _chunk_size(chunk_size)
    if srs is None:
        raise ValueError("srs is required when writing point arrays")
    srs = CRS.from_user_input(srs).to_wkt()
    _require_tiledb_driver("writers.tiledb")
    uri, ctx = _new_destination(output_uri, config_file)
    bounds = tuple(value for name in ("X", "Y", "Z") for value in
                   (min(array[name].min() for array in arrays),
                    max(array[name].max() for array in arrays)))
    writer = _writer_stage(uri, config_file, chunk_size, bounds)
    pipeline = pdal.Pipeline(json.dumps({"pipeline": [writer]}), arrays=arrays)
    pipeline.execute()
    _store_srs(uri, srs, ctx)


def convert_to_tiledb(input_file, output_uri, *, source_type=None, srs=None,
                      bounds=None, config_file=None, chunk_size=1_000_000) -> int:
    """Stream a LAS/LAZ, COPC, EPT, or TileDB point cloud to a new array.

    The conversion keeps the source coordinates and CRS. ``srs`` supplies a
    fallback only when the source lacks a CRS. Bounds use the source CRS.
    No complete point cloud is materialized as a NumPy array.

    Args:
        input_file (str or os.PathLike): Input point-cloud path or URI.
        output_uri (str or os.PathLike): New TileDB destination.
        source_type (str or None): Explicit "las", "copc", "ept", or
            "tiledb". Existing file extensions can be inferred.
        srs (str or None): Fallback source CRS; this does not reproject data.
        bounds (tuple or None): Ordered XY or XYZ [min, max] pairs.
        config_file (str or os.PathLike or None): TileDB configuration file
            passed to both Python and native TileDB operations.
        chunk_size (int): Streaming batch and writer buffer size.

    Returns:
        int: Number of converted points. An empty TileDB source or a query
            outside its data extent returns zero without creating an array.

    Raises:
        FileExistsError: If the destination exists.
        ValueError: If the source, bounds, or CRS are unsupported or invalid.
        RuntimeError: If a required native TileDB driver is unavailable.
    """
    # Local import avoids a cycle with handlers' TileDB reader integration.
    from pyforestscan.handlers import _reader_stage, _resolve_source_type

    chunk_size = _chunk_size(chunk_size)
    pairs = _bounds_pairs(bounds) if bounds is not None else None
    fallback_srs = CRS.from_user_input(srs).to_wkt() if srs is not None else None
    _require_tiledb_driver("writers.tiledb")
    uri, ctx = _new_destination(output_uri, config_file)
    kind = _resolve_source_type(input_file, source_type=source_type)
    reader = _reader_stage(input_file, source_type=kind, bounds=bounds,
                           srs=None, tiledb_config=config_file)
    if reader is None:
        return 0
    if fallback_srs is not None:
        reader.setdefault("default_srs", fallback_srs)

    if kind == "tiledb":
        info = get_tiledb_info(input_file, config_file=config_file)
        source_srs, source_bounds = info["srs"], info["bounds"]
    else:
        preview = pdal.Pipeline(json.dumps({"pipeline": [reader]})).quickinfo
        source = preview.get(reader["type"], {})
        reference = source.get("srs", {})
        source_srs = reference.get("compoundwkt") or reference.get("wkt")
        box = source.get("bounds", {})
        source_bounds = tuple(box.get(key) for key in
                              ("minx", "maxx", "miny", "maxy", "minz", "maxz"))
        if any(value is None for value in source_bounds):
            source_bounds = None
        else:
            # quickinfo serializes approximate bounds with fewer digits than
            # the points. These limits only size the storage domain; leave a
            # margin so rounding cannot exclude valid endpoint coordinates.
            approximate = np.asarray(source_bounds, dtype=float).reshape(3, 2)
            margin = np.maximum(1.0, np.max(np.abs(approximate), axis=1) * 1e-8)
            approximate[:, 0] -= margin
            approximate[:, 1] += margin
            source_bounds = approximate.ravel()
    source_srs = source_srs or fallback_srs
    if not source_srs:
        raise ValueError("The input has no stored CRS; provide srs to identify its coordinates")
    source_srs = CRS.from_user_input(source_srs).to_wkt()
    stages = [reader]
    if kind == "las" and pairs is not None:
        stages.append({"type": "filters.crop",
                       "bounds": "(" + ", ".join(str(pair) for pair in pairs.tolist()) + ")"})
    stages.append(_writer_stage(uri, config_file, chunk_size, source_bounds))
    pipeline = pdal.Pipeline(json.dumps({"pipeline": stages}))
    count = pipeline.execute_streaming(chunk_size=chunk_size)
    _store_srs(uri, source_srs, ctx)
    return int(count)
