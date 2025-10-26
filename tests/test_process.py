import pytest
import numpy as np
from unittest.mock import patch, MagicMock


from pyforestscan.process import process_with_tiles


# todo: look into why this test takes so long...
@pytest.mark.skip(reason="Takes too long to run. Run manually if needed.")
def test_process_with_tiles_chm_small_hawaii(tmp_path):
    """
    Integration test that runs process_with_tiles on a very small bounding region
    of the Hawaii EPT dataset. Ensures it creates a CHM GeoTIFF in tmp_path.
    """
    ept_file = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/HI_Hawaii_Island_2017/ept.json"
    bounds = ([-17348441.871880997,-17347398.335829224], [2245235.283966082,2246320.888103429])

    tile_size = (200, 200)
    voxel_size = (1, 1, 1)
    metric = "chm"
    interpolation = None

    output_dir = tmp_path / "chm_test_tiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    process_with_tiles(
        ept_file=ept_file,
        tile_size=tile_size,
        output_path=str(output_dir),
        metric=metric,
        voxel_size=voxel_size,
        buffer_size=0.1,
        srs=None,
        hag=True,
        hag_dtm=False,
        dtm=None,
        bounds=bounds,
        interpolation=interpolation
    )

    tifs = list(output_dir.glob("tile_*_chm.tif"))
    assert len(tifs) >= 1, "No CHM tiles were produced. Possibly empty area or PDAL config issue."


@patch("pyforestscan.process.pdal.Pipeline")
def test_process_with_tiles_no_data(mock_pipeline_cls, tmp_path):
    """
    If the pipeline returns no points, ensure we skip tile creation gracefully.
    """
    mock_pipeline = MagicMock()
    mock_pipeline.execute.return_value = True
    mock_pipeline.arrays = [np.array([], dtype=[('X', 'f8'), ('Y', 'f8'), ('Z', 'f8')])]
    mock_pipeline_cls.return_value = mock_pipeline

    out_dir = tmp_path / "test_fhd"
    out_dir.mkdir()

    process_with_tiles(
        ept_file="fake_ept_path",
        tile_size=(50, 50),
        output_path=str(out_dir),
        metric="fhd",
        voxel_size=(1, 1, 1),
        buffer_size=0.0,
        srs="EPSG:32610",
        hag=False,
        hag_dtm=False,
        dtm=None,
        bounds=([0, 50], [0, 50], [0, 50])  # bounding box
    )

    created_tifs = list(out_dir.glob("*.tif"))
    assert len(created_tifs) == 0, "No data => should produce no TIF output."


@patch("pyforestscan.process.pdal.Pipeline")
def test_process_with_tiles_no_data_with_thinning(mock_pipeline_cls, tmp_path):
    """
    If the pipeline returns no points and thinning is requested, ensure we still skip gracefully.
    """
    mock_pipeline = MagicMock()
    mock_pipeline.execute.return_value = True
    mock_pipeline.arrays = [np.array([], dtype=[('X', 'f8'), ('Y', 'f8'), ('Z', 'f8')])]
    mock_pipeline_cls.return_value = mock_pipeline

    out_dir = tmp_path / "test_fhd_thin"
    out_dir.mkdir()

    process_with_tiles(
        ept_file="fake_ept_path",
        tile_size=(50, 50),
        output_path=str(out_dir),
        metric="fhd",
        voxel_size=(1, 1, 1),
        buffer_size=0.0,
        srs="EPSG:32610",
        hag=False,
        hag_dtm=False,
        dtm=None,
        bounds=([0, 50], [0, 50], [0, 50]),
        thin_radius=1.0
    )

    created_tifs = list(out_dir.glob("*.tif"))
    assert len(created_tifs) == 0, "No data => should produce no TIF output (even with thinning)."


@patch("pyforestscan.process.downsample_poisson")
@patch("pyforestscan.process.pdal.Pipeline")
def test_process_with_tiles_thin_radius_applied(mock_pipeline_cls, mock_downsample, tmp_path):
    """
    When thin_radius is provided, ensure downsample_poisson is called and output is produced.
    """
    # Mock PDAL pipeline to return a small synthetic point set with required fields
    dtype = [("X", "f8"), ("Y", "f8"), ("HeightAboveGround", "f8")]
    pts = np.zeros(100, dtype=dtype)
    pts["X"] = np.random.uniform(0, 10, size=100)
    pts["Y"] = np.random.uniform(0, 10, size=100)
    pts["HeightAboveGround"] = np.random.uniform(0, 5, size=100)

    mock_pipeline = MagicMock()
    mock_pipeline.execute.return_value = True
    mock_pipeline.arrays = [pts]
    mock_pipeline_cls.return_value = mock_pipeline

    # Mock downsample to return half the points, and track calls
    def _fake_downsample(arrays, thin_radius):
        arr = arrays[0]
        return [arr[::2]]

    mock_downsample.side_effect = _fake_downsample

    out_dir = tmp_path / "test_pai_thin"
    out_dir.mkdir()

    process_with_tiles(
        ept_file="fake_ept_path",
        tile_size=(20, 20),
        output_path=str(out_dir),
        metric="pai",
        voxel_size=(2, 2, 1),
        voxel_height=1.0,
        buffer_size=0.0,
        srs="EPSG:32610",
        hag=False,
        hag_dtm=False,
        dtm=None,
        bounds=([0, 20], [0, 20], [0, 10]),
        thin_radius=1.0
    )

    # Should have called thinning at least once
    assert mock_downsample.called, "downsample_poisson should be called when thin_radius is set"

    # And produced at least one PAI output
    created_tifs = list(out_dir.glob("tile_*_pai.tif"))
    assert len(created_tifs) >= 1, "Expected at least one output tile for PAI."


@patch("pyforestscan.process.downsample_voxel")
@patch("pyforestscan.process.pdal.Pipeline")
def test_process_with_tiles_voxelgrid_applied(mock_pipeline_cls, mock_downsample_voxel, tmp_path):
    """
    When voxelgrid_cell is provided, ensure downsample_voxel is called and output is produced.
    """
    dtype = [("X", "f8"), ("Y", "f8"), ("HeightAboveGround", "f8")]
    pts = np.zeros(80, dtype=dtype)
    pts["X"] = np.random.uniform(0, 10, size=80)
    pts["Y"] = np.random.uniform(0, 10, size=80)
    pts["HeightAboveGround"] = np.random.uniform(0, 5, size=80)

    mock_pipeline = MagicMock()
    mock_pipeline.execute.return_value = True
    mock_pipeline.arrays = [pts]
    mock_pipeline_cls.return_value = mock_pipeline

    def _fake_voxel(arrays, cell, mode):
        arr = arrays[0]
        # Keep every 3rd point to mimic downsampling
        return [arr[::3]]

    mock_downsample_voxel.side_effect = _fake_voxel

    out_dir = tmp_path / "test_pai_voxelgrid"
    out_dir.mkdir()

    process_with_tiles(
        ept_file="fake_ept_path",
        tile_size=(20, 20),
        output_path=str(out_dir),
        metric="pai",
        voxel_size=(2, 2, 1),
        voxel_height=1.0,
        buffer_size=0.0,
        srs="EPSG:32610",
        hag=False,
        hag_dtm=False,
        dtm=None,
        bounds=([0, 20], [0, 20], [0, 10]),
        voxelgrid_cell=1.5,
        voxelgrid_mode="first",
    )

    assert mock_downsample_voxel.called, "downsample_voxel should be called when voxelgrid_cell is set"
    created_tifs = list(out_dir.glob("tile_*_pai.tif"))
    assert len(created_tifs) >= 1, "Expected at least one output tile for PAI with voxel-grid downsampling."


@patch("pyforestscan.process.downsample_voxel")
@patch("pyforestscan.process.pdal.Pipeline")
def test_process_with_tiles_voxelgrid_empty_skips(mock_pipeline_cls, mock_downsample_voxel, tmp_path):
    """
    If voxel-grid thinning removes all points, the tile should be skipped gracefully.
    """
    dtype = [("X", "f8"), ("Y", "f8"), ("HeightAboveGround", "f8")]
    pts = np.zeros(30, dtype=dtype)
    pts["X"] = np.random.uniform(0, 10, size=30)
    pts["Y"] = np.random.uniform(0, 10, size=30)
    pts["HeightAboveGround"] = np.random.uniform(0, 1, size=30)

    mock_pipeline = MagicMock()
    mock_pipeline.execute.return_value = True
    mock_pipeline.arrays = [pts]
    mock_pipeline_cls.return_value = mock_pipeline

    def _empty_voxel(arrays, cell, mode):
        # Simulate all points removed by voxel downsampling
        empty = np.array([], dtype=dtype)
        return [empty]

    mock_downsample_voxel.side_effect = _empty_voxel

    out_dir = tmp_path / "test_voxelgrid_empty"
    out_dir.mkdir()

    process_with_tiles(
        ept_file="fake_ept_path",
        tile_size=(20, 20),
        output_path=str(out_dir),
        metric="fhd",
        voxel_size=(2, 2, 1),
        buffer_size=0.0,
        srs="EPSG:32610",
        hag=False,
        hag_dtm=False,
        dtm=None,
        bounds=([0, 20], [0, 20], [0, 10]),
        voxelgrid_cell=1.0,
        voxelgrid_mode="first",
        verbose=True,
    )

    created_tifs = list(out_dir.glob("*.tif"))
    assert len(created_tifs) == 0, "No output should be produced when voxel-grid thinning empties the tile."

@patch("pyforestscan.process.pdal.Pipeline")
def test_process_with_tiles_pai_handles_low_top_height(mock_pipeline_cls, tmp_path):
    """
    When the available height is below the default PAI min_height (1 m),
    the process should not raise and should produce a tile (zeros allowed).
    """
    dtype = [("X", "f8"), ("Y", "f8"), ("HeightAboveGround", "f8")]
    pts = np.zeros(50, dtype=dtype)
    pts["X"] = np.random.uniform(0, 10, size=50)
    pts["Y"] = np.random.uniform(0, 10, size=50)
    # HAG strictly below 1 m to force a single Z layer when dz=1
    pts["HeightAboveGround"] = np.random.uniform(0, 0.5, size=50)

    mock_pipeline = MagicMock()
    mock_pipeline.execute.return_value = True
    mock_pipeline.arrays = [pts]
    mock_pipeline_cls.return_value = mock_pipeline

    out_dir = tmp_path / "test_pai_lowtop"
    out_dir.mkdir()

    process_with_tiles(
        ept_file="fake_ept_path",
        tile_size=(20, 20),
        output_path=str(out_dir),
        metric="pai",
        voxel_size=(2, 2, 1),  # dz=1
        voxel_height=1.0,
        buffer_size=0.0,
        srs="EPSG:32610",
        hag=False,
        hag_dtm=False,
        dtm=None,
        bounds=([0, 20], [0, 20], [0, 10])
    )

    created_tifs = list(out_dir.glob("tile_*_pai.tif"))
    assert len(created_tifs) >= 1, "Expected a PAI output tile even when top height < 1 m."


@patch("pyforestscan.process.downsample_poisson")
@patch("pyforestscan.process.pdal.Pipeline")
def test_process_with_tiles_by_flightline_sampling_applied(mock_pipeline_cls, mock_downsample, tmp_path):
    """
    With by_flightline=True, Poisson sampling should be applied per PointSourceId group.
    """
    dtype = [("X", "f8"), ("Y", "f8"), ("HeightAboveGround", "f8"), ("PointSourceId", "i4")]
    pts = np.zeros(120, dtype=dtype)
    pts["X"] = np.random.uniform(0, 20, size=120)
    pts["Y"] = np.random.uniform(0, 20, size=120)
    pts["HeightAboveGround"] = np.random.uniform(0, 5, size=120)
    # Two flightlines
    pts["PointSourceId"][:60] = 10
    pts["PointSourceId"][60:] = 11

    mock_pipeline = MagicMock()
    mock_pipeline.execute.return_value = True
    mock_pipeline.arrays = [pts]
    mock_pipeline_cls.return_value = mock_pipeline

    call_counts = {10: 0, 11: 0}

    def _per_group(arrays, thin_radius):
        arr = arrays[0]
        # Track which group is being thinned
        psid = int(arr["PointSourceId"][0])
        call_counts[psid] += 1
        return [arr[::2]]

    mock_downsample.side_effect = _per_group

    out_dir = tmp_path / "test_by_flightline"
    out_dir.mkdir()

    process_with_tiles(
        ept_file="fake_ept_path",
        tile_size=(20, 20),
        output_path=str(out_dir),
        metric="fhd",
        voxel_size=(2, 2, 1),
        buffer_size=0.0,
        srs="EPSG:32610",
        bounds=([0, 20], [0, 20], [0, 10]),
        by_flightline=True,
        thin_radius=0.5,
    )

    # One downsample call per PSID
    assert call_counts[10] == 1 and call_counts[11] == 1
    created_tifs = list(out_dir.glob("tile_*_fhd_psid*.tif"))
    assert len(created_tifs) >= 1, "Expected an FHD output tile when processing by flightline."


@patch("pyforestscan.process.pdal.Pipeline")
def test_process_with_tiles_by_flightline_missing_dim(mock_pipeline_cls, tmp_path):
    """
    If PointSourceId is missing and by_flightline=True, raise a clear error.
    """
    dtype = [("X", "f8"), ("Y", "f8"), ("HeightAboveGround", "f8")]
    pts = np.zeros(40, dtype=dtype)
    pts["X"] = np.random.uniform(0, 10, size=40)
    pts["Y"] = np.random.uniform(0, 10, size=40)
    pts["HeightAboveGround"] = np.random.uniform(0, 3, size=40)

    mock_pipeline = MagicMock()
    mock_pipeline.execute.return_value = True
    mock_pipeline.arrays = [pts]
    mock_pipeline_cls.return_value = mock_pipeline

    out_dir = tmp_path / "test_by_flightline_err"
    out_dir.mkdir()

    with pytest.raises(ValueError):
        process_with_tiles(
            ept_file="fake_ept_path",
            tile_size=(20, 20),
            output_path=str(out_dir),
            metric="pai",
            voxel_size=(2, 2, 1),
            voxel_height=1.0,
            buffer_size=0.0,
            srs="EPSG:32610",
            bounds=([0, 20], [0, 20], [0, 10]),
            by_flightline=True,
            thin_radius=0.3,
        )
