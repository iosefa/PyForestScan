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
