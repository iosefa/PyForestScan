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
