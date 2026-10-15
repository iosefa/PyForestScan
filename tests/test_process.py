import pytest
import numpy as np
import ast
import json
import rasterio
from unittest.mock import patch, MagicMock


from pyforestscan.process import process_with_tiles
from pyforestscan.calculate import calculate_rumple_grid, calculate_voxel_stat


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


@pytest.fixture
def rumple_ept(monkeypatch):
    """Simulate EPT bounds reads; calculate and write real rumple GeoTIFFs."""
    x, y = np.meshgrid(np.arange(11) * 2.0 + 1.0,
                       np.arange(9) * 3.0 + 1.5, indexing='ij')
    points = np.zeros(x.size, dtype=[('X', 'f8'), ('Y', 'f8'), ('HeightAboveGround', 'f8')])
    points['X'], points['Y'] = x.ravel(), y.ravel()
    points['HeightAboveGround'] = 10.0 + 0.03 * points['X'] ** 2 + 0.01 * points['Y'] ** 2
    # A gap and a peak exercise spatial variation, orientation, and NoData.
    points['HeightAboveGround'][12] = np.nan
    points['HeightAboveGround'][49] += 8.0
    # Returns exactly on read boundaries must be assigned consistently.
    edge_points = np.array([(10.0, 12.0, 25.0), (8.0, 15.0, 30.0)], dtype=points.dtype)
    points = np.concatenate([points, edge_points])
    reads = []

    def pipeline(pipeline_json):
        read = json.loads(pipeline_json)['pipeline'][0]
        bounds = ast.literal_eval(read['bounds'])
        reads.append(bounds)
        (xmin, xmax), (ymin, ymax) = bounds[:2]
        keep = ((points['X'] >= xmin) & (points['X'] <= xmax) &
                (points['Y'] >= ymin) & (points['Y'] <= ymax))
        result = MagicMock()
        result.arrays = [points[keep]]
        return result

    monkeypatch.setattr('pyforestscan.process.pdal.Pipeline', pipeline)
    return points, reads


def _whole_rumple(points, method="two_triangle", min_height=None, interpolation=None):
    from scipy.interpolate import griddata

    valid = np.isfinite(points['HeightAboveGround']) & (points['HeightAboveGround'] >= 0)
    chm, extent = calculate_voxel_stat(points[valid], (2, 3, 1), 'HeightAboveGround', 'max')
    if interpolation is not None:
        observed = np.isfinite(chm)
        chm[~observed] = griddata(np.argwhere(observed) * (2, 3), chm[observed],
                                 np.argwhere(~observed) * (2, 3), method=interpolation)
    return calculate_rumple_grid(chm, (2, 3), (4, 6), extent=extent,
                                 min_height=min_height, method=method)


def _assert_rumple_mosaic(directory, expected, extent):
    mosaic = np.full(expected.shape, np.nan)
    covered = np.zeros(expected.shape, dtype=int)
    for path in sorted(directory.glob('tile_*_rumple.tif')):
        with rasterio.open(path) as src:
            assert src.crs.to_epsg() == 32605
            assert src.res == (4.0, 6.0)
            assert src.nodata == -9999
            assert src.count == 1
            values = src.read(1, masked=True).filled(np.nan).T
            x0 = int(round((src.bounds.left - extent[0]) / 4))
            y0 = int(round((extent[3] - src.bounds.top) / 6))
            xs = slice(x0, x0 + src.width)
            ys = slice(y0, y0 + src.height)
            np.testing.assert_allclose(values, expected[xs, ys], equal_nan=True, atol=1e-6)
            mosaic[xs, ys] = values
            covered[xs, ys] += 1
    np.testing.assert_array_equal(covered, 1)
    np.testing.assert_allclose(mosaic, expected, equal_nan=True, atol=1e-6)


@pytest.mark.parametrize('method', ['two_triangle', 'jenness'])
@pytest.mark.parametrize('buffer_size', [0.0, 0.4])
def test_rumple_tiles_match_whole_grid_and_georeferencing(rumple_ept, tmp_path, buffer_size, method):
    points, reads = rumple_ept
    expected, extent = _whole_rumple(points, method=method)
    process_with_tiles(
        'fake_ept', (8, 12), str(tmp_path), 'rumple', (2, 3, 1),
        rumple_resolution=(4, 6), rumple_method=method,
        buffer_size=buffer_size, srs='EPSG:32605',
        bounds=([0.2, 21.8], [0.3, 26.8], [0, 100]),
    )
    assert len(list(tmp_path.glob('tile_*_rumple.tif'))) == 9
    assert len(reads) == 9
    assert reads[0][2] == [0, 100]
    # Fine CHM 22 x 27 becomes a 24 x 30 output footprint, padded east/south.
    np.testing.assert_allclose(extent, [0, 24, -3, 27])
    with rasterio.open(tmp_path / 'tile_0_0_rumple.tif') as src:
        assert tuple(src.bounds) == (0, 15, 8, 27)
    _assert_rumple_mosaic(tmp_path, expected, extent)


@pytest.mark.parametrize('method', ['two_triangle', 'jenness'])
def test_rumple_tiles_preserve_missing_outer_chm_cells(rumple_ept, tmp_path, method):
    points, _ = rumple_ept
    finite = np.isfinite(points['HeightAboveGround'])
    chm, extent = calculate_voxel_stat(points[finite], (2, 3, 1), 'HeightAboveGround', 'max')
    points['HeightAboveGround'][(points['X'] < 2) | (points['Y'] > 24)] = np.nan
    chm[0, :] = np.nan
    chm[:, 0] = np.nan
    expected, extent = calculate_rumple_grid(chm, (2, 3), (4, 6), extent=extent, method=method)
    process_with_tiles(
        'fake_ept', (8, 12), str(tmp_path), 'rumple', (2, 3, 1),
        rumple_resolution=(4, 6), rumple_method=method,
        buffer_size=0, srs='EPSG:32605', bounds=([0, 22], [0, 27]),
    )
    _assert_rumple_mosaic(tmp_path, expected, extent)


@pytest.mark.parametrize('method', ['two_triangle', 'jenness'])
@pytest.mark.parametrize('interpolation', ['linear', 'cubic', 'nearest'])
def test_rumple_tiles_interpolate_gaps_across_seams(rumple_ept, tmp_path, interpolation, method):
    points, _ = rumple_ept
    points['HeightAboveGround'] = 10.0 + 0.5 * points['X'] + 0.25 * points['Y']
    # Missing canopy cells straddle the seams at x=8 and y=15.
    gaps = ((points['X'] >= 7) & (points['X'] <= 9) &
            (points['Y'] >= 13.5) & (points['Y'] <= 16.5))
    points['HeightAboveGround'][gaps] = np.nan
    points['HeightAboveGround'][-2:] = np.nan  # Keep only samples at cell centers.
    process_with_tiles(
        'fake_ept', (8, 12), str(tmp_path), 'rumple', (2, 3, 1),
        rumple_resolution=(4, 6), rumple_method=method,
        interpolation=interpolation, buffer_size=3, srs='EPSG:32605',
        rumple_min_height=16, bounds=([0, 22], [0, 27]),
    )
    expected, extent = _whole_rumple(points, method=method, min_height=16,
                                     interpolation=interpolation)
    assert len(list(tmp_path.glob('tile_*_rumple.tif'))) == 9
    _assert_rumple_mosaic(tmp_path, expected, extent)


@pytest.mark.parametrize('method', ['two_triangle', 'jenness'])
def test_rumple_tiles_height_mask_selection_and_skip_existing(rumple_ept, tmp_path, method):
    points, reads = rumple_ept
    expected, _ = _whole_rumple(points, method=method, min_height=16)
    kwargs = dict(
        ept_file='fake_ept', tile_size=(8, 12), output_path=str(tmp_path),
        metric='rumple', voxel_size=(2, 3, 1), srs='EPSG:32605',
        rumple_resolution=(4, 6), rumple_method=method,
        bounds=([0, 22], [0, 27]), tile_indices={(1, 1)},
        rumple_min_height=16.0, buffer_size=0.0,
    )
    process_with_tiles(**kwargs)
    path = tmp_path / 'tile_1_1_rumple.tif'
    assert list(tmp_path.glob('*.tif')) == [path]
    with rasterio.open(path) as src:
        np.testing.assert_allclose(src.read(1, masked=True).filled(np.nan).T,
                                   expected[2:4, 2:4], equal_nan=True)
    original = path.read_bytes()
    process_with_tiles(**kwargs, skip_existing=True)
    assert len(reads) == 1
    assert path.read_bytes() == original


@pytest.mark.parametrize('method', ['two_triangle', 'jenness'])
def test_rumple_tiles_all_masked_write_nodata(rumple_ept, tmp_path, method):
    process_with_tiles(
        'fake_ept', (8, 12), str(tmp_path), 'rumple', (2, 3, 1),
        rumple_resolution=(4, 6), rumple_method=method,
        srs='EPSG:32605', bounds=([0, 22], [0, 27]),
        tile_indices={(1, 1)}, rumple_min_height=100.0,
    )
    with rasterio.open(tmp_path / 'tile_1_1_rumple.tif') as src:
        assert src.read(1, masked=True).mask.all()


@pytest.mark.parametrize('height', [None, -1.0, np.nan])
@patch('pyforestscan.process.pdal.Pipeline')
def test_rumple_tiles_empty_or_invalid_points_skip(mock_pipeline_cls, tmp_path, height):
    points = np.zeros(0 if height is None else 2,
                      dtype=[('X', 'f8'), ('Y', 'f8'), ('HeightAboveGround', 'f8')])
    if height is not None:
        points['HeightAboveGround'] = height
    mock_pipeline_cls.return_value.arrays = [points]
    process_with_tiles('fake_ept', (4, 4), str(tmp_path), 'rumple', (1, 1, 1),
                       rumple_resolution=2, srs='EPSG:32605', bounds=([0, 4], [0, 4]))
    assert not list(tmp_path.glob('*.tif'))


@pytest.mark.parametrize('overrides, message', [
    ({'voxel_size': (0, 1, 1)}, 'voxel_size'),
    ({'voxel_size': (1, np.inf, 1)}, 'voxel_size'),
    ({'rumple_resolution': None}, 'rumple_resolution'),
    ({'rumple_resolution': (3, 6)}, 'grid_resolution'),
    ({'rumple_resolution': (1, 3)}, 'grid_resolution'),
    ({'rumple_method': 'jensen'}, 'method'),
    ({'tile_size': (8,)}, 'tile_size'),
    ({'tile_size': (6, 12)}, 'tile_size'),
    ({'buffer_size': -0.1}, 'buffer_size'),
    ({'rumple_min_height': np.nan}, 'rumple_min_height'),
    ({'interpolation': 'bilinear'}, 'interpolation'),
    ({'bounds': ([0, np.inf], [0, 4])}, 'bounds'),
    ({'bounds': ([4, 0], [0, 4])}, 'bounds'),
])
def test_rumple_tiles_validate_grid_before_reading(tmp_path, overrides, message):
    kwargs = dict(ept_file='fake_ept', tile_size=(8, 12), output_path=str(tmp_path),
                  metric='rumple', voxel_size=(2, 3, 1), rumple_resolution=(4, 6))
    kwargs.update(overrides)
    with patch('pyforestscan.process.pdal.Pipeline') as pipeline:
        with pytest.raises(ValueError, match=message):
            process_with_tiles(**kwargs)
        pipeline.assert_not_called()


@pytest.mark.parametrize('method', ['two_triangle', 'jenness'])
def test_rumple_tiles_assign_points_on_outer_bounds(rumple_ept, tmp_path, method):
    points, _ = rumple_ept
    # Reuse the two extra returns for exact outer-edge samples.
    points['X'][-2:] = [22, 0]
    points['Y'][-2:] = [0, 27]
    points['HeightAboveGround'][-2:] = [60, 40]
    expected, extent = _whole_rumple(points, method=method)
    process_with_tiles(
        'fake_ept', (8, 12), str(tmp_path), 'rumple', (2, 3, 1),
        rumple_resolution=(4, 6), rumple_method=method, buffer_size=0,
        srs='EPSG:32605', bounds=([0, 22], [0, 27]),
    )
    _assert_rumple_mosaic(tmp_path, expected, extent)


@pytest.mark.parametrize('interpolation', ['linear', 'cubic'])
@patch('pyforestscan.process.pdal.Pipeline')
def test_rumple_tiles_sparse_interpolation_leaves_missing_heights(mock_pipeline_cls, tmp_path, interpolation):
    # A single sample cannot support a linear/cubic interpolation surface.
    points = np.array([(0.5, 0.5, 10.0)],
                      dtype=[('X', 'f8'), ('Y', 'f8'), ('HeightAboveGround', 'f8')])
    mock_pipeline_cls.return_value.arrays = [points]
    process_with_tiles(
        'fake_ept', (4, 4), str(tmp_path), 'rumple', (1, 1, 1),
        rumple_resolution=2, interpolation=interpolation,
        srs='EPSG:32605', bounds=([0, 4], [0, 4]),
    )
    with rasterio.open(tmp_path / 'tile_0_0_rumple.tif') as src:
        assert src.read(1, masked=True).mask.all()


@pytest.mark.parametrize('method', ['two_triangle', 'jenness'])
def test_rumple_tiles_large_output_cell_does_not_allocate_fine_padding(rumple_ept, tmp_path, method):
    from pyforestscan.calculate import calculate_rumple

    points, _ = rumple_ept
    points = points[np.isfinite(points['HeightAboveGround'])]
    chm, _ = calculate_voxel_stat(points, (2, 3, 1), 'HeightAboveGround', 'max')
    expected = calculate_rumple(chm, (2, 3), method=method)
    output_resolution = (100_000_000, 99_999_999)
    process_with_tiles(
        'fake_ept', output_resolution, str(tmp_path), 'rumple', (2, 3, 1),
        rumple_resolution=output_resolution, rumple_method=method,
        buffer_size=0, srs='EPSG:32605', bounds=([0, 22], [0, 27]),
    )
    with rasterio.open(tmp_path / 'tile_0_0_rumple.tif') as src:
        assert src.shape == (1, 1)
        assert src.res == output_resolution
        np.testing.assert_allclose(src.read(1), [[expected]])
