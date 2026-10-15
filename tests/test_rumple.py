"""Geometry, compatibility, and independent R references for rumple."""
import json
from pathlib import Path

import numpy as np
import pytest

from pyforestscan import calculate_rumple, calculate_rumple_grid


REFERENCE = json.loads((Path(__file__).parent / 'fixtures' / 'rumple_lidr_reference.json').read_text())


@pytest.mark.parametrize('case', REFERENCE['cases'], ids=lambda case: case['name'])
def test_jenness_matches_lidr_raster_reference_including_gaps_and_edges(case):
    chm = np.asarray(case['chm'], dtype=float)
    resolution = case['cell_resolution']
    expected = np.asarray(case['per_cell_rumple'], dtype=float)
    actual, _ = calculate_rumple_grid(chm, resolution, resolution, method='jenness')
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    scalar = np.nan if case['scalar_rumple'] is None else case['scalar_rumple']
    np.testing.assert_allclose(calculate_rumple(chm, resolution, method='jenness'), scalar,
                               rtol=1e-12, atol=1e-12)


def test_two_triangle_fixed_diagonal_has_known_area_and_cell_clipping():
    # Horizontal triangle: area 1/2. Other triangle: sqrt(3)/2.
    chm = np.array([[0., 0.], [0., 1.]])
    ratio = (1 + np.sqrt(3)) / 2
    assert calculate_rumple(chm, (1, 1)) == pytest.approx(ratio)
    fine, _ = calculate_rumple_grid(chm, (1, 1), 1)
    np.testing.assert_allclose(fine, [[1., ratio], [ratio, np.sqrt(3)]])
    coarse, _ = calculate_rumple_grid(chm, (1, 1), 2)
    assert coarse[0, 0] == pytest.approx(ratio)


@pytest.mark.parametrize('method', ['two_triangle', 'jenness'])
def test_grid_uses_fine_surface_and_recovers_scalar_in_one_cell(method):
    chm = np.random.default_rng(41).uniform(2, 30, (20, 20))
    original = chm.copy()
    result, extent = calculate_rumple_grid(chm, (1, 1), 20, method=method)
    assert result.shape == (1, 1)
    assert extent == [0, 20, 0, 20]
    assert result[0, 0] > 1
    assert result[0, 0] == pytest.approx(calculate_rumple(chm, (1, 1), method=method))
    np.testing.assert_array_equal(chm, original)


@pytest.mark.parametrize('method', ['two_triangle', 'jenness'])
def test_grid_missing_support_is_excluded_from_both_areas(method):
    chm = np.full((4, 4), np.nan)
    chm[1:3, 1:3] = [[10, 10], [10, 11]]
    result, _ = calculate_rumple_grid(chm, (1, 1), 2, method=method)
    assert np.isfinite(result).all()  # Valid patch crosses all four output cells.
    assert (result >= 1).all()
    if method == 'two_triangle':
        ratio = (1 + np.sqrt(3)) / 2
        np.testing.assert_allclose(result, [[1., ratio], [ratio, np.sqrt(3)]])
    whole, _ = calculate_rumple_grid(chm, (1, 1), 4, method=method)
    assert whole[0, 0] == pytest.approx(calculate_rumple(chm, (1, 1), method=method))


def test_two_triangle_grid_plane_including_partial_outer_cells():
    dx, dy = 2., 3.
    chm = 10 + .5 * np.arange(7)[:, None] * dx + .25 * np.arange(5)[None, :] * dy
    result, extent = calculate_rumple_grid(chm, (dx, dy), (6, 6),
                                         extent=[100, 114, 200, 215])
    assert result.shape == (3, 3)
    assert extent == [100, 118, 197, 215]
    np.testing.assert_allclose(result, np.sqrt(1 + .5**2 + .25**2))


def test_jenness_grid_uses_neighbors_across_output_boundaries():
    dx, dy = 2., 3.
    chm = 10 + .5 * np.arange(9)[:, None] * dx + .25 * np.arange(9)[None, :] * dy
    result, _ = calculate_rumple_grid(chm, (dx, dy), (6, 9), method='jenness')
    assert result[1, 1] == pytest.approx(np.sqrt(1 + .5**2 + .25**2))
    # lidR repeats heights at the true raster boundary, reducing slope there.
    assert 1 < result[0, 0] < result[1, 1]


def test_grid_does_not_average_ratios_with_unequal_supported_area():
    chm = np.array([[0., 0., 0.], [0., 1., 4.], [0., 3., 2.]])
    fine, _ = calculate_rumple_grid(chm, (1, 1), 1)
    coarse, _ = calculate_rumple_grid(chm, (1, 1), 3)
    # Corners have 1/4 m2, edge centers 1/2 m2, center 1 m2 support.
    ground = np.array([[.25, .5, .25], [.5, 1., .5], [.25, .5, .25]])
    assert coarse[0, 0] == pytest.approx(np.sum(fine * ground) / ground.sum())
    assert not np.isclose(coarse[0, 0], fine.mean())


@pytest.mark.parametrize('method', ['two_triangle', 'jenness'])
def test_masking_precedes_area_calculation_and_preserves_input(method):
    chm = np.full((6, 8), 12.)
    chm[1, 1] = 0
    chm[4, 4] = np.inf
    original = chm.copy()
    masked = np.where((chm >= 2) & np.isfinite(chm), chm, np.nan)
    actual, extent = calculate_rumple_grid(chm, (1, 1), 2, min_height=2, method=method)
    expected, _ = calculate_rumple_grid(masked, (1, 1), 2, method=method)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(chm, original)
    np.testing.assert_allclose(calculate_rumple(chm, (1, 1), 2, method=method),
                               calculate_rumple(masked, (1, 1), method=method))
    empty, same_extent = calculate_rumple_grid(chm, (1, 1), 2, min_height=100, method=method)
    assert np.isnan(empty).all()
    assert same_extent == extent


@pytest.mark.parametrize('method', ['two_triangle', 'jenness'])
def test_all_missing_output_and_partial_extent(method):
    result, extent = calculate_rumple_grid(np.full((3, 5), np.nan), (2, 3), (4, 9),
                                         method=method)
    assert result.shape == (2, 2)
    assert np.isnan(result).all()
    assert extent == [0, 8, -3, 15]
    assert np.isnan(calculate_rumple(np.full((3, 5), np.nan), (2, 3), method=method))


@pytest.mark.parametrize('shape', [(1, 1), (1, 3), (3, 1), (0, 2)])
def test_two_triangle_needs_a_complete_patch(shape):
    assert np.isnan(calculate_rumple(np.ones(shape), (1, 1)))
    if min(shape) > 0:
        result, _ = calculate_rumple_grid(np.ones(shape), (1, 1), 2)
        assert np.isnan(result).all()


@pytest.mark.parametrize('method', ['two_triangle', 'jenness'])
def test_empty_grid_is_rejected(method):
    with pytest.raises(ValueError, match='nonempty'):
        calculate_rumple_grid(np.ones((0, 2)), (1, 1), 2, method=method)


@pytest.mark.parametrize('resolution', [(0, 1), (-1, 1), (np.nan, 1), (1, np.inf), (1,), (1, 1, 1), 1])
def test_invalid_chm_resolution(resolution):
    with pytest.raises(ValueError, match='cell_resolution'):
        calculate_rumple(np.ones((2, 2)), resolution)
    with pytest.raises(ValueError, match='cell_resolution'):
        calculate_rumple_grid(np.ones((2, 2)), resolution, 2)


@pytest.mark.parametrize('resolution', [0, -1, np.nan, np.inf, (1,), (1, 2, 3), (2, 0), .5, 1.5, (2, 1.5)])
def test_invalid_output_resolution(resolution):
    with pytest.raises(ValueError, match='grid_resolution'):
        calculate_rumple_grid(np.ones((2, 2)), (1, 1), resolution)


@pytest.mark.parametrize('extent', [[0, 3, 0, 2], [2, 0, 0, 2], [0, 2, 2, 0], [0, 2, 0], [np.nan, 2, 0, 2]])
def test_extent_must_match_chm(extent):
    with pytest.raises(ValueError, match='extent'):
        calculate_rumple_grid(np.ones((2, 2)), (1, 1), 2, extent=extent)


@pytest.mark.parametrize('kwargs,match', [({'method': 'jensen'}, 'method'),
                                        ({'min_height': np.nan}, 'min_height')])
def test_invalid_method_or_height(kwargs, match):
    with pytest.raises(ValueError, match=match):
        calculate_rumple(np.ones((2, 2)), (1, 1), **kwargs)
    with pytest.raises(ValueError, match=match):
        calculate_rumple_grid(np.ones((2, 2)), (1, 1), 2, **kwargs)


def test_grid_requires_2d_chm():
    with pytest.raises(ValueError, match='2D'):
        calculate_rumple_grid(np.ones((2, 2, 2)), (1, 1), 2)


def test_output_cell_larger_than_chm_does_not_require_fine_grid_padding():
    chm = np.array([[0., 0.], [0., 1.]])
    result, extent = calculate_rumple_grid(chm, (1, 1), 1_000_000)
    assert result.shape == (1, 1)
    assert extent == [0, 1_000_000, -999_998, 2]
    assert result[0, 0] == pytest.approx(calculate_rumple(chm, (1, 1)))
