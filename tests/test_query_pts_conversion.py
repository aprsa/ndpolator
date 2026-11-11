"""
Tests for the conversion of query points by the C extension.

ndpolate() and distance() read query points as a C-contiguous float64
buffer. Anything else (integers, float32, byte-swapped or strided arrays,
nested lists) must be converted first rather than have its memory
reinterpreted, the same way axes, grids and import_query_pts() are handled.
"""

import gc
import sys

import numpy as np
import pytest

import cndpolator
import ndpolator


AXES = (np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0, 3.0]))
GRID = (AXES[0][:, None] + 10 * AXES[1][None, :])[..., None]  # f(x, y) = x + 10y

# on and off the grid, on and off vertices; every value is exact in float32:
FLOATS = np.array([[0.5, 1.25], [1.0, 2.0], [3.0, 0.0], [4.5, 1.0], [-1.0, 3.5]])
INTS = np.array([[1, 2], [0, 3], [3, 0], [4, 1], [-1, 5]])

QUERY_PTS = pytest.mark.parametrize('query_pts', [
    pytest.param(INTS.astype(np.int64), id='int64'),
    pytest.param(INTS.astype(np.int32), id='int32'),
    pytest.param(INTS.tolist(), id='list-of-ints'),
    pytest.param(FLOATS.astype(np.float32), id='float32'),
    pytest.param(FLOATS.astype(FLOATS.dtype.newbyteorder()), id='byte-swapped'),
    pytest.param(np.asfortranarray(FLOATS), id='fortran-order'),
    pytest.param(np.repeat(FLOATS, 2, axis=1)[:, ::2], id='strided'),
])


def _ndp():
    ndp = ndpolator.Ndpolator(AXES)
    ndp.register('t', None, GRID)
    return ndp


def _float64(query_pts):
    """The same query points as a new C-contiguous float64 array."""
    return np.array(query_pts, dtype=np.float64, order='C')


@QUERY_PTS
def test_ndpolate_converts_query_pts(query_pts):
    want = _ndp().ndpolate('t', _float64(query_pts), extrapolation_method='linear')
    np.testing.assert_allclose(want['interps'][:, 0], _float64(query_pts) @ [1, 10])  # guards the reference itself

    ndp = _ndp()
    for _ in range(2):  # the first call builds the table, the second one uses the cached capsule
        got = ndp.ndpolate('t', query_pts, extrapolation_method='linear')
        np.testing.assert_array_equal(got['interps'], want['interps'])
        np.testing.assert_array_equal(got['dists'], want['dists'])


@QUERY_PTS
def test_distance_converts_query_pts(query_pts):
    want = _ndp().distance('t', _float64(query_pts))
    assert np.count_nonzero(want) == 2  # two query points are off the grid

    ndp = _ndp()
    np.testing.assert_array_equal(ndp.distance('t', query_pts), want)  # temporary table
    ndp.ndpolate('t', FLOATS)
    np.testing.assert_array_equal(ndp.distance('t', query_pts), want)  # cached capsule


@QUERY_PTS
def test_extension_converts_query_pts(query_pts):
    # the same, without the np.ascontiguousarray() call of the python wrapper:
    linear = cndpolator.ExtrapolationMethod.LINEAR
    interps, dists, capsule = cndpolator.ndpolate(query_pts=_float64(query_pts), axes=AXES, grid=GRID, extrapolation_method=linear)

    got = cndpolator.ndpolate(query_pts=query_pts, axes=AXES, grid=GRID, extrapolation_method=linear)
    np.testing.assert_array_equal(got[0], interps)
    np.testing.assert_array_equal(got[1], dists)

    got = cndpolator.ndpolate(capsule=capsule, query_pts=query_pts, extrapolation_method=linear)
    np.testing.assert_array_equal(got[0], interps)
    np.testing.assert_array_equal(got[1], dists)

    want = cndpolator.distance(_float64(query_pts), capsule)
    np.testing.assert_array_equal(cndpolator.distance(query_pts, capsule), want)
    np.testing.assert_array_equal(cndpolator.distance(query_pts, None, AXES, GRID, len(AXES)), want)


def test_query_pts_references_are_released():
    ndp = _ndp()
    query_pts = FLOATS.copy()  # already C-contiguous float64, so it is passed on as is
    before = sys.getrefcount(query_pts)
    for _ in range(10):
        ndp.distance('t', query_pts)  # the first pass has no capsule yet
        ndp.ndpolate('t', query_pts)
    assert sys.getrefcount(query_pts) == before


def test_unconvertible_query_pts_raise_and_do_not_leak():
    grid = GRID.copy()
    ndp = ndpolator.Ndpolator(AXES)
    ndp.register('t', None, grid)
    before = sys.getrefcount(grid)
    text = np.array([['a', 'b']])

    # no capsule yet, so the table built for the call must be released again:
    with pytest.raises((TypeError, ValueError)):
        ndp.ndpolate('t', text)
    with pytest.raises((TypeError, ValueError)):
        ndp.distance('t', text)
    gc.collect()
    assert sys.getrefcount(grid) == before

    ndp.ndpolate('t', FLOATS)
    capsule = ndp.table['t']['capsule']
    with pytest.raises((TypeError, ValueError)):
        ndp.ndpolate('t', text)
    with pytest.raises((TypeError, ValueError)):
        ndp.distance('t', text)
    with pytest.raises(ValueError):
        cndpolator.ndpolate(capsule=capsule)  # no query points at all
    assert ndp.ndpolate('t', FLOATS)['interps'][0, 0] == pytest.approx(13.0)  # the capsule is still usable
