"""
Tests for the ownership of numpy arrays by the C extension.

The C-side ndp_table borrows the axis and grid buffers of the numpy arrays
it is built from. The capsule cached by the Ndpolator class must therefore
keep those arrays alive for as long as the table exists, and must release
them (and nothing more) afterwards. Temporary tables built by distance()
and hypercubes() must not leak references either.
"""

import gc
import sys
import weakref

import numpy as np
import pytest

import ndpolator


QUERY_PT = np.array([[0.5, 0.5]])
EXPECTED = 2.0  # mean of the four surrounding vertices: (0 + 1 + 3 + 4) / 4


def make_ndpolator(grid=None):
    axes = (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    ndp = ndpolator.Ndpolator(axes)
    if grid is None:
        grid = np.arange(9, dtype=float).reshape(3, 3, 1)
    ndp.register('t', None, grid)
    return ndp, grid


def test_distance_does_not_leak_grid_references():
    # distance() never caches a capsule, so each call builds and frees a table:
    ndp, grid = make_ndpolator()
    before = sys.getrefcount(grid)
    for _ in range(10):
        ndp.distance('t', QUERY_PT)
    assert sys.getrefcount(grid) == before


def test_hypercubes_does_not_leak_grid_references():
    ndp, grid = make_ndpolator()
    indices, flags, normed = ndp.import_query_pts('t', QUERY_PT)
    before = sys.getrefcount(grid)
    for _ in range(10):
        ndp.find_hypercubes('t', normed, indices, flags)
    assert sys.getrefcount(grid) == before


def test_capsule_holds_one_reference_and_releases_it():
    ndp, grid = make_ndpolator()
    before = sys.getrefcount(grid)  # includes the reference held by ndp.table

    ndp.ndpolate('t', QUERY_PT)  # creates and caches the capsule
    assert sys.getrefcount(grid) == before + 1

    for _ in range(10):
        ndp.ndpolate('t', QUERY_PT)  # cached capsule: no new references
        ndp.distance('t', QUERY_PT)  # cached capsule: no temporary table
    assert sys.getrefcount(grid) == before + 1

    del ndp
    gc.collect()
    assert sys.getrefcount(grid) == before - 1  # capsule and ndp.table are both gone


def test_capsule_keeps_grid_alive():
    ndp, grid = make_ndpolator()
    grid_ref = weakref.ref(grid)
    ndp.ndpolate('t', QUERY_PT)

    # drop every python-side reference to the grid:
    ndp.table['t']['grid'] = None
    del grid
    gc.collect()

    assert grid_ref() is not None, 'the capsule must keep the grid alive'
    assert ndp.ndpolate('t', QUERY_PT)['interps'][0, 0] == pytest.approx(EXPECTED)

    del ndp
    gc.collect()
    assert grid_ref() is None, 'the capsule must release the grid when collected'


@pytest.mark.parametrize('grid', [
    pytest.param(np.asfortranarray(np.arange(9, dtype=float).reshape(3, 3, 1)), id='non-contiguous'),
    pytest.param(np.arange(9, dtype=np.float32).reshape(3, 3, 1), id='float32'),
])
def test_converted_grid_is_owned_by_the_capsule(grid):
    # grids that are not C-contiguous float64 are converted; the capsule must
    # own the converted copy rather than the original array:
    ndp, grid = make_ndpolator(grid)
    before = sys.getrefcount(grid)

    assert ndp.ndpolate('t', QUERY_PT)['interps'][0, 0] == pytest.approx(EXPECTED)
    assert sys.getrefcount(grid) == before

    ndp.table['t']['grid'] = None
    del grid
    gc.collect()
    assert ndp.ndpolate('t', QUERY_PT)['interps'][0, 0] == pytest.approx(EXPECTED)


def test_find_converts_axes_and_query_pts():
    # import_query_pts() must accept the same inputs as ndpolate(): integer or
    # non-contiguous axes and query points are converted rather than reinterpreted.
    float_axes = (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    int_axes = (np.array([0, 1, 2]), np.array([0, 1, 2]))
    strided_axes = (np.array([0.0, 9.0, 1.0, 9.0, 2.0])[::2], np.array([0.0, 9.0, 1.0, 9.0, 2.0])[::2])
    grid = np.arange(9, dtype=float).reshape(3, 3, 1)

    float_qpts = np.array([[0.5, 1.5], [1.0, 1.0], [2.5, -0.5]])
    int_qpts = np.array([[1, 1]])

    reference = ndpolator.Ndpolator(float_axes)
    reference.register('t', None, grid)
    expected = reference.import_query_pts('t', float_qpts)

    for axes in (int_axes, strided_axes):
        ndp = ndpolator.Ndpolator(axes)
        ndp.register('t', None, grid)
        for got, want in zip(ndp.import_query_pts('t', float_qpts), expected):
            np.testing.assert_array_equal(got, want)

    expected_int = reference.import_query_pts('t', float_qpts[1:2])
    for got, want in zip(reference.import_query_pts('t', int_qpts), expected_int):
        np.testing.assert_array_equal(got, want)


def test_find_does_not_leak_references():
    axes = (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    ndp = ndpolator.Ndpolator(axes)
    ndp.register('t', None, np.arange(9, dtype=float).reshape(3, 3, 1))
    before = tuple(sys.getrefcount(axis) for axis in axes) + (sys.getrefcount(QUERY_PT),)
    for _ in range(10):
        ndp.import_query_pts('t', QUERY_PT)
    assert tuple(sys.getrefcount(axis) for axis in axes) + (sys.getrefcount(QUERY_PT),) == before


def test_integer_axes_are_converted():
    # axes are converted to float64 the same way the grid is:
    ndp = ndpolator.Ndpolator((np.array([0, 1, 2]), np.array([0, 1, 2])))
    ndp.register('t', None, np.arange(9, dtype=float).reshape(3, 3, 1))
    assert ndp.ndpolate('t', QUERY_PT)['interps'][0, 0] == pytest.approx(EXPECTED)
    assert ndp.distance('t', np.array([[2.5, 1.0]]))[0, 0] == pytest.approx(0.25)
