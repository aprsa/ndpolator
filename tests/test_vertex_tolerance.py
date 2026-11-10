import numpy as np
import pytest

from ndpolator import Ndpolator

# vertex matching must not depend on the units of the axis: the same grid is
# tested with a spacing far below, at, and far above the 1e-6 tolerance
SCALES = pytest.mark.parametrize("scale", [1e-9, 1.0, 2.0**40])

ON_VERTEX = 1


def _ndp(axis, values):
    nd = Ndpolator((np.asarray(axis, float),))
    nd.register("t", None, np.asarray(values, float).reshape(-1, 1))
    return nd


@SCALES
def test_interpolation_does_not_depend_on_axis_scale(scale):
    """The enclosing cell is found by exact bisection, whatever the spacing of the axis."""
    rng = np.random.default_rng(0)
    axis = scale * np.cumsum(rng.uniform(0.5, 1.5, 40))
    values = np.sin(axis / scale)  # non-linear, so that a wrong cell gives a wrong value
    cells = rng.integers(1, len(axis), 200)
    q = axis[cells - 1] + rng.uniform(0.01, 0.99, 200) * (axis[cells] - axis[cells - 1])

    nd = _ndp(axis, values)
    indices, flags, normed = nd.import_query_pts("t", q.reshape(-1, 1))
    assert np.array_equal(np.ravel(indices), cells)
    assert not np.any(flags)

    interps = nd.ndpolate("t", q.reshape(-1, 1))["interps"]
    np.testing.assert_allclose(np.ravel(interps), np.interp(q, axis, values), rtol=1e-9, atol=1e-12)


@SCALES
def test_vertex_matching_is_symmetric_and_scale_free(scale):
    """A query within tolerance of a vertex, on either side, takes the vertex value and nothing else."""
    axis = scale * np.arange(1.0, 6.0)
    # both neighbors of the middle vertex are undefined, so anything but an
    # on-vertex match of the middle vertex yields nan:
    nd = _ndp(axis, [1.0, np.nan, 3.0, np.nan, 5.0])
    q = (axis[2] * np.array([1 - 1e-9, 1.0, 1 + 1e-9])).reshape(-1, 1)

    indices, flags, normed = nd.import_query_pts("t", q)
    assert np.all(flags == ON_VERTEX)
    np.testing.assert_array_equal(np.ravel(nd.ndpolate("t", q)["interps"]), 3.0)


def test_roundoff_at_the_grid_boundary_is_on_vertex():
    """A query a rounding error outside of the axis span is on the boundary vertex, not out of bounds."""
    axis = np.array([0.1, 0.2, 0.3])
    nd = _ndp(axis, [1.0, 2.0, 4.0])
    q = np.array([[np.nextafter(0.1, 0.0)], [0.1 * 3]])
    assert q[0, 0] < axis[0] and q[1, 0] > axis[-1]

    indices, flags, normed = nd.import_query_pts("t", q)
    assert np.all(flags == ON_VERTEX)
    np.testing.assert_array_equal(np.ravel(nd.ndpolate("t", q)["interps"]), [1.0, 4.0])

    # anything beyond the tolerance is still out of bounds:
    far = np.array([[0.1 - 1e-3], [0.3 + 1e-3]])
    assert np.all(np.isnan(nd.ndpolate("t", far)["interps"]))
