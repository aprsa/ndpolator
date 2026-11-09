import numpy as np
import pytest

from ndpolator import Ndpolator

# the fix touches both ndp_find_nearest() branches, so every test runs with both search algorithms
SEARCH_ALGORITHMS = pytest.mark.parametrize("search_algorithm", ["kdtree", "linear"])


def _toy(m):
    """One basic axis x in [0, 1] and one associated axis m; the values depend on m only."""
    nd = Ndpolator((np.array([0.0, 1.0]),))
    vals = {2: [0.0, 10.0], 3: [0.0, 10.0, 11.0]}[len(m)]
    g = np.empty((2, len(m), 1))
    for k, v in enumerate(vals):
        g[:, k, 0] = v
    nd.register("toy", (np.asarray(m, float),), g)
    return nd


@SEARCH_ALGORITHMS
def test_linear_extrapolation_takes_the_enclosing_associated_cell(search_algorithm):
    """Off-grid in x, so 'linear' extrapolates; along m it must stay in the cell that encloses the query."""
    nd = _toy([0.0, 1.0, 2.0])
    q = np.array([[1.5, 1.3], [1.5, 1.7], [0.5, 1.3], [1.5, 0.4], [1.5, 0.0], [-0.5, -0.3], [1.5, 2.5]])
    r = nd.ndpolate("toy", q, extrapolation_method="linear", search_algorithm=search_algorithm)
    np.testing.assert_allclose(np.ravel(r["interps"]), [10.3, 10.7, 10.3, 4.0, 0.0, -3.0, 11.5], rtol=0, atol=1e-12)
    # distances keep their definition (basic-axis overshoot plus the offset from the nearest associated node)
    np.testing.assert_allclose(np.ravel(r["dists"]), [0.34, 0.34, 0.0, 0.41, 0.25, 0.34, 0.5], rtol=0, atol=1e-12)


@SEARCH_ALGORITHMS
def test_linear_extrapolation_length2_associated_axis(search_algorithm):
    nd = _toy([0.0, 1.0])
    q = np.array([[1.5, 0.3], [1.5, 0.8], [1.5, -0.2], [1.5, 1.4]])
    r = nd.ndpolate("toy", q, extrapolation_method="linear", search_algorithm=search_algorithm)
    np.testing.assert_allclose(np.ravel(r["interps"]), [3.0, 8.0, -2.0, 14.0], rtol=0, atol=1e-12)


@SEARCH_ALGORITHMS
def test_nearest_unchanged(search_algorithm):
    nd = _toy([0.0, 1.0, 2.0])
    q = np.array([[1.5, 1.3], [1.5, 0.4], [1.5, 1.7]])
    r = nd.ndpolate("toy", q, extrapolation_method="nearest", search_algorithm=search_algorithm)
    np.testing.assert_allclose(np.ravel(r["interps"]), [10.0, 0.0, 11.0], rtol=0, atol=1e-12)
