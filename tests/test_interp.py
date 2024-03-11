import pytest
import numpy as np

try:
    import ndpolator
    ndpolator.__version__
except ImportError:
    pytest.fail('Failed to import the ndpolator module.')


def test_instantiatiation():
    ax1 = np.linspace(1000, 5000, 5)
    ax2 = np.linspace(1, 5, 5)
    ax3 = np.linspace(0.1, 0.5, 5)

    basic_axes = (ax1, ax2, ax3)
    ndp = ndpolator.Ndpolator(basic_axes=basic_axes)

    assert ndp


def test_registration():
    ax1 = np.linspace(1000, 5000, 5)
    ax2 = np.linspace(1, 5, 5)
    ax3 = np.linspace(0.1, 0.5, 5)

    basic_axes = (ax1, ax2, ax3)
    ndp = ndpolator.Ndpolator(basic_axes=basic_axes)

    grid = np.random.normal(loc=1.0, scale=0.1, size=(len(ax1), len(ax2), len(ax3)))
    ndp.register('main', attached_axes=None, grid=grid)
    assert len(ndp.tables) == 1


def test_find_indices():
    ax1 = np.linspace(1000, 5000, 5)
    ax2 = np.linspace(1, 5, 5)
    ax3 = np.linspace(0.1, 0.5, 5)

    basic_axes = (ax1, ax2, ax3)
    ndp = ndpolator.Ndpolator(basic_axes=basic_axes)

    grid = np.random.normal(loc=1.0, scale=0.1, size=(len(ax1), len(ax2), len(ax3)))
    ndp.register('main', attached_axes=None, grid=grid)

    query_pts = np.array([
        [750, 2.25, 0.35],
        [1250, 2.5, 0.375],
        [2500, 3.75, 0.4],
        [2750, 4.0, 0.5],
        [3000, 4.25, 0.525],
    ])

    indices, flags, normed_query_pts = ndp.find_indices('main', query_pts=query_pts)

    expected_indices = np.array([
        [0, 2, 3],
        [1, 2, 3],
        [2, 3, 3],
        [2, 3, 4],
        [2, 4, 4]
    ])

    expected_flags = np.array([
        [2, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 2]
    ])

    expected_normed_query_pts = np.array([
        [-0.25, 0.25, 0.5],
        [0.25, 0.50, 0.75],
        [0.50, 0.75, 0.00],
        [0.75, 0.00, 0.00],
        [0.00, 0.25, 1.25],
    ])

    assert np.allclose(indices, expected_indices)
    assert np.allclose(flags, expected_flags)
    assert np.allclose(normed_query_pts, expected_normed_query_pts)


def test_find_hypercubes():
    ax1 = np.linspace(1000, 5000, 5)
    ax2 = np.linspace(1, 5, 5)
    ax3 = np.linspace(0.1, 0.5, 5)

    basic_axes = (ax1, ax2, ax3)
    ndp = ndpolator.Ndpolator(basic_axes=basic_axes)

    grid = np.random.normal(loc=1.0, scale=0.1, size=(len(ax1), len(ax2), len(ax3), 1))
    ndp.register('main', attached_axes=None, grid=grid)

    query_pts = np.array([
        [750, 2.25, 0.35],
        [1250, 2.5, 0.375],
        [2500, 3.75, 0.4],
        [2750, 4.0, 0.5],
        [3000, 4.25, 0.525],
        [3000, 4.0, 0.425],
        [3000, 4.0, 0.4]
    ])

    indices, flags, normed_query_pts = ndp.find_indices('main', query_pts=query_pts)
    hypercubes = ndp.find_hypercubes('main', indices=indices, flags=flags)

    expected_hypercube_shape = [(2, 2, 2, 1), (2, 2, 2, 1), (2, 2, 1), (2, 1), (2, 2, 1), (2, 1), (1,)]

    for i, hypercube in enumerate(hypercubes):
        assert hypercube.shape == expected_hypercube_shape[i]
