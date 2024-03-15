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
    ndp.register('main', associated_axes=None, grid=grid)
    assert len(ndp.tables) == 1


def test_ndp_query_pts_import():
    ax1 = np.linspace(1000, 5000, 5)
    ax2 = np.linspace(1, 5, 5)
    ax3 = np.linspace(0.1, 0.5, 5)

    basic_axes = (ax1, ax2, ax3)
    ndp = ndpolator.Ndpolator(basic_axes=basic_axes)

    grid = np.random.normal(loc=1.0, scale=0.1, size=(len(ax1), len(ax2), len(ax3)))
    ndp.register('main', associated_axes=None, grid=grid)

    query_pts = np.array([
        [750, 2.25, 0.35],
        [1250, 2.5, 0.375],
        [2500, 3.75, 0.4],
        [2750, 4.0, 0.5],
        [3000, 4.25, 0.525],
    ])

    indices, flags, normed_query_pts = ndp.import_query_pts('main', query_pts=query_pts)

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
    ndp.register('main', associated_axes=None, grid=grid)

    query_pts = np.array([
        [750, 2.25, 0.35],
        [1250, 2.5, 0.375],
        [2500, 3.75, 0.4],
        [2750, 4.0, 0.5],
        [3000, 4.25, 0.525],
        [3000, 4.0, 0.425],
        [3000, 4.0, 0.4]
    ])

    indices, flags, normed_query_pts = ndp.import_query_pts('main', query_pts=query_pts)
    hypercubes = ndp.find_hypercubes('main', indices=indices, flags=flags)

    expected_hypercube_shape = [(2, 2, 2, 1), (2, 2, 2, 1), (2, 2, 1), (2, 1), (2, 2, 1), (2, 1), (1,)]

    for i, hypercube in enumerate(hypercubes):
        assert hypercube.shape == expected_hypercube_shape[i]


def test_ndpolate():
    ax1 = np.linspace(1000, 5000, 5)
    ax2 = np.linspace(1, 5, 5)
    ax3 = np.linspace(0.1, 0.5, 5)

    basic_axes = (ax1, ax2, ax3)
    ndp = ndpolator.Ndpolator(basic_axes=basic_axes)

    def fv(query_pt):
        return query_pt[0]/1000 + query_pt[1] + query_pt[2]*10

    grid = np.empty((len(ax1), len(ax2), len(ax3), 1))
    for i, x in enumerate(ax1):
        for j, y in enumerate(ax2):
            for k, z in enumerate(ax3):
                grid[i, j, k, 0] = fv((x, y, z))

    ndp.register('main', associated_axes=None, grid=grid)

    # regular interpolation:
    query_pts = np.ascontiguousarray(np.vstack((np.random.uniform(1000, 5000, 1000), np.random.uniform(1, 5, 1000), np.random.uniform(0.1, 0.5, 1000))).T)
    expected_interps = np.array([fv(query_pt) for query_pt in query_pts])
    interps = ndp.ndpolate('main', query_pts, extrapolation_method='none')
    assert np.allclose(interps[:,0], expected_interps, rtol=1e-5)

    # extrapolation='none':
    query_pts = np.ascontiguousarray(np.vstack((np.random.uniform(500, 5500, 1000), np.random.uniform(0.5, 5.5, 1000), np.random.uniform(0.05, 0.55, 1000))).T)
    expected_interps = np.array([fv(query_pt) for query_pt in query_pts])
    interps = ndp.ndpolate('main', query_pts, extrapolation_method='none')
    out_of_bounds = np.argwhere(
        (query_pts[:,0] < ax1[0]) | (query_pts[:,0] > ax1[-1]) |
        (query_pts[:,1] < ax2[0]) | (query_pts[:,1] > ax2[-1]) |
        (query_pts[:,2] < ax3[0]) | (query_pts[:,2] > ax3[-1]))
    assert np.all(np.argwhere(np.isnan(interps[:,0])) == out_of_bounds)

    # extrapolation='nearest':
    nx = [np.argmin(np.abs(ax1-query_pt[0])) for query_pt in query_pts]
    ny = [np.argmin(np.abs(ax2-query_pt[1])) for query_pt in query_pts]
    nz = [np.argmin(np.abs(ax3-query_pt[2])) for query_pt in query_pts]
    nearest_indices = np.vstack((nx, ny, nz)).T
    nearest_values = np.array([fv((ax1[n[0]], ax2[n[1]], ax3[n[2]])) for n in nearest_indices])
    rounded_interps = expected_interps.copy()
    rounded_interps[out_of_bounds.flatten()] = nearest_values[out_of_bounds.flatten()]

    interps = ndp.ndpolate('main', query_pts, extrapolation_method='nearest')
    assert np.allclose(interps[:,0], rounded_interps, rtol=1e-5)

    # extrapolation='linear':
    interps = ndp.ndpolate('main', query_pts, extrapolation_method='linear')
    assert np.allclose(interps[:,0], expected_interps, rtol=1e-5)
