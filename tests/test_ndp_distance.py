import pytest
import numpy as np

try:
    import ndpolator
    ndpolator.__version__
except ImportError:
    pytest.fail('Failed to import the ndpolator module.')


def test_distances():
    """
    Test Ndpolator.distance() method comprehensively.

    Uses a 3D grid with non-uniform axes to test distance calculations
    to the nearest hypercube. Tests include:
    - Points inside the grid (distance = 0)
    - Points on edges, faces, and vertices (distance = 0)
    - Points outside the grid at various distances
    - Distance to vertex (all dims outside)
    - Distance to edge (two dims outside)
    - Distance to face (one dim outside)
    """
    # Create 3D grid with non-uniform axes to test normalization
    # Physical ranges: [10, 40] x [100, 400] x [0.5, 3.5]
    ax1 = np.array([10.0, 20.0, 30.0, 40.0])
    ax2 = np.array([100.0, 200.0, 300.0, 400.0])
    ax3 = np.array([0.5, 1.5, 2.5, 3.5])

    basic_axes = (ax1, ax2, ax3)
    ndp = ndpolator.Ndpolator(basic_axes=basic_axes)

    # Create random grid values
    grid = np.random.rand(len(ax1), len(ax2), len(ax3), 1)
    ndp.register('test', associated_axes=None, grid=grid)

    # Test cases: [point, expected_squared_distance, description]
    # Note: distances are computed in GRID coordinate space, not physical space
    # ax1: [10, 20, 30, 40] → spacing = 10 → 5 physical units = 0.5 grid units
    # ax2: [100, 200, 300, 400] → spacing = 100 → 50 physical units = 0.5 grid units
    # ax3: [0.5, 1.5, 2.5, 3.5] → spacing = 1.0 → 0.5 physical units = 0.5 grid units
    test_cases = [
        # Inside grid (distance = 0)
        ([25.0, 250.0, 2.0], 0.0, "inside grid"),
        ([15.0, 150.0, 1.0], 0.0, "inside grid near lower bounds"),
        ([35.0, 350.0, 3.0], 0.0, "inside grid near upper bounds"),

        # On boundaries (distance = 0)
        ([10.0, 250.0, 2.0], 0.0, "on lower x boundary"),
        ([40.0, 250.0, 2.0], 0.0, "on upper x boundary"),
        ([25.0, 100.0, 2.0], 0.0, "on lower y boundary"),
        ([25.0, 400.0, 2.0], 0.0, "on upper y boundary"),
        ([25.0, 250.0, 0.5], 0.0, "on lower z boundary"),
        ([25.0, 250.0, 3.5], 0.0, "on upper z boundary"),
        ([10.0, 100.0, 0.5], 0.0, "on corner vertex"),
        ([40.0, 400.0, 3.5], 0.0, "on opposite corner vertex"),

        # Outside grid - distance to face (one dimension outside)
        # x: 45 is 5 physical units = 0.5 grid units beyond 40 → 0.5^2 = 0.25
        ([45.0, 250.0, 2.0], 0.25, "0.5 grid units beyond upper x"),
        # x: 5 is 5 physical units = 0.5 grid units before 10 → 0.5^2 = 0.25
        ([5.0, 250.0, 2.0], 0.25, "0.5 grid units before lower x"),
        # y: 450 is 50 physical units = 0.5 grid units beyond 400 → 0.5^2 = 0.25
        ([25.0, 450.0, 2.0], 0.25, "0.5 grid units beyond upper y"),
        # y: 50 is 50 physical units = 0.5 grid units before 100 → 0.5^2 = 0.25
        ([25.0, 50.0, 2.0], 0.25, "0.5 grid units before lower y"),
        # z: 4.0 is 0.5 physical units = 0.5 grid units beyond 3.5 → 0.5^2 = 0.25
        ([25.0, 250.0, 4.0], 0.25, "0.5 grid units beyond upper z"),
        # z: 0.0 is 0.5 physical units = 0.5 grid units before 0.5 → 0.5^2 = 0.25
        ([25.0, 250.0, 0.0], 0.25, "0.5 grid units before lower z"),

        # Outside grid - distance to edge (two dimensions outside)
        # x: 0.5 grid, y: 0.5 grid → 0.5^2 + 0.5^2 = 0.5
        ([45.0, 450.0, 2.0], 0.5, "beyond x and y"),
        ([5.0, 50.0, 2.0], 0.5, "before x and y"),
        # x: 0.5 grid, z: 0.5 grid → 0.5^2 + 0.5^2 = 0.5
        ([45.0, 250.0, 4.0], 0.5, "beyond x and z"),
        ([5.0, 250.0, 0.0], 0.5, "before x and z"),
        # y: 0.5 grid, z: 0.5 grid → 0.5^2 + 0.5^2 = 0.5
        ([25.0, 450.0, 4.0], 0.5, "beyond y and z"),
        ([25.0, 50.0, 0.0], 0.5, "before y and z"),

        # Outside grid - distance to vertex (all three dimensions outside)
        # x, y, z all 0.5 grid units → 0.5^2 + 0.5^2 + 0.5^2 = 0.75
        ([45.0, 450.0, 4.0], 0.75, "beyond all dims"),
        ([5.0, 50.0, 0.0], 0.75, "before all dims"),
        ([45.0, 50.0, 4.0], 0.75, "mixed beyond/before x+y+z"),
        ([5.0, 450.0, 0.0], 0.75, "mixed before/beyond x+y+z"),
    ]

    query_pts = np.array([tc[0] for tc in test_cases])

    # Compute distances
    computed_dists = ndp.distance('test', query_pts)

    # Verify all distances match expectations
    for i, (query, expected, desc) in enumerate(test_cases):
        computed = computed_dists[i, 0]
        assert np.isclose(computed, expected, atol=1e-9), \
            f"Test {i+1} failed ({desc}): query={query}, expected={expected}, computed={computed}"


if __name__ == '__main__':
    test_distances()
    print("All distance tests passed!")
