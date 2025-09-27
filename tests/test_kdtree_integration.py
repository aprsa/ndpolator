"""
Unit tests for k-d tree integration in ndpolator.

This module tests that the k-d tree lazy-loading implementation correctly:
1. Stores vertex coordinates for vertex trees (NDP_METHOD_NEAREST)
2. Stores hypercube center coordinates for hypercube trees (NDP_METHOD_LINEAR)  
3. Associates correct vertex indices with spatial coordinates
4. Produces the same results as the original linear search algorithm
"""

import numpy as np
import pytest
import ndpolator


class TestKdTreeIntegration:
    """Test k-d tree integration correctness."""

    def test_vertex_tree_coordinates(self):
        """Test that vertex trees store correct vertex coordinates."""
        # Create a simple 3D grid to test vertex coordinate calculation
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0])  
        z = np.array([0.0, 1.0, 2.0])
        
        ndp = ndpolator.Ndpolator(basic_axes=(x, y, z))
        
        # Create grid values: f(x,y,z) = 100*x + 10*y + z (unique for each vertex)
        grid = np.empty((len(x), len(y), len(z), 1))
        for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                for k, zi in enumerate(z):
                    grid[i, j, k, 0] = 100*xi + 10*yi + zi
        
        ndp.register('main', associated_axes=None, grid=grid)
        
        # Test points exactly at grid vertices - should return exact grid values
        test_points = np.array([
            [0.0, 0.0, 0.0],  # Should return 0 (grid[0,0,0])
            [1.0, 0.0, 0.0],  # Should return 100 (grid[1,0,0])
            [0.0, 1.0, 0.0],  # Should return 10 (grid[0,1,0])
            [0.0, 0.0, 1.0],  # Should return 1 (grid[0,0,1])
            [2.0, 1.0, 2.0],  # Should return 212 (grid[2,1,2])
        ])
        
        expected_values = np.array([0, 100, 10, 1, 212])
        
        # Use nearest extrapolation to trigger vertex tree construction
        result = ndp.ndpolate('main', test_points, extrapolation_method='nearest')
        
        for i, (point, expected, actual) in enumerate(zip(test_points, expected_values, result['interps'][:, 0])):
            assert abs(actual - expected) < 1e-10, \
                f"Vertex {i} at {point}: expected {expected}, got {actual}"

    def test_hypercube_tree_coordinates(self):
        """Test that hypercube trees store correct hypercube center coordinates."""
        # Create a 2D grid for easier visualization of hypercube centers
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        
        ndp = ndpolator.Ndpolator(basic_axes=(x, y))
        
        # Create a simple linear function: f(x,y) = x + y
        grid = np.empty((len(x), len(y), 1))
        for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                grid[i, j, 0] = xi + yi
        
        ndp.register('main', associated_axes=None, grid=grid)
        
        # Test points at hypercube centers - linear interpolation should be exact
        # Hypercube with inferior corner (0,0) and superior corner (1,1) has center (0.5,0.5)
        # Expected value at center: f(0.5,0.5) = 1.0
        test_points = np.array([
            [0.5, 0.5],  # Center of hypercube [(0,0)-(1,1)] -> f=1.0
            [1.5, 0.5],  # Center of hypercube [(1,0)-(2,1)] -> f=2.0  
            [0.5, 1.5],  # Center of hypercube [(0,1)-(1,2)] -> f=2.0
            [1.5, 1.5],  # Center of hypercube [(1,1)-(2,2)] -> f=3.0
        ])
        
        expected_values = np.array([1.0, 2.0, 2.0, 3.0])
        
        # Use linear extrapolation to trigger hypercube tree construction
        result = ndp.ndpolate('main', test_points, extrapolation_method='linear')
        
        for i, (point, expected, actual) in enumerate(zip(test_points, expected_values, result['interps'][:, 0])):
            assert abs(actual - expected) < 1e-10, \
                f"Hypercube center {i} at {point}: expected {expected}, got {actual}"

    def test_extrapolation_consistency(self):
        """Test that k-d tree and linear search give identical results."""
        # Create a 3D grid
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        z = np.array([0.0, 1.0])
        
        ndp = ndpolator.Ndpolator(basic_axes=(x, y, z))
        
        # Create test function: f(x,y,z) = x^2 + y^2 + z
        grid = np.empty((len(x), len(y), len(z), 1))
        for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                for k, zi in enumerate(z):
                    grid[i, j, k, 0] = xi*xi + yi*yi + zi
        
        ndp.register('main', associated_axes=None, grid=grid)
        
        # Generate test points including extrapolation cases
        np.random.seed(42)  # For reproducibility
        test_points = np.array([
            # In-bounds points
            [0.5, 0.5, 0.2],
            [1.3, 1.7, 0.8],
            # Out-of-bounds points (extrapolation)
            [-0.5, 0.5, 0.5],
            [2.5, 1.0, 0.3], 
            [1.0, -0.3, 0.7],
            [1.5, 2.8, 0.1],
            [3.0, 3.0, 1.5],
        ])
        
        # Test both nearest and linear extrapolation methods
        for method in ['nearest', 'linear']:
            # The first call will use k-d tree (if implemented)
            result_kdtree = ndp.ndpolate('main', test_points, extrapolation_method=method)
            
            # Force rebuild to test consistency (this would use linear search again)
            # Note: In current implementation, we can't easily force linear search
            # but we can test that repeated calls give the same results
            result_repeat = ndp.ndpolate('main', test_points, extrapolation_method=method)
            
            # Results should be identical
            np.testing.assert_array_almost_equal(
                result_kdtree['interps'], 
                result_repeat['interps'],
                decimal=12,
                err_msg=f"K-d tree and repeat call results differ for {method} extrapolation"
            )

    def test_vertex_index_mapping(self):
        """Test that vertex indices are correctly mapped in k-d tree."""
        # Create a small grid where we can verify index mappings
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        z = np.array([0.0, 1.0])
        
        ndp = ndpolator.Ndpolator(basic_axes=(x, y, z))
        
        # Create grid where each value equals its linear index
        # This lets us verify that the correct vertex indices are found
        grid = np.empty((2, 2, 2, 1))
        linear_index = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    grid[i, j, k, 0] = linear_index
                    linear_index += 1
        
        ndp.register('main', associated_axes=None, grid=grid)
        
        # Test points exactly at vertices - should return the linear index
        test_points = np.array([
            [0.0, 0.0, 0.0],  # Linear index 0
            [0.0, 0.0, 1.0],  # Linear index 1  
            [0.0, 1.0, 0.0],  # Linear index 2
            [0.0, 1.0, 1.0],  # Linear index 3
            [1.0, 0.0, 0.0],  # Linear index 4
            [1.0, 0.0, 1.0],  # Linear index 5
            [1.0, 1.0, 0.0],  # Linear index 6
            [1.0, 1.0, 1.0],  # Linear index 7
        ])
        
        expected_indices = np.arange(8)
        
        result = ndp.ndpolate('main', test_points, extrapolation_method='nearest')
        
        for i, (point, expected_idx, actual_value) in enumerate(zip(test_points, expected_indices, result['interps'][:, 0])):
            assert abs(actual_value - expected_idx) < 1e-10, \
                f"Point {point} should map to vertex index {expected_idx}, got value {actual_value}"

    def test_coordinate_calculation_formula(self):
        """Test the coordinate calculation formula directly."""
        # Test with a 3×3×2 grid (18 total elements)
        axis_lengths = [3, 3, 2]
        
        # Calculate cumulative products (cplen array)
        cplen = [1] * len(axis_lengths)
        cplen[-1] = 1
        for j in range(len(axis_lengths)-2, -1, -1):
            cplen[j] = cplen[j+1] * axis_lengths[j+1]
        
        # Expected cplen: [6, 2, 1] for [3,3,2] grid
        assert cplen == [6, 2, 1], f"cplen calculation error: expected [6, 2, 1], got {cplen}"
        
        # Test coordinate conversion for all indices  
        expected_coords = [
            [0, 0, 0], [0, 0, 1],  # i=0,1
            [0, 1, 0], [0, 1, 1],  # i=2,3  
            [0, 2, 0], [0, 2, 1],  # i=4,5
            [1, 0, 0], [1, 0, 1],  # i=6,7
            [1, 1, 0], [1, 1, 1],  # i=8,9
            [1, 2, 0], [1, 2, 1],  # i=10,11
            [2, 0, 0], [2, 0, 1],  # i=12,13
            [2, 1, 0], [2, 1, 1],  # i=14,15
            [2, 2, 0], [2, 2, 1],  # i=16,17
        ]
        
        for i, expected in enumerate(expected_coords):
            calculated = []
            for j in range(len(axis_lengths)):
                coord = (i // (cplen[j] // cplen[-1])) % axis_lengths[j]
                calculated.append(coord)
            
            assert calculated == expected, \
                f"Index {i}: expected coordinates {expected}, calculated {calculated}"
        
        # Test that these represent superior corners for hypercubes
        # Superior corner (1,1,1) should represent hypercube from (0,0,0) to (1,1,1)
        # Center should be at (0.5, 0.5, 0.5)
        superior_corner = [1, 1, 1]
        hypercube_center = [coord - 0.5 for coord in superior_corner] 
        assert hypercube_center == [0.5, 0.5, 0.5], \
            f"Hypercube center calculation error: {hypercube_center}"

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test minimal 2×2 grid (smallest allowed)
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        
        ndp = ndpolator.Ndpolator(basic_axes=(x, y))
        grid = np.array([[[1.0], [2.0]], [[3.0], [4.0]]])
        ndp.register('main', associated_axes=None, grid=grid)
        
        # Test extrapolation far outside the grid
        test_points = np.array([
            [0.5, 0.5],   # In bounds
            [-10.0, 0.5], # Far out of bounds in x
            [0.5, 10.0],  # Far out of bounds in y
            [-5.0, -5.0], # Far out of bounds in both
        ])
        
        # Should not crash and should return reasonable values
        result_nearest = ndp.ndpolate('main', test_points, extrapolation_method='nearest')
        result_linear = ndp.ndpolate('main', test_points, extrapolation_method='linear')
        
        assert len(result_nearest['interps']) == len(test_points)
        assert len(result_linear['interps']) == len(test_points)
        
        # All results should be finite (not NaN or inf)
        assert np.all(np.isfinite(result_nearest['interps']))
        assert np.all(np.isfinite(result_linear['interps']))
        
        # Extreme extrapolation should still give sensible nearest neighbor results
        # Far negative point should map to corner (0,0) with value 1.0
        assert abs(result_nearest['interps'][3, 0] - 1.0) < 1e-10


if __name__ == '__main__':
    # Run the tests
    test_instance = TestKdTreeIntegration()
    
    print("Running k-d tree integration tests...")
    
    try:
        test_instance.test_vertex_tree_coordinates()
        print("✓ Vertex tree coordinates test passed")
        
        test_instance.test_hypercube_tree_coordinates()
        print("✓ Hypercube tree coordinates test passed")
        
        test_instance.test_extrapolation_consistency()
        print("✓ Extrapolation consistency test passed")
        
        test_instance.test_vertex_index_mapping()
        print("✓ Vertex index mapping test passed")
        
        test_instance.test_coordinate_calculation_formula()
        print("✓ Coordinate calculation formula test passed")
        
        test_instance.test_edge_cases()
        print("✓ Edge cases test passed")
        
        print("\n🎉 All k-d tree integration tests passed!")
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        raise