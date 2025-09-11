"""
Test suite for ndpolator extrapolation functionality.

This module provides comprehensive tests for the three extrapolation methods
available in ndpolator: 'none', 'nearest', and 'linear'.
"""

import numpy as np
import pytest
import ndpolator


def test_extrapolation_methods():
    """Test all three extrapolation methods with a simple 2D grid."""
    # Create a simple 2D grid
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    
    # Create ndpolator instance
    ndp = ndpolator.Ndpolator(basic_axes=(x, y))
    
    # Create grid values: f(x,y) = x + 2*y
    grid = np.empty((len(x), len(y), 1))
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            grid[i, j, 0] = xi + 2*yj
    
    ndp.register('main', associated_axes=None, grid=grid)
    
    # Test points: mix of in-bounds and out-of-bounds
    query_points = np.array([
        [0.5, 0.5],    # in-bounds
        [1.5, 1.5],    # in-bounds
        [-0.3, 0.2],   # out-of-bounds (x < 0), nearest to (0,0)
        [2.3, 1.8],    # out-of-bounds (x > 2), nearest to (2,2)
        [1.2, -0.3],   # out-of-bounds (y < 0), nearest to (1,0)
        [0.8, 2.3],    # out-of-bounds (y > 2), nearest to (1,2)
        [-0.2, -0.3],  # out-of-bounds (both coordinates), nearest to (0,0)
        [2.3, 2.4],    # out-of-bounds (both coordinates), nearest to (2,2)
    ])
    
    # Test 'none' extrapolation method
    result_none = ndp.ndpolate('main', query_points, extrapolation_method='none')
    
    # In-bounds points should interpolate normally
    assert abs(result_none['interps'][0, 0] - 1.5) < 1e-10  # f(0.5, 0.5) = 0.5 + 2*0.5 = 1.5
    assert abs(result_none['interps'][1, 0] - 4.5) < 1e-10  # f(1.5, 1.5) = 1.5 + 2*1.5 = 4.5
    
    # Out-of-bounds points should be NaN
    assert np.isnan(result_none['interps'][2, 0])  # (-0.3, 0.2)
    assert np.isnan(result_none['interps'][3, 0])  # (2.3, 1.8)
    assert np.isnan(result_none['interps'][4, 0])  # (1.2, -0.3)
    assert np.isnan(result_none['interps'][5, 0])  # (0.8, 2.3)
    assert np.isnan(result_none['interps'][6, 0])  # (-0.2, -0.3)
    assert np.isnan(result_none['interps'][7, 0])  # (2.3, 2.4)
    
    # Test 'nearest' extrapolation method
    result_nearest = ndp.ndpolate('main', query_points, extrapolation_method='nearest')
    
    # In-bounds points should interpolate normally (same as 'none')
    assert abs(result_nearest['interps'][0, 0] - 1.5) < 1e-10
    assert abs(result_nearest['interps'][1, 0] - 4.5) < 1e-10
    
    # Out-of-bounds points should use nearest grid point values
    # For (-0.3, 0.2): nearest is (0.0, 0.0) -> f(0.0, 0.0) = 0.0
    assert abs(result_nearest['interps'][2, 0] - 0.0) < 1e-10
    
    # For (2.3, 1.8): nearest is (2.0, 2.0) -> f(2.0, 2.0) = 6.0
    assert abs(result_nearest['interps'][3, 0] - 6.0) < 1e-10
    
    # For (1.2, -0.3): nearest is (1.0, 0.0) -> f(1.0, 0.0) = 1.0
    assert abs(result_nearest['interps'][4, 0] - 1.0) < 1e-10
    
    # For (0.8, 2.3): nearest is (1.0, 2.0) -> f(1.0, 2.0) = 5.0
    assert abs(result_nearest['interps'][5, 0] - 5.0) < 1e-10
    
    # Test 'linear' extrapolation method
    result_linear = ndp.ndpolate('main', query_points, extrapolation_method='linear')
    
    # In-bounds points should interpolate normally (same as others)
    assert abs(result_linear['interps'][0, 0] - 1.5) < 1e-10
    assert abs(result_linear['interps'][1, 0] - 4.5) < 1e-10
    
    # Out-of-bounds points should use linear extrapolation
    # The linear function is f(x,y) = x + 2*y, so it should work perfectly
    assert abs(result_linear['interps'][2, 0] - (-0.3 + 2*0.2)) < 1e-10  # f(-0.3, 0.2) = 0.1
    assert abs(result_linear['interps'][3, 0] - (2.3 + 2*1.8)) < 1e-10   # f(2.3, 1.8) = 5.9
    assert abs(result_linear['interps'][4, 0] - (1.2 + 2*(-0.3))) < 1e-10  # f(1.2, -0.3) = 0.6
    assert abs(result_linear['interps'][5, 0] - (0.8 + 2*2.3)) < 1e-10   # f(0.8, 2.3) = 5.4
    assert abs(result_linear['interps'][6, 0] - (-0.2 + 2*(-0.3))) < 1e-10  # f(-0.2, -0.3) = -0.8
    assert abs(result_linear['interps'][7, 0] - (2.3 + 2*2.4)) < 1e-10   # f(2.3, 2.4) = 7.1


def test_extrapolation_edge_cases():
    """Test extrapolation behavior in edge cases and special scenarios."""
    # Create a 1D case for simpler testing
    x = np.array([0.0, 1.0, 2.0])
    
    ndp = ndpolator.Ndpolator(basic_axes=(x,))
    
    # Create grid values: f(x) = 10*x + 10
    grid = np.empty((len(x), 1))
    for i, xi in enumerate(x):
        grid[i, 0] = 10*xi + 10
    
    ndp.register('main', associated_axes=None, grid=grid)
    
    # Test extrapolation at grid boundaries
    boundary_points = np.array([[0.0], [1.0], [2.0]])
    
    for method in ['none', 'nearest', 'linear']:
        result = ndp.ndpolate('main', boundary_points, extrapolation_method=method)
        # All methods should give exact values at grid points
        assert abs(result['interps'][0, 0] - 10.0) < 1e-10
        assert abs(result['interps'][1, 0] - 20.0) < 1e-10
        assert abs(result['interps'][2, 0] - 30.0) < 1e-10
    
    # Test far extrapolation
    far_points = np.array([[-1.0], [3.0]])
    
    # 'none' method
    result_none = ndp.ndpolate('main', far_points, extrapolation_method='none')
    assert np.isnan(result_none['interps'][0, 0])
    assert np.isnan(result_none['interps'][1, 0])
    
    # 'nearest' method
    result_nearest = ndp.ndpolate('main', far_points, extrapolation_method='nearest')
    assert abs(result_nearest['interps'][0, 0] - 10.0) < 1e-10  # nearest to x[0]
    assert abs(result_nearest['interps'][1, 0] - 30.0) < 1e-10  # nearest to x[2]
    
    # 'linear' method
    result_linear = ndp.ndpolate('main', far_points, extrapolation_method='linear')
    assert abs(result_linear['interps'][0, 0] - 0.0) < 1e-10   # f(-1) = 10*(-1) + 10 = 0
    assert abs(result_linear['interps'][1, 0] - 40.0) < 1e-10  # f(3) = 10*3 + 10 = 40


def test_extrapolation_single_points():
    """Test extrapolation with single query points."""
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    
    ndp = ndpolator.Ndpolator(basic_axes=(x, y))
    
    grid = np.array([[[1.0], [2.0]],
                     [[3.0], [4.0]]])
    
    ndp.register('main', associated_axes=None, grid=grid)
    
    # Test single out-of-bounds point
    single_point = np.array([[-0.5, 0.5]])
    
    result_none = ndp.ndpolate('main', single_point, extrapolation_method='none')
    assert result_none['interps'].shape == (1, 1)
    assert np.isnan(result_none['interps'][0, 0])
    
    result_nearest = ndp.ndpolate('main', single_point, extrapolation_method='nearest')
    assert result_nearest['interps'].shape == (1, 1)
    assert not np.isnan(result_nearest['interps'][0, 0])
    
    result_linear = ndp.ndpolate('main', single_point, extrapolation_method='linear')
    assert result_linear['interps'].shape == (1, 1)
    assert not np.isnan(result_linear['interps'][0, 0])


def test_extrapolation_method_validation():
    """Test that invalid extrapolation methods raise appropriate errors."""
    x = np.array([0.0, 1.0])
    ndp = ndpolator.Ndpolator(basic_axes=(x,))
    
    grid = np.array([[1.0], [2.0]])
    ndp.register('main', associated_axes=None, grid=grid)
    
    query_point = np.array([[0.5]])
    
    # Valid methods should work
    for method in ['none', 'nearest', 'linear']:
        result = ndp.ndpolate('main', query_point, extrapolation_method=method)
        assert result['interps'].shape == (1, 1)
        assert not np.isnan(result['interps'][0, 0])
    
    # Invalid method should raise an error
    with pytest.raises((ValueError, RuntimeError)):
        ndp.ndpolate('main', query_point, extrapolation_method='invalid_method')
