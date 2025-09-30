# Python API Tutorial {#python_tutorial}

[TOC]

This tutorial shows you how to use the ndpolator Python API for n-dimensional interpolation.

## Basic 2D Example {#basic_example}

Here's a simple 2D interpolation example:

```python
import numpy as np
from ndpolator import Ndpolator

# Step 1: Define your grid axes
x = np.linspace(0, 10, 11)  # [0, 1, 2, ..., 10]
y = np.linspace(0, 5, 6)    # [0, 1, 2, 3, 4, 5]

# Step 2: Create the ndpolator instance
ndp = Ndpolator(basic_axes=(x, y))

# Step 3: Define function values on your grid
# Grid shape: (len(x), len(y), num_function_values)
values = np.sin(x[:, None]) * np.cos(y[None, :])
grid = values.reshape(11, 6, 1)  # Reshape to add function value dimension

# Step 4: Register the interpolation grid
ndp.register(name='my_function', associated_axes=None, grid=grid)

# Step 5: Define points where you want interpolated values
query_points = np.array([[2.5, 1.5],    # Point inside the grid
                         [5.0, 3.0],     # Point exactly on a grid vertex
                         [0.0, 0.0]])    # Point at grid boundary

# Step 6: Perform interpolation
result = ndp.ndpolate('my_function', query_points)
print("Interpolated values:", result['interps'])
```

### Nearest Extrapolation {#extrapolation_example}

ndpolator can also extrapolate beyond the grid boundaries:

```python
# Query points outside the grid
query_points = np.array([[15.0, 2.0],   # Beyond x-axis
                         [5.0, 10.0],   # Beyond y-axis  
                         [-1.0, -1.0]]) # Beyond both axes

# Extrapolate using nearest neighbor method
result = ndp.ndpolate('my_function', query_points, 
                     extrapolation_method='nearest')

print("Extrapolated values:", result['interps'])
print("Distances to nearest points:", result['dists'])
```

### Linear Extrapolation {#linear_extrapolation_example}

```python
# Extrapolate using linear extrapolation
result = ndp.ndpolate('my_function', query_points, 
                     extrapolation_method='linear')

print("Extrapolated values:", result['interps'])
print("Distances to nearest points:", result['dists'])
```

## Advanced Usage {#advanced_usage}

### Working with Associated Axes {#associated_axes}

For more complex grids with both basic and associated dimensions:

```python
# Basic axes define the sparse grid structure
basic_axes = (np.linspace(0, 1, 5),      # Temperature
              np.linspace(0, 1, 4))      # Pressure

# Associated axes are guaranteed to have values wherever basic axes do
associated_axes = (np.linspace(0, 2*np.pi, 8),)  # Angle

ndp = Ndpolator(basic_axes)

# Grid shape: (basic_temp, basic_pressure, associated_angle, function_values)
grid = np.random.random((5, 4, 8, 2))  # 2 function values per point

ndp.register('complex_function', associated_axes, grid)

# Query all dimensions: (temperature, pressure, angle)
query_pts = np.array([[0.5, 0.5, np.pi],
                      [0.8, 0.2, np.pi/2]])

result = ndp.ndpolate('complex_function', query_pts)
```

### Multiple Interpolation Tables {#multiple_tables}

You can register multiple functions on the same axes:

```python
# Register multiple functions
ndp.register('temperature', None, temp_grid)
ndp.register('pressure', None, pressure_grid)
ndp.register('density', None, density_grid)

# Interpolate each function separately
temp_result = ndp.ndpolate('temperature', query_points)
pressure_result = ndp.ndpolate('pressure', query_points)
density_result = ndp.ndpolate('density', query_points)

# Check what tables are registered
print("Available tables:", ndp.tables)
```

## Sparse Grids {#sparse_grids}

Ndpolator handles sparse grids where some vertices contain NaN values. This is useful for irregular data or when some combinations of parameters are invalid:

```python
import numpy as np
from ndpolator import Ndpolator

# Create a 3x4 grid
x = np.array([1.0, 2.0, 3.0])
y = np.array([10.0, 20.0, 30.0, 40.0])
ndp = Ndpolator((x, y))

# Create a function grid with some NaN values (sparse grid)
# Shape: (3, 4, 1) for (len(x), len(y), function_values)
grid = np.array([
    [[1.0], [2.0], [np.nan], [4.0]],      # x=1.0: y=30.0 is undefined
    [[5.0], [np.nan], [7.0], [8.0]],      # x=2.0: y=20.0 is undefined
    [[np.nan], [10.0], [11.0], [12.0]]    # x=3.0: y=10.0 is undefined
])

ndp.register('sparse_function', None, grid)

# Query points - some will require interpolation around NaN values
query_points = np.array([
    [2.5, 35.0],   # Between defined points
    [1.5, 35.0],   # One vertex is NaN
    [1.0, 30.0],   # Exactly at NaN point
    [1.5, 15.0],   # Three vertices are NaN
    [3.5, 45.0]    # Off-grid entirely
])

# Interpolate first, with no imputing/extrapolation:
result = ndp.ndpolate('sparse_function', query_points)
print("Interpolated values:", result['interps'])

# For points that can't be interpolated due to insufficient defined neighbors,
# use extrapolation:
result = ndp.ndpolate('sparse_function', query_points, extrapolation_method='nearest')
print("With nearest extrapolation:", result['interps'])

result = ndp.ndpolate('sparse_function', query_points, extrapolation_method='linear')
print("With linear extrapolation:", result['interps'])
```

## Performance Tips {#performance_tips}

### Search Algorithms {#search_algorithms}

Choose the right search algorithm for your data size:

```python
# For large grids (>1000 points): use k-d tree (default)
result = ndp.ndpolate('my_function', query_points, 
                     search_algorithm='kdtree')

# For small grids (<100 points): linear search may be faster
result = ndp.ndpolate('my_function', query_points, 
                     search_algorithm='linear')
```

## Common Errors and Solutions {#common_errors}

### Grid Shape Errors {#shape_errors}

Make sure your grid dimensions match your axes:

```python
# WRONG: Grid shape doesn't match axes
x = np.linspace(0, 1, 5)  # 5 points
y = np.linspace(0, 1, 4)  # 4 points  
grid = np.random.random((3, 3, 1))  # 3x3 grid - MISMATCH!

# CORRECT: Grid shape matches axes
grid = np.random.random((5, 4, 1))  # 5x4 grid matches axes
```

### Extrapolation Errors {#extrapolation_errors}

Handle out-of-bounds queries appropriately:

```python
# This will return NaN for out-of-bounds points
result = ndp.ndpolate('my_function', query_points, 
                     extrapolation_method='none')

# This will extrapolate using nearest neighbors
result = ndp.ndpolate('my_function', query_points, 
                     extrapolation_method='nearest')
```

## See Also {#see_also}

- [Ndpolator Class Reference](\ref ndpolator::ndpolator::Ndpolator)
- [Main API Functions](\ref main_api)  
- [Data Structures](\ref data_structures)
