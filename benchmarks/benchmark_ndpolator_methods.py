#!/usr/bin/env python3

import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import ndpolator

def benchmark_extrapolation_methods():
    """Compare performance of interpolation vs nearest vs linear extrapolation methods."""
    
    print("Benchmarking interpolation vs nearest vs linear extrapolation methods")
    print("=" * 60)
    
    # Test different grid sizes
    grid_sizes = [10, 20, 30, 40, 50]
    dimensions = [2, 3, 4]
    n_queries = 1000
    
    results = {
        'nearest': {'2d': [], '3d': [], '4d': []},
        'linear': {'2d': [], '3d': [], '4d': []},
        'interpolation': {'2d': [], '3d': [], '4d': []}
    }
    
    for dim in dimensions:
        print(f"\n{dim}D Benchmarks:")
        print("-" * 30)
        
        for grid_size in grid_sizes:
            print(f"Grid size: {grid_size}^{dim} ({grid_size**dim:,} points)")
            
            # Create axes
            axes = tuple(np.linspace(0, 100, grid_size) for _ in range(dim))
            
            # Create ndpolator
            ndp = ndpolator.Ndpolator(basic_axes=axes)
            
            # Create synthetic data
            def test_function(*coords):
                return sum(coord**2 for coord in coords)
            
            # Generate grid
            if dim == 2:
                grid = np.empty((grid_size, grid_size, 1))
                for i, x in enumerate(axes[0]):
                    for j, y in enumerate(axes[1]):
                        grid[i, j, 0] = test_function(x, y)
            elif dim == 3:
                grid = np.empty((grid_size, grid_size, grid_size, 1))
                for i, x in enumerate(axes[0]):
                    for j, y in enumerate(axes[1]):
                        for k, z in enumerate(axes[2]):
                            grid[i, j, k, 0] = test_function(x, y, z)
            elif dim == 4:
                grid = np.empty((grid_size, grid_size, grid_size, grid_size, 1))
                for i, x in enumerate(axes[0]):
                    for j, y in enumerate(axes[1]):
                        for k, z in enumerate(axes[2]):
                            for l, w in enumerate(axes[3]):
                                grid[i, j, k, l, 0] = test_function(x, y, z, w)
            
            ndp.register('test', associated_axes=None, grid=grid)
            
            # Generate random query points for interpolation (inside grid)
            interp_query_pts = []
            for _ in range(n_queries):
                point = []
                for axis in axes:
                    min_val, max_val = axis[0], axis[-1]
                    # Inside grid only
                    coord = np.random.uniform(min_val, max_val)
                    point.append(coord)
                interp_query_pts.append(point)
            interp_query_pts = np.array(interp_query_pts)
            
            # Generate random query points for extrapolation (including outside points)
            extrap_query_pts = []
            for _ in range(n_queries):
                point = []
                for axis in axes:
                    # Generate points both inside and outside the grid
                    min_val, max_val = axis[0], axis[-1]
                    range_val = max_val - min_val
                    # 50% inside, 50% outside (extrapolation)
                    if np.random.random() < 0.5:
                        # Inside grid
                        coord = np.random.uniform(min_val, max_val)
                    else:
                        # Outside grid (extrapolation)
                        if np.random.random() < 0.5:
                            # Below minimum
                            coord = np.random.uniform(min_val - range_val * 0.3, min_val)
                        else:
                            # Above maximum
                            coord = np.random.uniform(max_val, max_val + range_val * 0.3)
                    point.append(coord)
                extrap_query_pts.append(point)
            extrap_query_pts = np.array(extrap_query_pts)
            
            # Benchmark interpolation (inside grid points only)
            start_time = time.time()
            result_interp = ndp.ndpolate('test', interp_query_pts)
            interp_time = time.time() - start_time
            
            # Benchmark nearest extrapolation
            start_time = time.time()
            result_nearest = ndp.ndpolate('test', extrap_query_pts, extrapolation_method='nearest')
            nearest_time = time.time() - start_time
            
            # Benchmark linear extrapolation
            start_time = time.time()
            result_linear = ndp.ndpolate('test', extrap_query_pts, extrapolation_method='linear')
            linear_time = time.time() - start_time
            
            # Store results
            dim_key = f'{dim}d'
            results['interpolation'][dim_key].append(interp_time)
            results['nearest'][dim_key].append(nearest_time)
            results['linear'][dim_key].append(linear_time)
            
            print(f"  Interpolation: {interp_time:.4f}s ({interp_time/n_queries*1000:.2f}ms per query)")
            print(f"  Nearest:       {nearest_time:.4f}s ({nearest_time/n_queries*1000:.2f}ms per query)")
            print(f"  Linear:        {linear_time:.4f}s ({linear_time/n_queries*1000:.2f}ms per query)")
            print(f"  Ratio (Linear/Nearest): {linear_time/nearest_time:.2f}x")
            print(f"  Ratio (Nearest/Interp): {nearest_time/interp_time:.2f}x")
            print(f"  Ratio (Linear/Interp): {linear_time/interp_time:.2f}x")
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, dim in enumerate(dimensions):
        ax = axes[i]
        dim_key = f'{dim}d'
        
        grid_points = [size**dim for size in grid_sizes]
        
        ax.loglog(grid_points, results['interpolation'][dim_key], 'go-', label='Interpolation', linewidth=2, markersize=8)
        ax.loglog(grid_points, results['nearest'][dim_key], 'bo-', label='Nearest', linewidth=2, markersize=8)
        ax.loglog(grid_points, results['linear'][dim_key], 'ro-', label='Linear', linewidth=2, markersize=8)
        
        ax.set_xlabel('Grid Points')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'{dim}D Extrapolation Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add performance ratio annotations
        for j, (size, grid_pts) in enumerate(zip(grid_sizes, grid_points)):
            ratio = results['linear'][dim_key][j] / results['nearest'][dim_key][j]
            ax.annotate(f'{ratio:.1f}x', 
                       xy=(grid_pts, results['linear'][dim_key][j]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('extrapolation_methods_benchmark.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for dim in dimensions:
        dim_key = f'{dim}d'
        interp_times = results['interpolation'][dim_key]
        nearest_times = results['nearest'][dim_key]
        linear_times = results['linear'][dim_key]
        ratios_ln = [linear/nearest for linear, nearest in zip(linear_times, nearest_times)]
        ratios_ni = [nearest/interp for nearest, interp in zip(nearest_times, interp_times)]
        ratios_li = [linear/interp for linear, interp in zip(linear_times, interp_times)]
        
        print(f"\n{dim}D Results:")
        print(f"  Average Linear/Nearest ratio: {np.mean(ratios_ln):.2f}x")
        print(f"  Average Nearest/Interp ratio: {np.mean(ratios_ni):.2f}x")
        print(f"  Average Linear/Interp ratio: {np.mean(ratios_li):.2f}x")
        
        # Performance times
        fastest_interp = np.min(interp_times)
        fastest_nearest = np.min(nearest_times)
        fastest_linear = np.min(linear_times)
        print(f"  Best interpolation time: {fastest_interp:.4f}s")
        print(f"  Best nearest time: {fastest_nearest:.4f}s")
        print(f"  Best linear time: {fastest_linear:.4f}s")

if __name__ == "__main__":
    benchmark_extrapolation_methods()