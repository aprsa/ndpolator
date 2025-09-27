#!/usr/bin/env python3
"""
Benchmark script to profile the speed-up achieved by k-d tree implementation
compared to linear search for nearest neighbor extrapolation.
"""

import numpy as np
import ndpolator
import time
import matplotlib.pyplot as plt
from collections import defaultdict


def create_test_grid(shape, func=None):
    """Create a test grid with specified shape."""
    if func is None:
        # Default function: sum of coordinates
        def func(coords):
            return sum(coords)
    
    # Create axes
    axes = tuple(np.linspace(0, s-1, s) for s in shape)
    ndp = ndpolator.Ndpolator(basic_axes=axes)
    
    # Create grid
    grid = np.zeros(shape + (1,))
    for idx in np.ndindex(shape):
        coords = [axes[i][idx[i]] for i in range(len(shape))]
        grid[idx + (0,)] = func(coords)
    
    ndp.register('main', associated_axes=None, grid=grid)
    return ndp, axes

def generate_random_queries(axes, n_queries, extrapolation_factor=0.3):
    """Generate random query points, including extrapolation points."""
    n_dims = len(axes)
    queries = np.zeros((n_queries, n_dims))
    
    for i in range(n_dims):
        axis_min, axis_max = axes[i][0], axes[i][-1]
        axis_range = axis_max - axis_min
        
        # Extend range for extrapolation
        extended_min = axis_min - extrapolation_factor * axis_range
        extended_max = axis_max + extrapolation_factor * axis_range
        
        queries[:, i] = np.random.uniform(extended_min, extended_max, n_queries)
    
    return queries

def benchmark_search_methods(grid_shapes, n_queries_list, n_runs=3):
    """Benchmark both search methods across different grid sizes and query counts."""
    results = defaultdict(list)
    
    print("Benchmarking k-d tree vs linear search performance...")
    print("=" * 60)
    
    for shape in grid_shapes:
        n_vertices = np.prod(shape)
        print(f"\nGrid shape: {shape} ({n_vertices:,} vertices)")
        
        # Create test grid
        ndp, axes = create_test_grid(shape)
        
        for n_queries in n_queries_list:
            print(f"  Testing {n_queries:,} queries...")
            
            # Generate queries
            queries = generate_random_queries(axes, n_queries)
            
            # Benchmark linear search
            linear_times = []
            for run in range(n_runs):
                start_time = time.time()
                result_linear = ndp.ndpolate('main', queries, 
                                           extrapolation_method='nearest', 
                                           search_algorithm='linear')
                linear_times.append(time.time() - start_time)
            
            # Benchmark k-d tree search
            kdtree_times = []
            for run in range(n_runs):
                start_time = time.time()
                result_kdtree = ndp.ndpolate('main', queries, 
                                           extrapolation_method='nearest', 
                                           search_algorithm='kdtree')
                kdtree_times.append(time.time() - start_time)
            
            # Verify results match (at least for a subset)
            test_subset = min(100, n_queries)
            linear_subset = result_linear['interps'][:test_subset]
            kdtree_subset = result_kdtree['interps'][:test_subset]
            
            if not np.allclose(linear_subset, kdtree_subset, rtol=1e-10):
                print("    WARNING: Results don't match!")
            
            # Calculate statistics
            linear_avg = np.mean(linear_times)
            kdtree_avg = np.mean(kdtree_times)
            speedup = linear_avg / kdtree_avg
            
            results['grid_shape'].append(shape)
            results['n_vertices'].append(n_vertices)
            results['n_queries'].append(n_queries)
            results['linear_time'].append(linear_avg)
            results['kdtree_time'].append(kdtree_avg)
            results['speedup'].append(speedup)
            
            print(f"    Linear:  {linear_avg:.4f}s ± {np.std(linear_times):.4f}s")
            print(f"    K-d tree: {kdtree_avg:.4f}s ± {np.std(kdtree_times):.4f}s")
            print(f"    Speedup: {speedup:.2f}x")
    
    return dict(results)

def plot_speedup_results(results):
    """Create visualization plots of the speedup results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('K-d Tree vs Linear Search Performance Comparison', fontsize=16)
    
    # Convert to numpy arrays for easier plotting
    n_vertices = np.array(results['n_vertices'])
    n_queries = np.array(results['n_queries'])
    linear_times = np.array(results['linear_time'])
    kdtree_times = np.array(results['kdtree_time'])
    speedups = np.array(results['speedup'])
    
    # Plot 1: Speedup vs Grid Size
    ax1 = axes[0, 0]
    unique_vertices = np.unique(n_vertices)
    for n_v in unique_vertices:
        mask = n_vertices == n_v
        ax1.plot(n_queries[mask], speedups[mask], 'o-', label=f'{n_v:,} vertices')
    ax1.set_xlabel('Number of Queries')
    ax1.set_ylabel('Speedup (x)')
    ax1.set_title('Speedup vs Number of Queries')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot 2: Absolute Times
    ax2 = axes[0, 1]
    unique_queries = np.unique(n_queries)
    for n_q in unique_queries:
        mask = n_queries == n_q
        ax2.loglog(n_vertices[mask], linear_times[mask], 'o-', label=f'Linear ({n_q:,} queries)')
        ax2.loglog(n_vertices[mask], kdtree_times[mask], 's--', label=f'K-d tree ({n_q:,} queries)')
    ax2.set_xlabel('Number of Vertices')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Absolute Performance: Linear vs K-d Tree')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Speedup vs Grid Size (for largest query set)
    ax3 = axes[1, 0]
    max_queries = np.max(n_queries)
    mask = n_queries == max_queries
    ax3.semilogx(n_vertices[mask], speedups[mask], 'ro-', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Vertices')
    ax3.set_ylabel('Speedup (x)')
    ax3.set_title(f'Speedup vs Grid Size ({max_queries:,} queries)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Time per Query
    ax4 = axes[1, 1]
    time_per_query_linear = linear_times / n_queries * 1000  # ms per query
    time_per_query_kdtree = kdtree_times / n_queries * 1000
    
    for n_v in unique_vertices:
        mask = n_vertices == n_v
        ax4.loglog(n_queries[mask], time_per_query_linear[mask], 'o-', label=f'Linear ({n_v:,} vertices)')
        ax4.loglog(n_queries[mask], time_per_query_kdtree[mask], 's--', label=f'K-d tree ({n_v:,} vertices)')
    
    ax4.set_xlabel('Number of Queries')
    ax4.set_ylabel('Time per Query (ms)')
    ax4.set_title('Time per Query Scaling')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def complexity_analysis(results):
    """Analyze the computational complexity of both methods."""
    print("\n" + "=" * 60)
    print("COMPLEXITY ANALYSIS")
    print("=" * 60)
    
    n_vertices = np.array(results['n_vertices'])
    n_queries = np.array(results['n_queries'])
    linear_times = np.array(results['linear_time'])
    kdtree_times = np.array(results['kdtree_time'])
    
    # Analyze scaling with number of vertices (for fixed number of queries)
    fixed_queries = 1000
    mask = n_queries == fixed_queries
    
    if np.sum(mask) > 1:
        vertices_subset = n_vertices[mask]
        linear_subset = linear_times[mask]
        kdtree_subset = kdtree_times[mask]
        
        # Fit power laws: time = a * vertices^b
        linear_log_fit = np.polyfit(np.log(vertices_subset), np.log(linear_subset), 1)
        kdtree_log_fit = np.polyfit(np.log(vertices_subset), np.log(kdtree_subset), 1)
        
        print(f"Scaling with grid size ({fixed_queries} queries):")
        print(f"  Linear search: O(N^{linear_log_fit[0]:.2f}) - Expected: O(N)")
        print(f"  K-d tree:      O(N^{kdtree_log_fit[0]:.2f}) - Expected: O(log N)")
    
    # Analyze scaling with number of queries (for fixed grid size)
    largest_grid_vertices = np.max(n_vertices)
    mask = n_vertices == largest_grid_vertices
    
    if np.sum(mask) > 1:
        queries_subset = n_queries[mask]
        linear_subset = linear_times[mask]
        kdtree_subset = kdtree_times[mask]
        
        # Fit linear relationships: time = a * queries + b
        linear_fit = np.polyfit(queries_subset, linear_subset, 1)
        kdtree_fit = np.polyfit(queries_subset, kdtree_subset, 1)
        
        print(f"\nScaling with number of queries ({largest_grid_vertices:,} vertices):")
        print(f"  Linear search: {linear_fit[0]*1000:.4f} ms per query")
        print(f"  K-d tree:      {kdtree_fit[0]*1000:.4f} ms per query")
        print(f"  Per-query speedup: {linear_fit[0]/kdtree_fit[0]:.2f}x")

def print_summary(results):
    """Print a summary of the performance results."""
    speedups = np.array(results['speedup'])
    
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Total benchmarks run: {len(speedups)}")
    print(f"Average speedup: {np.mean(speedups):.2f}x")
    print(f"Median speedup: {np.median(speedups):.2f}x")
    print(f"Best speedup: {np.max(speedups):.2f}x")
    print(f"Worst speedup: {np.min(speedups):.2f}x")
    
    # Find best and worst cases
    best_idx = np.argmax(speedups)
    worst_idx = np.argmin(speedups)
    
    print(f"\nBest case:")
    print(f"  Grid: {results['grid_shape'][best_idx]} ({results['n_vertices'][best_idx]:,} vertices)")
    print(f"  Queries: {results['n_queries'][best_idx]:,}")
    print(f"  Speedup: {speedups[best_idx]:.2f}x")
    
    print(f"\nWorst case:")
    print(f"  Grid: {results['grid_shape'][worst_idx]} ({results['n_vertices'][worst_idx]:,} vertices)")
    print(f"  Queries: {results['n_queries'][worst_idx]:,}")
    print(f"  Speedup: {speedups[worst_idx]:.2f}x")

def main():
    """Run the complete benchmark suite."""
    print("K-d Tree Performance Benchmark")
    print("==============================")
    
    # Define test parameters
    grid_shapes = [
        (10, 10),          # 100 vertices
        (20, 20),          # 400 vertices  
        (50, 50),          # 2,500 vertices
        (100, 100),        # 10,000 vertices
        (10, 10, 10),      # 1,000 vertices (3D)
        (20, 20, 20),      # 8,000 vertices (3D)
        (50, 50, 50),      # 125,000 vertices (3D)
    ]
    
    n_queries_list = [10, 100, 1000, 5000]
    
    # Run benchmarks
    results = benchmark_search_methods(grid_shapes, n_queries_list, n_runs=3)
    
    # Analysis and visualization
    print_summary(results)
    complexity_analysis(results)
    
    # Create plots
    fig = plot_speedup_results(results)
    plt.savefig('kdtree_speedup_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSpeedup analysis plot saved as 'kdtree_speedup_analysis.png'")
    
    # Save raw results
    import json
    with open('kdtree_benchmark_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if key == 'grid_shape':
                json_results[key] = [list(shape) for shape in value]
            else:
                json_results[key] = [float(v) for v in value]
        json.dump(json_results, f, indent=2)
    
    print("Raw benchmark data saved as 'kdtree_benchmark_results.json'")
    
    return results

if __name__ == "__main__":
    results = main()