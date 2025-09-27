#!/usr/bin/env python3
"""Fine-grained performance benchmark to show continuous O(N) vs O(log N) scaling."""

import numpy as np
import ndpolator
import time
import matplotlib.pyplot as plt


def create_test_grid(shape):
    """Create a test grid with specified shape."""
    # Create axes
    axes = tuple(np.linspace(0, s-1, s) for s in shape)
    ndp = ndpolator.Ndpolator(basic_axes=axes)
    
    # Create grid with simple function
    grid = np.zeros(shape + (1,))
    for idx in np.ndindex(shape):
        coords = [axes[i][idx[i]] for i in range(len(shape))]
        grid[idx + (0,)] = sum(coords)
    
    ndp.register('main', associated_axes=None, grid=grid)
    return ndp, axes


def generate_queries(axes, n_queries):
    """Generate random query points including extrapolation."""
    n_dims = len(axes)
    queries = np.zeros((n_queries, n_dims))
    
    for i in range(n_dims):
        axis_min, axis_max = axes[i][0], axes[i][-1]
        axis_range = axis_max - axis_min
        
        # Extend range for extrapolation
        extended_min = axis_min - 0.3 * axis_range
        extended_max = axis_max + 0.3 * axis_range
        
        queries[:, i] = np.random.uniform(extended_min, extended_max, n_queries)
    
    return queries


def benchmark_scaling(dimension, n_queries, n_runs=3):
    """Benchmark scaling behavior with fine-grained grid sizes."""
    print(f"Fine-grained scaling benchmark: {dimension}D grids, {n_queries} queries")
    print("=" * 70)
    
    # Define fine-grained grid sizes
    if dimension == 2:
        # 2D: from 5x5 to 100x100 with many intermediate points
        grid_sizes = [5, 7, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
        shapes = [(s, s) for s in grid_sizes]
    elif dimension == 3:
        # 3D: from 5x5x5 to 30x30x30 (smaller due to cubic growth)
        grid_sizes = [5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 25, 28, 30]
        shapes = [(s, s, s) for s in grid_sizes]
    elif dimension == 4:
        # 4D: from 3x3x3x3 to 12x12x12x12 (much smaller due to quartic growth)
        grid_sizes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        shapes = [(s, s, s, s) for s in grid_sizes]
    else:
        raise ValueError("Only 2D, 3D, and 4D supported")
    
    results = []
    
    for i, shape in enumerate(shapes):
        n_vertices = np.prod(shape)
        print(f"Progress: {i+1}/{len(shapes)} - Grid {shape} ({n_vertices:,} vertices)", end=" ... ")
        
        try:
            # Create test setup
            ndp, axes = create_test_grid(shape)
            queries = generate_queries(axes, n_queries)
            
            # Benchmark linear search
            linear_times = []
            for _ in range(n_runs):
                start_time = time.time()
                result_linear = ndp.ndpolate('main', queries,
                                           extrapolation_method='nearest',
                                           search_algorithm='linear')
                linear_times.append(time.time() - start_time)
            
            # Benchmark k-d tree search
            kdtree_times = []
            for _ in range(n_runs):
                start_time = time.time()
                result_kdtree = ndp.ndpolate('main', queries,
                                           extrapolation_method='nearest',
                                           search_algorithm='kdtree')
                kdtree_times.append(time.time() - start_time)
            
            # Verify results match (sample check)
            sample_size = min(100, n_queries)
            if not np.allclose(result_linear['interps'][:sample_size], 
                              result_kdtree['interps'][:sample_size], rtol=1e-10):
                print("WARNING: Results don't match!")
            
            # Calculate stats
            linear_avg = np.mean(linear_times)
            kdtree_avg = np.mean(kdtree_times)
            speedup = linear_avg / kdtree_avg
            
            time_per_query_linear = (linear_avg / n_queries) * 1000  # ms per query
            time_per_query_kdtree = (kdtree_avg / n_queries) * 1000
            
            results.append({
                'shape': shape,
                'n_vertices': n_vertices,
                'linear_time': linear_avg,
                'kdtree_time': kdtree_avg,
                'speedup': speedup,
                'time_per_query_linear': time_per_query_linear,
                'time_per_query_kdtree': time_per_query_kdtree
            })
            
            print(f"Speedup: {speedup:.1f}x")
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    return results


def plot_scaling_analysis(results_2d, results_3d, results_4d):
    """Create detailed scaling analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Fine-Grained Scaling Analysis: O(N) vs O(log N)', fontsize=16, fontweight='bold')
    
    # Extract data
    def extract_data(results):
        n_vertices = np.array([r['n_vertices'] for r in results])
        linear_times = np.array([r['time_per_query_linear'] for r in results])
        kdtree_times = np.array([r['time_per_query_kdtree'] for r in results])
        speedups = np.array([r['speedup'] for r in results])
        return n_vertices, linear_times, kdtree_times, speedups
    
    # Plot 1: Time per Query vs Grid Size (Log-Log)
    ax1 = axes[0, 0]
    
    if results_2d:
        n_vertices_2d, linear_2d, kdtree_2d, _ = extract_data(results_2d)
        ax1.loglog(n_vertices_2d, linear_2d, 'ro-', label='Linear (2D)', markersize=4, linewidth=2)
        ax1.loglog(n_vertices_2d, kdtree_2d, 'rs--', label='K-d tree (2D)', markersize=4, linewidth=2)
    
    if results_3d:
        n_vertices_3d, linear_3d, kdtree_3d, _ = extract_data(results_3d)
        ax1.loglog(n_vertices_3d, linear_3d, 'bo-', label='Linear (3D)', markersize=4, linewidth=2)
        ax1.loglog(n_vertices_3d, kdtree_3d, 'bs--', label='K-d tree (3D)', markersize=4, linewidth=2)
    
    if results_4d:
        n_vertices_4d, linear_4d, kdtree_4d, _ = extract_data(results_4d)
        ax1.loglog(n_vertices_4d, linear_4d, 'go-', label='Linear (4D)', markersize=4, linewidth=2)
        ax1.loglog(n_vertices_4d, kdtree_4d, 'gs--', label='K-d tree (4D)', markersize=4, linewidth=2)
    
    # Add theoretical complexity lines
    if results_2d:
        x_range = np.logspace(np.log10(n_vertices_2d.min()), np.log10(n_vertices_2d.max()), 100)
        # O(N) line normalized to fit
        linear_theory = x_range / x_range[0] * linear_2d[0]
        ax1.loglog(x_range, linear_theory, 'k-', alpha=0.5, linewidth=3, label='O(N) theory')
        
        # O(log N) line normalized to fit
        log_theory = np.log(x_range) / np.log(x_range[0]) * kdtree_2d[0]
        ax1.loglog(x_range, log_theory, 'g-', alpha=0.5, linewidth=3, label='O(log N) theory')
    
    ax1.set_xlabel('Number of Vertices')
    ax1.set_ylabel('Time per Query (ms)')
    ax1.set_title('Complexity Scaling Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup vs Grid Size
    ax2 = axes[0, 1]
    
    if results_2d:
        _, _, _, speedups_2d = extract_data(results_2d)
        ax2.semilogx(n_vertices_2d, speedups_2d, 'ro-', label='2D grids', markersize=6, linewidth=2)
    
    if results_3d:
        _, _, _, speedups_3d = extract_data(results_3d)
        ax2.semilogx(n_vertices_3d, speedups_3d, 'bo-', label='3D grids', markersize=6, linewidth=2)
    
    if results_4d:
        _, _, _, speedups_4d = extract_data(results_4d)
        ax2.semilogx(n_vertices_4d, speedups_4d, 'go-', label='4D grids', markersize=6, linewidth=2)
    
    ax2.set_xlabel('Number of Vertices')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('Speedup vs Grid Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Log Scale Time Comparison (All Dimensions)
    ax3 = axes[1, 0]
    if results_2d:
        ax3.semilogy(n_vertices_2d, linear_2d, 'ro-', label='Linear 2D', markersize=4, linewidth=2)
        ax3.semilogy(n_vertices_2d, kdtree_2d, 'rs--', label='K-d tree 2D', markersize=4, linewidth=2)
    if results_3d:
        ax3.semilogy(n_vertices_3d, linear_3d, 'bo-', label='Linear 3D', markersize=4, linewidth=2)
        ax3.semilogy(n_vertices_3d, kdtree_3d, 'bs--', label='K-d tree 3D', markersize=4, linewidth=2)
    if results_4d:
        ax3.semilogy(n_vertices_4d, linear_4d, 'go-', label='Linear 4D', markersize=4, linewidth=2)
        ax3.semilogy(n_vertices_4d, kdtree_4d, 'gs--', label='K-d tree 4D', markersize=4, linewidth=2)
    
    ax3.set_xlabel('Number of Vertices')
    ax3.set_ylabel('Time per Query (ms)')
    ax3.set_title('Performance Comparison (Log Y Scale)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: 4D Performance Comparison (if available)
    ax4 = axes[1, 1]
    if results_4d:
        ax4.loglog(n_vertices_4d, linear_4d, 'go-', label='Linear search', markersize=4, linewidth=2)
        ax4.loglog(n_vertices_4d, kdtree_4d, 'gs--', label='K-d tree', markersize=4, linewidth=2)
        ax4.set_xlabel('Number of Vertices')
        ax4.set_ylabel('Time per Query (ms)')
        ax4.set_title('4D Performance (Log-Log Scale)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        # Show a combined speedup comparison if no 4D data
        if results_2d and results_3d:
            ax4.semilogx(n_vertices_2d, speedups_2d, 'ro-', label='2D speedup', markersize=4, linewidth=2)
            ax4.semilogx(n_vertices_3d, speedups_3d, 'bo-', label='3D speedup', markersize=4, linewidth=2)
            ax4.set_xlabel('Number of Vertices')
            ax4.set_ylabel('Speedup (x)')
            ax4.set_title('Speedup Comparison (2D vs 3D)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def analyze_complexity(results, dimension):
    """Analyze the computational complexity from the results."""
    if not results:
        return
        
    n_vertices = np.array([r['n_vertices'] for r in results])
    linear_times = np.array([r['time_per_query_linear'] for r in results])
    kdtree_times = np.array([r['time_per_query_kdtree'] for r in results])
    
    # Fit power laws: time = a * N^b
    linear_coeffs = np.polyfit(np.log(n_vertices), np.log(linear_times), 1)
    kdtree_coeffs = np.polyfit(np.log(n_vertices), np.log(kdtree_times), 1)
    
    print(f"\n{dimension}D Complexity Analysis:")
    print(f"  Linear search: O(N^{linear_coeffs[0]:.3f}) - Expected: O(N^1.0)")
    print(f"  K-d tree:      O(N^{kdtree_coeffs[0]:.3f}) - Expected: O(log N ≈ N^0.1)")
    
    # Calculate R-squared for goodness of fit
    def r_squared(y_actual, y_predicted):
        ss_res = np.sum((y_actual - y_predicted) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    linear_predicted = np.exp(linear_coeffs[1]) * n_vertices ** linear_coeffs[0]
    kdtree_predicted = np.exp(kdtree_coeffs[1]) * n_vertices ** kdtree_coeffs[0]
    
    linear_r2 = r_squared(linear_times, linear_predicted)
    kdtree_r2 = r_squared(kdtree_times, kdtree_predicted)
    
    print(f"  Fit quality (R²): Linear = {linear_r2:.3f}, K-d tree = {kdtree_r2:.3f}")


def main():
    """Run fine-grained scaling analysis."""
    print("Fine-Grained K-d Tree Scaling Analysis")
    print("======================================")
    
    n_queries = 1000  # Fixed number of queries for consistent comparison
    
    # Run 2D benchmark
    print("\nRunning 2D scaling benchmark...")
    results_2d = benchmark_scaling(dimension=2, n_queries=n_queries)
    
    print("\nRunning 3D scaling benchmark...")
    results_3d = benchmark_scaling(dimension=3, n_queries=n_queries)
    
    print("\nRunning 4D scaling benchmark...")
    results_4d = benchmark_scaling(dimension=4, n_queries=n_queries)
    
    # Analyze complexity
    analyze_complexity(results_2d, "2D")
    analyze_complexity(results_3d, "3D")
    analyze_complexity(results_4d, "4D")
    
    # Summary statistics
    if results_2d or results_3d or results_4d:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        all_speedups = []
        if results_2d:
            speedups_2d = [r['speedup'] for r in results_2d]
            all_speedups.extend(speedups_2d)
            print(f"2D grids: {np.mean(speedups_2d):.1f}x average speedup ({len(results_2d)} points)")
            
        if results_3d:
            speedups_3d = [r['speedup'] for r in results_3d]
            all_speedups.extend(speedups_3d)
            print(f"3D grids: {np.mean(speedups_3d):.1f}x average speedup ({len(results_3d)} points)")
        
        if results_4d:
            speedups_4d = [r['speedup'] for r in results_4d]
            all_speedups.extend(speedups_4d)
            print(f"4D grids: {np.mean(speedups_4d):.1f}x average speedup ({len(results_4d)} points)")
        
        if all_speedups:
            print(f"Overall: {np.mean(all_speedups):.1f}x average speedup")
            print(f"Range: {np.min(all_speedups):.1f}x to {np.max(all_speedups):.1f}x")
    
    # Create plots
    fig = plot_scaling_analysis(results_2d, results_3d, results_4d)
    plt.savefig('fine_grained_scaling_analysis.png', dpi=300, bbox_inches='tight')
    print("\nScaling analysis plot saved as 'fine_grained_scaling_analysis.png'")
    
    return results_2d, results_3d, results_4d


if __name__ == "__main__":
    results_2d, results_3d, results_4d = main()