![GitHub commit activity](https://img.shields.io/github/commit-activity/w/aprsa/ndpolator)
![GitHub last commit](https://img.shields.io/github/last-commit/aprsa/ndpolator)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/aprsa/ndpolator)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/aprsa/ndpolator/on_pr.yml)
![Documentation](https://img.shields.io/github/actions/workflow/status/aprsa/ndpolator/docbuild.yml?label=docs-build)

# Ndpolator
Fast, n-dimensional linear interpolation and extrapolation on sparse grids.

Ndpolator is a combined interpolator/extrapolator that operates on sparse (incompletely populated) $n$-dimensional grids. It estimates scalar or vector function values within and beyond the definition range of the grid while still avoiding the need to impute missing data or sacrifice the benefits of structured grids. Ndpolator is written in C for speed and portability; a python wrapper that uses numpy arrays is provided for convenience.

A high-level introduction to ndpolator is available [here (pdf)](https://github.com/aprsa/ndpolator/actions/runs/8274798696/artifacts/1324607266).

# Installation

To install ndpolator, clone the github repo and install with pip:

```bash
$> git clone https://github.com/aprsa/ndpolator ndpolator
$> cd ndpolator
$> pip install .
```

Once installed, you can test it by running a pytest:

```bash
$> cd tests
$> pytest
```

# Documentation

API reference is available at [gh-pages](https://aprsa.github.io/ndpolator).

A draft [JOSS](https://joss.theoj.org/) paper that describes the operational details of ndpolator is available [here](https://github.com/aprsa/ndpolator/actions/runs/8274798696/artifacts/1324607266). The draft has been submitted on Mar 13, 2024.

# Usage example

To demonstrate the usage of ndpolator, let us consider a 3-dimensional space with three axes of vastly different vertex magnitudes. For comparison purposes, let the function that we want to interpolate and extrapolate be a linear scalar field:

$$ \mathbf a_1 = (1000, 2000, 3000, 4000, 5000), \quad \mathbf a_2 = (1, 2, 3, 4, 5), \quad \mathbf a_3 = (0.01, 0.02, 0.03, 0.04, 0.05), $$

$$ \mathbf F(x, y, z) = \frac{x}{1000} + y + 100 z. $$

A suitable ndpolator instance would be initiated and operated as follows:

```python
import numpy
import ndpolator

# initialize the axes:
a1 = np.linspace(1000, 5000, 5)
a2 = np.linspace(1, 5, 5)
a3 = np.linspace(0.01, 0.05, 5)

# initialize interpolation space:
ndp = ndpolator.Ndpolator(basic_axes=(a1, a2, a3))

# define a scalar function field and evaluate it across the grid:
def fv(pt):
    return pt[0]/1000 + pt[1] + 100*pt[2]

grid = np.empty((len(ax1), len(ax2), len(ax3), 1))
for i, x in enumerate(ax1):
        for j, y in enumerate(ax2):
            for k, z in enumerate(ax3):
                grid[i, j, k, 0] = fv((x, y, z))

# label the grid ('main') and register it with the ndpolator instance:
ndp.register(table='main', associated_axes=None, grid=grid)

# draw query points randomly within and beyond the definition ranges:
query_pts = np.ascontiguousarray(
    np.vstack((
        np.random.uniform(500, 5500, 1000),
        np.random.uniform(0.5, 5.5, 1000),
        np.random.uniform(0.005, 0.055, 1000))
    ).T
)

# interpolate and extrapolate linearly:
interps = ndp.ndpolate(table='main', query_pts, extrapolation_method='nearest')
```
