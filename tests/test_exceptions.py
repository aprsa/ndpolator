import pytest
import numpy as np

try:
    import ndpolator
    ndpolator.__version__
except ImportError:
    pytest.fail('Failed to import the ndpolator module.')


def test_wrapper_typing():
    axes = [1, 2, 3]
    with pytest.raises(TypeError):
        ndp = ndpolator.Ndpolator(axes)

    axes = (1, 2, 3)
    with pytest.raises(TypeError):
        ndp = ndpolator.Ndpolator(axes)

    ax = np.array([1, 2, 3])
    axes = (ax, True)
    with pytest.raises(TypeError):
        ndp = ndpolator.Ndpolator(axes)

    axes = (ax, ax)
    ndp = ndpolator.Ndpolator(axes)
    
    table = 123
    grid = 123
    with pytest.raises(TypeError):
        ndp.register(table=table, associated_axes=None, grid=grid)

    table = 'test'
    grid = 123
    with pytest.raises(TypeError):
        ndp.register(table=table, associated_axes=None, grid=grid)

    grid = np.random.uniform(low=0, high=1, size=(3, 3, 1))
    ndp.register(table=table, associated_axes=None, grid=grid)
