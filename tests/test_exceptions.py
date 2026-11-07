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
    
    name = 123
    grid = 123
    with pytest.raises(TypeError):
        ndp.register(name=name, associated_axes=None, grid=grid)

    name = 'test'
    grid = 123
    with pytest.raises(TypeError):
        ndp.register(name=name, associated_axes=None, grid=grid)

    grid = np.random.uniform(low=0, high=1, size=(3, 3, 1))
    ndp.register(name=name, associated_axes=None, grid=grid)


def test_missing_table_arguments_raise_value_error():
    # the C extension must raise a proper exception, not return NULL silently
    # (which python would report as SystemError), when it is given neither a
    # valid capsule nor the axes and grid needed to build a table:
    import cndpolator

    query_pts = np.array([[0.5, 0.5]])
    axes = (np.array([0., 1.]), np.array([0., 1.]))
    grid = np.zeros((2, 2, 1))

    with pytest.raises(ValueError):
        cndpolator.ndpolate(query_pts=query_pts)
    with pytest.raises(ValueError):
        cndpolator.ndpolate(query_pts=query_pts, axes=axes)
    with pytest.raises(ValueError):
        cndpolator.ndpolate(capsule=object(), query_pts=query_pts)
    with pytest.raises(ValueError):
        cndpolator.distance(query_pts, object())
    with pytest.raises(TypeError):
        cndpolator.ndpolate(query_pts=query_pts, axes=list(axes), grid=grid)


def test_enums_are_registered():
    import cndpolator

    assert [m.value for m in cndpolator.ExtrapolationMethod] == [0, 1, 2]
    assert cndpolator.ExtrapolationMethod.NONE == 0
    assert cndpolator.ExtrapolationMethod.NEAREST == 1
    assert cndpolator.ExtrapolationMethod.LINEAR == 2
    assert cndpolator.SearchAlgorithm.KDTREE == 0
    assert cndpolator.SearchAlgorithm.LINEAR == 1
