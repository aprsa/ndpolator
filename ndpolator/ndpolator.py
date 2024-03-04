import numpy as np
from enum import IntEnum

import cndpolator

__version__ = '0.1.0'


class ExtrapolationMethod(IntEnum):
    NONE = 0
    NEAREST = 1
    LINEAR = 2


class AxisFlag(IntEnum):
    SPANNING_AXIS = 1
    ADDITIONAL_AXIS = 2


class Ndpolator():
    """
    This class implements interpolation and extrapolation in n dimensions.
    """
    def __init__(self, basic_axes):
        """
        Instantiates an Nndpolator class. The class relies on `axes` to span
        the interpolation hypercubes. Only basic (spanning) axes should
        be passed here.

        Parameters
        ----------
        basic_axes : tuple of ndarrays
            Axes that span the atmosphere grid. Only the required (spanning)
            axes should be included here; any additional axes should be
            registered separately.
        """

        self.axes = basic_axes
        self.table = dict()

    def __repr__(self) -> str:
        return f'<Ndpolator N={len(self.axes)}, {len(self.table)} tables>'

    def __str__(self) -> str:
        return f'<Ndpolator N={len(self.axes)}, {len(self.table)} tables>'

    @property
    def tables(self):
        """
        Prints a list of tables attached to the ndpolator.

        Returns
        -------
        list of strings
            table names (references) attached to the ndpolator
        """
        return list(self.table.keys())

    def register(self, table, attached_axes, grid):
        if not isinstance(table, str):
            raise ValueError('parameter `table` must be a string')

        self.table[table] = [attached_axes, np.ascontiguousarray(grid), None]

    def find_indices(self, table, query_pts):
        adtl_axes = self.table[table][0]
        axes = self.axes if adtl_axes is None else self.axes + adtl_axes
        indices, flags, normed_query_pts = cndpolator.find(axes, query_pts)
        return indices, flags, normed_query_pts

    def find_hypercubes(self, table, indices, flags, adtl_axes=None):
        axes = self.axes if adtl_axes is None else self.axes + adtl_axes
        grid = self.table[table][1]
        hypercubes = cndpolator.hypercubes(indices, axes, flags, grid)
        return hypercubes

    def ndpolate(self, table, query_pts, extrapolation_method=0):
        if extrapolation_method == 'none':
            extrapolation_method = 0
        elif extrapolation_method == 'nearest':
            extrapolation_method = 1
        elif extrapolation_method == 'linear':
            extrapolation_method = 2
        else:
            raise ValueError(f"extrapolation_method={extrapolation_method} is not valid; it must be one of ['none', 'nearest', 'linear'].")

        capsule = self.table[table][2]
        if capsule:
            interps = cndpolator.ndpolate(capsule=capsule, query_pts=query_pts, nbasic=len(self.axes), extrapolation_method=extrapolation_method)
        else:
            attached_axes = self.table[table][0]
            grid = self.table[table][1]
            axes = self.axes if attached_axes is None else self.axes + attached_axes

            interps, capsule = cndpolator.ndpolate(query_pts=query_pts, axes=axes, grid=grid, nbasic=len(self.axes), extrapolation_method=extrapolation_method)
            self.table[table][2] = capsule

        return interps
