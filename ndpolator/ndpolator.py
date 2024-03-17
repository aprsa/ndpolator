import numpy as np
import cndpolator

from cndpolator import ExtrapolationMethod


class Ndpolator():
    """
    This class implements interpolation and extrapolation in n dimensions.
    """
    def __init__(self, basic_axes: tuple) -> None:
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
    def tables(self) -> list[str]:
        """
        Prints a list of tables attached to the ndpolator.

        Returns
        -------
        list of strings
            table names (references) attached to the ndpolator
        """
        return list(self.table.keys())

    def register(self, table: str, associated_axes: tuple, grid: np.ndarray) -> None:
        if not isinstance(table, str):
            raise ValueError('parameter `table` must be a string')

        self.table[table] = [associated_axes, np.ascontiguousarray(grid), None]

    def import_query_pts(self, table: str, query_pts: np.ndarray) -> tuple:
        # make sure that the array we're passing to C is contiguous:
        query_pts = np.ascontiguousarray(query_pts)

        associated_axes = self.table[table][0]
        axes = self.axes if associated_axes is None else self.axes + associated_axes
        indices, flags, normed_query_pts = cndpolator.find(axes=axes, query_pts=query_pts, nbasic=len(self.axes))
        return indices, flags, normed_query_pts

    def find_hypercubes(self, table: str, indices: np.ndarray, flags: np.ndarray, adtl_axes: tuple | None = None) -> np.ndarray:
        axes = self.axes if adtl_axes is None else self.axes + adtl_axes
        grid = self.table[table][1]
        hypercubes = cndpolator.hypercubes(indices=indices, axes=axes, flags=flags, grid=grid)
        return hypercubes

    def ndpolate(self, table: str, query_pts: np.ndarray, extrapolation_method: str = 'none') -> np.ndarray:
        extrapolation_methods = {
            'none': ExtrapolationMethod.NONE,
            'nearest': ExtrapolationMethod.NEAREST,
            'linear': ExtrapolationMethod.LINEAR
        }

        if extrapolation_method not in extrapolation_methods.keys():
            raise ValueError(f"extrapolation_method={extrapolation_method} is not valid; it must be one of {extrapolation_methods.keys()}.")
        extrapolation_method = extrapolation_methods.get(extrapolation_method, ExtrapolationMethod.NONE)

        # make sure that the array we're passing to C is contiguous:
        query_pts = np.ascontiguousarray(query_pts)

        capsule = self.table[table][2]
        if capsule:
            interps = cndpolator.ndpolate(capsule=capsule, query_pts=query_pts, nbasic=len(self.axes), extrapolation_method=extrapolation_method)
        else:
            associated_axes = self.table[table][0]
            grid = self.table[table][1]
            axes = self.axes if associated_axes is None else self.axes + associated_axes

            interps, capsule = cndpolator.ndpolate(query_pts=query_pts, axes=axes, grid=grid, nbasic=len(self.axes), extrapolation_method=extrapolation_method)
            self.table[table][2] = capsule

        return interps
