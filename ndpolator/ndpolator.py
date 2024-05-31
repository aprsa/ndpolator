import numpy as np
import cndpolator

from cndpolator import ExtrapolationMethod


class Ndpolator():
    """
    This class implements interpolation and extrapolation in n dimensions.
    It provides a "backbone" to the cndpolator C extension.
    """

    def __init__(self, basic_axes: tuple) -> None:
        """
        Instantiates an Nndpolator class. The class relies on `basic_axes` to
        span the interpolation hypercubes. Only basic (spanning) axes should
        be passed here.

        Parameters
        ----------
        basic_axes : tuple of ndarrays
            Axes that span the ndpolator grid.
        """
        if not isinstance(basic_axes, tuple):
            raise TypeError('parameter `basic_axes` must be a tuple of numpy ndarrays')
        for ti, telem in enumerate(basic_axes):
            if not isinstance(telem, np.ndarray):
                raise TypeError(f'the `basic_axes[{ti}]` element must be a numpy ndarray')

        self.axes = basic_axes
        self.table = dict()

    def __repr__(self) -> str:
        return f'<Ndpolator N={len(self.axes)}, {len(self.table)} tables>'

    def __str__(self) -> str:
        return f'<Ndpolator N={len(self.axes)}, {len(self.table)} tables>'

    @property
    def tables(self) -> list:
        """
        Prints a list of tables attached to the ndpolator instance.

        Returns
        -------
        list of strings
            table names (references) attached to the ndpolator
        """
        return list(self.table.keys())

    def register(self, table: str, associated_axes: tuple, grid: np.ndarray) -> None:
        """
        Registers an interpolation grid, along with any associated tables,
        with the ndpolator instance. It is referenced by the provided table
        label. The list of tables is held in the top-level `table` dictionary.
        Each entry in the list has three elements: a tuple of associated axes,
        the interpolation grid, and a capsule (initially None) that stores a
        pointer to the initialized cndpolator structure for caching purposes.

        Parameters
        ----------
        table : str
            reference label to the interpolation grid
        associated_axes : tuple or None
            any additional non-basic axes in the interpolation grid
        grid : np.ndarray
            interpolation grid; grid shape should be `(l(b1), ..., l(bn),
            l(a1), ..., l(bm), l(fv))`, where `bk` are basic axes, `ak` are
            associated axes, `fv` is the function value, and `l()` is the
            length operator.

        Raises
        ------
        TypeError
            if any of passed parameters have an incorrect type.
        """
        if not isinstance(table, str):
            raise TypeError('parameter `table` must be a string')
        if associated_axes:
            if not isinstance(associated_axes, tuple):
                raise TypeError('parameter `associated_axes` must be a tuple of numpy ndarrays')
            for ti, telem in enumerate(associated_axes):
                if not isinstance(telem, np.ndarray):
                    raise TypeError(f'the `associated_axes[{ti}]` element must be a numpy ndarray')
        if not isinstance(grid, np.ndarray):
            raise TypeError('parameter `grid` must be a numpy ndarray')

        self.table[table] = [associated_axes, np.ascontiguousarray(grid), None]

    def import_query_pts(self, table: str, query_pts: np.ndarray) -> tuple:
        """
        Imports and processes query points (points of interest). This entails
        finding the enclosing (or adjacent, if at the boundary of the grid)
        hypercubes for each query point; per query point component flags
        (whether the component is on grid, on vertex, or out of bounds); and
        hypercube-normalized query points. Indices identify hypercubes by
        their superior corner; for example, a (3, 4, 5) x (2, 3, 4) x (1, 2)
        hypercube would be identified by the (5, 4, 2) corner. Thus, for `N`
        query points and `M` basic axes, all three arrays (indices, flags and
        hypercube-normalized query points are `(N, M)`-shaped.
        
        Note: this class method is rarely called directly. The only time it
        would be called is when the calling function identifies query points
        within a certain hypercube and then reuses the indices, flags, and
        hypercube-normalized query points.

        Parameters
        ----------
        table : str
            reference label to the interpolation grid
        query_pts : np.ndarray
            a numpy ndarray of query points; the expected shape is `(N, M)`,
            where `N` is the number of query points and `M` is the number of
            basic axes.

        Returns
        -------
        tuple[np.ndarray]
            A tuple with three elements: an ndarray of containing hypercube
            indices, an ndarray of per-component flags, and an ndarray of
            hypercube-normalized query points.
        """

        # make sure that the array we're passing to C is contiguous:
        query_pts = np.ascontiguousarray(query_pts)

        associated_axes = self.table[table][0]
        axes = self.axes if associated_axes is None else self.axes + associated_axes
        indices, flags, normed_query_pts = cndpolator.find(axes=axes, query_pts=query_pts, nbasic=len(self.axes))
        return indices, flags, normed_query_pts

    def find_hypercubes(self, table: str, normed_query_pts: np.ndarray, indices: np.ndarray, flags: np.ndarray, associated_axes: tuple = None) -> np.ndarray:
        """
        Extracts and populates hypercubes for each query point based on the
        table reference, indices, flags and any associated axes.

        Note: this class method is rarely called directly. The only time it
        would be called is when the calling function identifies query points
        within a certain hypercube and then reuses the indices, flags, and
        hypercube-normalized query points.
        
        Parameters
        ----------
        table : str
            reference label to the interpolation grid
        normed_query_pts :
            an `(N, M)`-shaped array of normalized query points
        indices : np.ndarray
            an `(N, M)`-shaped array of superior hypercube corners
        flags : np.ndarray
            an `(N, M)`-shaped array of per-query-point-component flags
        associated_axes : tuple | None, optional
            A tuple of any associated (non-basic) axes, by default None

        Returns
        -------
        np.ndarray
            An (N, M, l(fv))-shaped array of containing (or adjacent, if on
            the grid boundary) hypercubes identified by their superior
            corners; `N` is the number of query points, `M` is the number of
            basic axes, and `l(fv)` is the length of the function values.
        """

        axes = self.axes if associated_axes is None else self.axes + associated_axes
        grid = self.table[table][1]
        hypercubes = cndpolator.hypercubes(normed_query_pts=normed_query_pts, indices=indices, axes=axes, flags=flags, grid=grid)
        return hypercubes

    def ndpolate(self, table: str, query_pts: np.ndarray, extrapolation_method: str = 'none') -> np.ndarray:
        """
        Performs n-dimensional interpolation or extrapolation. This is the
        main "workhorse" of the class and should be considered the default
        interface to the underlying C-based cndpolator extension. See the
        top-level README.md file for usage examples.

        Parameters
        ----------
        table : str
            reference label to the interpolation grid
        query_pts : np.ndarray
            a numpy ndarray of query points; the expected shape is `(N, M)`,
            where `N` is the number of query points and `M` is the number of
            basic axes.
        extrapolation_method : str, optional
            extrapolation method, one of 'none', 'nearest', 'linear'; by
            default 'none'

        Returns
        -------
        dict
            mandatory keys: 'interps'
            optional keys: 'dists'
            
            interps: np.ndarray
                an (N, l(fv))-shaped array of interpolated values, where `N`
                is the number of query points and `l(fv)` is the length of
                function values.

        Raises
        ------
        ValueError
            raised when the passed extrapolation method is not one of 'none',
            'nearest', 'linear'.
        """
        extrapolation_methods = {
            'none': ExtrapolationMethod.NONE,
            'nearest': ExtrapolationMethod.NEAREST,
            'linear': ExtrapolationMethod.LINEAR
        }

        if extrapolation_method not in extrapolation_methods.keys():
            raise ValueError(f"extrapolation_method={extrapolation_method} is not valid; it must be one of {extrapolation_methods.keys()}.")
        extrapolation_method = extrapolation_methods[extrapolation_method]

        # make sure that the array we're passing to C is contiguous:
        query_pts = np.ascontiguousarray(query_pts)

        # if cndpolator structures have been used before, use the cached
        # version, otherwise initialize and cache it for subsequent use:
        capsule = self.table[table][2]
        if capsule:
            interps, dists = cndpolator.ndpolate(capsule=capsule, query_pts=query_pts, nbasic=len(self.axes), extrapolation_method=extrapolation_method)
        else:
            associated_axes = self.table[table][0]
            grid = self.table[table][1]
            axes = self.axes if associated_axes is None else self.axes + associated_axes

            interps, dists, capsule = cndpolator.ndpolate(query_pts=query_pts, axes=axes, grid=grid, nbasic=len(self.axes), extrapolation_method=extrapolation_method)
            self.table[table][2] = capsule

        if extrapolation_method == ExtrapolationMethod.NONE:
            return {
                'interps': interps
            }
        else:
            return {
                'interps': interps,
                'dists': dists
            }
