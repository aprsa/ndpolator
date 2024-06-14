/**
 * @file ndpolator.c
 * @brief Main functions and python bindings.
 */

#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

/**
 * @private
 * @def PY_ARRAY_UNIQUE_SYMBOL
 * Required by numpy C-API. It defines a unique symbol to be used in other
 * C source files and header files.
 */

#define PY_ARRAY_UNIQUE_SYMBOL cndpolator_ARRAY_API

#include "ndpolator.h"
#include "ndp_types.h"

/**
 * @def min(a,b)
 * Computes the minimum of @p a and @p b.
 */

#define min(a,b) (((a)<(b))?(a):(b))

/**
 * @def max(a,b)
 * Computes the maximum of @p a and @p b.
 */

#define max(a,b) (((a)>(b))?(a):(b))

/**
 * @def sign(a)
 * Returns the sign of @p a.
 */

#define sign(a) ( ( (a) < 0 )  ?  -1   : ( (a) > 0 ) )

/**
 * <!-- _ainfo() -->
 * @private
 * @brief Internal function for printing ndarray flags from C.
 *
 * @param array numpy ndarray to be analyzed
 * @param print_data boolean, determines whether array contents should be
 * printed.
 *
 * @details
 * The function prints the dimensions, types, flags, and (if @p print_data is
 * TRUE) array contents. This is an internal function that should not be used
 * for anything other than debugging.
 */

void _ainfo(PyArrayObject *array, int print_data)
{
    int i, ndim, size;
    npy_intp *dims, *shape, *strides;

    ndim = PyArray_NDIM(array);
    size = PyArray_SIZE(array);

    printf("array->nd = %d\n", ndim);
    printf("array->flags = %d\n", PyArray_FLAGS(array));
    printf("array->type = %d\n", PyArray_TYPE(array));
    printf("array->itemsize = %d\n", (int) PyArray_ITEMSIZE(array));
    printf("array->size = %d\n", size);
    printf("array->nbytes = %d\n", (int) PyArray_NBYTES(array));

    dims = PyArray_DIMS(array);
    printf("array->dims = [");
    for (i = 0; i < ndim - 1; i++)
        printf("%d, ", (int) dims[i]);
    printf("%d]\n", (int) dims[i]);

    shape = PyArray_SHAPE(array);
    printf("array->shape = [");
    for (i = 0; i < ndim - 1; i++)
        printf("%d, ", (int) shape[i]);
    printf("%d]\n", (int) shape[i]);

    strides = PyArray_STRIDES(array);
    printf("array->strides = [");
    for (i = 0; i < ndim - 1; i++)
        printf("%d, ", (int) strides[i]);
    printf("%d]\n", (int) strides[i]);

    printf("array->is_c_contiguous: %d\n", PyArray_IS_C_CONTIGUOUS(array));
    printf("array->is_f_contiguous: %d\n", PyArray_IS_F_CONTIGUOUS(array));
    printf("array->is_fortran: %d\n", PyArray_ISFORTRAN(array));
    printf("array->is_writeable: %d\n", PyArray_ISWRITEABLE(array));
    printf("array->is_aligned: %d\n", PyArray_ISALIGNED(array));
    printf("array->is_behaved: %d\n", PyArray_ISBEHAVED(array));
    printf("array->is_behaved_ro: %d\n", PyArray_ISBEHAVED_RO(array));
    printf("array->is_carray: %d\n", PyArray_ISCARRAY(array));
    printf("array->is_farray: %d\n", PyArray_ISFARRAY(array));
    printf("array->is_carray_ro: %d\n", PyArray_ISCARRAY_RO(array));
    printf("array->is_farray_ro: %d\n", PyArray_ISFARRAY_RO(array));
    printf("array->is_isonesegment: %d\n", PyArray_ISONESEGMENT(array));

    if (print_data) {
        if (PyArray_TYPE(array) == 5) {
            int *data = (int *) PyArray_DATA(array);
            printf("data = [");
            for (i = 0; i < size - 1; i++)
                printf("%d, ", data[i]);
            printf("%d]\n", data[i]);
        } else {
            double *data = (double *) PyArray_DATA(array);
            printf("data = [");
            for (i = 0; i < size - 1; i++)
                printf("%lf, ", data[i]);
            printf("%lf]\n", data[i]);
        }
    }

    return;
}

/**
 * <!-- find_first_geq_than() -->
 * @brief Finds the superior hypercube vertex for the passed parameter.
 *
 * @param axis an #ndp_axis instance to be searched, must be sorted in
 * ascending order
 * @param l index of the left search boundary in the @p axis, normally 0, but
 * can be anything between 0 and @p r-1
 * @param r index of the right search boundary in the @p axis, normally len(@p
 * axis)-1, but can be anything between @p l+1 and len(@p axis)-1
 * @param x value to be found in @p axis
 * @param rtol relative (fractional) tolerance to determine if @p x coincides
 * with a vertex in @p axis
 * @param flag flag placeholder; it will be populated with one of
 * #NDP_ON_GRID, #NDP_ON_VERTEX, #NDP_OUT_OF_BOUNDS
 *
 * @details
 * Uses bisection to find the index in @p axis that points to the first value
 * that is greater or equal to the requested value @p x. Indices @p l and @p r
 * can be used to narrow the search within the array. When the suitable index
 * is found, a flag is set to #NDP_ON_GRID if @p x is in the array's value
 * span, #NDP_OUT_OF_BOUNDS is @p x is either smaller than @p axis[0] or
 * larger than @p axis[N-1], and #NDP_ON_VERTEX if @p x is within @p rtol of
 * the value in the array.
 *
 * @return index of the first value in the array that is greater-or-equal-to
 * the requested value @p x. It also sets the @p flag accordingly. 
 */

int find_first_geq_than(ndp_axis *axis, int l, int r, double x, double rtol, int *flag)
{
    int debug = 0;
    int m = l + (r - l) / 2;

    while (l != r) {
        // if (x > (1-sign(axis->val[m])*rtol)*axis->val[m])
        if (x + rtol > axis->val[m])
            l = m + 1;
        else
            r = m;

        m = l + (r - l) / 2;
    }

    *flag = (x < axis->val[0] || x > axis->val[axis->len-1]) ? NDP_OUT_OF_BOUNDS : NDP_ON_GRID;

    if ( fabs((x - axis->val[l-1])/(axis->val[l] - axis->val[l-1])) < rtol ||
         (l == axis->len - 1 && fabs((x - axis->val[l-1])/(axis->val[l] - axis->val[l-1])-1) < rtol) )
        *flag |= NDP_ON_VERTEX;

    if (debug)
        printf("l=%d x=%f a[l-1]=%f a[l]=%f rtol=%f flag=%d\n", l, x, axis->val[l-1], axis->val[l], rtol, *flag);

    return l;
}

/**
 * <!-- idx2pos() -->
 * @brief Converts an array of indices into an integer position of the array.
 *
 * @param axes a ndp_axes structure that holds all ndpolator axes
 * @param vdim vertex length (number of function values per grid point)
 * @param index a naxes-dimensional array of indices
 * @param pos placeholder for the position index in the NDP grid that
 * corresponds to per-axis indices
 *
 * @details
 * For efficiency, all ndpolator arrays are 1-dimensional, where axes are
 * stacked in the usual C order (last axis runs first). Referring to grid
 * elements can be done either by position in the 1-dimensional array, or
 * per-axis indices. This function converts from the index representation to
 * position.
 *
 * @return #ndp_status.
 */

int idx2pos(ndp_axes *axes, int vdim, int *index, int *pos)
{
    *pos = axes->cplen[0]*index[0];
    for (int i = 1; i < axes->len; i++)
        *pos += axes->cplen[i]*index[i];
    *pos *= vdim;

    return NDP_SUCCESS;
}

/**
 * <!-- pos2idx() -->
 * @brief Converts position in the array into an array of per-axis indices.
 *
 * @param axes a ndp_axes structure that holds all ndpolator axes
 * @param vdim vertex length (number of function values per grid point)
 * @param pos position index in the grid
 * @param idx an array of per-axis indices; must be allocated
 *
 * @details
 * For efficiency, all ndpolator arrays are 1-dimensional, where axes are
 * stacked in the usual C order (last axis runs first). Referring to grid
 * elements can be done either by position in the 1-dimensional array, or
 * per-axis indices. This function converts from position index representation
 * to an array of per-axis indices.
 * 
 * @return #ndp_status.
 */

int pos2idx(ndp_axes *axes, int vdim, int pos, int *idx)
{
    int debug = 0;

    for (int i=0; i < axes->len; i++)
        idx[i] = pos / vdim / axes->cplen[i] % axes->axis[i]->len;

    if (debug) {
        printf("pos = %d idx = [", pos);
        for (int j = 0; j < axes->len; j++)
            printf("%d ", idx[j]);
        printf("\b]\n");
    }

    return NDP_SUCCESS;
}

/**
 * <!-- c_ndpolate() -->
 * @brief Linear interpolation and extrapolation on a fully defined hypercube.
 *
 * @param naxes ndpolator dimension (number of axes)
 * @param vdim vertex length (number of function values per grid point)
 * @param x point of interest
 * @param fv naxes-dimensional unit hypercube of function values
 *
 * @details
 * Interpolates (or extrapolates) function values on a @p naxes -dimensional
 * fully defined hypercube in a query point @p x. Function values are
 * @p vdim -dimensional. The hypercube is assumed unit-normalized and @p x
 * components are relative to the hypercube. For example, if @p naxes = 3, the
 * hypercube will be an array of length 2<sup>3</sup> = 8, with hypercube
 * vertices at {(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0,
 * 1), (1, 1, 0), (1, 1, 1)}, and @p x a 3-dimensional array w.r.t. the
 * hypercube, i.e. @p x = (0.3, 0.4, 0.6) for interpolation, or @p x = (-0.2,
 * -0.3, 0.6) for extrapolation.
 *
 * @warning
 * For optimization purposes, the function overwrites @p fv values. The user
 * is required to make a copy of the @p fv array if the contents are meant to
 * be reused.
 *
 * @return #ndp_status status code.
 */

int c_ndpolate(int naxes, int vdim, double *x, double *fv)
{
    int debug = 0;
    int i, j, k;

    for (i = 0; i < naxes; i++) {
        if (debug) {
            printf("naxes=%d x[%d]=%3.3f\n", naxes, i, x[i]);
        }
        for (j = 0; j < (1 << (naxes - i - 1)); j++) {
            if (debug)
                printf("j=%d fv[%d]=%3.3f, fv[%d]=%3.3f, ", j, (1 << (naxes - i - 1)) + j, fv[(1 << (naxes - i - 1)) + j], j, fv[j]);
            for (k = 0; k < vdim; k++) {
                fv[j * vdim + k] += (fv[((1 << (naxes - i - 1)) + j) * vdim + k] - fv[j * vdim + k]) * x[i];
            }
            if (debug)
                printf("corr=%3.3f\n", fv[j]);
        }
    }

    return NDP_SUCCESS;
}

int _compare_indexed_dists(const void *a, const void *b)
{
    typedef struct {
        int idx;
        double dist;
    } indexed_dists;

    if (((indexed_dists *) a)->dist < ((indexed_dists *) b)->dist) return -1;
    if (((indexed_dists *) a)->dist > ((indexed_dists *) b)->dist) return 1;
    return 0;
}

/**
 * <!-- find_nearest() -->
 * @brief Finds the nearest defined value on the grid.
 *
 * @param normed_elem unit hypercube-normalized query point
 * @param elem_index superior corner of the containing/nearest hypercube
 * @param elem_flag flag per query point component
 * @param table #ndp_table instance with full ndpolator definition
 * @param extrapolation_method a #ndp_extrapolation_method that determines
 * whether to find the nearest defined vertex or the nearest fully defined
 * hypercube.
 * @param dist placeholder for distance between @p normed_elem and the nearest
 * fully defined hypercube
 *
 * @details
 * Find the nearest defined vertex or the nearest fully defined hypercube.
 *
 * Parameter @p normed_elem provides coordinates of the query point in unit
 * hypercube space. For example, `normed_elem=(0.3, 0.8, 0.2)` provides
 * coordinates of the query point with respect to the inferior hypercube
 * corner, which in this case would be within the hypercube. On the other
 * hand, `normed_elem=(-0.2, 0.3, 0.4)` would lie outside of the hypercube.
 *
 * Parameter @p elem_index provides coordinates of the superior hypercube
 * corner (i.e., indices of each axis where the corresponding value is the
 * first value greater than the query point coordinate). For example, if the
 * query point (in index space) is `(4.2, 5.6, 8.9)`, then `elem_index=(5, 6,
 * 9)`.
 *
 * Parameter @p elem_flag flags each coordinate of the @p normed_elem. Flags
 * can be either #NDP_ON_GRID, #NDP_ON_VERTEX, or #NDP_OUT_OF_BOUNDS. This is
 * important because @p elem_index points to the nearest larger axis value if
 * the coordinate does not coincide with the axis vertex, and it points to the
 * vertex itself if it coincides with the coordinate. For example, if
 * `axis=[0,1,2,3,4]` and the requested element is `1.5`, then @p elem_index
 * will equal 2; if the requested element is `1.0`, then @p elem_index will
 * equal 1; if the requested element is `-0.3`, then @p elem_index will equal
 * 0. In order to correctly account for out-of-bounds and on-vertex requests,
 *    the function needs to be aware of the flags.
 *
 * Parameter @p table is an #ndp_table structure that defines all relevant
 * grid parameters. Of particular use here is the grid of function values and
 * all axis definitions.
 *
 * Parameter @p extrapolation_method determines whether to find the nearest
 * vertex (NDP_METHOD_NEAREST) or the nearest fully defined hypercube
 * (NDP_METHOD_LINEAR). Under the hood this determines which mask is used from
 * the underlying table: @p table->vmask or @p table->hcmask. The first one
 * masks defined vertices, and the second one masks fully defined hypercubes.
 * If a vertex (or a hypercube) is defined, the value of the mask is set to 1;
 * otherwise it is set to 0. These arrays have @p table->nverts elements,
 * which equals to the product of the lengths of all basic axes.
 *
 * The function computes Euclidean square distances for each masked grid point
 * from the requested element and returns the pointer to the nearest function
 * value. The search is optimized by searching over basic axes first. The
 * @p dist parameter is set to the minimal distance.
 *
 * @return allocated pointer to the nearest coordinates. The calling function
 * must free the memory once done.
 */

int *find_nearest(double *normed_elem, int *elem_index, int *elem_flag, ndp_table *table, ndp_extrapolation_method extrapolation_method, double *dist)
{
    int debug = 0;
    int min_pos;
    double cdist;
    int *coords = malloc(table->axes->len * sizeof(*coords));

    typedef struct {
        int idx;
        double dist;
    } indexed_dists;
    
    indexed_dists *dists = malloc(table->nverts * sizeof(*dists));

    int *mask = extrapolation_method == NDP_METHOD_NEAREST ? table->vmask : table->hcmask;

    if (debug) {
        printf("normed_elem=[");
        for (int j = 0; j < table->axes->nbasic; j++) {
            printf("%3.3f ", normed_elem[j]);
        }
        printf("\b] elem_index=[");
        for (int j = 0; j < table->axes->nbasic; j++) {
            printf("%d ", elem_index[j]);
        }
        printf("\b]\n");
    }

    /* loop over all basic vertices: */
    for (int i = 0; i < table->nverts; i++) {
        dists[i].idx = i;

        /* skip if vertex is masked: */
        if (!mask[i]) {
            dists[i].dist = 1e10;
            continue;
        }

        if (debug) {
            printf("  i=% 4d coord=[", i);
        }

        /* find the distance to the basic vertex: */
        cdist = 0.0;

        if (debug) {
            for (int j = 0; j < table->axes->nbasic; j++) {
                int coord = i / (table->axes->cplen[j] / table->axes->cplen[table->axes->nbasic-1]) % table->axes->axis[j]->len;
                printf("%d ", coord);
            }
            printf("\b] cdist: 0 -> ");
        }

        for (int j = 0; j < table->axes->nbasic; j++) {
            /* converts the running index i to j-th coordinate: */
            int coord = i / (table->axes->cplen[j] / table->axes->cplen[table->axes->nbasic-1]) % table->axes->axis[j]->len;

            if (extrapolation_method == NDP_METHOD_NEAREST) {
                /* FIXME: rewrite this logic in terms of offset_normed_elem. */
                double offset_normed_elem = elem_index[j] - coord + normed_elem[j];
                if (normed_elem[j] < 0 || normed_elem[j] > 1)
                    cdist += (offset_normed_elem-1)*(offset_normed_elem-1);
                else
                    cdist += (round(elem_index[j]+normed_elem[j]-1)-coord)*(round(elem_index[j]+normed_elem[j]-1)-coord);
            }

            if (extrapolation_method == NDP_METHOD_LINEAR) {
                double offset_normed_elem = elem_index[j] - coord + normed_elem[j];
                if (offset_normed_elem < 0)
                    cdist += offset_normed_elem*offset_normed_elem;
                else if (offset_normed_elem > 1)
                    cdist += (offset_normed_elem-1)*(offset_normed_elem-1);
                else {
                    /* coordinate is within the hypercube, no cdist change. */
                }
            }

            if (debug)
                printf("%f -> ", cdist);
        }

        if (debug)
            printf("\b\b\b\b\n");

        dists[i].dist = cdist;
    }

    /* sort the distances: */
    qsort(dists, table->nverts, sizeof(*dists), _compare_indexed_dists);
    *dist = dists[0].dist;
    min_pos = dists[0].idx;

    if (debug)
        printf("  min_dist=%f min_pos=%d nearest=[", dists[0].dist, dists[0].idx);

    /* Assemble the coordinates: */
    for (int j = 0; j < table->axes->nbasic; j++) {
        coords[j] = min_pos / (table->axes->cplen[j] / table->axes->cplen[table->axes->nbasic-1]) % table->axes->axis[j]->len;
        if (debug)
            printf("%d ", coords[j]);
    }

    for (int j = table->axes->nbasic; j < table->axes->len; j++) {
        coords[j] = max(0, min(table->axes->axis[j]->len-1, round(elem_index[j]+normed_elem[j])));
        if (debug)
            printf("%d ", coords[j]);
    }

    if (debug)
        printf("\b]\n");

    return coords;
}

/**
 * <!-- ndp_query_pts_import() -->
 * @brief Determines hypercube indices, flags, and normalized query point
 * values based on the passed query points.
 *
 * @param nelems number of query points
 * @param qpts query points, an @p nelems -by- @p naxes array of doubles
 * @param axes a @p qpdim -dimensional array of axes
 *
 * @details
 * Computes superior index of the n-dimensional hypercubes that contain query
 * points @p qpts. It does so by calling #find_first_geq_than() sequentially
 * for all @p axes.
 *
 * When any of the query point components coincides with the grid vertex, that
 * component will be flagged by #NDP_ON_VERTEX. This is used in
 * #find_hypercubes() to reduce the dimensionality of the corresponding
 * hypercube. Any query point components that fall outside of the grid
 * boundaries are flagged by #NDP_OUT_OF_BOUNDS. Finally, all components that
 * do fall within the grid are flagged by #NDP_ON_GRID.
 *
 * @return a #ndp_query_pts instance.
 */

ndp_query_pts *ndp_query_pts_import(int nelems, double *qpts, ndp_axes *axes)
{
    int debug = 0;
    ndp_query_pts *query_pts = ndp_query_pts_new();
    double rtol = 1e-6;  /* relative tolerance for vertex matching */

    ndp_query_pts_alloc(query_pts, nelems, axes->len);

    if (debug) {
        printf("ndp_query_pts_import():\n  number of query points=%d\n  query point dimension=%d\n", nelems, axes->len);
        for (int i = 0; i < axes->len; i++) {
            printf("  axis %d (length %d):\n    [", i, axes->axis[i]->len);
            for (int j = 0; j < axes->axis[i]->len; j++) {
                printf("%2.2f ", axes->axis[i]->val[j]);
            }
            printf("\b]\n");
        }
    }

    for (int i = 0; i < axes->len; i++) {
        for (int j = 0; j < nelems; j++) {
            int k = j*axes->len + i;
            double lo, hi;
            query_pts->requested[k] = qpts[k];
            query_pts->indices[k] = find_first_geq_than(axes->axis[i], 1, axes->axis[i]->len - 1, qpts[k], rtol, &query_pts->flags[k]);
            lo = axes->axis[i]->val[query_pts->indices[k]-1];
            hi = axes->axis[i]->val[query_pts->indices[k]];
            query_pts->normed[k] = (qpts[k] - lo)/(hi - lo);
        }
    }

    if (debug) {
        for (int i = 0; i < nelems; i++) {
            printf("  query_pt[%d] = [", i);
            for (int j = 0; j < axes->len; j++) {
                printf("%2.2f ", qpts[i*axes->len + j]);
            }
            printf("\b]");

            printf("  indices = [");
            for (int j = 0; j < axes->len; j++) {
                printf("%d ", query_pts->indices[i*axes->len + j]);
            }
            printf("\b]");

            printf("  flags = [");
            for (int j = 0; j < axes->len; j++) {
                printf("%d ", query_pts->flags[i*axes->len + j]);
            }
            printf("\b]");

            printf("  normed_query_pt = [");
            for (int j = 0; j < axes->len; j++) {
                printf("%3.3f ", query_pts->normed[i*axes->len + j]);
            }
            printf("\b]\n");
        }
    }

    return query_pts;
}

/**
 * <!-- find_hypercubes() -->
 * @brief Determines n-dimensional hypercubes that contain (or are adjacent
 * to) the query points identified by indices.
 *
 * @param qpts an #ndp_query_pts instance that holds all query point
 * information
 * @param table an #ndp_table instance that holds all axis/grid information
 *
 * @details
 * Hypercubes are n-dimensional subgrids that contain the point of interest
 * (i.e., a query point). If the query point lies within the hypercube, the
 * ndpolator will interpolate based on the function values in the hypercube.
 * If the query point is adjacent to the hypercube, ndpolator will extrapolate
 * instead. The hypercubes here need not be fully defined, i.e. there may be
 * voids (nans) in the table grid.
 *
 * Depending on @p qpts->flags, the dimension of the hypercube can be reduced.
 * In particular, if any query point component flag is set to #NDP_ON_VERTEX,
 * then the corresponding dimension is eliminated (there is no need to
 * interpolate or extrapolate when the value is already on the axis).
 *
 * @return an array of #ndp_hypercube instances, one per query point.
 */

ndp_hypercube **find_hypercubes(ndp_query_pts *qpts, ndp_table *table)
{
    int debug = 0;

    int fdhc, tidx, *iptr;
    int dim_reduction, hc_size;
    double *hc_vertices;

    ndp_axes *axes = table->axes;
    int cidx[axes->len];

    int nelems = qpts->nelems;
    int *indices = qpts->indices;
    int *flags = qpts->flags;

    ndp_hypercube **hypercubes = malloc(nelems * sizeof(*hypercubes));

    for (int i = 0; i < nelems; i++) {
        /* assume the hypercube (or the relevant subcube) is fully defined: */
        fdhc = 1;

        /* if qpts are out of bounds, set fdhc to 0: */
        for (int j = 0; j < axes->len; j++) {
            int pos = i * axes->len + j;
            if ( (NDP_OUT_OF_BOUNDS & flags[pos]) == NDP_OUT_OF_BOUNDS )
                fdhc = 0;
        }

        /* point iptr to the i-th index multiplet: */
        iptr = indices + i*axes->len;

        // do not check whether the hypercube is fully defined before reducing
        // its dimension: it may happen that we don't need the undefined parts
        // of the hypercube!

        /* reduce hypercube dimension for each query point component that coincides with the grid vertex: */
        dim_reduction = 0;
        for (int j = 0; j < axes->len; j++)
            if ((NDP_ON_VERTEX & flags[i*axes->len+j]) == NDP_ON_VERTEX)
                dim_reduction++;

        hc_size = axes->len-dim_reduction;
        hc_vertices = malloc(table->vdim * (1 << hc_size) * sizeof(*hc_vertices));

        if (debug) {
            printf("hypercube %d:\n", i);
            printf("  basic indices: [");
            for (int j = 0; j < axes->nbasic; j++)
                printf("%d ", iptr[j]);
            printf("\b]\n");
            printf("  hypercube size: %d\n", hc_size);
        }

        for (int j = 0; j < (1 << hc_size); j++) {
            for (int k = 0, l = 0; k < axes->len; k++) {
                if ( (NDP_ON_VERTEX & flags[i*axes->len+k]) == NDP_ON_VERTEX) {
                    /* qpts->normed can either be 0 or 1, with 1 being only on the upper axis boundary: */
                    cidx[k] = qpts->normed[i*axes->len+k] > 0.5 ? iptr[k] : iptr[k]-1;
                    // cidx[k] = iptr[k]-1;
                    continue;
                }
                cidx[k] = max(iptr[k]-1+(j / (1 << (hc_size-l-1))) % 2, (j / (1 << (hc_size-l-1))) % 2);
                l++;
            }
            if (debug) {
                printf("    cidx = [");
                for (int k = 0; k < axes->len; k++)
                    printf("%d ", cidx[k]);
                printf("\b], ");
            }

            idx2pos(axes, table->vdim, cidx, &tidx);
            if (table->grid[tidx] != table->grid[tidx])  /* true if nan */
                fdhc = 0;

            if (debug)
                printf("  tidx = %d, table[tidx] = %f\n", tidx, table->grid[tidx]);

            memcpy(hc_vertices + j*table->vdim, table->grid + tidx, table->vdim*sizeof(*hc_vertices));
        }

        ndp_hypercube *hypercube = ndp_hypercube_new_from_data(hc_size, table->vdim, fdhc, hc_vertices);
        if (debug)
            ndp_hypercube_print(hypercube, "    ");

        hypercubes[i] = hypercube;
    }

    return hypercubes;
}

/**
 * <!-- ndpolate() -->
 * @brief Runs linear interpolation or extrapolation in n dimensions.
 *
 * @param qpts an #ndp_query_pts instance that holds all query point
 * information
 * @param table an #ndp_table instance that has all identifying information on
 * the interpolating grid itself
 * @param extrapolation_method how extrapolation should be done; one of
 * #NDP_METHOD_NONE, #NDP_METHOD_NEAREST, or #NDP_METHOD_LINEAR.
 *
 * @details
 * This is the main workhorse on the ndpolator module. It assumes that the
 * main #ndp_table @p table structure has been set up. It takes the points of
 * interest, @p qpts, and it calls #ndp_query_pts_import() and
 * #find_hypercubes() consecutively, to populate the #ndp_query structure.
 * While at it, the function also checks whether any of the query point
 * components are out of bounds (flag = #NDP_OUT_OF_BOUNDS) and it prepares
 * those query points for extrapolation, depending on the @p
 * extrapolation_method parameter.
 *
 * Once the supporting structures are initialized and populated, #ndpolate()
 * will first handle the out-of-bounds elements. It will set the value of NAN
 * if @p extrapolation_method = #NDP_METHOD_NONE, find the nearest defined
 * grid vertex by using #find_nearest() and set the value to the found nearest
 * value if @p extrapolation_method = #NDP_METHOD_NEAREST, and lookup the
 * nearest fully defined hypercube for extrapolation if
 * @p extrapolation_method = #NDP_METHOD_LINEAR.
 *
 * Finally, the ndpolator will loop through all hypercubes and call
 * #c_ndpolate() to get the interpolated or extrapolated function values for
 * each query point. The results are stored in the #ndp_query structure.
 *
 * @return a #ndp_query structure that holds all information on the specific
 * ndpolator run.
 */

ndp_query *ndpolate(ndp_query_pts *qpts, ndp_table *table, ndp_extrapolation_method extrapolation_method)
{
    int debug = 0;

    ndp_query *query = ndp_query_new();
    double reduced[table->axes->len];
    ndp_hypercube *hypercube;

    int nelems = qpts->nelems;

    query->hypercubes = find_hypercubes(qpts, table);

    if (debug) {
        for (int i = 0; i < qpts->nelems; i++) {
            ndp_hypercube *hypercube = query->hypercubes[i];
            printf("  hypercube %d: dim=%d vdim=%d fdhc=%d v=[", i, hypercube->dim, hypercube->vdim, hypercube->fdhc);
            for (int j = 0; j < 1 << hypercube->dim; j++) {
                printf("{");
                for (int k = 0; k < hypercube->vdim; k++)
                    printf("%2.2f, ", hypercube->v[j*hypercube->vdim+k]);
                printf("\b\b} ");
            }
            printf("\b] indices=[");
            for (int j = 0; j < table->axes->len; j++)
                printf("%d ", qpts->indices[i*table->axes->len+j]);
            printf("\b] flags=[");
            for (int j = 0; j < table->axes->len; j++)
                printf("%d ", qpts->flags[i*table->axes->len+j]);
            printf("\b]\n");
        }
    }

    query->interps = malloc(nelems * table->vdim * sizeof(*(query->interps)));
    query->dists = calloc(nelems, sizeof(*(query->dists)));

    for (int i = 0; i < nelems; i++) {
        /* handle out-of-bounds elements first: */
        if (!query->hypercubes[i]->fdhc) {
            switch (extrapolation_method) {
                case NDP_METHOD_NONE:
                    for (int j = 0; j < table->vdim; j++)
                        query->interps[i*table->vdim+j] = NAN;
                    continue;
                break;
                case NDP_METHOD_NEAREST: {
                    double *normed_elem = qpts->normed + i * table->axes->len;
                    int *elem_index = qpts->indices + i * table->axes->len;
                    int *elem_flag = qpts->flags + i * table->axes->len;
                    int pos;

                    int *coords = find_nearest(normed_elem, elem_index, elem_flag, table, extrapolation_method, &(query->dists[i]));
                    idx2pos(table->axes, table->vdim, coords, &pos);
                    memcpy(query->interps + i*table->vdim, table->grid + pos, table->vdim * sizeof(*(query->interps)));
                    free(coords);
                    continue;
                }
                break;
                case NDP_METHOD_LINEAR: {
                    double *normed_elem = qpts->normed + i * table->axes->len;
                    int *elem_index = qpts->indices + i * table->axes->len;
                    int *elem_flag = qpts->flags + i * table->axes->len;
                    int cidx[table->axes->len];  /* hypercube corner (given by table->axes->len coordinates) */
                    int pos;

                    /* superior corner coordinates of the nearest fully defined hypercube: */
                    int *coords = find_nearest(normed_elem, elem_index, elem_flag, table, extrapolation_method, &(query->dists[i]));
                    double *hc_vertices = malloc(table->vdim * (1 << table->axes->len) * sizeof(*hc_vertices));

                    if (debug) {
                        printf("  hc %d: normed_elem = [", i);
                        for (int k = 0; k < table->axes->len; k++) {
                            printf("%f ", normed_elem[k]);
                        }
                        printf("\b]\n");
                        printf("  elem_index = [");
                        for (int k = 0; k < table->axes->len; k++) {
                            printf("%d ", elem_index[k]);
                        }
                        printf("\b]\n");
                        printf("  nearest fdhc = [");
                        for (int k = 0; k < table->axes->len; k++) {
                            printf("%d ", coords[k]);
                        }
                        printf("\b]\n");
                    }

                    /* find all hypercube corners: */
                    for (int j = 0; j < (1 << table->axes->len); j++) {
                        for (int k = 0; k < table->axes->len; k++)
                            cidx[k] = max(coords[k]-1+(j / (1 << (table->axes->len-k-1))) % 2, (j / (1 << (table->axes->len-k-1))) % 2);

                        if (debug) {
                            printf("    cidx[%d] = [", j);
                            for (int k = 0; k < table->axes->len; k++)
                                printf("%d ", cidx[k]);
                            printf("\b]\n");
                        }

                        idx2pos(table->axes, table->vdim, cidx, &pos);
                        memcpy(hc_vertices + j * table->vdim, table->grid + pos, table->vdim * sizeof(*hc_vertices));
                    }

                    /* replace the incomplete hypercube with the nearest fully defined hypercube: */
                    ndp_hypercube_free(query->hypercubes[i]);
                    hypercube = query->hypercubes[i] = ndp_hypercube_new_from_data(table->axes->len, table->vdim, /* fdhc = */ 1, hc_vertices);
                    if (debug)
                        ndp_hypercube_print(hypercube, "    ");

                    /* shift normed query points to refer to the nearest fully defined hypercube: */
                    for (int j = 0; j < table->axes->len; j++)
                        qpts->normed[i * table->axes->len + j] += qpts->indices[i * table->axes->len + j] - coords[j];

                    if (debug) {
                        printf("  updated query_pt[%d] = [", i);
                        for (int j = 0; j < table->axes->len; j++)
                            printf("%3.3f ", qpts->normed[i * table->axes->len + j]);
                        printf("\b]\n");
                    }

                    c_ndpolate(hypercube->dim, hypercube->vdim, &qpts->normed[i * table->axes->len], hypercube->v);
                    memcpy(query->interps + i*table->vdim, hypercube->v, table->vdim * sizeof(*(query->interps)));
                    free(coords);
                    continue;
                }
                break;
                default:
                    /* invalid extrapolation method */
                    return NULL;
                break;
            }
        }
        else {
            /* continue with regular interpolation: */
            hypercube = query->hypercubes[i];
        }

        for (int j=0, k=0; j < table->axes->len; j++) {
            /* skip when queried coordinate coincides with a vertex: */
            if (qpts->flags[i * table->axes->len + j] == NDP_ON_VERTEX)
                continue;
            reduced[k] = qpts->normed[i * table->axes->len + j];
            k++;
        }

        if (debug) {
            printf("  i=%d dim=%d vdim=%d nqpts=[", i, hypercube->dim, hypercube->vdim);
            for (int j = 0; j < table->axes->len; j++)
                printf("%2.2f ", qpts->normed[i*table->axes->len + j]);
            printf("\b] reduced=[");
            for (int j = 0; j < hypercube->dim; j++)
                printf("%2.2f ", reduced[j]);
            printf("\b]\n");
        }

        c_ndpolate(hypercube->dim, hypercube->vdim, reduced, hypercube->v);
        memcpy(query->interps + i*table->vdim, hypercube->v, table->vdim * sizeof(*(query->interps)));
    }

    return query;
}

/**
 * <!-- py_import_query_pts() -->
 * @brief Python wrapper to the #ndp_query_pts_import() function.
 *
 * @param self reference to the module object
 * @param args tuple (axes, query_pts)
 *
 * @details
 * The wrapper takes a tuple of axes and an ndarray of query points, and it
 * calls #ndp_query_pts_import() to compute the indices, flags, and
 * unit-normalized query points w.r.t. the corresponding hypercube. These are
 * returned in a tuple to the calling function.
 *
 * @note: In most (if not all) practical circumstances this function should
 * not be used because of the C-python data translation overhead. Instead, use
 * #py_ndpolate() instead as all allocation is done in C.
 *
 * @return a tuple of (indices, flags, normed_query_pts).
 */

static PyObject *py_import_query_pts(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *py_axes, *py_indices, *py_flags, *py_normed_query_pts, *py_combo;
    PyArrayObject *py_query_pts;

    npy_intp *query_pts_shape;

    int naxes, nelems, nbasic;

    double *qpts;

    ndp_axis **axis;
    ndp_axes *axes;
    ndp_query_pts *query_pts;

    static char *kwlist[] = {"axes", "query_pts", "nbasic", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOi", kwlist, &py_axes, &py_query_pts, &nbasic))  /* all borrowed references */
        return NULL;

    naxes = PyTuple_Size(py_axes);
    nelems = PyArray_DIM(py_query_pts, 0);
    qpts = (double *) PyArray_DATA(py_query_pts);

    query_pts_shape = PyArray_SHAPE(py_query_pts);

    axis = malloc(naxes*sizeof(*axis));

    for (int i = 0; i < naxes; i++) {
        PyArrayObject *py_axis = (PyArrayObject *) PyTuple_GetItem(py_axes, i);
        axis[i] = ndp_axis_new_from_data(PyArray_SIZE(py_axis), (double *) PyArray_DATA(py_axis));
    }

    axes = ndp_axes_new_from_data(naxes, nbasic, axis);
    query_pts = ndp_query_pts_import(nelems, qpts, axes);

    /* clean up: */
    ndp_axes_free(axes);

    py_indices = PyArray_SimpleNewFromData(2, query_pts_shape, NPY_INT, query_pts->indices);
    PyArray_ENABLEFLAGS((PyArrayObject *) py_indices, NPY_ARRAY_OWNDATA);

    py_flags = PyArray_SimpleNewFromData(2, query_pts_shape, NPY_INT, query_pts->flags);
    PyArray_ENABLEFLAGS((PyArrayObject *) py_flags, NPY_ARRAY_OWNDATA);

    py_normed_query_pts = PyArray_SimpleNewFromData(2, query_pts_shape, NPY_DOUBLE, query_pts->normed);
    PyArray_ENABLEFLAGS((PyArrayObject *) py_normed_query_pts, NPY_ARRAY_OWNDATA);

    /* free memory that is not passed back to python: */
    free(query_pts->requested);
    free(query_pts);

    py_combo = PyTuple_New(3);
    PyTuple_SET_ITEM(py_combo, 0, py_indices);
    PyTuple_SET_ITEM(py_combo, 1, py_flags);
    PyTuple_SET_ITEM(py_combo, 2, py_normed_query_pts);

    return py_combo;
}

/**
 * <!-- py_hypercubes() -->
 * @brief Python wrapper to the #find_hypercubes() function.
 *
 * @param self reference to the module object
 * @param args tuple (indices, axes, flags, grid)
 *
 * @details
 * The wrapper takes a tuple of indices, axes, flags and function value grid,
 * and it calls #find_hypercubes() to compute the hypercubes, reducing their
 * dimension when possible. Hypercubes are returned to the calling function.
 *
 * @note: In most (if not all) practical circumstances this function should
 * not be used because of the C-python data translation overhead. Instead, use
 * #py_ndpolate() instead as all allocation is done in C.
 *
 * @return an ndarray of hypercubes.
 */

static PyObject *py_hypercubes(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *py_normed_query_pts, *py_indices, *py_flags, *py_grid;
    PyObject *py_axes;

    ndp_table *table;
    ndp_query_pts *qpts;

    double *normed_query_pts;
    int *indices, *flags;
    int nelems, naxes;
    int nbasic = 0;

    PyObject *py_hypercubes;
    ndp_hypercube **hypercubes;

    static char *kwlist[] = {"normed_query_pts", "indices", "axes", "flags", "grid", "nbasic", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOO|i", kwlist, &py_normed_query_pts, &py_indices, &py_axes, &py_flags, &py_grid, &nbasic))
        return NULL;

    nelems = PyArray_DIM(py_indices, 0);
    naxes = PyArray_DIM(py_indices, 1);
    if (nbasic == 0) nbasic = naxes;

    normed_query_pts = (double *) PyArray_DATA(py_normed_query_pts);
    indices = (int *) PyArray_DATA(py_indices);
    flags = (int *) PyArray_DATA(py_flags);

    qpts = ndp_query_pts_new_from_data(nelems, naxes, indices, flags, /* requested= */ NULL, /* normed= */ normed_query_pts);

    py_hypercubes = PyTuple_New(nelems);

    table = ndp_table_new_from_python(py_axes, nbasic, py_grid);

    hypercubes = find_hypercubes(qpts, table);

    for (int i = 0; i < nelems; i++) {
        npy_intp shape[hypercubes[i]->dim+1];
        PyObject *py_hypercube;
        int j;

        for (j = 0; j < hypercubes[i]->dim; j++)
            shape[j] = 2;
        shape[j] = hypercubes[i]->vdim;

        py_hypercube = PyArray_SimpleNewFromData(hypercubes[i]->dim+1, shape, NPY_DOUBLE, hypercubes[i]->v);
        PyArray_ENABLEFLAGS((PyArrayObject *) py_hypercube, NPY_ARRAY_OWNDATA);
        PyTuple_SetItem(py_hypercubes, i, py_hypercube);
    }

    for (int i = 0; i < nelems; i++)
        free(hypercubes[i]);
    /* don't free hypercube data, those go back to python */

    free(hypercubes);
    ndp_table_free(table);
    free(qpts->requested);
    free(qpts);

    return py_hypercubes;
}

/**
 * <!-- py_ainfo -->
 * @private
 * @brief Python wrapper to the #_ainfo() function.
 * 
 * @param self reference to the module object
 * @param args tuple (ndarray | print_data)
 * 
 * @details
 * Prints information on the passed array (its dimensions, flags and content
 * if @p print_data = True).
 * 
 * @return None
 */

static PyObject *py_ainfo(PyObject *self, PyObject *args)
{
    int print_data = 1;
    PyArrayObject *array;

    if (!PyArg_ParseTuple(args, "O|i", &array, &print_data))
        return NULL;

    _ainfo(array, print_data);

    return Py_None;
}

/**
 * <!-- py_ndpolate() -->
 * @brief Python wrapper to the #ndpolate() function.
 *
 * @param self reference to the module object
 * @param args tuple (query_pts, axes, flags, grid | extrapolation_method)
 *
 * @details
 * The wrapper takes a tuple of query points, axes, flags and function value
 * grid, and it calls #ndpolate() to run interpolation and/or extrapolation in
 * all query points. Interpolated/extrapolated values are returned to the
 * calling function in a (nelems-by-vdim)-dimensional ndarray.
 *
 * @note This is the main (and probably the only practical) entry point from
 * python code to the C ndpolator module.
 * 
 * @return an ndarray of interpolated/extrapolated values.
 */

static PyObject *py_ndpolate(PyObject *self, PyObject *args, PyObject *kwargs)
{
    ndp_table *table;
    int capsule_available = 0;

    PyObject *py_rv;

    /* default values: */
    PyObject *py_capsule = NULL;
    PyArrayObject *py_query_pts = NULL;
    PyObject *py_axes = NULL;
    PyArrayObject *py_grid = NULL;
    int nbasic = 0;
    ndp_extrapolation_method extrapolation_method = NDP_METHOD_NONE;

    static char *kwlist[] = {"capsule", "query_pts", "axes", "grid", "nbasic", "extrapolation_method", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOOOii", kwlist, &py_capsule, &py_query_pts, &py_axes, &py_grid, &nbasic, &extrapolation_method))
        return NULL;

    if (PyCapsule_IsValid(py_capsule, NULL)) {
        capsule_available = 1;
        table = (ndp_table *) PyCapsule_GetPointer(py_capsule, NULL);
    }
    else if (py_query_pts && py_axes && py_grid) {
        table = ndp_table_new_from_python(py_axes, nbasic, py_grid);
        py_capsule = PyCapsule_New((void *) table, NULL, NULL);
    }
    else {
        return NULL;
    }

    int nelems = PyArray_DIM(py_query_pts, 0);
    double *qpts = PyArray_DATA(py_query_pts);

    ndp_query_pts *query_pts = ndp_query_pts_import(nelems, qpts, table->axes);
    ndp_query *query = ndpolate(query_pts, table, extrapolation_method);

    npy_intp adim[] = {nelems, table->vdim};
    PyObject *py_interps = PyArray_SimpleNewFromData(2, adim, NPY_DOUBLE, query->interps);
    PyArray_ENABLEFLAGS((PyArrayObject *) py_interps, NPY_ARRAY_OWNDATA);

    npy_intp ddim[] = {nelems, 1};
    PyObject *py_dists = PyArray_SimpleNewFromData(2, ddim, NPY_DOUBLE, query->dists);
    PyArray_ENABLEFLAGS((PyArrayObject *) py_dists, NPY_ARRAY_OWNDATA);

    ndp_query_pts_free(query_pts);
    for (int i = 0; i < nelems; i++)
        ndp_hypercube_free(query->hypercubes[i]);
    /* do not free query->interps or query->dists because they are passed to python. */
    free(query->hypercubes);
    free(query);

    if (capsule_available) {
        py_rv = PyTuple_New(2);
        PyTuple_SetItem(py_rv, 0, py_interps);
        PyTuple_SetItem(py_rv, 1, py_dists);
        return py_rv;
    }

    py_rv = PyTuple_New(3);
    PyTuple_SetItem(py_rv, 0, py_interps);
    PyTuple_SetItem(py_rv, 1, py_dists);
    PyTuple_SetItem(py_rv, 2, py_capsule);
    return py_rv;
}

/**
 * <!-- _register_enum() -->
 * @private
 * @brief Helper function to transfer C enums to Python enums.
 * 
 * @param self reference to the module object
 * @param enum_name Python-side enum name string
 * @param py_enum Python dictionary that defines enumerated constants
 * 
 * @details
 * Registers an enumerated constant in Python.
 */

void _register_enum(PyObject *self, const char *enum_name, PyObject *py_enum)
{
    PyObject *py_enum_class = NULL;
    PyObject *py_enum_module = PyImport_ImportModule("enum");
    if (!py_enum_module)
        Py_CLEAR(py_enum);

    py_enum_class = PyObject_CallMethod(py_enum_module, "IntEnum", "sO", enum_name, py_enum);

    Py_CLEAR(py_enum);
    Py_CLEAR(py_enum_module);

    if (py_enum_class && PyModule_AddObject(self, enum_name, py_enum_class) < 0)
        Py_CLEAR(py_enum_class);
}

/**
 * <!-- ndp_register_enums() -->
 * @brief Translates and registers all C-side enumerated types into Python.
 * 
 * @param self reference to the module object
 * @return #ndp_status
 */

int ndp_register_enums(PyObject *self)
{
    PyObject* py_enum = PyDict_New();

    PyDict_SetItemString(py_enum, "NONE", PyLong_FromLong(NDP_METHOD_NONE));
    PyDict_SetItemString(py_enum, "NEAREST", PyLong_FromLong(NDP_METHOD_NEAREST));
    PyDict_SetItemString(py_enum, "LINEAR", PyLong_FromLong(NDP_METHOD_LINEAR));
    _register_enum(self, "ExtrapolationMethod", py_enum);

    return NDP_SUCCESS;
}

/**
 * @brief Standard python boilerplate code that defines methods present in this C module.
 */

static PyMethodDef cndpolator_methods[] =
{
    {"ndpolate", (PyCFunction) py_ndpolate, METH_VARARGS | METH_KEYWORDS, "C implementation of N-dimensional interpolation"},
    {"find", (PyCFunction) py_import_query_pts, METH_VARARGS | METH_KEYWORDS, "determine indices, flags and normalized query points"},
    {"hypercubes", (PyCFunction) py_hypercubes, METH_VARARGS | METH_KEYWORDS, "determine enclosing hypercubes"},
    {"ainfo", py_ainfo, METH_VARARGS, "array information for internal purposes"},
    {NULL, NULL, 0, NULL}
};

/**
 * @brief Standard python boilerplate code that defines the ndpolator module.
 */

static struct PyModuleDef cndpolator_module = 
{
    PyModuleDef_HEAD_INIT,
    "cndpolator",
    NULL, /* documentation */
    -1,
    cndpolator_methods
};

/**
 * <!-- PyInit_cndpolator() -->
 * @private
 * @brief Initializes the ndpolator C module for Python.
 * 
 * @return PyMODINIT_FUNC 
 */

PyMODINIT_FUNC PyInit_cndpolator(void)
{
    PyObject *module;
    import_array();
    module = PyModule_Create(&cndpolator_module);
    ndp_register_enums(module);
    return module;
}
