#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

/* Required by numpy C-API. It defines a unique symbol to be used in other
 * C source files and header files. */
#define PY_ARRAY_UNIQUE_SYMBOL cndpolator_ARRAY_API

#include "ndpolator.h"
#include "ndp_types.h"

/* Math utility macros */
#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))
#define sign(a) ( ( (a) < 0 )  ?  -1   : ( (a) > 0 ) )

/* Internal function for printing ndarray flags from C for debugging */
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

int find_first_geq_than(ndp_axis *axis, int l, int r, double x, double rtol, int *flag)
{
    int debug = 0;
    int m = l + (r - l) / 2;

    while (l != r) {
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

int ndp_idx2pos(ndp_axes *axes, int vdim, int *index, int *pos)
{
    *pos = axes->cplen[0]*index[0];
    for (int i = 1; i < axes->len; i++)
        *pos += axes->cplen[i]*index[i];
    *pos *= vdim;

    return NDP_SUCCESS;
}

int ndp_pos2idx(ndp_axes *axes, int vdim, int pos, int *idx)
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

/* Linear interpolation and extrapolation on a fully defined hypercube.
 * IMPORTANT: For optimization purposes, this function overwrites fv values.
 * The user must make a copy of the fv array if contents need to be reused.
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
    /* for equal distances select the upper index for consistency: */
    return (((indexed_dists *) a)->idx > ((indexed_dists *) b)->idx) ? 1 : -1;
}

int *ndp_find_nearest(double *normed_elem, int *elem_index, int *elem_flag, ndp_table *table, ndp_extrapolation_method extrapolation_method, ndp_search_algorithm search_algorithm, double *dist)
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

    if (search_algorithm == NDP_SEARCH_KDTREE) {
        /* if this is the first call to find_nearest and k-d trees are used, lazy-load the corresponding spatial index */
        struct kdtree **tree_ptr = extrapolation_method == NDP_METHOD_NEAREST ? &table->vtree : &table->hctree;
        if (!*tree_ptr) {
            *tree_ptr = kd_create(table->axes->nbasic);
            for (int i = 0; i < table->nverts; i++)
                if (mask[i]) {
                    double *coords = malloc(table->axes->nbasic * sizeof(double));

                    for (int j = 0; j < table->axes->nbasic; j++) {
                        coords[j] = i / (table->axes->cplen[j] / table->axes->cplen[table->axes->nbasic-1]) % table->axes->axis[j]->len;
                        /* For hypercube tree: subtract 0.5 from superior corner to get hypercube center */
                        if (extrapolation_method == NDP_METHOD_LINEAR)
                            coords[j] -= 0.5;
                    }

                    kd_insert(*tree_ptr, coords, (void *)(uintptr_t) i);
                }
        }
    }

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

    if (search_algorithm == NDP_SEARCH_KDTREE) {
        struct kdtree *tree = extrapolation_method == NDP_METHOD_NEAREST ? table->vtree : table->hctree;

        /* Convert query point to k-d tree coordinate space */
        double *query_coords = malloc(table->axes->nbasic * sizeof(double));
        
        for (int j = 0; j < table->axes->nbasic; j++) {
            if (extrapolation_method == NDP_METHOD_NEAREST) {
                query_coords[j] = (elem_index[j] - 1) + normed_elem[j];
            } else {
                /* For hypercube trees use hypercube center coordinates */
                query_coords[j] = elem_index[j] + normed_elem[j] - 0.5;
            }
        }

        /* Query the k-d tree */
        struct kdres *result = kd_nearest(tree, query_coords);
        if (result && kd_res_size(result) > 0) {
            /* Extract the vertex index from the result */
            int vertex_idx = (int)(uintptr_t) kd_res_item_data(result);

            /* Calculate distance for consistency with linear search */
            cdist = 0.0;
            for (int j = 0; j < table->axes->nbasic; j++) {
                int coord = vertex_idx / (table->axes->cplen[j] / table->axes->cplen[table->axes->nbasic-1]) % table->axes->axis[j]->len;
                
                if (extrapolation_method == NDP_METHOD_NEAREST) {
                    double offset_normed_elem = elem_index[j] - coord + normed_elem[j];
                    if (normed_elem[j] < 0 || normed_elem[j] > 1)
                        cdist += (offset_normed_elem-1)*(offset_normed_elem-1);
                    else
                        cdist += (round(elem_index[j]+normed_elem[j]-1)-coord)*(round(elem_index[j]+normed_elem[j]-1)-coord);
                }
                
                if (extrapolation_method == NDP_METHOD_LINEAR) {
                    /* For linear method, use the original normalized coordinate directly */
                    double offset_normed_elem = normed_elem[j];
                    if (offset_normed_elem < 0)
                        cdist += offset_normed_elem*offset_normed_elem;
                    else if (offset_normed_elem > 1)
                        cdist += (offset_normed_elem-1)*(offset_normed_elem-1);
                }
            }
            
            *dist = cdist;
            kd_res_free(result);
            free(query_coords);
            free(dists);
            
            /* Convert vertex index back to coordinates */
            int *coords = malloc(table->axes->nbasic * sizeof(int));
            if (!coords) return NULL;
            
            for (int j = 0; j < table->axes->nbasic; j++) {
                coords[j] = vertex_idx / (table->axes->cplen[j] / table->axes->cplen[table->axes->nbasic-1]) % table->axes->axis[j]->len;
            }
            
            if (debug) {
                printf("k-d tree found vertex %d at coords [", vertex_idx);
                for (int j = 0; j < table->axes->nbasic; j++) {
                    printf("%d ", coords[j]);
                }
                printf("\b] with distance %f\n", cdist);
            }
            
            return coords;
        }
        
        kd_res_free(result);
        free(query_coords);
    }

    /* Fallback to linear search */
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

ndp_hypercube **ndp_find_hypercubes(ndp_query_pts *qpts, ndp_table *table)
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

    if (debug)
        ndp_table_print(table);

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

        /* do not check whether the hypercube is fully defined before reducing
         * its dimension: it may happen that we don't need the undefined parts
         * of the hypercube! */

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
                    /* cidx[k] = iptr[k]-1; */
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

            ndp_idx2pos(axes, table->vdim, cidx, &tidx);
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

ndp_query *ndpolate(ndp_query_pts *qpts, ndp_table *table, ndp_extrapolation_method extrapolation_method, ndp_search_algorithm search_algorithm)
{
    int debug = 0;

    ndp_query *query = ndp_query_new();
    double reduced[table->axes->len];
    ndp_hypercube *hypercube;

    query->nelems = qpts->nelems;
    query->extrapolation_method = extrapolation_method;
    query->search_algorithm = search_algorithm;

    query->hypercubes = ndp_find_hypercubes(qpts, table);

    if (debug) {
        for (int i = 0; i < query->nelems; i++) {
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

    query->interps = malloc(query->nelems * table->vdim * sizeof(*(query->interps)));
    query->dists = calloc(query->nelems, sizeof(*(query->dists)));

    for (int i = 0; i < query->nelems; i++) {
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

                    int *coords = ndp_find_nearest(normed_elem, elem_index, elem_flag, table, extrapolation_method, search_algorithm, &(query->dists[i]));
                    ndp_idx2pos(table->axes, table->vdim, coords, &pos);
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
                    int *coords = ndp_find_nearest(normed_elem, elem_index, elem_flag, table, extrapolation_method, search_algorithm, &(query->dists[i]));
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

                        ndp_idx2pos(table->axes, table->vdim, cidx, &pos);
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

/* Python wrapper to the ndp_query_pts_import() function */
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

/* Python wrapper to the ndp_find_hypercubes() function */
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

    hypercubes = ndp_find_hypercubes(qpts, table);

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

/* Python wrapper to the _ainfo() function for debugging */
static PyObject *py_ainfo(PyObject *self, PyObject *args)
{
    int print_data = 1;
    PyArrayObject *array;

    if (!PyArg_ParseTuple(args, "O|i", &array, &print_data))
        return NULL;

    _ainfo(array, print_data);

    return Py_None;
}

/* Python wrapper to the ndpolate() function - main entry point */
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
    ndp_search_algorithm search_algorithm = NDP_SEARCH_KDTREE;

    static char *kwlist[] = {"capsule", "query_pts", "axes", "grid", "nbasic", "extrapolation_method", "search_algorithm", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOOOiii", kwlist, &py_capsule, &py_query_pts, &py_axes, &py_grid, &nbasic, &extrapolation_method, &search_algorithm))
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
    ndp_query *query = ndpolate(query_pts, table, extrapolation_method, search_algorithm);

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

/* Helper function to transfer C enums to Python enums */
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

/* Translates and registers all C-side enumerated types into Python */
int ndp_register_enums(PyObject *self)
{
    PyObject *py_enum;

    py_enum = PyDict_New();
    PyDict_SetItemString(py_enum, "NONE", PyLong_FromLong(NDP_METHOD_NONE));
    PyDict_SetItemString(py_enum, "NEAREST", PyLong_FromLong(NDP_METHOD_NEAREST));
    PyDict_SetItemString(py_enum, "LINEAR", PyLong_FromLong(NDP_METHOD_LINEAR));
    _register_enum(self, "ExtrapolationMethod", py_enum);

    py_enum = PyDict_New();
    PyDict_SetItemString(py_enum, "LINEAR", PyLong_FromLong(NDP_SEARCH_LINEAR));
    PyDict_SetItemString(py_enum, "KDTREE", PyLong_FromLong(NDP_SEARCH_KDTREE));
    _register_enum(self, "SearchAlgorithm", py_enum);

    return NDP_SUCCESS;
}

/* Standard python boilerplate code that defines methods present in this C module */
static PyMethodDef cndpolator_methods[] =
{
    {"ndpolate", (PyCFunction) py_ndpolate, METH_VARARGS | METH_KEYWORDS, "C implementation of N-dimensional interpolation"},
    {"find", (PyCFunction) py_import_query_pts, METH_VARARGS | METH_KEYWORDS, "determine indices, flags and normalized query points"},
    {"hypercubes", (PyCFunction) py_hypercubes, METH_VARARGS | METH_KEYWORDS, "determine enclosing hypercubes"},
    {"ainfo", py_ainfo, METH_VARARGS, "array information for internal purposes"},
    {NULL, NULL, 0, NULL}
};

/* Standard python boilerplate code that defines the ndpolator module */
static struct PyModuleDef cndpolator_module = 
{
    PyModuleDef_HEAD_INIT,
    "cndpolator",
    NULL, /* documentation */
    -1,
    cndpolator_methods
};

/* Initializes the ndpolator C module for Python */
PyMODINIT_FUNC PyInit_cndpolator(void)
{
    PyObject *module;
    import_array();
    module = PyModule_Create(&cndpolator_module);
    ndp_register_enums(module);
    return module;
}
