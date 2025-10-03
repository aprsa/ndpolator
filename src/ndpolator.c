/* Core ndpolator functionality
 * 
 * This file contains the core C implementation of n-dimensional interpolation.
 */

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ndpolator.h"
#include "ndp_types.h"

/* Math utility macros */
#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))
#define sign(a) ( ( (a) < 0 )  ?  -1   : ( (a) > 0 ) )

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

    for (int i = 0; i < axes->len; i++)
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

struct kdtree *ndp_kdtree_create(ndp_table *table, ndp_extrapolation_method extrapolation_method)
{
    struct kdtree *tree = kd_create(table->axes->nbasic);
    int *mask = extrapolation_method == NDP_METHOD_NEAREST ? table->vmask : table->hcmask;
    
    for (int i = 0; i < table->nverts; i++) {
        if (mask[i]) {
            double *node = malloc(table->axes->nbasic * sizeof(*node));
            
            for (int j = 0; j < table->axes->nbasic; j++) {
                node[j] = i / (table->axes->cplen[j] / table->axes->cplen[table->axes->nbasic-1]) % table->axes->axis[j]->len;
                /* For hypercube tree: subtract 0.5 from superior corner to get hypercube center */
                if (extrapolation_method == NDP_METHOD_LINEAR)
                    node[j] -= 0.5;
            }
            
            kd_insert(tree, node, (void *)(uintptr_t) i);
            free(node);
        }
    }
    
    return tree;
}

double ndp_distance(double *normed_elem, int *elem_index, int *nearest_coords, ndp_table *table)
{
    /* Computes the squared distance from the query point to the nearest edge/face/vertex 
     * of the nearest hypercube (for LINEAR method only).
     * 
     * For BASIC axes (0 to nbasic-1):
     *   nearest_coords[j] is the superior corner of the hypercube.
     *   The hypercube spans from nearest_coords[j]-1 to nearest_coords[j].
     *   Distance is 0 if inside hypercube, otherwise distance to nearest edge/face/vertex.
     *
     * For ASSOCIATED axes (nbasic to len-1):
     *   nearest_coords[j] is the rounded grid coordinate.
     *   Distance is the absolute distance from query point to this coordinate.
     *   (Associated axes are not part of the hypercube structure for LINEAR interpolation;
     *    they simply contribute their deviation from the nearest grid point)
     */

    double cdist = 0.0;

    /* Basic axes: hypercube edge/face/vertex logic */
    for (int j = 0; j < table->axes->nbasic; j++) {
        double query_coord = elem_index[j] + normed_elem[j] - 1.0;
        double diff;
        
        if (query_coord < nearest_coords[j] - 1.0) {
            /* Query point is below the hypercube */
            diff = (nearest_coords[j] - 1.0) - query_coord;
        }
        else if (query_coord > nearest_coords[j]) {
            /* Query point is above the hypercube */
            diff = query_coord - nearest_coords[j];
        }
        else {
            /* Query point is inside the hypercube in this dimension */
            diff = 0.0;
        }
        
        cdist += diff * diff;
    }
    
    /* Associated axes: distance from rounded coordinate */
    for (int j = table->axes->nbasic; j < table->axes->len; j++) {
        double query_coord = elem_index[j] + normed_elem[j] - 1.0;
        double diff = query_coord - nearest_coords[j];
        cdist += diff * diff;
    }

    return cdist;
}

int *ndp_find_nearest(double *normed_elem, int *elem_index, int *elem_flag, ndp_table *table, ndp_extrapolation_method extrapolation_method, ndp_search_algorithm search_algorithm, double *dist)
{
    int debug = 0;
    int min_pos;
    double cdist = 0.0;
    int nearest;

    typedef struct {
        int idx;
        double dist;
    } indexed_dists;

    indexed_dists *dists = malloc(table->nverts * sizeof(*dists));

    int *coords = malloc(table->axes->len * sizeof(*coords));
    int *mask = extrapolation_method == NDP_METHOD_NEAREST ? table->vmask : table->hcmask;
    struct kdtree **tree_ptr = extrapolation_method == NDP_METHOD_NEAREST ? &table->vtree : &table->hctree;

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

    if (search_algorithm == NDP_SEARCH_KDTREE && !*tree_ptr) {
        /* lazy-load the spatial index */
        *tree_ptr = ndp_kdtree_create(table, extrapolation_method);
    }

    if (search_algorithm == NDP_SEARCH_KDTREE) {
        double query_coords[table->axes->nbasic];
        for (int j = 0; j < table->axes->nbasic; j++)
            query_coords[j] = elem_index[j] + normed_elem[j] - 1.0;

        /* Query the k-d tree */
        struct kdres *result = kd_nearest(*tree_ptr, query_coords);

        if (result && kd_res_size(result) > 0) {
            nearest = (int)(uintptr_t) kd_res_item_data(result);
            
            /* Convert vertex index to grid coordinates (basic axes) */
            for (int j = 0; j < table->axes->nbasic; j++) {
                coords[j] = nearest / (table->axes->cplen[j] / table->axes->cplen[table->axes->nbasic-1]) % table->axes->axis[j]->len;
            }
            
            /* Add associated axes: */
            for (int j = table->axes->nbasic; j < table->axes->len; j++) {
                coords[j] = max(0, min(table->axes->axis[j]->len-1, round(elem_index[j]+normed_elem[j]-1)));
            }
            
            /* Calculate squared distance */
            if (extrapolation_method == NDP_METHOD_NEAREST) {
                /* For NEAREST method, calculate squared distance from query point to the found vertex.
                 * Include ALL axes - a fully defined vertex requires being on-grid for all dimensions. */
                *dist = 0.0;
                for (int j = 0; j < table->axes->len; j++) {
                    double query_coord = elem_index[j] + normed_elem[j] - 1.0;
                    double diff = query_coord - coords[j];
                    *dist += diff * diff;
                }
            }
            else if (extrapolation_method == NDP_METHOD_LINEAR) {
                /* For LINEAR method, calculate squared distance to nearest edge/face/vertex of hypercube */
                *dist = ndp_distance(normed_elem, elem_index, coords, table);
            }
            
            if (debug) {
                printf("k-d tree found vertex/hypercube %d at coords [", nearest);
                for (int j = 0; j < table->axes->len; j++) {
                    printf("%d ", coords[j]);
                }
                printf("\b] with distance %f\n", *dist);
            }
            
            kd_res_free(result);
            free(dists);
            return coords;
        }
        
        kd_res_free(result);
        /* Fall through to linear search if kdtree query failed */
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

        /* Convert vertex index to grid coordinates (basic axes) */
        int temp_coords[table->axes->len];
        for (int j = 0; j < table->axes->nbasic; j++) {
            temp_coords[j] = i / (table->axes->cplen[j] / table->axes->cplen[table->axes->nbasic-1]) % table->axes->axis[j]->len;
        }
        
        /* Add associated axes coordinates (nearest grid point to query) */
        for (int j = table->axes->nbasic; j < table->axes->len; j++) {
            temp_coords[j] = max(0, min(table->axes->axis[j]->len-1, round(elem_index[j]+normed_elem[j]-1)));
        }

        /* Calculate squared distance */
        if (extrapolation_method == NDP_METHOD_NEAREST) {
            /* For NEAREST method, calculate squared Euclidean distance to vertex.
             * Include ALL axes - a fully defined vertex requires being on-grid for all dimensions. */
            cdist = 0.0;
            for (int j = 0; j < table->axes->len; j++) {
                double query_coord = elem_index[j] + normed_elem[j] - 1.0;
                double diff = query_coord - temp_coords[j];
                cdist += diff * diff;
            }
        }
        else if (extrapolation_method == NDP_METHOD_LINEAR) {
            /* For LINEAR method, calculate squared distance to nearest edge/face/vertex of hypercube */
            cdist = ndp_distance(normed_elem, elem_index, temp_coords, table);
        }

        if (debug) {
            printf("  i=% 4d coord=[", i);
            for (int j = 0; j < table->axes->nbasic; j++) {
                printf("%d ", temp_coords[j]);
            }
            printf("\b] dist=%f\n", cdist);
        }

        dists[i].dist = cdist;
    }

    /* sort the distances: */
    qsort(dists, table->nverts, sizeof(*dists), _compare_indexed_dists);
    *dist = dists[0].dist;
    min_pos = dists[0].idx;

    if (debug)
        printf("  min_dist=%f min_pos=%d nearest=[", *dist, dists[0].idx);

    /* Assemble the coordinates: */
    for (int j = 0; j < table->axes->nbasic; j++) {
        coords[j] = min_pos / (table->axes->cplen[j] / table->axes->cplen[table->axes->nbasic-1]) % table->axes->axis[j]->len;
        if (debug)
            printf("%d ", coords[j]);
    }

    for (int j = table->axes->nbasic; j < table->axes->len; j++) {
        coords[j] = max(0, min(table->axes->axis[j]->len-1, round(elem_index[j]+normed_elem[j]-1)));
        if (debug)
            printf("%d ", coords[j]);
    }

    if (debug)
        printf("\b]\n");

    free(dists);
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
