#include <stdlib.h>
#include <stdio.h>

#include "ndp_types.h"
#include "ndpolator.h"
#include "kdtree.h"

ndp_axis *ndp_axis_new()
{
    ndp_axis *axis = malloc(sizeof(*axis));

    axis->len = 0;
    axis->val = NULL;
    axis->owns_data = 1;

    return axis;
}

ndp_axis *ndp_axis_new_from_data(int len, double *val, int owns_data)
{
    ndp_axis *axis = malloc(sizeof(*axis));

    axis->len = len;
    axis->val = val;
    axis->owns_data = owns_data;

    return axis;
}

int ndp_axis_free(ndp_axis *axis)
{
    if (axis->owns_data && axis->val)
        free(axis->val);
    free(axis);

    return NDP_SUCCESS;
}

ndp_axes *ndp_axes_new()
{
    ndp_axes *axes = malloc(sizeof(*axes));

    axes->len = 0;
    axes->nbasic = 0;
    axes->axis = NULL;
    axes->cplen = NULL;

    return axes;
}

ndp_axes *ndp_axes_new_from_data(int naxes, int nbasic, ndp_axis **axis)
{
    ndp_axes *axes = ndp_axes_new();

    axes->len = naxes;
    axes->nbasic = nbasic;
    axes->axis = axis;

    /* add a cumulative product array: */
    axes->cplen = malloc(naxes*sizeof(*(axes->cplen)));
    for (int i = 0; i < naxes; i++) {
        axes->cplen[i] = 1.0;
        for (int j = i+1; j < naxes; j++)
            axes->cplen[i] *= axes->axis[j]->len;
    }

    return axes;
}

int ndp_axes_free(ndp_axes *axes)
{
    if (axes->cplen)
        free(axes->cplen);

    for (int i = 0; i < axes->len; i++)
        ndp_axis_free(axes->axis[i]);

    free(axes->axis);
    free(axes);

    return NDP_SUCCESS;
}

ndp_query_pts *ndp_query_pts_new()
{
    ndp_query_pts *qpts = malloc(sizeof(*qpts));

    qpts->nelems = 0;
    qpts->naxes = 0;
    qpts->indices = NULL;
    qpts->flags = NULL;
    qpts->requested = NULL;
    qpts->normed = NULL;

    return qpts;
}

ndp_query_pts *ndp_query_pts_new_from_data(int nelems, int naxes, int *indices, int *flags, double *requested, double *normed)
{
    ndp_query_pts *qpts = malloc(sizeof(*qpts));

    qpts->nelems = nelems;
    qpts->naxes = naxes;
    qpts->indices = indices;
    qpts->flags = flags;
    qpts->requested = requested;
    qpts->normed = normed;

    return qpts;
}

int ndp_query_pts_alloc(ndp_query_pts *qpts, int nelems, int naxes)
{
    qpts->nelems = nelems;
    qpts->naxes = naxes;

    qpts->indices = malloc(nelems * naxes * sizeof(*(qpts->indices)));
    qpts->flags = malloc(nelems * naxes * sizeof(*(qpts->flags)));
    qpts->requested = malloc(nelems * naxes * sizeof(*(qpts->requested)));
    qpts->normed = malloc(nelems * naxes * sizeof(*(qpts->normed)));

    return NDP_SUCCESS;
}

int ndp_query_pts_free(ndp_query_pts *qpts)
{
    if (qpts->indices)
        free(qpts->indices);
    if (qpts->flags)
        free(qpts->flags);
    if (qpts->requested)
        free(qpts->requested);
    if (qpts->normed)
        free(qpts->normed);
    
    free(qpts);

    return NDP_SUCCESS;
}

ndp_table *ndp_table_new()
{
    ndp_table *table = malloc(sizeof(*table));
    table->vdim = 0;
    table->axes = NULL;
    table->grid = NULL;
    table->owns_data = 1;

    table->vtree = NULL;
    table->hctree = NULL;

    table->nverts = 0;
    table->vmask = NULL;
    table->hcmask = NULL;

    return table;
}

ndp_table *ndp_table_new_from_data(ndp_axes *axes, int vdim, double *grid, int owns_data)
{
    int debug = 0;
    int pos;
    int cpsum = 0;
    int ith_corner[axes->nbasic], cidx[axes->nbasic];

    ndp_table *table = ndp_table_new();

    table->axes = axes;
    table->owns_data = owns_data;
    table->vdim = vdim;
    table->grid = grid;

    /* count all vertices in the grid: */
    table->nverts = 1;
    for (int i = 0; i < axes->nbasic; i++)
        table->nverts *= axes->axis[i]->len;

    /* collect all non-nan vertices: */
    table->vmask = calloc(table->nverts, sizeof(*(table->vmask)));
    for (int i = 0; i < table->nverts; i++) {
        pos = i*axes->cplen[axes->nbasic-1]*vdim;
        if (grid[pos] == grid[pos])  /* false if nan */
            table->vmask[i] = 1;
    }

    /* cpsum is the number of hypercube vertices on the lower edge */
    for (int i = 0; i < axes->nbasic; i++)
        cpsum += axes->cplen[i];
    cpsum /= axes->cplen[axes->nbasic-1];

    if (debug) {
        printf("axlen=[");
        for (int i = 0; i < axes->len; i++)
            printf("%d ", axes->axis[i]->len);
        printf("\b] cplen=[");
        for (int i = 0; i < axes->len; i++)
            printf("%d ", axes->cplen[i]);
        printf("\b] cpsum=%d nverts=%d\n", cpsum, table->nverts);
    }

    table->hcmask = calloc(table->nverts, sizeof(*(table->hcmask)));
    for (int i = cpsum; i < table->nverts; i++) {
        int nan_encountered = 0;

        /* skip undefined vertices: */
        if (table->vmask[i] == 0)
            continue;

        /* convert running index to per-axis indices of the superior corner of the hypercube: */
        for (int k = 0; k < axes->nbasic; k++) {
            ith_corner[k] = (i / (axes->cplen[k] / axes->cplen[axes->nbasic-1])) % axes->axis[k]->len;
            if (debug)
                printf("i=%d k=%d normed_cplen[k]=%d ith_corner[k]=%d\n", i, k, axes->cplen[k]/axes->cplen[axes->nbasic-1], i / (axes->cplen[k] / axes->cplen[axes->nbasic-1]));
            /* skip edge elements: */
            if (ith_corner[k] == 0) {
                nan_encountered = 1;
                break;
            }
        }

        if (nan_encountered)
            continue;

        if (debug) {
            printf("corners of hc=[");
            for (int k = 0; k < axes->nbasic; k++)
                printf("%d ", ith_corner[k]);
            printf("\b]:\n");
        }

        /* loop over all basic hypercube vertices and see if they're all defined: */
        for (int j = 0; j < 1 << table->axes->nbasic; j++) {                
            for (int k = 0; k < table->axes->nbasic; k++)
                cidx[k] = ith_corner[k]-1+(j / (1 << (table->axes->nbasic-k-1))) % 2;

            if (debug) {
                printf("  c%d=[", j);
                for (int k = 0; k < table->axes->nbasic; k++)
                    printf("%d ", cidx[k]);
                printf("\b]\n");
            }

            /* convert per-axis indices to running index: */
            pos = 0;
            for (int k = 0; k < table->axes->nbasic; k++)
                pos += cidx[k] * axes->cplen[k] / axes->cplen[axes->nbasic-1];

            if (!table->vmask[pos]) {
                nan_encountered = 1;
                break;
            }
        }

        if (nan_encountered)
            continue;
        
        table->hcmask[i] = 1;
    }

    if (debug) {
        for (int i = 0, sum = 0; i < table->nverts; i++) {
            sum += table->hcmask[i];
            if (i == table->nverts-1)
                printf("%d fully defined hypercubes found.\n", sum);
        }
    }

    return table;
}

void ndp_table_print(ndp_table *table)
{
    int undef = 0;
    printf("ndp_table properties:\n");
    printf("  vdim = %d\n", table->vdim);
    printf("  naxes = %d\n", table->axes->len);
    printf("    axlen = [");
    for (int i = 0; i < table->axes->len; i++)
        printf("%d ", table->axes->axis[i]->len);
    printf("\b]\n");
    printf("  basic vertices = %d\n", table->nverts);
    for (int i = 0; i < table->nverts; i++)
        undef += (1-table->vmask[i]);
    printf("  undefined vertices = %d\n", undef);
}

int ndp_table_free(ndp_table *table)
{
    if (table->axes)
        ndp_axes_free(table->axes);

    if (table->owns_data && table->grid)
        free(table->grid);

    if (table->vtree)
        kd_free(table->vtree);

    if (table->hctree)
        kd_free(table->hctree);

    if (table->vmask)
        free(table->vmask);

    if (table->hcmask)
        free(table->hcmask);

    free(table);

    return NDP_SUCCESS;
}

ndp_hypercube *ndp_hypercube_new()
{
    ndp_hypercube *hc = malloc(sizeof(*hc));
    hc->dim = 0;
    hc->vdim = 0;
    hc->fdhc = 0;
    hc->v = NULL;
    return hc;
}

ndp_hypercube *ndp_hypercube_new_from_data(int dim, int vdim, int fdhc, double *v)
{
    ndp_hypercube *hc = malloc(sizeof(*hc));

    hc->dim = dim;
    hc->vdim = vdim;
    hc->fdhc = fdhc;
    hc->v = v;

    return hc;
}

int ndp_hypercube_alloc(ndp_hypercube *hc, int dim, int vdim)
{
    hc->dim = dim;
    hc->vdim = vdim;
    hc->fdhc = 0;
    hc->v = malloc(vdim * (1 << dim) * sizeof(*(hc->v)));

    return NDP_SUCCESS;
}

/* Helper function that prints hypercube values for debugging */
void ndp_hypercube_print(ndp_hypercube *hc, const char *prefix)
{
    printf("%shc->dim = %d\n", prefix, hc->dim);
    printf("%shc->vdim = %d\n", prefix, hc->vdim);
    printf("%shc->fdhc = %d\n", prefix, hc->fdhc);

    printf("%shc->v = [", prefix);
    for (int i = 0; i < (1<<hc->dim); i++) {
        printf("{");
        for (int j = 0; j < hc->vdim; j++) {
            printf("%f ", hc->v[i*hc->vdim+j]);
        }
        printf("\b}, ");
    }
    printf("\b\b]\n");
}

int ndp_hypercube_free(ndp_hypercube *hc)
{
    free(hc->v);
    free(hc);
    return NDP_SUCCESS;
}

ndp_query *ndp_query_new()
{
    ndp_query *query = malloc(sizeof(*query));

    query->nelems = 0;
    query->hypercubes = NULL;
    query->interps = NULL;
    query->dists = NULL;

    query->extrapolation_method = NDP_METHOD_NONE;
    query->search_algorithm = NDP_SEARCH_KDTREE;

    return query;
}

int ndp_query_free(ndp_query *query)
{
    if (query->interps)
        free(query->interps);
    if (query->dists)
        free(query->dists);
    if (query->hypercubes) {
        for (int i = 0; i < query->nelems; i++)
            ndp_hypercube_free(query->hypercubes[i]);
        free(query->hypercubes);
    }

    free(query);

    return NDP_SUCCESS;
}
