/**
 * @file ndp_types.h
 * @brief Ndpolator's type definitions and prototypes for instantiating and
 * destroying data structures.
 */

#ifndef NDP_TYPES_H
    #define NDP_TYPES_H 1

/* Include k-d tree library */
#include "kdtree.h"

/** @defgroup enumerators Enumerations
  * @brief Core enumeration types defined and used by ndpolator
  * 
  * This section defines the core enumeration types used throughout the ndpolator library.
  */

/** @defgroup data_structures Ndpolator structures
  * @brief Core data types defined and used by ndpolator
  * 
  * These structures define the grid, axes, and query data that ndpolator works with.
  */

/** @defgroup constructors Memory Management
  * @brief Functions for instantiating and destroying ndpolator data structures
  * 
  * Use these functions to properly allocate and free memory for ndpolator structures.
  */

/** @defgroup convenience Convenience Functions
  * @brief Helper functions that help inspect and debug ndpolator structures
  * 
  * These functions are used to validate and inspect ndpolator structures.
  */


/** @enum ndp_status
  * @ingroup enumerators
  * @brief Return codes for ndpolator functions.
  * 
  * @details
  * All ndpolator functions that do not allocate memory should return
  * ndp_status. If no exceptions occurred, NDP_SUCCESS should be returned.
  * Otherwise, a suitable error code enumerated by ndp_status should be returned.
  */

enum ndp_status {
    NDP_SUCCESS = 0,  /*!< Normal exit, no exceptions occurred */
    NDP_INVALID_TYPE  /*!< Invalid argument type, function cannot continue */
};

typedef enum ndp_status ndp_status;

/** @struct ndp_axis
  * @ingroup data_structures
  * @brief Single axis structure containing vertices for one dimension.
  * @details
  * An axis, in ndpolator language, is an array of length len and vertices
  * val. Axes span ndpolator dimensions: for `N`-dimensional interpolation
  * and/or extrapolation, there need to be `N` axes. Note that axes themselves
  * do not have any function values associated to them; they only span the
  * `N`-dimensional grid.
  */

typedef struct ndp_axis {
    int len;       /*!< Axis length (number of vertices) */
    double *val;   /*!< Axis vertices array */
    int owns_data; /*!< Flag indicating if ndp_axis owns the val array */
} ndp_axis;

/** @fn ndp_axis *ndp_axis_new(void)
  * @ingroup constructors
  * @brief Default constructor for ndp_axis.
  *
  * @details
  * Initializes a new ndp_axis instance, sets axis->len to 0 and
  * axis->val to NULL.
  *
  * @return Initialized ndp_axis instance
  */
 
ndp_axis *ndp_axis_new();

/** @fn ndp_axis *ndp_axis_new_from_data(int len, double *val, int owns_data)
  * @ingroup constructors
  * @brief Constructor for ndp_axis from existing data.
  *
  * @param len Length of the val array
  * @param val Array of vertices that span the axis
  * @param owns_data Flag indicating if the val array is owned by self
  *
  * @details
  * Initializes a new ndp_axis instance, sets axis->len to len and axis->val
  * to val. Note that the function does not copy the array, it only assigns a
  * pointer to it. Thus, the calling function needs to determine data
  * ownership and pass an allocated copy if the array is (re)used elsewhere.
  * Ndpolator treats the val array as read-only and will not change it. On
  * destruction, the val array will be freed only if owns_data is set to true.
  *
  * @return Initialized ndp_axis instance
  */

ndp_axis *ndp_axis_new_from_data(int len, double *val, int owns_data);

/** @fn int ndp_axis_free(ndp_axis *axis)
  * @ingroup constructors
  * @brief Destructor for ndp_axis.
  * 
  * @param axis ndp_axis instance to be freed
  * 
  * @details
  * Frees memory allocated for the ndp_axis instance. This includes the
  * val array memory and the ndp_axis instance itself.
  * 
  * @return ndp_status code
  */

int ndp_axis_free(ndp_axis *axis);

/** @struct ndp_axes
  * @ingroup data_structures
  * 
  * @brief Multi-axis collection with basic and associated axes.
  * 
  * @details
  * This structure stores all axes that span the ndpolator grid. Each axis must
  * be of the ndp_axis type. Function values are associated to each
  * combination (cartesian product) of axis indices.
  *
  * There are two types of axes that ndpolator recognizes: _basic_ and
  * _associated_. Basic axes span the sparse grid: function values can either
  * be defined, or null. Associated axes, on the other hand, are _guaranteed_
  * to have function values defined for all combinations of basic indices that
  * have function values defined. For example, if `(i, j, k)` are basic indices
  * that have a defined function value, then `(i, j, k, l, m)` are guaranteed
  * to be defined as well, where `l` and `m` index associated axes.
  */

typedef struct ndp_axes {
    int len;          /*!< Number of axes */
    int nbasic;       /*!< Number of basic axes (must precede any associated axes in the axis array) */
    ndp_axis **axis;  /*!< Array of ndp_axis pointers */
    int *cplen;       /*!< @private Cumulative product of axis lengths for indexing */
} ndp_axes;

/** @fn ndp_axes *ndp_axes_new(void)
  * @ingroup constructors
  * @brief Default constructor for ndp_axes.
  * 
  * @details
  * Initializes a new ndp_axes instance, sets axes->len and axes->nbasic to 0, and
  * sets axes->axis and axes->cplen to NULL.
  * 
  * @return Initialized ndp_axes instance with default values
  */

ndp_axes *ndp_axes_new();

/** @fn ndp_axes *ndp_axes_new_from_data(int naxes, int nbasic, ndp_axis
  * **axis)
  * @ingroup constructors
  * @brief Constructor for ndp_axes from existing data.
  *
  * @param naxes Number of axes
  * @param nbasic Number of basic axes
  * @param axis Array of ndp_axis pointers
  *
  * @return Initialized ndp_axes instance
  *
  * @details
  * Initializes a new ndp_axes instance, sets axes->len to naxes, axes->nbasic
  * to nbasic, and axes->axis to axis. Note that the function does not copy
  * the array, it only assigns a pointer to it. Thus, the calling function
  * needs to pass an allocated copy if the array is (re)used elsewhere.
  * Ndpolator treats the axis array as read-only and will not change it. On
  * destruction, the axis array, along with all individual axis instances,
  * will be freed.
  */

ndp_axes *ndp_axes_new_from_data(int naxes, int nbasic, ndp_axis **axis);

/** @fn int ndp_axes_free(ndp_axes *axes)
  * @ingroup constructors
  * @brief Destructor for ndp_axes.
  * 
  * @param axes ndp_axes instance to be freed
  * 
  * @details
  * Frees memory allocated for the ndp_axes instance. This includes the
  * cplen array memory and the ndp_axes instance, along with all individual
  * axis instances.
  * 
  * @return ndp_status code
  */

int ndp_axes_free(ndp_axes *axes);

/** @struct ndp_query_pts
  * @ingroup data_structures
  * @brief Query points structure with indices, flags and coordinates.
  * @details
  *
  * Query points (points of interest) are given by n coordinates that
  * correspond to n ndp_axis instances stored in ndp_axes. Their number is
  * given by the nelems field and their dimension by the naxes field. The
  * indices array provides superior corners of the hypercube that contains a
  * query point; the flags array tags each query point component with one of
  * the ndp_vertex_flag flags: NDP_ON_GRID, NDP_ON_VERTEX, or
  * NDP_OUT_OF_BOUNDS. The actual query points (as passed to ndpolator, in
  * axis units) are stored in the requested array, and the unit-hypercube
  * normalized coordinates are stored in the normed array.
  */

typedef struct ndp_query_pts {
    int nelems;         /*!< Number of query points */
    int naxes;          /*!< Query point dimension (number of axes) */
    int *indices;       /*!< Array of superior hypercube indices */
    int *flags;         /*!< Array of flags, one per query point component */
    double *requested;  /*!< Array of absolute query points (in axis units) */
    double *normed;     /*!< Array of unit-hypercube normalized query points */
} ndp_query_pts;

/** @fn ndp_query_pts *ndp_query_pts_new(void)
  * @ingroup constructors
  * @brief Default constructor for ndp_query_pts.
  * 
  * @details
  * Initializes a new ndp_query_pts instance, sets nelems and naxes to 0, and
  * sets indices, flags, requested, and normed array pointers to NULL.
  * 
  * @return Initialized ndp_query_pts instance
  */

ndp_query_pts *ndp_query_pts_new();

/** @fn ndp_query_pts *ndp_query_pts_new_from_data(int nelems, int naxes, int
  * *indices, int *flags, double *requested, double *normed)
  * @ingroup constructors
  * @brief Constructor for ndp_query_pts from existing data.
  *
  * @param nelems Number of query points
  * @param naxes Number of axes (query point dimension)
  * @param indices Flattened array of parent hypercube indices, of length
  *     nelems * naxes
  * @param flags Flattened array of ndp_vertex_flag flags, one for each query
  *     point component, of length nelems * naxes
  * @param requested Flattened array of query points (in physical/axis units),
  *     of length nelems * naxes
  * @param normed Flattened array of hypercube-normalized query points (in
  *     grid/tick units), of length nelems * naxes
  *
  * @details
  * Initializes a new ndp_query_pts instance, sets nelems and naxes to the
  * provided values, and points the indices, flags, requested, and normed
  * arrays to the provided arrays. Memory for these arrays must be allocated
  * by the caller. Ndpolator treats these arrays as read-only and will not
  * change them. On destruction, the arrays will be freed along with the
  * ndp_query_pts instance.
  *
  * @return Initialized ndp_query_pts instance
  */

ndp_query_pts *ndp_query_pts_new_from_data(int nelems, int naxes, int *indices, int *flags, double *requested, double *normed);

/** @fn int ndp_query_pts_alloc(ndp_query_pts *qpts, int nelems, int naxes)
  * @ingroup constructors
  * @brief Allocates memory for ndp_query_pts arrays.
  * 
  * @param qpts ndp_query_pts instance to allocate memory for
  * @param nelems Number of query points
  * @param naxes Number of axes (query point dimension)
  * 
  * @details
  * Allocates memory for the indices, flags, requested, and normed arrays;
  * each array will be of length nelems * naxes.
  * 
  * @return ndp_status code
  */

int ndp_query_pts_alloc(ndp_query_pts *qpts, int nelems, int naxes);

/** @fn int ndp_query_pts_free(ndp_query_pts *qpts)
  * @ingroup constructors
  * @brief Destructor for ndp_query_pts.
  * 
  * @param qpts ndp_query_pts instance to be freed
  * 
  * @details
  * Frees memory allocated for the ndp_query_pts instance. This includes the
  * indices, flags, requested, and normed arrays.
  * 
  * @return ndp_status code
  */

int ndp_query_pts_free(ndp_query_pts *qpts);

/** @struct ndp_table
  * @ingroup data_structures
  * @struct ndp_table
  * @brief self-contained ndpolation table with axes, grid values and spatial indices.
  *
  * @details
  * Ndpolator uses #ndp_table to store all relevant parameters for
  * interpolation and/or extrapolation. It stores the axes that span the
  * interpolation hyperspace (in a #ndp_axes structure), the function values
  * across the interpolation hyperspace (grid and kdtree), function value
  * length (vdim), and several private fields that further optimize
  * interpolation.
  */

typedef struct ndp_table {
    int vdim;               /*!< Vertex dimension (i.e., function value length): 1 for scalars, >1 for arrays */
    ndp_axes *axes;         /*!< ndp_axes instance that defines all axes */
    double *grid;           /*!< Array holding all function values in C-native order */
    int owns_data;          /*!< Flag indicating if ndp_table owns the grid array */
    struct kdtree *vtree;   /*!< Vertex k-d tree spatial index for nearest neighbor search */
    struct kdtree *hctree;  /*!< Hypercube k-d tree spatial index for nearest neighbor search */
    int nverts;             /*!< @private Number of basic grid points (cartesian product of all basic axes) */
    int *vmask;             /*!< @private Mask of defined grid points (nverts length) */
    int *hcmask;            /*!< @private Mask of fully defined hypercubes (nverts length) */
} ndp_table;

/** @fn ndp_table *ndp_table_new(void)
  * @ingroup constructors
  * @brief Default constructor for ndp_table.
  * 
  * @details
  * Initializes a new ndp_table instance, sets vdim to 0, and
  * sets axes, grid, vtree and hctree to NULL.
  * 
  * @return Initialized ndp_table instance
  */

ndp_table *ndp_table_new();

/** @fn ndp_table *ndp_table_new_from_data(ndp_axes *axes, int vdim, double *grid, int owns_data)
  * @ingroup constructors
  * @brief Constructor for ndp_table from existing data.
  *
  * @param axes ndp_axes instance defining all axes
  * @param vdim Vertex dimension (function value length)
  * @param grid Flattened array holding all function values, in C-native
  *     order, of size vdim * nverts
  * @param owns_data Flag indicating if the grid array is owned by self
  *
  * @details
  * Initializes a new ndp_table instance from passed data. Note that the
  * function does not copy the arrays, it only assigns pointers to them. Thus,
  * the calling function needs to determine data ownership and pass an
  * allocated copy if the array is (re)used elsewhere. Ndpolator treats all
  * arrays as read-only and will not change them. On destruction, the grid
  * array will be freed only if owns_data is set to true.
  *
  * This constructor also initializes a private list of all non-nan vertices
  * and fully defined hypercubes in the grid. It does so by traversing the
  * grid and storing their count in a private array. These fields are only
  * used internally by ndpolator for the linear search of nearest neighbors
  * and are likely to be made obsolete in the future.
  *
  * @return Initialized ndp_table instance
  */

ndp_table *ndp_table_new_from_data(ndp_axes *axes, int vdim, double *grid, int owns_data);

/** @fn void ndp_table_print(ndp_table *table)
  * @ingroup convenience
  * @brief Prints ndp_table information.
  *
  * @param table ndp_table instance to print
  *
  * @details
  * Prints basic information about the ndp_table instance, including vertex
  * (function value) dimension, number of axes, axis lengths, number of basic
  * vertices, and number of undefined vertices.
  */

void ndp_table_print(ndp_table *table);

/** @fn int ndp_table_free(ndp_table *table)
  * @ingroup constructors
  * @brief Destructor for ndp_table.
  * 
  * @param table ndp_table instance to be freed
  * 
  * @details
  * Frees memory allocated for the ndp_table instance. This includes the
  * axes, grid, vtree and hctree arrays. All private structures are also
  * freed.
  * 
  * @return ndp_status code
  */

int ndp_table_free(ndp_table *table);

/** @struct ndp_hypercube
  * @ingroup data_structures
  * @struct ndp_hypercube
  * 
  * @brief N-dimensional hypercube containing vertex values for interpolation.
  * 
  * @details
  * Hypercubes are subgrids that enclose (or are adjacent to, in the case of
  * extrapolation) the passed query points, one per query point. They are
  * qualified by their dimension (an N-dimensional hypercube has 2<sup>N</sup>
  * vertices) and their function value length. Note that hypercube dimension
  * can be less than the dimension of the grid itself: if a query point
  * coincides with any of the axes, that will reduce the dimensionality of the
  * hypercube. If all query point components coincide with the axes (i.e, the
  * vertex itself is requested), then the hypercube dimension equals 0, so
  * there is no interpolation at all -- only that vertex's function value is
  * returned.
  */

typedef struct ndp_hypercube {
    int dim;      /*!< Dimension of the hypercube */
    int vdim;     /*!< Function value length */
    int fdhc;     /*!< Flag indicating whether the hypercube is fully defined */
    double *v;    /*!< Hypercube vertex function values in C order (last dimension stride-1) */
} ndp_hypercube;

/** @fn ndp_hypercube *ndp_hypercube_new(void)
  * @ingroup constructors
  * @brief Default constructor for ndp_hypercube.
  * 
  * @details
  * Initializes a new ndp_hypercube instance, sets dim and vdim to 0, fdhc
  * to 0 (not fully defined), and vertex array v to NULL.
  * 
  * @return Initialized ndp_hypercube instance with default values
  */

ndp_hypercube *ndp_hypercube_new();

/** @fn ndp_hypercube *ndp_hypercube_new_from_data(int dim, int vdim, int fdhc, double *v)
  * @ingroup constructors
  * @brief Constructor for ndp_hypercube from existing data.
  * 
  * @param dim Dimension of the hypercube
  * @param vdim Function value length
  * @param fdhc Flag indicating whether hypercube is fully defined
  * @param v Hypercube vertex function values
  * 
  * @details
  * Initializes a new ndp_hypercube instance and populates all fields from
  * passed arguments. Note that the function does not copy the array, it
  * only assigns a pointer to it. Thus, the calling function needs to pass an
  * allocated copy if the array is (re)used elsewhere. Ndpolator treats the
  * v array as read-only and it will not change it. On destruction, the v
  * array is freed along with the ndp_hypercube instance.
  *
  * @return Initialized ndp_hypercube instance
  */

ndp_hypercube *ndp_hypercube_new_from_data(int dim, int vdim, int fdhc, double *v);

/** @fn int ndp_hypercube_alloc(ndp_hypercube *hc, int dim, int vdim)
  * @ingroup constructors
  * @brief Allocates memory for ndp_hypercube vertex array.
  * 
  * @param hc ndp_hypercube instance to allocate memory for
  * @param dim Dimension of the hypercube
  * @param vdim Function value length
  * 
  * @details
  * Allocates memory for the vertex array v, of length vdim * 2^dim. It also
  * sets the dim and vdim fields of the hypercube instance, and sets fdhc to
  * 0 (not fully defined).
  * 
  * @return ndp_status code
  */

int ndp_hypercube_alloc(ndp_hypercube *hc, int dim, int vdim);

/** @fn void ndp_hypercube_print(ndp_hypercube *hc, const char *prefix)
  * @ingroup convenience
  * @brief Prints ndp_hypercube information for debugging.
  *
  * @param hc ndp_hypercube instance to print
  * @param prefix String to prefix each printed line with (can be NULL)
  *
  * @details
  * Prints basic information about the ndp_hypercube instance, including
  * dimension, vertex dimension (function value length), whether the hypercube
  * is fully defined, and all vertex function values.
  */

void ndp_hypercube_print(ndp_hypercube *hc, const char *prefix);

/** @fn void ndp_hypercube_free(ndp_hypercube *hc)
  * @ingroup constructors
  * @brief Destructor for ndp_hypercube.
  * 
  * @param hc ndp_hypercube instance to be freed
  * 
  * @details
  * Frees memory allocated for the ndp_hypercube instance. This includes the
  * vertex array v and the ndp_hypercube instance itself.
  * 
  * @return ndp_status code
  */

int ndp_hypercube_free(ndp_hypercube *hc);

/* defined in ndpolator.c: */
/** @fn extern int ndp_idx2pos(ndp_axes *axes, int vdim, int *index, int *pos)
  * 
  * @ingroup convenience
  * @brief Converts array of indices to integer position in flattened array.
  *
  * @param axes ndp_axes structure with all axis definitions
  * @param vdim Vertex dimension (function value length)
  * @param index Array of per-axis indices
  * @param[out] pos Integer position in the flattened array
  *
  * @return ndp_status code
  *
  * @details
  * For efficiency, all ndpolator arrays are flattened (1-dimensional), where
  * axes are stacked in the usual C order (last axis runs first, i.e. last
  * dimension is stride-1). Referring to grid elements can be done either by
  * position in the 1-dimensional array, or per-axis indices. This function
  * converts index representation to position.
  */

extern int ndp_idx2pos(ndp_axes *axes, int vdim, int *index, int *pos);

/** @fn ndp_pos2idx(ndp_axes *axes, int vdim, int pos, int *idx)
  * 
  * @ingroup convenience
  * @brief Converts integer position to array of per-axis indices.
  * 
  * @param axes ndp_axes structure with all axis definitions
  * @param vdim Function value length (number of values per grid point)
  * @param pos Integer position in flattened array
  * @param[out] idx Array of per-axis indices (must be pre-allocated)
  * 
  * @return ndp_status code
  * 
  * @details
  * For efficiency, all ndpolator arrays are flattened (1-dimensional), where
  * axes are stacked in the usual C order (last axis runs first, i.e. last
  * dimension is stride-1). Referring to grid elements can be done either by
  * position in the 1-dimensional array, or per-axis indices. This function
  * converts position representation to index.
  */

extern int ndp_pos2idx(ndp_axes *axes, int vdim, int pos, int *idx);

/** @struct ndp_query
  * @ingroup data_structures
  * @brief Main query structure containing query points, hypercubes and results.
  * 
  * @details
  * Query is ndpolator's main work structure. It stores the query points
  * (called elements in the structure), the corresponding axis indices,
  * flags, and hypercubes. Once interpolation/extrapolation is done (by
  * calling ndpolate), interpolated values are also stored in it.
  */

typedef struct ndp_query {
    int nelems;                  /*!< Number of query points */
    int extrapolation_method;    /*!< ndp_extrapolation_method used for this query */
    int search_algorithm;        /*!< ndp_search_algorithm used for this query */
    ndp_hypercube **hypercubes;  /*!< Array of hypercubes, one per query point */
    double *interps;             /*!< Array of interpolation/extrapolation results */
    double *dists;               /*!< Array of distances to nearest fully defined hypercube */
} ndp_query;

/** @fn ndp_query *ndp_query_new(void)
  * @ingroup constructors
  * @brief Default constructor for ndp_query.
  *
  * @details
  * Initializes a new ndp_query instance, sets nelems to 0,
  * extrapolation_method to NDP_METHOD_NONE, search algorithm to
  * NDP_SEARCH_KDTREE and sets hypercubes, interps, and dists to NULL.
  *
  * @return Initialized ndp_query instance with default values
  */

ndp_query *ndp_query_new();

/** @fn int ndp_query_free(ndp_query *query)
  * @ingroup constructors
  * @brief Destructor for ndp_query.
  * 
  * @param query ndp_query instance to be freed
  * 
  * @details
  * Frees memory allocated for the ndp_query instance. This includes the
  * hypercubes, interps, and dists arrays, along with all individual
  * hypercube instances.
  * 
  * @return ndp_status code
  */

int ndp_query_free(ndp_query *query);

#endif
