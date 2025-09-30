/**
 * @file ndpolator.h
 * @brief Enumerators and function prototypes for ndpolator.
 */

#ifndef NDPOLATOR_H
    #define NDPOLATOR_H 1

#include "ndp_types.h"

/**
 * @defgroup main_api Main C API
 * @brief Core functions for n-dimensional interpolation and extrapolation
 * 
 * These are the main functions users will call to perform interpolation.
 * Start here if you're new to ndpolator.
 */

/** @enum ndp_extrapolation_method
  * @ingroup enumerators
  * @brief Determines how ndpolator handles out-of-bounds query points.
  * 
  * @details
  * Controls the behavior when query points fall outside the fully defined
  * hypercube boundaries during interpolation operations.
  */

typedef enum {
    NDP_METHOD_NONE = 0,     /*!< Do not extrapolate; return NAN instead */
    NDP_METHOD_NEAREST,      /*!< Find nearest defined vertex and use its value */
    NDP_METHOD_LINEAR        /*!< Find nearest fully defined hypercube for linear extrapolation */
} ndp_extrapolation_method;

/** @enum ndp_search_algorithm
  * @ingroup enumerators
  * @brief Determines which algorithm to use for nearest neighbor searches.
  * 
  * @details
  * The default is to use a k-d tree spatial index, but for small
  * datasets a linear search may be faster. K-d tree has O(log N)
  * complexity while linear search has O(N) complexity.
  */

typedef enum {
    NDP_SEARCH_KDTREE = 0,   /*!< Use k-d tree spatial index for nearest neighbor searches */
    NDP_SEARCH_LINEAR        /*!< Use linear search for nearest neighbor searches */
} ndp_search_algorithm;

/** @enum ndp_vertex_flag
  * @ingroup enumerators
  * @brief Flags each component of a query point relative to axis boundaries.
  * 
  * @details
  * Used to classify whether each component of a query point is within
  * the axis span, coincides with a vertex, or lies outside the bounds.
  * This information is crucial for determining interpolation vs extrapolation.
  */

typedef enum {
    NDP_ON_GRID = 0,         /*!< Query point component is on-grid and can be interpolated */
    NDP_ON_VERTEX,           /*!< Query point component coincides with a vertex (within tolerance) */
    NDP_OUT_OF_BOUNDS        /*!< Query point component is off-grid and requires extrapolation */
} ndp_vertex_flag;

/** @fn int *ndp_find_nearest(double *normed_elem, int *elem_index, int *elem_flag, ndp_table *table, ndp_extrapolation_method extrapolation_method, ndp_search_algorithm search_algorithm, double *dist)
  * @ingroup main_api
  * @brief Finds the nearest defined vertex or fully defined hypercube.
  * 
  * @param normed_elem Unit hypercube-normalized query point coordinates
  * @param elem_index Superior corner indices of containing/nearest hypercube
  * @param elem_flag Per-component flags for the query point
  * @param table Self-contained ndpolator table with grid definition
  * @param extrapolation_method Method for handling out-of-bounds points
  * @param search_algorithm Algorithm to use for nearest neighbor search
  * @param[out] dist Distance to nearest fully defined hypercube
  * 
  * @return Allocated array to nearest coordinates (caller must free)
  * 
  * @details
  * Finds the nearest defined vertex or the nearest fully defined hypercube.
  * 
  * The normed_elem parameter provides coordinates of the query point in unit
  * hypercube space. For example, normed_elem=(0.3, 0.8, 0.2) provides
  * coordinates with respect to the inferior hypercube corner (within the hypercube).
  * Conversely, normed_elem=(-0.2, 0.3, 0.4) would lie outside the hypercube.
  * 
  * The elem_index parameter provides coordinates of the superior hypercube
  * corner (indices of each axis where the corresponding value is the
  * first value greater than the query point coordinate). For example, if the
  * query point (in index space) is (4.2, 5.6, 8.9), then elem_index=(5, 6, 9).
  * 
  * The elem_flag parameter flags each coordinate of the normed_elem with
  * NDP_ON_GRID, NDP_ON_VERTEX, or NDP_OUT_OF_BOUNDS. This is important because
  * elem_index points to the nearest larger axis value if the coordinate does not
  * coincide with the axis vertex, and points to the vertex itself if it coincides.
  * 
  * The extrapolation_method parameter determines whether to find the nearest
  * vertex (NDP_METHOD_NEAREST) or nearest fully defined hypercube (NDP_METHOD_LINEAR).
  * This determines which mask is used: table->vmask (defined vertices) or
  * table->hcmask (fully defined hypercubes).
  * 
  * The search_algorithm parameter determines whether to use k-d tree or linear search.
  * K-d tree has O(log N) complexity vs O(N) for linear search, but has upfront
  * construction cost that may not be worthwhile for small grids.
  */

int *ndp_find_nearest(double *normed_elem, int *elem_index, int *elem_flag, ndp_table *table, ndp_extrapolation_method extrapolation_method, ndp_search_algorithm search_algorithm, double *dist);

/** @fn ndp_query_pts *ndp_query_pts_import(int nelems, double *qpts, ndp_axes *axes)
  * @ingroup main_api
  * @brief Imports and pre-processes query points for ndpolator operations.
  *
  * @param nelems Number of query points to process
  * @param qpts Array of query point coordinates
  * @param axes Complete axes structure defining the grid
  *
  * @details
  * Looks up superior index of the n-dimensional hypercube that contains each
  * query point in qpts. The index is found by binary search for each axis
  * sequentially.
  *
  * When any of the query point components coincides with the grid vertex,
  * that component will be flagged by NDP_ON_VERTEX. This is used by the
  * ndp_find_hypercubes function to reduce the dimensionality of the
  * corresponding hypercube. Any query point components that fall outside of
  * the grid boundaries are flagged by NDP_OUT_OF_BOUNDS. Finally, all
  * components that do fall within the grid are flagged by NDP_ON_GRID.
  *
  * @return Allocated ndp_query_pts structure (caller must free)
  */

ndp_query_pts *ndp_query_pts_import(int nelems, double *qpts, ndp_axes *axes);

/** @fn ndp_hypercube **ndp_find_hypercubes(ndp_query_pts *qpts, ndp_table *table)
  * @ingroup main_api
  * @brief Finds hypercubes containing each query point.
  * 
  * @param qpts Pre-processed ndp_query_pts structure
  * @param table Self-contained ndpolator table with grid definition
  * 
  * @return Array of hypercube pointers, one per query point (caller must free)
  * 
  * @details
  * For each query point, the function identifies the N-dimensional hypercube
  * that contains or is nearest to the point. A hypercube is defined by 2^N
  * vertices in N-dimensional space.
  * 
  * The function handles several cases:
  * - **Interior points**: Creates hypercubes with defined vertices for interpolation
  * - **Boundary points**: Creates lower-dimensional hypercubes when points
  *   lie exactly on grid lines or vertices
  * - **Exterior points**: Finds the nearest fully defined hypercube for extrapolation
  * 
  * Each returned hypercube contains the vertex values needed for interpolation,
  * organized in C-order (last dimension varies fastest).
  */

ndp_hypercube **ndp_find_hypercubes(ndp_query_pts *qpts, ndp_table *table);

/** @fn ndp_query *ndpolate(ndp_query_pts *qpts, ndp_table *table, ndp_extrapolation_method extrapolation_method, ndp_search_algorithm search_algorithm)
  * @ingroup main_api
  * @brief Performs n-dimensional interpolation and extrapolation.
  *
  * @param qpts Query points structure with all necessary information
  * @param table Complete ndpolator table with grid definition
  * @param extrapolation_method Method for handling out-of-bounds points
  * @param search_algorithm Algorithm to use for nearest neighbor search
  *
  * @return Allocated ndp_query structure with interpolation results (caller
  * must free)
  *
  * @details
  * This is the main ndpolator function that performs n-dimensional
  * interpolation and extrapolation on sparse grids. It processes all query
  * points in the qpts structure and returns interpolated/extrapolated values.
  *
  * The function handles three main scenarios:
  * - **Interpolation**: When query points lie within the defined grid, linear
  *   interpolation is performed using the enclosing hypercube vertices.  
  * - **Extrapolation**: When query points lie outside the grid, the specified
  *   extrapolation method determines the behavior (nearest neighbor or
  *   linear).
  * - **On-node queries**: When any one or more components of the query point
  *   coincide with axis nodes to a specified tolerance, hypercube
  *   dimensionality is reduced by eliminating interpolation along those axes.
  *   For example, if the query point is (0, 0.5,
  *   1) and hypercube vertices are [0, 1]^3, then interpolation will only be
  *      performed along the second axis, i.e. in 1D.
  * - **On-vertex queries**: When query points coincide with grid vertices to
  *   a specified tolerance, vertex values are returned directly, without
  *   interpolation.
  *
  * The extrapolation_method parameter determines ndpolator's behavior for
  * off-grid query points:
  * - **NDP_METHOD_NONE**: No extrapolation is done, and nan is returned
  * - **NDP_METHOD_NEAREST**: Nearest (Euclidean distance) defined vertex
  *   is found and its value is returned
  * - **NDP_METHOD_LINEAR**: Nearest (Euclidean distance) fully defined
  *   hypercube is found and the returned value is found by linear
  *   extrapolation.
  * 
  * The search_algorithm parameter controls the method to find nearest
  * vertices or fully defined hypercubes:
  * - **NDP_SEARCH_KDTREE**: Fast O(log N) searches, ideal for large grids
  * - **NDP_SEARCH_LINEAR**: Simple O(N) searches, better for small grids
  *
  * @note The returned ndp_query structure contains interpolation results in
  * the interps array and distances to nearest defined points in the dists
  * array.
  */

ndp_query *ndpolate(ndp_query_pts *qpts, ndp_table *table, ndp_extrapolation_method extrapolation_method, ndp_search_algorithm search_algorithm);

#endif
