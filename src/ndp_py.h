/**
 * @file ndp_py.h
 * @brief Python-specific functions for ndpolator C extension
 * 
 * This header contains declarations for functions that interface between
 * Python and the core C ndpolator functionality. These functions handle
 * conversion of Python objects to C data structures.
 */

#ifndef NDP_PY_H
    #define NDP_PY_H 1

/* Python and NumPy headers - required for Python interface */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "ndp_types.h"

/**
 * @defgroup python_constructors Python Interface Constructors
 * @brief Constructors that create ndpolator structures from Python objects
 * 
 * These functions are part of the Python C extension and handle conversion
 * of Python objects (NumPy arrays, tuples, etc.) into native C data structures
 * used by the core ndpolator library.
 */

/** @struct ndp_py_table
  * @ingroup python_constructors
  * @brief An ndp_table together with the Python arrays that own its memory.
  *
  * @details
  * The core ndp_table only borrows its axis and grid arrays (owns_data = 0),
  * so the numpy arrays they point into must outlive the table. This wrapper
  * holds one reference to each of those arrays (converted to C-contiguous
  * NPY_DOUBLE arrays if they were not already) and releases them in
  * ndp_py_table_free(). The capsule handed back to Python points to an
  * ndp_py_table, so the arrays live exactly as long as the capsule does.
  */

typedef struct ndp_py_table {
    ndp_table *table;  /*!< Wrapped table; its axis and grid pointers borrow from the arrays below */
    PyObject *py_axes; /*!< Owned reference: tuple of C-contiguous NPY_DOUBLE ndarrays, one per axis */
    PyObject *py_grid; /*!< Owned reference: C-contiguous NPY_DOUBLE ndarray holding the grid */
} ndp_py_table;

/** Capsule name for capsules that wrap an ndp_py_table. Passing it to
  * PyCapsule_IsValid() makes capsules of any other origin invalid. */
#define NDP_PY_TABLE_CAPSULE_NAME "ndpolator.table"

/** @fn ndp_py_table *ndp_py_table_new(PyObject *py_axes, int nbasic, PyObject *py_grid)
  * @ingroup python_constructors
  * @brief Constructor for ndp_py_table (and its ndp_table) from Python objects.
  * 
  * @param py_axes Tuple of ndarrays, one for each axis
  * @param nbasic Number of basic axes; 0 means all axes are basic
  * @param py_grid ndarray of shape (n1, ..., nN, vdim), where nk is the length of the k-th axis
  * 
  * @details
  * Converts each axis and the grid to a C-contiguous NPY_DOUBLE ndarray
  * (reusing the original array when it already is one), keeps a reference to
  * each converted array, and builds the ndp_table on top of their data
  * buffers by calling ndp_axes_new_from_data() and ndp_table_new_from_data().
  * 
  * @return Initialized ndp_py_table, or NULL with a Python exception set
  */

ndp_py_table *ndp_py_table_new(PyObject *py_axes, int nbasic, PyObject *py_grid);

/** @fn void ndp_py_table_free(ndp_py_table *pytable)
  * @ingroup python_constructors
  * @brief Frees an ndp_py_table, its ndp_table, and the array references it holds.
  * 
  * @param pytable ndp_py_table to free; NULL is allowed and ignored
  */

void ndp_py_table_free(ndp_py_table *pytable);

#endif
