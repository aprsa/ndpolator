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

/** @fn ndp_axes *ndp_axes_new_from_python(PyObject *py_axes, int nbasic)
  * @ingroup python_constructors
  * @brief Constructor for ndp_axes from Python objects.
  * 
  * @param py_axes Tuple of ndarrays, one for each axis
  * @param nbasic Number of basic axes
  * 
  * @details
  * Initializes a new ndp_axes instance by translating python data into C
  * and then calling ndp_axes_new_from_data(). The py_axes argument must be a
  * tuple of ndarrays, one for each axis. Each ndarray must be one-dimensional
  * and of type NPY_DOUBLE. The nbasic argument specifies how many of the axes
  * are basic axes; the rest are associated axes. If nbasic is set to 0, all
  * axes are considered basic axes.
  * 
  * @return Initialized ndp_axes instance
  */

ndp_axes *ndp_axes_new_from_python(PyObject *py_axes, int nbasic);

/** @fn ndp_table *ndp_table_new_from_python(PyObject *py_axes, int nbasic, PyArrayObject *py_grid)
  * @ingroup python_constructors
  * @brief Constructor for ndp_table from Python objects.
  * 
  * @param py_axes Python object containing axis data
  * @param nbasic Number of basic axes
  * @param py_grid Python numpy array containing grid data
  * 
  * @details
  * Initializes a new ndp_table instance by translating python data into C and
  * then calling ndp_table_new_from_data(). The passed py_axes parameter
  * must be a tuple of numpy arrays, one for each axis; the passed nbasic
  * must be an integer that provides the number of basic axes (<=
  * len(py_axes)), and the passed py_grid parameter must be a numpy array of
  * the shape (n1, n2, ..., nk, ..., nN, vdim), where nk is the length of the
  * k-th axis.
  * 
  * @return Initialized ndp_table instance
  */

ndp_table *ndp_table_new_from_python(PyObject *py_axes, int nbasic, PyArrayObject *py_grid);

#endif
