/* Python wrappers for ndpolator
 * 
 * This file contains all Python-specific code for the ndpolator C extension.
 * The core interpolation functionality is in ndpolator.c.
 */

#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

/* Required by numpy C-API. It defines a unique symbol to be used in other
 * C source files and header files. */
#define PY_ARRAY_UNIQUE_SYMBOL cndpolator_ARRAY_API

#include "ndpolator.h"
#include "ndp_types.h"
#include "ndp_py.h"

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

/* ====================================================================
 * Python-specific constructor functions
 * These functions create ndpolator data structures from Python objects
 * ==================================================================== */

ndp_axes *ndp_axes_new_from_python(PyObject *py_axes, int nbasic)
{
    ndp_axes *axes;

    int naxes = PyTuple_Size(py_axes);
    ndp_axis **axis = malloc(naxes*sizeof(*axis));

    if (nbasic == 0) nbasic = naxes;

    for (int i = 0; i < naxes; i++) {
        PyArrayObject *py_axis = (PyArrayObject *) PyTuple_GetItem(py_axes, i);
        int py_axis_len = PyArray_DIM(py_axis, 0);
        double *py_axis_data = (double *) PyArray_DATA(py_axis);
        axis[i] = ndp_axis_new_from_data(py_axis_len, py_axis_data, /* owns_data = */ 0);
    }

    axes = ndp_axes_new_from_data(naxes, nbasic, axis);

    return axes;
}

ndp_table *ndp_table_new_from_python(PyObject *py_axes, int nbasic, PyArrayObject *py_grid)
{
    ndp_axes *axes = ndp_axes_new_from_python(py_axes, nbasic);

    int ndims = PyArray_NDIM(py_grid);
    int vdim = PyArray_DIM(py_grid, ndims-1);

    /* work around the misbehaved array: */
    PyArrayObject *py_behaved_grid = (PyArrayObject *) PyArray_FROM_OTF((PyObject *) py_grid, NPY_DOUBLE, NPY_ARRAY_CARRAY);
    double *grid = (double *) PyArray_DATA(py_behaved_grid);

    return ndp_table_new_from_data(axes, vdim, grid, /* owns_data = */ 0);
}

/* ====================================================================
 * Python wrapper functions
 * ==================================================================== */

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
        axis[i] = ndp_axis_new_from_data(PyArray_SIZE(py_axis), (double *) PyArray_DATA(py_axis), /* owns_data = */ 0);
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

    /* indices, flags, and normed are now owned by Python, so NULLify them: */
    query_pts->indices = NULL;
    query_pts->flags = NULL;
    query_pts->normed = NULL;

    /* free all the rest: */
    ndp_query_pts_free(query_pts);

    py_combo = PyTuple_New(3);
    PyTuple_SET_ITEM(py_combo, 0, py_indices);
    PyTuple_SET_ITEM(py_combo, 1, py_flags);
    PyTuple_SET_ITEM(py_combo, 2, py_normed_query_pts);

    return py_combo;
}

/* Python wrapper to the ndp_distance() function */
static PyObject *py_distance(PyObject *self, PyObject *args, PyObject *kwargs)
{
    ndp_table *table;
    int owns_table = 0;

    PyObject *py_capsule = NULL;
    PyArrayObject *py_query_pts = NULL;
    PyObject *py_axes = NULL;
    PyArrayObject *py_grid = NULL;
    int nbasic = 0;

    static char *kwlist[] = {"query_pts", "capsule", "axes", "grid", "nbasic", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OOOi", kwlist, &py_query_pts, &py_capsule, &py_axes, &py_grid, &nbasic))
        return NULL;

    if (!PyCapsule_IsValid(py_capsule, NULL) && !(py_axes && py_grid)) {
        PyErr_SetString(PyExc_ValueError, 
            "Either capsule must be valid or axes and grid must be provided");
        return NULL;
    }

    if (PyCapsule_IsValid(py_capsule, NULL)) {
        owns_table = 0;
        table = (ndp_table *) PyCapsule_GetPointer(py_capsule, NULL);
    }
    else if (py_query_pts && py_axes && py_grid) {
        owns_table = 1;
        table = ndp_table_new_from_python(py_axes, nbasic != 0 ? nbasic : PyArray_DIM(py_grid, 0), py_grid);
    }
    else
        return NULL;

    int nelems = PyArray_DIM(py_query_pts, 0);
    double *qpts = (double *) PyArray_DATA(py_query_pts);
    ndp_query_pts *query_pts = ndp_query_pts_import(nelems, qpts, table->axes);
    double *distances = malloc(nelems * sizeof(*distances));

    for (int i = 0; i < nelems; i++) {
        double *normed_elem = query_pts->normed + i * table->axes->len;
        int *elem_index = query_pts->indices + i * table->axes->len;
        int *elem_flag = query_pts->flags + i * table->axes->len;

        int *coords = ndp_find_nearest(normed_elem, elem_index, elem_flag, table, NDP_METHOD_LINEAR, NDP_SEARCH_KDTREE, &distances[i]);
        free(coords);
    }

    ndp_query_pts_free(query_pts);
    if (owns_table)
        ndp_table_free(table);

    npy_intp ddim[] = {nelems, 1};
    PyObject *py_distances = PyArray_SimpleNewFromData(2, ddim, NPY_DOUBLE, distances);
    PyArray_ENABLEFLAGS((PyArrayObject *) py_distances, NPY_ARRAY_OWNDATA);
    return py_distances;
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

    for (int i = 0; i < nelems; i++) {
        /* nullify hypercube data pointers, those are now owned by python: */
        hypercubes[i]->v = NULL;
        ndp_hypercube_free(hypercubes[i]);
    }
    free(hypercubes);

    ndp_table_free(table);

    /* only free the container; arrays are owned by python */
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

/* Destructor for PyCapsule containing ndp_table */
static void py_table_capsule_destructor(PyObject *capsule)
{
    ndp_table *table = (ndp_table *) PyCapsule_GetPointer(capsule, NULL);
    if (table)
        ndp_table_free(table);
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
        py_capsule = PyCapsule_New((void *) table, NULL, py_table_capsule_destructor);
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
    {"distance", (PyCFunction) py_distance, METH_VARARGS | METH_KEYWORDS, "compute squared distance to nearest edge/face/vertex of hypercube"},
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
