#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject *mat_mul(PyObject *self, PyObject *args)
{
    PyObject* a;
    PyObject* b;

    if (!PyArg_ParseTuple(args, "OO", &a, &b))
        return NULL;

    Py_ssize_t rows_a = PySequence_Length(a),
        cols_a = PySequence_Length(PySequence_GetItem(a, 0)),
        cols_b = PySequence_Length(PySequence_GetItem(b, 0));

    PyObject* c = PyList_New(rows_a);
    for (Py_ssize_t i = 0; i < rows_a; ++i) {
        PyObject* rowList = PyList_New(cols_b);

        for (Py_ssize_t j = 0; j < cols_b; ++j) {
            PyList_SET_ITEM(rowList, j, PyFloat_FromDouble(0));
        }

        PyList_SetItem(c, i, rowList);
    }

    for(Py_ssize_t i = 0;i < rows_a;i++) {
        PyObject* c_row = PySequence_GetItem(c, i);

        for(Py_ssize_t j = 0;j < cols_b;j++) {
            for(Py_ssize_t k = 0;k < cols_a;k++) {
                PyObject* c_cell = PySequence_GetItem(c_row, j);
                PyObject* a_cell = PySequence_GetItem(PySequence_GetItem(a, i), k);
                PyObject* b_cell = PySequence_GetItem(PySequence_GetItem(b, k), j);
                double c_val = PyFloat_AsDouble(c_cell),
                    a_val = PyFloat_AsDouble(a_cell),
                    b_val = PyFloat_AsDouble(b_cell);
                
                PyList_SetItem(c_row, j, PyFloat_FromDouble(c_val + a_val * b_val));
            }
        }
    }

    return c;
}

static PyMethodDef ModuleMethods[] = {
    { "mul", mat_mul, METH_VARARGS, "description" }
};

static PyModuleDef ModuleDef = {
    PyModuleDef_HEAD_INIT,
    "MatModule",
    "description",
    -1,
    ModuleMethods
};

PyMODINIT_FUNC PyInit_mat(void)
{
    Py_Initialize();
    return PyModule_Create(&ModuleDef);
}
