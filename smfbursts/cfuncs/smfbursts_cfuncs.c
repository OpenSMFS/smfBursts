#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>

#include "smfbursts_burstwise.h"


static PyObject* smfbursts_cfuncs_index_range(PyObject* self, PyObject* args, PyObject* kwargs)
{
	// set input argument names
	char *kwlist[] = {"times", "start", "stop", "prev", NULL};
	PyObject *pytimes = NULL;
	Py_ssize_t prev = 0;
	long long start, stop;
	// parse args/kwargs, types are Object, unsigned long long, unsigned long long, and bool
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OLL|n:index_range", kwlist, &pytimes, &start, &stop, &prev)){
		return NULL;
	}
	// ensure/convert object times is/to numpy array, and is uint64_t and is 1D
	PyArrayObject *nptimes = (PyArrayObject*) PyArray_FROMANY(pytimes, NPY_INT64, 1, 1, NPY_ARRAY_ENSUREARRAY);
	if (nptimes == NULL){
		return NULL;
	}
	// dimensions of output array, and create (with zeros, in case not times in range
	const npy_intp odims[] = {2, };
	PyArrayObject *out = (PyArrayObject*) PyArray_ZEROS(1, odims, NPY_INT64, FALSE);
	if (out == NULL){
		Py_DECREF(pytimes);
		return NULL;
	}
	// get important pointers/sizes
	char *times = (char*) PyArray_DATA(nptimes);
	int64_t *ot = (npy_intp*) PyArray_DATA(out);
	const npy_intp size = PyArray_DIM(nptimes, 0);
	const npy_intp stride = PyArray_STRIDE(nptimes, 0);
	if (prev < 0){
		prev = size - prev;
		if (prev < 0) prev = 0;
	}
	// core algorithm
	npy_intp i = prev;
	while (i < size && *(int64_t*)&times[i*stride] < start) i++;
	ot[0] = i;
	while (i < size && *(int64_t*)&times[i*stride] < stop) i++;
	ot[1] = i;
	Py_DECREF(nptimes);
	return (PyObject*) out;
}

static PyObject* smfbursts_cfuncs_index_ranges(PyObject* self, PyObject* args, PyObject* kwargs)
{
	char *kwlist[] = {"times", "start", "stop", "nonoverlap", NULL};
	PyObject *pytimes = NULL, *pystarts = NULL, *pystops = NULL;
	int overlap = FALSE;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|p:index_ranges", kwlist, &pytimes, &pystarts, &pystops, &overlap)){
		return NULL;
	}
	PyArrayObject *nptimes = (PyArrayObject*) PyArray_FROMANY(pytimes, NPY_INT64, 1, 1, NPY_ARRAY_ENSUREARRAY);
	PyArrayObject *npstarts = (PyArrayObject*) PyArray_FROMANY(pystarts, NPY_INT64, 1, 1, NPY_ARRAY_ENSUREARRAY);
	PyArrayObject *npstops = (PyArrayObject*) PyArray_FROMANY(pystops, NPY_INT64, 1, 1, NPY_ARRAY_ENSUREARRAY);
	if ((nptimes == NULL)||(npstarts == NULL)||(npstops == NULL)){
		Py_XDECREF(nptimes);
		Py_XDECREF(npstarts);
		Py_XDECREF(npstops);
		return NULL;
	}
	const npy_intp odims[] = {PyArray_DIM(npstarts, 0), };
	if (odims[0] != PyArray_DIM(npstops, 0)){
		PyErr_Format(PyExc_ValueError, "mismatched size of starts and stops, got %li and %li", odims[0], PyArray_DIM(npstops, 0));
		Py_DECREF(nptimes);
		Py_DECREF(npstarts);
		Py_DECREF(npstops);
		return NULL;
	}
	PyArrayObject *npistarts = (PyArrayObject*) PyArray_ZEROS(1, odims, NPY_INT64, FALSE);
	if (npistarts == NULL){
		Py_DECREF(nptimes);
		Py_DECREF(npstarts);
		Py_DECREF(npstops);
		return NULL;
	}
	PyArrayObject *npistops = (PyArrayObject*) PyArray_ZEROS(1, odims, NPY_INT64, FALSE);
	if (npistops == NULL){
		Py_DECREF(nptimes);
		Py_DECREF(npstarts);
		Py_DECREF(npstops);
		Py_DECREF(npistarts);
		return NULL;
	}
	// extract constants for stides etc.
	npy_intp timesize = PyArray_DIM(nptimes, 0);
	npy_intp timestride = PyArray_STRIDE(nptimes, 0);
	char *times = (char*) PyArray_DATA(nptimes);
	npy_intp startstride = PyArray_STRIDE(npstarts, 0);
	char *starts = (char*) PyArray_DATA(npstarts);
	npy_intp stopstride = PyArray_STRIDE(npstops, 0);
	char *stops = (char*) PyArray_DATA(npstops);
	int64_t *ostarts = PyArray_DATA(npistarts);
	int64_t *ostops = PyArray_DATA(npistops);
	int64_t i = 0;
	// loop to find locations
	for (npy_intp n = 0; n < odims[0]; n++){
		while (i < timesize && *(int64_t*)&times[i*timestride] < *(int64_t*)&starts[n*startstride]) i++;
		ostarts[n] = i;
		while (i < timesize && *(int64_t*)&times[i*timestride] < *(int64_t*)&stops[n*stopstride]) i++;
		ostops[n] = i;
		i = overlap ? ostarts[n] : i;
	}
	Py_DECREF(nptimes);
	Py_DECREF(npstarts);
	Py_DECREF(npstops);
	PyObject *out = PyTuple_Pack(2, npistarts, npistops);
	Py_DECREF(npistarts);
	Py_DECREF(npistops);
	return out;
}


static PyObject* smfbursts_cfuncs_burstsearch(PyObject* self, PyObject* args, PyObject* kwargs){
	char *kwlist[] = {"times", "dets", "periods", "bg", "clk_p", "det_ids", "m", "F", "c", "fuse", "bg_is_thresh", "alloc_size", "ncore", NULL};
	PyObject *pytimes = NULL, *pydets = NULL, *pyperiods = NULL, *pybg = NULL, *pydetids = NULL;
	int64_t m = 10;
	double clk_p = NAN, F = 6.0, c = -1.0, fuse = 0.0;
	Py_ssize_t alloc_size = 512, ncore = 8;
	int asT = FALSE;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOd|OLdddpnn:burstsearch", kwlist, &pytimes, &pydets, &pyperiods, &pybg, &clk_p, &pydetids, &m, &F, &c, &fuse, &asT, &alloc_size, &ncore)){
		return NULL;
	}
	// check for invalid values
	if ((m < 1) || ((((double)(m-1))-c) < 1.0)){
		PyErr_SetString(PyExc_ValueError, "non-positive window for burst search, m must be positive, and m-1-c must also be positive");
		return NULL;
	}
	if (F <= 0.0){
		PyErr_SetString(PyExc_ValueError, "F must be positive");
		return NULL;
	}
	if (clk_p <= 0.0){
		PyErr_SetString(PyExc_ValueError, "clk_p must be positive");
		return NULL;
	}
	if ((fuse < 0.0) && (fuse != -1.0)){
		PyErr_SetString(PyExc_ValueError, "fuse must be -1.0 or non-negative");
		return NULL;
	}
	if (alloc_size < 1){
		PyErr_SetString(PyExc_ValueError, "alloc_size must be positive");
		return NULL;
	}
	if (ncore < 1){
		PyErr_SetString(PyExc_ValueError, "ncore must be positive");
		return NULL;
	}
	// cast all array arguments to well behaved arrays
	PyObject *out = NULL;
	PyArrayObject *nptimes = (PyArrayObject*) PyArray_FROMANY(pytimes, NPY_INT64, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
	PyArrayObject *npdets = (PyArrayObject*) PyArray_FROMANY(pydets, NPY_UINT8, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
	PyArrayObject *npperiods = (PyArrayObject*) PyArray_FROMANY(pyperiods, NPY_INT64, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
	PyArrayObject *npbg = (PyArrayObject*) PyArray_FROMANY(pybg, NPY_DOUBLE, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
	PyArrayObject *npdetids = NULL;
	int64_t nphot = 0, nper=0, dsize = 0;
	int64_t *times = NULL, *periods = NULL;
	uint8_t *dets=NULL, *dset = NULL;
	double *bg = NULL;
	if (nptimes == NULL) goto decrefs;
	if (npdets == NULL) goto decrefs;
	if (npperiods == NULL) goto decrefs;
	if (npbg == NULL) goto decrefs;
	// check for errors in arrays
	if ((nphot = (int64_t)PyArray_DIM(nptimes, 0)) != ((int64_t) PyArray_DIM(npdets, 0))){
		PyErr_Format(PyExc_ValueError, "mismatched sizes of times and dets arrays, must be identical, but got %ld, and %ld", PyArray_DIM(nptimes, 0), PyArray_DIM(npdets,0));
		goto decrefs;
	}
	if ((nper = (int64_t) PyArray_DIM(npbg, 0)) != ((int64_t) PyArray_DIM(npperiods, 0) - 1)){
		PyErr_Format(PyExc_ValueError, "mismatched sizes of periods and bg arrays, periods must be 1 greater in size that bg, but got %ld, and %ld", PyArray_DIM(npperiods, 0), PyArray_DIM(npbg,0));
		goto decrefs;
	}
	// assign pointers
	times   = (int64_t*) PyArray_DATA(nptimes);
	dets    = (uint8_t*) PyArray_DATA(npdets);
	periods = (int64_t*) PyArray_DATA(npperiods);
	bg      = (double*)  PyArray_DATA(npbg);
	// if det_ids is included, extract array, otherwise make detids all detectors
	if ((pydetids != NULL)&&(pydetids != Py_None)){
		npdetids = (PyArrayObject*) PyArray_FROMANY(pydetids, NPY_UINT8, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
		if (npdetids == NULL) goto decrefs;
		dsize = (int64_t) PyArray_DIM(npdetids, 0);
		if (dsize < 1){
			PyErr_SetString(PyExc_ValueError, "must specify at least one detector id");
			goto decrefs;
		}
		dset = (uint8_t*) PyArray_DATA(npdetids);
	}
	else{
		uint8_t maxdet = 0;
		for (int64_t i = 0; i < nphot; i++){
			if (dets[i] > maxdet){
				maxdet = dets[i];
			}
		}
		dsize = 1 + (int64_t) maxdet;
		dset = (uint8_t*) malloc(dsize*sizeof(uint8_t));
		if (dset == NULL) goto decrefs;
		uint8_t detid = 0;
		for (int64_t d = 0; d < dsize; d++){
			dset[d] = detid++;
		}
	}
	// perform calculation
	Bursts *bursts = NULL;
	Py_BEGIN_ALLOW_THREADS;
	if (burst_search_sliding_window(m, F, clk_p, c, nphot, times, dets, dsize, dset, nper, periods, bg, alloc_size, ncore, fuse, asT, &bursts)){
		PyErr_SetString(PyExc_MemoryError, "insufficent memory to allocate for bursts");
		goto decrefs;
	}
	Py_END_ALLOW_THREADS;
	// build output
	const npy_intp dims[] = { (npy_intp) bursts->size , };
	PyObject *pystarts = NULL;
	PyObject *pystops = NULL;
	if (bursts->size != 0){
		//~ pystarts = PyArray_SimpleNewFromData(1, dims, NPY_INT64, bursts->starts);
		//~ pystops= PyArray_SimpleNewFromData(1, dims, NPY_INT64, bursts->stops);
		pystarts = PyArray_EMPTY(1, dims, NPY_INT64, 0);
		pystops = PyArray_EMPTY(1, dims, NPY_INT64, 0);
		if ((pystarts == NULL) || (pystops == NULL)){
			free_bursts_fields(bursts);
			Py_XDECREF(pystarts);
			pystarts = NULL;
			Py_XDECREF(pystops);
			pystops = NULL;
			goto decrefs;
		}
		else{
			memcpy(PyArray_DATA((PyArrayObject*)pystarts), bursts->starts, sizeof(int64_t)*bursts->size);
			memcpy(PyArray_DATA((PyArrayObject*)pystops), bursts->stops, sizeof(int64_t)*bursts->size);
		}
	}
	else{
		pystarts = PyArray_ZEROS(1, dims, NPY_INT64, FALSE);
		pystops = PyArray_ZEROS(1, dims, NPY_INT64, FALSE);
		if ((pystarts == NULL) || (pystops == NULL)){
			Py_XDECREF(pystarts);
			pystarts = NULL;
			Py_XDECREF(pystops);
			pystops = NULL;
		}
	}
	free_bursts_fields(bursts);
	if (bursts != NULL){
		free(bursts);
		bursts = NULL;
	}
	// build output
	out = PyTuple_Pack(2, pystarts, pystops);
	// decref created arrays
	decrefs:
	Py_XDECREF(nptimes);
	nptimes = NULL;
	Py_XDECREF(npdets);
	npdets = NULL;
	Py_XDECREF(npperiods);
	npperiods = NULL;
	Py_XDECREF(npbg);
	npbg = NULL;
	Py_XDECREF(npdetids);
	npdetids = NULL;
	return out;
}

typedef struct{
	int64_t size;
	int64_t *lens;
	PyArrayObject **arrays;
	int64_t **datas;
} PyArrayList;

int free_arraylist(PyArrayList *arrlist){
	if (arrlist == NULL){
		return FALSE;
	}
	if (arrlist->lens != NULL){
		free(arrlist->lens);
		arrlist->lens = NULL;
	}
	if (arrlist->datas != NULL){
		free(arrlist->datas);
		arrlist->datas = NULL;
	}
	if (arrlist->arrays == NULL){
		return FALSE;
	}
	for (int64_t i = 0; i < arrlist->size; i++){
		Py_XDECREF(arrlist->arrays[i]);
		arrlist->arrays[i] = NULL;
	}
	free(arrlist->arrays);
	arrlist->arrays = NULL;
	return FALSE;
}

int converter_listofarrays_int64(PyObject *obj, void *result){
	PyArrayList *arrlist = (PyArrayList*) result;
	Py_ssize_t pylen = PySequence_Length(obj);
	if ( pylen == -1 ){
		PyErr_SetString(PyExc_TypeError, "cannot interpret starts/stops as sequence of arrays");
		return 0;
	}
	else if ( pylen == 0 ) {
		PyErr_SetString(PyExc_ValueError, "must specify at least 1 array");
		return 0;
	}
	arrlist->size = (int64_t) pylen;
	arrlist->lens = (int64_t*) calloc(pylen, sizeof(int64_t));
	arrlist->arrays = (PyArrayObject**) calloc(pylen, sizeof(PyArrayObject*));
	arrlist->datas = (int64_t**) calloc(pylen, sizeof(int64_t*));
	if ((arrlist->lens == NULL)||(arrlist->arrays == NULL)||(arrlist->datas == NULL)){
		PyErr_NoMemory();
		return 0;
	}
	PyObject *temp = NULL;
	for (Py_ssize_t i = 0; i < pylen; i++){
		temp = PySequence_GetItem(obj, i);
		if (temp == NULL){
			return 0;
		}
		arrlist->arrays[i] = (PyArrayObject*) PyArray_FROMANY(temp, NPY_INT64, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
		Py_DECREF(temp);
		if (arrlist->arrays[i] == NULL){
			return 0;
		}
		arrlist->lens[i] = (int64_t) PyArray_DIM(arrlist->arrays[i], 0);
		arrlist->datas[i] = (int64_t*) PyArray_DATA(arrlist->arrays[i]);
	}
	return 1;
}

static PyObject* smfbursts_cfuncs_burstgate(PyObject* self, PyObject* args, PyObject* kwargs){
	char *kwlist[] = {"starts", "stops", "truthtable", "starttime", "stoptime", "alloc_size", NULL};
	PyArrayList startslist, stopslist;
	startslist.size = 0;
	startslist.lens = NULL;
	startslist.arrays = NULL;
	startslist.datas = NULL;
	stopslist.size = 0;
	stopslist.lens = NULL;
	stopslist.arrays = NULL;
	stopslist.datas = NULL;
	PyObject *pytruthtable = NULL, *pystarttime = NULL, *pystoptime = NULL;
	PyArrayObject *nptruthtable = NULL;
	PyObject *out = NULL;
	Bursts *inbursts = NULL, *outbursts = NULL;
	Py_ssize_t alloc_size = 512;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&O|OOn:burstgate", kwlist, converter_listofarrays_int64, &startslist, 
																		converter_listofarrays_int64, &stopslist, 
																		&pytruthtable, &pystarttime, &pystoptime, &alloc_size)){
		goto decrefs;
	}
	// check sizes of arrays consistent
	if (startslist.size != stopslist.size){
		PyErr_Format(PyExc_ValueError, "inconsistent number of arrays to merge in starts/stops, %ld vs %ld", startslist.size, stopslist.size);
		goto decrefs;
	}
	nptruthtable = (PyArrayObject*) PyArray_FROMANY(pytruthtable, NPY_BOOL, 0, 0, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
	if (startslist.size != (int64_t) PyArray_NDIM(nptruthtable)){
		PyErr_Format(PyExc_ValueError, "incorrect number of dimentions of truthtable, has %ld, expected %ld", startslist.size, PyArray_NDIM(nptruthtable));
		goto decrefs;
	}
	for (int64_t i = 0 ; i < startslist.size; i++){
		if (PyArray_DIM(nptruthtable, i) != 2){
			PyErr_SetString(PyExc_ValueError, "all truthtable dimentions must be size 2");
			goto decrefs;
		}
		if (startslist.lens[i] != stopslist.lens[i]){
			PyErr_Format(PyExc_ValueError, "inconsistent number of bursts in starts/stops in element %ld of lists", i);
			goto decrefs;
		}
	}
	// process optional start/stoptime arguments
	int64_t starttime = INT64_MAX, stoptime = 0;
	if ((pystarttime != NULL)&&(pystarttime != Py_None)){
		starttime = (int64_t) PyLong_AsLongLong(pystarttime);
		if ( PyErr_Occurred() != NULL){
			PyErr_SetString(PyExc_TypeError, "starttime must be integer value");
			goto decrefs;
		}
	}
	else{
		for (int64_t i = 0; i < startslist.size; i++){
			if (startslist.lens[i] != 0){
				if (startslist.datas[i][0] < starttime){
					starttime = startslist.datas[i][0];
				}
			}
		}
	}
	if ((pystoptime != NULL)&&(pystoptime != Py_None)){
		stoptime = (int64_t) PyLong_AsLongLong(pystoptime);
		if ( PyErr_Occurred() != NULL) {
			PyErr_SetString(PyExc_TypeError, "stoptime must be integer value");
			goto decrefs;
		}
	}
	else{
		for (int64_t i = 0; i < stopslist.size; i++){
			if (stopslist.lens[i] != 0){
				if (stopslist.datas[i][stopslist.lens[i]-1] > stoptime){
					stoptime = stopslist.datas[i][stopslist.lens[i]-1];
				}
			}
		}
	}
	// ensure starttime is less than stoptime
	if (starttime == INT64_MAX){
		starttime = 0;
	}
	if (starttime > stoptime){
		stoptime = starttime + 1;
	}
	// get alloc_size if non-positive, works by either taking the maximum number of bursts, or from (-alloc_size -1)th bursts size
	if (alloc_size < 1){
		if (alloc_size == 0){
			for (int64_t i = 0 ; i < startslist.size; i++){
				if (startslist.lens[i] > alloc_size){
					alloc_size = startslist.lens[i];
				}
			}
		}
		else {
			if (startslist.size < (-alloc_size)){
				PyErr_Format(PyExc_ValueError, "Negative alloc_size out of bounts, requested %ld of %ld arrays", (-alloc_size - 1), startslist.size);
				goto decrefs;
			}
			alloc_size = startslist.lens[(-alloc_size -1)];
		}
	}
	// allocate bursts for input
	inbursts = (Bursts*) calloc(startslist.size, sizeof(Bursts));
	if (inbursts == NULL){
		PyErr_NoMemory();
		goto decrefs;
	}
	// ensure burst inputs are valid
	for (int64_t i = 0; i < startslist.size; i++){
		inbursts[i].size = startslist.lens[i];
		inbursts[i].pos = 0;
		inbursts[i].starts = startslist.datas[i];
		inbursts[i].stops = stopslist.datas[i];
		if (check_bursts_fused(&inbursts[i], 1)){
			PyErr_SetString(PyExc_ValueError, "invalid burst definitions, starts and stops are non-monotonic or overlapping");
			goto decrefs;
		}
	}
	// allocate output array
	outbursts = (Bursts*) alloc_burst_array(1, alloc_size);
	if (outbursts == NULL){
		PyErr_NoMemory();
		goto decrefs;
	}
	uint8_t *truthtable = (uint8_t*) PyArray_DATA(nptruthtable);
	// compute gated array
	if (burst_gate(startslist.size, inbursts, truthtable, starttime, stoptime, alloc_size, outbursts)){
		free_bursts_fields(outbursts);
		goto decrefs;
	}
	// make output arrays
	PyArrayObject *outstarts = NULL, *outstops = NULL;
	const npy_intp dims[] = {(npy_intp) outbursts->size, };
	if (outbursts->size != 0){
		//~ outstarts = (PyArrayObject*) PyArray_SimpleNewFromData(1, dims, NPY_INT64, outbursts->starts);
		//~ outstops= (PyArrayObject*) PyArray_SimpleNewFromData(1, dims, NPY_INT64, outbursts->stops);
		outstarts = (PyArrayObject*) PyArray_EMPTY(1, dims, NPY_INT64, 0);
		outstops= (PyArrayObject*) PyArray_EMPTY(1, dims, NPY_INT64, 0);
		if ((outstarts == NULL) || (outstops == NULL)){
			free_bursts_fields(outbursts);
			Py_XDECREF(outstarts);
			Py_XDECREF(outstops);
			goto decrefs;
		}
		memcpy(PyArray_DATA(outstarts), outbursts->starts, sizeof(int64_t)*outbursts->size);
		memcpy(PyArray_DATA(outstops), outbursts->stops, sizeof(int64_t)*outbursts->size);
		free_bursts_fields(outbursts);
		//~ PyArray_ENABLEFLAGS((PyArrayObject*) outstarts, NPY_ARRAY_OWNDATA);
		//~ PyArray_ENABLEFLAGS((PyArrayObject*) outstops, NPY_ARRAY_OWNDATA);
	}
	else{
		outstarts = (PyArrayObject*) PyArray_ZEROS(1, dims, NPY_INT64, FALSE);
		outstops = (PyArrayObject*) PyArray_ZEROS(1, dims, NPY_INT64, FALSE);
		if ((outstarts == NULL) || (outstops == NULL)){
			free_bursts_fields(outbursts);
			Py_XDECREF(outstarts);
			outstarts = NULL;
			Py_XDECREF(outstops);
			outstops = NULL;
			goto decrefs;
		}
	}
	out = PyTuple_Pack(2, outstarts, outstops);
	// exit, decref cast numpy arrays
	decrefs:
	if (inbursts != NULL){
		free(inbursts);
		inbursts = NULL;
	}
	if (outbursts != NULL){
		free(outbursts);
		outbursts = NULL;
	}
	free_arraylist(&startslist);
	free_arraylist(&stopslist);
	Py_XDECREF(nptruthtable);
	nptruthtable = NULL;
	return out;
}

static PyObject* smfbursts_cfuncs_fusebursts(PyObject* self, PyObject* args, PyObject* kwargs){
	char *kwlist[] = {"starts", "stops", "max_sep", NULL};
	PyObject *pystarts = NULL, *pystops = NULL, *out = NULL;
	long long max_sep = 0;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOL:fusebursts", kwlist, &pystarts, &pystops, &max_sep)){
		return NULL;
	}
	if (max_sep < 0){
		PyErr_SetString(PyExc_ValueError, "max_sep must be positive");
		return NULL;
	}
	PyArrayObject *npstarts = (PyArrayObject*) PyArray_FROMANY(pystarts, NPY_INT64, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
	PyArrayObject *npstops = (PyArrayObject*) PyArray_FROMANY(pystops, NPY_INT64, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
	if ((npstarts == NULL) || (npstops == NULL)){
		goto decrefs;
	}
	int64_t nbursts = (int64_t) PyArray_DIM(npstarts, 0);
	if (nbursts != (int64_t) PyArray_DIM(npstops, 0)){
		PyErr_Format(PyExc_ValueError, "mismatched sizes of starts and stops, must be equal, got %ld, %ld respectively", nbursts, PyArray_DIM(npstops, 0));
		goto decrefs;
	}
	Bursts inbursts, outbursts;
	inbursts.size = nbursts;
	inbursts.pos = nbursts;
	inbursts.starts = (int64_t*) PyArray_DATA(npstarts);
	inbursts.stops = (int64_t*) PyArray_DATA(npstops);
	outbursts.size = 0;
	outbursts.pos = 0;
	outbursts.starts = NULL;
	outbursts.stops = NULL;
	if (fuse_bursts(&inbursts, max_sep, &outbursts)){
		PyErr_NoMemory();
		goto decrefs;
	}
	const npy_intp dims[] = {(npy_intp) outbursts.size, };
	PyArrayObject *outstarts = NULL, *outstops = NULL;
	if (outbursts.size == 0){
		outstarts = (PyArrayObject*) PyArray_ZEROS(1, dims, NPY_INT64, FALSE);
		outstops = (PyArrayObject*) PyArray_ZEROS(1, dims, NPY_INT64, FALSE);
	}
	else{
		//~ outstarts = (PyArrayObject*) PyArray_SimpleNewFromData(1, dims, NPY_INT64, outbursts.starts);
		//~ outstops = (PyArrayObject*) PyArray_SimpleNewFromData(1, dims, NPY_INT64, outbursts.stops);
		outstarts = (PyArrayObject*) PyArray_EMPTY(1, dims, NPY_INT64, 0);
		outstops = (PyArrayObject*) PyArray_EMPTY(1, dims, NPY_INT64, 0);
		if ((outstarts == NULL) || (outstops == NULL)){
			 Py_XDECREF(outstarts);
			 outstarts = NULL;
			 Py_XDECREF(outstops);
			 outstops = NULL;
			 goto decrefs;
		}
		memcpy(PyArray_DATA(outstarts), outbursts.starts, sizeof(int64_t)*outbursts.size);
		memcpy(PyArray_DATA(outstops), outbursts.stops, sizeof(int64_t)*outbursts.size);
		//~ PyArray_ENABLEFLAGS((PyArrayObject*) outstarts, NPY_ARRAY_OWNDATA);
		//~ PyArray_ENABLEFLAGS((PyArrayObject*) outstops, NPY_ARRAY_OWNDATA);
	}
	out = PyTuple_Pack(2, outstarts, outstops);
	decrefs:
	free_bursts_fields(&outbursts);
	Py_XDECREF(npstarts);
	Py_XDECREF(npstops);
	return out;
}

static PyObject* smfbursts_cfuncs_maximum_rate(PyObject* self, PyObject* args, PyObject* kwargs){
	char *kwlist[] = {"times", "dets", "istarts", "istops", "clk_p", "det_ids", "m", "ncore", NULL};
	PyObject *pytimes = NULL, *pydets = NULL, *pyistarts = NULL, *pyistops = NULL, *pydetids = NULL;
	int64_t m = 10;
	double clk_p = NAN;
	Py_ssize_t ncore = 8;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOd|OLn:maximum_rate", kwlist, &pytimes, &pydets, &pyistarts, &pyistops, &clk_p, &pydetids, &m, &ncore)){
		return NULL;
	}
	// check for invalid values
	if (m < 1){
		PyErr_SetString(PyExc_ValueError, "m must be positive");
		return NULL;
	}
	if (clk_p <= 0.0){
		PyErr_SetString(PyExc_ValueError, "clk_p must be positive");
		return NULL;
	}
	if (ncore < 1){
		PyErr_SetString(PyExc_ValueError, "ncore must be positive");
		return NULL;
	}
	// cast all array arguments to well behaved arrays
	PyArrayObject *nptimes = (PyArrayObject*) PyArray_FROMANY(pytimes, NPY_INT64, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
	PyArrayObject *npdets = (PyArrayObject*) PyArray_FROMANY(pydets, NPY_UINT8, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
	PyArrayObject *npistarts = (PyArrayObject*) PyArray_FROMANY(pyistarts, NPY_INT64, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
	PyArrayObject *npistops = (PyArrayObject*) PyArray_FROMANY(pyistops, NPY_INT64, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
	PyArrayObject *npdetids = NULL;
	PyObject *out = NULL;
	int64_t dsize = 0;
	uint8_t *dset = NULL;
	int64_t nphot = (int64_t) PyArray_DIM(nptimes, 0);
	if ((nptimes == NULL) || (npdets == NULL) || (npistarts == NULL) || (npistops == NULL)){
		goto decrefs;
	}
	if (nphot != (int64_t) PyArray_DIM(npdets, 0)){
		PyErr_Format(PyExc_ValueError, "times and dets must be of the same size, got %ld, and %ld", nphot, PyArray_DIM(npdets,0));
		goto decrefs;
	}
	int64_t nbursts = (int64_t) PyArray_DIM(npistarts, 0);
	if (nbursts != (int64_t) PyArray_DIM(npistops, 0)){
		PyErr_Format(PyExc_ValueError, "istarts and istops must be of the same size, got %ld, and %ld", nbursts, PyArray_DIM(npistops,0));
		goto decrefs;
	}
	int64_t *times = (int64_t*) PyArray_DATA(nptimes);
	uint8_t *dets = (uint8_t*) PyArray_DATA(npdets);
	int64_t *istarts = (int64_t*) PyArray_DATA(npistarts);
	int64_t *istops = (int64_t*) PyArray_DATA(npistops);
	const npy_intp dims[] = {(npy_intp) nbursts, };
	if (nbursts == 0){
		out = PyArray_ZEROS(1, dims, NPY_DOUBLE, FALSE);
		goto decrefs;
	}
	// check that no istart/istop exceeds size of photons (out of range index)
	for (int64_t i = 0; i < nbursts; i++){
		if ((istarts[i] >= nphot) || (istops[i] > nphot)){
			PyErr_SetString(PyExc_ValueError, "istarts/istops index values out of range of size of times");
			goto decrefs;
		}
	}
	// process det_ids, 
	if (pydetids != NULL){
		npdetids = (PyArrayObject*) PyArray_FROMANY(pydetids, NPY_UINT8, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
		if (npdetids == NULL) goto decrefs;
		dsize = (int64_t) PyArray_DIM(npdetids, 0);
		if (dsize < 1){
			PyErr_SetString(PyExc_ValueError, "must specify at least one detector id");
			goto decrefs;
		}
		dset = (uint8_t*) PyArray_DATA(npdetids);
	}
	else{
		uint8_t maxdet = 0;
		for (int64_t i = 0; i < nphot; i++){
			if (dets[i] > maxdet){
				maxdet = dets[i];
			}
		}
		dsize = 1 + (int64_t) maxdet;
		dset = (uint8_t*) malloc(dsize*sizeof(uint8_t));
		if (dset == NULL) goto decrefs;
		uint8_t detid = 0;
		for (int64_t d = 0; d < dsize; d++){
			dset[d] = detid++;
		}
	}
	out = PyArray_ZEROS(1, dims, NPY_DOUBLE, FALSE);
	if (out == NULL){
		goto decrefs;
	}
	double *maxrates = (double*) PyArray_DATA((PyArrayObject*) out);
	// calculate
	Py_BEGIN_ALLOW_THREADS;
	if (bursts_max_rate(m, clk_p, nphot, times, dets, dsize, dset, nbursts, istarts, istops, ncore, maxrates)){
		PyErr_SetString(PyExc_MemoryError, "insufficient memory for threads");
	}
	Py_END_ALLOW_THREADS;
	// decref arrays
	decrefs:
	Py_XDECREF(nptimes);
	Py_XDECREF(npdets);
	Py_XDECREF(npistarts);
	Py_XDECREF(npistops);
	if (pydetids == NULL){
		if (dset != NULL){
			free(dset);
			dset = NULL;
		}
	}
	else{
		Py_DECREF(npdetids);
	}
	return out;
}

static PyObject* smfbursts_cfuncs_burst_variance_analysis(PyObject* self, PyObject* args, PyObject* kwargs){
	char *kwlist[] = {"dets", "istarts", "istops", "dets_All", "dets_Sub", "n", "ncore", NULL};
	PyObject *pydets = NULL, *pyistarts = NULL, *pyistops = NULL, *pydetidAll = NULL, *pydetidSub = NULL;
	int64_t n = 10;
	Py_ssize_t ncore = 8;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOO|Ln:burst_variance_analysis", kwlist, &pydets, &pyistarts, &pyistops, &pydetidAll, &pydetidSub, &n, &ncore)){
		return NULL;
	}
	// check for invalid values
	if (n < 2){
		PyErr_SetString(PyExc_ValueError, "n must greater than 1");
		return NULL;
	}
	if (ncore < 1){
		PyErr_SetString(PyExc_ValueError, "ncore must greater than 0");
		return NULL;
	}
	// cast all array arguments to well behaved arrays
	PyArrayObject *npdets = (PyArrayObject*) PyArray_FROMANY(pydets, NPY_UINT8, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
	PyArrayObject *npistarts = (PyArrayObject*) PyArray_FROMANY(pyistarts, NPY_INT64, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
	PyArrayObject *npistops = (PyArrayObject*) PyArray_FROMANY(pyistops, NPY_INT64, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
	PyArrayObject *npdetidAll = (PyArrayObject*) PyArray_FROMANY(pydetidAll, NPY_UINT8, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
	PyArrayObject *npdetidSub = (PyArrayObject*) PyArray_FROMANY(pydetidSub, NPY_UINT8, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
	PyObject*out = NULL;
	int64_t nphot = (int64_t) PyArray_DIM(npdets, 0);
	int64_t nbursts = (int64_t) PyArray_DIM(npistarts, 0);
	int64_t dsizeAll = (int64_t) PyArray_DIM(npdetidAll, 0);
	int64_t dsizeSub = (int64_t) PyArray_DIM(npdetidSub, 0);
	if (nbursts != (int64_t) PyArray_DIM(npistops, 0)){
		PyErr_Format(PyExc_ValueError, "istarts and istops must be the same size, got %ld, %ld", nbursts, PyArray_DIM(npistops, 0));
		goto decrefs;
	}
	const npy_intp dims[] = {(npy_intp) nbursts, };
	if (nbursts == 0){
		out = PyArray_ZEROS(1, dims, NPY_DOUBLE, FALSE);
		goto decrefs;
	}
	// get raw data pointers
	uint8_t *dets = (uint8_t*) PyArray_DATA(npdets);
	int64_t *istarts = (int64_t*) PyArray_DATA(npistarts);
	int64_t *istops = (int64_t*) PyArray_DATA(npistops);
	uint8_t *dsetAll = (uint8_t*) PyArray_DATA(npdetidAll);
	uint8_t *dsetSub = (uint8_t*) PyArray_DATA(npdetidSub);
	int64_t i, j;
	for (i = 0; i < nbursts; i++){
		if ((istarts[i] >= nphot) || (istops[i] > nphot)){
			PyErr_SetString(PyExc_ValueError, "istarts/istops index values out of range of size of dets");
			goto decrefs;
		}
	}
	for (i = 0; i < dsizeSub; i++){
		for (j = 0; j < dsizeAll; j++){
			if (dsetSub[i] == dsetAll[j]){
				break;
			}
		}
		if (j == dsizeAll){
			PyErr_SetString(PyExc_ValueError, "all elements of dets_Sub must be present in dets_All");
			goto decrefs;
		}
	}
	out = PyArray_ZEROS(1, dims, NPY_DOUBLE, FALSE);
	double *bvas = (double*) PyArray_DATA((PyArrayObject*) out);
	// compute BVA
	if (burst_variance_analysis(n, dets, nbursts, istarts, istops, dsizeAll, dsetAll, dsizeSub, dsetSub, ncore, bvas) ){
		PyErr_SetString(PyExc_MemoryError, "insufficient memory for threads");
	}
	// decrefs
	decrefs:
	Py_XDECREF(npdets);
	Py_XDECREF(npistarts);
	Py_XDECREF(npistops);
	Py_XDECREF(npdetidAll);
	Py_XDECREF(npdetidSub);
	return out;
}


static inline int pyeval_kde_single(PyObject* pyfunc, int64_t tl, int64_t tt, double tau, double* res){
	PyObject* funcres = PyObject_CallFunction(pyfunc, "LLd", tl, tt, tau);
	if (funcres == NULL){
		return TRUE;
	}
	*res = PyFloat_AsDouble(funcres);
	Py_DECREF(funcres);
	return ((*res == -1.0) && PyErr_Occurred());
}

static inline int kde_self_npf(npy_intp nphot, npy_intp stride, char* times, double tau, int64_t lim, PyObject* pyfunc, double* out){
	npy_intp iloc, iphot, imin = 0, imax = 0;
	int64_t cloc, tmin, tmax;
	double kde_val = 0.0;
	for (iloc = 0; iloc < nphot; iloc++){
		cloc = *(int64_t*)&times[iloc*stride];
		tmin = cloc - lim; // get minimum and maximum times of range to compute in KDE
		tmax = cloc + lim;
		while ((imin < nphot) && (*(int64_t*)&times[imin*stride] < tmin)){ imin++; } // advance until imin is index in range
		while ((imax < nphot) && (*(int64_t*)&times[imax*stride] < tmax)){ imax++; } // advance until imax is just out of range
		for (iphot = imin; iphot < imax; iphot++){
			if (pyeval_kde_single(pyfunc, *(int64_t*)&times[iloc*stride], *(int64_t*)&times[iphot*stride], tau, &kde_val)){ return TRUE; }
			out[iloc] += kde_val;
		}
	}
	return FALSE;
}

static inline int kde_self_exclude_zero_npf(npy_intp nphot, int64_t stride, char* times, double tau, int64_t lim, PyObject* pyfunc, double* out){
	npy_intp iloc, iphot, imin = 0, imax = 0;
	int64_t cloc, tmin, tmax;
	double kde_val = 0.0;
	for (iloc = 0; iloc < nphot; iloc++){
		cloc = *(int64_t*)&times[iloc*stride];
		tmin = cloc - lim; // get minimum and maximum times of range to compute in KDE
		tmax = cloc + lim;
		while ((imin < nphot) && (*(int64_t*)&times[imin*stride] < tmin)){ imin++; } // advance until imin is index in range
		while ((imax < nphot) && (*(int64_t*)&times[imax*stride] < tmax)){ imax++; } // advance until imax is just out of range
		for (iphot = imin; iphot < imax; iphot++){
			if (cloc == *(int64_t*)&times[iphot*stride]){ 
				continue;
			}
			if (pyeval_kde_single(pyfunc, *(int64_t*)&times[iloc*stride], *(int64_t*)&times[iphot*stride], tau, &kde_val)){ return TRUE; }
			out[iloc] += kde_val;
		}
	}
	return FALSE;
}

static inline int kde_other_npf(const npy_intp nphot, const int64_t tstride, char* times, const int64_t nloc, const int64_t lstride, char* locs, const double tau, int64_t lim, PyObject* pyfunc, double* out){
	npy_intp iloc, iphot, imin = 0, imax = 0;
	int64_t cloc, tmin, tmax;
	double kde_val = 0.0;
	for (iloc = 0; iloc < nloc; iloc++){
		cloc = *(int64_t*)&locs[iloc*lstride];
		tmin = cloc - lim; // get minimum and maximum times of range to compute in KDE
		tmax = cloc + lim;
		while ((imin < nphot) && (*(int64_t*)&times[imin*tstride] < tmin)){ imin++; } // advance until imin is index in range
		while ((imax < nphot) && (*(int64_t*)&times[imax*tstride] < tmax)){ imax++; } // advance until imax is just out of range
		for (iphot = imin; iphot < imax; iphot++){
			if (pyeval_kde_single(pyfunc, *(int64_t*)&locs[iloc*lstride], *(int64_t*)&times[iphot*tstride], tau, &kde_val)){ return TRUE; }
			out[iloc] += kde_val;
		}
	}
	return FALSE;
}

static inline int kde_other_exclude_zero_npf(const npy_intp nphot, const int64_t tstride, char* times, const npy_intp nloc, const int64_t lstride, char* locs, const double tau, int64_t lim, PyObject* pyfunc, double* out){
	npy_intp iloc, iphot, imin = 0, imax = 0;
	int64_t cloc, tmin, tmax;
	double kde_val = 0.0;
	for (iloc = 0; iloc < nloc; iloc++){
		cloc = *(int64_t*)&locs[iloc*lstride];
		tmin = cloc - lim; // get minimum and maximum times of range to compute in KDE
		tmax = cloc + lim;
		while ((imin < nphot) && (*(int64_t*)&times[imin*tstride] < tmin)){ imin++; } // advance until imin is index in range
		while ((imax < nphot) && (*(int64_t*)&times[imax*tstride] < tmax)){ imax++; } // advance until imax is just out of range
		for (iphot = imin; iphot < imax; iphot++){
			if (cloc == *(int64_t*)&times[iphot*tstride]){ 
				continue;
			}
			if (pyeval_kde_single(pyfunc, *(int64_t*)&locs[iloc*lstride], *(int64_t*)&times[iphot*tstride], tau, &kde_val)){ return TRUE; }
			out[iloc] += kde_val;
		}
	}
	return FALSE;
}



static PyObject* smfbursts_cfuncs_kde_photons(PyObject* self, PyObject* args, PyObject* kwargs){
	char *kwlist[] = {"times", "tau", "locs", "lim", "func", "drop_self", NULL};
	PyObject *pytimes=NULL, *pylocs=NULL, *pyfunc=NULL;
	double tau, flim = -1.0;
	int drop_self = FALSE, func = 0, err = FALSE;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Od|OdOp:kde_photons", kwlist, &pytimes, &tau, &pylocs, &flim, &pyfunc, &drop_self)){
		return NULL;
	}
	if ((pyfunc != NULL)){
		if (PyCallable_Check(pyfunc)) { func = 3;}
		else{
			func = (int) PyLong_AsLong(pyfunc);
			if (PyErr_Occurred()){
				return NULL;
			}
			if ((func < 0) || (func > 2))
			{
				PyErr_SetString(PyExc_ValueError, "func must be 0 (laplace), 1 (gaussian), 2 (rect), or callable (custom)");
				return NULL;
			}
		}
	}
	int64_t lim;
	if (flim == -1.0){
		switch (func){
			case 1:
				lim = (int64_t) (3.0*tau);
				break;
			case 2:
				lim = (int64_t) tau;
				break;
			default:
				lim = (int64_t) (5.0 * tau);
				break;
		}
	}
	else{
		lim = (int64_t) flim;
	}
	PyArrayObject* nptimes = (PyArrayObject*) PyArray_FROMANY(pytimes, NPY_INT64, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
	if (nptimes == NULL){
		return NULL;
	}
	PyArrayObject* nplocs = (pylocs != NULL) ? (PyArrayObject*) PyArray_FROMANY((PyObject*) pylocs, NPY_INT64, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY) : NULL;
	if ((pylocs !=NULL ) && (nplocs == NULL)){
		Py_DECREF(nptimes);
		return NULL;
	}
	npy_intp nphot = PyArray_DIM(nptimes, 0);
	npy_intp tstride = PyArray_STRIDE(nptimes, 0);
	char *times = PyArray_DATA(nptimes);
	npy_intp nloc = (pylocs != NULL) ? PyArray_DIM(nplocs, 0) : 0;
	npy_intp lstride = (pylocs != NULL) ? PyArray_STRIDE(nplocs, 0) : 0;
	char *locs = (pylocs != NULL) ? PyArray_DATA(nplocs) : NULL;
	npy_intp dims[] = {(pylocs==NULL)? nphot : nloc, };
	PyArrayObject* out = (PyArrayObject*) PyArray_ZEROS(1, dims, NPY_DOUBLE, FALSE);
	if (out == NULL){
		goto decrefs;
	}
	double *dout = (double*) PyArray_DATA(out);
	int fswitch = drop_self + 2*(pylocs != NULL) + 4*(func == 3);
	double (*kdefuncs[3])(int64_t, int64_t, double) = {laplace_kdefunc, gaussian_kdefunc, rect_kdefunc};
	switch(fswitch){
		case 0:
		err = kde_self_np((int64_t)nphot, (int64_t)tstride, times, tau, lim, kdefuncs[func], dout);
		break;
		case 1:
		err = kde_self_exclude_zero_np((int64_t)nphot, (int64_t)tstride, times, tau, lim, kdefuncs[func], dout);
		break;
		case 2:
		err = kde_other_np((int64_t)nphot, (int64_t)tstride, times, (int64_t)nloc, (int64_t)lstride, locs, tau, lim, kdefuncs[func], dout);
		break;
		case 3:
		err = kde_other_exclude_zero_np((int64_t)nphot, (int64_t)tstride, times, (int64_t)nloc, (int64_t)lstride, locs, tau, lim, kdefuncs[func], dout);
		break;
		case 4:
		err = kde_self_npf(nphot, tstride, times, tau, lim, pyfunc, dout);
		break;
		case 5:
		err = kde_self_exclude_zero_npf(nphot, tstride, times, tau, lim, pyfunc, dout);
		break;
		case 6:
		err = kde_other_npf(nphot, tstride, times, nloc, lstride, locs, tau, lim, pyfunc, dout);
		break;
		case 7:
		err = kde_other_exclude_zero_npf(nphot, tstride, times, nloc, lstride, locs, tau, lim, pyfunc, dout);
		break;
	}
	decrefs:
	Py_XDECREF(nptimes);
	nptimes = NULL;
	Py_XDECREF(nplocs);
	nplocs = NULL;
	if (err){
		Py_XDECREF(out);
		out = NULL;
	}
	return (PyObject*) out;
}

static PyObject* smfbursts_cfuncs_cpburstsearch(PyObject* self, PyObject* args, PyObject* kwargs){
	char *kwlist[] = {"times", "dets", "periods", "bg", "sbr", "clk_p", "alpha", "beta", "det_ids", "fuse", "alloc_size", "ncore", NULL};
	PyObject *pytimes = NULL, *pydets = NULL, *pyperiods = NULL, *pybg = NULL, *pysbr = NULL, *pydetids = NULL;
	double clk_p = 5e-8, alpha = 1e-4, beta = 1e-2, fuse = 0.0;
	Py_ssize_t alloc_size = 512, ncore = 8;
	// parse input arguments as python function
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOOd|ddOdnn:cpburstsearch", kwlist, &pytimes, &pydets, &pyperiods, &pybg, &pysbr, &clk_p, &alpha, &beta, &pydetids, &fuse, &alloc_size, &ncore)){
		return NULL;
	}
	// Check that non-array arguments are within relevant ranges
	if (ncore < 1){
		PyErr_SetString(PyExc_ValueError, "must specify at least 1 core (ncore >= 1");
		return NULL;
	}
	if (alloc_size < 1){
		PyErr_SetString(PyExc_ValueError, "alloc_size must be greater than 1");
		return NULL;
	}
	if ( fuse < 0.0){
		PyErr_SetString(PyExc_ValueError, "fuse must be greater than 0");
		return NULL;
	}
	if ((alpha <= 0.0) || (alpha >= 1.0)){
		PyErr_SetString(PyExc_ValueError, "alpha must be in the open interval (0, 1)");
		return NULL;
	}
	if ((beta <= 0.0) || (beta >= 1.0)){
		PyErr_SetString(PyExc_ValueError, "beta must be in the open interval (0, 1)");
		return NULL;
	}
	// Define pointers for input to processing
	PyObject *out = NULL;
	int64_t nphot, nper, dsize = 0; // number of photons, periods, and valid detectors respectively
	int64_t *times = NULL, *periods = NULL; // pointers to times and periods arrays (access numpy arrays)
	double *bg = NULL, *sbr = NULL; // pointers to background and signal to background arrays (access numpy arrays 
	uint8_t *dets = NULL, *dset = NULL; // pointers to the detectors array, and array of detector ids to consider in search
	Bursts *bursts = NULL; // where burstsearch_cp will allocate and store outputs
	// Cast array inputs to the correct types of array 
	PyArrayObject* nptimes = (PyArrayObject*) PyArray_FROMANY(pytimes, NPY_INT64, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
	PyArrayObject* npdets = (PyArrayObject*) PyArray_FROMANY(pydets, NPY_UINT8, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
	PyArrayObject* npperiods = (PyArrayObject*) PyArray_FROMANY(pyperiods, NPY_INT64, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
	PyArrayObject* npbg = (PyArrayObject*) PyArray_FROMANY(pybg, NPY_DOUBLE, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
	PyArrayObject* npsbr = (PyArrayObject*) PyArray_FROMANY(pysbr, NPY_DOUBLE, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
	PyArrayObject *npdetids = (pydetids == NULL)||(pydetids == Py_None) ? NULL: (PyArrayObject*) PyArray_FROMANY(pydetids, NPY_UINT8, 1, 1, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY);
	// if any error, go to decrefs, where all arrays decrefed and cleaned up
	if ((nptimes == NULL)||(npdets == NULL)||(npperiods == NULL)||(npbg == NULL)||(npsbr == NULL)){goto decrefs;}
	if (!((pydetids == NULL)||(pydetids == Py_None))){ // special check if detids not specified
		if (npdetids == NULL) { goto decrefs; }
	}
	// get sizes of each type of array
	nphot = (int64_t) PyArray_DIM(nptimes, 0);
	nper = (int64_t) PyArray_DIM(npbg, 0);
	// compare sizes of arrays with other that should have the same size, raise error if inconsistent
	if (nphot != (int64_t) PyArray_DIM(npdets, 0)){
		PyErr_SetString(PyExc_ValueError, "times and dets arrays must be the same size");
		goto decrefs;
	}
	if (nper != (int64_t) PyArray_DIM(npsbr, 0)){
		PyErr_SetString(PyExc_ValueError, "bg and sbr arrays must be the same size");
		goto decrefs;
	}
	if (nper != (int64_t) (PyArray_DIM(npperiods, 0) - 1)){
		PyErr_SetString(PyExc_ValueError, "periods array must be 1 element larger than bg and sbr arrays");
		goto decrefs;
	}
	// get pointers to data in numpy arrays
	times = (int64_t*) PyArray_DATA(nptimes);
	dets = (uint8_t*) PyArray_DATA(npdets);
	periods = (int64_t*) PyArray_DATA(npperiods);
	bg = (double*) PyArray_DATA(npbg);
	sbr = (double*) PyArray_DATA(npsbr);
	if ((pydetids == NULL)||(pydetids == Py_None)){
		for (int64_t i = 0; i < nphot; i++){ if (dets[i] > dsize) { dsize = (int64_t) dets[i]; } }
		dsize++;
		if ((dset = (uint8_t*) malloc(dsize * sizeof(uint8_t))) == NULL){
			PyErr_SetString(PyExc_MemoryError, "insufficent memory to allocate dset array (how did this happend it should be tiny!)");
			goto decrefs;
		}
		for (uint8_t i = 0; i < dsize; i++){ dset[i] = i; }
	}
	else {
		dsize = (int64_t) PyArray_DIM(npdetids, 0);
		dset = PyArray_DATA(npdetids);
	}
	// perform burst search
	Py_BEGIN_ALLOW_THREADS;
	if (burst_search_cp(alpha, beta, clk_p, nphot, times, dets, dsize, dset, nper, periods, bg, sbr, fuse, alloc_size, ncore, &bursts)){
		PyErr_SetString(PyExc_MemoryError, "insufficent memory for all bursts found");
		goto decrefs;
	}
	Py_END_ALLOW_THREADS;
	// copy data to output numpy arrays
	const npy_intp dims[] = {(npy_intp) bursts->size, };
	PyArrayObject *npstarts = (PyArrayObject*) PyArray_EMPTY(1, dims, NPY_INT64, 0);
	PyArrayObject *npstops = (PyArrayObject*) PyArray_EMPTY(1, dims, NPY_INT64, 0);
	if ((npstarts == NULL)||(npstops == NULL)){
		Py_XDECREF(npstarts);
		Py_XDECREF(npstops);
		PyErr_SetString(PyExc_MemoryError, "insufficent memory for output of bursts");
		goto decrefs;
	}
	memcpy(PyArray_DATA(npstarts), bursts->starts, bursts->size * sizeof(int64_t));
	memcpy(PyArray_DATA(npstops), bursts->stops, bursts->size * sizeof(int64_t));
	// pack output into tuple
	out = PyTuple_Pack(2, (PyObject*)npstarts, (PyObject*) npstops);
	// Decref created numpy arrays
	decrefs:
	Py_XDECREF(nptimes);
	nptimes = NULL;
	Py_XDECREF(npdets);
	npdets = NULL;
	Py_XDECREF(npperiods);
	npperiods = NULL;
	Py_XDECREF(npbg);
	npbg = NULL;
	Py_XDECREF(npsbr);
	npsbr = NULL;
	if ((pydetids == NULL)||(pydetids == Py_None)){if (dset != NULL) {free(dset); dset = NULL;}}
	Py_XDECREF(npdetids);
	npdetids = NULL;
	if (bursts != NULL){
		free_bursts_fields(bursts);
		free(bursts);
		bursts = NULL;
	}
	return out;
}

static PyMethodDef smfbursts_cfuncs_funcs[] = {
	{"index_range", (PyCFunction)smfbursts_cfuncs_index_range, METH_VARARGS|METH_KEYWORDS, 
		"index_range(times, start, stop, prev)\n"
		"--\n\n"
		"Determine first index of first time in times after start and\n"
		"first time in times after stop\n\n"
		"Parameters\n"
		"----------\n"
		"times: np.ndarray[np.int64]\n"
		"    Arrival times of photons\n"
		"start: int\n"
		"    start time of range\n"
		"stop: int\n"
		"    stop time of range\n"
		"prev: int\n"
		"    for speeding computation, the last know index before start\n"
		"\n"
		"Returns\n"
		"-------\n"
		"out: np.ndarray[np.int64]\n"
		"    (2,) shaped array of start and stop indexes\n\n"},
	{"index_ranges", (PyCFunction)smfbursts_cfuncs_index_ranges, METH_VARARGS|METH_KEYWORDS,
		"index_ranges(times, start, stop)\n"
		"--\n\n"
		"Determine first index of first time in times after start and\n"
		"first time in times after stop\n\n"
		"Parameters\n"
		"----------\n"
		"times: np.ndarray[np.uint64]\n"
		"    Arrival times of photons\n"
		"start: np.ndarray[np.uint64]\n"
		"    start times of range\n"
		"stop: np.ndarray[np.uint64]\n"
		"    stop times of range\n"
		"overlap: bool, optional\n"
		"    True if ranges can overlap (i.e. start of next range less\n"
		"    than stop of previous) if True algortithm is less efficient\n"
		"    The default is False.\n\n"
		"Returns\n"
		"-------\n"
		"istarts: np.ndarray[np.int64]\n"
		"    array of start indexes of bursts (closed)\n"
		"istops: np.ndarray[np.int64]]n"
		"    array of stop indexes (open) of bursts\n\n"},
	{"burstsearch", (PyCFunction)smfbursts_cfuncs_burstsearch, METH_VARARGS|METH_KEYWORDS, 
		"burstsearch(times, dets, periods, bg, clk_p, det_ids=None, m=10, F=6.0, c=-1.0, fuse=0.0, alloc_size=512, ncore=8)\n"
		"--\n\n"
		"Perform sliding window burst search on data.\n"
		"Iterates over all windows of consecutive photons of size :math;`m`\n"
		"and if the instantanous rate excedes the computed threshold, it\n"
		"is treated as a burst.\n\n"
		"If ``bg_is_thresh=False`` the a burst is defined as any window\n"
		"Where\n\n"
		".. math::\n\n"
		"    F*\\tau_{bg} \\leq \\frac{m-1-c}{\\Delta_{m}t_{i}}\n\n\n"
		"If ``bg_is_thresh=True`` the a burst is defined as any window\n\n"
		".. math::\n\n"
		"    bg \\leq \\frac{m}{\\Delta_{m}t_{i}}\n\n\n"
		"Note that the start/stop times are \"expanded\" such to included\n"
		"the \"maximum\" degree so that if the first photon were moved\m"
		"earlier, the above inequality would still hold true\n"
		"and likewise, if the last photon where moved later, the\n"
		"inequality would also still hold true.\n"
		"This is done by defining\\:\n\n"
		"#. If ``bg_is_thresh=False``\n\n"
		"   :math:`b_{start} = t_{i_{first}+m} - (F\\tau_{bg})^{-1}`, and\n\n"
		"   :math:`b_{stop} = t_{i_{last}} + (F\\tau_{bg})^{-1}`, or\n"
		"#. If ``bg_thresh=True``\n\n" 
		"   :math:`b_{start} = t_{i_{first}+m} - \\mathrm{Erlang}_{PPF}(P, k=m, \\lambda = \\tau_{bg}^{-1})^{-1}`\n\n"
		"   :math:`b_{stop} = t_{i_{last}} + \\mathrm{Erlang}_{PPF}(P, k=m, \\lambda = \\tau_{bg}^{-1})^{-1}`\n\n\n"
		"Parameters\n"
		"----------\n"
		"times: np.ndarray[np.int64]\n"
		"    Arrival times of photons\n"
		"dets: np.ndarray[np.uint8]\n"
		"    detector indices of photons\n"
		"periods: np.ndarray[np.int64]\n"
		"    bins of background periods, size is 1 larger than number of periods\n"
		"bg: np.ndarray[np.float64]\n"
		"    background photon rate for given photon selection is photons/s\n"
		"clk_p: float\n"
		"    time of single clock in seconds (ie unit of clocks, in s/clock)\n"
		"det_ids: np.ndarray[np.uint8]\n"
		"    array of all detector ids to include in burst search\n"
		"m: int, optional\n"
		"    size of sliding window. The default is 10.\n"
		"F: float, optional\n"
		"    Multiple of background rate for which the rate must exceed\n"
		"    in order for a window of m photons to be considered in a burst\n"
		"c: float\n"
		"    Correction factor used in the rate vs time-lags relation.\n"
		"    Min count rate is given as:math:`(F * bg) / (m - 1 - c)`\n"
		"    Practically the inverse is used, ie  the maximum time between\n"
		"    m photons must be :math:`\\delta T_{m} = (m - 1 - c) / (F * bg)`\n"
		"    The default is -1.0\n"
		"fuse: float, optional\n"
		"    Minimum delta between bursts to fuse (in seconds), \n"
		"    if -1.0, do not fuse bursts.\n"
		"    The default is 0.0\n"
		"bg_is_thresh: bool, optional\n"
		"    If True, ignore F and c, and treat bg array as maximum\n"
		"    separation between m photons to be considered in a burst\n"
		"    The default is False\n"
		"alloc_size: int, optional\n"
		"    The size of array to initially allocate for bursts,\n"
		"    and the ammount by which the array is exteneded when number\n"
		"    of bursts has reached the current size\n"
		"    This parameter does not affect the results, but optimizes\n"
		"    preallocation time. The default is 512\n"
		"ncore: int, optional\n"
		"    Number of threads to use in burst search. This parameter does\n"
		"    not affect the results, only optimizes time, should be related\n"
		"    to number of cores on the current system.\n"
		"\n"
		"Returns\n"
		"-------\n"
		"starts: np.ndarray[np.int64]\n"
		"    (n,) shaped array of start indices\n"
		"stops: np.ndarray[np.int64]\n"
		"    (n,) shaped array of stop indices\n"
		"\n"},
	{"cpburstsearch", (PyCFunction)smfbursts_cfuncs_cpburstsearch, METH_VARARGS|METH_KEYWORDS, 
		"cpburstsearch(times, dets, periods, bg, sbr, clk_p, alpha=0.0001, beta=0.01, det_ids=None, fuse=0.0, alloc_size=512, ncore=8)\n"
		"--\n\n"
		"Perform change point burst search.\n"
		"This is a custom implementation of |Yang|.\n"
		"This burst search presumes that photons are either the result\n"
		"of background (:math:`H_{B}`) or a molecule, ie a burst (:math:`H_{0}`.\n"
		"The :math:`alpha` and :math:`beta` parameters define the probabilities\n"
		"of type I false positives, and type II false negatives respectively.\n\n"
		".. |Yang| replace:: `Kai Zheng, Haw Yang. J. Phys. Chem. B 2005, 109, 46, 21930–21937 <https://doi.org/10.1021/jp0546047>`__\n\n"
		"Parameters\n"
		"----------\n"
		"times: np.ndarray[np.int64]\n"
		"    Arrival times of photons\n"
		"dets: np.ndarray[np.uint8]\n"
		"    detector indices of photons\n"
		"periods: np.ndarray[np.int64]\n"
		"    bins of background periods, size is 1 larger than number of periods\n"
		"bg: np.ndarray[np.float64]\n"
		"    background photon rate for given photon selection is photons/s\n"
		"sbr: np.ndarray[np.float64]\n"
		"    signal to background ratio of expected bursts.\n"
		"clk_p: float\n"
		"    time of single clock in seconds (ie unit of clocks, in s/clock)\n"
		"alpha: float, optional\n"
		"    Likelihood of false positive detection. The default is 0.0001.\n"
		"beta: float, optional\n"
		"    Likelihood of false negative detection. The default is 0.001.\n"
		"det_ids: np.ndarray[np.uint8], optional\n"
		"    array of all detector ids to include in burst search.\n"
		"    Optional, default is None\n"
		"alloc_size: int, optional\n"
		"    The size of array to initially allocate for bursts,\n"
		"    and the ammount by which the array is exteneded when number\n"
		"    of bursts has reached the current size\n"
		"    This parameter does not affect the results, but optimizes\n"
		"    preallocation time. The default is 512\n"
		"ncore: int, optional\n"
		"    Number of threads to use in burst search. This parameter does\n"
		"    not affect the results, only optimizes time, should be related\n"
		"    to number of cores on the current system.\n"
		"\n"
		"Returns\n"
		"-------\n"
		"starts: np.ndarray[np.int64]\n"
		"    (n,) shaped array of start indices\n"
		"stops: np.ndarray[np.int64]\n"
		"    (n,) shaped array of stop indices\n"
		"\n"},
	{"burstgate", (PyCFunction)smfbursts_cfuncs_burstgate, METH_VARARGS|METH_KEYWORDS, 
		"burstgate(starts, stops, truthtable, starttime=None, stoptime=None, alloc_size=512)\n"
		"--\n\n"
		"Perform logical operation on burst ranges. Compile ranges into 2\n"
		"lists, one of starts and the other of stops. Truthtable indexed\n"
		"in same order as appear in 2 lists.\n\n"
		"Parameters\n"
		"----------\n"
		"starts: list[np.ndarray[np.int64]]\n"
		"    N length list of arrays of each set of start times in bursts\n"
		"stops: list[np.ndarray[np.int64]]\n"
		"    N length list of arrays of each set of stop times in bursts\n"
		"truthtable: np.ndarray[np.bool\\_]\n"
		"    Boolean array, with ndim = N and all dimensions size = 2.\n"
		"    Defining which combinations of bursts to count as in a burst.\n"
		"starttime: int, optional\n"
		"    Time at which data is considered to start. If not specified,\n"
		"    defaults to earliest start time in starts. Default is None.\n"
		"stoptime: int, optional\n"
		"    Time at which data is considered to stop. If not specified,\n"
		"    defaults to latest stop time in stops. Default is None.\n"
		"alloc_size: int, optional\n"
		"    The size of array to initially allocate for bursts,\n"
		"    and the ammount by which the array is exteneded when number\n"
		"    of bursts has reached the current size\n"
		"    This parameter does not affect the results, but optimizes\n"
		"    preallocation time. The default is 512\n\n"
		"Returns\n"
		"-------\n"
		"starts: np.ndarray[np.int64]\n"
		"    gated start times\n"
		"stops: np.ndarray[np.int64]\n"
		"    gated stop times\n"
		"\n"},
	{"fusebursts", (PyCFunction)smfbursts_cfuncs_fusebursts, METH_VARARGS|METH_KEYWORDS, 
		"fusebursts(starts, stops, max_sep)\n"
		"--\n\n"
		"Fuse bursts with difference in stop of previous and start of next\n"
		"less than max_sep (in clock units).\n\n"
		"Parameters\n"
		"----------\n"
		"starts: np.ndarray[np.int64]\n"
		"    Start times of input bursts\n"
		"stops: np.ndarray[np.int64]\n"
		"    Start times of input bursts\n"
		"max_sep: int\n"
		"    maximum separation between stop of previous and start of next\n"
		"    allowed for bursts to be fused. Bursts with separation less\n"
		"    than this value will be fused.\n\n"
		"Returns\n"
		"-------\n"
		"starts: np.ndarray[np.int64]\n"
		"    Fused start times\n"
		"stops: np.ndarray[np.int64]\n"
		"    fused stop times\n"
		"\n"},
	{"maximum_rate", (PyCFunction)smfbursts_cfuncs_maximum_rate, METH_VARARGS|METH_KEYWORDS, 
		"maximum_rate(times, dets, istarts, istops, clk_p, det_ids=None, m=10, ncore=8)\n"
		"--\n\n"
		"Compute the maximum m-photon rate of per burst.\n\n"
		"Parameters\n"
		"----------\n"
		"times: np.ndarray[np.int64]\n"
		"    Arrival times of photons in data\n"
		"dets: np.ndarray[np.uint8]\n"
		"    Detector indices of photons in data\n"
		"istarts: np.ndarray[np.int64]\n"
		"    Indices of starts of bursts within times.\n"
		"    On half open interval [istart, istop]\n"
		"istops: np.ndarray[np.int64]\n"
		"    Indices of stops of bursts within times.\n"
		"    On half open interval [istart, istop]\n"
		"clk_p: float\n"
		"    Clock of photons, given in s/clock\n"
		"det_ids: np.ndarray[np.uint8]\n"
		"    Indices in dets over which to compute max rate\n"
		"m: int, optional\n"
		"    Size of sliding window over which to compute max rate.\n"
		"    Default is 10\n"
		"ncore: int, optional\n"
		"    Number of threads to use in burst search. This parameter does\n"
		"    not affect the results, only optimizes time, should be related\n"
		"    to number of cores on the current system.\n\n"
		"Returns\n"
		"-------\n"
		"np.ndarray[np.float64]\n"
		"    Maximum photon rate for the given detectors (in photons/s)\n"
		"    for each burst, computed over the sliding window of size m.\n"
		"\n"},
	{"burst_variance_analysis", (PyCFunction)smfbursts_cfuncs_burst_variance_analysis, METH_VARARGS|METH_KEYWORDS, 
		"burst_variance_analysis(dets, istarts, istops, dets_All, dets_Sub, n=10, ncore=8)\n"
		"--\n\n"
		"Compute burst variance analysis (BVA) of each burst defined by.\n"
		"indices in istarts and istops, for detectors in det\n\n"
		"Parameters\n"
		"----------\n"
		"dets : np.ndarray[np.uint8]\n"
		"    Detector indices of photons in data\n"
		"istarts : np.ndarray[np.int64]\n"
		"    Indices of starts of bursts within times.\n"
		"    On half open interval [istart, istop]\n"
		"istops : np.ndarray[np.int64]\n"
		"    Indices of stops of bursts within times.\n"
		"    On half open interval [istart, istop]\n"
		"dets_All : np.ndarray[np.uint8]\n"
		"    Indices in dets considered in the denominator of the ratio.\n"
		"dets_Sub : np.ndarray[np.uint8]\n"
		"    Indices in dets considered in the numerator of the ratio.\n"
		"n: int, optional\n"
		"    Size of chuck to calculate the variance of the ratio of Sub:All\n"
		"    Default is 10\n"
		"ncore : int, optional\n"
		"    Number of threads to use in burst search. This parameter does\n"
		"    not affect the results, only optimizes time, should be related\n"
		"    to number of cores on the current system.\n\n"
		"Returns\n"
		"-------\n"
		"np.ndarray[np.float64]\n"
		"    Variance of the ratio of Sub:All of all chuncks of size n\n"
		"    in each bursts.\n\n"},
	{"kde_photons", (PyCFunction)smfbursts_cfuncs_kde_photons, METH_VARARGS|METH_KEYWORDS, 
		"kde_photons(times, tau, locs=None, lim=0.0, drop_self=False, func=0)\n"
		"--\n\n"
		"Compute kernel density estimator of photons.\n\n"
		"Parameters\n"
		"----------\n"
		"times: np.ndarray[np.int16]\n"
		"    Arrival times of photons, assuemd to be monotnically increasing\n"
		"tau: float\n"
		"    Decay constant of kde function.\n"
		"locs: np.ndarray[np.int64], optional\n"
		"    Times at which to evaluate the kde. If not specified, use\n"
		"    times array as source of times.\n"
		"    Size of output is same as size of locs. The default is None\n"
		"lim: float, optional\n"
		"    Factor by which to multiply tau at which to include photons\n"
		"    in evaluating KDE. If 0.0, set based on selected func. The default is 0.0.\n"
		"drop_self: np.ndarray[np.uint8]\n"
		"    Photons with same time not dropped from KDE computation.\n"
		"func: int | Callable[[int,int,float],float], optional\n"
		"    Integer index specifying which KDE function to use.\n\n"
		"    - ``0`` laplace KDE :math:`exp( -|t_{j} - t_{i}| / \\tau)`, lim defaults to 5.0\n"
		"    - ``1`` gaussian KDE :math:`exp(-(t_{j}-t_{i})^2/(2*\tau^{2}))`, lim defaults to 3.0\n"
		"    - ``2`` rectanular kde 1.0 if :math:`|t_{j}-t_{i}| < \\tau / 2` 0.0 otherwise\n\n\n"
		"    Or a callable with the signature \n"
		"    ``func(timeloc:int, timephot:int, tau:float)->float``\n"
		"    Which evaluatues the kde contributeion of timephot at point\n"
		"    timeloc, with time constant tau\n\n"
		"Returns\n"
		"-------\n"
		"np.ndarray[np.float64]\n"
		"    Kernel Density for each time in locs or timesn\n"
		},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef smfbursts_cfuncs_module =
{
	PyModuleDef_HEAD_INIT, "smfbursts.cfuncs",
	"C accelerated functions for smfBursts.\n", -1,
	smfbursts_cfuncs_funcs
};

PyMODINIT_FUNC PyInit_cfuncs(void)
{
	PyObject *module = PyModule_Create(&smfbursts_cfuncs_module);
	import_array();
	return module;
};
