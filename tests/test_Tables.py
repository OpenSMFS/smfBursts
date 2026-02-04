#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 13:31:29 2025

@author: paul
"""
import numpy as np

from fretbursts import datamodel as dmd

import pytest

global Prds, DSet
Prds, DSet = None, None


def test_dset():
    global DSet
    class DSet(dmd.DataSet):
        def __init_data__(self, data, **kwargs):
            self.data = data
    try:
        dset = DSet(np.logspace(-1.0,3.0,1000))
    except Exception as e:
        DSet = None
        raise e
    assert dset.data.size == 1000, "data mismanaged"


@pytest.fixture(params=[(-2.0,3.0, 1000), (-3.1, 4.0, 2000)])
def dset_sample(request):
    global DSet
    if DSet is None:
        pytest.skip("could not subclass DataSet")
    return DSet(np.logspace(*request.param))


def test_base_param():
    global Prds
    class Prds(dmd.BaseTable):
        param_defs = (dmd.ParamDef('period', dmd.TV_float, default=np.pi),)
        parent_defs = tuple()
        column_defs = (
            dmd.ColumnDef('prd', tuple(), 0, 'all'),
            dmd.ColumnDef('sep', tuple(), offset=-1, iter_func='_iter_sep', atomic=False)
                       )
    
        def __init_columns__(self):
            self.p_add_column('prd', tuple(),  self.origin.data/self.param.params['period'])
        
        def _iter_sep(self):
            yield from np.diff(self['prd'])
    #try except blocks so that if fail, the Prds global can be set to None
    try:
        prd = dmd.Param(Prds, {'prd':10.0})
    except Exception as e:
        Prds = None
        raise e
    try:
        assert prd.params['prd'] == 10.0, 'param has wrong value'
    except Exception as e:
        Prds = None
        raise e
    try:
        assert len(prd.parents) == 0, 'hallucinates parents'
    except Exception as e:
        Prds = None
        raise e
    with pytest.raises(Exception):
        _ = dmd.Param(Prds, {'prd':10.0}, (prd, ))


@pytest.fixture(params=[{}, {'period':10.0}])
def prds_param(request):
    if Prds is None:
        pytest.skip("cannot create sublcass of BaseTable")
    return dmd.Param(Prds, request)


global make_column
make_column = False


@pytest.mark.skipif(Prds is None, reason="Coutl not create BaseTable to create columns")
def test_basecolumns():
    global Prds, make_column
    param = dmd.Param(Prds, {})
    col = dmd.Column(param, 'prd')
    assert col.atomic, "prd column should be atomic"
    assert col.base_param == param, "wrong base param"
    with pytest.raises(Exception):
        dmd.Column(prds_param, 'prd', offset=1)
    scol = dmd.Column(param, 'sep', offset=0)
    assert scol.param == param
    assert scol.atomic, 'sep marked as atomic when it should not be'
    make_column = True


@pytest.mark.skipif(DSet is None or Prds is None, "Cannot create either DataSet, Table or both")
def test_gettable(prds_param, dset_sample):
    assert isinstance(t:=dset_sample.get_table(prds_param), Prds), "Get returns wrong type"
    assert t.param == prds_param, "prds_param, mismatched param to table"
    assert ('prd', ) in t._cache, "prd not in cache, ColumnDef store all error"
    assert ('sep', ) not in t._cache, "sep in cache, ColumnDef store none error"


@pytest.mark.skipif(not make_column, "Cannot create Column objects")
def test_getcolumn(dset_sample, prds_param):
    global Prds
    pcol = dmd.Column(prds_param, 'prd')
    parr = dset_sample.get_column(pcol)
    assert np.allclose(parr, dset_sample.data/prds_param.params['period']), "back column computation"
    scol = dmd.Column(prds_param, 'sep')
    sarr = dset_sample.get_column(scol)
    sarri = np.array(list(dset_sample.iter_column(scol)))
    assert scol.size == sarri.size, "columns changed sizes"
    scol0 = dmd.Column(prds_param, 'sep', offset=0, fill=1.0)
    sarr0 = dset_sample.get_column(scol0)
    assert sarr0[-1] == -1.0, "wrong fill value"
    scol1 = dmd.Column(prds_param, 'sep', offset=1, fill=-1.0)
    sarr1 = np.ndarray(list(dset_sample.iter_column(scol1)))
    assert sarr1[0] == -1.0, "wrong fill value"
    assert sarr.size + 1 == sarr0.size == sarr1.size, "Columns of inconsistent sizes"
    assert parr is dset_sample.get_column(pcol), "recreating stored column"


global Back
Back = None


@pytest.mark.skipif(Prds is None, reason="Cannot create BaseTable for ChildTable to inherit from")
def test_childtable(dset_sample, prds_param):
    global Back, Prds
    class Back(dmd.ChildTable):
        param_defs = (dmd.ParamDef('bg', dmd.TV_ndarray(dtype=np.int64, dims=dmd.utils.arr_slc[2:5]),),)
        parent_defs = (dmd.ParentDef('base', Prds, is_base=True),)
        column_defs = (dmd.ColumnDef('level',(int,), 0, 'some', get_func='_get_level', dtype=np.int64), )
        
        def _get_level(self, level:int)->np.ndarray[np.int64]:
            out = np.round(self.parents['base']['prd'] + self.param.params['bg'][level%self.param.params['bg'].size]).astype(np.int64)
            if level < self.param.params['bg'].size:
                self.p_add_column('level', (level, ), out)
            return out
    
    try:
        bgpr = dmd.Param(Back, {'bg':np.array([2,10]) }, {'base':prds_param})
    except Exception as e:
        Back = None
        raise e
    assert bgpr.base_param == prds_param, "child table does not recognize correct base param"
    assert bgpr.params['bg'].size == 1, "reshape numpy array"
    lcol = dmd.Column(bgpr, 'level', (1,))
    larr = dset_sample.get_column(lcol)
    assert larr.size == dset_sample.get_column(dmd.Param(Prds,{}), 'prd').size
    assert np.all(np.round(dset_sample.data%10).astype(np.int64) == larr), "incorrect calculation"
    tabel = dset_sample.get_table(bgpr)
    assert ('label', 1) in tabel._cache, "failed to store table of 'some'"
    lcol = dmd.Column(bgpr, 'level', (2,))
    larr = dset_sample.get_column(lcol)
    assert ('label', 2) not in tabel._cache, "stored all when should store only one"


global SPrd
SPrd = None


@pytest.mark.skipif(Back is None, reason="Cannot create ChildTable to derive from")
def test_derived_table(dset_sample, prds_param):
    global SPrd, Back
    def append_n(params:dict)->tuple[dmd.ParamDef,...]:
        if params.get('n') == 2:
            return (dmd.ParamDef('d', dmd.TV_int), )
        return tuple()
        
    class SPrd(dmd.BaseTable):
        param_defs = (dmd.ParamDef('n',dmd.TV_int(mx=0, mx=5), default=2, append_params=append_n), )
        parent_defs = (dmd.ParentDef('bg', Back, is_base=True),)
        column_defs = (dmd.ColumnDef('tt', tuple(), 0, 'user', get_func='_get_tt'),
                       dmd.ColumnDef('bmap', tuple(), 0, mapto=Prds, get_func='_get_bmap'))
        @classmethod
        def validate_params(cls, param:dmd.Param):
            if param.params['n'] == 2 and param.params['d'] == 0:
                raise ValueError("the one illegal param combination")
        
        def _get_tt(self)->np.ndarray[np.float64]:
            return self.parents['bg']['level',0][::2].astype(np.float64)
        
        def _get_bmap(self, param:dmd.Param)->np.ndarray[np.float64]:
            dest = self.origin.get_table(param).size
            dsz, tt = dest['prd'].size, self['tt']
            ssz = tt.size
            out = np.zeros(dsz, dtype=np.float64)
            sz = dsz if dsz < ssz else ssz
            out[:sz] = tt[:sz]
            return out
    
    try:
        bparam = dmd.Param(SPrd, {'n':1}, {'base':prds_param})
    except Exception as e:
        SPrd = None
        raise e
    btbl = dset_sample.get_table(bparam)
    assert btbl.size is None, "Size set prematurely"
    bcol = dmd.Column(bparam, 'tt')
    barr = dset_sample.get_column(bcol)
    assert barr.size == (dset_sample.get_column(dmd.Column(prds_param, 'prd')).size + 1) // 2, "wrong expected size of derived column"
    assert ('tt',) not in btbl._cach, "user value set before user records"
    btbl.record_column('tt')


@pytest.fixture()
def sprd_param():
    global SPrd
    if SPrd is None:
        pytest.skip("could not create parented base table")
    return dmd.Param(SPrd, {'n':1}, {'base':prds_param})


@pytest.mark.skipif(SPrd is None, reason="Cannot create column with parents")
def test_mapedcolumns(dset_sample, prds_param, sprd_param):
    global SPrd
    mp = dmd.Column(sprd_param, 'bmap', (prds_param, ))
    assert mp.base_param == prds_param, "mapped param failed to correctly map"
    assert mp.source_param == sprd_param
    sarr = dset_sample.get_column(mp)
    assert sarr.size == dset_sample.get_column(dmd.Column(prds_param, 'prd')).size, "Mapped columns have incorrect size"