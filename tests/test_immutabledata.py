# -*- coding: utf-8 -*-
# Author : Paul David Harris
# Email : harripd@gmail.com
# Created : 18/04/2025
"""
Testing immutabledata submodule of datamodel of FRETBursts
"""

import numpy as np
import tables as tb

from fretbursts.datamodel.utils import ImDict, FixedDict, arr_slc, tupledict, _eq
from fretbursts.datamodel.immutabledata import (
    _ImData, TypeValidator, register_PyCode, 
    TV_int, TV_float, TV_ndarray, TV_str, TV_bool, TV_type, TV_PyCode, 
    TV_typewithnodename, TV_tuple, TV_tupledict, TV_ImData
    )

import pytest


def get_ImClass(name):
    for cname, imclass in _ImData._registered.items():
        if cname.split('.')[-1] == name:
            return imclass
    return None


def get_PyCode(name):
    for cname, pcode in TypeValidator.get_subgroup('byteslike', 'pycode').items():
        if cname.split('.')[-1] == name:
            return pcode
    return None


def test_TypeValidatorCall():
    """Test making sub-TypeValidator"""
    assert isinstance(TV_int(mn=3), TypeValidator)

@pytest.fixture(params=[(TV_int, 1, 0.2, {'mn':0, 'mx':4}, (-1, 5)), 
                        (TV_float, 1.0, 'a', {'mn':0, 'mx':4}, (-0.1, 5.0)), 
                        (TV_bool, True, 1, {}, tuple())])
def base_TV(request):
    """Simple type validator, return TV, good value, fail value, kwargs for call, and fail examples with kwargs"""
    return request.param

def test_TV_number(base_TV):
    """Test TypeValidator on numbers"""
    tv, valid, failtype, failkws, fails = base_TV
    assert valid == tv.check_val(valid, {}), "Incorrect return type"
    assert valid == tv.check_val(valid), "Incorrect return type"
    with pytest.raises(TypeError):
        tv.check_val(failtype)
    if not failkws:
        return
    tv_new = tv(**failkws)
    for fail in fails:
        with pytest.raises(ValueError):
            tv_new.check_val(fail)


def test_ndarray_num():
    """Test typevalidator on numpy arrays"""
    arr1d = np.arange(0,4,1)
    arrsq = arr1d.reshape(2,2)
    assert np.all(arr1d == TV_ndarray.check_val(arr1d))
    TV_2d = TV_ndarray(mindim=2)
    assert np.all(arrsq==TV_2d.check_val(arrsq))
    TV_sq = TV_ndarray(square=True)
    assert np.all(arrsq==TV_sq.check_val(arrsq))
    with pytest.raises(Exception):
        TV_2d.check_val(arr1d)
    with pytest.raises(Exception):
        TV_sq.check_val(arr1d.reshape(1,-1))
    TV_slc = TV_ndarray(dims=arr_slc[1:3, ...])
    assert np.all(TV_slc.check_val(arrsq) == arrsq)
    assert np.all(arr1d[:2] == TV_slc.check_val(arr1d[:2]))
    with pytest.raises(Exception):
        TV_slc.check_val(arr1d)
    tv_rng = TV_ndarray(mn=1, mx=2)
    with pytest.raises(ValueError):
        tv_rng.check_val(np.arange(0,2,1))
    with pytest.raises(ValueError):
        tv_rng.check_val(np.arange(1,4,1))


def test_type():
    assert TV_type.check_val(int) == int
    with pytest.raises(Exception):
        TV_type.check_val(1.2)
    assert TV_typewithnodename.check_val(float) == float
    with pytest.raises(Exception):
        TV_typewithnodename.check_val(np.ndarray)
        

@pytest.mark.dependency(name='gate')
def test_PyCode():
    """Test TypeValidator on new PyCode objects"""
    def ge_gate(col, val=0.0):
        return col > val
    register_PyCode(ge_gate)
    def le_gate(col, val=np.inf):
        return col < val
    TV_PyCode.check_val(ge_gate)
    with pytest.raises(Exception):
        TV_PyCode.check_val(le_gate)


def test_tupledict():
    """Test TypeValidator on tupledicts"""
    typedefs = {'a':TV_int, 'b':TV_float(mn=0.0), 'c':TV_str}
    td = tupledict(('a',1), ('b', 2.0), ('c', 'hello world'))
    tdf = tupledict(('a', 1), ('b', -0.25), ('c', 'hello world'))
    TV_td_abc = TV_tupledict(order=('c', 'b', 'a'), required={'a', 'b', 'c'},
                             typedefs=typedefs)
    TV_tdpass = TV_tupledict(data_proc=lambda x: dict(typedefs=x))
    assert td == TV_tupledict.check_val(td)
    TV_td_abc.check_val(td)
    with pytest.raises(Exception):
        TV_td_abc.check_val(tdf)
    tds = td[:2]
    assert td == TV_tupledict.check_val(td)
    with pytest.raises(Exception):
        TV_td_abc.check_val(tds)
    assert td == TV_tdpass.check_val(td, {'a':TV_int, 'b':TV_float, 'c':TV_str})
    with pytest.raises(Exception):
        TV_tdpass.check_val(tdf, typedefs)


def test_tuple():
    """Test TypeValidator on tuples"""
    TV_comp = TV_tuple(typedefs=(TV_int, TV_float, TV_str))
    t = (1, 2.3, 'hello world')
    TV_arrays = TV_tuple(typedefs=TV_ndarray(dtype=np.float64))
    ta = (np.array([1,2,3]), np.array([[0.1, 0.2], [0.3, 0.5]]))
    assert _eq(TV_tuple.check_val(t), t)
    assert _eq(TV_comp.check_val(t), t)
    with pytest.raises(Exception):
        TV_comp.check_val(ta)
    assert _eq(ta, TV_arrays.check_val(ta))
    with pytest.raises(Exception):
        TV_arrays.check_val(t)


def test_validator():
    """Test validator argument to TypeValidator works"""
    def int_validator(val, isin=None, **kwargs):
        if isin is not None and val not in isin:
            raise ValueError("invalid isin option")
        return val
    TV_isinint = TV_int(validator=int_validator)
    assert 5 == TV_isinint.check_val(5, {})
    TV_is1to4 = TV_isinint(isin=(1,2,3,4))
    assert 2 == TV_is1to4.check_val(2, {})
    with pytest.raises(ValueError):
        assert 5 == TV_is1to4.check_val(5, {})
    def predata_process(predata):
        if 'isin_' in predata:
            return dict(isin=predata['isin_'])
        return dict()
    TV_checkint = TV_isinint(data_proc=predata_process)
    assert TV_checkint.check_val(5, {}) == 5
    assert TV_checkint.check_val(5, {'isin_':(1,2,3,4,5)}) == 5
    with pytest.raises(ValueError):
        TV_checkint.check_val(5, {'isin_':(1,2,3,4)})


@pytest.mark.dependency(name='simple')
def test_ImData_attrs():
    """Test attribute retreval from ImData"""
    class ImDataTest_simple(_ImData):
        __slots__ = ('i', 'f')
        _typeconversions = ImDict(i=TV_int, f=TV_float)
    data = ImDataTest_simple(1)
    assert 'i' in data
    assert 'f' not in data
    assert data.i == 1
    data = ImDataTest_simple(1,2.0)
    assert 'i' in data
    assert 'f' in data
    assert data.i == 1
    assert data.f == 2.0
    data = ImDataTest_simple(i=1)
    assert 'i' in data
    assert 'f' not in data
    assert data.i == 1
    data = ImDataTest_simple(1, f=2.0)
    assert 'i' in data
    assert 'f' in data
    assert data.i == 1
    assert data.f == 2.0


@pytest.mark.dependency(name='defaults')
def test_ImData_defaults():
    """Test that _defaults class var works with ImData"""
    class ImDataTest_defaults(_ImData):
        __slots__ = ('i', 'f')
        _typeconversions = ImDict(i=TV_int, f=TV_float)
        _defaults = FixedDict(i=2, f=0.5)
    data = ImDataTest_defaults()
    assert 'i' in data
    assert 'f' in data
    assert data.i == 2
    assert data.f == 0.5
    data = ImDataTest_defaults(f=9.0)
    assert data['f'] == 9.0
    assert TV_ImData.check_val(data) == data
    assert TV_ImData(subclass=ImDataTest_defaults).check_val(data) == data


@pytest.mark.dependency(depends=['simple', 'defaults'])
def test_ImData_subclassfail():
    """Test TV_ImData correctly handles subclass argument (ie fails when given wrong type)"""
    data = get_ImClass('ImDataTest_defaults')()
    with pytest.raises(Exception):
        TV_ImData(subclass=get_ImClass('ImDataTest_simple')).check_val(data)


@pytest.mark.dependency(name='required')
def test_ImData_required():
    """Test ImData object with _required class var"""
    class ImDataTest_required(_ImData):
        __slots__ = ('i', 'f')
        _typeconversions = ImDict(i=TV_int, f=TV_float)
        _required = frozenset({'i',})
    data = ImDataTest_required(23)
    assert 'i' in data
    assert 'f' not in data
    assert data.i == 23
    data = ImDataTest_required(32, 2.3)
    assert data.i == 32
    assert data.f == 2.3
    assert TV_ImData.check_val(data) == data
    with pytest.raises(Exception):
        data = ImDataTest_required(f=5.6)


@pytest.mark.dependency(name='complete')
def test_ImData_complete():
    """Test that ImData subclass specifying everything remains functional"""
    class ImDataTest_complete(_ImData):
        __slots__ = ('i', 'f', 'string', 'tpl', 'tdict')
        _typeconversions = ImDict(i=TV_int, f=TV_float, string=TV_str, tpl=TV_tuple, tdict=TV_tupledict)
        _defaults = dict(i=3, f=2.9, string='hello world', tpl=(1,np.array([1,2,3]),3), tdict=tupledict(('a','monkey'), ('b', 23)))
    td = tupledict(('a','gorilla'), ('b', 42))
    data = ImDataTest_complete(tdict=td)
    assert data.tdict == td


def all_pass(*args, **kwargs):
    return None


def call_none(func):
    if func is None:
        return all_pass
    return func


@pytest.fixture(params=["array_", "int_", "float_", "str_", "tuple_", "tupledict_", 
                        "imsimple_", "imdefaults_", "imrequired_", "imcomplete_"])
def all_val_types(request):
    """Fixture to test each type of type validator"""
    tps = dict (array_= ('ndarray', np.arange(4).reshape(2,2)),
                int_ = ('int', 1),
                float_ = ('float', 1.0),
                bool_ = ('bool', True),
                str_ = ("str", "hello world"),
                tuple_ = ('tuple', (1, 1.2, 'hello wolrd')),
                tupledict_ = ('tupledict', tupledict(('a', 1), ('b', 2.0), ('c', 'hello world'))),
                pycode_ = ('py_code', get_PyCode('ge_gate')),
                imsimple_ = ('ImDataTest_simple', call_none(get_ImClass('ImDataTest_simple'))(4,2.71)),
                imdefaults_ = ('ImDataTest_defaults', call_none(get_ImClass('ImDataTest_defaults'))()),
                imrequired_ = ('ImDataTest_required', call_none(get_ImClass('ImDataTest_required'))(8,3.14)),
                imcomplete_ = ('ImDataTest_complete', call_none(get_ImClass('ImDataTest_complete'))()))
    if tps[request.param][1] is None:
        pytest.skip(f'cannot create {request.param}')
        return None
    return tps[request.param]


@pytest.fixture()
def file_(tmp_path, request):
    """Open HDF5 file to test"""
    f = tb.open_file(tmp_path / 'test.hdf5', 'w')
    yield f
    f.close()


def test_write(file_, all_val_types):
    """Test writing files to HDF5"""
    TypeValidator.write_any(file_.root, *all_val_types)
    assert _eq(TypeValidator.read_any(file_.root[all_val_types[0]]), all_val_types[1])
