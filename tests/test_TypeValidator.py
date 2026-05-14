#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 07:53:39 2025

@author: paul
"""
import re

import numpy as np
import tables as tb

from smfbursts.datamodel.utils import tupledict, arr_slc
from smfbursts.datamodel.immutabledata import (
    TypeValidator, TV_int, TV_float, TV_bool, TV_bytes, TV_str, TV_attrstr, TV_attrstr_allow_empty, 
    TV_type, TV_typewithnodename, TV_dtype, TV_ndarray, TV_tuple, TV_frozenset, 
    TV_tupledict, TV_PyCode, attr_regex, _echo
    )

import pytest


@pytest.fixture
def file_(tmp_path, request):
    f = tb.open_file(tmp_path / 'test.hdf5', 'w')
    yield f
    f.close()

    
def equals(a,b):
    if type(a) != type(b):
        return False
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype:
            return False
        if a.shape != b.shape:
            return False
        if a.dtype == np.object_:
            return all(equals(aa, bb) for aa, bb in zip(a.flat, b.flat))
        return np.array_equal(a, b, equal_nan=True)
    return a == b


@pytest.fixture(params=[('int', 21, TV_int), ('float', 3.14159, TV_float), 
                        ('true', True, TV_bool), ('false', False, TV_bool),
                        ('bytes', b'hello world', TV_bytes), 
                        ('str', 'the quick brown fox jumped over the lazy dog', TV_str),
                        ('attrstr', 'bg_cache', TV_str),
                        ('type', int, TV_type), ('dtype', np.dtype('<f8'), TV_dtype), 
                        ('pycode', _echo, TV_PyCode),
                        ('ndarray', np.array([1,4,2,6,100]), TV_ndarray),
                        ('ndarray_ragged', np.array([np.arange(i+1) for i in range(16)], dtype=np.object_).reshape(8,2), TV_ndarray),
                        ('tuple', (1,2,3), TV_tuple), ('tuple_noattr', (1, np.dtype('<i1')), TV_tuple),
                        ('fset', frozenset({1,2,3}), TV_frozenset), ('fset_noattr', frozenset({1, np.dtype('<f4')}), TV_frozenset),
                        ('tdict', tupledict(('a', 12), ('b', 'helloworld')), TV_tupledict),
                        ('tdict', tupledict(('a', 12), ('b', 'hello world')), TV_tupledict),
                         ])
def write_vals(request):
    return request.param


def test_tv_retrieve(write_vals):
    _, val, tv = write_vals
    assert TypeValidator.convert_type(type(val)) == tv, 'convert_type retireives wrong type'


def test_write(file_, write_vals):
    name, val, _ = write_vals
    g = TypeValidator.write_any(file_.root, name, val)
    r = TypeValidator.read_any(g)
    assert equals(r, val), f'{name} read back incorrectly: {r} vs {val}'


def test_noderepr(write_vals):
    name, val, tv = write_vals
    if TypeValidator.val_has_node_repr(val):
        nr = TypeValidator.val_to_nodename(val)
        assert nr.split('_', 1)[1] == tv.node_repr(val), 'typevalidor/type mismatch'
        assert equals(val, TypeValidator.read_nodename(nr)), f"{name} has bad node_read or node_repr function"
    else:
        npass = 0
        with pytest.raises(Exception):
            nr = tv.node_repr(val)
            npass += 1
            assert attr_regex.match(nr)
        with pytest.raises(Exception):
            nrtv = TypeValidator.val_to_nodename(val).split('_', 1)[1]
            npass += 1
            assert attr_regex.match(nr)
        assert npass != 1
        if npass == 2:
            assert nr == nrtv, f"{name} gives bad failing node names"
            assert not attr_regex.match(nr), f"{name} have valid node name when node_check indicates it should not"
        
        

@pytest.fixture(params=[TV_int, TV_float])
def tv_numeric(request):
    return request.param


def test_num_limits(tv_numeric):
    tv = tv_numeric(mn=0, mx=10)
    assert tv.check_val(5) == 5
    with pytest.raises(Exception):
        tv.check_val(-1)
    with pytest.raises(Exception):
        tv.check_val(20)


def test_TV_str():
    tv = TV_str(startswith='hello')
    tv.check_val('hello world')
    with pytest.raises(Exception):
        tv.check_val('')
    with pytest.raises(Exception):
        tv.check_val('the quick brown fox')
    tv = TV_str(endswith='world')
    tv.check_val('hello world')
    with pytest.raises(Exception):
        tv.check_val('hello universe')
    rgx = re.compile(r'hello (world|universe)')
    tv = TV_str(pattern=rgx)
    tv.check_val('hello universe')
    tv.check_val('hello world')
    with pytest.raises(Exception):
        tv.check_val('hello earth')
    tv = TV_str(startswith='hello', endswith='moon', pattern=rgx, allow_empty=True)
    tv.check_val('hello universe moon')
    tv.check_val('')
    with pytest.raises(Exception):
        tv.check_val('hello')
    with pytest.raises(Exception):
        tv.check_val('moon')
    with pytest.raises(Exception):
        tv.check_val('great hello universe moon')


def test_TV_attrstr():
    TV_attrstr.check_val('abc')
    with pytest.raises(Exception):
        TV_attrstr.check_val('abc.de')


def test_TV_attstr_allow_empty():
    TV_attrstr_allow_empty.check_val('')
    TV_attrstr_allow_empty.check_val('abc')
    with pytest.raises(Exception):
        TV_attrstr_allow_empty.check_val('abc.def')


def test_TV_type_withnodename():
    TV_typewithnodename.check_val(int)
    with pytest.raises(Exception):
        TV_typewithnodename.check_val(np.ndarray)


def test_TV_tuple():
    tv = TV_tuple(minsize=2, maxsize=4, typedefs=TV_int)
    assert (1,2) == tv.check_val((1,2), )
    assert (1,2,3,4) == tv.check_val((1,2,3,4), )
    with pytest.raises(Exception):
        tv.check_val((1,), )
    with pytest.raises(Exception):
        tv.check_val((1,2,3,4,5), )
    with pytest.raises(Exception):
        tv.check_vals(('abc', 1),)
    tv = TV_tuple(typedefs=(TV_int, TV_float))
    assert (1, 2.0) == tv.check_val((1, 2.0),)
    assert (1, 2.0) == tv.check_val((1, 2),)
    with pytest.raises(Exception):
        tv.check_val((1, "hello world"),)


def test_TV_tupledict():
    tv = TV_tupledict(order=('a', 'b', 'c'))
    td = tupledict(('a', 2), ('b', 3), ('c', 5))
    assert tv.check_val(td) == td, "convert changed tupledict"
    assert tv.check_val((('b', 3), ('a', 2), ('c', 5)), ) == td, 'convert from tuple of tuples fails to reorder'
    assert tv.check_val((2,3,5), ) == td, 'convert from sequence returns incorrectly'
    assert tv.check_val({'c':5, 'b':3, 'a':2}) == td, "convert from dictionary fails to reorder"
    tdbad = tupledict(('d', 44))
    with pytest.raises(Exception):
        tv.check_val(tdbad)
    with pytest.raises(Exception):
        tv.check_val((1,2,3,4), )
    tv = TV_tupledict(required=('a', 'b'))
    tv.check_val({'a':23, 'b':34}) == tupledict(('a', 23), ('b', 34))
    with pytest.raises(Exception):
        tv.check_val({'c':23, 'a':11})
    tv = TV_tupledict(typedefs=TV_float)
    assert tv.check_val(td) == tupledict(('a', 2.0), ('b', 3.0), ('c', 5.0))
    with pytest.raises(Exception):
        tv.check_val({'a':23.3, 'b':'hello world'})
    tv = TV_tupledict(order=('a', 'b', 'c'), required=('a', 'b'), typedefs=(TV_int, TV_float, TV_attrstr))
    td = tupledict(('a', 10), ('b', 20.0), ('c', 'bg_cache'))
    assert tv.check_val(td) == td, 'check_tupledict returned bad value'
    assert tv.check_val((('a', 10), ('b', 20), ('c', 'bg_cache')), ) == td, 'sequence read failed with type convesions'
    assert tv.check_val((10, 20, 'bg_cache'), ) == td, 'sequence read failed with type convesions'
    assert tv.check_val((('b', 20), ('a', 10), ('c', 'bg_cache')), ) == td, 'sequence read failed to reorder with type convesions'
    with pytest.raises(Exception):
        tv.check_val(('a', 32), )
    with pytest.raises(Exception):
        tv.check_val(tupledict(('a', 10), ('b', 20.0), ('c', 'bg_cache'), ('d', 'more than expected')))


def test_TV_ndarray():
    tv = TV_ndarray(mn=0, mx=20, superdtype=np.integer, mindim=2, maxdim=3, dims=arr_slc[2:4,...,4:8])
    a = np.arange(0,21, 1).reshape(3,7)
    assert equals(tv.check_val(a), a), 'unequal array, int64'
    b = np.ones((3,2,7), dtype=np.uint8)
    assert equals(tv.check_val(b), b), 'unequal array, uint8'
    # test out of bounds values
    c = a - 10
    with pytest.raises(Exception):
        tv.check_val(c)
    c = a + 10
    with pytest.raises(Exception):
        tv.check_val(c)
    # test bad shapes
    c = np.ones((2,2,2,4), dtype=np.int64)
    with pytest.raises(Exception):
        tv.checkval(c)
    c = np.ones((1, 4), dtype=np.int64)
    with pytest.raises(Exception):
        tv.checkval(c)
    
