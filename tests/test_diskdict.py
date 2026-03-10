#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 06:48:35 2025

@author: paul
"""

import numpy as np
import tables as tb

from fretbursts.datamodel.diskdict import (
    DiskDict, VattrDD, AttrDD, MappedAttrDD, 
    TypedValueDD, NestedDD, SubDiskDict, MaskedDD
    )

import pytest


@pytest.fixture
def file_(tmp_path, request):
    f = tb.open_file(tmp_path / 'pytestdump.hdf5', 'w')
    yield f
    f.close()

@pytest.fixture
def strkeys():
    return ('a', 'b', 'c', 'd')

@pytest.fixture
def arrvals():
    return (np.arange(10), np.arange(10,20, 0.5), 
            np.ones(4, dtype=np.uint8), np.arange(20,30,2, dtype=np.int32))

@pytest.fixture
def strarrdct(strkeys, arrvals):
    return {k:v for k, v in zip(strkeys, arrvals)}


@pytest.mark.dependency(name='diskdict')
def test_cachedkeys(strarrdct):
    dct = DiskDict(strarrdct)
    assert all(k in strarrdct for k in dct.keys()), 'missing key'
    assert all(np.all(strarrdct[k] == v) and strarrdct[k].dtype == v.dtype 
               for k, v in dct.items()), 'value assigned incorrectly'

@pytest.fixture
def ddct(strarrdct):
    try:
        return DiskDict(strarrdct)
    except:
        return None

@pytest.mark.dependency(depends=['diskdict',])
def test_assignment(ddct):
    bdct = DiskDict()
    for k, v in ddct.items():
        bdct[k] = v
        with pytest.raises(TypeError):
            bdct[k] = np.arange(100)


def test_sethdf5(file_, ddct):
    g = file_.create_group(file_.root, 'sub')
    ddct.group = g        
    assert ddct.group is g
    ddct.save()    
    o = DiskDict.load_group(file_.root.sub)
    assert frozenset(o.keys()) == frozenset(ddct.keys())
    for key in o.keys():
        assert np.all(o[key] == ddct[key])
    