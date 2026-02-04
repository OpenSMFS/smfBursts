#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 06:48:35 2025

@author: paul
"""

import numpy as np
import tables as tb

from fretbursts.DiskDicts import DiskDict, StrDD, AnyValueDD, NestedDD, MultiArrayValueDD, VattrDD

import pytest

class StrArrayDD_test(StrDD, AnyValueDD):
    pass

class NestedColumnDD_test(NestedDD, MultiArrayValueDD):
    pass


class StrColumnDD_test(VattrDD, MultiArrayValueDD):
    pass

@pytest.fixture
def file_(tmp_path, request):
    f = tb.open_file(tmp_path / 'test.hdf5', 'w')
    yield f
    f.close()
    
@pytest.fixture
def bdct():
    return StrArrayDD_test()

@pytest.fixture
def idct():
    return StrArrayDD_test(dct={'a':np.sin(np.arange(100)), 'b':np.cos(np.arange(100))})

@pytest.mark.incremental
class Test_strdd:
        
    def test_cachedkeys(self, idct):
        self.keys = list()
        for k, v in idct.items():
            assert k in idct
            assert np.all(v == idct[k])
        
    def test_assignment(self, idct, bdct):
        for k in idct.keys():
            bdct[k] = idct[k]
            with pytest.raises(TypeError):
                bdct[k] = np.arange(100)
        
    def test_sethdf5(self, file_, idct):
        g = file_.create_group(file_.root, 'sub')
        idct.hdf5_group = g        
        assert idct.hdf5_group is g
        idct.save()    
        o = DiskDict.from_hdf5_group(file_.root.sub)
        assert frozenset(o.keys()) == frozenset(idct.keys())
        for key in o.keys():
            assert np.all(o[key] == idct[key])
        