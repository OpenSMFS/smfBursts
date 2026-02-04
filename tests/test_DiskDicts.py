#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 17:16:48 2025

@author: paul
"""

import numpy as np
import tables as tb

from fretbursts import DiskDict, NestedDiskDict

import pytest



def all_equal(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape: return False
        return np.all(a==b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b): return False
        return all(all_equal(c, d) for c, d in zip(a, b))
    try:
        out = a == b
    except:
        out = False
    else:
        if not isinstance(out, bool):
            out = False
    return out
        

ddct_params = ((DiskDict, {'times':np.arange(100), 2:np.arange(0,50, 0.5)}, None),
               (NestedDiskDict, {}, {}))

@pytest.fixture
def file_(tmp_path, request):
    f = tb.open_file(tmp_path / 'test.hdf5', 'w')
    yield f
    f.close()
    
@pytest.mark.incremental
class TestDiskDict:
    def test_init(self, sampledict):
        self.tp, self.idct, self.edct = sampledict
        self.test_key = list(self.edct.keys())[0]
        self.test_value = self.edct[self.test_key]
        if self.edct is None: self.edct = self.idct
        self.dct = self.tp(dct=self.idct)
        assert self.dct.hdf5_group is None
        
    
    def test_sethdf5(self, tmp_path):
        f = tb.open_file(tmp_path / 'test.hdf5', 'w')
        g = f.create_group(f.root, 'sub')
        assert self.dct.hdf5_group is None
        self.dct.hdf5_group = g
        assert self.dct.hdf5_group is g
    
    def test_cachedkeys(self):
        for k, v in self.edct.items():
            assert k in self.dct
            assert all_equal(v, self.dct[k])
        for k, v in self.dct.items():
            assert k in self.edct
            assert all_equal(k, self.edct[k])