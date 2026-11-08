#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 10:14:15 2026

@author: paul
"""

import smfbursts as smf

import pytest


def test_CheckDict():
    dct = smf.datamodel.utils.CheckDict(lambda x, y: (int(x), float(y)))
    dct[1.0] = 1
    dct[2] = 2.0
    dct.setdefault(3.0, 3)
    dct.setdefault(2, 5.0)
    assert all(isinstance(k, int) for k in dct.keys())
    assert all(isinstance(v, float) for v in dct.values())
    assert all(k == v for k, v in dct.items())
    with pytest.raises(Exception):
        dct['help'] = 2.0
    with pytest.raises(Exception):
        dct[3] = 'python'
    l = len(dct)
    assert dct.get(1) == 1.0
    assert len(dct) == l
    assert dct.pop(1) == 1.0
    assert len(dct) == (l - 1)
    assert 1 not in dct
    assert 2 in dct


def test_CheckNestDict():
    def int_key(key):
        if len(key) > 1:
            raise ValueError("integer keys only allow single key specification")
        if not isinstance(key[0], int):
            raise TypeError("only integer keys allowed")
        return key
    
    def str_key(key):
        if len(key) > 2:
            raise ValueError("string keys must be no more than 2 elements long")
        if any(s not in ('a', 'b', 'c', 'd', 'e', 'f', 'g') for s in key):
            raise ValueError("string keys must be between a and g")
        return key
    
    
    def fuse_key(key):
        if len(key) == 0: return key
        if isinstance(key[0], int): return int_key(key)
        return str_key(key)
    
    
    def fuse_check(key, value):
        key = fuse_key(key)
        if isinstance(key[0], int):
            value = int(value)
        else:
            value = float(value)
        return key, value
    dct = smf.datamodel.utils.CheckNestDict(fuse_check, fuse_key)
    dct['a','b'] = 10
    assert isinstance(dct['a','b'], float)
    o = dct['a']
    assert isinstance(o, dict)
    assert len(o) == 1
    assert o['b'] == 10
    dct[0] = 0.0
    assert isinstance(dct[0], int)
    assert dct[0] == 0
    assert 'a' in dct
    assert 0 in dct
    assert 'b' not in dct
    assert 1 not in dct
    assert ('a', 'b') in dct
    assert ('a', 'c') not in dct
    assert ('c', 'b') not in dct
    assert isinstance(dct.setdefault('d', {'a':5.2, 'b':5.3}), dict)
    