#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing the core cfuncs, the main objective of these tests is to verify
that reference counting is done correctly, ie input variables do not change
refcount after function is called.

Later this may be improved with tests for each individual function that it
creates a realistic value.


Created on Wed Aug 26 15:10:27 2026

@author: paul
"""
from sys import getrefcount as grc
from itertools import chain, product
from numbers import Number
from collections.abc import Sequence

import numpy as np
from scipy.stats import erlang

import smfbursts as smf
import smfbursts.cfuncs as smc

import pytest


def check_sub(val):
    if isinstance(val, np.ndarray):
        return val.dtype == np.object_
    if isinstance(val, Sequence):
        return not isinstance(val[0], Number)
    return False


def refcounts(val):
    if check_sub(val):
        return val, tuple(refcounts(v) for v in val)
    return grc(val)


def refcount_map(args, kwargs):
    kwarg_ord = sorted(kwargs.keys())
    return tuple(refcounts(arg) for arg in args) + tuple(refcounts(kwargs[k]) for k in kwarg_ord)


def check_refcount(func, args, kwargs, skip):
    rmap = refcount_map(args, kwargs)
    out = func(*args, **{k:v for k, v in chain(kwargs.items(), skip.items())})
    assert rmap == refcount_map(args, kwargs), 'refcounts changed'
    return out


def pint(gen:np.random.Generator, default:int=10):
    n = 0
    while (val := gen()) < 1.0 and n < 10: n += 1
    return int(val) if val >= 1.0 else default
        

def molsim(rng):
    return np.cumsum(rng.poisson(50, pint(rng.normal(50.0, 10.0))))


def check_fused(start, stop, fuse):
    if fuse >= 0.0:
        assert np.all(start[1:] > stop[:-1])
        assert np.all(np.diff(start)) >= 0 and np.all(np.diff(stop) >= 0)


def test_burstsearch(data, default_bg):
    bgall = data.get_table(default_bg)['bg', smf.PhSel('all')]
    bg0ex = data.get_table(default_bg)['bg', smf.PhSel('0ex')]
    times = data.times.copy()
    periods = data.get_table(default_bg.parents['base'])['periods']
    dets = data.dets.copy()
    clk_p = data.clk_p
    det_id = np.array([0,1], dtype=np.uint8)
    Pall = erlang.ppf(0.95, 10, 1/bgall)
    P0ex = erlang.ppf(0.95, 10, 1/bg0ex)
    for f in (-1.0, 0.0, 10*clk_p):
        start, stop = check_refcount(smc.burstsearch, (times, dets, periods, bgall), {}, {'clk_p':clk_p, 'fuse':f})
        check_fused(start, stop, f)
        start, stop = check_refcount(smc.burstsearch, (times, dets, periods, Pall), {}, {'clk_p':clk_p, 'fuse':f, 'bg_is_thresh':True})
        check_fused(start, stop, f)
        start, stop = check_refcount(smc.burstsearch, (times, dets, periods, bg0ex), {'det_ids':det_id}, {'clk_p':clk_p, 'fuse':f})
        check_fused(start, stop, f)
        start, stop = check_refcount(smc.burstsearch, (times, dets, periods, P0ex), {'det_ids':det_id}, {'clk_p':clk_p, 'fuse':f, 'bg_is_thresh':True})
        check_fused(start, stop, f)
        

    
def test_burstsearch_cp(data, sper_bg):
    bgall = data.get_table(sper_bg)['bg', smf.PhSel('all')]
    bg0ex = data.get_table(sper_bg)['bg', smf.PhSel('0ex')]
    times = data.times.copy()
    periods = data.get_table(sper_bg.parents['base'])['periods']
    dets = data.dets.copy()
    clk_p = data.clk_p
    det_id = np.array([0,1], dtype=np.uint8)
    sbr = np.ones(bgall.size)*20.0
    check_refcount(smc.cpburstsearch, (times, dets, periods, bgall, sbr), {}, {'clk_p':clk_p, 'alpha':1e-4, 'beta':1e-2})
    check_refcount(smc.cpburstsearch, (times, dets, periods, bg0ex, sbr), {'det_ids':det_id}, {'clk_p':clk_p, 'alpha':1e-4, 'beta':1e-2})


def test_index_range(data):
    times = data.times.copy()
    stst = check_refcount(smc.index_range, (times, ), {}, {'start':times[0]-3, 'stop':times[2]+1})
    assert stst.shape == (2, ), "wrong size of index_range"
    assert stst[0] == 0 and stst[1] == 3, 'incorrect indexing'
    stst = check_refcount(smc.index_range, (times, ), {}, {'start':times[1000]-3, 'stop':times[1009]+2, 'prev':980})
    assert stst.shape == (2, ), "wrong size of index_range"
    assert stst[0] == 1000 and stst[1] == 1010, 'incorrect indexing'


def test_index_ranges(data, default_bg):
    bgall = data.get_table(default_bg)['bg', smf.PhSel('all')]
    periods = data.get_table(default_bg.parents['base'])['periods']
    times = data.times.copy()
    dets = data.dets.copy()
    clk_p = data.clk_p
    starts, stops = smc.burstsearch(times, dets, periods, bgall, clk_p)
    istarts, istops = check_refcount(smc.index_ranges, (times, starts, stops), {}, {})
    prev = 0
    index_list = list()
    for start, stop in zip(starts, stops):
        b, e = smc.index_range(times, start, stop, prev)
        prev = e
        index_list.append([b, e])
    index_arr = np.array(index_list)
    assert np.all(index_arr[:,0] == istarts) and np.all(index_arr[:,1] == istops), "index_range and index_ranges produce differing results"
    
    
def test_bva(data, default_bg):
    bgall = data.get_table(default_bg)['bg', smf.PhSel('all')]
    periods = data.get_table(default_bg.parents['base'])['periods']
    times = data.times.copy()
    dets = data.dets.copy()
    clk_p = data.clk_p
    starts, stops = smc.burstsearch(times, dets, periods, bgall, clk_p, fuse=0.0)
    istarts, istops = smc.index_ranges(times, starts, stops)
    dets_All = np.array([0, 1], dtype=np.uint8)
    dets_Sub = np.array([1, ], dtype=np.uint8)
    check_refcount(smc.burst_variance_analysis, (dets, istarts, istops, dets_All, dets_Sub), {}, {})


def test_fusebursts():
    starts = np.array([ 10,  20,  40, 100, 140, 450, 500], dtype=np.int64)
    stops  = np.array([ 20,  30,  60, 120, 180, 501, 600], dtype=np.int64)
    starts, stops = check_refcount(smc.fusebursts, (starts, stops), {}, {'max_sep':0})
    assert np.all(starts == np.array([ 10,  20,  40, 100, 140, 450]))
    assert np.all(stops == np.array([ 20,  30,  60, 120, 180, 600]))
    starts, stops = check_refcount(smc.fusebursts, (starts, stops), {}, {'max_sep':1})
    assert np.all(starts == np.array([ 10,  40, 100, 140, 450]))
    assert np.all(stops == np.array([ 30,  60, 120, 180, 600]))
    starts, stops = check_refcount(smc.fusebursts, (starts, stops), {}, {'max_sep':11})
    assert np.all(starts == np.array([ 10, 100, 140, 450]))
    assert np.all(stops == np.array([ 60, 120, 180, 600]))
   

def test_maxrate(data, default_bg):
    bgall = data.get_table(default_bg)['bg', smf.PhSel('all')]
    periods = data.get_table(default_bg.parents['base'])['periods']
    times = data.times.copy()
    dets = data.dets.copy()
    clk_p = data.clk_p
    starts, stops = smc.burstsearch(times, dets, periods, bgall, clk_p, np.array([0,1], dtype=np.uint8))
    istarts, istops = smc.index_ranges(times, starts, stops)
    check_refcount(smc.maximum_rate, (times, dets, istarts, istops), {}, {'clk_p':clk_p, 'm':5})
    check_refcount(smc.maximum_rate, (times, dets, istarts, istops), {'det_ids':np.array([0,1], dtype=np.uint8)}, {'clk_p':clk_p, 'm':5})
    

def test_kde_photons(data, default_bg):
    times = data.times[:2000].copy()
    dets = data.dets[:2000].copy()
    ta = times[np.isin(dets, [0,1])]
    tb = times[np.isin(dets, [3,])]
    tc = times[np.isin(dets, [0,3])]
    clk_p = data.clk_p
    for func, ds in product(range(3), (False, True)):
        check_refcount(smc.kde_photons, (ta,), {}, {'tau':5e-4/clk_p, 'func':func})
        check_refcount(smc.kde_photons, (ta, ), {'locs':tc}, {'tau':5e-4/clk_p, 'lim':5.0, 'func':func, 'drop_self':ds})
        check_refcount(smc.kde_photons, (ta,), {'locs':tb}, {'tau':5e-4/clk_p, 'func':func, 'drop_self':ds})
    def exp(i, j, f):
        return np.exp(abs(i-j)/f)
    check_refcount(smc.kde_photons, (ta, ), {'func':exp}, {'tau':5e-4/clk_p})
    check_refcount(smc.kde_photons, (ta, ), {'func':exp, 'locs':tb}, {'tau':5e-4/clk_p, 'lim':5.0, 'drop_self':False})
    check_refcount(smc.kde_photons, (ta, ), {'func':exp, 'locs':tc}, {'tau':5e-4/clk_p, 'drop_self':True})