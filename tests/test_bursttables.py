#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 14:38:21 2025

@author: paul
"""
import numpy as np

import fretbursts as frb

import pytest


def test_burst_param(data):
    bg = frb.bg.make_bg_param(data)
    brst = frb.Param(frb.Bursts, {'streams':frb.PhSel('all'), 'm':10, 'F':6.0}, {'bg':bg})
    assert len(brst.params['streams']) == 1
    assert len(brst.parents['bg']) == 1
    assert brst.params['m'].size == 1
    brst = frb.Param(frb.Bursts, {'streams':(frb.PhSel('0ex'), frb.PhSel('1ex1em')), 'm':10, 'F':6.0}, {'bg':bg})
    assert len(brst.params['streams']) == 2
    assert len(brst.parents['bg']) == 2
    assert brst.params['m'].size == 2
    assert brst.params['F'].size == 2
    assert np.all(brst.params['m'] == 10)
    assert np.all(brst.params['F'] == 6.0)
    brst1 = frb.Param(frb.Bursts, {'streams':(frb.PhSel('0ex'), frb.PhSel('1ex1em')), 
                                   'm':np.array([10,12]), 'F':np.array([6.0, 10.0])},
                      {'bg':(bg, frb.bg.make_bg_param(data, func=frb.bg.exp_cdffit))})
    assert len(brst.params['streams']) == 2
    assert len(brst.parents['bg']) == 2
    assert brst.params['m'].size == 2
    assert brst.params['F'].size == 2
    assert brst.params['truthtable'].sum() == 1
    brst2 = frb.Param(frb.Bursts, {'streams':(frb.PhSel('1ex1em'), frb.PhSel('0ex')), 
                                   'm':np.array([12,10]), 'F':np.array([10.0, 6.0])},
                      {'bg':(frb.bg.make_bg_param(data, func=frb.bg.exp_cdffit), bg)})
    assert brst1 == brst2


@pytest.fixture()
def burst(data):
    bg = frb.bg.make_bg_param(data)
    return frb.Param(frb.Bursts, {'streams':frb.PhSel('all'), 'm':10, 'F':6.0}, {'bg':bg})
    

def test_geq_gate(data, burst):
    nph = frb.Column(burst, 'nph_raw', frb.PhSel('0ex_1ex1em'))
    g50 = frb.make_geq_gate(nph, 50)
    nph50 = nph.regate(g50)
    nphcol = data.get_column(nph)
    assert np.all(data.get_column(nph50) == nphcol[nphcol >= 50]), 'Improper masking'


def test_bursts_startstops(data, burst):
    cstart = frb.Column(burst, 'start')
    cstop = frb.Column(burst, 'stop')
    cistart = frb.Column(burst, 'istart')
    cistop = frb.Column(burst, 'istop')
    cistarttime = frb.Column(burst, 'istarttime')
    cistoptime = frb.Column(burst, 'istoptime')
    assert np.all(data.get_column(cstart) < data.get_column(cstop)), "start before stop"
    assert np.all(data.get_column(cistart) < data.get_column(cistop)), "istart before istop"
    assert np.all(data.get_column(cistarttime) <= data.get_column(cistoptime)), "istarttime before istoptime"
    assert np.all(data.get_column(cstart) <= data.get_column(cistarttime)), "istarttime before start"
    assert np.all(data.get_column(cstop) >= data.get_column(cistoptime)), "istoptime before stop"


def test_burst_midtime(data, burst, colstart, colstop):
    mt = frb.Column(burst, 'sep', (colstart, colstop))
    data.get_column(mt)


def test_nph_raw(data, burst):
    nphs = [frb.Column(burst, 'nph_raw', frb.PhSel(f'0ex{i}em')) for i in range(data.detdef.em)]
    nph_d = frb.Column(burst, 'nph_raw', frb.PhSel('0ex'))
    assert np.all(np.sum([data.get_column(c) for c in nphs], axis=0) == data.get_column(nph_d)), "nph_raw computed incorrectly"


def test_burst_ratio_raw(data, burst):
    E = frb.Column(burst, 'ratio_raw', (frb.PhSel('0ex0em'), frb.PhSel('0ex')))
    E_i = frb.Column(burst, 'ratio_raw', (frb.PhSel('0ex'), frb.PhSel('0ex0em')))
    eraw = data.get_column(E)
    erawi = data.get_column(E_i)
    erawi = 1 / erawi
    eraw[eraw==np.inf] = np.nan
    erawi[erawi==np.inf] = np.nan
    assert np.allclose(eraw, erawi, equal_nan=True), "ratio_raw not computing inverse"


def test_meanT(data, burst):
    meanT = frb.Column(burst, 'meanT', frb.PhSel('all'))
    assert np.all(np.diff(data.get_column(meanT)) > 0), "non-monotonic mean T"


def test_mTdiff(data, burst):
    mTdiff = frb.Column(burst, 'mTdiff', (frb.PhSel('0ex0em'), frb.PhSel('0ex1em')))
    data.get_column(mTdiff)


def test_burst_brightness(data, burst):
    br = frb.Column(burst, 'brightness', frb.PhSel('0ex0em'))
    data.get_column(br)


def test_burst_dur(data, burst, colstart, colstop):
    dur = frb.Column(burst, 'dur', ('start', 'stop'))
    mdur = data.get_column(dur)
    dur = frb.Column(burst, 'dur', (colstart, colstop))
    assert np.all(mdur >= data.get_column(dur)), "incorrect calculation of duration of burst"


def test_burst_sep(data, burst, colstart, colstop):
    sep = frb.Column(burst, 'sep', (colstart, colstop))
    data.get_column(sep)


def test_max_rate(data, burst):
    mrall = frb.Column(burst, 'max_rate', frb.PhSel('all'))
    mrDD = frb.Column(burst, 'max_rate', frb.PhSel('0ex0em'))
    mall, mdd = data.get_column(mrall), data.get_column(mrDD)
    mask = ~(np.isnan(mall) | np.isnan(mdd))
    mall, mdd = mall[mask], mdd[mask]
    mask = ~(np.isnan(mall) | np.isnan(mdd))
    assert np.all(mall[mask] >= mdd[mask]), "DD stream max rate larger than all"


def test_bva(data, burst):
    data.get_column(frb.Column(burst, 'bva', (frb.PhSel('0ex1em'), frb.PhSel('0ex'), 10)))
    

def test_ebva(data, burst):
    bva = data.get_column(frb.Column(burst, 'bva', (frb.PhSel('0ex1em'), frb.PhSel('0ex'), 10)))
    ebva = data.get_column(frb.Column(burst, 'ebva', (frb.PhSel('0ex1em'), frb.PhSel('0ex'), 10)))
    mask = ~np.isnan(ebva)
    assert np.all(bva[mask] >= ebva[mask]), "ebva larger than bva"


def test_nanohist(data, burst):
    nh = frb.Column(burst, 'nanohist', (frb.PhSel('0ex0em'), False))
    nhf = frb.Column(burst, 'nanohist', (frb.PhSel('0ex0em'), True))
    hsize = np.diff(data.setup.ex_ranges[0])[0]
    for h, hf in zip(data.iter_column(nh), data.iter_column(nhf)):
        assert hsize == h.size, 'incorrect size of sub-nanohist column'
        assert hsize < hf.size, "full column is smaller than sub column"
        assert h.sum() == hf.sum(), 'inconsistent histograms between sub and full nanohist'


def test_nanomean(data, burst):
    # set irf_thresh
    for sel in (frb.PhSel('0ex0em'), frb.PhSel('0ex1em'), frb.PhSel('1ex1em')):
        data.irf_thresh[sel] = np.argmax(data.get_column(frb.Column(burst, 'nanohist', (sel, True))).sum(axis=0))
        data.get_column(frb.Column(burst, 'nanomean', sel))


def test_make_NphBG(burst):
    frb.Param(frb.NphBG, {'single':True}, {'base':burst, 'bg':burst.parents['bg'][0]})


@pytest.fixture()
def nph(burst):
    return frb.Param(frb.NphBG, {'single':True}, {'base':burst, 'bg':burst.parents['bg'][0]})


def test_nph_bg(data, nph, colstart, colstop):
    nphDD = frb.Column(nph, 'nph_bg', (frb.PhSel('0ex0em'), colstart, colstop))
    nphAA = frb.Column(nph, 'nph_bg', (frb.PhSel('1ex1em'), colstart, colstop))
    nphAll = frb.Column(nph, 'nph_bg', (frb.PhSel('all'), colstart, colstop))
    nphDD_raw = frb.Column(nph.parents['base'], 'nph_raw', frb.PhSel('0ex0em'))
    nphAA_raw = frb.Column(nph.parents['base'], 'nph_raw', frb.PhSel('1ex1em'))
    assert np.all(data.get_column(nphDD) <= data.get_column(nphDD_raw)), "bg adjusted column greater than raw"
    assert np.all(data.get_column(nphAA) <= data.get_column(nphAA_raw)), "bg adjusted column greater than raw"
    data.get_column(nphAll)


def test_ratio_bg(data, nph, colstart, colstop):
    nphD = frb.Column(nph, 'nph_bg', (frb.PhSel('0ex'), colstart, colstop))
    nphDAA = frb.Column(nph, 'nph_bg', (frb.PhSel('0ex_1ex1em'), colstart, colstop))
    nphDA = frb.Column(nph, 'nph_bg', (frb.PhSel('0ex1em'), colstart, colstop))
    E = frb.Column(nph, 'ratio_bg', (frb.PhSel('0ex1em'), frb.PhSel('0ex'), colstart, colstop))
    S = frb.Column(nph, 'ratio_bg', (frb.PhSel('0ex'), frb.PhSel('0ex_1ex1em'), colstart, colstop))
    d = data.get_column(nphD)
    daa = data.get_column(nphDAA)
    da = data.get_column(nphDA)
    e = data.get_column(E)
    s = data.get_column(S)
    assert np.allclose(e, da / d, equal_nan=True), "ratio of e calculated incorrectly"
    assert np.allclose(s, d / daa, equal_nan=True), "ratio of s calculated incorrectly"


def test_anisotropy_bg(data, nph, colstart, colstop):
    nphDD = frb.Column(nph, 'nph_bg', (frb.PhSel('0ex0em'), colstart, colstop))
    nphDA = frb.Column(nph, 'nph_bg', (frb.PhSel('0ex1em'), colstart, colstop))
    Ani = frb.Column(nph, 'anisotropy_bg', (frb.PhSel('0ex0em'), frb.PhSel('0ex1em'), colstart, colstop))
    dd = data.get_column(nphDD)
    da = data.get_column(nphDA)
    ani = data.get_column(Ani)
    rani = (dd-da)/(dd+2*da)
    assert np.allclose(ani, rani, equal_nan=True), "incorrect anisotropy calculation"


def test_ESbgr(data, nph, colstart, colstop):
    E = frb.Column(nph, 'E_bg', (colstart, colstop))
    S = frb.Column(nph, 'S_bg', (colstart, colstop))
    assert E == frb.Column(nph, 'ratio_bg', (frb.PhSel('0ex1em'), frb.PhSel('0ex'), colstart, colstop))
    assert S == frb.Column(nph, 'ratio_bg', (frb.PhSel('0ex'), frb.PhSel('0ex_1ex1em'), colstart, colstop))


def test_brightness_bg(data, nph, colstart, colstop):
    dur = frb.Column(nph.parents['base'], 'dur', (colstart, colstop))
    n = frb.Column(nph, 'nph_bg', (frb.PhSel('0ex'), colstart, colstop))
    br = frb.Column(nph, 'brightness_bg', (frb.PhSel('0ex'), colstart, colstop))
    durs, ns, brs = data.get_column(dur), data.get_column(n), data.get_column(br)
    assert np.allclose(brs, ns/durs, equal_nan=True), "Incorrect calculation of brightness"


def test_ratio(nph):
    corr_mat = np.eye(4)
    ratio = frb.Param(frb.Ratios, {'corr_mat':corr_mat}, {'nph':nph})
    assert ratio.base_param == nph.base_param, "base_param incorrectly determined"


@pytest.fixture()
def ratio(nph):
    cmat = np.array([[0.9, 0.0, 0.0, 0.0],
                     [0.1, 1.0, 0.0, 0.1],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 0.9]])
    return frb.Param(frb.Ratios, {'corr_mat':cmat}, {'nph':nph})


def test_nph_c(data, ratio, colstart, colstop):
    nphDD = frb.Column(ratio, 'nph_c', (frb.PhSel('0ex0em'), colstart, colstop))
    nphDA = frb.Column(ratio, 'nph_c', (frb.PhSel('0ex1em'), colstart, colstop))
    nphDD_bg = frb.Column(ratio.parents['nph'], 'nph_bg', (frb.PhSel('0ex0em'), colstart, colstop))
    nphDA_bg = frb.Column(ratio.parents['nph'], 'nph_bg', (frb.PhSel('0ex1em'), colstart, colstop))
    nphAA_bg = frb.Column(ratio.parents['nph'], 'nph_bg', (frb.PhSel('1ex1em'), colstart, colstop))
    dd = data.get_column(nphDD_bg)
    da = data.get_column(nphDA_bg)
    aa = data.get_column(nphAA_bg)
    ddc = data.get_column(nphDD)
    dac = data.get_column(nphDA)
    assert np.allclose(ddc, 0.9*dd, equal_nan=True), 'bad correction calculation on single channel'
    assert np.allclose(dac, 0.1*dd+da+0.1*aa, equal_nan=True), "bad correction of multiple channel"


def test_ratio_c(data, ratio, colstart, colstop):
    nphD = frb.Column(ratio, 'nph_c', (frb.PhSel('0ex'), colstart, colstop))
    nphDAA = frb.Column(ratio, 'nph_c', (frb.PhSel('0ex_1ex1em'), colstart, colstop))
    nphDA = frb.Column(ratio, 'nph_c', (frb.PhSel('0ex1em'), colstart, colstop))
    E = frb.Column(ratio, 'ratio_c', (frb.PhSel('0ex1em'), frb.PhSel('0ex'), colstart, colstop))
    S = frb.Column(ratio, 'ratio_c', (frb.PhSel('0ex'), frb.PhSel('0ex_1ex1em'), colstart, colstop))
    d = data.get_column(nphD)
    daa = data.get_column(nphDAA)
    da = data.get_column(nphDA)
    e = data.get_column(E)
    s = data.get_column(S)
    assert np.allclose(e, da/d, equal_nan=True), "ratio e calculated incorrectly"
    assert np.allclose(s, d/daa, equal_nan=True), "ratio s calculated incorrectly"


def test_anisotropy_c(data, ratio, colstart, colstop):
    nphDD = frb.Column(ratio, 'nph_c', (frb.PhSel('0ex0em'), colstart, colstop))
    nphDA = frb.Column(ratio, 'nph_c', (frb.PhSel('0ex1em'), colstart, colstop))
    Ani = frb.Column(ratio, 'anisotropy_c', (frb.PhSel('0ex0em'), frb.PhSel('0ex1em'), colstart, colstop))
    dd = data.get_column(nphDD)
    da = data.get_column(nphDA)
    ani = data.get_column(Ani)
    assert np.allclose(ani, (dd-da)/(dd+2*da), equal_nan=True), "incorrect anisotropy calculation"


def test_ES(data, nph, colstart, colstop):
    E = frb.Column(nph, 'E_bg', (colstart, colstop))
    S = frb.Column(nph, 'S_bg', (colstart, colstop))
    data.get_column(E)
    data.get_column(S)
    assert E == frb.Column(nph, 'ratio_bg', (frb.PhSel('0ex1em'), frb.PhSel('0ex'), colstart, colstop))
    assert S == frb.Column(nph, 'ratio_bg', (frb.PhSel('0ex'), frb.PhSel('0ex_1ex1em'), colstart, colstop))
