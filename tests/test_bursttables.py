#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 14:38:21 2025

@author: paul
"""
from itertools import product

import numpy as np

import fretbursts as frb

import pytest


@pytest.fixture()
def data()->frb.PhotonData:
    raw = frb.photonHDF5.load('HP3_TE300_SPC630.hdf5')
    return frb.photonHDF5.normalize(raw)


global has_burst
has_burst = False

def test_burst_param(data):
    global has_burst
    bg = frb.bg.make_bg_param(data)
    brst = frb.Param(frb.Bursts, {'channels':frb.Ph_sel('all'), 'm':10, 'F':6.0}, {'bg':bg})
    assert len(brst.params['channels']) == 1
    assert len(brst.parents['bg']) == 1
    assert brst.params['m'].size == 1
    brst = frb.Param(frb.Bursts, {'channels':(frb.Ph_sel('0ex'), frb.Ph_sel('1ex1em')), 'm':10, 'F':6.0}, {'bg':bg})
    has_burst = True
    assert len(brst.params['channels']) == 2
    assert len(brst.parents['bg']) == 2
    assert brst.params['m'].size == 2
    assert brst.params['F'].size == 2
    assert np.all(brst.params['m'] == 10)
    assert np.all(brst.params['F'] == 6.0)
    brst1 = frb.Param(frb.Bursts, {'channels':(frb.Ph_sel('0ex'), frb.Ph_sel('1ex1em')), 
                                   'm':np.array([10,12]), 'F':np.ndarray([6.0, 10.0])},
                      {'bg':(bg, frb.bg.make_bg_param(data, func=frb.bg.exp_cdffit))})
    assert len(brst.params['channels']) == 2
    assert len(brst.parents['bg']) == 2
    assert brst.params['m'].size == 2
    assert brst.params['F'].size == 2
    assert brst.params['truthtable'].sum() == 1
    brst2 = frb.Param(frb.Bursts, {'channels':(frb.Ph_sel('1ex1em'), frb.Ph_sel('0ex')), 
                                   'm':np.array([12,10]), 'F':np.ndarray([10.0, 6.0])},
                      {'bg':(frb.bg.make_bg_param(data, func=frb.bg.exp_cdffit), bg)})
    assert brst1 == brst2


@pytest.fixture()
def burst():
    global has_burst
    if not  has_burst:
        pytest.skip("cannot create burst Param")
    bg = frb.bg.make_bg_param(data)
    return frb.Param(frb.Bursts, {'channels':frb.Ph_sel('all'), 'm':10, 'F':6.0}, {'bg':bg})
    

def test_gt_gate(data, burst):
    nph = frb.Column(burst, 'nph', frb.Ph_sel('0ex_1ex1em'))
    g50 = frb.make_gt_gate(nph, 50)
    nph50 = nph.regate(g50)
    nphcol = data.get_column(nph)
    assert np.all(data.get_column(nph50) == nphcol[nphcol > 50]), 'Improper masking'


def test_bursts_startstops(data, burst):
    cstart = frb.Column(burst, 'start')
    cstop = frb.Column(burst, 'stop')
    cistart = frb.Column(burst, 'istart')
    cistop = frb.Column(burst, 'istop')
    cistarttime = frb.Column(burst, 'istarttime')
    cistoptime = frb.Column(burst, 'istoptime')
    assert np.all(data.get_column(cstart) < data.get_column(cstop)), "start before stop"
    assert np.all(data.get_column(cistart) <= data.get_column(cistop)), "istart before istop"
    assert np.all(data.get_column(cistarttime) <= data.get_column(cistoptime)), "istarttime before istoptime"
    assert np.all(data.get_column(cstart) <= data.get_column(cistarttime)), "istarttime before start"
    assert np.all(data.get_column(cstop) <= data.get_column(cistoptime)), "istoptime before stop"


def test_burst_midtime(data, burst):
    for start, stop in product(('start', 'istarttime'), ('stop', 'istoptime')):    
        mt = frb.Column(burst, 'sep', (start, stop))
        data.get_column(mt)


def test_nph_raw(data, burst):
    nphs = [frb.Column(burst, 'nph_raw', frb.PhSel(f'0ex{i}em')) for i in range(data.detdef.em)]
    nph_d = frb.Column(burst, 'nph_raw', frb.PhSel('0ex'))
    assert np.all(nsum = sum(data.get_column(c) for c in nphs) == data.get_column(nph_d)), "nph_raw computed incorrectly"


def test_burst_ratio_raw(data, burst):
    E = frb.Column(burst, 'ratio_raw', (frb.PhSel('0ex0em'), frb.PhSel('0ex')))
    E_i = frb.Column(burst, 'ratio_raw', (frb.PhSel('0ex'), frb.PhSel('0ex0em')))
    eraw = data.get_column(E)
    erawi = data.get_column(E_i)
    assert np.allclose(eraw, 1/erawi), "ratio_raw not computing inverse"


def test_meanT(data, burst):
    meanT = frb.Column(burst, 'meanT', frb.PhSel('all'))
    assert np.all(np.diff(data.get_column(meanT)) > 0), "non-monotonic mean T"


def test_mTdiff(data, burst):
    mTdiff = frb.Column(burst, 'mTdiff', (frb.PhSel('0ex0em'), frb.PhSel('0ex1em')))
    data.get_column(mTdiff)


def test_burst_brightness(data, burst):
    br = frb.Column(burst, 'brightness')
    data.get_column(br)


def test_burst_dur(data, burst):
    dur = frb.Column(burst, 'dur', ('start', 'stop'))
    mdur = data.get_column(dur)
    for start, stop in product(('start', 'istarttime'), ('stop', 'istoptime')):    
        dur = frb.Column(burst, 'dur', (start, stop))
        assert np.all(mdur >= data.get_column(dur)), "incorrect calculation of duration of burst"


def test_burst_sep(data, busrt):
    for start, stop in product(('start', 'istarttime'), ('stop', 'istoptime')):
        sep = frb.Column(burst, 'sep', (start, stop))
        data.get_column(sep)


def test_max_rate(data, burst):
    mrall = frb.Column(burst, 'max_rate', frb.PhSel('all'))
    mrDD = frb.Column(burst, 'max_rate', frb.PhSel('0ex0em'))
    mall, mdd = data.get_column(mrall) >= data.get_column(mrDD)
    mask = ~np.isnan(mall) + ~np.isnan(mdd)
    mall, mdd = mall[mask], mdd[mask]
    assert np.all(mall >= mdd), "DD stream max rate larger than all"


def test_bva(data, burst):
    data.get_column(frb.Column(burst, 'bva', (frb.PhSel('0ex1em'), frb.PhSel('0ex'), 10)))
    

def test_ebva(data, burst):
    bva = data.get_column(frb.Column(burst, 'bva', (frb.PhSel('0ex1em'), frb.PhSel('0ex'), 10)))
    ebva = data.get_column(frb.Column(burst, 'ebva', (frb.PhSel('0ex1em'), frb.PhSel('0ex'), 10)))
    assert np.all(bva >= ebva), "ebva larger than bva"


def test_nanohist(data, burst):
    nh = frb.Column(burst, 'nanohist', (frb.PhSel('0ex0em'), False))
    nhf = frb.Column(burst, 'nanohist', (frb.PhSel('0ex0em'), True))
    hsize = np.diff(data.setup.ex_ranges[0])[0]
    hfsize = np.max(data.setup.tcspc_num_bins)
    for h, hf in zip(data.iter_column(nh), data.iter_column(nhf)):
        assert hsize == h.size, 'incorrect size of sub-nanohist column'
        assert hfsize < hf.size, 'mismatched size of full nanohist column'
        assert h.sum() == hf.sum(), 'inconsistent histograms between sub and full nanohist'


global has_nph
has_nph = False


def test_make_NphBG(burst):
    global has_nph
    frb.Param(frb.NphBG, {'single':True}, {'base':burst})
    has_nph = True


@pytest.fixture()
def nph(burst):
    global has_nph
    if not has_nph:
        pytest.skip("cannot build NphBG params")
    return frb.Param(frb.NphBG, {'single':True}, {'base':burst})


def test_nph_bg(data, nph):
    nphDD = frb.Column(nph, 'nph_bg', frb.PhSel('0ex0em'))
    nphAA = frb.Column(nph, 'nph_bg', frb.PhSel('1ex1em'))
    nphAll = frb.Column(nph, 'nph_bg', frb.PhSel('all'))
    nphDD_raw = frb.Column(nph.parents['base'], 'nph_raw', frb.PhSel('0ex0em'))
    nphAA_raw = frb.Column(nph.parents['base'], 'nph_raw', frb.PhSel('1ex1em'))
    assert np.all(data.get_column(nphDD) <= data.get_column(nphDD_raw)), "bg adjusted column greater than raw"
    assert np.all(data.get_column(nphAA) <= data.get_column(nphAA_raw)), "bg adjusted column greater than raw"
    data.get_column(nphAll)


def test_ratio_bg(data, nph):
    nphD = frb.Column(nph, 'nph_bg', frb.PhSel('0ex'))
    nphDAA = frb.Column(nph, 'nph_bg', frb.PhSel('0ex_1ex1em'))
    nphAA = frb.Column(nph, 'nph_bg', frb.PhSel('1ex1em'))
    E = frb.Column(nph, 'ratio_bg', (frb.PhSel('0ex0em'), frb.PhSel('0ex')))
    S = frb.Column(nph, 'ratio_bg', (frb.PhSel('0ex'), frb.PhSel('0ex_1ex1em')))
    d = data.get_column(nphD)
    daa = data.get_column(nphDAA)
    aa = data.get_column(nphAA)
    e = data.get_column(E)
    s = data.get_column(S)
    assert np.allclose(e, aa/d), "ratio calculated incorrectly"
    assert np.allclose(s, d/daa), "ratio clculated incorrectly"


def test_anisotropy_bg(data, nph):
    nphDD = frb.Column(nph, 'nph_bg', frb.PhSel('0ex0em'))
    nphDA = frb.Column(nph, 'nph_bg', frb.PhSel('0ex1em'))
    Ani = frb.Column(nph, 'anisotropy_bg', (frb.PhSel('0ex0em'), frb.PhSel('0ex1em')))
    dd = data.get_column(nphDD)
    da = data.get_column(nphDA)
    ani = data.get_column(Ani)
    assert np.allclose(ani, (dd-da)/(dd+2*da)), "incorrect anisotropy calculation"


def test_ES_pr(data, nph):
    E = frb.Column(nph, 'E_pr')
    S = frb.Column(nph, 'S_pr')
    assert E == frb.Column(nph, 'ratio_bg', (frb.PhSel('0ex1em'), frb.PhSel('0ex')))
    assert S == frb.Column(nph, 'ratio_bg', (frb.PhSel('0ex'), frb.PhSel('0ex_1ex1em')))


global has_ratio
has_ratio = False


def test_ratio(nph):
    global has_ratio
    corr_mat = np.eye(4)
    ratio = frb.Param(frb.Ratios, {'corr_mat':corr_mat}, {'nph':nph})
    assert ratio.base_param == nph.base_param, "base_param incorrectly determined"
    has_ratio = True


@pytest.fixture()
def ratio(nph):
    global has_ratio
    if not has_ratio:
        pytest.skip("cannot build ratio")
    cmat = np.array([[0.9, 0.0, 0.0, 0.0],
                     [0.1, 1.0, 0.0, 0.1],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 0.9]])
    return frb.Param(frb.Ratios, {'corr_mat':cmat}, {'nph':nph})


def test_nph_c(data, ratio):
    nphDD = frb.Column(ratio, 'nph_c', frb.PhSel('0ex0em'))
    nphDA = frb.Column(ratio, 'nph_c', frb.PhSel('0ex1em'))
    nphDD_bg = frb.Column(ratio.parents['nph'], 'nph_raw', frb.PhSel('0ex0em'))
    nphDA_bg = frb.Column(ratio.parents['nph'], 'nph_raw', frb.PhSel('0ex1em'))
    nphAA_bg = frb.Column(ratio.parents['nph'], 'nph_raw', frb.PhSel('1ex1em'))
    dd = data.get_column(nphDD_bg)
    da = data.get_column(nphDA_bg)
    aa = data.get_column(nphAA_bg)
    ddc = data.get_column(nphDD)
    dac = data.get_column(nphDA)
    assert np.allclose(ddc, 0.9*dd), 'bad correction calculation on single channel'
    assert np.allclose(dac, 0.1*dd+da+0.1*aa)

def test_ratio_c(data, ratio):
    nphD = frb.Column(nph, 'nph_c', frb.PhSel('0ex'))
    nphDAA = frb.Column(ratio, 'nph_c', frb.PhSel('0ex_1ex1em'))
    nphAA = frb.Column(ratio, 'nph_c', frb.PhSel('1ex1em'))
    E = frb.Column(ratio, 'ratio_c', (frb.PhSel('0ex0em'), frb.PhSel('0ex')))
    S = frb.Column(ratio, 'ratio_c', (frb.PhSel('0ex'), frb.PhSel('0ex_1ex1em')))
    d = data.get_column(nphD)
    daa = data.get_column(nphDAA)
    aa = data.get_column(nphAA)
    e = data.get_column(E)
    s = data.get_column(S)
    assert np.allclose(e, aa/d), "ratio calculated incorrectly"
    assert np.allclose(s, d/daa), "ratio clculated incorrectly"


def test_anisotropy_c(data, ratio):
    nphDD = frb.Column(nph, 'nph_c', frb.PhSel('0ex0em'))
    nphDA = frb.Column(nph, 'nph_c', frb.PhSel('0ex1em'))
    Ani = frb.Column(nph, 'anisotropy_c', (frb.PhSel('0ex0em'), frb.PhSel('0ex1em')))
    dd = data.get_column(nphDD)
    da = data.get_column(nphDA)
    ani = data.get_column(Ani)
    assert np.allclose(ani, (dd-da)/(dd+2*da)), "incorrect anisotropy calculation"


def test_ES(data, nph):
    E = frb.Column(nph, 'E_pr')
    S = frb.Column(nph, 'S_pr')
    data.get_column(E)
    data.get_column(S)
    assert E == frb.Column(nph, 'ratio_bg', (frb.PhSel('0ex1em'), frb.PhSel('0ex')))
    assert S == frb.Column(nph, 'ratio_bg', (frb.PhSel('0ex'), frb.PhSel('0ex_1ex1em')))
