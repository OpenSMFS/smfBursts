#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Paul David Harris
# Created: 27/10/2025
"""
Test for background.py params/columns
"""
from itertools import product, chain

import numpy as np

import fretbursts as frb

import pytest


@pytest.mark.dependency(name='period')
def test_periods_timerangecolumns(data):
    prds = frb.Param(frb.Periods, {'period':60.0, 'detdef':data.detdef})
    assert prds.params['period'] == 60.0
    # test for proper BasePhotonTable handling
    start, stop = frb.Column(prds, 'start'), frb.Column(prds, 'stop')
    assert np.all(data.get_column(start)[1:] == data.get_column(stop)[:-1])
    istart, istop = frb.Column(prds, 'start'), frb.Column(prds, 'stop')
    assert np.all(data.get_column(istart) >= data.get_column(start)), "istart before start"
    assert np.all(data.get_column(istop) <= data.get_column(stop)), 'istop after stop'
    

@pytest.fixture()
def p_periods(data)->frb.Param:
    try:
        return frb.Param(frb.Periods, {'period':60.0, 'detdef':data.detdef})
    except:
        return None


@pytest.fixture()
def c_start_p(p_periods)->frb.Column:
    if p_periods is None:
        return None
    try:
        return frb.Column(p_periods, 'start')
    except:
        return None


@pytest.mark.dependency(depends=['period',])
def test_sepcolumn(data, p_periods, c_start_p):
    sep = frb.Column(p_periods, 'sep', ('istarttime', 'istoptime'))
    assert data.get_column(sep).size + 1 == data.get_column(c_start_p).size, "sep column has wrong size"
    assert np.all(data.get_column(sep) >= 0.0), "negative speartion between periods"


@pytest.fixture(params=[frb.bg.exp_mlefit, frb.bg.exp_cdffit, frb.bg.exp_histfit])
def bgfunc(request):
    return request.param


@pytest.fixture(params =[{k:v for k, v in chain(tm.items(), ath.items(), f.items())}
                         for tm, ath, f in product([{'compute_stream':'single', 'tail_min':5e-4}, 
                                                    {'compute_stream':'single', 
                                                     'tail_min':(5e-4, 6e-4, 3e-4, 4e-4)},
                                                    {'compute_stream':'single_all', 
                                                     'tail_min':5e-4},
                                                    {'compute_stream':'single_all', 
                                                     'tail_min':(5e-4, 6e-4, 7e-4, 4e-4, 3e-4)},
                                                    {'compute_stream':'any', 'tail_min':5e-4}],
                                                   [{'auto_threshold':True, 'F_bg':2.0}, 
                                                    {'auto_threshold':False}],
                                                   [{'func':frb.bg.exp_mlefit}, 
                                                    {'func':frb.bg.exp_cdffit}, 
                                                    {'func':frb.bg.exp_histfit}])])
def bgparam(request):
    return request.param


@pytest.mark.dependency(name='backgroundparam', depends=['period',])
def test_make_background(data, p_periods, bgparam):
    """Smoke test for make_background function"""
    bg = frb.Param(frb.BG, bgparam, {'base':p_periods})
    assert bg.parents['base'].params['period'] == p_periods.params['period']
    assert np.all(bg.params['tail_min'] == np.asarray(bgparam['tail_min']))
    assert bg.params['func'] == bgparam['func']


@pytest.fixture()
def p_bg(data, p_periods, bgparam)->frb.Param:
    return frb.Param(frb.BG, bgparam, {'base':p_periods})


@pytest.mark.dependency(depends=['backgroundparam',])
def test_bg_bg_column(data, p_bg):
    bg00 = frb.Column(p_bg, 'bg', frb.PhSel('0ex0em'))
    bg01 = frb.Column(p_bg, 'bg', frb.PhSel('0ex1em'))
    bg0A = frb.Column(p_bg, 'bg', frb.PhSel('0ex'))
    if p_bg.params['compute_stream'] != 'any':
        assert np.allclose(data.get_column(bg00) + data.get_column(bg01), data.get_column(bg0A), equal_nan=True), "bg not compute as a sum"
    else:
        data.get_column(bg00)
        data.get_column(bg01)
        data.get_column(bg0A)
    data.clear_memory()


@pytest.mark.dependency(depends=['backgroundparam'])
def test_KS_column(data, p_bg):
    # smoke tests for error
    bg00err_KS = frb.Column(p_bg, 'err_KS', frb.PhSel('0ex0em'))
    data.get_column(bg00err_KS)
    data.clear_memory()

@pytest.mark.dependency(depends=['backgroundparam'])
def test_CM_column(data, p_bg):
    bg00err_CM = frb.Column(p_bg, 'err_CM', frb.PhSel('0ex0em'))
    data.get_column(bg00err_CM)
    data.clear_memory()


@pytest.mark.dependency(depends=['backgroundparam'])
def test_tail_min(data, p_bg):
    tmin = frb.Column(p_bg, 'tail_min', frb.PhSel('0ex1em'))
    data.get_column(tmin)
    # if p_bg.params['compute_stream'] == 'all':
    #     tminDex = frb.Column(p_bg, 'tail_min', frb.PhSel('0ex'))
    #     data.get_column(tminDex)
    # else:
    #     with pytest.raises(Exception):
    #         tminDex = frb.Column(p_bg, 'tail_min', frb.PhSel('0ex'))