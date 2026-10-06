#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 09:48:30 2026

@author: paul
"""
import numpy as np


import smfbursts as smf

import pytest


@pytest.fixture(params=[0.0, -1.0, np.pi/2.5e-9, np.array([0.0, 0.0])])
def omega_style(request):
    return request.param


@pytest.fixture(params=[False, True])
def exclude_phasor(request):
    return request.param


@pytest.fixture(params=[smf.PhSel('0ex0em'), smf.PhSel('1ex1em')])
def phsel_test(request):
    return request.param


def test_create_phasor(data, sper_bg, irfstyle, omega_style, exclude_phasor, phsel_test):
    bursts = smf.Param(smf.Bursts, m=10, F=6.0, bg=sper_bg)
    phasor = smf.Param(smf.Phasor, base=bursts, start=irfstyle, omega=omega_style, exclude=exclude_phasor)
    phG = smf.Column(phasor, 'phasor_g', phsel_test)
    phS = smf.Column(phasor, 'phasor_s', phsel_test)
    data.get_column(phG)
    data.get_column(phS)


def test_fit(data, phsel_test):
    out = smf.lt.fit_fldecay(data, phsel_test, ndec=1)
    assert out['taus'].size == 1
    assert out['amps'].size == 0
    
    out = smf.lt.fit_fldecay(data, phsel_test, ndec=2)
    assert out['taus'].size == 2
    assert out['amps'].size == 1
    
def test_fit_init(data, sper_bg):
    bursts = smf.Param(smf.Bursts, m=10, F=6.0, bg=sper_bg)
    nph = smf.Column(bursts, 'nph_raw', smf.PhSel('0ex'))
    gate = smf.make_geq_gate(nph, 60)
    out = smf.lt.fit_fldecay(data, smf.PhSel('0ex0em'), gate, amp_init=np.array([0.7]))
    assert out['taus'].size == 2
    assert out['amps'].size == 1
    out = smf.lt.fit_fldecay(data, smf.PhSel('0ex0em'), gate, tau_init=np.array([0.7e-9]))
    assert out['taus'].size == 1
    assert out['amps'].size == 0
    out = smf.lt.fit_fldecay(data, smf.PhSel('0ex0em'), gate, tau_init=np.array([0.7e-9, 4.8e-9]), amp_init=np.array([0.2]))
    assert out['taus'].size == 2
    assert out['amps'].size == 1
    

def test_fit_irfstyle(data, sper_bg):
    bursts = smf.Param(smf.Bursts, m=10, F=6.0, bg=sper_bg)
    nph = smf.Column(bursts, 'nph_raw', smf.PhSel('0ex'))
    gate = smf.make_geq_gate(nph, 60)
    out = smf.lt.fit_fldecay(data, smf.PhSel('0ex0em'), gate, irf=False)
    assert 'irf_params' in out
    assert 'irf_func' in out
    out = smf.lt.fit_fldecay(data, smf.PhSel('0ex0em'), gate, irf=True)
    assert 'irf_params' not in out
    assert 'irf_func' not in out
    out = smf.lt.fit_fldecay(data, smf.PhSel('0ex0em'), gate, irf=data.irf[smf.PhSel('0ex0em')])
    assert 'irf_params' not in out
    assert 'irf_func' not in out
