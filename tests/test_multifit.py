#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 22:19:46 2026

@author: paul
"""

import numpy as np

import smfbursts as smf
import smfbursts.datamodel.multifit as smfit


import pytest


@pytest.fixture
def default_bursts_all(default_bg)->smf.Param:
    return smf.Param(smf.Bursts, {'m':10, 'F':6.0}, {'bg':default_bg})


@pytest.fixture
def default_burst_gate(default_bursts_all):
    nph = smf.Column(default_bursts_all, 'nph_raw', smf.PhSel('0ex'))
    gate = smf.make_geq_gate(nph, 50)
    return default_bursts_all.regate(gate)


@pytest.fixture
def default_nphbg(default_burst_gate, default_bg):
    return smf.Param(smf.NphBG, base=default_burst_gate, bg=default_bg)


@pytest.fixture
def ehist(data, default_burst_gate):
    E = smf.Column(default_burst_gate, 'ratio_raw', (smf.PhSel('0ex1em'), smf.PhSel('0ex')))
    bins = np.linspace(0, 1, 21)
    ehst = smfit.Hist.from_columns(data, E, bins=bins)
    return ehst

def test_Hist(data, default_burst_gate):
    E = smf.Column(default_burst_gate, 'ratio_raw', (smf.PhSel('0ex1em'), smf.PhSel('0ex')))
    S = smf.Column(default_burst_gate, 'ratio_raw', (smf.PhSel('0ex'), smf.PhSel('0ex_1ex1em')))
    bins = np.linspace(0, 1, 21)
    ehst = smfit.Hist.from_columns(data, E, bins=bins)
    assert len(ehst.bins) == 1
    assert np.all(ehst.bins[0] == bins)
    assert ehst.hist.sum() == data.get_column(E).size
    eshst = smfit.Hist.from_columns(data, (E, S), bins=(bins,bins))
    assert len(eshst.bins) == 2
    assert eshst.hist.sum() == data.get_column(E).size
    assert eshst.pdf.shape == (20, 20)
    ## future tests should add more tests of pdf/cdf etc


@pytest.fixture
def gaus1_init():
    mu = [0.3, 0.9]
    sigma = [0.1, 0.1]
    amps = [0.7, 0.3]
    return smfit.make_init(mu=mu, sigma=sigma, amps=amps, func=smfit.ngaus_cdf)


@pytest.fixture
def gaus1_bounds():
    mubound = np.array([[0.0, 1.0], [0.0, 1.0]])
    sigmabound = np.array([[1e-4, 0.5],[1e-4, 0.5]])
    return smfit.make_bounds(mu=mubound, sigma=sigmabound, func=smfit.ngaus_cdf)



def test_gauss_fit(ehist, gaus1_init, gaus1_bounds):
    smfit.fit_hist_cdf(ehist, smfit.ngaus_cdf, gaus1_init, bounds=gaus1_bounds, method="Nelder-Mead")


def test_mle(data, default_burst_gate, gaus1_init, gaus1_bounds):
    E = smf.Column(default_burst_gate, 'ratio_raw', (smf.PhSel('0ex1em'), smf.PhSel('0ex')))
    smfit.fit_column_mle(data, E, smfit.ngaus_pdf, gaus1_init, bounds=gaus1_bounds, method='Nelder-Mead')
