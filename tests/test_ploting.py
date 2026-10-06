#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 19:41:49 2026

@author: paul
"""
import numpy as np
import matplotlib.pyplot as plt

import smfbursts as smf

import pytest


def test_scatter(data, default_burst):
    E = smf.Column(default_burst, 'E_raw')
    S = smf.Column(default_burst, 'S_raw')
    gate = smf.make_range_gate(E, 0.3, 0.8)
    smf.plot.scatter(data, E, S)
    plt.close()
    fig, ax = plt.subplots()
    smf.plot.scatter(data, E, S, rescale=(-1, 2), s=1.0, ax=ax, gate=gate)
    plt.close()


def test_density_kde(data, default_burst):
    E = smf.Column(default_burst, 'E_raw')
    S = smf.Column(default_burst, 'S_raw')
    gate = smf.make_range_gate(E, 0.3, 0.8)
    smf.plot.scatter(data, E, S, point_func=smf.plot.density_kde, point_kwargs={'minzero':False}, gate=gate)
    plt.close()


@pytest.fixture(params=[smf.plot.hist_bar, smf.plot.hist_stair, smf.plot.hist_line])
def histfunc(request):
    return request.param


def test_hist(data, default_burst, histfunc):
    E = smf.Column(default_burst, 'E_raw')
    gate = smf.make_range_gate(E, 0.3, 0.8)
    histfunc(data, E, gate=gate)
    plt.close()
    fig, ax = plt.subplots()
    histfunc(data, E, gate=gate, bins=np.linspace(-0.1, 1.1), ax=ax)
    plt.close()


def test_interphoton(data, default_bg):
    smf.plot.hist_interphoton(data)
    plt.close()
    fig, ax = plt.subplots()
    smf.plot.hist_interphoton(data, default_bg, ax=ax, 
                              streams=[smf.PhSel('0ex0em'), smf.PhSel('1ex1em')],
                              streams_kwargs=[{'c':'g'}, {'c':'r'}])
    plt.close()


def test_timeplot(data, default_bg):
    bgDD = smf.Column(default_bg, 'bg', smf.PhSel('0ex0em'))
    fig, ax = plt.subplots()
    smf.plot.time_plot(data, bgDD, ax=ax)