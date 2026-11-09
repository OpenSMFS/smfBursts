#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:39:27 2026

@author: paul
"""
from itertools import product

import numpy as np

import smfbursts as smf

import pytest


def test_make_NphBG(burstAll):
    smf.Param(smf.NphBG, {'single':True}, {'base':burstAll, 'bg':burstAll.parents['bg']})


@pytest.fixture()
def nph(burstAll):
    return smf.Param(smf.NphBG, {'single':True}, {'base':burstAll, 'bg':burstAll.parents['bg']})


def test_nph_bg(data, nph, colstart, colstop):
    nphDD = smf.Column(nph, 'nph_bg', (smf.PhSel('0ex0em'), colstart, colstop))
    nphAA = smf.Column(nph, 'nph_bg', (smf.PhSel('1ex1em'), colstart, colstop))
    nphAll = smf.Column(nph, 'nph_bg', (smf.PhSel('all'), colstart, colstop))
    nphDD_raw = smf.Column(nph.parents['base'], 'nph_raw', smf.PhSel('0ex0em'))
    nphAA_raw = smf.Column(nph.parents['base'], 'nph_raw', smf.PhSel('1ex1em'))
    assert np.all(data.get_column(nphDD) <= data.get_column(nphDD_raw)), "bg adjusted column greater than raw"
    assert np.all(data.get_column(nphAA) <= data.get_column(nphAA_raw)), "bg adjusted column greater than raw"
    data.get_column(nphAll)


def test_ratio_bg(data, nph, colstart, colstop):
    nphD = smf.Column(nph, 'nph_bg', (smf.PhSel('0ex'), colstart, colstop))
    nphDAA = smf.Column(nph, 'nph_bg', (smf.PhSel('0ex_1ex1em'), colstart, colstop))
    nphDA = smf.Column(nph, 'nph_bg', (smf.PhSel('0ex1em'), colstart, colstop))
    E = smf.Column(nph, 'ratio_bg', (smf.PhSel('0ex1em'), smf.PhSel('0ex'), colstart, colstop))
    S = smf.Column(nph, 'ratio_bg', (smf.PhSel('0ex'), smf.PhSel('0ex_1ex1em'), colstart, colstop))
    d = data.get_column(nphD)
    daa = data.get_column(nphDAA)
    da = data.get_column(nphDA)
    e = data.get_column(E)
    s = data.get_column(S)
    assert np.allclose(e, da / d, equal_nan=True), "ratio of e calculated incorrectly"
    assert np.allclose(s, d / daa, equal_nan=True), "ratio of s calculated incorrectly"


def test_bad_anisotropy_bg(data, nph, colstart, colstop):
    nphDD = smf.Column(nph, 'nph_bg', (smf.PhSel('0ex0em'), colstart, colstop))
    nphDA = smf.Column(nph, 'nph_bg', (smf.PhSel('0ex1em'), colstart, colstop))
    with pytest.warns():
        Ani = smf.Column(nph, 'anisotropy_bg', (smf.PhSel('0ex0em'), smf.PhSel('0ex1em'), colstart, colstop))
    dd = data.get_column(nphDD)
    da = data.get_column(nphDA)
    with pytest.warns():
        ani = data.get_column(Ani)
    rani = (dd-da)/(dd+2*da)
    assert np.allclose(ani, rani, equal_nan=True), "incorrect anisotropy calculation"


def test_ESbgr(data, nph, colstart, colstop):
    E = smf.Column(nph, 'E_bg', (colstart, colstop))
    S = smf.Column(nph, 'S_bg', (colstart, colstop))
    assert E == smf.Column(nph, 'ratio_bg', (smf.PhSel('0ex1em'), smf.PhSel('0ex'), colstart, colstop))
    assert S == smf.Column(nph, 'ratio_bg', (smf.PhSel('0ex'), smf.PhSel('0ex_1ex1em'), colstart, colstop))


def test_brightness_bg(data, nph, colstart, colstop):
    dur = smf.Column(nph.parents['base'], 'dur', (colstart, colstop))
    n = smf.Column(nph, 'nph_bg', (smf.PhSel('0ex'), colstart, colstop))
    br = smf.Column(nph, 'brightness_bg', (smf.PhSel('0ex'), colstart, colstop))
    durs, ns, brs = data.get_column(dur), data.get_column(n), data.get_column(br)
    assert np.allclose(brs, ns/durs, equal_nan=True), "Incorrect calculation of brightness"


def test_nanomean_bg(data, nph, irfstyle, colstart, colstop):
    phsel = data.detdef.stream_ids_to_PhSel(0)
    nm = smf.Column(nph, 'nanomean_bg', (phsel, irfstyle, colstart, colstop))
    data.get_column(nm)


def test_ratio(nph):
    corr_mat = np.eye(4)
    ratio = smf.Param(smf.Ratios, {'corr_mat':corr_mat}, {'nph':nph})
    assert ratio.base_param == nph.base_param, "base_param incorrectly determined"


@pytest.fixture()
def ratio(nph):
    cmat = np.array([[0.9, 0.0, 0.0, 0.0],
                     [0.1, 1.0, 0.0, 0.1],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 0.9]])
    return smf.Param(smf.Ratios, {'corr_mat':cmat}, {'nph':nph})


def test_nph_c(data, ratio, colstart, colstop):
    nphDD = smf.Column(ratio, 'nph_c', (smf.PhSel('0ex0em'), colstart, colstop))
    nphDA = smf.Column(ratio, 'nph_c', (smf.PhSel('0ex1em'), colstart, colstop))
    nphDD_bg = smf.Column(ratio.parents['nph'], 'nph_bg', (smf.PhSel('0ex0em'), colstart, colstop))
    nphDA_bg = smf.Column(ratio.parents['nph'], 'nph_bg', (smf.PhSel('0ex1em'), colstart, colstop))
    nphAA_bg = smf.Column(ratio.parents['nph'], 'nph_bg', (smf.PhSel('1ex1em'), colstart, colstop))
    dd = data.get_column(nphDD_bg)
    da = data.get_column(nphDA_bg)
    aa = data.get_column(nphAA_bg)
    ddc = data.get_column(nphDD)
    dac = data.get_column(nphDA)
    assert np.allclose(ddc, 0.9*dd, equal_nan=True), 'bad correction calculation on single channel'
    assert np.allclose(dac, 0.1*dd+da+0.1*aa, equal_nan=True), "bad correction of multiple channel"


def test_ratio_c(data, ratio, colstart, colstop):
    nphD = smf.Column(ratio, 'nph_c', (smf.PhSel('0ex'), colstart, colstop))
    nphDAA = smf.Column(ratio, 'nph_c', (smf.PhSel('0ex_1ex1em'), colstart, colstop))
    nphDA = smf.Column(ratio, 'nph_c', (smf.PhSel('0ex1em'), colstart, colstop))
    E = smf.Column(ratio, 'ratio_c', (smf.PhSel('0ex1em'), smf.PhSel('0ex'), colstart, colstop))
    S = smf.Column(ratio, 'ratio_c', (smf.PhSel('0ex'), smf.PhSel('0ex_1ex1em'), colstart, colstop))
    d = data.get_column(nphD)
    daa = data.get_column(nphDAA)
    da = data.get_column(nphDA)
    e = data.get_column(E)
    s = data.get_column(S)
    assert np.allclose(e, da/d, equal_nan=True), "ratio e calculated incorrectly"
    assert np.allclose(s, d/daa, equal_nan=True), "ratio s calculated incorrectly"


def test_bad_anisotropy_c(data, ratio, colstart, colstop):
    nphDD = smf.Column(ratio, 'nph_c', (smf.PhSel('0ex0em'), colstart, colstop))
    nphDA = smf.Column(ratio, 'nph_c', (smf.PhSel('0ex1em'), colstart, colstop))
    dd = data.get_column(nphDD)
    da = data.get_column(nphDA)
    with pytest.warns():
        Ani = smf.Column(ratio, 'anisotropy_c', (smf.PhSel('0ex0em'), smf.PhSel('0ex1em'), colstart, colstop))
    with pytest.warns():
        ani = data.get_column(Ani)
    assert np.allclose(ani, (dd-da)/(dd+2*da), equal_nan=True), "incorrect anisotropy calculation"


def test_ES(data, nph, colstart, colstop):
    E = smf.Column(nph, 'E_bg', (colstart, colstop))
    S = smf.Column(nph, 'S_bg', (colstart, colstop))
    data.get_column(E)
    data.get_column(S)
    assert E == smf.Column(nph, 'ratio_bg', (smf.PhSel('0ex1em'), smf.PhSel('0ex'), colstart, colstop))
    assert S == smf.Column(nph, 'ratio_bg', (smf.PhSel('0ex'), smf.PhSel('0ex_1ex1em'), colstart, colstop))


def test_anisotropy(datapolgroup:smf.PhotonDataList):
    bg = smf.ff.make_bg(datapolgroup, auto_threshold=True, F_bg=1.7)
    bursts = smf.ff.make_burst_search(bg['bg'], m=10, F=6.0, streams=smf.PhSel('0ex_1em'))
    ani_r = smf.Column(bursts['bursts'], 'anisotropy_raw', (smf.PhSel('0ex0em0pol'), smf.PhSel('0ex0em1pol')))
    datapolgroup.get_column(ani_r)
    ani_b = smf.Column(bursts['nphbg'], 'anisotropy_bg', (smf.PhSel('0ex0em0pol'), smf.PhSel('0ex0em1pol')))
    an_b = datapolgroup.get_column(ani_b)
    assert isinstance(an_b, tuple)
    rat = smf.Param(smf.Ratios, corr_mat=2.0*np.eye(8), nph=bursts['nphbg'])
    ani_c = smf.Column(rat, 'anisotropy_c', (smf.PhSel('0ex0em0pol'), smf.PhSel('0ex0em1pol')))
    an_c = datapolgroup.get_column(ani_c)
    assert all(np.allclose(b, c, equal_nan=True) for b, c in zip(an_b, an_c))


def test_rotational_corr_c(datapolgroup, irfstyle):
    bg = smf.ff.make_bg(datapolgroup, auto_threshold=True, F_bg=1.7)
    bursts = smf.ff.make_burst_search(bg['bg'], m=10, F=6.0, streams=smf.PhSel('0ex_1em'))
    rat = smf.Param(smf.Ratios, corr_mat=2.0*np.eye(8), nph=bursts['nphbg'])
    rcor = smf.Column(rat, 'rotational_corr_c', (smf.PhSel('0ex0em0pol'), smf.PhSel('0ex0em1pol'), irfstyle))
    datapolgroup.concatenate_column(rcor)


def test_single_ex(data1ex):
    bg = smf.ff.make_bg(data1ex, auto_threshold=True)
    bursts = smf.ff.make_burst_search(bg['bg'], m=10, F=6.0, lifetime=False)
    gate = smf.make_geq_gate(bursts['NphDD_bg'], 50)
    smf.ff.make_correction_factors(bursts['nphbg'], update=bursts, alpha=0.1, gamma=0.95)
    data1ex.get_frame(*smf.ff.get_columns(bursts, gate))


def test_make_KDE(burstAll):
    smf.Param(smf.KDE, parents={'base':burstAll}, params={'kernel':smf.cpt.laplace_kde, 'tau':5e-4, "thresh":25e-4})


@smf.datamodel.utils.fjit('f8(i8,f8)')
def my_kernel(delta, tau):
    return tau / (np.abs(delta) + 1)
my_kernel.name = r'\mathcal{Y}'
my_kernel.thresh_func = lambda param: 5.0*param.get('tau')

smf.cpt.register_kde_func(my_kernel, smf.datamodel.ParamDef('tau', smf.datamodel.TV_float(mn=0.0)))

@pytest.fixture(params=[{'kernel':func[0], func[1]:5e-5, 'all_phot':all_phot, 'exclude_zero':ex_zero} 
                        for func, all_phot, ex_zero in 
                        product(((smf.cpt.laplace_kde, 'tau'), (smf.cpt.gaussian_kde, 'sigma'), (my_kernel, 'tau')), (False, True), (False, True))])
def kde(burstAll, request):
    return smf.Param(smf.KDE, base=burstAll, **request.param)


def test_ph_kde(data, kde):
    phkde = smf.Column(kde, 'ph_kde', (smf.PhSel('0ex0em'), smf.PhSel('0ex1em'), False))
    nph = smf.Column(kde.base_param, 'nph_raw', smf.PhSel('0ex1em'))
    for pkde, n in zip(data.get_column(phkde), data.get_column(nph)):
        assert pkde.size == n

def test_tomovE(data, kde):
    tomovE = smf.Column(kde, 'tomovE', (smf.PhSel('0ex1em'), smf.PhSel('0ex0em')))
    data.get_column(tomovE)


def test_tomovBR(data, kde):
    tomovE = smf.Column(kde, 'tomovBR', (smf.PhSel('0ex'), smf.PhSel('1ex1em')))
    data.get_column(tomovE)


def test_fret2cde(data, kde):
    f2cde = smf.Column(kde, 'fret2cde', (smf.PhSel('0ex1em'), smf.PhSel('0ex1em')))
    data.get_column(f2cde)


def test_alex2cde(data, kde):
    f2cde = smf.Column(kde, 'alex2cde', (smf.PhSel('0ex'), smf.PhSel('1ex1em')))
    data.get_column(f2cde)
    