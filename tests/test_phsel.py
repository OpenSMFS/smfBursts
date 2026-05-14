#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:13:20 2026

@author: paul
"""
from itertools import product, permutations, chain

import numpy as np

from smfbursts.ph_sel import DetDef, ChannelSet, PhStream, PhSel

import pytest


def test_detdef():
    for arg in product(range(1,3), range(1,3), range(1,3), range(1,3)):
        nd = np.array(arg)
        d1 = DetDef(*arg)
        assert np.all(d1.shape == nd)
        kws = {k:v for k, v in zip(('ex', 'em', 'pol', 'split'), arg)}
        d2 = DetDef(**kws)
        assert d1 == d2
        kws_r = {k:v for k, v in zip(('ex', 'em', 'pol', 'split'), arg) if v != 1}
        d3 = DetDef(**kws_r)
        assert d2 == d3
        nmin = len(arg) - 1
        while nmin > 0 and arg[nmin] == 1:
            nmin -= 1
        d4 = DetDef(*arg[:nmin+1])
        assert d3 == d4

@pytest.mark.dependency(name='channelset_pos')
def test_channelset_pos():
    pos0 = ChannelSet(True, {0})
    pos1 = ChannelSet(True, {1})
    pos01 = ChannelSet(True, {0,1})
    cnone = ChannelSet(True, {})
    assert pos0 | pos1 == pos01
    assert pos0 & pos1 == cnone
    assert pos01 - pos1 == pos0
    

@pytest.mark.dependency(name='channelset_neg')
def test_channelset_neg():
    neg0 = ChannelSet(False, {0})
    neg1 = ChannelSet(False, {1})
    neg01 = ChannelSet(False, {0,1})
    call = ChannelSet(False, {})
    assert neg0 | neg1 == call
    assert neg0 & neg1 == neg01
    assert neg0 - neg1 == ChannelSet(True, {1})



@pytest.mark.dependency(depends=['channelset_pos', 'channelset_neg'], name='channelset_logicalor')
def test_channel_logical_or():
    perms = tuple(chain.from_iterable((set(p) for p in permutations(range(4), i)) for i in range(4)))
    for s0, s1 in product(perms, perms):
        c0p, c1p = ChannelSet(True, s0), ChannelSet(True, s1)
        corpp = c0p | c1p
        assert corpp.kind == True, f"Channel set {s0}|{s1} incorrect kind"
        assert corpp.elements == s0 | s1, f"Channel set {s0}|{s1} incorrect elements"
        c0n, c1n = ChannelSet(False, s0), ChannelSet(False, s1)
        cornn = c0n | c1n
        assert cornn.kind == False, f"Channel set ~{s0}|~{s1} incorrect kind"
        assert cornn.elements == s0 & s1, f"Channel set ~{s0}|~{s1} incorrect elements"
        corpn = c0p | c1n
        assert corpn.kind == False, f'Channel set {s0}|~{s1} incorrect kind'
        assert corpn.elements == s1.difference(s0), f'Channel set {s0}|~{s1} incorrect elements'
        cornp = c0n | c1p
        assert cornp.kind == False, f'Channel set {s0}|~{s1} incorrect kind'
        assert cornp.elements == s0.difference(s1), f'Channel set ~{s0}|{s1} incorrect elements'
        

@pytest.mark.dependency(depends=['channelset_pos', 'channelset_neg'], name='channelset_logicalor')
def test_channel_logical_and():
    perms = tuple(chain.from_iterable((set(p) for p in permutations(range(4), i)) for i in range(4)))
    for s0, s1 in product(perms, perms):
        c0p, c1p = ChannelSet(True, s0), ChannelSet(True, s1)
        corpp = c0p & c1p
        assert corpp.kind == True, f"Channel set {s0}&{s1} incorrect kind"
        assert corpp.elements == s0 & s1, f"Channel set {s0}&{s1} incorrect elements"
        c0n, c1n = ChannelSet(False, s0), ChannelSet(False, s1)
        cornn = c0n & c1n
        assert cornn.kind == False, f"Channel set ~{s0}&~{s1} incorrect kind"
        assert cornn.elements == s0 | s1, f"Channel set ~{s0}&~{s1} incorrect elements"
        corpn = c0p & c1n
        assert corpn.kind == True, f'Channel set {s0}&~{s1} incorrect kind'
        assert corpn.elements == s0.difference(s1), f'Channel set {s0}&~{s1} incorrect elements'
        cornp = c0n & c1p
        assert cornp.kind == True, f'Channel set {s0}&~{s1} incorrect kind'
        assert cornp.elements == s1.difference(s0), f'Channel set ~{s0}&{s1} incorrect elements'


@pytest.mark.dependency(depends=['channelset_pos', 'channelset_neg'], name='phstream')
def test_phstream():
    call = ChannelSet(False, {})
    pos0 = ChannelSet(True, {0})
    neg0 = ChannelSet(False, {0})
    for s in ('ex', 'em', 'pol', 'split'):
        spos = PhStream(**{s:pos0})
        sneg = PhStream(**{s:neg0})
        for attr in ('ex', 'em', 'pol', 'split'):
            assert getattr(spos, attr) == (pos0 if s == attr else call)
            assert getattr(sneg, attr) == (neg0 if s == attr else call)


@pytest.mark.dependency(depends=['channelset_pos', 'channelset_neg'], name='phstream_and')
def test_phstream_and():
    # human readable combinations
    assert PhSel('0ex') & PhSel('0em') == PhSel('0ex0em')
    assert PhStream(ex=ChannelSet(True, {0, 1})) & PhStream(ex=ChannelSet(True, {1, 2}))
    csets = tuple(ChannelSet(bool(kind), p) for kind, p in product(range(2), ({0},{1},{2},{0,1},{})))
    


@pytest.mark.dependency(depends=['channelset_pos', 'channelset_neg'], name='phstream_or')
def test_phstream_or():
    assert PhSel('0ex') | PhSel('0em') == PhSel('0ex_0em')


@pytest.mark.dependency(depends=['channelset',])
def test_phsel_allnone():
    pall = PhSel('all')
    assert len(pall.streams) == 1 and list(pall.streams)[0] == ChannelSet(False, {})
    pnone = PhSel('none')
    assert len(pnone.streams) == 1 and list(pnone.streams)[0] == ChannelSet(True, {})


