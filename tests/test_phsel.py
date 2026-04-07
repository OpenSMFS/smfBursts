#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:13:20 2026

@author: paul
"""
from itertools import product

import numpy as np

from fretbursts.ph_sel import DetDef, ChannelSet, PhStream, PhSel

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


@pytest.mark.dependency(depends=['channelset_pos', 'channelset_neg'])
def test_phstream_pos():
    call = ChannelSet(False, {})
    pos0 = ChannelSet(True, {0})
    neg0 = ChannelSet(False, {0})
    for s in ('ex', 'em', 'pol', 'split'):
        spos = PhStream(**{s:pos0})
        sneg = PhStream(**{s:neg0})
        for attr in ('ex', 'em', 'pol', 'split'):
            assert getattr(spos, attr) == (pos0 if s == attr else call)
            assert getattr(sneg, attr) == (neg0 if s == attr else call)