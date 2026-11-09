#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small script for creating files needed for cptester.py to verify no
out of bounds reads etc. with ``valgrind python cptester.py``
Note that memory leaks are difficult with valgrind, as Python has its
own rules.
"""

import smfbursts as smf

data = smf.hdf5.regularize_dets(smf.hdf5.load('data/HP3_TE300_SPC630.hdf5'))
bg = smf.ff.make_bg(data, period=60.0)
bursts = smf.ff.make_burst_search(bg['bg'], streams=smf.PhSel('0ex_1em'))
data.times.tofile('data/times.dat')
data.dets.tofile('data/dets.dat')
data.get_table(bg['periods'])['periods'].tofile('data/periods.dat')
data.get_column(bg['BgAll']).tofile('data/bg.dat')
data.get_column(bg['BgDex']).tofile('data/bgd.dat')
data.get_column(bg['BgAA']).tofile('data/bga.dat')