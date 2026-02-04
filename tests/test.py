#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 21:08:55 2025

@author: paul
"""

from fretbursts.read_photonHDF5 import PhotonHDF5Data, normalize_photon_hdf5
from fretbursts.Tables import Periods, Param, BG
from fretbursts.ph_sel import Ph_sel

phdata = PhotonHDF5Data.load_hdf5('HP3_TE300_SPC630.hdf5')
print('loaded phdata')
phd = normalize_photon_hdf5(phdata)
print("normalized data")
p30 = Param(Periods, {'period':30.0, 'style':'independent_min'}, dict())
print('made period param')
col = Periods(p30, phd)
print("make col Table")
print(col.periods[0].size - col.start[0].size)
print(col.istart[0][-1])

for m in col.iter_phdata(('nanos',), ph_sel=Ph_sel('0ex')):
    print("next spot")
    for i, (b,) in enumerate(m):
        pass

for m in col.iter_phdata(('nanos',), ph_sel=Ph_sel('1ex')):
    print("next spot")
    for i, (b,) in enumerate(m):
        pass
bg30_50 = Param(BG, {'tail_min':5e-4}, {'base':p30})
# print("make bg param")
col_bg = BG(bg30_50, phd)
print('has col_bg')
print(col_bg.bg[Ph_sel('all')])