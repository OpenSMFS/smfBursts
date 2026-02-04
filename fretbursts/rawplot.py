#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module contains functions for plotting data from raw objects
"""
from typing import Union
import matplotlib.pyplot as plt
import phconvert.plotter as plotter

from .photonHDF5 import PhotonHDF5Data


def alternation_hist(raw:PhotonHDF5Data, ich:int=0, ax:plt.Axes=None, group_dets:Union[None,bool]=None,
                     **kwargs):
    if group_dets is None:
        group_dets = any(plotter._ch_rgx.fullmatch(k) for k 
                         in raw.photon_data[ich].meas_specs['detectors_specs'].keys())
    plotter.alternation_hist(raw.as_photonHDF5_dict, ich=ich, ax=ax, group_dets=group_dets,**kwargs)