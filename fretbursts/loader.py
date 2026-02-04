#
# FRETBursts - A single-molecule FRET burst analysis toolkit.
#
# Copyright (C) 2014-2016 The Regents of the University of California,
#               Antonino Ingargiola <tritemio@gmail.com>
#
"""
The `loader` module contains functions to load each supported data format.
The loader functions load data from a specific format and
return a new :class:`fretbursts.burstlib.Data()` object containing the data.

This module contains the high-level function to load a data-file and
to return a `Data()` object. The low-level functions that perform the binary
loading and preprocessing can be found in the `dataload` folder.
"""

from .legacy_burstlib import Data

from .read_photonHDF5 import PhotonHDF5Data, normalize_photon_hdf5

def photon_hdf5(filename, ondisk=False, require_setup=True, validate=False, fix_order=True)->Data:
    return Data.new_raw(PhotonHDF5Data.load_hdf5(filename, ondisk=ondisk))


def alex_apply_period(data:Data, alex_type:str=None)->None:
    data._data = normalize_photon_hdf5(data._rawdata).datas
    delattr(data, '_rawdata')
    return data