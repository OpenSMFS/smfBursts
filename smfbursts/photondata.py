#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 3/12/2024
# author: Paul David Harris
# email: harripd@gmail.com
"""
The ``smfbursts.photondata`` module defines the main handling of photon data 
for smfBursts.
This incldudes the primary class definitions for the handling of 
stream sorted photon data, 
and the various attributes and tables that can be defined thereof.

.. |TypeValidator| replace:: :class:`TypeValidator <smfbursts.datamodel.immutabledata.TypeValidator>`
.. |DiskDict| replace:: :class:`smfbursts.datamodel.diskdict.DiskDict`
.. |DataSet| replace:: :class:`smfbursts.datamodel.tables.DataSet`
"""
from typing import Union, Any, ClassVar, Literal
from collections.abc import Sequence, Iterator, Hashable, Callable
import weakref
from numbers import Real
from os import PathLike
import warnings
import hashlib

import numpy as np
import tables as tb

from .datamodel.utils import (
    tupledict, ImDict, arr_slc, union_multi, enumerate_intersects, get_unit_prefix
    )
from .datamodel.immutabledata import (
    _ImData, TV_int, TV_float, TV_ndarray, TV_ImData, TV_str, TypeValidator
    )
from .ph_sel import DetDef, PhSel, TV_DetDef, ChannelSet, _csall, phsel_all
from .datamodel.diskdict import AttrDD, TypedValueDD, SubDiskDict
from .datamodel.tables import (
    TableLike, BaseTable, ChildTable, DataSet, DataSetList, TableConstructionError,
    paramproperty, Param, ColumnDef, Column, Gate, GateGroup, GroupFuture
    )
from .cite import cite

import smfbursts.cfuncs as smc

_alloc_size:int = 512


def _proc_detdef(imdata:"PhSpec", kwarg_append:dict)->dict:
    """data_proc function for creating detdef in PhSpec"""
    ex = imdata.ex_ranges.shape[0] if 'ex_ranges' in imdata else 1
    em = imdata.em_wv.shape[0] if 'em_wv' in imdata else None
    pol = imdata.pol_anlge.shape[0] if 'pol_anlge' in imdata else None
    split = imdata.split_ratio.shape[0] if 'split_ratio' in imdata else None
    return dict(ex=ex, em=em, pol=pol, split=split)


class PhSpec(_ImData):
    r"""
    Settings for **processed** data on photon arrival times.
    This class acts as an immutable dictionary, all parameters can be accesed
    as either keys or attributes.
    
    Parameters
    ----------
    clk_p : float
        time (in seconds) of unit of times array.
    tcspc_unit : np.ndarray[np.float64]
        array of units (in seconds) of nanotimes array, 1 per detector.
    tcspc_num_bins : np.ndarray[np.uint16]
        array of number of bins in TCSPC window, 1 per detector.
    tcspc_range : float
        duration (in seconds) of TCSPC window. Shold be tcspc_unit*tcspc_num_bins
    alex_type : str
        On which time scale lasers are alternated. Must be one of
        'macro', 'nano', 'none'. 'none' only for single cw excitation.
    ex_ranges : np.ndarray[np.ndarray[np.int64]]
        ranges (in macrotime if alex_type='macro', or nanotime if alex_type='nano')
        for each excitation spot. Array of nx2 arrays (n usually = 1). nx2 arrays
        indicate [start, stop) times of excitation for given laser. If stop > start
        the window is treated as [start, max_window) + [start_window, stop).
    alex_period : int
        **alex_type = 'macro' only.** Duration of complete excitation period in units
        of clk_p.
    alex_offset : int
        **alex_type = 'macro' only.** Offset in units of clk_p to be applied to
        timestamps to compute alex period. Alex period computed as
        :math:`(timestamp - alex_offset) \mod alex_period`.
    alternated : np.ndarray[np.bool\_]
        If given laser excitation is cw with alternation.
    pulsed : np.ndarray[np.bool\_]
        If a given laser excitation is from a pulsed (ps) laser source
    ex_wv : np.ndarray[np.float64]
        wavelength of each excitation laser (in nm).
    ex_pol : np.ndarray[np.float64]
        Polarization of each excitation laser (in degrees).
    ex_pow : np.ndarray[np.float64]
        Excitation power (in W) of each laser at focal point.
    ex_intensities : np.ndarray[np.float64]
        Excitation intensity (in W*m\ :sup:`-2`) of each laser at focal point.
    em_wv_centers : np.ndarray[np.float64]
        width of emmission filter for each emission index (physical detector).
    em_wv_widths : np.ndarray[np.float64]
        width of emmission filter for each emission index (physical detector).
    pol_angle : np.ndarray[np.float64]
        Angle of polarization (in degrees) for each split index.
    split_ratio : np.ndarray[np.float64]
        ratio [0, 1.0] of split to each index.
    detdef : DetDef
        DetDef object defining nummber of ex, em, pol and split channels.
    
    """
    __slots__ = ('clk_p', 'tcspc_unit', 'tcspc_num_bins', 'tcspc_range', 'alex_type', 
                 'ex_ranges', 'alex_period', 'alex_offset', 'alternated', 'pulsed',
                 'ex_wv', 'ex_pol', 'ex_pow', 'ex_intensities',
                 'em_wv_centers', 'em_wv_widths', 'pol_angle', 'split_ratio',
                 'detdef')
    _match_size = ImDict(ex=('ex_ranges', 'ex_wv', 'ex_pol', 'ex_pow', 'ex_intensities'),
                         em=('em_wv_centers', 'em_wv_widths'), pol=('pol_angle',),
                         split=('split_ratio',))
    _typeconversions = ImDict(clk_p=TV_float(mn=0.0),
                              tcspc_unit=TV_ndarray(dims=arr_slc[:], mn=0.0, 
                                                    dtype=np.float64, superdtype=np.number),
                              tcspc_num_bins=TV_ndarray(dims=arr_slc[:], mn=1, 
                                                        dtype=np.uint16, superdtype=np.integer),
                              tcspc_range=TV_float, alex_type=TV_str(isin=('macro', 'nano', 'none')),
                              ex_ranges=TV_ndarray(dims=arr_slc[:], dtype=np.object_, 
                                                   typedefs=TV_ndarray(dtype=np.int64, 
                                                                       superdtype=np.number, 
                                                                       dims=arr_slc[:,2])),
                              alex_period=TV_int(mn=0), alex_offset=TV_int,
                              alternated=TV_ndarray(dims=arr_slc[:], dtype=np.bool_),
                              pulsed=TV_ndarray(dims=arr_slc[:], dtype=np.bool_),
                              ex_wv=TV_ndarray(dtype=np.float64, superdtype=np.number, 
                                               dims=arr_slc[:]),
                              ex_pol=TV_ndarray(dims=arr_slc[:], dtype=np.float64),
                              ex_pow=TV_ndarray(dims=arr_slc[:], dtype=np.float64),
                              ex_intensities=TV_ndarray(dims=arr_slc[:], dtype=np.float64),
                              em_wv_centers=TV_ndarray(dtype=np.float64, 
                                                       superdtype=np.number, dims=arr_slc[:]),
                              em_wv_widths=TV_ndarray(dtype=np.float64, 
                                                      superdtype=np.number, dims=arr_slc[:]),
                              pol_angle=TV_ndarray(dtype=np.float64, superdtype=np.number, 
                                                   dims=arr_slc[:]),
                              split_ratio=TV_ndarray(dtype=np.float64, superdtype=np.number, 
                                                     dims=arr_slc[:]), 
                              detdef=TV_DetDef(data_proc=_proc_detdef))
    _defaults = ImDict(detdef=None, alex_type='none')
    
    alex_type : Literal['none','macro','nano']
    clk_p: float
    tcspc_unit: np.ndarray[np.float64]
    tcspc_num_bins: np.ndarray[np.int64]
    tcspc_range: float
    ex_ranges: np.ndarray[np.ndarray[np.int64]]
    alex_period: int
    alex_offset: int
    alternated: np.ndarray[np.bool_]
    pulsed: np.ndarray[np.bool_]
    em_wv_centers: np.ndarray[np.float64]
    em_wv_widths: np.ndarray[np.float64]
    pol_angle: np.ndarray[np.float64]
    split_ratio: np.ndarray[np.float64]
    detdef: DetDef

    def __post_init__(self):
        for stattr, attrs in self._match_size.items():
            for attr in attrs:
                if attr in self and getattr(self, attr).size != getattr(self.detdef, stattr):
                    raise ValueError(f'incorrect number of {attr} ({getattr(self, attr)}), defined {getattr(self.detdef, stattr)}')
        if not hasattr(self, 'tcspc_unit') and (hasattr(self, 'tcspc_num_bins') + hasattr(self, 'tcspc_range')) == 1:
            raise ValueError('Cannot compute tcspc_unit')
        if not hasattr(self, 'tcspc_unit') and hasattr(self, 'tcspc_num_bins') and hasattr(self, 'tcspc_range'):
            super(_ImData, self).__setattr__("tcspc_unit", self.tcspc_range[:,np.newaxis] / self.tcspc_num_bins)
        if self.alex_type == 'none' and hasattr(self, 'ex_ranges'):
            if hasattr(self, 'tcspc_unit'):
                super(_ImData, self).__setattr__('alex_type', 'nano')
            else:
                super(_ImData, self).__setattr__('alex_type', 'macro')
        if self.alex_type == 'macro':
            if not hasattr(self, 'alex_period'):
                raise ValueError("macrotime based ALEX measurements must specify alex_period")
            if not hasattr(self, 'alex_offset'):
                super(_ImData, self).__setattr__('alex_offset', 0)
        elif self.alex_type == 'nano' and (hasattr(self, 'alex_period') or hasattr(self, 'alex_offset')):
            raise ValueError("nanotime base ALEX measurements cannot specify alex_period or alex_offset")
        # check and convert valid ex_ranges
        if 'alternated' in self and self.detdef.ex != self.alternated.size:
            raise ValueError("alternated must be same size as number of excitations")
        if 'alternated' not in self:
            super(_ImData, self).__setattr__('alternated', np.ones(self.detdef.ex, dtype=np.bool_)*(self.alex_type == 'macro'))
        if 'pulsed' in self and self.detdef.ex != self.alternated.size:
            raise ValueError("pulsed must be same size as number of excitations")
        if 'pulsed' not in self:
            super(_ImData, self).__setattr__('pulsed', (~self.alternated)*(self.alex_type=='nano'))
        if np.any(self.pulsed & self.alternated):
            raise ValueError("excitations cannot be both pulsed and alternated")


def _proc_size(imdata:"PhArray", kwarg_append:dict)->dict:
    """Pre-process for typevalidator of pharray of PhArray, ensure correct size of arrays"""
    for k in imdata.keys():
        if k == 'setup':
            continue
        return dict(dims=imdata._get_prop(k, 'shape'))
    return dict()


TV_pharray_mtch = TV_ndarray(dims=arr_slc[:], data_proc=_proc_size)


class PhArray(AttrDD, TypedValueDD):
    """
    |DiskDict| of data from single photon counting measurement.
    All keys can be accessed as attrs.
    
    Parameters
    ----------
    setup : Ph_sec
        :class:`PhSpec` of settings informing about timestamps/tcspc_unit units, and
        excitation ranges etc.
    times : np.ndarray[np.int64]
        photon arrival times (macrotimes) of photons
    dets : np.ndarray[np.uint8]
        (sorted) photon indexes, matching a correct size detdef
    nanos : np.ndarray[np.uint16]
        nanotimes of photons, only in data with TCSPC
    particles : np.ndarray[np.uint8]
        **Simulated data only** particle index of photon.
    
    """
    #: :class:PhSpec` of settings used when acquiring data
    setup: PhSpec
    #: Array of photon arrival times (macrotime)
    times: np.ndarray[np.int64]
    #: Array of photon detector indexes (identified according to DetDef in setup)
    dets: np.ndarray[np.uint8]
    #: Array of photon nanotimes (microtime), pulsed excitation only
    nanos: np.ndarray[np.uint16]
    #: Array of photon particle indexes, simulated data only
    particles: np.ndarray[np.uint8]
    
    _attrs = frozenset({'setup', 'times', 'dets', 'nanos', 'particles'})
    _typemap = ImDict(setup=TV_ImData(sublcass=PhSpec),
                      times=TV_pharray_mtch(dtype=np.int64), dets=TV_pharray_mtch(dtype=np.uint8),
                      nanos=TV_pharray_mtch(dtype=np.uint16), particles=TV_pharray_mtch(np.uint8))
    
    @classmethod
    def _valtype(cls, key):
        """Get expected dtype of key"""
        return cls._typemap[key]
    
    def __getattr__(self, attr):
        if attr not in self._attrs or attr not in self:
            raise AttributeError(f"PhArray has no attribute {attr}")
        return self[attr]
    
    @property
    def detdef(self)->DetDef:
        """:class:`smfbursts.ph_sel.DetDef` defining :attr:`PhArray.dets` indexes"""
        return self['setup'].detdef
        

# Functions for converting raw data dets to processed
# NOTE: these are split into many functions so that numba can be optimized
# Build masks for the alternating periods
def _select_outer_range(times:np.ndarray[np.int64], edges:np.ndarray[np.int64])->np.ndarray[np.bool_]:
    """Mask for times in given range [start, stop)"""
    return (times >= edges[0]) + (times < edges[1])


def _select_inner_range(times:np.ndarray[np.int64], edges:np.ndarray[np.int64])->np.ndarray[np.bool_]:
    """Mask for times in given range [start, ...) + [0, stop)"""
    return (times >= edges[0]) * (times < edges[1])


def _select_range(times:np.ndarray[np.int64], edges:np.ndarray[np.int64])->np.ndarray[np.bool_]:
    """Mask for times in range of start, stop, detecting if stop > start, and treating as inner range"""
    return _select_inner_range(times, edges) if edges[0] < edges[1] else _select_outer_range(times, edges)

# function shifts nanotimes
def _apply_offset(nanos:np.ndarray[np.uint16], dets:np.ndarray[np.uint8], 
                  ids:np.ndarray[np.uint8], offsets:np.ndarray[np.uint16])->np.ndarray[np.uint16]:
    """Apply tcspc_offset to nanos"""
    for id_, offset in zip(ids, offsets):
        nanos[dets == id_] -= offset
    return nanos


def _sort_ex(times:np.ndarray[np.integer], ex_rngs:np.ndarray[np.integer])->tuple[np.ndarray[np.uint8], np.ndarray[np.bool_]]:
    """Sort excitation based on ranges, times may be either times array (if usALEX)
    or nanotimes (if nsALEX)"""
    if (nmissing := sum(ex is None for ex in ex_rngs)) == 1:
        for missing, ex in enumerate(ex_rngs):
            if ex is None:
                break
    elif nmissing > 1:
        raise ValueError("can only have 1 missing excitation range")
    out = np.ones(times.size, dtype=np.uint8)*0xff if nmissing == 0 else np.ones(times.size, dtype=np.uint8)*missing
    mask, mtemp = np.zeros(times.size, dtype=np.bool_), np.zeros(times.size, dtype=np.bool_)
    for i, ex_rng in enumerate(ex_rngs):
        if ex_rng is None:
            continue
        mtemp[:] = False
        for ex_r in ex_rng.reshape(-1,2):
            mtemp += _select_range(times, ex_r)
        out[mtemp] = i
        mask += mtemp
    return out, mask


def _nonmonotonic(times:np.ndarray[np.integer])->bool:
    """
    Test if array is *not* nomonotonically increasing, 
    True if non-monotonic so error can be raised
    """
    return np.any(times[:-1] > times[1:])


def _single_object_array(array:np.ndarray)->np.ndarray[np.object_]:
    """Place input into a (1,) shaped numpy object array"""
    out = np.empty(1, dtype=np.object_)
    out[0] = array
    return out


def _one_none_match(n:int, ids:np.ndarray[np.ndarray[np.uint8]], 
                    valid:np.ndarray[np.uint8], spec:str)->np.ndarray[np.ndarray[np.uint8]]:
    """Test if given channel is correct size, and ensure is object array."""
    if ids is None or len(ids) == 0:
        ids = _single_object_array(valid)
    if len(ids) != n:
        raise ValueError(f"Setup expects {n} {spec} channels, but {spec}_dets spcifies {len(ids)}")
    return np.asarray(ids, dtype=np.object_)


def _apply_mask(mask:np.ndarray[np.bool_], *args:np.ndarray)->tuple[np.ndarray,...]:
    """Mask all arrays in args by mask"""
    return tuple(None if arg is None else arg[mask] for arg in args)


def regularize_photon_data(setup:PhSpec,
                          times:np.ndarray[np.int64]|tb.Node,
                          dets:np.ndarray[np.uint8]|tb.Node,
                          nanos:np.ndarray[np.uint16]|tb.Node|None=None,
                          particles:np.ndarray[np.uint8]|tb.Node|None=None,
                          em_dets:tuple[np.ndarray[np.uint8],...]=None,
                          pol_dets:tuple[np.ndarray[np.uint8],...]=None,
                          split_dets:tuple[np.ndarray[np.uint8]]=None,
                          det_ids:np.ndarray[np.uint8]=None,
                          tcspc_offsets:np.ndarray[np.uint16]=None,
                          sort:bool=True, group:GroupFuture=None)->PhArray:
    """
    Sort raw photon data into defined streams, and return pharray.
    This is a convenience method for correctly processing raw photon data once
    the sorce file has been processed into timestamps, detectors, (and nanotimes,
    if relevant), as well as correct excitation ranges identified.
    
    This ensures that the streams are assigned consistent with the conventions
    used in smfBursts. The returned DetDef object should either be used in the
    StreamSpec passed to the |DataSet|, or compared with other DetDefs from other
    excitation spots to ensure the are identical

    Parameters
    ----------
    setup : PhSpec
        PhSpec settings for new PhArray object.
    times : np.ndarray[np.int64]|tb.Node
        Array of photon arrival times (macrotimes).
    dets : np.ndarray[np.uint8]|tb.Node
        Array of detector indexes photons.
    nanos : tuple[np.ndarray[np.uint16]]|tb.Node|None, optional
        If present, array of nanotimes for photons. The default is None.
    particles : np.ndarray[np.uint8]|tb.Node|None, optional
        If present (only when using simulated data) particle index for photons. 
        The default is None.
    em_dets : tuple[np.ndarray[np.uint8]], optional
        tuple of arrays, each array is the detectors that belong to a single
        (spectral) emmission index. The default is None.
    pol_dets : tuple[np.ndarray[np.uint8]], optional
        tuple of arrays, each array is the detectors that belong to a single
        polarization index. The default is None.
    split_dets : tuple[np.ndarray[np.uint8]], optional
        tuple of arrays, each array is the detectors that belong to a single
        split index. The default is None.
    det_ids : np.ndarray[np.uint8], optional
        Array of detector IDs, used to map id to offset.
    tcspc_offsets : np.ndarray[np.uint16], optional
        Map of raw detector-id to tcspc offset to apply to detector.
    sort : bool, optional
        Whether or not to ensure times are monotonic. Should only be set to False
        if it is already **guaranteed** that time is monotonically increasing.
        The default is True.

    Raises
    ------
    TypeError
        usAlex without alex_period.

    Returns
    -------
    PhArray
        Photon data properly sorted for excitation and emission characteristics.

    """
    # reqularize type
    times = np.asarray(times, dtype=np.int64)
    nanos = None if nanos is None else np.asarray(nanos, dtype=np.uint16)
    if sort and _nonmonotonic(times):
        sort = np.argsort(times)
        times, dets, nanos, particles = _apply_mask(sort, times, dets, nanos, particles)
        sort = True # so sort array can be freed
    valid_ids = union_multi(*(union_multi(*ids) for ids in (em_dets, pol_dets, split_dets) if ids is not None))
    detdef = setup.detdef
    em_dets = _one_none_match(setup.detdef.em, em_dets, valid_ids, "em")
    pol_dets = _one_none_match(setup.detdef.pol, pol_dets, valid_ids, "pol")
    split_dets = _one_none_match(setup.detdef.split, split_dets, valid_ids, "split")
    if 'ex_ranges' in setup:
        ex_ranges = setup.ex_ranges
        if nanos is not None:
            # uses nsALEX
            if tcspc_offsets is not None:
                if det_ids is None:
                    raise ValueError("must supply det_ids to offset nanos")
                nanos = _apply_offset(nanos[:], dets[:], det_ids, tcspc_offsets)
            odets, mask = _sort_ex(nanos[:], ex_ranges)
        else:
            # uses usALEX
            if 'alex_period' not in setup:
                raise ValueError("if nanos are not specified, must specify alex_period")
            odets, mask = _sort_ex((times[:]-setup.alex_offset)%setup.alex_period, setup.ex_ranges)
        odets *= detdef.ex_stride
        mask *= np.isin(dets[:], valid_ids)
    else:
        odets = np.zeros(dets[:].size, dtype=np.uint8)
        mask = np.isin(dets[:], valid_ids)
    times, odets, nanos, particles = _apply_mask(mask, times, odets, nanos, particles)
    ndets = dets[mask]
    del mask
    # sort photons for emission characteristics
    for shift, isect in enumerate_intersects(em_dets, pol_dets, split_dets):
        odets[np.isin(ndets, isect)] += shift
    kwargs = dict(setup=setup, times=times, dets=odets)
    if nanos is not None:
        kwargs['nanos']  = nanos
    if particles is not None:
        kwargs['particles'] = particles
    pharray = PhArray(kwargs, group=group)
    return pharray


def get_phsel_ex_range(setup:PhSpec, phsel:PhSel)->int:
    """
    Extract the number of bins in the excitation window of ph_sel

    Parameters
    ----------
    setup : PhSpec
        photon data settings to be queried.
    ph_sel : PhSel
        :class:`smfbursts.ph_sel.Ph_sel` of interest.

    Returns
    -------
    int
        number of bins in ex range of ph_sel.

    """
    phsel = phsel.render_positive(setup['detdef'])
    if len(phsel.ex.elements) != 1:
        raise ValueError("can only determine range for single ex stream PhSel objects")
    iex = list(phsel.ex.elements)[0]
    rng = setup['ex_ranges'][iex]
    if rng.shape[0] != 1:
        warnings.warn("non-contiguous excitation range")
    if np.any(rng[:,0] > rng[:,1]):
        warnings.warn("wrapped excitation rang")
        return 0, setup['tcspc_num_bins'][iex]
    return np.min(rng), np.max(rng)


def in_ph_range(thresh:int, setup:PhSpec, phsel:PhSel)->bool:
    """
    Determine of a number is in the available excitation range for phsel

    Parameters
    ----------
    thresh : int
        Threshold for assessing nanomean.
    setup : PhSpec
        Setup dictionary of reference photon data.
    phsel : PhSel
        DESCRIPTION.

    Raises
    ------
    ValueError
        Invalid PhSel.

    Returns
    -------
    bool
        If PhSel in range or not.

    """
    mn, mx = get_phsel_ex_range(setup, phsel)
    return mn <= thresh and thresh < mx


def get_phsel_range_size(setup:PhSpec, phsel:PhSel)->int:
    """
    Get size of excitation window of setup for a given excitation defined by phsel.

    Parameters
    ----------
    setup : PhSpec
        Setup spec of :class:`PhotonData` defining excitation windows.
    phsel : PhSel
        Photon selection to interogate.

    Returns
    -------
    int
        Number of TCSPC channels (end - start) in excitation window.

    """
    res = get_phsel_ex_range(setup, phsel)
    return res[1] - res[0]


def _pol_names(angle:float)->str:
    """
    Get long name of polarization angle, parallel if 0, perpendicular if 90.0
    and return number otherwise
    """
    if angle == 0.0:
        return r'\parallel'
    if angle == 90.0:
        return r'\perp'
    return f'{angle}'


class _MutTracked(type):
    """Metaclass implementing check for a property in class called mut"""
    def __subclasscheck__(self, subclass):
        return hasattr(subclass, 'mut') and isinstance(subclass.mut, property)


class MutTracked(metaclass=_MutTracked):
    """"Metaclass- any class that implements a mut property is subclass"""
    pass


class PhotonData(DataSet):
    r"""
    |DataSet| for PhotonData
    
    Parameters
    ----------
    pharray : PhArray
        PhArray defining the core data of the object.
    group : None | tb.Group | Callable, optional
        Where to store tables in HDF5 file. Default is None
    autosave : bool, optional
        Whether to save data to HDF5 file upon computation. Default is False.
    irf : dict[PhSel:np.ndarray[np.int64]], optional
        Dictionary of IRF histograms, 1 per stream. Defualt is None
    irf_thresh : dict[PhSel:int], optional
        Dictionary of thresholds for computing nanomean, 1 per stream.
        Default is None.
    meta : dict, optional
        dictionary of arbitrary additional metadata
    save_memory : bool, optional
        flag for tables to select between memory expensive or memory saving 
        algorithms. The default is False.
    track : bool, optional
        Whether to "track" the file, ie close file when all tracking datasets
        have ceased to exist. The default is True.
    file : tb.File, optional
        HDF5 file in which to save data, is inferior to group. The default is None
    group_no : int | bool, optional
        If True, then group is created within group with name "photon_data", 
        if number, then data saved in subgroup with name "photon_data[group_no]"
        if False, data saved directly in group. The default is 1.
    ref : PhotonHDF5Data, optional
        Original data, used to determine if data was modified after loading. 
        The default is None.
    meta_conflict_policy : {'check', 'warn', 'pass', 'error'}
        How to handle saved meta-diskdict keys already present in the saved
        HDF5 group. This is passed to the init of |DiskDict| ``save_conflict_policy``
        keyword argument of the meta dictionary.
        Options are\:
        
        - "pass" (default) if key is present in HDF5 group and input dictionary,
          use the value of the HDF5 group, regardless of differences
        - "warn" if key is present in HDF5 group and input dictionary,
          use the value of the HDF5 group, and raise warning if the values are
          different
        - "check" if key is present in HDF5 group and input dictionary,
          values are checked, if they are not identical, raise an error
        - 'error' raise an error automatically if key is present in both HDF5
          group an input dictionary
        
        The default is 'pass'.
    """
    _group_name = 'photon_data'
    
    # : dictionary of codes for (relatitive GateGroup, Gate) : map
    _gates: dict[Gate:tuple[GateGroup, np.ndarray[np.bool_]]]
    # : nested dictionary of masks {requested:{relative:mask}} for gategroups
    _gategroups: dict[GateGroup:dict[GateGroup,np.ndarray[np.bool_]]]
    _group: GroupFuture  # location to store HDF5 data, None otherwise
    _temp_group: None|tuple[Param,tb.Group]
    autosave: bool  # : whether to automatically record computed columns in HDF5 file

    _pharray: PhArray
    _save_memory: bool
    _irf: SubDiskDict
    _irf_thresh: SubDiskDict
    _reference: weakref.ReferenceType
    _array_cache: weakref.WeakValueDictionary

    def __init__(self, pharray:None|PhArray=None, group:GroupFuture=None, autosave:bool=False, 
                 irf:dict[PhSel:np.ndarray[np.int64]]=None, irf_thresh:dict[PhSel:int]=None, 
                 meta:dict=None, save_memory:bool=False, track:bool=True, file:tb.File=None,
                 group_no:int|bool=1, ref:MutTracked=None, 
                 meta_conflict_policy:Literal['check','warn','error','pass']='pass'):
        super().__init__(group, autosave, meta, pharray=pharray, irf=irf, irf_thresh=irf_thresh, 
                         save_memory=save_memory, track=track, file=file, group_no=group_no, ref=ref,
                         meta_conflict_policy=meta_conflict_policy)

    def __init_data__(self, pharray:None|PhArray=None, 
                    irf:dict[PhSel:np.ndarray[np.int64]]=None, irf_thresh:dict[PhSel:int]=None,
                    save_memory=False, ref:Any=None):
        # intercept creating from group
        if pharray is None:
            if not self._group._creatable:
                raise TypeError("cannot load from non-creatable group")
            self._group._create() # ensures group is created
            # processed data saved
            if 'pharray' in self._group:
                pharray = PhArray(self._group._create_groupfuture('pharray'))
            else:
                pharray = None
        if pharray is not None:
            if not isinstance(pharray, PhArray):
                raise TypeError(f"pharray must be PhArray, got {type(pharray)}")
            self._pharray = pharray
        self._reference = ref
        self._save_memory = bool(save_memory)
        irf = dict() if irf is None else irf
        if not isinstance(irf, dict):
            try:
                irf = dict(*irf)
            except Exception as e:
                raise TypeError(f"irf must be specified as dict of keys as PhSel:array, not {type(irf)}") from e
        self._irf = SubDiskDict(self._meta, 'irf', self._check_irf)
        irf_thresh = dict() if irf_thresh is None else irf_thresh
        self._irf_thresh = SubDiskDict(self._meta, 'irf_thresh', self._check_irf_thresh)
        for ph_sel, hst in irf.items():
            self._irf[ph_sel] = hst
        if not isinstance(irf_thresh, dict):
            try:
                irf_thresh = dict(*irf_thresh)
            except Exception as e:
                raise TypeError(f"irf_thresh must be specified as dict of keys as PhSel:int, not {type(irf_thresh)}") from e
        for ph_sel, thresh in irf_thresh.items():
            self._irf_thresh[ph_sel] = thresh

    def _check_irf(self, phsel:PhSel, hst:np.ndarray)->tuple[PhSel,np.ndarray[np.int64]]:
        """Check irf key value pair- ie phsel in range and hst correct size/type"""
        phsel = phsel.render_positive(self.detdef, convert_all=True)
        if self.detdef.get_stream_ids(phsel).size != 1:
            raise ValueError("PhSel must specifiy a single index for the given detdef")
        try:
            hst = np.asarray(hst, dtype=np.int64)
        except Exception as e:
            raise ValueError("Values of irf must be 1D numpy arrays with size matching ex range") from e
        if hst.shape != (get_phsel_range_size(self.setup, phsel), ):
            raise ValueError(f"irf for {phsel} does has incorrect size, expected {get_phsel_range_size(self.setup, phsel)}, got {hst.size}")
        phsel = phsel.render_positive(self.detdef, convert_all=True)
        return phsel, hst

    def _check_irf_thresh(self, phsel:PhSel, thresh:int)->tuple[PhSel,int]:
        """Check irf_thresh key value pair- phsel in range and thresh in range"""
        phsel = phsel.render_positive(self.detdef, convert_all=True)
        if self.detdef.get_stream_ids(phsel).size != 1:
            raise ValueError("PhSel must specifiy a single index for the given detdef")
        try:
            thresh = int(thresh)
        except Exception as e:
            raise TypeError("thresholds must be integers between 0 and excitation range") from e
        if not in_ph_range(thresh, self.setup, phsel):
            raise ValueError("thresholds must be inside excitation range of phsel")
        phsel = phsel.render_positive(self.detdef, convert_all=True)
        return phsel, thresh

    def _calc_dataID(self)->bytes:
        """Compute hash of raw data to get dataID"""
        hs = hashlib.sha256(self.times)
        hs.update(self.dets)
        if 'nanos' in self._pharray:
            hs.update(self.nanos)
        if 'particles' in self._pharray:
            hs.update(self.particles)
        return hs.digest()

    @property
    def _ref(self)->MutTracked:
        """
        Get ref data used to create PhotonData.
        Main purpose is to check if it was modified post-creation, and thus
        modified raw data should be saved.
        """
        if isinstance(self._reference, weakref.ReferenceType):
            return self._reference()
        return self._reference
    
    @property
    def source_filename(self):
        return self._meta['filename']
    
    def _get_from_pharray(self, name:str, phsel:PhSel)->np.ndarray:
        """Get masked photon data array"""
        if phsel == phsel_all:
            return self._pharray[name]
        stream_ids = self.detdef.get_stream_ids(phsel)
        return self._pharray[name][np.isin(self._pharray['dets'], stream_ids)]

    @property
    def times(self)->np.ndarray[np.int64]:
        """Arrival times (macrotimes) of photons, in unit of clk_p"""
        if not hasattr(self, '_pharray'):
            raise AttributeError("PhArray not recorded, cannot access original times")
        return self._pharray['times']

    def get_times(self, phsel:PhSel=phsel_all)->np.ndarray[np.int64]:
        """
        Get photon arrival times (macrotimes) of photons belonging to phsel.

        Parameters
        ----------
        phsel : PhSel, optional
            PhSel defining streams to return. The default is PhSel('all').

        Returns
        -------
        np.ndarray[np.int64]
            photon arival times (macrotimes) masked by phsel.

        """
        return self._get_from_pharray('times', phsel)

    @property
    def dets(self)->np.ndarray[np.uint8]:
        """*Sorted* detector indexes of photons"""
        if not hasattr(self, '_pharray'):
            raise AttributeError("PhArray not recorded, cannot access original dets")
        return self._pharray['dets']

    def get_dets(self, phsel:PhSel=phsel_all)->np.ndarray[np.uint8]:
        """
        Get detectors array, of photon belonging to phsel.

        Parameters
        ----------
        phsel : PhSel, optional
            PhSel defining streams to return. The default is PhSel('all').

        Returns
        -------
        np.ndarray[np.uint8]
            Detectors array masked by phsel.

        """
        return self._get_from_pharray('dets', phsel)

    @property
    def nanos(self)->np.ndarray[np.uint16]:
        """Nanotimes of photons, in unit of tcspc_unit"""
        if not hasattr(self, '_pharray'):
            raise AttributeError("PhArray not recorded, cannot access original nanotimes")
        if 'nanos' not in self._pharray:
            raise AttributeError("non-pulsed excitation data")
        return self._pharray['nanos']

    def get_nanos(self, phsel:PhSel=phsel_all)->np.ndarray[np.uint16]:
        """
        Get nanotimes of photon belonging to phsel. Pulsed excitation only.

        Parameters
        ----------
        phsel : PhSel, optional
            PhSel defining streams to return. The default is PhSel('all').

        Returns
        -------
        np.ndarray[np.uint16]
            Photon nanotimes, of photons in phsel.

        """
        return self._get_from_pharray('nanos', phsel)

    @property
    def pulsed(self)->bool:
        """If data is has pulsed excitation"""
        return 'nanos' in self._pharray

    @property
    def particles(self)->np.ndarray[np.uint8]:
        """*Simulated data only.* particle index of photons"""
        if not hasattr(self, '_pharray'):
            raise AttributeError("PhArray not recorded, cannot access original particles")
        
        if 'particles'not in self._pharray:
            raise AttributeError("real data, no particles array")
        return self._pharray['particles']
    
    def get_particles(self, phsel:PhSel=phsel_all)->np.ndarray[np.uint8]:
        """
        Get particle indexes on photons in phsel. Simulated data only.

        Parameters
        ----------
        phsel : PhSel, optional
            PhSel defining streams to return. The default is PhSel('all').

        Returns
        -------
        np.ndarray[np.uint8]
            Indexes of particles in phsel.

        """
        return self._get_from_pharray('particles', phsel)

    @property
    def simulated(self)->bool:
        """If data was produced from simulation"""
        return hasattr(self, '_pharray') and 'particles' in self._pharray

    @property
    def clk_p(self)->float:
        """clock unit of times (macroitimes), in unit of seconds"""
        return hasattr(self, '_pharray') and self._pharray['setup'].clk_p

    @property
    def detdef(self)->DetDef:
        """DetDef of data, defins number of ex, em, pol, split streams"""
        return self._pharray['setup'].detdef

    @property
    def setup(self)->PhSpec:
        """:class:`PhSec` dictionary of all settings of processed photon data"""
        return self._pharray['setup']

    @property
    def save_memory(self)->bool:
        """Switch for using memory saving functions in tables"""
        return self._save_memory

    @save_memory.setter
    def save_memory(self, value:bool):
        self._save_memory = bool(value)
        for table in self._tables.values():
            if hasattr(table, "_save_memory_switch"):
                table._save_memory_switch(self._save_memory)

    @property
    def irf(self)->dict[PhSel:np.ndarray[np.int64]]:
        """Dictionary of irfs (1 per single stream PhSel)"""
        return self._irf

    @irf.setter
    def irf(self, val:dict):
        for k, v in val.items():
            if k in self._irf:
                if np.any(self._irf[k] != v):
                    raise ValueError(f'value of {k} already set')
            self._irf[k] = v

    @property
    def irf_thresh(self)->dict[PhSel:int]:
        """Dictionary of each single stream PhSel, of nanotime threshold for computing mean nanotime"""
        return self._irf_thresh
    
    @irf_thresh.setter
    def irf_thresh(self, val:dict):
        for k, v in val.items():
            if k in self._irf_thresh:
                if self._irf_thresh[k] != v:
                    raise ValueError(f"value of {k} already set")
                continue
            self._irf_thresh[k] = v

    def get_stream_names(self)->dict[str:dict[int,str]]:
        """
        Get a dictionary providing names for each channel-type and stream.
        Dictionary is nested, has structure:
        ``{'ex_wv':{n:name,...}, 'em_wv_centers':{n:name,...}}, 'pol_angle':{n:name,...}``
        Where n is stream number (int), and name is name of stream (str) specifying
        wavelength or angle of polarization.
        """
        out = dict()
        if 'ex_wv' in self.setup:
            out['ex'] = {i:f'{wv*1e9:.0f}'  for i, wv in enumerate(self.setup.ex_wv)}
        if 'em_wv_centers' in self.setup:
            out['em'] = {i:f'{wv*1e9:.0f}'  for i, wv in enumerate(self.setup.em_wv_centers)}
        if 'pol_angle' in self.setup:
            out['pol'] = {i:_pol_names(angle) for i, angle in enumerate(self.setup.pol_angle)}
        return out
    
    def _save(self, *args:Param, group:tb.Group=None, save_sorted:bool=None, _strict:bool=True)->tb.Group:
        """Internal function for saving data, wrapped by :meth:`PhotonData.save`"""
        if save_sorted is None:
            save_sorted = True if self._ref is None else self._ref.mut
        if _strict and (self._ref is None or self._ref.mut) and not save_sorted:
            raise ValueError("Original data modified, must save_sorted=True")
        outgroup = super().save(*args, group=group)
        if save_sorted and 'pharray' not in outgroup:
            self._pharray.save(outgroup._v_file.create_group(outgroup, 'pharray'))
        return outgroup

    def save(self, *args:Param, group:tb.Group=None, save_sorted:bool=None)->tb.Group:
        return self._save(*args, group=group, save_sorted=save_sorted)

    def save_photonHDF5(self, file:str|PathLike|tb.File, save_sorted:bool=False, 
                        close:bool=None, **kwargs)->tb.File:
        """
        Create a photonHDF5 file with all saved analysis in the user/smfBursts
        group.

        Parameters
        ----------
        file : str | os.PathLike | tb.File
            Path to file or tb.File object.
        close : bool, optional
            Whether to close file after saving. If None, will determine whether
            to save based on how file inputed and if other objects are tracking
            it. The default is None.
        save_sorted : bool, optional
            Whether to also record the sorted photons in 
            ``user/smfBursts/photon_data[x]`` group. 
            The default is False

        Returns
        -------
        tb.File
            File where data saved.

        """
        if self._ref is None:
            raise ValueError("No raw data to save")
        if self._ref.n_ch != 1:
            raise ValueError("multi-spot measurements must be saved from PhotonDataList")
        close = not isinstance(file, tb.File) if close is None else bool(close)
        dkwargs = dict(mode='a', filters=tb.Filters(complevel=6))
        dkwargs.update(kwargs)
        try:
            file = file if isinstance(file, tb.File) else tb.open_file(file, **dkwargs)
            self._ref.save_photonHDF5(file, close=False)
            if 'user/smfBursts/photon_data0' not in file.root:
                file.create_group('/user/smfBursts', 'photon_data0', createparents=True)
            group = file.root.user.smfBursts.photon_data0
            self._save(group=group, save_sorted=save_sorted, _strict=False)
        except Exception as e:
            raise e
        finally:
            if close:
                file.close()
            elif not self._group._created:
                self.set_group(group)
        return file


class PhotonDataList(DataSetList):
    """
    DataSetList object for PhotonData. Stores mutliple :class:`PhotonDataList`
    useful for multi-spot photon-HDF5 files, or when wanting to perform the same
    operation on multiple data sets.
    """
    _group_name = 'photon_data'
    def __init__(self, datas:Sequence[PhotonData]):
        if any(d.detdef != datas[0].detdef for d in datas[1:]):
            raise ValueError("Detector definitions of PhotonData objects incompatible")
        super().__init__(datas)

    @property
    def detdef(self)->DetDef:
        """Detector definition of :class:`PhotonData`, since all must be same,
        can DetDef directly"""
        return self._datas[0].detdef

    @property
    def setup(self)->tuple[PhSpec,...]:
        """Tuple of :class:`PhSpec` objects, 1 for each :class:`PhotonData`"""
        return tuple(d._pharray['setup'] for d in self._datas)

    @property
    def clk_p(self)->float:
        return self.datas[0].clk_p

    def iter_times(self, phsel:PhSel=phsel_all)->Iterator[np.ndarray[np.int64]]:
        """
        Iterate over macrotimes arrays in each :class:`PhotonData`

        Parameters
        ----------
        phsel : PhSel, optional
            PhSel defining streams to incldue in array. The default is PhSel('all').
        
        Yields
        ------
        np.ndarray[np.uint8]
            Array of macrotimes filtered by phsel.

        """
        for data in self._datas:
            yield data.get_times(phsel)

    def get_times(self, phsel:PhSel=phsel_all)->np.ndarray[np.ndarray[np.int64]]:
        """
        Get macrotimes arrays (pulsed excitation only) in each :class:`PhotonData`
        
        Parameters
        ----------
        phsel : PhSel, optional
            PhSel defining streams to incldue in array. The default is PhSel('all').
        
        Returns
        -------
        np.ndarary[np.ndarray[np.uint8]]
            Array of arrays of macrotimes filetered by phsel.
        """
        return np.array(list(self.iter_times(phsel)), dtype=np.object_)

    @property
    def times(self)->np.ndarray[np.ndarray[np.int64]]:
        """All photon macrotimes"""
        return np.array(list(self.iter_times()), dtype=np.object_)

    def iter_dets(self, phsel:PhSel=phsel_all)->Iterator[np.ndarray[np.uint8]]:
        """
        Iterate over detector arrays in each :class:`PhotonData`

        Parameters
        ----------
        phsel : PhSel, optional
            PhSel defining streams to incldue in array. The default is PhSel('all').
        
        Yields
        ------
        np.ndarray[np.uint8]
            Array of detector indexes filtered by phsel.

        """
        for data in self._datas:
            yield data.get_dets(phsel)

    def get_dets(self, phsel:PhSel=phsel_all)->np.ndarray[np.ndarray[np.uint8]]:
        """
        Get detectors arrays (pulsed excitation only) in each :class:`PhotonData`
        
        Parameters
        ----------
        phsel : PhSel, optional
            PhSel defining streams to incldue in array. The default is PhSel('all').
        
        Returns
        -------
        np.ndarary[np.ndarray[np.uint8]]
            Array of arrays of detectors filetered by phsel.
        """
        return np.array(list(self.iter_nanos(phsel)), dtype=np.object_)

    @property
    def dets(self)->np.ndarray[np.ndarray[np.uint8]]:
        """All detectors in each PhotonData group"""
        return np.array(list(self.iter_dets()), dtype=np.object_)

    def iter_nanos(self, phsel:PhSel=phsel_all)->Iterator[np.ndarray[np.uint16]]:
        """
        Iterate over nanotime arrays (pulsed excitation only) in each :class:`PhotonData`

        Parameters
        ----------
        phsel : PhSel, optional
            PhSel defining streams to incldue in array. The default is PhSel('all').
        
        Yields
        ------
        np.ndarray[np.uint16]
            Array of nanotimes filtered by phsel.

        """
        for data in self._datas:
            yield data.get_nanos(phsel)

    def get_nanos(self, phsel:PhSel=phsel_all)->np.ndarray[np.ndarray[np.uint16]]:
        """
        Get nanotimes arrays (pulsed excitation only) in each :class:`PhotonData`
        
        Parameters
        ----------
        phsel : PhSel, optional
            PhSel defining streams to incldue in array. The default is PhSel('all').
        
        Returns
        -------
        np.ndarary[np.ndarray[np.uint16]]
            Array of arrays of nanotimes filtered by phsel.
        """
        return np.array(list(self.iter_nanos(phsel)), dtype=np.object_)

    @property
    def nanos(self)->np.ndarray[np.ndarray[np.uint16]]:
        """All nanotimes in each PhotonData group"""
        return np.array(list(self.iter_times()), dtype=np.object_)

    def iter_particles(self, phsel:PhSel=phsel_all)->Iterator[np.ndarray[np.uint8]]:
        """
        Iterate over particles arrays (simulated data only) in each :class:`PhotonData`

        Parameters
        ----------
        phsel : PhSel, optional
            PhSel defining streams to incldue in array. The default is PhSel('all').
        
        Yields
        ------
        np.ndarray[np.uint8]
            Array of particle indexes filtered by phsel.

        """
        for data in self._datas:
            yield data.get_particles(phsel)

    def get_particles(self, phsel:PhSel=phsel_all)->np.ndarray[np.ndarray[np.uint8]]:
        """
        Get particles arrays (simulated data only) in each :class:`PhotonData`
        
        Parameters
        ----------
        phsel : PhSel, optional
            PhSel defining streams to incldue in array. The default is PhSel('all').
        
        Returns
        -------
        np.ndarary[np.ndarray[np.uint8]]
            Array of arrays of particles indexes.
        """
        return np.array(list(self.iter_particles(phsel)), dtype=np.object_)

    @property
    def particles(self)->np.ndarray[np.ndarray[np.int64]]:
        """All particle indexes in each PhotonData group"""
        return np.array(list(self.iter_particles()), dtype=np.object_)

    def save(self, *args:Param, group:tb.Group=None, name:Callable[[int],str]=None, 
             save_sorted:bool=None)->list[tb.Group]:
        """
        Save specified tables to HDF5 file.

        Parameters
        ----------
        *args : Param
            Tables to save to file.
        group : tb.Group, optional
            Group in which to save, if not specified, use default group. 
            If no default group is set, will raise an error.
            The default is None.
        name : Callable[[int],str], optional
            Callable that takes int/bool and outputs name for each :class:`PhotonData`
            in datas. The default is None.
        save_sorted : bool, optional
            Whether, for each data set, to save raw photons as well as each table. 
            The default is None.

        Returns
        -------
        list[tb.Group]
            List of groups (1 per data-set) where data was saved.

        """
        return super().save(*args, group=group, name=name, save_sorted=save_sorted)

    def save_photonHDF5(self, file:str|PathLike|tb.File, 
                               save_sorted:bool=False, close:bool=None)->tb.File:
        """
        Save data in PhotonHDF5 format. Computed tables will be saved
        under ``/user/smfBursts/photon_data[x]`` groups.

        Parameters
        ----------
        file : str|PathLike|tb.File
            Path to file where data is to be saved.
        save_sorted : bool, optional
            Whether to also record the sorted photons in 
            ``user/smfBursts/photon_data[x]`` group. 
            The default is False
        close : bool, optional
            Whether to close file after saving. If None, infer from input and if
            file is being used by othe objects.. The default is None.

        Returns
        -------
        file : tb.File
            File object where data was saved.

        """
        # TODO: change this part of the checking to make sure saving rules are correct, currenlty fails when no previous HDF5 created
        ref = self._datas[0]._ref
        if any(data._ref is None or data._ref != ref for data in self._datas):
            raise ValueError("Inconsistent or deleted raw data, cannot save raw HDF5 data")
        close = isinstance(file, str) if close is None else bool(close)
        try:
            file = file if isinstance(file, tb.File) else tb.open_file(file, 'a')
            self._ref.save_photonHDF5(file, close=False)
            if 'user/smfBursts/photon_data0' not in file.root:
                file.create_group('/user/smfBursts', 'photon_data0', createparents=True)
            group = file.root.user.smfBursts.photon_data0
            for i, phdata in enumerate(self._datas):
                phdata._save(group=file.create_group(group, f'photon_data{i}'), 
                             save_sorted=save_sorted, _strict=False)
        except Exception as e:
            raise e
        finally:
            if close:
                file.close()
        return file
                                                        

PhotonDataS = PhotonData|PhotonDataList


def _echo_first(*args):
    """Return first argument to function"""
    return args[0]


def _mask_byset(mask:np.ndarray, inset:np.ndarray)->np.ndarray:
    """return only elements in mask that are in inset"""
    return mask[np.isin(mask, inset)]


def _regularize_ph_sel(val:PhSel|Sequence[PhSel], 
                      detdef:DetDef, convert_all:bool=True)->PhSel|Sequence[PhSel]:
    """
    Ensure all ph_sels are rendered positive based on detef.

    Parameters
    ----------
    val : PhSel|Sequence[PhSel]
        ph_sels to convert.
    detdef : DetDef
        DetDef definition.
    convert_all : bool, optional
        Whether to convert PhSel('all') to stream enumeration. The default is True.

    Returns
    -------
    PhSel|Sequence[PhSel]
        Rendered sequence of PhSel.

    """
    if isinstance(val, PhSel):
        return val.render_positive(detdef, convert_all=convert_all)
    if isinstance(val, tupledict):
        return tupledict(*((k, _regularize_ph_sel(v, detdef, convert_all)) for k, v in val.items()))
    if isinstance(val, dict):
        return {k:_regularize_ph_sel(v, detdef, convert_all) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return type(val)((_regularize_ph_sel(v, detdef, convert_all)for v in val))
    if isinstance(val, np.ndarray) and val.dtype == np.object_:
        return np.array([_regularize_ph_sel(v, detdef, convert_all) for v in val.reshape(-1)]).reshape(val.shape)
    return val


TV_str_start = TV_str(isin=('start', 'istarttime'))
TV_str_stop = TV_str(isin=('stop', 'istoptime'))


def _regularize_column_startstop(source_param:Param, *args:Hashable)->tuple[str,str]:
    r"""
    Convenience function for column regularization of columns whose keytup
    ends in `starttime, stoptime`. \*args should be the args after other
    keys peeled off of beginning.
    If one or the other is not specified, defaults them to `istarttime` and
    `istoptime`.
    """
    startoptions = source_param._colstarttypes
    stopoptions = source_param._colstoptypes
    startdefault = source_param._colstartdefaultstr
    stopdefault = source_param._colstopdefaultstr
    ln = len(args)
    if ln == 2:
        starttype, stoptype = args if args[0] in startoptions else args[::-1]
    elif ln == 1:
        starttype, stoptype = (args[0], stopdefault) if args[0] in startoptions else (startdefault, args[0])
    elif ln == 0:
        starttype, stoptype = startdefault, stopdefault
    else:
        raise TableConstructionError('Too many args passed to _regularize_column_startstop, this is most likely an incorrectly written reg_func')
    if not isinstance(starttype, str):
        starttype = startoptions[starttype]
    if not isinstance(stoptype, str):
        stoptype = stopoptions[stoptype]
    if starttype not in startoptions:
        err = ', '.join(f"'{opt}'" for opt in startoptions)
        raise ValueError(f"starttype must be one of [{err}]")
    if stoptype not in stopoptions:
        err = ', '.join(f"'{opt}'" for opt in stopoptions)
        raise ValueError(f"stoptype must be [{err}]")
    return starttype, stoptype


class PhotonTable:
    """
    Mixin class used by :class:`BasePhotonTable` and :class:`ChildPhotonTable`
    
    .. note::
        
        This class should be used for ``isinstance`` calls, :class:`BasePhotonTable`
        and :class:`ChildPhotonTable` subclass this class. Since all 
        :class:`smfbursts.datamodel.tables.Table` should be either
        :class:`smfbursts.datamodel.tables.BaseTable` or 
        :class:`smfbursts.datamodel.tables.ChildTable`, other Tables classes
        should not directly subclass this class.
        
    Primary function of this class is to cause all instances of Ph_sel objects
    specified at instantiation of :class:`smfbursts.datamodel.Param`, and
    :class:`smfbursts.datamodel.Column` objects to be rendered positive based
    on detdef of most basal param.
    
    """
    @classmethod
    def _validate_param(cls, param:Param)->None:
        """Validate intercept to ensure all phsel are positive, convert all based on detdef"""
        pdict = tupledict(*((k,_regularize_ph_sel(v, param.detdef, convert_all=True)) 
                            for k, v in param.params.items()))
        super(_ImData, param).__setattr__('params', pdict)
        cls.validate_param(param)

    @classmethod
    def _regularize_column_kwargs(cls, **kwargs)->dict[str:Any]:
        """
        Intercept regularize_column_kwargs ensuring all phsel are positive, 
        convert all based on detdef
        """
        detdef = kwargs['source_param'].detdef
        kwargs['keytup'] = _regularize_ph_sel(kwargs['keytup'], detdef, convert_all=True)
        return kwargs

    def _get_keys(self, keys:tuple[str,Hashable,...])->tuple[ColumnDef,tuple[Hashable,...],int,Any]:
        """
        Intercept keys for accessing columns, ensures all phsel are positive,
        convert all based on detdef
        """
        coldef, keys, offset, fill = super()._get_keys(keys)
        keys = tuple(key.render_positive(self.param.detdef, convert_all=True) 
                     if isinstance(key, PhSel) else key for key in keys)
        return coldef, keys, offset, fill

    @paramproperty
    def detdef(cls, param:Param)->DetDef:
        """
        All final subclasses must implement this. 
        Returns the :class:`DetDef` of a :class:`Param` based on subclass
        """
        raise NotImplementedError("Subclasses must implement this method")


ColKeyStart = Literal['istarttime', 'start']
ColKeyStop = Literal['istoptime', 'stop']


def _title_startstop_append(name:str, start:ColKeyStart, stop:ColKeyStop)->str:
    """Smart appending start/stop to end of column name based on type of start/stop"""
    if start == 'start' and stop == 'stop':
        name += r'\: full'
    elif start != 'istarttime' or stop != 'istoptime':
        name += rf'\: [{start},{stop}]'
    return name


def _title_unit_append(title:str, unit:str, include_unit:Real|bool)->str:
    """Apppend unit to title based on unit name and rescale"""
    if include_unit is not False:
        title += rf'\: ({get_unit_prefix(include_unit)}{unit})'
    return title


def _title_sels(name:str, origin:PhotonData, *args:PhSel)->str:
    """Get column names when keys contain phsel based on origin data"""
    kw = {'name':name}
    if origin is not None:
        kw.update(detdef=origin.detdef, stream_names=origin.get_stream_names())
    return tuple(sel.tex_str(**kw) for sel in args)


_cs01 = ChannelSet(True, {0,1})


def _pol_ps(sel:PhSel, detdef:DetDef=None, setup:PhSpec=None)->bool:
    """Determine if sel covers all polarization channels- if True, can render as anisotropy"""
    sel = sel if detdef is None else sel.render_positive(detdef, convert_all=True)
    if setup is None and detdef is None:
        return all(s == _csall or s == _cs01 for s in sel.streams)
    if setup is not None and 'pol_angle' in setup:
        ipar = np.argwhere(setup.pol_angle == 0.0)
        iperp = np.argwhere(setup.pol_angle == 90.0)
        if ipar.size and iperp.size:
            _cs = ChannelSet(True, {ipar[0], iperp[0]}).render_positive(setup.pol_angle.size, 
                                                                        conver_all=True)
        else:
            return False
    elif detdef is not None and (setup is None or 'pol_angle' not in setup):
        if detdef.pol == 1:
            return False
        _cs = ChannelSet(True, {0, detdef.pol-1}).render_positive(detdef.pol, convert_all=True)
    else:
        return all(s == _cs01 or s == _csall for s in sel.streams)
    return all(s == _cs for s in sel.streams)


class BasePhotonTableLike(metaclass=TableLike):
    """
    Metaclass determining if table can be accessed like a BasePhotonTable-
    ie if start/stop and similar required columns are implemented.
    Allows child table to "borrow" from BaseTables, and the still be used
    as base for other child tables.
    """
    required_columns = ('start', 'stop', 'istart', 'istop', 'ph_times', 'ph_dets')


class BasePhotonTable(PhotonTable, BaseTable):
    r"""
    Class for :class:`smfbursts.datamodel.tables.BaseTable` which has a :class:`PhotonData`
    origin, and defines a set of *monotonically increasing* ranges of times, such
    as bursts. This class provides methods for columns that can be considered
    "universal" for ranges of times within single photon data.
    
    Ranges of times are defined by ``start`` and ``stop`` columns. Further,
    the ranges of indexes corresponding to all photons in the :attr:`PhotonData.times`
    are defined by the ``istart`` and ``istop`` columns. *The definition and computation
    of ``start``, ``stop``, ``istart`` and ``istop`` columns is the primary role
    of subclasses of this class* **subclasses of this ``BasePhotonTable`` must
    implement these columns**
    
    It is also necessary to define a ``detdef`` paramproperty which can extract
    the :class:`smfbursts.ph_sel.DetDef` from either the params or parents of
    a :class:`smfbursts.datamodel.tables.Param` of the particlar table type.
    
    
    The :attr:`BasePhotonTable.column_defs` class attribute should be set to a
    tuple of :class:`smfbursts.datamodel.tables.ColumnDef` objects defining
    ``start``, ``stop``, ``istart``, ``istop`` columns concatenated with the
    convenience function :func:`make_base_column_defs` to create tuple of column
    defs, any additional user-defined columns can be added by adding to returned
    tuple.
    
    BasePhotonTable defines the following columns:
    
    .. _basephotoncolumns:
    
    Base Photon Columns
    -------------------
    
        istarttime : int
            time of first photon in range
        istoptime : int
            time of last photon in range
        ph_mask : np.ndarray[np.bool\_], (ph_sel:PhSel, )
            mask of photons in ``ph_sel`` vs all photons in range for each range
        ph_times : np.ndarray[np.int64], (ph_sel:PhSel, )
            times (macrotimes) of photons in range belonging to ``ph_sel``.
        ph_nanos : np.ndarray[np.uint16], (ph_sel:PhSel, )
            nanotimes of photons in range belonging to ``ph_sel``.
        ph_dets : np.ndarray[np.uint8], (ph_sel:PhSel, )
            detector indexes (sorted) of photons in range belonging to ``ph_sel``.
        ph_particles : np.ndarray[np.int64], (ph_sel:PhSel, )
            particle indexes (simulated data only) of photons in range belonging to ``ph_sel``
        nph_raw : int, (ph_sel:PhSel, )
            number of photons in ``ph_sel`` in range. No corrections applied (even background)
        ratio_raw : float (phsel_num:PhSel, phsel_dem:PhSel)
            Ratio of ``[nph_raw, phsel_num] / [nph_raw, phsel_dem]``
        anisotropy_raw: float (phsel_p:PhSel, phsel_s:PhSel)
            Anisotropy of two channels 
            (only sensible if phsel_p and phsel_s are parallel/perpendicular compliments of each other).
            Computes as  ``([nph_raw, phsel_p] - [nph_raw, phsel_p])/([nph_raw, phsel_p] + 2*[nph_raw, phsel_p])``
        brightness : float, (ph_sel:PhSel, starttype:{'istarttime', 'start'}, stoptype:{'istoptime', 'stop'})
            photons*s\ :sup:`-1` for stream ``ph_sel`` in range
        max_rate : float, (ph_sel:PhSel, m:int)
            maximum photon rate for ``ph_sel`` based on sliding window of of size ``m`` in range
        dur : float (starttype:{'istarttime', 'start'}, stoptype:{'istoptime', 'stop'})
            duration (in seconds) of range
        midtime : (starttype:{'istarttime','start'}, stoptype:{'istoptime', 'stop'})
            Midpoint time of burst (in s)
        sep : float, offset=-1, non-atomic, (starttype:{'istarttime', 'start'}, stoptype:{'istoptime', 'stop'})
            suparation (in seconds) between successive ranges
        bva : float, (ph_sel_num:PhSel, ph_sel_dem:PhSel, n:int) 
            variance of ratio of :math:`N(ph\_sel\_num)/N(ph\_sel\_dem)` of chuncks 
            of size ``n``.
        ebva : float, (ph_sel_num:PhSel, ph_sel_dem:PhSel, n:int) 
            "Excess" variance of bva, defined as :math:`S^{2} = s^{2} - \sigma^{2}`
            where :math:`s` is the classical BVA, and 
            :math:`$\sigma^{2}=\frac{\langle\epsilon\rangle (1-\langle\epsilon\rangle)}{m}$`
            and :math:`\epsilon` is the ratio of the raw number of photons in
            phsel_num to phsel_dem of the entire burst.
        nanohist : np.ndarray[np.int64] (phsel:PhSel, full:bool)
            histogram (1 per range) of nanotimes of photons in range. If full, the
            return histogram using TCSPC raw bins, if full=False, then trim to excitation range.
        nanomean : float, (:class:`smfbursts.ph_sel.PhSel`, )
            mean nanotime (in seconds) of ph_sel of photons in range. All streams in
            ph_sel should have same irf_thresh in origin data, and be reasonable to
            be treated collectively (single stream, or at least same excitation and
            emission).
        
    Remapped Columns
    ----------------
        E_raw : float ()
            FRET efficiency, maps to ``ratio_raw, PhSel(0ex1em), PhSel(0ex)``
        S_raw : float ()
            Stoichiometry efficiency, maps to ``ratio_raw, PhSel(0ex), PhSel(0ex_1ex1em)``
    
    """
    _parent_ph_subrange:ClassVar[str] = False
    
    _origin: PhotonData
    
    #: |TypeValidator| for start-type column value options for time range columns
    _colstarttype:ClassVar[TypeValidator] = TV_str_start
    #: |TypeValidator| for stop-type column value options for time range columns
    _colstoptype:ClassVar[TypeValidator] = TV_str_stop
    #: Default value for start-type of time range columns
    _colstartdefault:ClassVar[str] = 'istarttime'
    #: Default value for stop-type of time range columns
    _colstopdefault:ClassVar[str] = 'istoptime'
    
    @paramproperty
    def _colstarttypes(cls, param:Param)->tuple[str,...]:
        return cls._colstarttype.ckwargs['isin']
    
    @paramproperty
    def _colstoptypes(cls, param:Param)->tuple[str,...]:
        return cls._colstoptype.ckwargs['isin']
    
    @paramproperty
    def _colstartdefaultstr(cls, param:Param)->str:
        return cls._colstartdefault

    @paramproperty
    def _colstopdefaultstr(cls, param:Param)->str:
        return cls._colstopdefault


    def _init_new_(self):
        if self.param.detdef != self.origin.detdef:
            raise ValueError("mismatched detdefs")
        super()._init_new_()

    def record_photondata(self)->None:
        """
        Save core photon data arrays (array per row).
        This can speed procesing of data.

        """
        if self._derived:
            return self._base.record_photondata()
        if 'nanos' not in self._cache and self.origin.pulsed:
            self._cache['nanos'] = self['ph_nanos', phsel_all]
        if 'particles' not in self._cache and self.origin.simulated:
            self._cache['particles'] = self['ph_particles', phsel_all]
        if 'dets' not in self._cache:
            self._cache['dets'] = self['ph_dets', phsel_all]
        # record times last so that check in _get_parent_ph_subrange returns false for all iterations
        if 'times' not in self._cache:
            self._cache['times'] = self['ph_times', phsel_all]

    @paramproperty
    def detdef(cls, param:Param)->DetDef:
        """Method should take param and extract the DetDef"""
        raise NotImplementedError("subclasses of BasePhotonTable must implement detdef method")
    
    def _get_istarttime(self)->np.ndarray[np.int64]:
        """Getter function, time of first photon in each row, in clk_p units"""
        return self.origin.times[self['istart']]

    def _get_istoptime(self)->np.ndarray[np.int64]:
        """Getter function, time+1 (so can be treated as half-open interval) 
        of last photon in each row, in clk_p units"""
        return self.origin.times[self['istop']-1]+1

    def _get_midtime(self, starttime:ColKeyStart, stoptime:ColKeyStop)->np.ndarray[np.double]:
        """Getter function, midpoint time of burst"""
        start, stop = self[stoptime], self[starttime]
        return ((stop - start)/2 + start)*self.origin.clk_p

    @classmethod
    def _get_midtime_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """Title function for midtime"""
        if col.keytup[0] == 'istarttime' and col.keytup[1] == 'istoptime':
            out = 'time'
        else:
            out = f'time:({col.keytup[0]}, {col.keytup[1]})'
        if include_unit is not False:
            out += f' ({get_unit_prefix(include_unit)}s)'
        return out

    def _get_parent_ph_subrange(self)->Union["BasePhotonTable",None]:
        """Get parent table if it has stored photon arrays"""
        if 'times' in self._bcache:
            return self
        parent_name = self._parent_ph_subrange
        if not parent_name:
            return None
        parent_param = self.param.parents[parent_name].base_param
        if self.origin.has_table_saved(parent_param): 
            return self._origin.get_table(parent_param)._get_parent_ph_subrange()
        return None

    def _iter_ph_array_all(self, key:str)->Iterator[np.ndarray]:
        """General iterator, if available iterate over stored photon arrays of type key"""
        # iterate from stored photon arrays (not from origin)
        if key in self._bcache:
            yield from self._bcache.iter_key(key)
        # iterate from arrays stored in parent base table
        elif (parent := self._get_parent_ph_subrange()) is not None:
            istartstop = zip(self.iter_column('istart'), self.iter_column('istop'))
            istart, istop = next(istartstop)
            for pistart, pistop, arr in zip(parent.iter_column('istart'), 
                                            parent.iter_column('istop'), 
                                            parent._bcache.iter_key(key)):
                while istop <= pistop:
                    yield arr[istart-pistart:istop-pistart]
                    istart, istop = next(istartstop)
        # iterate from origin photon arrays
        else:
            if self.origin.save_memory:
                for istart, istop in zip(self.iter_column('istart'), self.iter_column('istop')):
                    yield self.origin._pharray.get_from_index(key, slice(istart, istop))
            else:
                pharray = getattr(self.origin._pharray, key)
                for istart, istop in zip(self.iter_column('istart'), self.iter_column('istop')):
                    yield pharray[istart:istop]

    def _iter_ph_mask(self, phsel:PhSel)->np.ndarray[np.bool_]:
        """Iterator for mask of phsel of photon arrays"""
        stream_ids = self.origin.detdef.get_stream_ids(phsel)
        if not self.origin.save_memory and hasattr(self.origin, 'dets'):
            mask = self.origin._get_from_cache(BasePhotonTable, tupledict(('phsel',phsel),), 'mask',
                                               lambda:np.isin(self.origin.dets, stream_ids))
            for istart, istop in zip(self.iter_column('istart'), self.iter_column('istop')):
                yield mask[istart:istop]
        else:
            for dets in self._iter_ph_array_all('dets'):
                yield np.isin(dets, stream_ids)

    def _iter_ph_array(self, key:str, phsel:PhSel)->np.ndarray:
        """Iterator over ph_array key masked by phsel"""
        if phsel.render_positive(self.origin.detdef, convert_all=True) == phsel_all:
            yield from self._iter_ph_array_all(key)
        else:
            for arr, mask in zip(self._iter_ph_array_all(key), self._iter_ph_mask(phsel)):
                yield arr[mask]

    def _iter_ph_dets(self, phsel:PhSel)->Iterator[np.ndarray[np.uint8]]:
        """Iterator function, photon detector indices"""
        yield from self._iter_ph_array('dets', phsel)

    def _iter_ph_times(self, phsel:PhSel)->Iterator[np.ndarray[np.int64]]:
        """Iterator function, photon times"""
        yield from self._iter_ph_array('times', phsel)

    def _iter_ph_nanos(self, phsel:PhSel)->Iterator[np.ndarray[np.uint16]]:
        """Iterator function, photon nanotimes"""
        yield from self._iter_ph_array('nanos', phsel)

    def _iter_ph_particles(self, phsel:PhSel)->Iterator[np.ndarray[np.uint8]]:
        """Iterator function, photon detector particles (simulated)"""
        yield from self._iter_ph_array('particles', phsel)

    def _iter_nph_raw(self, phsel:PhSel)->Iterator[int]:
        """Iterator function, for nph_raw column, raw photon counts for phsel"""
        if phsel == phsel_all:
            yield from (istop-istart for istart, istop 
                        in zip(self.iter_column('istart'), self.iter_column('istop')))
        else:
            yield from (mask.sum() for mask in self.iter_column('ph_mask', phsel))

    @classmethod
    def _get_nph_raw_title(cls, col:Column, include_unit:Real=False, origin:PhotonData=None)->str:
        """Title getter function for nph_raw"""
        title = _title_sels('_{raw}n', origin, col.keytup[0])[0]
        title = _title_unit_append(title, 'cnts', include_unit)
        return f'${title}$'

    def _get_ratio_raw(self, phsel_num:PhSel, phsel_dem:PhSel)->np.ndarray[np.float64]:
        """Getter function for ratio_raw column"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self['nph_raw', phsel_num] / self['nph_raw', phsel_dem]
        return out

    @classmethod
    def _get_ratio_raw_title(cls, col:Column, include_unit:Real=False, origin:PhotonData=None)->str:
        """Title getter function for ratio_raw column"""
        return '$%s/%s$' % _title_sels('_{raw}n', origin, *col.keytup[:2])

    @classmethod
    def _replace_E_raw(cls, col:str, keytup:tuple)->tuple:
        """Column map function E_raw->ratio_raw"""
        return 'ratio_raw', (PhSel('0ex1em'), PhSel('0ex'),)+keytup, {'title':'E_{raw}'}

    @classmethod
    def _replace_S_raw(cls, col:str, keytup:tuple)->tuple:
        return 'ratio_raw', (PhSel('0ex'), PhSel('0ex_1ex1em'),)+keytup, {'title':'S_{raw}'}

    def _get_anisotropy_raw(self, phsel_p:PhSel, phsel_s:PhSel)->np.ndarray[np.float64]:
        """Getter function for anisotropy_raw column"""
        p, s = self['nph_raw', phsel_p], self['nph_raw', phsel_s]
        with np.errstate(divide='ignore'):
            out = (p-s)/(p+2*s)
        return out

    @classmethod
    def _get_anisotropy_raw_title(cls, col:Column, include_unit:Real=False, origin:PhotonData=None)->str:
        """Title getter function for anisotropy_raw column"""
        kw = {'name':'_{raw}I'}
        par, perp, start, stop = col.keytup
        fuse = par | perp
        overlap = par | perp
        detdef = None
        if origin is not None:
            kw.update(detdef=origin.detdef, stream_names=origin.get_stream_names())
            fuse = fuse.render_positive(origin.detdef, convert_all=True)
            overlap = overlap.render_positive(origin.detdef, convert_all=True)
            detdef = origin.detdef
        if not overlap and _pol_ps(fuse, detdef, None if origin is None else origin.setup):
            kw['name'] = 'r'
            title = fuse.tex_str(kw)
        else:
            title = rf'anis({par.tex_str(**kw)},\: {perp.tex_str(**kw)})'
        return title

    def _iter_meanT(self, ph_sel:PhSel)->Iterator[float]:
        """Iterator function for meanT column, mean time of given photon stream"""
        for time, s in zip(self.iter_column('ph_times', ph_sel), self.iter_column('istart')):
            yield (np.mean(time-s)+s)*self.origin.clk_p if time.size else np.nan

    @classmethod
    def _get_meanT_title(cls, col:Column, include_unit:Real=False, origin:PhotonData=None)->str:
        """Title getter function for meanT columns"""
        title = _title_sels('_{raw}n', origin, *col.keytup)
        title = _title_unit_append(title, 's', include_unit)
        return f'${title}$'

    def _iter_mTdiff(self, phsel_a:PhSel, phsel_b:PhSel)->Iterator[float]:
        """Iterator function for mTdiff, difference in s in mean time between phsel_a and phsel_b"""
        for timea, timeb, s in zip(self.iter_column('ph_times', phsel_a), self.iter_column('ph_times', phsel_b), self.iter_column('istart')):
            yield (np.mean(timea-s) - np.mean(timeb-s))*self.origin.clk_p if timea.size and timeb.size else np.nan

    @classmethod
    def _get_mTdiff_title(cls, col:Column, include_unit:Real=False, origin:PhotonData=None)->str:
        """Title getter function for column mTdiff"""
        ta, tb = _title_sels(r'\bar t', origin, *col.keytup[:2])
        title = _title_unit_append(f'{ta}-{tb}', 's', include_unit)
        return f'${title}$'

    @classmethod
    def _regularizecolumn_brightness(cls, source_param:Param, *args):
        """Column regularizetion function for brightness column"""
        return args[:1] + cls._regularize_column_startstop(source_param, *args[1:])

    def _get_brightness(self, phsel:PhSel, starttype:ColKeyStart, stoptype:ColKeyStop)->np.ndarray[np.double]:
        """Getter function for brightness column"""
        return self['nph_raw', phsel] / self['dur', starttype, stoptype]

    @classmethod
    def _get_brightness_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """Title getter function for brightness column"""
        title = _title_sels('_{raw}br', origin, col.keytup[0])[0]
        title = _title_unit_append(title, 'cnts s^{-1}', include_unit)
        return f'${title}$'

    @classmethod
    def _regularize_column_startstop(cls, source_param:Param, *args:str)->tuple[str,str]:
        """Sub function for regularizing start/stop times of columns using said keys"""
        return _regularize_column_startstop(source_param, *args)

    @classmethod
    def _regularizecolumn_middur(cls, source_param:Param, *args:str)->tuple[str, str]:
        """regularization function for midtime and dur columns"""
        return cls._regularize_column_startstop(source_param, *args)

    def _get_dur(self, starttype:ColKeyStart, stoptype:ColKeyStop)->np.ndarray[np.float64]:
        """Getter function for dur column"""
        return self.origin.clk_p*(self[stoptype,]-self[starttype,])

    @classmethod
    def _title_startstop_append(cls, name:str, start:ColKeyStart, stop:ColKeyStop)->str:
        """Sub-function for appending start/stop type to column titles"""
        return _title_startstop_append(name, start, stop)

    @classmethod
    def _get_dur_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """Title getter function for column dur"""
        title = cls._title_startstop_append('duration', col.keytup[0], col.keytup[1])
        title = _title_unit_append(title, 's', include_unit)
        return f'${title}$'

    @classmethod
    def _regularizecolumn_sep(cls, source_param:Param, *args:str)->tuple[str, str]:
        """Column regularization function for sep column"""
        if len(args) > 1 and not isinstance(args[-2], str):
            args, post = args[:-2], args[-2:]
        elif len(args) > 0 and  not isinstance(args[-1], str):
            args, post = args[:-1], args[-1:]
        else:
            post = tuple()
        return cls._regularize_column_startstop(source_param, *args) + post

    def _get_sep(self, starttype:ColKeyStart, stoptype:ColKeyStop)->np.ndarray[np.float64]:
        """Getter function for sep column"""
        return self.origin.clk_p*(self[starttype,][1:]-self[stoptype,][:-1])

    @classmethod
    def _get_sep_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """Title getter function for sep column"""
        sep = ''
        if 'offset' in col:
            sep = r'bwd\:' if col.offset else r'fwd\:'
        sep += 'separation'
        title = cls._title_startstop_append(sep, col.keytup[0], col.keytup[1])
        title =  _title_unit_append(title, 's', include_unit)
        return f'${title}$'

    @classmethod
    def _regularizecolumn_max_rate(self, source_param:Param, *args)->tuple[PhSel, int]:
        """Column regularization function for max_rate column"""
        phsel, m = args[0:1], args[1:2]
        phsel = phsel_all if not phsel else phsel[0]
        m = 10 if not m else m[0]
        try:
            m = int(m)
        except Exception as e:
            raise TypeError(f'{type(m)} cannot be interpreted as an int') from e
        if m < 2:
            raise ValueError('m must be 2 or greater')
        return phsel, m

    @cite('IngargiolaPLOSOne2016')
    def _get_max_rate(self, phsel:PhSel, m:int)->np.ndarray[np.float64]:
        """Getter function for max_rate column"""
        stream_ids = self.origin.detdef.get_stream_ids(phsel)
        return smc.maximum_rate(self.origin.times, self.origin.dets, 
                                self['istart',], self['istop',], 
                                self.origin.clk_p, stream_ids, m=m)

    @classmethod
    def _get_max_rate_title(cls, col:Column, include_unit:bool=True, origin:PhotonData=None)->str:
        """Title getter function for max_rate column"""
        title = _title_sels(r'peak\: rate _{%d}r' % (col.keytup[1],), origin, col.keytup[0])[0]
        title = _title_unit_append(title, 'cnts s^{-1}', include_unit)
        return f'${title}$'

    @classmethod
    def _regularizecolumn_bva(cls, source_param:Param, *args)->tuple[PhSel, int]:
        """Column regularization function for bva column"""
        phsel_num, phsel_dem, n = args[0:1], args[1:2], args[2:3]
        phsel_num = PhSel('0ex0em') if not phsel_num else phsel_num[0]
        phsel_dem = PhSel('0ex') if not phsel_dem else phsel_dem[0]
        if phsel_num not in phsel_dem:
            phsel_dem = phsel_num | phsel_dem
        n = 10 if not n else n[0]
        try:
            n = int(n)
        except Exception as e:
            raise TypeError(f'{type(n)} cannot be interpreted as an int') from e
        if n < 2:
            raise ValueError('n must be 2 or greater')
        return phsel_num, phsel_dem, n

    @cite('TorellaBioPhyJ2011', purpose='Burst Variance Analysis')
    def _get_bva(self, phsel_num:PhSel, phsel_dem:PhSel, n:int)->np.ndarray[np.float64]:
        """Getter function for bva column"""
        stream_idsSub = self.origin.detdef.get_stream_ids(phsel_num)
        stream_idsAll = self.origin.detdef.get_stream_ids(phsel_dem)
        return smc.burst_variance_analysis(self.origin.dets, 
                                           self['istart',], self['istop'], 
                                           stream_idsAll, stream_idsSub, n=n)

    @classmethod
    def _get_bva_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """Title getter function for bva column"""
        num, dem = _title_sels('n', origin, *col.keytup[:2])
        return r'$_{%d}\sigma_{%s/%s}$' % (col.keytup[2], num, dem)

    @cite('TorellaBioPhyJ2011', purpose='Burst Variance Analysis')
    def _get_ebva(self, phsel_num:PhSel, phsel_dem:PhSel, n:int)->np.ndarray[np.float64]:
        """Getter function for ebva column"""
        bva, r = self['bva', phsel_num, phsel_dem, n], self['ratio_raw', phsel_num, phsel_dem]
        with np.errstate(divide='ignore'):
            out = bva**2 - ((r*(1-r))/n)
        return out

    @classmethod
    def _get_ebva_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """Title getter function for ebva column"""
        num, dem = _title_sels('n', origin, *col.keytup[:2])
        return r'$_{%d,\: excess}\sigma_{%s/%s}$' % (col.keytup[2], num, dem)

    def _iter_nanomean(self, phsel:PhSel)->float:
        """Iter function for nanomean column"""
        phsel = phsel.render_positive(self.origin.detdef, convert_all=True)
        stream_ids = self.origin.detdef.get_stream_ids(phsel)
        if stream_ids.size == 1:
            tcspc_unit = self.origin.setup['tcspc_unit'][stream_ids[0] % self.origin.detdef.ex_stride]
            thresh = self.origin.irf_thresh[phsel]
            for nanos in self.iter_column('ph_nanos', phsel):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = np.mean((nanos[nanos>=thresh]-thresh)*tcspc_unit)
                yield out
        else:
            thresh_dict = {i:self.origin.irf_thresh[self.origin.detdef.stream_ids_to_PhSel(i)] for i in stream_ids}
            threshs = np.array([thresh_dict.get(i, 0) for i in range(self.origin.detdef.size)])
            exstride = self.origin.detdef.ex_stride
            tcspc_unit_ref = self.origin.setup['tcspc_unit']
            for nanos, dets in zip(self.iter_column('ph_nanos', phsel), self.iter_column('ph_dets', phsel)):
                offset = threshs[dets]
                mask = nanos >= offset
                tcspc_unit = tcspc_unit_ref[dets % exstride]
                nano_off = nanos[mask] - offset[mask]
                yield np.mean((nano_off)*tcspc_unit) if nano_off.size else np.nan

    @classmethod
    def _get_nanomean_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """Title getter function for nanomean"""
        title = _title_sels(r'\bar \tau', origin, col.keytup[0])[0]
        title = _title_unit_append(title, 's', include_unit)
        return f'${title}$'

    def _iter_nanohist(self, phsel:PhSel, full:bool)->Iterator[np.uint16]:
        """Iter function for nanohist column"""
        if not phsel.positive:
            phsel = phsel.render_positive(self.origin.detdef, convert_all=False)
        if full:
            mn = 0
            ln = np.max(self.origin.setup.tcspc_num_bins)
        else:
            ex_ranges = np.concatenate([self.origin.setup.ex_ranges[i] for i in phsel.ex.elements])
            mn, mx = np.min(ex_ranges), np.max(ex_ranges)
            ln = mx - mn
        for nanos in self.iter_column('ph_nanos',phsel):
            yield np.bincount(nanos-mn, minlength=ln)

    @classmethod
    def _regularizecolumn_nanohist(self, source_param:Param, *args)->tuple[PhSel, bool]:
        """Column regularization function for nanohist column"""
        phsel, fill, err = args[:1], args[1:2], args[2:]
        if err:
            raise TypeError("too many keys for nanohist, maximumn two, PhSel and full, (full optional)")
        if not fill:
            fill = (False, )
        return phsel + fill


_basetimecolumndefs = (
    ColumnDef('start', tuple(), 0, 'all', dtype=np.int64, title='start', unit='clk_p'), 
    ColumnDef('stop', tuple(), 0, 'all', dtype=np.int64, title='stop', unit='clk_p'),
    ColumnDef('istart', tuple(), 0, 'all', dtype=np.int64, title='istart'), 
    ColumnDef('istop', tuple(), 0, 'all', dtype=np.int64, title='istop'), 
    ColumnDef('istarttime', tuple(), 0, 'never', dtype=np.dtype('<i8'), 
              get_func='_get_istarttime', get_derived=True, unit='(clk_p)'),
    ColumnDef('istoptime', tuple(), 0, 'never', dtype=np.dtype('<i8'), 
              get_func='_get_istoptime', get_derived=True, unit='(clk_p)'),
    ColumnDef('ph_mask', (PhSel, ), 0, 'never', iter_func='_iter_ph_mask', 
              get_derived=True, dtype=np.object_, typedef=np.dtype(np.bool_)),
    ColumnDef('ph_times', (PhSel, ), 0, 'never', iter_func='_iter_ph_times', 
              get_derived=True, dtype=np.object_, typedef=np.dtype('<i8')),
    ColumnDef('ph_nanos', (PhSel, ), 0, 'never', iter_func='_iter_ph_nanos', 
              get_derived=True, dtype=np.object_, typedef=np.dtype('<u2')),
    ColumnDef('ph_dets', (PhSel, ), 0, 'never', iter_func='_iter_ph_dets', 
              get_derived=True, dtype=np.object_, typedef=np.dtype('<u1')), 
    ColumnDef('ph_particles', (PhSel, ), 0, 'never', iter_func='_iter_ph_particles', 
              get_derived=True, dtype=np.object_, typedef=np.dtype('<u1')),
    ColumnDef('nph_raw', (PhSel, ), 0, 'user', dtype=np.dtype('<i8'), iter_func='_iter_nph_raw', 
              get_derived=True, title_func='_get_nph_raw_title', unit='cnts', 
              index_unit='cnts'),
    ColumnDef('ratio_raw', (PhSel, PhSel), 0, 'user', dtype=np.dtype('<f8'), 
              get_func='_get_ratio_raw', get_derived=True, 
              title_func='_get_ratio_raw_title'),
    ColumnDef('anisotropy_raw', (PhSel, PhSel), 0, 'user', dtype=np.dtype('<f8'),
              get_func='_get_anisotropy_raw', get_derived=True,
              title_func='_get_anisotropy_raw_title'),
    ColumnDef('meanT', (PhSel, ), 0, 'user', iter_func='_iter_meanT', get_derived=True,
              title_func='_get_meanT_title', unit='s'),
    ColumnDef('mTdiff', (PhSel, PhSel), 0, 'user', iter_func='_iter_mTdiff', get_derived=True,
              title_func='_get_mTdiff_title', unit='s'),
    ColumnDef('max_rate', (PhSel, int), 0, 'user', get_func='_get_max_rate',
              dtype=np.dtype('<f8'), get_derived=True, reg_func='_regularizecolumn_max_rate',
              title_func='_get_max_rate_title', unit=r'cnts\: s^{-1}', index_unit='cnts s-1'),
    ColumnDef('bva', (PhSel, PhSel, int), 0, 'user', get_func='_get_bva', 
              dtype=np.dtype('<f8'), get_derived=True, reg_func='_regularizecolumn_bva',
              title_func='_get_bva_title'),
    ColumnDef('ebva', (PhSel, PhSel, int), 0, 'user', get_func='_get_ebva', 
              dtype=np.dtype('<f8'), get_derived=True, reg_func='_regularizecolumn_bva',
              title_func='_get_ebva_title'),
    ColumnDef('nanohist', (PhSel, bool), 0, 'never', iter_func='_iter_nanohist',
              reg_func='_regularizecolumn_nanohist',
              get_derived=True, dtype=np.dtype('<i8'), ndim=2),
    ColumnDef('nanomean', (PhSel, ), 0, 'user', iter_func='_iter_nanomean', get_derived=True,
              dtype=np.dtype('<f8'), title_func='_get_nanomean_title', unit='s'),
    ColumnDef('E_raw', tuple(), 0, remap='_replace_E_raw'),
    ColumnDef('S_raw', tuple(), 0, remap='_replace_S_raw'),
                  )


def make_base_column_defs(startV:TypeValidator=TV_str_start, stopV:TypeValidator=TV_str_stop, skip:Sequence[str]=None):
    skip = tuple() if skip is None else skip
    out = _basetimecolumndefs + (
        ColumnDef('midtime', (startV, stopV), 0, 'never', get_func='_get_midtime', 
                  get_derived=True, reg_func='_regularizecolumn_middur', 
                  title_func='_get_midtime_title', unit='(s)'),
        ColumnDef('sep', (startV, stopV), -1, 'never', get_func='_get_sep', atomic=False, 
                  dtype=np.dtype('<f8'), reg_func='_regularizecolumn_sep', 
                  title_func='_get_sep_title', unit='s', index='sep', index_unit='s'),
        ColumnDef('brightness', (PhSel, startV, stopV), 0, 'user', dtype=np.dtype('<f8'), 
                  get_func='_get_brightness', get_derived=True, 
                  reg_func='_regularizecolumn_brightness', title_func='_get_brightness_title',
                  unit=r'cnts\: s^{-1}', index_unit='cnts s-1'),
        ColumnDef('dur', (startV, stopV), 0, 'never', dtype=np.dtype('<f8'), get_func='_get_dur', 
                  get_derived=True, reg_func='_regularizecolumn_middur', 
                  title_func='_get_dur_title', unit='s', index='dur', index_unit='s')
        )
    return tuple(o for o in out if o.name not in skip)


class ChildPhotonTable(PhotonTable, ChildTable):
    """
    Class for :class:`smfbursts.datamodel.ChildTable` which has a :class:`PhotonData`
    origin, and :class:`BasePhotonTable` as their base table.
    
    Subclasses need only define the ``param_defs``, ``parent_defs``, and
    ``column_defs`` class attributes, and necessary accompanying methods
    to compute columns. If the table should compute any columns upon calling
    :meth:`smfbursts.datamodel.tables.DataSet.get_table`, these should be 
    instantiated in the ``__init_columns__`` method.
    
    This class defines the ``detdef`` paramproperty to obtain the 
    :class:`smfbursts.ph_sel.DetDef` of the param.
    """
    _origin: PhotonData
    _base_table:BasePhotonTable

    def _init_new_(self):
        if self.param.detdef != self.origin.detdef:
            raise ValueError("mismatched detdefs")
        super()._init_new_()

    @paramproperty
    def detdef(cls, param:Param)->DetDef:
        """DetDef of param, shortcut to access fields about detdef in subclasses"""
        return param.base_param.detdef
    
    @paramproperty
    def _colstarttypes(cls, param:Param)->tuple[str,...]:
        return param.base_param._colstarttypes
    
    @paramproperty
    def _colstoptypes(cls, param:Param)->tuple[str,...]:
        return param.base_param._colstoptypes
    
    @paramproperty
    def _colstartdefaultstr(cls, param:Param)->str:
        return param.base_param._colstartdefaultstr
    
    @paramproperty
    def _colstopdefaultstr(cls, param:Param)->str:
        return param.base_param._colstopdefaultstr
    
    @classmethod
    def _regularize_column_startstop(cls, source_param:Param, *args)->tuple[str,str]:
        return _regularize_column_startstop(source_param.base_param, *args)


def as_irf(data:PhotonData)->dict[PhSel:np.ndarray[np.int64]]:
    """
    Convert a :class:`PhotonData` into an IRF dictionar, usable as :attr:`PhotonData.irf`
    in another :class:`PhotonData` object.

    Parameters
    ----------
    data : PhotonData
        PhotonData to be turned into IRF, should be of scatter or quenched fluorephore.

    Returns
    -------
    irf : dict[PhSel:np.ndarray[np.int64]]
        Dictionary of photon-stream:irf histogram key value pairs.

    """
    irf = dict()
    for i in range(data._pharray.detdef.size):
        phsel = data.detdef.stream_ids_to_PhSel(i)
        mn, mx = get_phsel_ex_range(data.setup, phsel)
        irf[phsel] = np.bincount(data.nanos[data.dets==i]-mn, minlength=mx-mn)
    return irf