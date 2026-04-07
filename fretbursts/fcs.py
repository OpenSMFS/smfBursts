#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created: 2026-03-24
# Author: Paul David Harris
# FCS module for pySMFS
"""
Functions for computing various FCS-related correlations.
"""
from typing import Literal
from collections.abc import Sequence
from numbers import Integral, Real
from itertools import repeat, chain

import numpy as np

import pyFCS as fcs

from .photondata import (
    PhotonData, PhotonDataList, PhotonDataS
    )

import fretbursts.cfuncs as fbc
from fretbursts.datamodel.utils import fjit, fnumba
from fretbursts.datamodel.tables import Param, Column, GateGroup
from fretbursts.photondata import PhotonData, PhotonDataList, PhotonDataS
from fretbursts import PhSel
from .cite import cite


def _concatenate_arrays(array):
    """Converts array or list of arrays to concatenated array"""
    if not np.issubdtype(type(array[0]), np.number):
        array = np.concatenate(array)
    return np.asarray(array)


def _norm_fcs_bins(bins:None|Real|np.ndarray, data:PhotonData, max_time:float=1.0)->np.ndarray[np.int64]:
    """Process bins argument of any correlation function."""
    if not isinstance(bins, np.ndarray):
        mx = np.log(max_time/data.clk_p)/np.log(10)
        fac = 5 if bins is None else bins
        ibins = np.logspace(np.log(2), mx, int(fac*mx-np.log(2)))
    elif np.issubdtype(bins.dtype, np.floating):
        ibins = bins/data.clk_p
    ibins = ibins.astype(np.int64)
    m = np.empty(ibins.shape, dtype=np.bool_)
    m[0] = True
    m[1:] = ibins[:-1] != ibins[1:]
    ibins = ibins[m]
    if isinstance(bins, np.ndarray):
        if np.any(m):
            bins = bins[m]
    else:
        bins = ibins * data.clk_p
    return bins, ibins


@cite('LaurenceOptLett2006', purpose='FCS algorithm')
def correlate(data:PhotonDataS, streamT:PhSel=PhSel('all'), streamU:PhSel=None, 
              bins:None|Real|np.ndarray=None)->tuple[np.ndarray[np.float64],np.ndarray[np.float64]]:
    r"""
    Compute the correlation between streamT and streamU in data.
    
    The calculation is as follows\:

    :math:`\hat{C}(\tau_{b}) = \sum_{k=1}^{K}{n(\{(i, j) \ni \tau_{b} \le u_{k,j} - t_{k,i} < \tau_{b+1})\})}`
    
    Where 

    Parameters
    ----------
    data : PhotonDataS
        Processed data on which to compute correlation.
    streamT : PhSel, optional
        Stream to correlate from. The default is PhSel('all').
    streamU : PhSel, optional
        Stream to correlate to, if None compute autocorrelation of streamT. The default is None.
    bins : None|Real|np.ndarray, optional
        Definition for bins if array. If the array is integral value, the units
        are clock rate, while if floating, bins unit will be seconds.
        If a real number value, the number of logspaced bins per order of magnitude
        (decade), starting from clock rate to 1 second.
        None is equivalent to 5. The default is None.

    Returns
    -------
    corrl : np.ndarray[np.float64]
        Auto/cross Correlation of data.
    bins : np.ndarray[np.float64]
        Time bins of correlation (in seconds).
    
    """
    bins, ibins = _norm_fcs_bins(bins, data)
    streamB = streamT if streamU is None else streamU
    corr = fcs.correlate(data.get_times(streamT), data.get_times(streamB), ibins)
    return corr, bins


def _get_nanorange(data:PhotonData, sel:PhSel)->tuple[np.ndarray, np.ndarray]:
    """Generate nanorange bins of data"""
    data = data if isinstance(data, PhotonData) else data.datas[0]
    detdef = data.detdef
    idxs = detdef.get_stream_ids(sel)
    windows = np.concatenate(data.setup.ex_ranges[idxs // detdef.ex_stride]).astype(np.uint16)
    if windows.shape[0] != idxs.shape[0]:
        raise ValueError("split excitation windows not supported")
    return idxs, windows


def _make_nanomap_arrays(data:PhotonData, stream:PhSel, nanobin:int, nanorange:None|np.ndarray[np.uint16]=None):
    """Generate arrays for maping/sorting nanotimes"""
    idxs, nrange = _get_nanorange(data, stream)
    nanomask = False
    if nanorange is not None:
        if nanorange.shape != nrange.shape:
            raise ValueError(f"Mismatched nanorange shape, expected {nrange.shape}, got {nanorange.shape}")
        if np.any(nanorange[:,0] < nrange[:,0]) or np.any(nrange[:,1] < nanorange[:,1]):
            raise ValueError("nanorange contains values outside excitation ranges")
        nanorange = np.any(nanorange != nanomask)
    else:
        nanorange = nrange
    nanomin = nanorange[:,0]
    nanoshift = np.zeros(idxs.size, dtype=np.uint16)
    nanoshift[1:] = np.cumsum(((np.diff(nanorange[:-1,:], axis=1)[:,0] - 1) // nanobin) + 1)
    nanomax = nanorange[:,1] if nanomask else repeat(None)
    return nanoshift, nanomin, nanomax


def _timesnanossort(data:PhotonData, stream:PhSel, sort_nanos:bool=False, nanobin:int=1, nanorange:np.ndarray[np.uint16]=None):
    """Retreive times in stream, and if sort_nanos is true, retrieve and remap/bin nanotimes"""
    # sort times, always necessary
    times, dets = data.times, data.dets
    idxs = data.detdef.get_stream_ids(stream)
    mask = np.isin(dets, idxs)
    times = times[mask]
    if sort_nanos is False:
        return times, None
    # get arrays for mapping nanotimes
    nanoshift, nanomin, nanomax = _make_nanomap_arrays(data, stream, nanobin, nanorange)
    nanos = data.nanos
    dets, nanos = dets[mask], nanos[mask]
    # sort each detector, must do individually to acount for partially filled final nanotime bin
    for d, s, mn, mx in zip(idxs, nanoshift, nanomin, nanomax):
        dmask = d == dets
        if mx is not None:
            rmask = (mn <= nanos) | (nanos < mx)
            mask = ~dmask | ~rmask
            times, dets, nanos, dmask = times[mask], dets[mask], nanos[mask], dmask[mask]
        nanos[dmask] -= mn
        nanos[dmask] //= nanobin
        nanos[dmask] += s
    return times, nanos


def _comp_nanorange(rA:None|np.ndarray, rB:None|np.ndarray)->bool:
    """Check two nanoranges are compatible"""
    if rA is None:
        return rB is None
    if rB is None:
        return False
    return rA.shape == rB.shape and np.all(rA == rB)


@cite('LaurenceBiophysJ2007', purpose='burst purification of FCS data')
def _expand_bursts(data:PhotonData, bursts:Param, streamT:PhSel=PhSel('all'), streamU:PhSel=PhSel('all'), 
                   expand:float=0.1, fuse:float=0.0, sort_nanos:bool=False, 
                   nanoTbin:int=1, nanoTrng:np.ndarray[np.int16]=None, 
                   nanoUbin:int=1, nanoUrng:np.ndarray[np.int16]=None
                  )->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Retrieve burst-expanded arrays and start/stops from PhotonData NOTE: must be PhotonData, not PhotonDataList"""
    # ensure nanoA/Brng are either None or 2d arrays
    nanoTrng = None if nanoTrng is None else np.atleast_2d(nanoTrng)
    nanoUrng = None if nanoUrng is None else np.atleast_2d(nanoUrng)
    # get times/nanos of whole data set
    timesT, nanosT = _timesnanossort(data, streamT, sort_nanos, nanoTbin, nanoTrng)
    # peform burst expansion
    expand, fuse = int(expand/data.clk_p), int(fuse/data.clk_p)
    btable = data.get_table(bursts)
    start, stop = fbc.fusebursts(btable['start']-expand, btable['stop']+expand, fuse)
    start[start < 0] = 0
    istart, istop = fbc.index_ranges(timesT, start, stop)
    # sort times into expanded bursts
    timesT = tuple(timesT[b:e] for b, e in zip(istart, istop))
    nanosT = tuple(nanosT[b:e] for b, e in zip(istart, istop)) if sort_nanos else None
    # process streamB separtely, 
    if streamT != streamU or nanoTbin != nanoUbin or not _comp_nanorange(nanoTrng, nanoUrng):
        timesU, nanosU = _timesnanossort(data, streamU, sort_nanos, nanoUbin, nanoUrng)
        istart, istop = fbc.index_ranges(timesU, start, stop)
        timesU = tuple(timesU[b:e] for b, e in zip(istart, istop))
        nanosU = tuple(nanosU[b:e] for b, e in zip(istart, istop)) if sort_nanos else None
    else:
        timesU, nanosU = timesT, nanosT
    return start, stop, timesT, timesU, nanosT, nanosU


def purified_fcs(data:PhotonDataS, bursts:Param, bins:None|Real|np.ndarray=None, gate:GateGroup=None,
                 streamT:PhSel=PhSel('all'), streamU:PhSel=None,
                 expand:float=0.1, fuse:float=0.0)->tuple[np.ndarray,np.ndarray]:
    """
    Compute correlation between streamT and streamU of data filtered by bursts.
    See `Laurence et. al. 2007 <https://doi.org/10.1529/biophysj.106.093591>`_

    Parameters
    ----------
    data : PhotonDataS
        Processed data on which to compute correlation.
    bursts : Param
        BasePhontonTable based param defining time ranges over which to correlate.
    bins : None|Real|np.ndarray, optional
        Definition for bins if array. If the array is integral value, the units
        are clock rate, while if floating, bins unit will be seconds.
        If a real number value, the number of logspaced bins per order of magnitude
        (decade), starting from clock rate to longest duration of burst.
        None is equivalent to 5. The default is None.
    gate : GateGroup, optional
        Gate to apply to bursts. The default is None.
    streamT : PhSel, optional
        Stream to correlate from. The default is PhSel('all').
    streamU : PhSel, optional
        Stream to correlate to, if None compute autocorrelation of streamT. The default is None.
    expand : float, optional
        Time (in seconds) around bursts by to exand time ranges to correlate. 
        The default is 0.1.
    fuse : float, optional
        If two time ranges are separted by less than this, ranges are fused. 
        The default is 0.0.

    Returns
    -------
    corrl : np.ndarray[np.float64]
        Auto/cross Correlation of data.
    bins : np.ndarray[np.float64]
        Time bins of correlation (in seconds).

    """
    streamU = streamT if streamU is None else streamU
    bins, ibins = _norm_fcs_bins(bins, data, expand)
    if gate is not None:
        bursts = bursts.regate(gate)
    if isinstance(data, PhotonData):
        b, e, tT, tU, _, _ = _expand_bursts(data, bursts, streamT, streamU, expand, fuse)
    elif isinstance(data, PhotonDataList):
        b, e, tT, tU, _, _ = zip(*(_expand_bursts(d, bursts, streamT, streamU, expand, fuse, False, False) for d in data.datas))
        b = np.concatenate(b)
        e = np.concatenate(e)
        if tU is tT:
            tT = tuple(chain(*tT))
            tU = tT
        else:
            tT = tuple(chain(*tT))
            tU = tuple(chain(*tU))
    edges = np.vstack([b, e]).T
    return fcs.correlate(tT, tU, ibins, edges=edges), bins


@cite('flcs', purpose='compute weights for FLCS')
def _compute_weights(nanos:np.ndarray[np.int64], M:np.ndarray[np.float64])->np.ndarray[np.float64]:
    """
    Compute the orthonormal set of weights arrays given the nanotimaes and the decays.
    
    Parameters
    ----------
    nanos: np.ndarray
        The array of nanotimes used in the data. Note: to count the number of instances
        of each nanotime, so input the data used in correlation.
    M: np.ndarray
        The decay histograms of each decay. Note that since the output is
        an orthonormal basis set, adding decays changes all
        previous outputs.
    
    Returns
    -------
    U: np.ndarray
        2D array of weights arranged [decayN, tcspc_bin], so the vectors
        from U[0,:], U[1,:] ... form an orthonormal basis set
    """
    # so that lists of arrays can be inputed directly
    nanos = _concatenate_arrays(nanos)
    # normalize decays and form into matrix M
    M = M / M.sum(axis=1)[:,np.newaxis]
    # compute <I_{i}>
    nanodist = np.bincount(nanos, minlength=M.shape[1])
    # account for 0 occupancy bins
    mask = nanodist != 0
    Mn = M[:,mask]
    dIinv = 1/nanodist[mask]
    # final computation of [M diag<I>^{-1} M^{T}]^{-1} M diag<I> 
    Mi = Mn*dIinv # M diag<I>^{-1}
    U = np.linalg.inv(Mi@Mn.T) @ Mi
    # re-expand U
    newU = np.zeros(M.shape)
    newU[:,mask] = U
    return U


def compute_weights(nanos:np.ndarray[np.uint16], *args)->np.ndarray[np.float64]:
    r"""
    Compute the orthonormal set of weights arrays given the nanotimaes and the decays.
    
    Parameters
    ----------
    nanos: np.ndarray
        The array of nanotimes used in the data. Note: to count the number of instances
        of each nanotime, so input the data used in correlation.
    *args: np.ndarray
        The decay histograms of each decay. Note that since the output is
        an orthonormal basis set, adding additional arguments changes all
        previous outputs, ie ``compute_weights(nanos, decay1, decay2)``
        will give different weights for decay1 and decay2 compared with
        compute_weights(nanos, decay1, decay2, decay3). Must supply at least
        2 decays
    
    Returns
    -------
    U: np.ndarray
        2D array of weights arranged [decayN, tcspc_bin], so the vectors
        from ``U[0,:], U[1,:] ...`` form an orthonormal basis set
    """
    M = np.asarray(args, dtype=np.float64)
    return _compute_weights(nanos, M)


def get_weights(data:PhotonData, decays:Sequence[np.ndarray[np.float64]], streams:PhSel=PhSel('all'),
                 bursts=None,expand:float=0.1, fuse:float=0.0,
                 nanorange:tuple[int,int]|tuple[tuple[int,int],...]=None, nanobin:int=1,
                 )->tuple[np.ndarray[np.float64],np.ndarray[np.float64]]:
    if bursts is None:
        if isinstance(data, PhotonData):
            times, nanos = _timesnanossort(data, streams, True, nanobin, nanorange)
        else:
            times, nanos = zip(*(_timesnanossort(d, streams, True, nanobin, nanorange) 
                                 for d in data.datas))
    else:
        if isinstance(data, PhotonData):
            temp = _expand_bursts(data, bursts, streams, streams, expand, fuse, 
                                  True, nanobin, nanorange, nanobin, nanorange)
            _, _, times, _, nanos, _ = temp
        else:
            temp = zip(*(_expand_bursts(d, bursts, streams, streams, expand, fuse, 
                                        True, nanobin, nanorange, nanobin, nanorange) 
                         for d in data.datas))
            _, _, times, _, nanos, _ = temp
            times = tuple(chain(*times))
            nanos = tuple(chain(*nanos))
    decays = np.atleast_2d(decays)
    weights = _compute_weights(nanos, decays)
    return weights


CorrSpec = Literal['auto','all', 'cross']|tuple[int,int]|Sequence[tuple[int,int]]


@cite('flcs', purpose='FLCS analysis')
def flcs(data:PhotonDataS, decays:Sequence[np.ndarray[np.float64]], 
         streams:PhSel=PhSel('all'), bins:None|Real|np.ndarray=None, bursts:Param=None, 
         gate:GateGroup=None, correlations:CorrSpec='cross', expand:float=0.1, fuse:float=0.0,
         nanobin:int=1, nanorange:tuple[int,int]|tuple[tuple[int,int],...]=None,
         )->tuple[np.ndarray[np.float64],np.ndarray[np.float64]]:
    """
    Compute fluorescence lifetime correlation(s) based on filters defined by decays
    given in decays argument.

    Parameters
    ----------
    data : PhotonDataS
        Processed data on which to compute correlation.
    decays : Sequence[np.ndarray[np.float64]]
        TCSPC decays (binned, and each det index stacked sequentially). Will
        peform automatic normalization and computation of orthonormal masks
        automatically.
    streams : PhSel, optional
        Photon stream(s) to use in computing correlations. The default is PhSel('all').
    bins : None|Real|np.ndarray, optional
        Definition for bins if array. If the array is integral value, the units
        are clock rate, while if floating, bins unit will be seconds.
        If a real number value, the number of logspaced bins per order of magnitude
        (decade), starting from clock rate to longest duration of burst if bursts
        is specified, or 1 second if bursts is not specified.
        None is equivalent to 5. The default is None.
    bursts : Param
        BasePhontonTable based param defining time ranges over which to correlate.
    gate : GateGroup
        Gate to apply to bursts
    correlations : CorrSpec, optional
        Which correlation(s) to compute. The default is 'cross'.
    expand : float, optional
        Time (in seconds) around bursts by to exand time ranges to correlate. 
        The default is 0.1.
    fuse : float, optional
        If two time ranges are separted by less than this, ranges are fused. 
        The default is 0.0.
    nanobin : int, optional
        Number of TCSPC bins to group together into single weights bin. The default is 1.
    nanorange : tuple[int,int]|tuple[tuple[int,int],...], optional
        Range(s) of raw nanotime per stream, in order of streams in stream based
        on detdef of data. Must be Nx2 dimensional array-like. If None, use whole
        excitation window for each stream. The default is None.

    Raises
    ------
    ValueError
        Bad option specified.

    Returns
    -------
    corrl : np.ndarray[np.float64]
        Auto/cross Correlation of data. Shape will depend on option specified
        in ``correlations`` kwarg
    bins : np.ndarray[np.float64]
        Time bins of correlation (in seconds).

    """
    bins, ibins = _norm_fcs_bins(bins, data)
    edges = None
    if bursts is None:
        bursts = bursts if gate is None else bursts.regate(gate)
        if isinstance(data, PhotonData):
            times, nanos = _timesnanossort(data, streams, True, nanobin, nanorange)
        else:
            times, nanos = zip(*(_timesnanossort(d, streams, True, nanobin, nanorange) 
                                 for d in data.datas))
    else:
        if isinstance(data, PhotonData):
            temp = _expand_bursts(data, bursts, streams, streams, expand, fuse, 
                                  True, nanobin, nanorange, nanobin, nanorange)
            start, stop, times, _, nanos, _ = temp
        else:
            temp = zip(*(_expand_bursts(d, bursts, streams, streams, expand, fuse, 
                                        True, nanobin, nanorange, nanobin, nanorange) 
                         for d in data.datas))
            start, stop, times, _, nanos, _ = temp
            times = tuple(chain(*times))
            nanos = tuple(chain(*nanos))
            start = np.concatenate(start)
            stop = np.concatenate(stop)
        edges = np.vstack([start, stop]).T
    decays = np.atleast_2d(decays)
    weights = _compute_weights(nanos, decays)
    crosscorr = False
    if isinstance(correlations, str):
        if correlations.lower() not in ('auto', 'cross', 'all'):
            raise ValueError("invalid correlation type, must be 'auto', or 'all'")
        crosscorr = correlations.lower() != 'all'
        weightsT = weights
        weightsU = weights
    else:
        correlations = np.asarray(correlations, dtype=np.int64).reshape(-1,2)
        weightsT = weights[correlations[:,0]]
        weightsU = weights[correlations[:,1]]
    corr = fcs.correlate(times, times, ibins, weightsT=weightsT, weightsU=weightsU, nanosT=nanos, nanosU=nanos, cross_correlate=crosscorr, edges=edges)
    return corr, bins


@cite('flcs', purpose='FLCS analysis')
def flcs_weights(data:PhotonDataS, weights:Sequence[np.ndarray[np.float64]],
                 streams:PhSel=PhSel('all'), bins:None|Real|np.ndarray=None, 
                 bursts:Param=None, gate:GateGroup=None, correlations:CorrSpec='cross', 
                 expand:float=0.1, fuse:float=0.0,
                 nanobin:int=1, nanorange:tuple[int,int]|tuple[tuple[int,int],...]=None,
                 )->tuple[np.ndarray[np.float64],np.ndarray[np.float64]]:
    """
    Compute fluorescence lifetime correlation(s) based on filters given in weights
    argument.

    Parameters
    ----------
    data : PhotonDataS
        Processed data on which to compute correlation.
    weights : Sequence[np.ndarray[np.float64]]
        Already computed weights functions (binned, and each det index stacked sequentially).
        Should form proper orthonormal mask.
    streams : PhSel, optional
        Photon stream(s) to use in computing correlations. The default is PhSel('all').
    bins : None|Real|np.ndarray, optional
        Definition for bins if array. If the array is integral value, the units
        are clock rate, while if floating, bins unit will be seconds.
        If a real number value, the number of logspaced bins per order of magnitude
        (decade), starting from clock rate to longest duration of burst if bursts
        is specified, or 1 second if bursts is not specified.
        None is equivalent to 5. The default is None.
    bursts : Param
        BasePhontonTable based param defining time ranges over which to correlate.
    gate : GateGroup
        Gate to apply to bursts
    correlations : CorrSpec, optional
        Which correlation(s) to compute. The default is 'cross'.
    expand : float, optional
        Time (in seconds) around bursts by to exand time ranges to correlate. 
        The default is 0.1.
    fuse : float, optional
        If two time ranges are separted by less than this, ranges are fused. 
        The default is 0.0.
    nanobin : int, optional
        Number of TCSPC bins to group together into single weights bin. The default is 1.
    nanorange : tuple[int,int]|tuple[tuple[int,int],...], optional
        Range(s) of raw nanotime per stream, in order of streams in stream based
        on detdef of data. Must be Nx2 dimensional array-like. If None, use whole
        excitation window for each stream. The default is None.

    Raises
    ------
    ValueError
        Bad option specified.

    Returns
    -------
    corrl : np.ndarray[np.float64]
        Auto/cross Correlation of data. Shape will depend on option specified
        in ``correlations`` kwarg
    bins : np.ndarray[np.float64]
        Time bins of correlation (in seconds).

    """
    bins, ibins = _norm_fcs_bins(bins, data)
    edges = None
    if bursts is None:
        bursts = bursts if gate is None else bursts.regate(gate)
        if isinstance(data, PhotonData):
            times, nanos = _timesnanossort(data, streams, True, nanobin, nanorange)
        else:
            times, nanos = zip(*(_timesnanossort(d, streams, True, nanobin, nanorange) 
                                 for d in data.datas))
    else:
        if isinstance(data, PhotonData):
            temp = _expand_bursts(data, bursts, streams, streams, expand, fuse, 
                                  True, nanobin, nanorange, nanobin, nanorange)
            start, stop, times, _, nanos, _ = temp
        else:
            temp = zip(*(_expand_bursts(d, bursts, streams, streams, expand, fuse, 
                                        True, nanobin, nanorange, nanobin, nanorange) 
                         for d in data.datas))
            start, stop, times, _, nanos, _ = temp
            times = tuple(chain(*times))
            nanos = tuple(chain(*nanos))
            start = np.concatenate(start)
            stop = np.concatenate(stop)
        edges = np.vstack([start, stop]).T
    crosscorr = False
    if isinstance(correlations, str):
        if correlations.lower() not in ('auto', 'cross', 'all'):
            raise ValueError("invalid correlation type, must be 'auto', or 'all'")
        crosscorr = correlations.lower() != 'all'
        weightsT = weights
        weightsU = weights
    else:
        correlations = np.asarray(correlations, dtype=np.int64).reshape(-1,2)
        weightsT = weights[correlations[:,0]]
        weightsU = weights[correlations[:,1]]
    corr = fcs.correlate(times, times, ibins, weightsT=weightsT, weightsU=weightsU, nanosT=nanos, nanosU=nanos, cross_correlate=crosscorr, edges=edges)
    return corr, bins


@fjit(fnumba.int64[:](fnumba.int64[:], fnumba.int64))
def bin_nanohist(nanohist:np.ndarray[np.int64], binsize:int)->np.ndarray[np.int64]:
    """
    Convenience function, bins a nanotime histogram by binsize. Use to create
    binned decays for input in ``decays`` argument of :func:`flcs` .

    Parameters
    ----------
    nanohist : np.ndarray[np.int64]
        Nanotime histogram.
    binsize : int
        Number of consequtive elements to bin into one output bin..

    Returns
    -------
    out : np.ndarray[np.int64]
        binned nanotime histogram.

    """
    if binsize == 1:
        return nanohist
    bsize = ((nanohist.size - 1) // binsize) + 1
    out = np.zeros(bsize, dtype=np.int64)
    for i in range(nanohist.size):
        out[i//binsize] += nanohist[i]
    return out


def _as_slice(slc:Integral|np.ndarray|slice)->slice:
    if isinstance(slc, slice):
        return slc
    if isinstance(slc, Integral):
        return slice(slc, slc+1)
    slc = np.asarray(slc).reshape(-1)
    if slc.size == 1:
        return slice(slc[0], slc[0]+1)
    elif slc.size == 2:
        return slice(slc[0], slc[1])
    raise ValueError(f"specifying range requires specifying either 1 or 2 values, not {slc.size}")


def extract_decays(data:PhotonDataS, streams:PhSel, bursts:Param=None, gate:GateGroup=None, 
                   nanobin:int=1, nanorange:None|np.ndarray[np.uint16]=None, 
                   bg_ranges:dict[PhSel:int|slice]=None, zero_ranges:dict[PhSel:int|slice]=None
                   )->np.ndarray[np.int64]:
    """
    Return concatenated per-stream fluorescence decays for streams from the given
    burst selection from data. If data is not specified, then all photons in data
    are considered. This is primarily used for extracting single "species" decays
    for subsequent analysis with :func:`flcs`
    
    When multiple photon streams compose streams, then each decays is concatenated
    with the order always being in increasing index, with split having smallest
    increment, then polarization, then emission, then excitation. This is the
    same as the streams being in asscending order of their cooresponding detector
    index.

    Parameters
    ----------
    data : PhotonDataS
        Data from which to extrac the decays.
    streams : PhSel
        All streams to include in concatenated decays.
    bursts : Param, optional
        Burst selection defining from which ranges of times to histogram the
        photon nanotimes. If none, entire dataset is included. The default is None.
    gate : GateGroup, optional
        Gate to apply to bursts, if None, use gate already speceified in bursts. 
        The default is None.
    nanobin : int, optional
        Number of consecutive TCSPC bins (nanotimes) to group into single output bin. 
        The default is 1.
    nanorange : None|np.ndarray[np.uint16], optional
        2xN array-like, where N is the nubmer of detector IDs in streams, specified
        in same order as concatentation/ascending order of included detector ids. 
        These specify start:stop ranges of nanotimes to include. If None, use
        the excitation windows defined in :attr:`data.setup.ex_ranges`
        The default is None.
    bg_ranges : dict[PhSel:int|slice], optional
        DESCRIPTION. The default is None.
    zero_ranges : dict[PhSel:int|slice], optional
        DESCRIPTION. The default is None.

    Returns
    -------
    np.ndarray[np.int64]
        Concatenated and binned fluorescent decay histograms.

    """
    idxs, nanoranges = _get_nanorange(data, streams) if isinstance(data, PhotonData) else _get_nanorange(data.datas[0], streams)
    if bursts is None:
        if isinstance(data, PhotonData):
            _, nanos = _timesnanossort(data, streams, True, nanobin, nanorange)
        else:
            _, nanos = zip(*(_timesnanossort(d, streams, True, nanobin, nanorange)
                             for d in data.datas))
            nanos = np.concatenate(nanos)
        nbins = np.sum(((np.diff(nanoranges, axis=1)-1) // nanobin) + 1)
        nanohist = np.bincount(nanos, minlength=nbins)
        if bg_ranges is not None or zero_ranges is not None:
            nanoshift, nanomin, nanomax = _make_nanomap_arrays(data, streams, nanobin)
            nmax = nanoshift[-1] + (((nanoranges[-1,1] - nanoranges[-1,0])-1) // nanobin) + 1
            nanoshift = np.concatenate([nanoshift, [nmax]])
        if bg_ranges is not None:
            if not isinstance(bg_ranges, dict):
                bg_ranges = {data.detdef.stream_ids_to_PhSel(idx):bg_ranges for idx in idxs}
            for idx, mn, mx in zip(idxs, nanoshift[:-1], nanoshift[1:]):
                nrng = bg_ranges.get(data.detdef.stream_ids_to_PhSel(idx), None)
                if nrng is None:
                    continue
                nanohist[mn:mx] -= np.mean(nanohist[mn:mx][_as_slice(nrng)], dtype=np.int64)
            nanohist[nanohist < 0] = 0
        if zero_ranges is not None:
            if not isinstance(zero_ranges, dict):
                zero_ranges = {data.detdef.stream_ids_to_PhSel(idx):zero_ranges for idx in idxs}
            for idx, mn, mx in zip(idxs, nanoshift[:-1], nanoshift[1:]):
                nrng = bg_ranges.get(data.detdef.stream_ids_to_PhSel(idx), None)
                if nrng is None:
                    continue
                nanohist[mn:mx][_as_slice(nrng)] = 0
        return nanohist
    gfunc = data.get_column if isinstance(data, PhotonData) else data.concatenate_column
    bursts = bursts if gate is None else bursts.regate(gate)
    hists = list()
    for idx, rng in zip(idxs, nanoranges):
        sel = data.detdef.stream_ids_to_PhSel(idx)
        col = Column(bursts, 'nanohist', (sel, True))
        hist = bin_nanohist(gfunc(col).sum(axis=0)[rng[0]:rng[1]], nanobin)
        if bg_ranges is not None:
            sp = bg_ranges.get(sel, None) if isinstance(bg_ranges, dict) else bg_ranges
            if sp is not None:
                hist -= np.mean(hist[_as_slice(sp)], dtype=np.int64)
                hist[hist < 0] = 0
        if zero_ranges is not None:
            sp = zero_ranges.get(sel, None) if isinstance(zero_ranges, dict) else zero_ranges
            hist[_as_slice(sp)] = 0
        hists.append(hist)
    return np.concatenate(hists)
        
        