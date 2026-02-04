# -*- coding: utf-8 -*-
# File: plot.py
# Author: Paul Harris
# Created 2/11/2025
# Purpose: plotting photondata param tables
"""
Core plotting functions (based on matplotlib) for FRETBursts, extends
datamodel.plotting with functions that incorporate the time of an event
into the plot (without creating many gates based therein).
"""
from typing import Union, Any
from numbers import Integral, Real

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .datamodel.utils import get_unit_prefix
from .datamodel.tables import Param, Column, GateGroup
from .datamodel.plotting import (hist_bar, hist_stair, hist_line, hist_kdeoverlay, 
    hist2d, hist, hexbin, kdeplot, scatter, jointplot,
    density_kde, rescale_size, density_kdesize, gaus_2Dkde, gaus_2Dkde_cmap,
    _check_ax, _get_column_arrays)

from .ph_sel import PhSel, DetDef, phsel_union
from .photondata import PhotonData, PhotonDataS
from .background import Periods

import fretbursts.cfuncs as fbc


def _dict_update(dct:dict, update:dict)->dict:
    dct.update(update)
    return dct


def _time_plot(func:str, ax:plt.Axes, data:PhotonDataS, col:Column, gate:GateGroup,
               include_unit:bool, xlabel:str, xlabel_kwargs:dict, ylabel:str, ylabel_kwargs:dict,
               kwargs:dict)->tuple[Any,plt.Text,plt.Text]:
    ax = _check_ax(ax)
    xlabel_kwargs = dict() if xlabel_kwargs is None else xlabel_kwargs
    ylabel_kwargs = dict() if ylabel_kwargs is None else ylabel_kwargs
    xcol = Column(col.base_param, 'midtime', ('istarttime', 'istoptime'))
    (xarr, yarr), (cxname, cyname) = _get_column_arrays(data, xcol, col, gate=gate, include_unit=include_unit)
    xlabel = cxname if xlabel is None else xlabel
    ylabel = cyname if ylabel is None else ylabel
    out = getattr(ax, func)(xarr, yarr, **kwargs)
    xlbl = ax.set_xlabel(xlabel, **xlabel_kwargs)
    ylbl = ax.set_ylabel(ylabel, **ylabel_kwargs)
    return out, xlbl, ylbl


def time_plot(data:PhotonDataS, col:Column, gate:GateGroup=None, ax:plt.Axes=None,
              include_unit:bool=False, 
              xlabel:str=None, xlabel_kwargs:dict=None, 
              ylabel:str=None, ylabel_kwargs:dict=None, 
              **kwargs:Any)->tuple[plt.Line2D,plt.Text,plt.Text]:
    return _time_plot('plot', ax, data, col, gate, include_unit, xlabel, xlabel_kwargs, 
                      ylabel, ylabel_kwargs, kwargs)

def time_scatter(data:PhotonDataS, col:Column, gate:GateGroup=None, ax:plt.Axes=None,
                 include_unit:bool=False, 
                 xlabel:str=None, xlabel_kwargs:dict=None, 
                 ylabel:str=None, ylabel_kwargs:dict=None, 
                 **kwargs:Any)->tuple[mpl.collections.PathCollection,plt.Text,plt.Text]:
    return _time_plot('scatter', ax, data, col, gate, include_unit, xlabel, xlabel_kwargs, 
                      ylabel, ylabel_kwargs, kwargs)


def _process_startstop(times:np.ndarray[np.int64], clk_p:float, period:Union[int,float], 
                       start:Union[None,float], stop:Union[None,float], 
                       start_at:Union[None,str], stop_at:Union[None,str]):
    if start is None:
        if start_at not in ('over', 'under', 'zero', 'time_min'):
            raise ValueError("Invalid value of start at ({start_at}), must be one of 'over', 'under', 'zero', 'time_min'")
        tstart = times[0] if start_at != 0 else 0
    else:
        tstart = start
    if stop is None:
        if stop_at not in ('over', 'under'):
            raise ValueError("Invalid value of start at ({start_at}), must be one of 'over', 'under', 'zero', 'time_min'")
        tstop = times[-1]
    else:
        tstop = stop
    if isinstance(period, Integral):
        period = (tstop - tstart) / period
    if start is not None:
        start = start/clk_p
    elif start_at == 'time_min':
        start = times[0]
    elif start_at == 'zero':
        start = np.int64(0)
    elif start_at == 'under':
        start = (times[0] // period) * period
    elif start_at == 'over':
        start = ((times[0]//period) + 1)* period
    # find stop time
    if stop is not None:
        stop = stop
    elif stop_at == 'under':
        stop = (times[-1] // period) * period
    elif stop_at == 'over':
        stop = ((times[-1] // period) + 1) * period
    # create periods
    return np.arange(start, stop, period)
    

def time_course_hist(data:PhotonDataS, col:Column, gate:GateGroup=None, ax:plt.Axes=None, 
                     period:Union[Integral,Real,np.ndarray]=10, bins:Union[Integral,np.ndarray]=10, 
                     include_unit:bool=True, time_direction:str='x', rescale=None,
                     norm:bool=True, norm_zero:bool=False, min_cnts:int=0, min_frac:float=0.0,
                     start_at:str='under', stop_at:str='over', start:float=None, stop:float=None,
                     xlabel:str=None, xlabel_kwargs:dict=None, ylabel:str=None, ylabel_kwargs:dict=None,
                     **kwargs:Any)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,mpl.collections.PolyQuadMesh,plt.Text,plt.Text]:
    ax = plt.gca() if ax is None else ax
    tcol = Column(col.base_param, 'midtime', ('istarttime', 'istoptime'))
    (varr, times), (vname, tname) = _get_column_arrays(data, col, tcol, gate=gate, include_unit=include_unit, rescale=rescale)
    # process time divisions
    period = np.asarray(period)
    # 0d array indicates as number, so process as a Periods param
    if period.size == 1:
        period = period.reshape(1)[0]
        periods = _process_startstop(times, data.clk_p, period, start, stop, start_at, stop_at)
    pidx = (periods / data.clk_p).astype(np.int64)
    starts, stops = fbc.index_ranges((times/data.clk_p).astype(np.int64), pidx[:-1], pidx[1:])
    del pidx
    # build histograms
    _, bns = np.histogram(varr[starts[0]:stops[0]], bins=bins)
    hst = np.array([np.histogram(varr[b:e], bins=bns)[0] for b, e in zip(starts, stops)])
    if callable(norm):
        nhst = norm(hst)
    else:
        nhst = hst / hst.max(axis=1)[:,np.newaxis] if norm else hst / hst.max()
        nhst[(hst < min_cnts) | (nhst < min_frac)] = np.nan
        if norm_zero:
            nhst -= np.nanmin(nhst, axis=1)[:,np.newaxis] if norm else np.nanmin(nhst)
            nhst /= np.nanmax(nhst, axis=1)[:,np.newaxis] if norm else np.nanmax(nhst)
    if time_direction.lower() in ('x', 'horizontal'):
        args = periods , bns, nhst.T
        xlabel = tname if xlabel is None else xlabel
        ylabel = vname if ylabel is None else ylabel
    elif time_direction.lower() in ('y', 'vertical'):
        args = bns, periods, nhst
        xlabel = vname if xlabel is None else xlabel
        ylabel = tname if ylabel is None else ylabel
    xlabel_kwargs = dict() if xlabel_kwargs is None else xlabel_kwargs
    ylabel_kwargs = dict() if ylabel_kwargs is None else ylabel_kwargs
    cmesh = ax.pcolor(*args, **kwargs)
    xttl = None if xlabel is False else ax.set_xlabel(xlabel, **xlabel_kwargs)
    yttl = None if ylabel is False else ax.set_ylabel(ylabel, **ylabel_kwargs)
    return hst, nhst, periods, bns, cmesh, xttl, yttl


def _get_factor(data:PhotonData, rescale:Union[None,int,float])->tuple[str, float]:
    rescale = 1.0 if rescale is None else rescale
    if isinstance(rescale, Integral):
        rescale = 10**rescale
    return get_unit_prefix(rescale)+'s', data.clk_p/rescale


def burst_dets(data:PhotonData, param:Param, burst:int, ax:plt.Axes=None, time_direction:str='x', 
               rescale:Union[int,float]=None, det_pos:dict[Union[int,PhSel],float]=None, 
               det_kwargs:dict[Union[int,PhSel],dict[str,Any]]=None, label_kwargs:dict[str,Any]=None,
               nanotime=False, nanotime_scale=1e9,
               **kwargs:Any)->tuple[mpl.collections.PathCollection,...,plt.Text]:
    detdef = data.detdef
    ax = plt.gca() if ax is None else ax
    label_kwargs = dict() if label_kwargs is None else label_kwargs
    if det_pos is None:
        if det_kwargs is None:
            det_pos = {detdef.stream_ids_to_PhSel(i,convert_all=True):i for i in range(data.detdef.size)}
        else:
            det_pos = {k:i for i, k in enumerate(det_kwargs.keys())}
    det_pos = {detdef.stream_ids_to_PhSel(k, convert_all=True) if not isinstance(k, PhSel) else k:i 
               for k, i in det_pos.items()}
    if det_kwargs is None:
        det_kwargs = {k:dict() for k in det_pos.keys()}
    det_kwargs = {detdef.stream_ids_to_PhSel(k, convert_all=True) if not isinstance(k, PhSel) else k:v 
                 for k, v in det_kwargs.items()}
    phsel = phsel_union(*det_pos.keys())
    if nanotime in (True, 'abs', 'window', 'thresh'):
        for i, (times, dets, nanos) in enumerate(zip(data.iter_column(Column(param, 'ph_times', phsel)),
                                                     data.iter_column(Column(param, 'ph_dets', phsel)),
                                                     data.iter_column(Column(param, 'ph_nanos', phsel)))):
            if i == burst:
                nanos = nanos.astype(np.int16)
                break
    else:
        for i, (times, dets) in enumerate(zip(data.iter_column(Column(param, 'ph_times', phsel)),
                                              data.iter_column(Column(param, 'ph_dets', phsel)))):
            if i == burst:
                break        
    out = list()
    unit, factor = _get_factor(data, rescale)
    time_direction = time_direction.lower()
    nanotime_scale *= data.setup['tcspc_unit'][0]
    for k, p in det_pos.items():
        dids = detdef.get_stream_ids(k)
        mask = np.isin(dets, dids)
        x, y = times[mask]*factor, np.repeat(p, mask.sum())
        if nanotime == 'abs':
            y = y + nanos[mask]*nanotime_scale
        elif nanotime == 'window':
            ex = list(list(k.render_positive(detdef, convert_all=False).streams)[0].ex.elements)[0]
            ws = data.setup['ex_ranges'][ex][0,0]
            y = y + (nanos[mask]-ws)*nanotime_scale
        elif nanotime == 'thresh' or nanotime is True:
            y = y + (nanos[mask]-data.irf_thresh[k])*nanotime_scale
        xy = x, y
        if time_direction in ('y', 'vertical'):
            xy = xy[::-1]
        out.append(ax.scatter(*xy, **_dict_update(det_kwargs.get(k, dict()), kwargs)))
    if time_direction in ('y', 'vertical'):
        out.append(ax.set_ylabel(unit, **label_kwargs))
    else:
        out.append(ax.set_xlabel(unit, **label_kwargs))
    return out