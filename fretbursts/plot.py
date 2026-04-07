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
from typing import Union, Any, Literal
from collections.abc import Callable, Sequence
from numbers import Integral, Real
from itertools import repeat

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import phconvert.plotter as plotter

from .datamodel.utils import get_unit_prefix, _dict_update, fjit
from .datamodel.immutabledata import get_pycode_subval
from .datamodel.tables import Param, Column, GateGroup
from .datamodel.plot import (
    hist_bar, hist_stair, hist_line, hist_kdeoverlay, 
    hist2d, hist, hexbin, kdeplot, scatter, errorbars, jointplot,
    density_kde, rescale_size, density_kdesize, gaus_2Dkde, gaus_2Dkde_cmap,
    _check_ax, _get_column_arrays, _rescale_value, plot_meaninterval, 
    scatter_meaninterval, colorcategory, plot_multi_dist
    )


from .ph_sel import PhSel, phsel_union
from .photondata import PhotonData, PhotonDataS, _title_sels
from .background import BG
from .bursttables import Bursts

import fretbursts.cfuncs as fbc



from .photonHDF5 import PhotonHDF5Data


def alternation_hist(raw:PhotonHDF5Data, ich:int=0, ax:plt.Axes=None, group_dets:None|bool=None,
                     **kwargs):
    """
    Plot alternation histogram of raw data.

    Parameters
    ----------
    raw : PhotonHDF5Data
        Raw data to plot.
    ich : int, optional
        For multi-spot data, which spot to plot. The default is 0.
    ax : plt.Axes, optional
        `plt.Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_  
        in which to plot data, if None, use 
        `plt.gca() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.gca.html>`_ . 
        The default is None.
    group_dets : None|bool, optional
        Whether to group detectors based on names. The default is None.
    **kwargs : Any
        Kwargs passed to phc.plotter.alternation_hist.

    """
    if group_dets is None:
        group_dets = any(plotter._ch_rgx.fullmatch(k) for k 
                         in raw.photon_data[ich].meas_specs['detectors_specs'].keys())
    return plotter.alternation_hist(raw.as_photonHDF5_dict, ich=ich, ax=ax, group_dets=group_dets,**kwargs)


def _time_plot(func:str, ax:plt.Axes, data:PhotonDataS, col:Column, gate:GateGroup,
               include_unit:bool, plotlabel:str, 
               xlabel:str, xlabel_kwargs:dict, ylabel:str, ylabel_kwargs:dict,
               kwargs:dict)->tuple[Any,plt.Text,plt.Text]:
    """
    Internal function for ploting time series. Avoids making gates to save memory.
    """
    ax = _check_ax(ax)
    xlabel_kwargs = dict() if xlabel_kwargs is None else xlabel_kwargs
    ylabel_kwargs = dict() if ylabel_kwargs is None else ylabel_kwargs
    plotlabel = col.name(origin=data) if plotlabel is None else plotlabel
    if plotlabel and 'label' not in kwargs:
        kwargs = _dict_update(kwargs, {'label':plotlabel})
    xcol = Column(col.base_param, 'midtime', ('istarttime', 'istoptime'))
    (xarr, yarr), (cxname, cyname) = _get_column_arrays(data, xcol, col, gate=gate, include_unit=include_unit)
    xlabel = cxname if xlabel is None else xlabel
    ylabel = cyname if ylabel is None else ylabel
    out = getattr(ax, func)(xarr, yarr, **kwargs)
    xlbl = ax.set_xlabel(xlabel, **xlabel_kwargs) if xlabel else None
    ylbl = ax.set_ylabel(ylabel, **ylabel_kwargs) if xlabel else None
    return out, xlbl, ylbl


def time_plot(data:PhotonDataS, col:Column, gate:GateGroup=None, ax:plt.Axes=None,
              include_unit:bool=False, plotlabel:str=None,
              xlabel:str=None, xlabel_kwargs:dict=None, 
              ylabel:str=None, ylabel_kwargs:dict=None, 
              **kwargs:Any)->tuple[plt.Line2D,plt.Text,plt.Text]:
    """
    Plot column col vs time of row. 
    Wrapper of `ax.plot() <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_

    Parameters
    ----------
    data : PhotonDataS
        Source of data to plot.
    col : Column
        Column to plot time series.
    gate : GateGroup, optional
        Gate to apply to col. The default is None.
    ax : plt.Axes, optional
        `plt.Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_  
        in which to plot data, if None, use 
        `plt.gca() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.gca.html>`_ . 
        The default is None.
    include_unit : bool, optional
        Whether to include unit in column labels. The default is False.
    plotlabel : str optional
        Name to set the 'label' keyword argument handed to ax.plot, if None, will
        automatically assigne based on col. If False, will not set 'label' kwarg.
        The default is None.
    xlabel : str, optional
        Name to set xlabel, if None, will automatically assign based on col.
        The default is None.
    xlabel_kwargs : dict, optional
        Keyword arguments passed to 
        `ax.set_xlabel <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlabel.html>`_. 
        The default is None.
    ylabel : str, optional
        Name to set ylabel, if None, will automatically assign based on col.
        The default is None.
    ylabel_kwargs : dict, optional
        Keyword arguments passed to 
        `ax.set_ylabel <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ylabel.html>`_. 
        The default is None.
    **kwargs : Any
        Kwargs passed to `ax.plot() <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_ .

    Returns
    -------
    out : plt.Line2D
        Matplotib line2D of histogram
    xttl : plt.Text
        Matplotlib Text object of xlabel.
    yttl : plt.Text
        Matplotlib Text object of ylabel.

    """
    return _time_plot('plot', ax, data, col, gate, include_unit, plotlabel, 
                      xlabel, xlabel_kwargs, ylabel, ylabel_kwargs, kwargs)

def time_scatter(data:PhotonDataS, col:Column, gate:GateGroup=None, ax:plt.Axes=None,
                 include_unit:bool=False, plotlabel:str=None,
                 xlabel:str=None, xlabel_kwargs:dict=None, 
                 ylabel:str=None, ylabel_kwargs:dict=None, 
                 **kwargs:Any)->tuple[mpl.collections.PathCollection,plt.Text,plt.Text]:
    """
    Scatter plot column col vs time of row.
    Wrapper of `ax.scatter() <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html>`_

    Parameters
    ----------
    data : PhotonDataS
        Source of data to plot.
    col : Column
        Column to plot time series.
    gate : GateGroup, optional
        Gate to apply to col. The default is None.
    ax : plt.Axes, optional
        `plt.Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_  
        in which to plot data, if None, use 
        `plt.gca() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.gca.html>`_ . 
        The default is None.
    include_unit : bool, optional
        Whether to include unit in column labels. The default is False.
    plotlabel : str optional
        Name to set the 'label' keyword argument handed to ax.plot, if None, will
        automatically assigne based on col. If False, will not set 'label' kwarg.
        The default is None.
    xlabel : str, optional
        Name to set xlabel, if None, will automatically assign based on col.
        The default is None.
    xlabel_kwargs : dict, optional
        Keyword arguments passed to 
        `ax.set_xlabel <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlabel.html>`_. 
        The default is None.
    ylabel : str, optional
        Name to set ylabel, if None, will automatically assign based on col.
        The default is None.
    ylabel_kwargs : dict, optional
        Keyword arguments passed to 
        `ax.set_ylabel <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ylabel.html>`_. 
        The default is None.
    **kwargs : Any
        Kwargs passed to `ax.plot() <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_ .

    Returns
    -------
    pathcollection : mpl.collections.PathCollection
        Path collection of scatter plot points.
    xttl : plt.Text
        Matplotlib Text object of xlabel.
    yttl : plt.Text
        Matplotlib Text object of ylabel.

    """
    return _time_plot('scatter', ax, data, col, gate, include_unit, plotlabel,
                      xlabel, xlabel_kwargs, ylabel, ylabel_kwargs, kwargs)


def _process_startstop(times:np.ndarray[np.int64], clk_p:float, period:int|float,
                       start:None|float, stop:None|float,
                       start_at:None|str, stop_at:None|str)->np.ndarray[np.float64]:
    """Process periods information into array of time bins for data"""
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
                     period:Integral|Real|np.ndarray=10, bins:Integral|np.ndarray=10, 
                     include_unit:bool=True, 
                     time_direction:Literal['x', 'horizontal', 'y', 'vertical']='x', 
                     rescale:Real=None, 
                     norm:bool|Callable[[np.ndarray],np.ndarray[np.double]]=True, 
                     norm_zero:bool=False, min_cnts:int=0, min_frac:float=0.0,
                     start_at:Literal['zero','time_min','under','over']='under', 
                     stop_at:Literal['under','over']='over', start:float=None, stop:float=None,
                     xlabel:str=None, xlabel_kwargs:dict=None, ylabel:str=None, ylabel_kwargs:dict=None,
                     **kwargs:Any)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,mpl.collections.PolyQuadMesh,plt.Text,plt.Text]:
    """
    Create 2-D histogram of column vs time.

    Parameters
    ----------
    data : PhotonDataS
        Source of data to histogram.
    col : Column
        Column to histogram per time bin.
    gate : GateGroup, optional
        Gate for bursts/rows. The default is None.
    ax : plt.Axes, optional
        `plt.Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_  
        in which to plot data, if None, use 
        `plt.gca() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.gca.html>`_ . 
        The default is None.
    period : Integral|Real|np.ndarray, optional
        Time division period. If int, sets number of time bins, if float, sets size
        of single period, if array, sets bins of time period. The default is 10.
    bins : Integral|np.ndarray, optional
        Bins of column axis. The default is 10.
    include_unit : bool, optional
        Whether to include unit in column labels. The default is False.
    time_direction : {'x', 'horizontal', 'y', 'vertical'}, optional
        Axis of time axis. The default is 'x'.
    rescale : Real, optional
        Factor (u be power of 10) by which to rescale values of col. The default is None.
    norm : bool|Callable[[np.ndarray],np.ndarray[np.double]], optional
        Whether to normalize column histograms so all time slice histograms have
        max = 1. Can specify callable to perform custom normalization, handed
        one 2-d numpy array, 
        The default is True.
    norm_zero : bool, optional
        Subtract min of histogram so normalized bins have min of 0. The default is False.
    min_cnts : int, optional
        Min number of points in bin for histogrm bin to recieve non-transparent 
        color. The default is 0.
    min_frac : float, optional
        Minimum fraction of histogram for histogram bin to recieve non-transparent
        color. The default is 0.0.
    start_at : {'zero','time_min','under','over'}, optional
        Definition based on minimum time in data of stop time. The default is 'under'.
    stop_at : {'under', 'over'}, optional
        Definition based on maximum time in data, of stop time. The default is 'over'.
    start : float, optional
        Start time, overrides start_at. The default is None.
    stop : float, optional
        Stop time, overrides stop_at. The default is None.
    xlabel : str, optional
        Name to set xlabel, if None, will automatically assign based on col and normalize.
        The default is None.
    xlabel_kwargs : dict, optional
        Keyword arguments passed to 
        `ax.set_xlabel <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlabel.html>`_. 
        The default is None.
    ylabel : str, optional
        Name to set ylabel, if None, will automatically assign based on col and normalize.
        The default is None.
    ylabel_kwargs : dict, optional
        Keyword arguments passed to 
        `ax.set_ylabel <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ylabel.html>`_. 
        The default is None.
    **kwargs : Any
        Kwargs hannded to `ax.pcolor() <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pcolor.html>`_.

    Returns
    -------
    hst : np.ndarray
        Non-normalized (raw counts) 2D histogram of column x times.
    nhst : np.ndarray
        Normalized 2D histogram of column x times.
    periods : np.ndarray
        Time bins of histogram.
    bns : np.ndarray
        Column bins of histogram.
    cmesh : mpl.collections.PolyQuadMesh
        `mpl.collections.PolyQuadMesh <https://matplotlib.org/stable/api/collections_api.html>`_
        returned by ax.pcolor
    xttl : plt.Text
        Matplotlib Text object of xlabel.
    yttl : plt.Text
        Matplotlib Text object of ylabel.

    """
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
    """Parse rescale into (unit, float of rescale factor)"""
    rescale = 1.0 if rescale is None else rescale
    if isinstance(rescale, Integral):
        rescale = 10**rescale
    return get_unit_prefix(rescale)+'s', data.clk_p/rescale


def burst_dets(data:PhotonData, param:Param, burst:int, ax:plt.Axes=None, 
               time_direction:Literal['x', 'horizontal', 'y', 'vertical']='x', 
               rescale:Real=None, zerostart:bool=False, det_pos:dict[int|PhSel:float]=None, 
               det_kwargs:dict[int|PhSel:dict[str,Any]]=None, label_kwargs:dict[str:Any]=None,
               nanotime:bool=False, nanotime_scale:float=1e9,
               **kwargs:Any)->tuple[mpl.collections.PathCollection,...,plt.Text]:
    """
    Create a plot of photons vs time for a single time range (usually burst)

    Parameters
    ----------
    data : PhotonData
        Source data for plot.
    param : Param
        Definition of time range, must be BasePhotonTable based :class:`Param`.
    burst : int
        Burst number.
    ax : plt.Axes, optional
        `plt.Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_  
        in which to plot data, if None, use 
        `plt.gca() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.gca.html>`_ . 
        The default is None.
    time_direction : Literal['x', 'horizontal', 'y', 'vertical'], optional
        Direction of time axis. The default is 'x'.
    rescale : Real, optional
        Rescale factor for time axis. The default is None.
    zerostart : bool, optional
        If True, photon times start at 0. The default is True
    det_pos : dict[int|PhSel:float], optional
        Dictionary of detector to position mappings. The default is None.
    det_kwargs : dict[int|PhSel:dict[str,Any]], optional
        Dictionary of kwargs dictionaries passed per detector key to `ax.scatter()`. 
        The default is None.
    label_kwargs : dict[str:Any], optional
        Keyword arguments passed to 
        `ax.set_x/ylabel`. 
    nanotime : bool, optional
        Whether to add shift for nanotimes. The default is False.
    nanotime_scale : float, optional
        Scale value for nanotimes. The default is 1e9.
    **kwargs : Any
        Universal kwargs hannded to 
        `ax.scatter() <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html>`_
        for each plot of detector class

    Returns
    -------
    tuple[mpl.collections.PathCollection,...,plt.Text]
        Path collection for each detector index plotted, finished with Text object
        for set_x/ylabel.

    """
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
    times = times - times[0] if zerostart else times
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


def _sel_wrap(name:str, origin:PhotonData, *args:PhSel)->str:
    selstr = ', '.join(_title_sels(name, origin, *args))
    return f'${selstr}$'


def _makebg_thresh(ax:plt.Axes, data:PhotonData, bg:Param, stream:PhSel, 
                   bins:np.ndarray[np.float64], bsize:np.ndarray[np.float64], 
                   factor:float, bg_kwargs:dict):
    bg = data.get_table(bg)
    periods = bg.parents['base']['periods', 1]
    locs = np.searchsorted(periods, bins)
    locs[locs >= periods.size-1] = periods.size - 2
    bgr = bg['bg', stream][locs]*bsize*factor
    return ax.plot(bins, bgr, **bg_kwargs)


def _timetrace_bg_sort(bg:Param, sel:PhSel)->tuple[Param,float]:
    if bg.tp == Bursts:
        detdef = bg.tp._detdef(bg)
        sub, sbg, sF, sublen = None, None, 0, np.inf
        for bsel, b, F in zip(bg.params['streams'], bg.parents['bg'], bg.params['F']):
            if bsel == sel:
                return b, F
            temp = sel - bsel
            templen = detdef.get_stream_ids(temp).size
            if  templen < sublen:
                sub, sublen, sbg, sF = temp, templen, b, F
        if sub is not None:
            return sbg, sF
        return bg.params['streams'][0], bg.params['F'][0]
    if bg.tp != BG:
        if 'bg' not in bg.parents:
            raise ValueError("cannot determine background param from bg")
        bg = bg.parents['bg']
    return bg, 1.0


def _trace_label_proc(data:PhotonData, labels:None|bool|Sequence[str], streams:None|Sequence[PhSel], name:str):
    """Process label sequence for time/ratetrace labels"""
    set_labels = labels is not None and labels is not False
    if set_labels:
        if labels is True:
            labels = (True for _ in range(len(streams)))
        labels = (_sel_wrap(name, data, s) if l is True else l for s, l in zip(streams, labels))
    else:
        labels = repeat(None)
    return set_labels, labels


def _trtrace_proc(data, streams, bg, stream_kwargs, bg_kwargs, labels, bg_labels, direction):
    """Process args common to timetrace and ratetrace"""
    # sort streams
    if streams is None:
        if isinstance(bg, Param) and bg.base_param.tp == Bursts:
            bg = bg.base_param
            streams = bg.params['streams']
        else:
            streams = tuple(data.detdef.stream_ids_to_PhSel(i) for i in range(data.detdef.size))
        bg = repeat(bg)
    bg = repeat(bg) if isinstance(bg, Param) or bg is None else bg
    stream_kwargs = dict() if stream_kwargs is None else stream_kwargs
    stream_kwargs = repeat(stream_kwargs) if isinstance(stream_kwargs, dict) else stream_kwargs
    set_labels, labels = _trace_label_proc(data, labels, streams, 'n')
    set_bglabels, bg_labels = _trace_label_proc(data, bg_labels, streams, 'bg')
    set_legend = set_labels or set_bglabels
    bg_kwargs = dict() if bg_kwargs is None else bg_kwargs
    bg_kwargs = bg_kwargs if isinstance(bg_kwargs, Sequence) else repeat(bg_kwargs)
    direction = repeat(True) if direction is None else direction
    return set_legend, zip(bg, streams, stream_kwargs, bg_kwargs, labels, bg_labels, direction)


def _idx_bounds(array:np.ndarray[np.int64], size:int)->np.ndarray[np.int64]:
    array[array<0] = 0
    array[array>=size] = size - 1
    return array


def _idx_sorted(array:np.ndarray[np.int64], loc:Integral|Sequence[Integral])->np.ndarray[np.int64]:
    idx = np.searchsorted(array, loc) - 1
    return _idx_bounds(idx, array.size)


def _colapse_transitions(idx:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    tloc = np.diff(idx) != 0
    mask = np.zeros(idx.shape, dtype=np.bool_)
    mask[1:] |= tloc
    mask[:-1] |= tloc
    mask[0] = True
    mask[-1] = True
    y = idx[mask]
    x = np.empty_like(y)
    x[0] = y[0]
    x[:-1] = y[1:]
    x[-1] = y[-1]
    return x, y
    

def timetrace(data:PhotonData, ax:plt.Axes=None, streams:Sequence[PhSel]=None,
              bg:Param=None, bins:Integral|Real|np.ndarray=None,
              tmin:float=0.0, tmax:float=1.0, binwidth:float=1e-3,
              stream_kwargs:Sequence[dict]=None, bg_kwargs:Sequence[dict]=None,
              direction:Sequence[bool|int]=None, 
              labels:Sequence[str]|bool=True, bg_labels:Sequence[str]|bool=True,
              xlabel:str=None, xlabel_kwargs:dict=None, ylabel:str=None, ylabel_kwargs:dict=None, 
              **kwargs)->tuple[list[plt.Line2D],list[plt.Line2D],plt.Text,plt.Text,mpl.legend.Legend]:
    """
    Plot a binned timetrace of a section of photon arrival times in data.

    Parameters
    ----------
    data : PhotonData
        Data containing photon times.
    ax : plt.Axes, optional
        `plt.Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_  
        in which to plot data, if None, use 
        `plt.gca() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.gca.html>`_ . 
        The default is None.
    streams : Sequence[PhSel], optional
        Sequence of photon streams to plot, if None, will plot each stream separately. 
        The default is None.
    bg : Param, optional
        If specified, overlay a plot the computed background photon rate. 
        The default is None.
    bins : np.ndarray | int | float, optional
        Specification of bins for timetrace. If int will plot that number of bins
        in time range, if float will plot bins of that size (unit = s), if 
        array, the array will define the bin edges. The default is None.
    tmin : float, optional
        Start time of bins, if bins is array this argument is ignored. 
        The default is 0.0
    tmax : float, optional
        Stop time of bins, if bins is array this argument is ignored. 
        The default is 1.0
    binwidth : float, optional
        Size of single time bin, ignored if bins is specified. The default is 1e-4.
    stream_kwargs : Sequence[dict], optional
        Kwargs hannded to ax.plot per stream. The default is None.
    bg_kwargs : dict, optional
        kwargs handed to time_plot for plotting background/background threshold.
    direction : Sequence[bool|int], optional
        Direction for each stream, if stream is True/1 plot in positive direction, 
        if stream is False/-1 plot with higher values in negative direction.
        The default is None.
    labels : Sequence[str|bool]|bool, optional
        Labels (sequence of str) per stream. If True, add labels with automatic
        names. If False, do not add labels. If sequence, can specify True to create
        automatic label for specific stream. The default is True.
    bg_labels : Sequence[str|bool]|bool, optional
        Labels (sequence of str) per stream for bg lines. If True, add labels 
        with automatic names. If False, do not add labels. 
        If sequence, can specify True to create automatic label for specific stream. 
        The default is True.
    xlabel : str, optional
        Name to set xlabel, if None, will automatically assign based on col and normalize.
        The default is None.
    xlabel_kwargs : dict, optional
        Keyword arguments passed to 
        `ax.set_xlabel <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlabel.html>`_. 
        The default is None.
    ylabel : str, optional
        Name to set ylabel, if None, will automatically assign based on col and normalize.
        The default is None.
    ylabel_kwargs : dict, optional
        Keyword arguments passed to 
        `ax.set_ylabel <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ylabel.html>`_. 
        The default is None.
    **kwargs : Any
        Keyword arguments handed to each call to ax.plot for plotting interphton
        time histograms.

    Returns
    -------
    hst_lines : list[plt.Line2D]
        list of Matplotib line2D of histograms of bins per stream
    bg_lines : list[plt.Line2D]
        list of Matplotib line2D of background-rate or threshold
    xttl : plt.Text
        Text object from ax.set_xlabel.
    yttl : plt.Text
        Text object from ax.set_ylabel.
    leg : mpl.legend.Legend
        Legend object from ax.legend.

    """
    # sort bins
    if bins is None:
        bins = np.arange(tmin, tmax+binwidth, binwidth)
    elif isinstance(bins, Integral):
        bins = np.linspace(tmin, tmax, bins+1)
    elif isinstance(bins, Real):
        bins = np.arange(tmin, tmax+bins, bins)
    mbins = np.diff(bins)/2 + bins[:-1]
    # sort streams
    set_legend, stream_zip = _trtrace_proc(data, streams, bg, stream_kwargs, bg_kwargs, labels, bg_labels, direction)
    xlabel = 'time (s)' if xlabel is None or xlabel is True else xlabel
    xlabel_kwargs = dict() if xlabel_kwargs is None else xlabel_kwargs
    ylabel = 'cnts' if ylabel is None or ylabel is True else ylabel
    ylabel_kwargs = dict() if ylabel_kwargs is None else ylabel_kwargs
    bins_clk = (bins / data.clk_p).astype(np.int64)
    bsize = np.diff(bins)
    bsize = np.concatenate([bsize, bsize[-1:]])
    idx = fbc.index_range(data.times, bins_clk[0], bins_clk[-1], 0)
    times, dets = data.times[idx[0]:idx[1]], data.dets[idx[0]:idx[1]]
    times = times * data.clk_p
    # get axis if not supplied
    ax = plt.gca() if ax is None else ax
    hst_lines, bg_lines = list(), list()
    # loop over all specified streams, adding to plot
    for sbg, stream, stream_kws, bg_kws, label, bg_label, direct in stream_zip:
        hst, _ = np.histogram(times[np.isin(dets, data.detdef.get_stream_ids(stream))], bins)
        direct = -1 if direct < 1 else 1
        hst *= direct
        if label is not None:
            stream_kws.setdefault('label', label)
        hst_lines.append(ax.plot(mbins, hst, **_dict_update(stream_kws, kwargs))[0])
        if sbg is None:
            bg_lines.append(None)
            continue
        sbg, factor = _timetrace_bg_sort(sbg, stream)
        sbg = data.get_table(sbg)
        prd = sbg.parents['base']['periods']
        bgc = sbg['bg', stream]
        idx = _idx_sorted(prd, bins_clk)
        idx, idy = _colapse_transitions(idx)
        ttemp = prd[idx] * data.clk_p
        ttemp[0] = bins[0]
        ttemp[-1] = bins[-1]
        # corrections for overflows
        bgc = bgc[idy]*factor*direct
        bgc *= bsize[idx]
        bg_kws = dict() if bg_kws is None else bg_kws
        if 'c' not in bg_kws or 'color' not in bg_kws:
            bg_kws = _dict_update(bg_kws, {'c':hst_lines[-1].get_color()})
        if bg_label is not None:
            bg_kws.setdefault('label', bg_label)
        bg_lines.append(ax.plot(ttemp, bgc, **bg_kws)[0])
    xttl = None if xlabel is False else ax.set_xlabel(xlabel, **xlabel_kwargs)
    yttl = None if ylabel is False else ax.set_ylabel(ylabel, ylabel_kwargs)
    leg = ax.legend() if set_legend else None
    return hst_lines, bg_lines, xttl, yttl, leg


@fjit('Tuple((i8[:],i8[:]))(i8[:],i8)')
def _ratetrace(photons:np.ndarray[np.int64], m:int)->tuple[np.ndarray[np.int64], np.ndarray[np.int64]]:
    diff = np.empty(photons.size-m, dtype='i8')
    mean = np.empty(photons.size-m, dtype='i8')
    for i in range(photons.size-m):
        diff[i] = photons[i+m] - photons[i]
        mean[i] = np.sum(photons[i:m+i]) // m
    return diff, mean


def ratetrace(data:PhotonData, ax:plt.Axes=None, streams:Sequence[PhSel]=None,
              bg:Param=None, m:Integral=10, tmin:float=0.0, tmax:float=1.0,
              stream_kwargs:Sequence[dict]=None, bg_kwargs:Sequence[dict]=None,
              direction:Sequence[bool|int]=None, 
              labels:Sequence[str]|bool=True, bg_labels:Sequence[str]|bool=True, 
              xlabel:str=None, xlabel_kwargs:dict=None, ylabel:str=None, ylabel_kwargs:dict=None, 
              **kwargs)->tuple[list[plt.Line2D],list[plt.Line2D],plt.Text,plt.Text,mpl.legend.Legend]:
    ax = _check_ax(ax)
    # sort streams
    set_legend, stream_zip = _trtrace_proc(data, streams, bg, stream_kwargs, bg_kwargs, labels, bg_labels, direction)
    xlabel = 'time (s)' if xlabel is None or xlabel is True else xlabel
    ylabel = r'$cnts\:s^{-1}$' if ylabel is None or ylabel is True else ylabel
    xlabel_kwargs = dict() if xlabel_kwargs is None else xlabel_kwargs
    ylabel_kwargs = dict() if ylabel_kwargs is None else ylabel_kwargs
    idx = fbc.index_range(data.times, int(tmin/data.clk_p), int(tmax/data.clk_p), 0)
    times, dets = data.times[idx[0]:idx[1]], data.dets[idx[0]:idx[1]]
    rt_lines = list()
    bg_lines = list()
    # loop over all specified streams, adding to plot
    for sbg, stream, stream_kws, bg_kws, label, bg_label, direct in stream_zip:
        direct = -1 if direct < 1 else 1
        mask = np.isin(dets, data.detdef.get_stream_ids(stream))
        ttemp = times[mask]
        delays, means = _ratetrace(ttemp, m)
        rates, means = 1/delays/data.clk_p, data.clk_p * means
        if label is not None:
            stream_kws.setdefault('label', label)
        rt_lines.append(ax.plot(means, direct*rates, **_dict_update(stream_kws, kwargs))[0])
        if sbg is None:
            bg_lines.append(None)
            continue
        if bg_label is not None:
            bg_kws.setdefault('label', bg_label)
        sbg, factor = _timetrace_bg_sort(sbg, stream)
        bg = data.get_table(sbg)
        sbg = data.get_table(sbg)
        prd = sbg.parents['base']['periods']
        bgc = sbg['bg', stream]*factor*direct
        idx = _idx_sorted(prd, [tmin/data.clk_p, tmax/data.clk_p])
        brate = bgc[idx]
        btimes = np.array([tmin, tmax])
        if idx[0] != idx[1]:
            # index of transitions between background periods in the ttemp array
            nidx = _idx_sorted(ttemp, prd[list(range(idx[0]+1, idx[1]+1))])
            # process bg rate transistions
            nbrate = np.empty(nidx.size*2+2, dtype=np.float64)
            nbrate[0] = brate[0]
            nbrate[-1] = brate[1]
            nbrate[1:-1:2] = bgc[idx[0]:idx[1]]
            nbrate[2:-1:2] = bgc[idx[0]+1:idx[1]+1]
            brate = nbrate
            # process location of time transitions
            nbtimes = np.empty(nidx.size*2, dtype=np.int64)
            # location of transitions shifted by +/- m
            nbtimes[::2] = ttemp[_idx_bounds(nidx-m, ttemp.size)]
            nbtimes[1::2] = ttemp[_idx_bounds(nidx+m, ttemp.size)]
            print()
            # fill new times array
            nbbtimes = np.empty(nbtimes.size+2, dtype=np.float64)
            nbbtimes[0] = btimes[0]
            nbbtimes[-1] = btimes[1]
            nbbtimes[1:-1] = nbtimes * data.clk_p
            btimes = nbbtimes
            if btimes[-2] > btimes[-1]:
                btimes = btimes[:-1]
                brate = brate[:-1]
        if 'c' not in bg_kws or 'color' not in bg_kws:
            bg_kws = _dict_update(bg_kws, {'c':rt_lines[-1].get_color()})
        if bg_label is not None:
            bg_kws.setdefault('label', bg_label)
        bg_lines.append(ax.plot(btimes, brate, **bg_kws)[0])
    xttl = None if xlabel is False else ax.set_xlabel(xlabel, **xlabel_kwargs)
    yttl = None if ylabel is False else ax.set_ylabel(ylabel, ylabel_kwargs)
    leg = ax.legend() if set_legend else None
    return rt_lines, bg_lines, xttl, yttl, leg


def _get_nth(data:PhotonData, n:int, *args:Column)->tuple[np.ndarray[np.int64,...]]:
    """Get the nth row of each column"""
    for i, *c in zip(range(n+1), *(data.iter_column(arg) for arg in args)):
        pass
    return c


def _sort_times(data:PhotonData, phsel:PhSel)->np.ndarray[np.int64]:
    """Get all times of data in phsel"""
    stream_ids = data.detdef.get_stream_ids(phsel)
    return data.times[np.isin(data.dets, stream_ids)]


def hist_interphoton(data:PhotonData, bg:Param=None, n:int=0, ax:plt.Axes=None,
                     streams:Sequence[PhSel]=None, labels:Sequence[str]=True,
                     bins:np.ndarray[np.float64]=None, tmin=0.0, tmax=1e-2, binwidth=1e-4,
                     streams_kwargs:Sequence[dict]=None, 
                     fit_streams_kwargs:dict=None, fit_kwargs:dict=None, 
                     fit_labels:Sequence[str]=True, fit_cps_labels:Sequence[bool]|bool=True,
                     xscale:str='linear', yscale:str='log', time_scale=-3, 
                     set_legend:bool=True, legend_kwargs:dict=None,
                     xlabel:str=None, xlabel_kwargs:dict=None, ylabel:str=None, ylabel_kwargs:dict=None,
                     **kwargs)->tuple[list[mpl.collections.PathCollection],list[plt.Line2D],plt.Text,plt.Text,mpl.legend.Legend]:
    """
    Plot interphoton time histogram. If bg is specified, it must be a :class:`BG` 
    based :class:`Param`, specifying bg will overlay histogram with background-rate fit line.

    Parameters
    ----------
    data : PhotonData
        Data to plot.
    bg : Param, optional
        Background fit to plot, if None will not plot fits (none to plot), and 
        interphoton time histogram will be performed over entire dataset. 
        The default is None.
    n : int, optional
        Which background period to plot. The default is 0.
    ax : plt.Axes, optional
        Axis in which to plot interphoton time histogram. The default is None.
    streams : Sequence[PhSel], optional
        Photon streams to plot interphoton times. The default is None.
    labels : Sequence[str], optional
        Labels for each stream. The default is True.
    bins : np.ndarray[np.float64], optional
        Bins for interphoton time histogram. The default is None.
    tmin : TYPE, optional
        Time of first bin of interphoton time histogram bins if bins is not specified. 
        The default is 0.0.
    tmax : TYPE, optional
        Time of last bin of interphoton time histogram bins if bins is not specified. 
        The default is 1e-2.
    binwidth : TYPE, optional
        If bins not specified, the width of each bin in histogram. The default is 1e-4.
    streams_kwargs : Sequence[dict], optional
        Per-stream kwargs (specify as sequence) for each interphoton time histogram. 
        The default is None.
    fit_streams_kwargs : dict, optional
        Per-stream kwargs (specify as sequence) for each fit line. The default is None.
    fit_kwargs : dict, optional
        Univeral kwargs handed to all calls of ax.plot for fit line of each stream. 
        The default is None.
    fit_labels : Sequence[str], optional
        Labels for each stream of fit line. The default is True.
    fit_cps_labels : Sequence[bool]|bool, optional
        Whether to add kilo counts per second to fit labels. The default is True.
    xscale : str, optional
        Y-axis scale (usually either log or linear). The default is 'linear'.
    yscale : str, optional
        Y-axis scale (usually either log or linear). The default is 'log'.
    time_scale : TYPE, optional
        Rescale factor of x-axis (time).
        If Integral interpret as power of 10, if float direct multiplier. 
        The default is -3.
    set_legend : bool, optional
        Whether to call ax.legend at end of function. The default is True.
    legend_kwargs : dict, optional
        Kwargs hannded to ax.legend. The default is None.
    xlabel : str, optional
        Label for x-axis, if False, will not call ax.set_xlabel, if None, set
        automatically, if str, will set that string as xlabel. The default is None.
    xlabel_kwargs : dict, optional
        Kwargs handed to ax.set_xlabel. The default is None.
    ylabel : str, optional
        Label for y-axis, if False, will not call ax.set_ylabel, if None, set
        automatically, if str, will set that string as ylabel. The default is None.
    ylabel_kwargs : dict, optional
        Kwargs handed to ax.set_ylabel. The default is None.
    **kwargs : Any
        Additional kwargs handed directly to each ax.scatter for each interphoton
        time histogram.

    Returns
    -------
    pathcol : list[mpl.collections.PathCollection]
        List of each return of ax.scatter from plotting individual stream interphoton
        time histogram.
    line2d : list[plt.Line2D]
        List of each return of ax.plot from plotting fit of stream background count rate.
    xttl : plt.Text
        Text object from ax.set_xlabel.
    yttl : plt.Text
        Text object from ax.set_ylabel.
    leg : mpl.legend.Legend
        Legend object from ax.legend.

    """
    # interpret kwargs (convert None to usuable default)
    ax = plt.gca() if ax is None else ax
    if streams is None:
        streams = tuple(data.detdef.stream_ids_to_PhSel(i) for i in range(data.detdef.size))
    streams = (streams, ) if isinstance(streams, PhSel) else streams
    streams_kwargs = repeat(dict()) if streams_kwargs is None else streams_kwargs
    fit_kwargs = dict() if fit_kwargs is None else fit_kwargs
    fit_streams_kwargs = repeat(dict()) if fit_streams_kwargs is None else fit_streams_kwargs
    labels = labels if isinstance(labels, Sequence) else repeat(labels)
    fit_labels = fit_labels if isinstance(fit_labels, Sequence) else repeat(fit_labels)
    fit_cps_labels = fit_cps_labels if isinstance(fit_cps_labels, Sequence) else repeat(fit_cps_labels)
    bins = np.arange(tmin, tmax+binwidth, binwidth) if bins is None else bins
    # determine proper peridos
    periods = None
    if bg is None:
        get_times = _sort_times 
    else:
        periods = bg.base_param
        _, _, _, stream_proc = get_pycode_subval('BG_func', bg.params['func'])
        get_times = lambda data, phsel: _get_nth(data, n, Column(periods, 'ph_times', phsel))
    bg = bg if issubclass(bg.tp, BG) else None
    pathcol, line2d = list(), list()
    # loop over each stream
    szip = zip(streams, labels, streams_kwargs, fit_streams_kwargs, fit_labels, fit_cps_labels)
    for stream, label, stream_kwarg, f_stream_kwarg, f_label, f_cps in szip:
        tms = get_times(data, stream)
        hst, bns = np.histogram(np.diff(tms)*data.clk_p, bins)
        rbns = _rescale_value(bns, time_scale)
        if label is True:
            label = stream.tex_str(data.detdef, r'\lambda', data.get_stream_names())
            lbl = f'${label}$'
        else:
            lbl = label
        skw = _dict_update(kwargs, stream_kwarg)
        if label:
            skw['label'] = lbl
        pathcol.append(ax.scatter(rbns[:-1]+np.diff(rbns)/2, hst, **skw))
        if bg is not None:
            if bg.params['auto_threshold']:
                cps, tail_min = _get_nth(data, n, Column(bg, 'bg', stream), 
                                         Column(bg, 'tail_min', stream))
            else:
                tail_min = stream_proc(bg.params.asdict,
                                       data.detdef.get_stream_ids(stream))['tail_min']
                cps = _get_nth(data, n, Column(bg, 'bg', stream))[0]
            i_th = np.searchsorted(bns, tail_min)
            decay = np.exp(-cps*bns)
            dnorm = decay[i_th:].sum()
            if dnorm > 0.0:
                decay *= hst[i_th:].sum()/decay[i_th:].sum()
            else:
                decay *= hst.sum() / decay.sum()
            fkws = _dict_update(fit_kwargs, f_stream_kwarg)
            if 'c' not in fkws and 'color' not in fkws:
                if 'c' in skw:
                    fkws['c'] = skw['c']
                elif 'color' in skw:
                    fkws['color'] = skw['color']
            if f_label is True:
                f_label = label
            if f_label:
                if f_cps:
                    f_label = f'${f_label}$- {cps/1000:.2f} kcps'
                fkws['label'] = f_label
            line2d.append(ax.plot(rbns, decay, **fkws))
    if xlabel is not False:
        xlabel_kwargs = dict() if xlabel_kwargs is None else xlabel_kwargs
        xlabel = f'Interphoton delays $({get_unit_prefix(time_scale)}s)$' if xlabel is None else xlabel
        xttl = ax.set_xlabel(xlabel, **xlabel_kwargs)
    if ylabel is not False:
        ylabel_kwargs = dict() if ylabel_kwargs is None else ylabel_kwargs
        ylabel = '# Delays'
        yttl = ax.set_ylabel(ylabel, **ylabel_kwargs)
    legend_kwargs = dict() if legend_kwargs is None else legend_kwargs
    leg = ax.legend(**legend_kwargs) if set_legend else None
    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)
    return pathcol, line2d, xttl, yttl, leg
