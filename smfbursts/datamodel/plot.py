#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convenience functions for plotting parameters.

Most functions in this module have the follow the basic formula below for their signature: 

``func(data:DataSet|DataSetList, col:Column, [coly:Column, ...], gate:GateGroup=None, ax:plt.Axes=None, include_unit:bool=False, [keyword arguments], **kwargs)``

where func is either the same as the underlying matplotlib function being
called, or close to it. The first argument is the data object, the next are
the column(s) to plot, depeinding on the dimensionality of the plot, and the
first keyword argument is the gate, which if not ``None`` is the gate to applied
to the data, overriding any gates in the input.


.. |nphistogram| replace:: `np.histogram() <https://numpy.org/doc/stable/reference/generated/numpy.histogram.html>`__
.. |gaussian_kde| replace:: `scipy.stats.gaussian_kde <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html>`__
.. |Text| replace:: `plt.Text <https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text>`__
.. |axxlabel| replace:: `ax.set_xlabel <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlabel.html>`__
.. |axylabel| replace:: `ax.set_ylabel <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ylabel.html>`__
"""
from collections.abc import Callable, Sequence
from typing import Union, Any, Literal
from inspect import signature
from itertools import repeat, zip_longest
from numbers import Real, Integral
import re
# from functools import wraps

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde, sem

from .utils import fjit, fnumba, _dict_update
from .tables import DataSet, DataSetList, Param, Column, Gate_, GateGroup
from .multifit import FitFunc, unwrap_param

from smfbursts import rcParams

rcParams['plot.record'] = False



DataS = Union[DataSet,DataSetList]
AxKw = Union[None,plt.Axes]


def _check_ax(ax:AxKw)->plt.Axes:
    """convenience function, if ax is None, return current axes"""
    return plt.gca() if ax is None else ax
    

def _regate(gate:None|Gate_|GateGroup, *args:Param|Column)->list[Param|Column]:
    """
    For normalizing columns to same gate, if gate is None, take intersect of 
    all columns, if gate is not None, regate all columns to gate, columns specified as args
    """
    if gate is not None:
        return [arg.regate(gate) for arg in args]
    gate = args[0].base_gate
    for arg in args[1:]:
        gate &= arg.base_gate
    return [arg.regate(gate) for arg in args]


def _kwdct(kwarg:Union[dict,None])->dict:
    """Similar to _check_ax, converts None to dict for []_kwargs argumments"""
    return dict() if kwarg is None else kwarg


def _rescale_value(val:np.ndarray, factor:Real)->np.ndarray[np.float64]:
    """
    Rescale val array (usually a column) by a constant. For setting column value
    "prefix", If value is integral, treat as power of 10, 
    if floating point multiply by factor.
    """
    if factor == 1.0 and not isinstance(val, Integral):
        return val
    if isinstance(factor, Integral):
        return val*(10**-factor)
    return val/factor


def _get_column(data:DataS, record:bool):
    def _getter(col:Column):
        col = data.record_column(col) if record else data.get_column(col)
        if isinstance(col, tuple):
            col = np.concatenate(col)
        return col
    return _getter


def _get_column_arrays(data:DataS, *cols:Column, gate:GateGroup=None, 
                       include_unit:bool=False, rescale:Sequence[float]=None,
                       record:bool=False)->tuple[list[np.ndarray],list[str]]:
    """Get 2 tuple of ([column arrays,...], [column names, ...]) from input to any plot"""
    cols = _regate(gate, *cols)
    get_col = _get_column(data, rcParams['plot.record'] if record is None else record)
    include_unit = repeat(include_unit) if isinstance(include_unit, bool) else include_unit
    if rescale is None:
        rescale = 1.0
    rescale = tuple(rescale for _ in range(len(cols))) if isinstance(rescale, Real) else rescale
    arrs = [_rescale_value(get_col(col), rs) for col, rs in zip_longest(cols, rescale, fillvalue=1.0)]
    names = [col.name(rs if inc_u else False, data) for col, inc_u, rs in zip(cols, include_unit, rescale)]
    return arrs, names

NormLiteral = Literal[None,"none",'PMF','sum','max','PDF', 'cumulative','icumulative','CDF','iCDF']

def _histcol(data:DataS, col:Column, gate:Union[None,GateGroup], 
             include_unit:bool, rescale:Sequence[float], record:bool, remove_nan:bool,
             normalize:NormLiteral, bins:Union[int,np.ndarray], minmax:tuple[float,float], 
             )->tuple[np.ndarray,np.ndarray,str,str]:
    """
    Histogram columns of data and normalize appropriately, 
    returns bins, hist, cname (column) and olabel (based on normalization type)
    """
    (colarr, ), (cname, ) = _get_column_arrays(data, col, gate=gate, 
                                               include_unit=include_unit, 
                                               rescale=rescale, record=record)
    normalize = 'none' if normalize is None else normalize
    if remove_nan and np.issubdtype(colarr.dtype, np.floating):
        colarr = colarr[~np.isnan(colarr)]
    hst, bns = np.histogram(colarr, bins=bins, range=minmax)
    normalize = normalize.lower()
    if normalize in ('pmf', 'sum'):
        hst = hst / np.sum(hst)
        olabel = 'PMF'
    elif  normalize == 'max':
        hst = hst / np.max(hst)
        olabel = f'# {col.base_param.tp.row_name} / max' if hasattr(col.base_param.tp, 'row_name') else "cnts / max(cnts)"
    elif normalize == 'pdf':
        hst = hst / np.sum(hst) / np.diff(bns)
        olabel = 'PDF'
    elif normalize == 'cumulative':
        hst = np.cumsum(hst)
        olabel = 'cum. cnts'
    elif normalize == 'icumulative':
        hst = np.cumsum(hst[::-1])[::-1]
        olabel = 'inv. cum. cnts'
    elif normalize == 'cdf':
        hst = np.cumsum(hst)
        hst = hst / hst[-1]
        olabel = 'CDF'
    elif normalize == 'icdf':
        hst = np.cumsum(hst[::-1])[::-1]
        hst = hst / hst[0]
        olabel = 'iCDF'
    elif normalize != 'none':
        raise ValueError("normalize must be None or 'none', 'sum', 'max', or 'PDF'")
    else:
        olabel = f'# {col.base_param.tp.row_name}' if hasattr(col.base_param.tp, 'row_name') else 'counts'
    return bns, hst, cname, olabel


OrientLiteral = Literal['vertical', 'horizontal']


def hist_bar(data:DataS, col:Column, gate:GateGroup=None, ax:plt.Axes=None, 
             include_unit:bool=False, rescale:Real=1.0, record:bool=False,
             normalize:NormLiteral=None, 
             remove_nan:bool=True, bins:Union[int,np.ndarray]=100, 
             minmax:tuple[float,float]=None, orientation:OrientLiteral='vertical',
             xlabel:str=None, xlabel_kwargs:dict[str:Any]=None,
             ylabel:str=None, ylabel_kwargs:dict[str:Any]=None,
             **kwargs)->tuple[np.ndarray,np.ndarray,mpl.container.BarContainer,plt.Text,plt.Text]:
    """
    Plot a bar-histogram of the selected column. Automatically sets axes labels
    appropriately, and allows selection of vertical or horizontal bar chart.

    Parameters
    ----------
    data : DataSet | DataSetList
        Data from which to pull column histogram.
    col : Column
        :class:`Column` to retrieve from data and histogram.
    gate : GateGroup, optional
        Gate to apply to column, if None, uses gate of ``col``. The default is None.
    ax : plt.Axes, optional
        Axes in which to plot histogram, if None, use current axes.
        The default is None.
    include_unit : bool, optional
        Whether to include unit in axes label. The default is False.
    rescale : Real, optional
        Factor (if integral, treat as a power of 10) by which to rescale values of col. 
        The default is 1.0.
    record : bool, optional
        Whether to record the column in cache or not. The default is False.
    normalize : NormLiteral, optional
        How to normalize data, can be one of ``'none'``, ``'sum'``, ``'max'``, 
        ``'PDF'``, ``'cumulative'``, ``'icumulativ'``, ``'CDF'``. 
        If ``None``, convert to ``'none'``.
        
            - 'none': no normalization, values are raw counts in each bin (default)
            - 'sum' sum of all bins = 1.
            - 'max' the bin with the largest counts is equal to 1 (max = 1).
            - 'PDF' probability density function, the area under the histogram is 1.
            - 'cumulative' histogram is cumulative counts
            - 'icumulative' histogram is inverse cumulative counts (cumulative starting from max instead of min)
            - 'CDF' cumulative density function
            - 'iCDF' inverse cumulative density function
        
        The default is None.
    remove_nan : bool
        If true, before using np.histogram, remove any nan values. This prevents
        errors when auto-detecting the range. The default is True.
    bins : Union[int,np.ndarray], optional
        Bins to use, handed to |nphistogram|. 
        Either bin edges or number of bins to divide the data into.
        The default is 10.
    minmax : tuple[float,float], optional
        Minimum and maximum values, handed to |nphistogram|.
        as the *range* keyword argument. Note that this is only valid when bins
        is specified as an integer. The default is None.
    xlabel : str, optional
        Name to set xlabel, if None, will automatically assign based on col and normalize.
        The default is None.
    xlabel_kwargs : dict, optional
        Keyword arguments passed to |axxlabel|. 
        The default is None.
    ylabel : str, optional
        Name to set ylabel, if None, will automatically assign based on col and normalize.
        The default is None.
    ylabel_kwargs : dict, optional
        Keyword arguments passed to |axylabel|.
        The default is None.
    orientation : OrientLiteral, optional
        Either 'vertical' or 'horizontal', which way the bars will face. 
        The default is 'vertical'.
    **kwargs : TYPE
        Keyword arguments passed to 
        `plt.bar <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html>`_ 
        or 
        `plt.barh <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.barh.html>`_

    Raises
    ------
    ValueError
        bad specification of orientation.

    Returns
    -------
    hst : np.ndarray[np.number]
        Array of histogram values.
    bns : np.ndarray[np.number]
        Bin edges.
    br : mpl.container.BarContainer
        Matplotib container created of histogram.
    xttl : plt.Text
        |Text| object of xlabel.
    yttl : plt.Text
        |Text| object of ylabel.

    """
    if not isinstance(orientation, str):
        orientation = 'horizontal' if orientation else 'horizontal'
    if orientation not in ('horizontal', 'vertical'):
        raise ValueError("orientation must be 'horizontal' or 'vertical'")
    ax = _check_ax(ax)
    xlabel_kwargs, ylabel_kwargs = _kwdct(xlabel_kwargs), _kwdct(ylabel_kwargs)
    bns, hst, cname, olabel = _histcol(data, col, gate, include_unit, rescale, 
                                       record, remove_nan, normalize, bins, minmax)
    bar = ax.bar if orientation == 'vertical' else ax.barh
    kwargs['width' if orientation == 'vertical' else 'height'] = np.diff(bns)
    kwargs['align'] = 'edge'
    if orientation == 'vertical':
        xlabel = cname if xlabel is None else xlabel
        ylabel = olabel if ylabel is None else ylabel
    else:
        xlabel = olabel if xlabel is None else xlabel
        ylabel = cname if ylabel is None else ylabel
    br = bar(bns[:-1], hst, **kwargs)
    xttl = None if xlabel is False else ax.set_xlabel(xlabel, **xlabel_kwargs)
    yttl = None if ylabel is False else ax.set_ylabel(ylabel, **ylabel_kwargs)
    return hst, bns, br, xttl, yttl


def hist_stair(data:DataS, col:Column, gate:GateGroup=None, ax:plt.Axes=None, 
               include_unit:bool=False, rescale:float=1.0, record:bool=False,
               normalize:NormLiteral=None, 
               remove_nan:bool=True, bins:Union[int,np.ndarray]=100, 
               minmax:tuple[float,float]=None, orientation:OrientLiteral='vertical',
               xlabel:str=None, xlabel_kwargs:dict[str,:Any]=None,
               ylabel:str=None, ylabel_kwargs:dict[str:Any]=None,
               **kwargs)->tuple[np.ndarray,np.ndarray,plt.Line2D,plt.Text,plt.Text]:
    """
    Plot a stair-histogram of the selected column. Automatically sets axes labels
    appropriately, and allows selection of vertical or horizontal bar chart.
    
    Very similar to :func:`hist_bar`, except uses the
    `ax.stairs <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.stairs.html>`_
    function instead of ``ax.bar`` or ``ax.barh``

    Parameters
    ----------
    data : DataSet | DataSetList
        Data from which to pull column histogram.
    col : Column
        :class:`Column` to retrieve from data and histogram.
    gate : GateGroup, optional
        Gate to apply to column, if None, uses gate of ``col``. The default is None.
    ax : plt.Axes, optional
        Axes in which to plot histogram, if None, pull axes from
        `plt.gca() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.gca.html>`_ . 
        The default is None.
    include_unit : bool, optional
        Whether to include unit in axes label. The default is False.
    rescale : float, optional
        Factor (if integral, treat as a power of 10) by which to rescale values of col. 
        The default is 1.0.
    record : bool, optional
        Whether to record the column in cache or not. The default is False.
    normalize : NormLiteral, optional
        How to normalize data, can be one of ``'none'``, ``'sum'``, ``'max'``, 
        ``'PDF'``, ``'cumulative'``, ``'icumulativ'``, ``'CDF'``. 
        If ``None``, convert to ``'none'``.
        
            - 'none': no normalization, values are raw counts in each bin (default)
            - 'sum' sum of all bins = 1.
            - 'max' the bin with the largest counts is equal to 1 (max = 1).
            - 'PDF' probability density function, the area under the histogram is 1.
            - 'cumulative' histogram is cumulative counts
            - 'icumulative' histogram is inverse cumulative counts (cumulative starting from max instead of min)
            - 'CDF' cumulative density function
            - 'iCDF' inverse cumulative density function
        
        The default is None.
    remove_nan : bool
        If true, before using np.histogram, remove any nan values. This prevents
        errors when auto-detecting the range. The default is True.
    bins : Union[int,np.ndarray], optional
        Bins to use, handed to |nphistogram|. 
        Either bin edges or number of bins to divide the data into.
        The default is 10.
    minmax : tuple[float,float], optional
        Minimum and maximum values, handed to |nphistogram|.
        as the *range* keyword argument. Note that this is only valid when bins
        is specified as an integer. The default is None.
    xlabel : str, optional
        Name to set xlabel, if None, will automatically assign based on col and normalize.
        The default is None.
    xlabel_kwargs : dict, optional
        Keyword arguments passed to |axxlabel|. 
        The default is None.
    ylabel : str, optional
        Name to set ylabel, if None, will automatically assign based on col and normalize.
        The default is None.
    ylabel_kwargs : dict, optional
        Keyword arguments passed to |axylabel|.
        The default is None.
    orientation : OrientLiteral, optional
        Either 'vertical' or 'horizontal', which axis will be assigned to the normalized counts.
        The default is 'vertical'.
    **kwargs : TYPE
        Keyword arguments passed to 
        `ax.stairs <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.stairs.html>`_

    Raises
    ------
    ValueError
        bad specification of orientation.

    Returns
    -------
    hst : np.ndarray[np.number]
        Array of histogram values.
    bns : np.ndarray[np.number]
        Bin edges.
    br : plt.Line2D
        Matplotib line2D of histogram
    xttl : plt.Text
        |Text| object of xlabel.
    yttl : plt.Text
        |Text| object of ylabel.

    """
    
    if not isinstance(orientation, str):
        orientation = 'horizontal' if orientation else 'horizontal'
    if orientation not in ('horizontal', 'vertical'):
        raise ValueError("orientation must be 'horizontal' or 'vertical'")
    ax = _check_ax(ax)
    xlabel_kwargs, ylabel_kwargs = _kwdct(xlabel_kwargs), _kwdct(ylabel_kwargs)
    bns, hst, cname, olabel = _histcol(data, col, gate, include_unit, rescale,
                                       record, remove_nan, normalize, bins, minmax)
    if orientation == 'vertical':
        xlabel = cname if xlabel is None else xlabel
        ylabel = olabel if ylabel is None else ylabel
    else:
        xlabel = olabel if xlabel is None else xlabel
        ylabel = cname if ylabel is None else ylabel
    br = ax.stairs(hst, bns, orientation=orientation, **kwargs)
    xttl = None if xlabel is False else ax.set_xlabel(xlabel, **xlabel_kwargs)
    yttl = None if ylabel is False else ax.set_ylabel(ylabel, **ylabel_kwargs)
    return hst, bns, br, xttl, yttl


def hist_line(data:DataS, col:Column, gate:GateGroup=None, ax:plt.Axes=None, 
              include_unit:bool=False, rescale:float=1.0, record:bool=False,
              normalize:NormLiteral=None, remove_nan:bool=True, bins:Union[int,np.ndarray]=100, 
              minmax:tuple[float,float]=None, orientation:OrientLiteral='vertical',
              xlabel:str=None, xlabel_kwargs:dict[str:Any]=None,
              ylabel:str=None, ylabel_kwargs:dict[str:Any]=None,
              **kwargs:Any)->tuple[np.ndarray,np.ndarray,plt.Line2D,plt.Text,plt.Text]:
    """
    Plot a histogram of the selected column as a line. Automatically sets axes labels
    appropriately, and allows selection of vertical or horizontal bar chart.
    
    Very similar to :func:`hist_bar`, except uses the
    `ax.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html#matplotlib.axes.Axes.plot>`_
    function instead of ``ax.bar`` or ``ax.barh``


    Parameters
    ----------
    data : DataSet | DataSetList
        Data from which to pull column histogram.
    col : Column
        :class:`Column` to retrieve from data and histogram.
    gate : GateGroup, optional
        Gate to apply to column, if None, uses gate of ``col``. The default is None.
    ax : plt.Axes, optional
        Axes in which to plot histogram, if None, pull axes from
        `plt.gca() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.gca.html>`_ . 
        The default is None.
    include_unit : bool, optional
        Whether to include unit in axes label. The default is False.
    rescale : float, optional
        Factor (if integral, treat as a power of 10) by which to rescale values of col. 
        The default is 1.0.
    record : bool, optional
        Whether to record the column in cache or not. The default is False.
    normalize : NormLiteral, optional
        How to normalize data, can be one of ``'none'``, ``'sum'``, ``'max'``, 
        ``'PDF'``, ``'cumulative'``, ``'icumulativ'``, ``'CDF'``. 
        If ``None``, convert to ``'none'``.
        
            - 'none': no normalization, values are raw counts in each bin (default)
            - 'sum' sum of all bins = 1.
            - 'max' the bin with the largest counts is equal to 1 (max = 1).
            - 'PDF' probability density function, the area under the histogram is 1.
            - 'cumulative' histogram is cumulative counts
            - 'icumulative' histogram is inverse cumulative counts (cumulative starting from max instead of min)
            - 'CDF' cumulative density function
            - 'iCDF' inverse cumulative density function
        
        The default is None.
    bins : Union[int,np.ndarray], optional
        Bins to use, handed to |nphistogram|. 
        Either bin edges or number of bins to divide the data into.
        The default is 10.
    minmax : tuple[float,float], optional
        Minimum and maximum values, handed to |nphistogram|.
        as the *range* keyword argument. Note that this is only valid when bins
        is specified as an integer. The default is None.
    xlabel : str, optional
        Name to set xlabel, if None, will automatically assign based on col and normalize.
        The default is None.
    xlabel_kwargs : dict, optional
        Keyword arguments passed to |axxlabel|. 
        The default is None.
    ylabel : str, optional
        Name to set ylabel, if None, will automatically assign based on col and normalize.
        The default is None.
    ylabel_kwargs : dict, optional
        Keyword arguments passed to |axylabel|.
        The default is None.
    orientation : OrientLiteral, optional
        Either 'vertical' or 'horizontal', which axis will be assigned to the normalized counts.
        The default is 'vertical'.
    **kwargs : TYPE
        Keyword arguments passed to 
        `ax.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html#matplotlib.axes.Axes.plot>`_

    Raises
    ------
    ValueError
        bad specification of orientation.

    Returns
    -------
    hst : np.ndarray[np.number]
        Array of histogram values.
    bns : np.ndarray[np.number]
        Bin edges.
    br : plt.Line2D
        Matplotib line2D of histogram
    xttl : plt.Text
        |Text| object of xlabel.
    yttl : plt.Text
        |Text| object of ylabel.
    
    """
    if not isinstance(orientation, str):
        orientation = 'horizontal' if orientation else 'horizontal'
    if orientation not in ('horizontal', 'vertical'):
        raise ValueError("orientation must be 'horizontal' or 'vertical'")
    ax = _check_ax(ax)
    xlabel_kwargs, ylabel_kwargs = _kwdct(xlabel_kwargs), _kwdct(ylabel_kwargs)
    bns, hst, cname, olabel = _histcol(data, col, gate, include_unit, rescale,
                                       record, remove_nan, normalize, bins, minmax)
    bcntrs = (bns[:-1] + bns[1:]) / 2
    if orientation == 'vertical':
        xlabel = cname if xlabel is None else xlabel
        ylabel = olabel if ylabel is None else ylabel
    else:
        xlabel = olabel if xlabel is None else xlabel
        ylabel = cname if ylabel is None else ylabel
    if orientation == 'vertical':
        lns = ax.plot(bcntrs, hst, **kwargs)[0]
    else:
        lns =  ax.plot(hst, bcntrs, **kwargs)[0]
    xttl = None if xlabel is False else ax.set_xlabel(xlabel, **xlabel_kwargs)
    yttl = None if ylabel is False else ax.set_ylabel(ylabel, **ylabel_kwargs)
    return hst, bns, lns, xttl, yttl


def hist_kdeoverlay(data:DataS, col:Column, gate:GateGroup=None, ax:plt.Axes=None, 
                    include_unit:bool=False, rescale:float=1.0, record:bool=False,
                    hist_func:Callable=hist_bar, normalize:NormLiteral=None, 
                    remove_nan:bool=True, bins:Union[int,np.ndarray]=100, 
                    kde_bins:Union[int, np.ndarray]=500, minmax:tuple[float,float]=None, 
                    orientation:OrientLiteral='vertical',
                    xlabel:str=None, xlabel_kwargs:dict[str:Any]=None,
                    ylabel:str=None, ylabel_kwargs:dict[str:Any]=None,
                    kde_kwargs:dict[str:Any]=None, kdeplot_kwargs:dict[str:Any]=None, 
                    **kwargs:Any)->tuple[np.ndarray,np.ndarray,plt.Line2D,plt.Text,plt.Text]:
    """
    Plot a 1-D histogram with a KDE-line overlay. Function ensures KDE overlay
    have matching amplitudes (ie KDE scaled as histogram is normalized).

    Parameters
    ----------
    data : DataSet | DataSetList
        Data from which to pull column histogram.
    col : Column
        :class:`Column` to retrieve from data and histogram.
    gate : GateGroup, optional
        Gate to apply to column, if None, uses gate of ``col``. The default is None.
    ax : plt.Axes, optional
        Axes in which to plot histogram, if None, pull axes from
        `plt.gca() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.gca.html>`_ . 
        The default is None.
    include_unit : bool, optional
        Whether to include unit in axes label. The default is False.
    rescale : float, optional
        Factor (if integral, treat as a power of 10) by which to rescale values of col. 
        The default is 1.0.
    record : bool, optional
        Whether to record the column in cache or not. The default is False.
    hist_func : Callable, optional
        **smfBursts** function call to plot histogram. **Note** this is **not** 
        the matplotlib function. The default is :func:`hist_bar`.
    normalize : NormLiteral, optional
        How to normalize data, can be one of ``'none'``, ``'PMF'``, ``'max'``, 
        ``'PDF'``, ``'cumulative'``, ``'icumulativ'``, ``'CDF'``. 
        If ``None``, convert to ``'none'``.
        
            - 'none': no normalization, values are raw counts in each bin (default)
            - 'sum' sum of all bins = 1.
            - 'max' the bin with the largest counts is equal to 1 (max = 1).
            - 'PDF' probability density function, the area under the histogram is 1.
            - 'cumulative' histogram is cumulative counts
            - 'icumulative' histogram is inverse cumulative counts (cumulative starting from max instead of min)
            - 'CDF' cumulative density function
            - 'iCDF' inverse cumulative density function
        
        The default is None.
    remove_nan : bool, optional
        Whether to mask nan values in columns during evaluation, if column contains
        nans, if False, may result in errors. The default is True.
    bins : Union[int,np.ndarray], optional
        Number of bins or bin edges in/of histogram. The default is 100.
    kde_bins : Union[int, np.ndarray], optional
        Bin/evaluation points of KDE, same basic behavior as bins. The default is 500.
    minmax : tuple[float,float], optional
        Passed to hist_func, Minimum and maximum values, handed to |nphistogram|. 
        as the *range* keyword argument. Note that this is only valid when bins
        is specified as an integer. The default is None.
    orientation : OrientLiteral, optional
        Passed to hist_func, Either 'vertical' or 'horizontal', which axis will 
        be assigned to the normalized counts.
        The default is 'vertical'.
    xlabel : str, optional
        Name to set xlabel, if None, will automatically assign based on col and normalize.
        The default is None.
    xlabel_kwargs : dict, optional
        Keyword arguments passed to |axxlabel|.
        The default is None.
    ylabel : str, optional
        Name to set ylabel, if None, will automatically assign based on col and normalize.
        The default is None.
    ylabel_kwargs : dict, optional
        Keyword arguments passed to |axylabel|.
        The default is None.
    kde_kwargs : dict[str:Any], optional
        Keywords arguments passed to |gaussian_kde|.
        The default is None.
    kdeplot_kwargs : dict[str:Any], optional
        Keyword arguments passed to 
        `ax.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_. 
        The default is None.
    **kwargs : Any
        Keyword arguments passed to histogram plotting function (passed through hist_func).

    Returns
    -------
    h : np.ndarray[np.number]
        Array of histogram values.
    b : np.ndarray[np.number]
        Bin edges.
    p : plt.Line2D
        Matplotib line2D of histogram
    tx : plt.Text
        |Text| object of xlabel.
    ty : plt.Text
        |Text| object of ylabel.
    k : plt.Line2D
        Matplotib line2D of kde plott

    """
    ax = _check_ax(ax)
    h, b, p, tx, ty = hist_func(data, col, gate=gate, ax=ax, 
                                include_unit=include_unit, rescale=rescale, 
                                record=record,
                                normalize=normalize, remove_nan=remove_nan,
                                bins=bins, minmax=minmax, orientation=orientation, 
                                xlabel=xlabel, xlabel_kwargs=xlabel_kwargs, 
                                ylabel=ylabel, ylabel_kwargs=ylabel_kwargs, **kwargs)
    kde_bins = np.linspace(b[0], b[-1], kde_bins) if isinstance(kde_bins, Integral) else kde_bins
    kde_kwargs = dict() if kde_kwargs is None else kde_kwargs
    col = data.get_column(col, gate)
    y = gaussian_kde(col[(~np.isnan(col))&(col>=b[0])&(col<=b[-1])], **kde_kwargs).evaluate(kde_bins)
    kde_args = kde_bins, y*(h*np.diff(b)).sum()
    kde_args = kde_args[::-1] if orientation == 'horizontal' else kde_args
    kdeplot_kwargs = dict() if kdeplot_kwargs is None else kdeplot_kwargs
    k = ax.plot(*kde_args, **kdeplot_kwargs)
    return h, b, p, tx, ty, k


def hist2d(data:DataS, colx:Column, coly:Column, gate:GateGroup=None, ax:plt.Axes=None, 
                 include_unit:bool=False, rescale:tuple[float, float]=None,
                 record:bool=False,
                 xlabel:str=None, xlabel_kwargs:dict[str:Any]=None, 
                 ylabel:str=None, ylabel_kwargs:dict[str:Any]=None, 
                 **kwargs:Any):
    """
    Show color-map image of 2D histogram of ``colx`` and ``coly``.
    
    Internally relies on 
    `ax.hist2d <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist2d.html>`_
    

    Parameters
    ----------
    data : DataS
        Data on which columns are based.
    colx : Column
        :class:`Column` assigned to X axis.
    coly : Column
        :class:`Column` assigned to Y axis.
    gate : GateGroup, optional
        If specified, colx and coly will be regated to gate, defining which points
        (rows) included. The default is None.
    ax : plt.Axes, optional
        Axes in which to plot histogram, if None, pull axes from
        `plt.gca() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.gca.html>`_ . 
        The default is None.
    include_unit : bool, optional
        Whether to include unit in axes label. The default is False.
    rescale : tuple[float, float] optional
        Factors (usually powers of 10) by which to rescale values of colx and coly. 
        If not specified, default is converted to (1.0, 1.0).
        The default is None.
    record : bool, optional
        Whether to record the column in cache or not. The default is False.
    xlabel_kwargs : dict, optional
        Keyword arguments passed to |axxlabel|.
        The default is None.
    ylabel : str, optional
        Name to set ylabel, if None, will automatically assign based on col and normalize.
        The default is None.
    ylabel_kwargs : dict, optional
        Keyword arguments passed to |axylabel|.
        The default is None.
    **kwargs : Any
        Key word argumetns passed to
        `ax.hist2d <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist2d.html>`_

    Returns
    -------
    h : np.ndarray
        Values of 2d histogram
    xedges : np.ndarray
        bin edges of x axis
    yedges : np.ndarray
        bin edges of y axis
    image : mpl.collections.QuadMesh
        Image object
    xttl : plt.Text
        |Text| object of xlabel.
    yttl : plt.Text
        |Text| object of ylabel.
    
    """
    ax = _check_ax(ax)
    (cx, cy), (nx, ny) = _get_column_arrays(data, colx, coly, gate=gate, 
                                            include_unit=include_unit, 
                                            rescale=rescale, record=record)
    xlabel = nx if xlabel is None else xlabel
    xlabel_kwargs = dict() if xlabel_kwargs is None else xlabel_kwargs
    ylabel = ny if ylabel is None else ylabel
    ylabel_kwargs = dict() if ylabel_kwargs is None else ylabel_kwargs
    out = ax.hist2d(cx, cy, **kwargs)
    xttl = None if xlabel is False else ax.set_xlabel(xlabel, **xlabel_kwargs)
    yttl = None if ylabel is False else ax.set_ylabel(ylabel, **ylabel_kwargs)
    return out + (xttl, yttl)


def hist(data:DataS, *args:Column, gate:GateGroup=None, ax:plt.Axes=None, 
         style:Literal[None,'auto', '2D','line','stair','bar']=None, 
         orientation:OrientLiteral='vertical', **kwargs:Any)->tuple[np.ndarray,np.ndarray,Any,plt.Text,plt.Text]:
    """
    Plot 1 or 2 columns as a histogram. This function selects the plot style
    and calls one of :func:`hist_bar`, :func:`hist_stair`, :func:`hist_line` or
    :func:`hist2d` depending on inputs.


    Parameters
    ----------
    data : DataS
        Data on which columns are based..
    *args : Column
        :class:`Column` of axis/axes, if single column given, plot 1D histogram,
        if 2, plot 2D histogram (colormap).
    gate : GateGroup, optional
        If specified, colx and coly will be regated to gate, defining which points
        (rows) included. The default is None.
    ax : plt.Axes, optional
        Axes in which to plot histogram, if None, pull axes from
        `plt.gca() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.gca.html>`_ . 
        The default is None.
    style : Literal[None,'auto', '2D','line','stair','bar'], optional
        String indicating type of plot to use, 
        if None/auto, choose either 2D or bar based on whether 2 or 1 column specified. 
        The default is None.
    orientation : OrientLiteral, optional
        DESCRIPTION. The default is 'vertical'.
    **kwargs : Any
        Additional keyword arguments passed to plotting function.

    Raises
    ------
    ValueError
        Invalid style string.

    Returns
    -------
    hst : np.ndarray[np.number]
        Array of histogram values.
    bns : np.ndarray[np.number]
        Bin edges.
    br : plt.Line2D
        Matplotib line2D of histogram
    xttl : plt.Text
        |Text| object of xlabel.
    yttl : plt.Text
        |Text| object of ylabel.
    """
    if style is None or style == 'auto':
        style = '2D' if len(args) == 2 else 'bar'
    if style == '2D':
        return hist2d(data, *args, gate=gate, ax=ax, **kwargs)
    elif style == 'line':
        return hist_line(data, *args, gate=gate, ax=ax, orientation=orientation, **kwargs)
    elif style == 'stair':
        return hist_stair(data, *args, gate=gate, ax=ax, orientation=orientation, **kwargs)
    elif style == 'bar':
        return hist_bar(data, *args, gate=gate, ax=ax, orientation=orientation, **kwargs)
    raise ValueError(f"'{style} must be 'auto', 'line', 'stair', or 'bar'")


def hexbin(data:DataS, colx:Column, coly:Column, gate:GateGroup=None, ax:plt.Axes=None, 
           include_unit:bool=False, rescale:tuple[float, float]=None, record:bool=False,
           xlabel:str=None, xlabel_kwargs:dict[str:Any]=None, 
           ylabel:str=None, ylabel_kwargs:dict[str:Any]=None, 
           **kwargs)->tuple[mpl.collections.PolyCollection, plt.Text, plt.Text]:
    """
    Plot hexagonal 2-D histogram of ``colx`` and ``coly``.
    
    Internally relies on 
    `ax.hexbin <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hexbin.html>`_
    

    Parameters
    ----------
    data : DataS
        Data on which columns are based.
    colx : Column
        :class:`Column` assigned to X axis.
    coly : Column
        :class:`Column` assigned to Y axis.
    gate : GateGroup, optional
        If specified, colx and coly will be regated to gate, defining which points
        (rows) included. The default is None.
    ax : plt.Axes, optional
        Axes in which to plot histogram, if None, pull axes from
        `plt.gca() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.gca.html>`_ . 
        The default is None.
    include_unit : bool, optional
        Whether to include unit in axes label. The default is False.
    rescale : tuple[float, float] optional
        Factors (usually powers of 10, if integral, treat as a power of 10) 
        by which to rescale values of colx and coly. 
        If not specified, default is converted to (1.0, 1.0).
        The default is None.
    record : bool, optional
        Whether to record the column in cache or not. The default is False.
    xlabel_kwargs : dict, optional
        Keyword arguments passed to |axxlabel|. 
        The default is None.
    ylabel : str, optional
        Name to set ylabel, if None, will automatically assign based on col and normalize.
        The default is None.
    ylabel_kwargs : dict, optional
        Keyword arguments passed to |axylabel|.
        The default is None.
    **kwargs : Any
        Key word argumetns passed to
        `ax.hexbin <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hexbin.html>`_

    Returns
    -------
    pc : mpl.collection.PolyCollection
        Output of `call to
        ax.hexbin <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hexbin.html>`_
    xttl : plt.Text
        |Text| object of xlabel.
    yttl : plt.Text
        |Text| object of ylabel.
    
    """
    ax = _check_ax(ax)
    (cx, cy), (nx, ny) = _get_column_arrays(data, colx, coly, gate=gate, 
                                            include_unit=include_unit, 
                                            rescale=rescale, record=record)
    xlabel = nx if xlabel is None else xlabel
    xlabel_kwargs = dict() if xlabel_kwargs is None else xlabel_kwargs
    ylabel = ny if ylabel is None else ylabel
    ylabel_kwargs = dict() if ylabel_kwargs is None else ylabel_kwargs
    pc = ax.hexbin(cx, cy, **kwargs)
    xttl = None if xlabel is False else ax.set_xlabel(xlabel, **xlabel_kwargs)
    yttl = None if ylabel is False else ax.set_ylabel(ylabel, **ylabel_kwargs)
    return pc, xttl, yttl


def _minmax_rescale(vals:np.ndarray, minzero:bool, maxone:bool)->np.ndarray:
    """Function to make method universal, rescales vals according to minzero
    and maxone arguments, if either is True, the min or max is ensured to be
    zero or one, respectively, uses nanmin/max so that nan values ignored"""
    if minzero:
        vals = vals - np.nanmin(vals)
    if maxone:
        vals = vals / np.nanmax(vals)
    return vals


def density_kde(colx:np.ndarray, coly:np.ndarray, minzero:bool=True, maxone:bool=True, 
                **kwargs:Any)->dict[str:np.ndarray]:
    """
    Function for use as value of ``point_func`` of :func:`scatter`.
    
    Create dictionary to color a scatter plot of arrays colx and coly based on
    gaussian kde.

    Parameters
    ----------
    colx : np.ndarray
        Array of X values.
    coly : np.ndarray
        Array of Y values.
    minzero : bool, optional
        Whether to rescale so that color values have minimum of 0. 
        The default is True.
    maxone : bool, optional
        Whether to rescale so that color values have maximum of 1. 
        The default is True.
    **kwargs : Any
        Keword arguments for |gaussian_kde| used to compute color argument.

    Returns
    -------
    dict
        Dictionary of kwargs for scatter plot with color based on kde.

    """
    mask = ~np.isnan(colx) & ~np.isnan(coly)
    if np.all(mask):
        xy = np.vstack([colx, coly])
        c = gaussian_kde(xy, **kwargs).evaluate(xy)
        return dict(c=_minmax_rescale(c, minzero, maxone))
    xy = np.vstack([colx, coly])[:,mask]
    ctemp = gaussian_kde(xy, **kwargs).evaluate(xy)
    c = np.zeros(mask.size, dtype=xy.dtype)
    c[mask] = _minmax_rescale(ctemp, minzero, maxone)
    return dict(c=c)


def rescale_size(col:np.ndarray[np.float64])->np.ndarray[np.float64]:
    """
    Function for use as value of ``rescale_func`` kwarg in :func:`density_size`
    and :func:`density_kdesize` functions.

    Parameters
    ----------
    col : np.ndarray[np.float64]
        Values to rescale.

    Returns
    -------
    np.ndarray[np.float64]
        col rescaled so min is 0.0 and max is 1.0.

    """
    return _minmax_rescale(col, True, True)


def point_size(colx:np.ndarray, coly:np.ndarray, cols:np.ndarray,
                 scale_factor:float=5.0, 
                 rescale_func:Callable[[np.ndarray],np.ndarray]=rescale_size)->dict:
    """
    Function to use as value of ``point_func`` kwarg in :func:`scatter` to
    size dots according to point_cols argument (should be a single :class:`Column`).

    Parameters
    ----------
    colx : np.ndarray
        Ignorred, array of X column.
    coly : np.ndarray
        Ignorred, array of Y column.
    cols : np.ndarray
        array of column to define size.
    scale_factor : float, optional
        Factor to multiply rescale_func by. The default is 5.0.
    rescale_func : Callable[[np.ndarray],np.ndarray], optional
        Function which takes single array as only argument and returns rescaled
        array (e.g. normalized so all values are from 0 to 1 in the case of default). 
        The default is :func:`rescale_size`
    
    Returns
    -------
    dict
        Dict ``dict(c=rescale_func(cols)*scale_factor)``.

    """
    return dict(s=rescale_func(cols)*scale_factor)


def density_kdesize(colx:np.ndarray, coly:np.ndarray, cols:np.ndarray,
                    kdefunc:Callable=density_kde, 
                    sizefunc:Callable=point_size,
                    kdekwargs=None, sizekwargs=None)->dict[str:np.ndarray]:
    """
    Function to use as value of ``point_func`` kwarg in :func:`scatter` to
    cause color to be determine by gaussian kde, and size by cols argument rescaled
    by ``rescale_func``.

    Parameters
    ----------
    colx : np.ndarray
        Array of X column.
    coly : np.ndarray
        Array of Y column.
    cols : np.ndarray
        Array of column to use to determine size of dots.
    kdeminzero : bool, optional
        Whether color values rescaled so minimum is 0. The default is True.
    kdemaxone : bool, optional
        Whether color values rescaled so minimum is 0. The default is True.
    scale_factor : float, optional
        Factor by which to rescale ``rescale_func(cols)`` by to determine size.
        The default is 5.0.

    Returns
    -------
    out : dict[str:np.ndarray]
        Combined dict of kdefunc and sizefunc arrays.

    """
    kdekwargs = dict() if kdekwargs is None else kdekwargs
    sizekwargs = dict() if sizefunc is None else sizefunc
    out = kdefunc(colx, coly, **kdekwargs)
    out.update(sizefunc(colx, coly, cols, **sizekwargs))
    return out


def colorcategory(*args:np.ndarray, cmap:str|mpl.colors.Colormap=None, ncat:int|None|bool=None, reduce:bool=False, **kwargs)->dict[str:Any]:
    """
    Function to use as value of ``point_func`` kwarg in :func:`scatter` to color
    each row based on the value of a third column 
    (supplied in the ``point_cols`` kwarg).

    Parameters
    ----------
    *args : np.ndarray
        Column values os catter plot, function uses last column to assign category
        colors.
    cmap : str|mpl.colors.Colormap, optional
        Colormap to use to recolor column. If specified, uses the create a
        listed colormap based on categorical values. Should be a tabular colormap.
        The default is None.
    ncat : int, optional
        Number of exected categories
    reduce: bool, optional
        
    Returns
    -------
    out : dict[str:Any]
        Kwargs dictionary with key 'c' array as inverse of unique values of last arg.
        If cmap is set, adds 'cmap', 'vmin', and 'vmax' keys. 

    """
    out = dict()
    out.update(kwargs)
    cat = args[-1]
    if reduce:
        uni, cat = np.unique(cat, return_inverse=True)
    if cmap is not None:
        cmap = plt.colormaps[cmap] if isinstance(cmap, str) else cmap
    if isinstance(cmap, mpl.colors.ListedColormap):
        if ncat is True:
            ncat = np.max(cat)
        if ncat:
            cmap = mpl.colors.ListedColormap([c for c in cmap.colors[:ncat]])
        out.setdefault('vmin', 0)
        out.setdefault('vmax', len(cmap.colors)-1)
    out['c'] = cat
    if cmap is not None:
        out['cmap'] = cmap
    return out


def scatter(data:DataS, colx:Column, coly:Column, gate:GateGroup=None, ax:plt.Axes=None,
            include_unit:bool=True, rescale:tuple[float,float]=None, record:bool=False,
            point_func:Callable[[np.ndarray,np.ndarray,...],dict[str:Any]]=None, 
            point_cols:Union[Column,tuple[Column,...]]=None, point_kwargs:dict[str:Any]=None,
            xlabel:str=None, xlabel_kwargs:dict[str:Any]=None, 
            ylabel:str=None, ylabel_kwargs:dict[str:Any]=None,
            **kwargs)->tuple[mpl.collections.PathCollection,plt.Text,plt.Text]:
    r"""
    Scatter plot of columns specified by ``colx`` and ``coly`` of ``data``

    Parameters
    ----------
    data : DataS
        Data on which columns are based.
    colx : Column
        :class:`Column` assigned to X axis.
    coly : Column
        :class:`Column` assigned to Y axis.
    gate : GateGroup, optional
        If specified, colx and coly will be regated to gate, defining which points
        (rows) included. The default is None.
    ax : plt.Axes, optional
        Axes in which to plot histogram, if None, pull axes from
        `plt.gca() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.gca.html>`_ . 
        The default is None.
    include_unit : bool, optional
        Whether to include unit in axes label. The default is False.
    rescale : tuple[float,float], optional
        Factors by which (usually powers of 10) to rescale values of colx and coly.
        If None, defaults to (1.0, 1.0). The default is None.
    record : bool, optional
        Whether to record the column in cache or not. The default is False.
    point_func : Callable[[np.ndarray,np.ndarray,...],dict[str:Any]], optional
        Callable that takes arrays from colx, coly, and those specified in 
        ``point_cols`` as args, and point_kwargs as kwargs, and returns dictionary
        of keyowrd arguments to pass to
        `ax.scatter <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html>`_ . 
        The default is None.
    point_cols : Union[Column,tuple[Column,...]], optional
        Additional columns, if any to pass to point_func as ``*args``. The default is None.
    point_kwargs : dict[str:Any], optional
        Keyword arguments to pass to point_func. The default is None.
    xlabel : str, optional
        Name to set xlabel, if None, will automatically assign based on col and normalize.
        The default is None.
    xlabel_kwargs : dict, optional
        Keyword arguments passed to |axxlabel|. 
        The default is None.
    ylabel : str, optional
        Name to set ylabel, if None, will automatically assign based on col and normalize.
        The default is None.
    ylabel_kwargs : dict, optional
        Keyword arguments passed to |axylabel|.
        The default is None.
    **kwargs : Any
        keyword arguments passed to
        `ax.scatter() <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html>`_ .

    Returns
    -------
    pathcollection : mpl.collections.PathCollection
        Path collection of scatter plot points.
    xttl : plt.Text
        |Text| object of xlabel.
    yttl : plt.Text
        |Text| object of ylabel.
    
    """
    ax = _check_ax(ax)
    xlabel_kwargs, ylabel_kwargs = _kwdct(xlabel_kwargs), _kwdct(ylabel_kwargs)
    point_cols = tuple() if point_cols is None else point_cols
    point_cols = point_cols if isinstance(point_cols, tuple) else (point_cols, )
    cols, names = _get_column_arrays(data, colx, coly, *point_cols, gate=gate, 
                                     include_unit=include_unit, 
                                     rescale=rescale, record=record)
    colx, coly, colsp = cols[0], cols[1], cols[2:]
    namex, namey, _ = names[0], names[1], names[2:]
    if point_func is not None:
        point_kwargs = _kwdct(point_kwargs)
        kwargs.update(point_func(colx, coly, *colsp, **point_kwargs))
    pathcollection = ax.scatter(colx, coly, **kwargs)
    xlabel = namex if xlabel is None else xlabel
    ylabel = namey if ylabel is None else ylabel
    xttl = None if xlabel is False else ax.set_xlabel(xlabel, **xlabel_kwargs)
    yttl = None if ylabel is False else ax.set_ylabel(ylabel, **ylabel_kwargs)
    return pathcollection, xttl, yttl


def _mean_intervalrange(l:float, h:float, thresh:int, bn:np.ndarray, av:np.ndarray)->float:
    """Find mean of av for bn between l and h if more than thresh"""
    m = (l < bn) & (bn < h)
    if m.sum() < thresh:
        return np.nan
    return np.nanmean(av[m])


def _mean_interval(carrbin:np.ndarray, carrav:np.ndarray, 
                   bins:np.ndarray, thresh:int)->np.ndarray[np.float64]:
    """Find mean of intervals of points between bins"""
    return np.array([_mean_intervalrange(l, h, thresh, carrbin, carrav) 
                     for l, h in zip(bins[:-1],bins[1:])])
    

def _mean_intervalplot(data:DataS, colbin:Column, colav:Column, gate:GateGroup=None, 
                       ax:plt.Axes=None, bins:np.ndarray|int=10, include_unit:bool=True,
                       rescale:tuple[float,float]=None, record:bool=False,
                       thresh:int=20, 
                       orientation:Literal['vertical','horizontal']='vertical',
                       xlabel:str=False, xlabel_kwargs:dict[str:Any]=None, 
                       ylabel:str=False, ylabel_kwargs:dict[str:Any]=None,
                       func:str='plot',
                       **kwargs)->tuple[Any,plt.Text,plt.Text]:
    """Plot mean interval of one column based on another"""
    ax = _check_ax(ax)
    (cbin, cav), names = _get_column_arrays(data, colbin, colav, gate=gate, 
                                            include_unit=include_unit, 
                                            rescale=rescale, record=record)
    if not isinstance(bins, np.ndarray):
        bins = np.linspace(np.nanmin(cbin), np.nanmax(cbin), bins+1)
    av = _mean_interval(cbin, cav, bins, thresh)
    cbins = bins[:-1]+np.diff(bins)/2
    args = (cbins, av) if orientation == 'vertical' else (av, cbins)
    names = names if orientation == 'vertical' else names[::-1]
    out = getattr(ax, func)(*args, **kwargs)
    if xlabel is not False:
        xlabel = names[0] if xlabel is None else xlabel
        xlabel_kwargs = _kwdct(xlabel_kwargs)
        xlbl = ax.set_xlabel(xlabel, **xlabel_kwargs)
    else:
        xlbl = None
    if ylabel is not False:
        ylabel = names[1] if ylabel is None else xlabel
        ylabel_kwargs = _kwdct(ylabel_kwargs)
        ylbl = ax.set_ylabel(ylabel, **ylabel_kwargs)
    else:
        ylbl = None
    return out, xlbl, ylbl, av, cbins
        

def plot_meaninterval(data:DataS, colbin:Column, colav:Column, gate:GateGroup=None, 
                      ax:plt.Axes=None, bins:np.ndarray|int=10, include_unit:bool=True,
                      rescale:tuple[float,float]=None, record:bool=False,
                      thresh:int=20, 
                      orientation:Literal['vertical','horizontal']='vertical',
                      xlabel:str=False, xlabel_kwargs:dict[str:Any]=None, 
                      ylabel:str=False, ylabel_kwargs:dict[str:Any]=None,
                      **kwargs)->tuple[plt.Line2D,plt.Text,plt.Text, np.ndarray,np.ndarray]:
    """
    Plots the mean of rows in intervals. That is, plot the mean values of column
    colav for the rows that are in the ranges defined by bins of colbin.

    Parameters
    ----------
    data : DataS
        Data on which columns are based.
    colbin : Column
        Column to use to define ranges within bins.
    colav : Column
        Column to use as source of values for computing mean.
    gate : GateGroup, optional
        Gate to apply to colav and colbin, if not specified, use intersect of
        gates of colav and colbin. The default is None.
    ax : plt.Axes, optional
        Axes in which to plot histogram, if None, pull axes from
        `plt.gca() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.gca.html>`_ . 
        The default is None.
    bins : np.ndarray|int, optional
        Continuous bin ranges to use to group rows by value of colbin. The default is 10.
    include_unit : bool, optional
        Whether to include unit in axes label. The default is False.
    rescale : tuple[float,float], optional
        Factors by which (usually powers of 10) to rescale values of colx and coly.
        If None, defaults to (1.0, 1.0). The default is None.
    record : bool, optional
        Whether to record the column in cache or not. The default is False.
    thresh : int, optional
        Minimum number of rows in a given range for colbin in order for point to
        be plotted. If number of rows is below, that bin will be given a NaN. 
        The default is 20.
    orientation : Literal['vertical','horizontal'], optional
        Orientation to plot values. 'vertical' will place colbin in the xaxis,
        while 'horizontal' will place colbin in the y-axis.
        The default is 'vertical'.
    xlabel : str, optional
        Name to set xlabel, if None, will automatically assign based on col and normalize.
        The default is None.
    xlabel_kwargs : dict, optional
        Keyword arguments passed to |axxlabel|.
        The default is None.
    ylabel : str, optional
        Name to set ylabel, if None, will automatically assign based on col and normalize.
        The default is None.
    ylabel_kwargs : dict, optional
        Keyword arguments passed to |axylabel|.
        The default is None.
    **kwargs : Any
        Additional kwargs passed directly to
        `ax.plot() <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_ .

    Returns
    -------
    line : plt.Line2D
        Matplotlib Line2D of mean values.
    xlbl : plt.Text
        |Text| object of xlabel.
    ylbl : plt.Text
        |Text| object of ylabel.
    av : np.ndarray[np.float64]
        Array of computed mean values of colav.
    cbins : np.ndarray[np.float64]
        Array of bin centers.

    """
    return _mean_intervalplot(data, colbin, colav, gate=gate, ax=ax, bins=bins, 
                              include_unit=include_unit, rescale=rescale, record=record,
                              thresh=thresh, orientation=orientation, xlabel=xlabel,
                              xlabel_kwargs=xlabel_kwargs, ylabel=ylabel_kwargs, 
                              func='plot', **kwargs)


def scatter_meaninterval(data:DataS, colav:Column, colbin:Column, gate:GateGroup=None, 
                         ax:plt.Axes=None, bins:np.ndarray|int=10, include_unit:bool=True,
                         rescale:tuple[float,float]=None, record:bool=False,
                         thresh:int=20, 
                         orientation:Literal['vertical','horizontal']='vertical',
                         xlabel:str=False, xlabel_kwargs:dict[str:Any]=None, 
                         ylabel:str=False, ylabel_kwargs:dict[str:Any]=None,
                         **kwargs)->tuple[mpl.collections.PathCollection,plt.Text,plt.Text,np.ndarray,np.ndarray]:
    """
    Scatter plot the mean of rows in intervals. That is, plot the mean values of column
    colav for the rows that are in the ranges defined by bins of colbin.

    Parameters
    ----------
    data : DataS
        Data on which columns are based.
    colbin : Column
        Column to use to define ranges within bins.
    colav : Column
        Column to use as source of values for computing mean.
    gate : GateGroup, optional
        Gate to apply to colav and colbin, if not specified, use intersect of
        gates of colav and colbin. The default is None.
    ax : plt.Axes, optional
        Axes in which to plot histogram, if None, pull axes from
        `plt.gca() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.gca.html>`_ . 
        The default is None.
    bins : np.ndarray|int, optional
        Continuous bin ranges to use to group rows by value of colbin. The default is 10.
    include_unit : bool, optional
        Whether to include unit in axes label. The default is False.
    rescale : tuple[float,float], optional
        Factors by which (usually powers of 10) to rescale values of colx and coly.
        If None, defaults to (1.0, 1.0). The default is None.
    record : bool, optional
        Whether to record the column in cache or not. The default is False.
    thresh : int, optional
        Minimum number of rows in a given range for colbin in order for point to
        be plotted. If number of rows is below, that bin will be given a NaN. 
        The default is 20.
    orientation : Literal['vertical','horizontal'], optional
        Orientation to plot values. 'vertical' will place colbin in the xaxis,
        while 'horizontal' will place colbin in the y-axis.
        The default is 'vertical'.
    xlabel : str, optional
        Name to set xlabel, if None, will automatically assign based on col and normalize.
        The default is None.
    xlabel_kwargs : dict, optional
        Keyword arguments passed to |axxlabel|.
        The default is None.
    ylabel : str, optional
        Name to set ylabel, if None, will automatically assign based on col and normalize.
        The default is None.
    ylabel_kwargs : dict, optional
        Keyword arguments passed to |axylabel|.
        The default is None.
    **kwargs : Any
        Additional kwargs passed directly to
        `ax.plot() <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_ .

    Returns
    -------
    path : mpl.collections.PathCollection
        Matplotlib PathCollection of mean values.
    xlbl : plt.Text
        |Text| object of xlabel.
    ylbl : plt.Text
        |Text| object of ylabel.
    av : np.ndarray[np.float64]
        Array of computed mean values of colav.
    cbins : np.ndarray[np.float64]
        Array of bin centers.

    """
    
    return _mean_intervalplot(data, colav, colbin, gate=gate, ax=ax, bins=bins, 
                              include_unit=include_unit, rescale=rescale, record=record,
                              thresh=thresh, orientation=orientation, xlabel=xlabel,
                              xlabel_kwargs=xlabel_kwargs, ylabel=ylabel_kwargs, 
                              func='scatter', **kwargs)


def _nansem(a:np.ndarray)->float:
    return sem(a, nan_policy='omit')


_ebar_funcs = {'mean':np.nanmean, 'median':np.nanmedian, 'std':np.nanstd, 'sem':_nansem}


def _proc_col(a:np.ndarray, c:np.ndarray, func:Literal['mean','median','str','err']|Callable[[np.ndarray],float]|None)->float:
    if func is None or func is False:
        return None
    if not callable(func):
        if func not in _ebar_funcs:
            raise ValueError("invalid errorbar string, must be in {'mean', 'median', 'std', 'sem'}, got '%s'" % func)
        func = _ebar_funcs[func]
    return np.array([func(a[c==v]) for v in np.unique(c)])


def errorbars(data:DataS, colx:Column, coly:Column, colcat:Column,
              gate:GateGroup=None, ax:plt.Axes=None,
              include_unit:bool=True, rescale:tuple[float,float]=None, record:bool=False,
              xcenter:Literal['mean','median']|Callable[[np.ndarray],float]='mean', 
              ycenter:Literal['mean','median']|Callable[[np.ndarray],float]='mean',
              xerr:Literal['std','sem']|Callable[[np.ndarray],float]|None='std', 
              yerr:Literal['std','sem']|Callable[[np.ndarray],float]|None='std',
              xlabel:str=False, xlabel_kwargs:dict[str:Any]=None, 
              ylabel:str=False, ylabel_kwargs:dict[str:Any]=None,
             **kwargs:Any)->tuple[mpl.container.ErrorbarContainer,plt.Text,plt.Text]:
    """
    Plot error bars based on catagory of colcat. Takes the means or medians of
    colx and coly masked based on the categories of colcat. Errorbar sizes
    can be either standard deviation or standard error.

    Parameters
    ----------
    data : DataS
        Data on which the columns are based.
    colx : Column
        Column to plot errors of in the x-axis.
    coly : Column
        Column to plot errors of in the y-axis.
    colcat : Column
        Categorical column on which to create subsets to plot individual errors.
    gate : GateGroup, optional
        Gate to apply to colav and colbin, if not specified, use intersect of
        gates of colav and colbin. The default is None.
    ax : plt.Axes, optional
        Axes in which to plot histogram, if None, pull axes from
        `plt.gca() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.gca.html>`_ . 
        The default is None.
    include_unit : bool, optional
        Whether to include unit in axes label. The default is False.
    rescale : tuple[float,float], optional
        Factors by which (usually powers of 10) to rescale values of colx and coly.
        If None, defaults to (1.0, 1.0). The default is None.
    record : bool, optional
        Whether to record the column in cache or not. The default is False.
    xcenter : Literal['mean','median']|Callable[[np.ndarray],float], optional
        Type of center moment to use for x axis, may be ``'mean'`` or ``'median'``, or *callable* .
        If callable, must take single array and return a float of the center value.
        The default is 'mean'.
    ycenter : Literal['mean','median']|Callable[[np.ndarray],float], optional
        Type of center moment to use for y axis, may be ``'mean'`` or ``'median'``, or *callable* .
        If callable, must take single array and return a float of the center value.
        The default is 'mean'.
    xerr : Literal['std','sem']|Callable[[np.ndarray],float]|None, optional
        Type of error to use for x axis error bar, may be ``'std'`` (standard devaiation)
        or ``'sem'`` (standard error), or *callable* .
        If callable, must take single array and return a float of the error value.
        The default is 'std'.
    yerr : Literal['std','sem']|Callable[[np.ndarray],float]|None, optional
        DESCRIPTION. The default is 'std'.
    xlabel : str, optional
        Name to set xlabel, if None, will automatically assign based on col and normalize.
        The default is None.
    xlabel_kwargs : dict, optional
        Keyword arguments passed to |axxlabel|. 
        The default is None.
    ylabel : str, optional
        Name to set ylabel, if None, will automatically assign based on col and normalize.
        The default is None.
    ylabel_kwargs : dict, optional
        Keyword arguments passed to |axylabel|.
        The default is None.
    **kwargs : Any
        Additional keyword arguments handed to
        `ax.errorbar <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html#matplotlib.axes.Axes.errorbar>`_ .

    Returns
    -------
    ebar : mpl.container.ErrorbarContainer
        Return value of the call to ``ax.errorbar`` .
    xlbl : plt.Text
        |Text| object representing the x axis caption.
    ylbl : plt.Text
        |Text| object representing the y axis caption.

    """
    ax = _check_ax(ax)
    (x, y, c), (nx, ny, nc) = _get_column_arrays(data, colx, coly, colcat, gate=gate, 
                                                 include_unit=include_unit, 
                                                 rescale=rescale, record=record)
    cx, cy = _proc_col(x, c, xcenter), _proc_col(y, c, ycenter)
    dx, dy = _proc_col(x, c, xerr), _proc_col(y, c, yerr)
    ebar = ax.errorbar(cx, cy, xerr=dx, yerr=dy, **kwargs)
    if xlabel is not False:
        xlabel = nx if xlabel is None else xlabel
        xlabel_kwargs = _kwdct(xlabel_kwargs)
        xlbl = ax.set_xlabel(xlabel, **xlabel_kwargs)
    else:
        xlbl = None
    if ylabel is not False:
        ylabel = ny if xlabel is None else xlabel
        ylabel_kwargs = _kwdct(ylabel_kwargs)
        ylbl = ax.set_xlabel(ylabel, **ylabel_kwargs)
    else:
        ylbl = None
    return ebar, xlbl, ylbl


@fjit(fnumba.float64[:](fnumba.float64[:], fnumba.float64[:], fnumba.float64[:], 
                            fnumba.float64[:], fnumba.float64[:], fnumba.float64[:], 
                            fnumba.float64[:], fnumba.float64[:]))
def nb_gaus_2dkde(datax:np.ndarray[np.float64], datay:np.ndarray[np.float64], 
                  weights:np.ndarray[np.float64], 
                  sigmax:np.ndarray[np.float64], sigmay:np.ndarray[np.float64], 
                  rho:np.ndarray[np.float64], 
                  outx:np.ndarray[np.float64], outy:np.ndarray[np.float64]):
    """
    Compute gaussian KDE based on variable 2D kernels.

    Parameters
    ----------
    datax : :np.ndarray[np.float64]
        KDE x data points.
    datay : :np.ndarray[np.float64]
        KDE y data points.
    weights : :np.ndarray[np.float64]
        weights of data points.
    sigmax : :np.ndarray[np.float64]
        x sigma of each data point.
    sigmay : :np.ndarray[np.float64]
        y sigma of each data point.
    rho : :np.ndarray[np.float64]
        cross-correlation between each x and y data point.
    outx : :np.ndarray[np.float64]
        X values of output locations.
    outy : :np.ndarray[np.float64]
        Y values of output locations.

    Returns
    -------
    val : np.ndarray[np.float64]
        Kernel density at each out location.

    """
    if outx.shape[0] != outy.shape[0]:
        raise ValueError(f"outx and outy must have same shape, got {outx.shape} and {outy.shape}")
    if datax.shape[0] != datay.shape[0]:
        raise ValueError(f"datax and datay must have same shape, got {datax.shape} and {datay.shape}")
    if datax.shape[0] != weights.shape[0]:
        raise ValueError(f"datax and weights must have same shape, got {datax.shape} and {weights.shape}")
    if datax.shape[0] != sigmax.shape[0]:
        raise ValueError(f"datax and datay must have same shape, got {datax.shape} and {sigmax.shape}")
    if datax.shape[0] != sigmay.shape[0]:
        raise ValueError(f"datax and datay must have same shape, got {datax.shape} and {sigmay.shape}")
    if datax.shape[0] != rho.shape[0]:
        raise ValueError(f"datax and datay must have same shape, got {datax.shape} and {rho.shape}")
    val = np.zeros(outx.shape, dtype=np.float64)
    for i in range(outx.shape[0]):
        for j in range(datax.shape[0]):
            
            dx, dy = (outx[i]-datax[j])/sigmax[j], (outy[i]-datay[j])/sigmay[j]
            temp = weights[j]*np.exp(-(dx**2-2*rho[j]*dx*dy+dy**2))/(2*np.pi*sigmax[j]*sigmay[j]*np.sqrt(1-rho[j]**2))
            if np.isnan(temp):
                continue
            val[i] += temp
    return val


ArrReal = np.ndarray[np.float64]|Real
ArrInt = np.ndarray[np.int64]|Integral

def _reshape_single(val:Real|np.ndarray, shape:tuple[int,...], name:str)->np.ndarray:
    """Return array of shape shape, from val, either number repeated, or reshaped array"""
    if isinstance(val, Real):
        return np.ones(shape)*val
    if val.shape != shape:
        raise ValueError(f"{name} must be same shape as datax and datay")
    return val


def _infer_sigma(x:np.ndarray, y:np.ndarray, weights:np.ndarray, meth:str='scott')->tuple[float, float]:
    """Retrieve sigma value for kde of x, y, for kde estimation, also returns dummy co-variance of 0.0"""
    mask = ~(np.isnan(x)+np.isnan(y))
    x, y, weights = x[mask], y[mask], weights[mask]
    if meth is None or meth == 'scott':
        f = (weights.sum()/weights.mean())**(-1.0/6.0)
    elif meth == 'silverman':
        f = ((weights.sum()/weights.mean())*1.5)**(-1.0/6.0)
    else:
        raise ValueError("unrecognized bw_method {meth}, must be either 'scott' or 'silverman'")
    xstd, ystd = x.std(), y.std()
    xstd = 1.0 if xstd == 0.0 else xstd
    ystd = 1.0 if ystd == 0.0 else ystd
    return xstd*f, ystd*f, 0.0


BWCall = Callable[[np.ndarray[np.float64],np.ndarray[np.float64],np.ndarray[np.float64]],tuple[ArrReal,ArrReal,ArrReal]]


def gaus_2Dkde(datax:np.ndarray[np.float64], datay:np.ndarray[np.float64], 
              outx:np.ndarray[np.float64], outy:np.ndarray[np.float64], 
              weights:np.ndarray=None, sigmax:ArrReal=None, sigmay:ArrReal=None, rho:ArrReal=None,
              bw_method:str|BWCall=None)->np.ndarray[np.float64]:
    r"""
    Compute Kernel Density Estimator for data points (``datax``, ``datay``) at locations 
    (``outx``, ``outy``). Allows specification of weights for each point, sigma
    for both x and y values, and cross-correlation (rho) between them in gaussian
    model. This function wraps :func:`nb_gaus_2dkde` providind automatic reshaping
    of ``weights``, ``sigmax``, ``sigmay`` and ``rho`` values. Alternatively
    the ``bw_method`` can be used to dynamically compute ``weights``, ``sigmax``
    ``sigmay`` and ``rho`` arrays.

    Parameters
    ----------
    datax : :np.ndarray[np.float64]
        KDE x data points.
    datay : :np.ndarray[np.float64]
        KDE y data points.
    outx : :np.ndarray[np.float64]
        X values of output locations.
    outy : :np.ndarray[np.float64]
        Y values of output locations.
    weights : np.ndarray, optional
        Weight of each point in datax/datay, if None, assume equal weight to all
        points. The default is None.
    sigmax : Real|np.ndarray, optional
        :math:`\sigma_{x}` values of gaussian, if scalar, then treat all data
        points the same, if array, independent sigma for each point. The default is None.
    sigmay : Real|np.ndarray, optional
        :math:`\sigma_{y}` values of gaussian, if scalar, then treat all data
        points the same, if array, independent sigma for each point. The default is None.
    rho : Real|np.ndarray, optional
        :math:`\rho` (cross correlation) values of gaussian, if scalar, then treat all data
        points the same, if array, independent rho for each point. The default is None.
    bw_method : str | BWCall, optional
        If all ``sigmax``, ``sigmay`` and ``rho`` are ``None``. The function called
        for generating said values from data given. If specified as str, use
        parameters close to |gaussian_kde|. The default is None.

    Raises
    ------
    ValueError
        Inconsistent shape of arrays.

    Returns
    -------
    np.ndarray[np.float64]
        Array of same shape as outx and outy specifying KDE value at each point in
        (``outx``, ``outy``).

    """
    if datax.ndim != 1 or datay.ndim != 1:
        raise ValueError("datax and datay must be 1d")
    if datax.shape != datay.shape:
        raise ValueError("datax and datay must be same shape")
    if outx.shape != outy.shape:
        raise ValueError("outx and outy must be same shape")
    otx, oty = outx.reshape(-1), outy.reshape(-1)
    weights = np.ones(datax.shape) if weights is None else weights
    if sigmax is None or sigmay is None or rho is None:
        if not callable(bw_method):
            sigx, sigy, rh = _infer_sigma(datax, datay, weights, meth=bw_method)
        else:
            sigx, sigy, rh = bw_method(datax, datay, weights)
        sigmax = sigx if sigmax is None else sigmax
        sigmay = sigy if sigmay is None else sigmay
        rho = rh if rho is None else rho
    mask = np.isfinite(datax) | np.isfinite(datay)
    datax, datay = datax[mask], datay[mask]
    sigmax = _reshape_single(sigmax, datax.shape, 'sigmay')
    sigmay = _reshape_single(sigmay, datax.shape, 'sigmax')
    rho = _reshape_single(rho, datax.shape, 'rho')
    return nb_gaus_2dkde(datax, datay, weights, sigmax, sigmay, rho, otx, oty).reshape(outx.shape)


def gaus_2Dkde_cmap(datax:np.ndarray, datay:np.ndarray, outx:np.ndarray, outy:np.ndarray, 
                    weights:np.ndarray=None, sigmax:ArrReal=None, sigmay:ArrReal=None, rho:ArrReal=None,
                    bw_method:str|BWCall=None, minzero:bool=True, maxone:bool=True, 
                    thresh:float=1.0, thresh_raw:bool=False)->np.ndarray[np.float64]:
    r"""
    Create a normalized color array for plotting :func:`gaus_2Dkde` distribution.
    
    Parameters
    ----------
    datax : :np.ndarray[np.float64]
        KDE x data points.
    datay : :np.ndarray[np.float64]
        KDE y data points.
    outx : :np.ndarray[np.float64]
        X values of output locations.
    outy : :np.ndarray[np.float64]
        Y values of output locations.
    weights : np.ndarray, optional
        Weight of each point in datax/datay, if None, assume equal weight to all
        points. The default is None.
    sigmax : Real|np.ndarray, optional
        :math:`\sigma_{x}` values of gaussian, if scalar, then treat all data
        points the same, if array, independent sigma for each point. The default is None.
    sigmay : Real|np.ndarray, optional
        :math:`\sigma_{y}` values of gaussian, if scalar, then treat all data
        points the same, if array, independent sigma for each point. The default is None.
    rho : Real|np.ndarray, optional
        :math:`\rho` (cross correlation) values of gaussian, if scalar, then treat all data
        points the same, if array, independent rho for each point. The default is None.
    bw_method : str | BWCall, optional
        If all ``sigmax``, ``sigmay`` and ``rho`` are ``None``. The function called
        for generating said values from data given. If specified as str, use
        parameters close to |gaussian_kde|. The default is None.
    minzero : bool, optional
        Whether to rescale minimum value to 0. The default is True.
    maxone : bool, optional
        Whether to rescale maximum value to 1. The default is True.
    thresh : float, optional
        Minimum KDE value to display, otherwise set to transprent (nan). 
        The default is 1.0.
    thresh_raw : bool, optional
        If ``True`` thresh is evaluated against as raw value of kde.
        If ``False`` thresh is defined by as kde divided by number of points 
        (ie if kde / # data poins < thresh, point is not displayed). This makes
        thresh a "fractional" KDE.
        The default is False.

    Returns
    -------
    dens : np.ndarray
        Density of KDE at specified points.

    """
    dens = gaus_2Dkde(datax, datay, outx, outy, weights, sigmax, sigmay, rho, bw_method=bw_method)
    if thresh_raw:
        dens[dens<thresh] = np.nan
    dens = _minmax_rescale(dens, minzero, maxone)
    if not thresh_raw:
            dens[dens<thresh] = np.nan
    return dens


def _get_lim(arr):
    """get axes limits for array, extending min/max values by 0.1 of difference between them"""
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mn == mx:
        return (0.0, 1.0)
    shift = 0.1*(mx-mn)
    return mn-shift, mx+shift


def _plot_kde1D(func:Callable, arr, bins:ArrInt=None, xlim:tuple[float,float]=None, 
                bw_method:str|Callable[[gaussian_kde],float]=None, weights:None|np.ndarray[np.float64]=None,
                rescale_factor:Callable[[np.ndarray],float]=None, edges:bool=False, **kwargs)->Any:
    """Plot 1D KDE using func (should be plot/bar like function)"""
    bins = 1024 if bins is None else bins
    if isinstance(bins, Integral):
        xmin, xmax = _get_lim(arr) if xlim is None else xlim
        bins = np.linspace(xmin, xmax, bins + 1 if edges else bins)
    bins = bins[:-1] + np.diff(bins)/2 if edges else bins
    mask = np.isfinite(arr)
    weights = weights if weights is None else weights[mask]
    y = gaussian_kde(arr[mask], bw_method=bw_method, weights=weights).evaluate(bins)
    if callable(rescale_factor):
        y *= rescale_factor(y)
    elif rescale_factor is not None:
        y *= rescale_factor
    return func(bins, y, **kwargs)


def _plot_kde2D(func:Callable, datax:np.ndarray, datay:np.ndarray, weights:np.ndarray=None,
                edges:bool=False, sigmax:ArrReal=None, sigmay:ArrReal=None, rho:ArrReal=None,
                minzero:bool=True, maxone:bool=True, 
                thresh:float=0.0, thresh_raw:bool=False, bins:tuple[ArrInt,ArrInt]=None,
                xlim:tuple[float,float]=None, ylim:tuple[float,float]=None, 
                bw_method:str|BWCall=None, **kwargs)->Any:
    """Plot a 2D KDE usuing func, func should be a pcolor like function"""
    binx, biny = (512, 512) if bins is None else bins
    if isinstance(binx, Real):
        xmin, xmax = _get_lim(datax) if xlim is None else xlim
        binx = np.linspace(xmin, xmax, binx)
    if isinstance(biny, Real):
        ymin, ymax = _get_lim(datay) if ylim is None else ylim
        biny = np.linspace(ymin, ymax, biny)
    yg, xg = np.meshgrid(biny, binx)
    if edges:    
        xl = xg[:-1,:-1]+(xg[0,1]-xg[0,0])/2
        yl = yg[:-1,:-1]+(yg[1,0]-yg[0,0])/2
    else:
        xl, yl = xg, yg
    kde = gaus_2Dkde_cmap(datax, datay, xl, yl, weights=weights, 
                         sigmax=sigmax, sigmay=sigmay, rho=rho, 
                         minzero=minzero, maxone=maxone, bw_method=bw_method,
                         thresh=thresh, thresh_raw=thresh_raw)
    return func(xg, yg, kde, **kwargs)


def _axim(ax:plt.Axes)->Callable[[np.ndarray,np.ndarray,np.ndarray,...],mpl.image.AxesImage]:
    """Return callable to plot image in ax"""
    def plot(x,y,c, **kwargs):
        kw = {'aspect':'auto', 'extent':(x[0,0], x[-1,0], y[0,0], y[0,-1])}
        kw.update(kwargs)
        ax.imshow(c.T[::-1,:],**kw)
    return plot


def kdeplot(data:DataS, *args:Column, gate:GateGroup=None, ax:plt.Axes=None, 
            plot_style:str|Callable=None,
            include_unit:bool=False, rescale:Sequence[float]=None, record:bool=False,
            weights:np.ndarray=None,
            rescale_factor:Callable[[np.ndarray],float]|float=None, edges:bool=False,
            sigmax:ArrReal=None, sigmay:ArrReal=None, rho:ArrReal=None,
            minzero:bool=True, maxone:bool=True, bins:tuple[ArrInt,ArrInt]=None,
            xlim:tuple[float,float]=None, ylim:tuple[float,float]=None,
            thresh:float=0.0, thresh_raw:bool=False, 
            xlabel:str=None, xlabel_kwargs:dict=None,
            ylabel:str=None, ylabel_kwargs:dict=None, 
            **kwargs)->tuple[mpl.image.AxesImage, plt.Text, plt.Text]:
    """
    Generate a KDE plot of the specified columns. May specify 1D with 1 column 
    argument, in which case results in line plot, if 2 column arguments specified,
    then produces heat-map like plot.

    Parameters
    ----------
    data : DataS
        Source of data.
    *args : Column
        Input :class:`Column` objects to be plotted.
    gate : GateGroup, optional
        Gate to apply to columns. The default is None.
    ax : plt.Axes, optional
        Axes in which to plot kde. The default is None.
    plot_style : str|Callable, optional
        Either name of function (method of axes) to use to plot kde result, or
        callable that plots. When called, will have signature ``(bins, vals, **kwargs)``. 
        The default is None.
    include_unit : bool, optional
        Whether to include unit in axis labels. The default is False.
    rescale : Sequence[float]|float, optional
        Factor by which to rescale values of each colvalumn. If not specified, assume
        1.0. The default is None.
    record : bool, optional
        Whether to record the column in cache or not. The default is False.
    weights : np.ndarray, optional
        Weight of each point in datax/datay, if None, assume equal weight to all
        points. The default is None.
    rescale_factor : Callable[[np.ndarray],float]|float, optional
        Factor by which to rescale KDE (1D only) (usually to match histogram). 
        Can specify callable that takes the kde and returns a float.
        This is used primarily for scaling KDE to histogram normalization scheme.
        The default is None.
    edges : bool, optional
        If True, compute KDE around midpoint of bins. This results in
        closer match between histogram and KDE points. 1-D only.
        The default is False.
    sigmax : ArrReal, optional
        sigma value around X-axis, 2-D only. Can supply as array to give per-point
        sigma values. The default is None.
    sigmay : ArrReal, optional
        sigma value around Y-axis, 2-D only. Can supply as array to give per-point
        sigma values. The default is None.
    rho : ArrReal, optional
        Rho value giving cross-correlation between X and Y axis, 2-D only. 
        Can supply as array to give per-point rho values. The default is None.
    minzero : bool, optional
        If True, rescale output values so minimum is 0. The default is True.
    maxone : bool, optional
        If True, rescale output values so maximum is 1. The default is True.
    bins : tuple[ArrInt,ArrInt], optional
        Values at which to evaluate KDE. The default is None.
    xlim : tuple[float,float], optional
        Limits of KDE evaluation along x-axis. The default is None.
    ylim : tuple[float,float], optional
        Limits of KDE evaluation along y-axis, 2-d only. The default is None.
    thresh : float, optional
        Minimum KDE value to display, otherwise set to transprent (nan). 
        The default is 0.0.
    thresh_raw : bool, optional
        If ``True`` thresh is evaluated against as raw value of kde.
        If ``False`` thresh is defined by as kde divided by number of points 
        (ie if kde / # data poins < thresh, point is not displayed). This makes
        thresh a "fractional" KDE.
        The default is False.
    xlabel : str, optional
        Name for x-axis label. The default is None.
    xlabel_kwargs : dict, optional
        keyword arguments given to |axxlabel|. The default is None.
    ylabel : str, optional
        Name for y-axis label. The default is None.
    ylabel_kwargs : dict, optional
        keyword arguments given to |axylabel|.
        The default is None.
    **kwargs : Any
        Keyword argument passed to matplotlib plotting function. Function set
        by plot_style.

    Raises
    ------
    TypeError
        Either no columns or too many columns specified.

    Returns
    -------
    out : mpl.image.AxesImage
        Output of plotting function.
    xlbl : plt.Text
        |Text| object of xlabel.
    ylbl : plt.Text
        |Text| object of ylabel.

    """
    if not args:
        raise TypeError("must specify at least one column")
    if len(args) > 2:
        raise TypeError("can specify maximum of 2 columns")
    carrs, names = _get_column_arrays(data, *args, gate=gate, 
                                      include_unit=include_unit, 
                                      rescale=rescale, record=record)
    ax = plt.gca() if ax is None else ax
    if callable(plot_style):
        func = plot_style
    elif plot_style is None:
        func = _axim(ax) if len(args) == 2 else ax.plot
    else:
        func = getattr(ax, plot_style)
    xlabel = names[0] if xlabel is None else xlabel
    xlabel_kwargs, ylabel_kwargs = _kwdct(xlabel_kwargs), _kwdct(ylabel_kwargs)
    if len(carrs) == 1:
        out = _plot_kde1D(func, *carrs, weights=weights, rescale_factor=rescale_factor, 
                          edges=edges, bins=bins, xlim=xlim, **kwargs)
        ylbl =  None if ylabel is False or rescale_factor is not None else ax.set_ylabel("PDF", **ylabel_kwargs)
    else:
        out = _plot_kde2D(func, *carrs, weights=weights, sigmax=sigmax, sigmay=sigmay,
                          rho=rho, minzero=minzero, maxone=maxone, bins=bins,
                          xlim=xlim, ylim=ylim, thresh=thresh, thresh_raw=thresh_raw, **kwargs)
        ylabel = names[1] if ylabel is None else ylabel
        ylbl = None if ylabel is False else ax.set_ylabel(ylabel, **ylabel_kwargs)
    xlbl = None if xlabel is False else ax.set_xlabel(names[0], **xlabel_kwargs)
    return out, xlbl, ylbl


_cpos_rgx = re.compile(r'(?P<y>l|lower|u|upper)(?P<x>l|left|r|right)')
_cpos_map = {0:(1,0), 1:(1,1), 2:(0,1), 3:(0,0)}


def jointplot(data:DataS, colx:Column, coly:Column, gate:GateGroup=None, 
              fig:plt.Figure=None, axmat:np.ndarray[plt.Axes]=None,
              include_unit:bool=True, rescale:tuple[Real,Real]=None, record:bool=False,
              cxlabel:str=None, cxlabel_kwargs:dict=None, cylabel:str=None, cylabel_kwargs:dict=None,
              xxlabel:str=None, xxlabel_kwargs:dict=None, xylabel:str=None, xylabel_kwargs:dict=None,
              yxlabel:str=None, yxlabel_kwargs:dict=None, yylabel:str=None, yylabel_kwargs:dict=None,
              cfunc:Callable=scatter, hfunc:Callable=hist, xfunc:Callable=None, yfunc:Callable=None, 
              ratio:float=5.0, width_ratio:float=None, height_ratio:float=None,
              cplot_kwargs:dict=None, hplot_kwargs:dict=None, xplot_kwargs:dict=None, yplot_kwargs:dict=None,
              gridspec_kwargs:dict=None, cpos:str='ll', 
              set_axis:Literal['auto','center','xhist','yhist',False,None]='auto'
              )->tuple[np.ndarray[plt.Axes],tuple[Any,...],tuple[Any,...],tuple[Any,...]]:
    """
    Create a "jointplot" ie a 2D representation of 2 columns flanked by the 
    1-D histogram projectiosn of each axis in 2D plot.

    Parameters
    ----------
    data : DataS
        Data on which columns are based.
    colx : Column
        :class:`Column` assigned to X axis.
    coly : Column
        :class:`Column` assigned to Y axis.
    gate : GateGroup, optional
        If specified, colx and coly will be regated to gate, defining which points
        (rows) included. The default is None.
    fig : plt.Figure, optional
        `plt.Figure <https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.html>`_ 
        in which to plot the jointplot, function will create axes within this figure. 
        The default is None.
    axmat : np.ndarray[plt.Axes], optional
        3-element sequence of ``plt.Axes`` objects in which to place plots.
        Sequence must be 2d, x-axis plot, y-axis plot
        If specifed overrides fig.
        The default is None.
    include_unit : bool, optional
        Whether to include unit in axes label. The default is False.
    rescale : tuple[float,float], optional
        Factors by which (usually powers of 10) to rescale values of colx and coly.
        If None, defaults to (1.0, 1.0). The default is None.
    record : bool, optional
        Whether to record the column in cache or not. The default is False.
    cxlabel : str, optional
        Name for x-axis label in center axis. The default is None.
    cxlabel_kwargs : dict, optional
        Keyword arguments passed to 
        axx;abe;_. 
        for the 2D plot axes
        The default is None.
    cylabel : str, optional
        Name for y-axis label in center axis. The default is None.
    cylabel_kwargs : dict, optional
        Keyword arguments passed to |axylabel| for the 2D plot axes.
        The default is None.
    xxlabel : str, optional
        Name for x-axis label in x-axis histogram. The default is None.
    xxlabel_kwargs : dict, optional
        Keyword arguments passed to |axxlabel|.
        For the x-axis histogram axes.
        The default is None.
    xylabel : str, optional
        Name for x-axis label in y-axis histogram. The default is None.
    xylabel_kwargs : dict, optional
        Keyword arguments passed to |axylabel| for the x-axis histogram.
        The default is None.
    yxlabel : str, optional
        Name for x-axis label in y-axis histogram. The default is None.
    yxlabel_kwargs : dict, optional
        Keyword arguments passed to |axxlabel|.
        for the y-axis histogram.
        The default is None.
    yylabel : str, optional
        Name for y-axis label in y-axis histogram. The default is None.
    yylabel_kwargs : dict, optional
        Keyword arguments passed to |axylabel| for the y-axis histogram.
        The default is None.
    cfunc : Callable, optional
        **smfBursts** function for plotting in 2D axes. The default is :func:`scatter`.
    hfunc : Callable, optional
        **smfBursts** function for plotting histograms, has lower priority than
        xfunc and yfunc. The default is :func`hist`.
    xfunc : Callable, optional
        **smfBursts** function for plotting x-axis histogram. The default is None.
    yfunc : Callable, optional
        **smfBursts** function for plotting y-axis histogram. The default is None.
    ratio : float, optional
        Ratio of 2D axis length/width to histogram axes, has lower priority than
        width_ratio and height_ratio. The default is 5.0.
    width_ratio : float, optional
        Ratio of width of x-axis of 2D to y-axis histogram width. The default is None.
    height_ratio : float, optional
        Ratio of height of y-axis of 2D to x-axis histogram height. The default is None.
    cplot_kwargs : dict, optional
        Keyword arguments passed to 2D plotting function. The default is None.
    hplot_kwargs : dict, optional
        Keyword arguments passed to histogram plotting functions, has lower priority
        that xplot_kwargs and yplot_kwargs, all kwargs specified here will be
        passed to both histogram plotting functions *unless* key is over-written
        by xplot_kwargs for x-axis histogram, or yplot_kwargs for y-axis histogram. 
        The default is None.
    xplot_kwargs : dict, optional
        Keyword arguments passed to x-axis histogram function. The default is None.
    yplot_kwargs : dict, optional
        Keyword arguments passed to y-axis histogram function. The default is None.
    gridspec_kwargs : dict, optional
        Keyword arguments passed to
        `mpl.gridspec.GridSpec <https://matplotlib.org/stable/api/_as_gen/matplotlib.gridspec.GridSpec.html>`_
        . The default is None.
    cpos : str, optional
        Position (lower/upper, and left/right) of 2D plot in grid. The default is 'll'.

    Raises
    ------
    ValueError
        Bad value in cpos or rescale.

    Returns
    -------
    axmat : np.ndarray[plt.Axes]
        3 element numpy array of ``plt.Axes`` objects, the axes of the
        2D plot, x-axis histogram and y-axis histogram, in that order.
    cout : tuple
        Output of 2D plotting function.
    xout : tuple
        Output of x-axis histogram function.
    yout : tuple
        Output of y-axis histogram function.

    """
    # make axes layout
    if axmat is None:
        set_axis = 'center' if set_axis == 'auto' or set_axis is True else set_axis
        # get figure if None
        fig = plt.gcf() if fig is None else fig
        # build joint grid axes
        if isinstance(cpos, str):
            cposm = _cpos_rgx.match(cpos)
            if cposm is None:
                raise ValueError("")
            cpos = (int(cposm.group('y')[0] == 'l'), int(cposm.group('x')[0] == 'r'))
        elif isinstance(cpos, Integral):
            if cpos not in _cpos_map:
                raise ValueError(f"invalid cpos code {cpos}, must be in {list(_cpos_map.keys())}")
            cpos = _cpos_map.get(cpos)
        width_ratio = ratio if width_ratio is None else width_ratio
        height_ratio = ratio if height_ratio is None else height_ratio
        spkwargs = dict(height_ratios=np.array([height_ratio, 1.0])[::1-2*cpos[0]],
                        width_ratios=np.array([width_ratio, 1.0])[::1-2*cpos[1]])
        spkwargs.update(dict() if gridspec_kwargs is None else gridspec_kwargs)
        gs = mpl.gridspec.GridSpec(2, 2, figure=fig, **spkwargs)
        axc = fig.add_subplot(gs[cpos])
        axx = fig.add_subplot(gs[1-cpos[0],cpos[1]], sharex=axc)
        axy = fig.add_subplot(gs[cpos[0], 1-cpos[1]], sharey=axc)
        axmat = np.array([axc, axx, axy])
        cxlabel = False if cxlabel is None and cpos[0] == 0 else cxlabel
        cylabel = False if cylabel is None and cpos[1] == 1 else cylabel
        xxlabel = False if xxlabel is None and cpos[0] == 1 else xxlabel
        yylabel = False if yylabel is None and cpos[1] == 0 else yylabel
        axx.yaxis.set_inverted(cpos[0] == 0)
        axy.xaxis.set_inverted(cpos[1] == 1)
    else:
        axc, axx, axy = axmat
    # process cross-plot kwargs
    if rescale is None:
        rescale = (1,1)
    elif not isinstance(rescale, Sequence):
        rescale = (rescale, rescale)
    elif len(rescale) != 2:
        raise ValueError("rescale for jointplot must be 2 elements")
    rescalex, rescaley = rescale
    xfunc = hfunc if xfunc is None else xfunc
    yfunc = hfunc if yfunc is None else yfunc
    ckwargs = dict(include_unit=include_unit, rescale=rescale, record=record,
                   xlabel = cxlabel, ylabel = cylabel, 
                   xlabel_kwargs=cxlabel_kwargs, ylabel_kwargs=cylabel_kwargs)
    xkwargs = dict(include_unit=include_unit, rescale=rescalex, record=False,
                   xlabel=xxlabel, ylabel=xylabel, 
                   xlabel_kwargs=xxlabel_kwargs, ylabel_kwargs=xylabel_kwargs)
    ykwargs = dict(include_unit=include_unit, rescale=rescaley, record=False,
                   xlabel = yxlabel, ylabel=yylabel, 
                   xlabel_kwargs=yxlabel_kwargs, ylabel_kwargs=yylabel_kwargs)
    if 'orientation' in signature(xfunc).parameters.keys():
        xkwargs['orientation'] = 'vertical'
    if 'orientation' in signature(yfunc).parameters.keys():
        ykwargs['orientation'] = 'horizontal'
    ckwargs.update(dict() if cplot_kwargs is None else cplot_kwargs)
    xkwargs.update(dict() if hplot_kwargs is None else hplot_kwargs)
    ykwargs.update(dict() if hplot_kwargs is None else hplot_kwargs)
    xkwargs.update(dict() if xplot_kwargs is None else xplot_kwargs)
    ykwargs.update(dict() if yplot_kwargs is None else yplot_kwargs)
    # call plotting functions
    cout = cfunc(data, colx, coly, gate=gate, ax=axc, **ckwargs)
    xout = xfunc(data, colx, gate=gate, ax=axx, **xkwargs)
    yout = yfunc(data, coly, gate=gate, ax=axy, **ykwargs)
    if cpos[0]:
        axy
    if set_axis == 'center':
        plt.sca(axmat[0])
    elif set_axis == 'xhist':
        plt.sca(axmat[1])
    elif set_axis == 'yhist':
        plt.sca(axmat[2])
    return axmat, cout, xout, yout


def plot_multi_dist(func:FitFunc, params:np.ndarray[np.float64], x:np.ndarray[np.float64], 
                    ax:plt.Axes=None, dist_kwargs:dict=None, plot_sub:bool=True,
                    subdist_kwargs:dict|Sequence[dict]=None, **kwargs:Any)->tuple[plt.Line2D,list[plt.Line2D]]:
    dist_kwargs = dict() if dist_kwargs is None else dist_kwargs
    subdist_kwargs = dict() if subdist_kwargs is None else subdist_kwargs
    subdist_kwargs = repeat(subdist_kwargs) if isinstance(subdist_kwargs, dict) else subdist_kwargs
    ax = _check_ax(ax)
    distplot = ax.plot(x, func(params, x), **_dict_update(kwargs, dist_kwargs))
    if plot_sub:
        parrays = unwrap_param(func, params, as_dict=False)
        if func._free:
            subs = [ax.plot(x, func(np.asarray(param), x), **_dict_update(kwargs, subkws)) 
                    for subkws, param in zip(subdist_kwargs, *parrays)]
        else:
            subs = [ax.plot(x, amp*func(np.asarray(param), x), **_dict_update(kwargs, subkws)) 
                    for subkws, *param, amp in zip(subdist_kwargs, *parrays)]
    else:
        subs = list()
    return distplot, subs


# TODO: complete this method
def plot_residuals_cdfbins(func:FitFunc, params:np.ndarray[np.float64], bins:np.ndarray[np.float64],
                           ax:plt.Axes=None, dist_kwargs:dict=None, plot_sub:bool=True,
                           subdist_kwargs:dict|Sequence[dict]=None, **kwargs):
    dist_kwargs = dict() if dist_kwargs is None else dist_kwargs
    subdist_kwargs = dict() if subdist_kwargs is None else subdist_kwargs
    subdist_kwargs = repeat(subdist_kwargs) if isinstance(subdist_kwargs, dict) else subdist_kwargs
    ax = _check_ax(ax)

