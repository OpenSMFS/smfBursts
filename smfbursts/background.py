#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created : 9/10/20025
# Author: Paul David Harris
# email: harrip@gmail.com
"""
Module for assesment of background rates in data.

Defines base and child tables :class:`Periods` and :class:`BG` respectively for
division of data in to consecutive periods (for background assemsment) and 
computation of background, respectively.

Additional background functions are defined.
"""
from typing import Union, ClassVar, Literal
from collections.abc import Hashable, Callable, Sequence, Iterator
import inspect
import warnings

import numpy as np
from scipy.stats import linregress, expon
from scipy.optimize import leastsq

from .datamodel.utils import tupledict, arr_slc
from .datamodel.immutabledata import (TypeValidator, TV_float, TV_str, TV_bool, 
                                      TV_PyCode, TV_ndarray, register_PyCode, get_pycode_subval)
from .datamodel.tables import ParamDef, ParentDef, ColumnDef, Param, Column, DataSet, as_paramdict, paramproperty
from .cite import cite
from .ph_sel import PhSel, DetDef, TV_DetDef
from .photondata import (
    BasePhotonTable, ChildPhotonTable, PhotonData, 
    _regularize_column_startstop, _title_sels, _title_startstop_append, _title_unit_append, 
    make_base_column_defs, ColKeyStart, ColKeyStop
    )

import smfbursts.cfuncs as smc


def _periods_title_func(col:Column, include_unit:bool=False)->str:
    """Title func for periods column"""
    if 'offset' not in col:
        out = 'periods'
    else:
        out = 'stop' if col.offset else 'start'
    if include_unit:
        out += ' clk_p'
    return out


class Periods(BasePhotonTable):
    """
    Define a range of equal duration, consecutive periods in :class:`PhotonData`
    
    Params
    ------
        period : float
            The duration of each period (time range) in seconds.
        start_at : {'time_min', 'zero', 'under', 'over'}
            One of ``'time_min'``, ``'zero'``, ``'under'``, or ``'over'``. Defines when 
            first period starts relative to first photon in data.
            
            Options:
                    
                - ``'time_min'`` start of first period is first photon in data
                - ``'zero'`` start of first period is time = 0
                - ``'under'`` start of first period is the time that is a integer multiple of 
                  period, and less than (greatest possible) the time of the first photon
                - ``'over'`` similar to under, the least possible integer multiple of 
                  period greater than the time of the first photon
            
            ``start_at`` must be specified with ``stop_at```, and these are exclusive of 
            specifying ``start`` and ``stop``. Default (if ``start`` is not defined)
            is ``'time_min'``
        stop_at : {'under', 'over'}
            One of ``'under'`` or ``'over'``. Defines stop time of last period.
            
            Options:
                
                - ``'under'`` last period ends before last time of data
                - ``'over'`` last period ends after last time of data
            
            Default (if ``stop`` is not defined) is ``'over'``.
        start : float
            Start of first period (in seconds). Must be specified with ``stop``, 
            cannot be specified with ``start_at`` or ``stop_at``
        stop : float
            End of final period (in seconds), rounded down to integer multiple of 
            periods + start. Must be specified with ``start``, cannot be specified
            with ``start_at`` or ``stop_at``
        detdef : DetDef
            DetDet object that the data must have to be compatible.
    
    Parents
    -------
    This method has no parents
    
    Columns
    -------
    periods : int, offset = 1, limits of periods
        
    And all columns in :any:`basephotoncolumns` for full list of columns.
    
    """
    row_name:ClassVar[str] = "Periods"
    _origin: PhotonData
    #: :meta private:
    param_defs = (
        ParamDef('period', TV_float(mn=0.0), default=60.0),
        ParamDef('start_at', TV_str(isin=('time_min', 'zero', 'under', 'over')), required=False),
        ParamDef('stop_at', TV_str(isin=('under', 'over')), required=False), 
        ParamDef('start', TV_float, required=False), 
        ParamDef('stop', TV_float, required=False),
        ParamDef('detdef', TV_DetDef, required=True)
                  )
    #: :meta private:
    parent_defs = tuple()
    #: :meta private:
    column_defs = (
        ColumnDef('periods', tuple(), 1, 'all', dtype=np.int64, 
                  title_func='_get_periods_title', index_func='_get_periods_index', unit='clk_p'),
        ColumnDef('start', tuple(), 0, remap='_replace_column_startstop'),
        ColumnDef('stop', tuple(), 0, remap='_replace_column_startstop')
                  ) + make_base_column_defs(skip=('start', 'stop'))

    def __init_columns__(self):
        period = np.int64(np.round(self.param.params['period']/self.origin.clk_p))
        # find start time
        if 'start' in self.param.params:
            start = np.int64(self.param.params['start']/self.origin.clk_p)
        elif self.param.params['start_at'] == 'time_min':
            start = self.origin.times[0]
        elif self.param.params['start_at'] == 'zero':
            start = np.int64(0)
        elif self.param.params['start_at'] == 'under':
            start = (self.origin.times[0] // period) * period
        elif self.param.params['start_at'] == 'over':
            start = ((self.origin.times[0]//period) + np.int64(1))* period
        # find stop time
        if 'stop' in self.param.params:
            stop = np.int64(self.param.params['stop']/self.origin.clk_p)
        elif self.param.params['stop_at'] == 'under':
            stop = (self.origin.times[-1] // period) * period
        elif self.param.params['stop_at'] == 'over':
            stop = ((self.origin.times[-1] // period) + np.int64(1)) * period
        # create periods
        periods = np.arange(start, stop, period, dtype=np.int64)
        self._add_column('periods',tuple(), periods)
        # compute index partitions
        istart, istop = smc.index_ranges(self.origin.times, periods[:-1], periods[1:])
        self._add_column('istart', tuple(), istart)
        self._add_column('istop', tuple(), istop)

    @classmethod
    def param_preprocess(cls, params:Union[Sequence,dict,tupledict], parents:dict[str,Param])->dict:
        params = as_paramdict(params, tuple(pdef.name for pdef in cls.param_defs))
        # check for competing definitions
        if 'start' in params and 'start_at' in params:
            raise ValueError("may only specify 'start' or 'start_at', but not both")
        if 'stop' in params and 'stop_at' in params:
            raise ValueError("may only specify 'stop' or 'stop_at', but not both")
        # set defaults
        if 'start' not in params and 'start_at' not in params:
            params['start_at'] = 'time_min'
        if 'stop' not in params and 'stop_at' not in params:
            params['stop_at'] = 'over'
        return params, parents

    @classmethod
    def _replace_column_startstop(cls, col:str, keys)->tuple[str,tuple[Hashable,...],int]:
        """Remap function for remapped coluns start and stop"""
        return 'periods', keys, 0 if col == 'start' else 1

    @paramproperty
    def detdef(cls, param:Param)->DetDef:
        """Return :class:`DetDef` of Periods :class:`Param`."""
        return param.params['detdef']

    @classmethod
    def _get_periods_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """Title func for periods column"""
        if 'offset' not in col:
            out = 'periods'
        else:
            out = 'stop' if col.offset else 'start'
        if include_unit:
            out += ' clk_p'
        return out

    @classmethod
    def _get_periods_index(cls, col:Column, include_unit:bool=False)->str:
        """Index name func for periods column"""
        return cls._periods_title_func(col, include_unit)


#########################################################
### Functions for computing BackGround
#########################################################
BGFuncType = Callable[[np.ndarray[np.int64],float,...], float]


def _make_default_bg_preprocess(params:dict, parents:dict)->dict:
    return params

def _make_default_bg_paramdefs(params:dict)->tuple[ParamDef,...]:
    params = tuple(inspect.signature(params['func']).parameters.values())
    params = params[1:] if params[0].kind == params[0].VAR_POSITIONAL else params[2:]
    return tuple(param_to_ParamDef(param) for param in params)


def _make_default_bg_postvalidate(params:tupledict, parents:tupledict)->None:
    pass


def _make_default_bg_streamconversion(param:tupledict, stream_ids:np.ndarray[np.uint8])->dict:
    return param 


def register_bg_func(func:BGFuncType, 
                     param_preprocess:Callable[[dict, dict],dict]=None,
                     param_validator:Callable[[dict,],tuple[ParamDef,...]]=None,
                     post_init_validator:Callable[[tupledict,tupledict],None]=None,
                     stream_conversion:Callable[[tupledict,np.ndarray[np.uint8]],dict[str:Hashable]]=None)->None:
    """
    Register a method in PyCode for computing background with :class:`BG`.
    Includes ability to add special param_validator

    Parameters
    ----------
    func : BGFuncType
        Funtion to register, must take at least 2 positional optional arguments, 
        rest should be keyword optional, and return a float. First argument
        is photon arrival times, second is timestamps unit.
    param_validator : Callable[[dict,],tuple[ParamDef,...]], optional
        Function to validate/convert params dict to be compatible with func. 
        The default is None.

    Raises
    ------
    TypeError
        param_validator is not callable.
    ValueError
        Incompatible combination of param_validator function parameters.

    """
    # Check bg func signature is valid
    func_params = tuple(p for p in inspect.signature(func).parameters.values())
    if any(param.kind == param.VAR_POSITIONAL for param in func_params):
        warnings.warn("bg_func contains VAR_POSITIONAL arguments, these will be ignored")
        # drop var_positional argument unless is first argument
        func_params = tuple(param for i, param in enumerate(func_params) 
                            if i == 0 or param.kind != param.VAR_POSITIONAL)
    if any(param.kind == param.VAR_KEYWORD for param in func_params):
        warnings.warn("bg_func contains VAR_KEYWORD arguments, these will be ignorred")
        func_params = tuple(param for param in func_params if param.kind != param.VAR_KEYWORD)
    if sum(p.default is inspect._empty and p.kind not in (p.VAR_POSITIONAL, p.KEYWORD_ONLY) 
           for p in func_params) > 2:
        raise ValueError("Too many required artuments for bg_func")
    # check param_preprocess
    param_preprocess = _make_default_bg_preprocess if param_preprocess is None else param_preprocess
    # Check param validator
    if param_validator is not None:
        if not callable(param_validator):
            raise TypeError("param_validator must be callable accepting single dict and returning tuple of ParamDefs")
            val_params = tuple(p for p in inspect.signature(param_validator))
            # make sure param_validator will accept single positional argument
            if sum(p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) or p.default is not inspect._empty
                   for p in val_params) > 1:
                raise ValueError("param_validator required too many positional ")
            if any(p.kind == p.KEYWORD_ONLY for p in val_params):
                raise ValueError("param_validator cannot have keyword-only arguments")
    else:
        param_validator = _make_default_bg_paramdefs
    # check post_init_validator
    # TODO: Write signature checkers for param_preprocess, post_init_validator and stream_conversion
    post_init_validator = _make_default_bg_postvalidate if post_init_validator is None else post_init_validator
    stream_conversion = _make_default_bg_streamconversion if stream_conversion is None else stream_conversion
    register_PyCode(func, 'BG_func', (param_preprocess, param_validator, post_init_validator, stream_conversion))


def get_ecdf(s:np.ndarray, offset:float=0.5):
    """
    Return arrays (x, y) for the empirical CDF curve of sample `s`.

    See the code for more info (is a one-liner!).

    Parameters
    ----------
    s : np.ndarray
        sample data
    offset : float, optional 
        Offset to add to the y values of the CDF, the default is 0.5

    Returns
    -------
    x : np.ndarray
        the x values of the empirical CDF
    y : np.ndarray
        the y values of the emperical CDF
    """
    return np.sort(s), np.arange(offset, s.size+offset)*1.0/s.size


def _bg_get_ndet(params:dict, parents:dict)->int:
    """Get number of streams expected to require tail_min specification from param"""
    if params['compute_stream'] == 'any':
        return 1
    ndet = parents['base'].detdef.size
    if params['compute_stream'] == 'single_all':
        ndet += 1
    return ndet


def _bg_tailmin_preprocess(params:tupledict, parents:dict)->tuple[dict]:
    parents = parents.asdict if isinstance(parents, tupledict) else dict(parents)
    params = params.asdict if isinstance(params, tupledict) else dict(params)
    tail_min = np.asarray(params.get('tail_min', 5e-4), dtype=np.float64).reshape(-1)
    ndet = _bg_get_ndet(params, parents)
    if tail_min.size == 1:
        params['tail_min'] = np.repeat(tail_min, ndet)
    elif tail_min.size != ndet:
        raise ValueError(f"Incompatible size of tail_min: Base DetDef requires {ndet} streams, tail_min only has {tail_min.size}")
    parents['base'] = parents['base'].degate()
    return params, parents


def _bg_tailmin_postvalidate(param:Param)->None:
    if param.params['tail_min'].size != _bg_get_ndet(param.params, param.parents):
        raise ValueError("Incorrect number of streams in tail_min")
    

def _bg_tailmin_stream_conversion(params:tupledict, stream_ids:np.ndarray[np.uint8])->dict[str:Hashable]:
    params = params.asdict
    compute_stream = params.pop('compute_stream')
    params.pop('func', None)
    if compute_stream == 'any':
        params['tail_min'] = params['tail_min'][0]
    elif stream_ids.size != 1:
        params['tail_min'] = params['tail_min'][-1]
    else:
        params['tail_min'] = params['tail_min'][stream_ids[0]]
    return params


def exp_mlefit(times:np.ndarray[np.int64], clk_p:float, tail_min:float=500e-6, 
               auto_threshold:bool=False, F_bg:float=2.0)->float:
    """
    Fit sample ``times`` to an exponential distribution using the ML estimator.

    This function computes the rate (Lambda) using the maximum likelihood (ML)
    estimator of the mean waiting-time (Tau), that for an exponentially
    distributed sample is the sample-mean.
    
    This function is primarily used in the ``func`` parameter of :class:`BG` tables.
    

    Parameters
    ----------
    times : np.ndarray[np.int64]
        array of exponetially-distributed samples
    clk_p : float
        Clock time of the times array (seconds in 1 clock of times, ie the unit of times)
    tail_min : float
        Minimum inter-photon time to consider in fitting histogram.
    auto_threshold : bool
        If True, set true tail_min threshold based on initial calculation.
        This threshold is set as :math:`$thresh_{final} = F_{bg} bg_{init}$`
        where :math:`F_{bg}` is the ``F_bg`` argument (default of 2.0), and
        :math:`bg_{init}` is the background rate computed using ``tail_min``.
        If False, use the initial calculation.
    F_bg : float
        Only used when auto_threshold is True. Multiplier of background rate
        to compute the minmum interphoton time to include in final MLE computation.
    
    Returns
    -------
    Lambda : float
        background photon rate
    
    """
    deltaT = np.diff(times)
    tlmin = int(tail_min/clk_p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if auto_threshold:
            tlmin = F_bg * np.mean(deltaT[deltaT>=tlmin]-tlmin)
            if np.isnan(tlmin):
                return np.nan
            tlmin = int(tlmin)
        out = 1.0/np.mean(deltaT[deltaT>=tlmin]-tlmin) / clk_p
    return out


#: ParamDefs for exp_mlefit type BG Param
_bg_mle_paramdefs = (
    ParamDef('tail_min', TV_ndarray(dims=arr_slc[:], mn=0.0), default=500e-6, unit="s"),
    ParamDef('auto_threshold', TV_bool, default=False),
    ParamDef('F_bg', TV_float(mn=0.0), default=2.0),
    )


def _make_bg_mle_paramdefs(params:dict)->tuple[ParamDef,...]:
    """Generate append_params tuple for :func:`exp_mlefit`"""
    return  _bg_mle_paramdefs if params.get('auto_threshold', False) else _bg_mle_paramdefs[:-1]


register_bg_func(exp_mlefit, _bg_tailmin_preprocess, _make_bg_mle_paramdefs,
                 _bg_tailmin_postvalidate, _bg_tailmin_stream_conversion)


def exp_cdffit(times:np.ndarray, clk_p:float, tail_min:float=500e-6, offset:float=0.5,
               auto_threshold:bool=False, F_bg:float=2.0):
    """
    Fit of an exponential model to the empirical CDF of `times`.

    This function computes the rate (Lambda) fitting a line (linear
    regression) to the log of the empirical CDF.
    
    This function is primarily used in the ``func`` parameter of :class:`BG` tables.

    Parameters
    ----------
    times : np.ndarray[np.int64]
        array of exponetially-distributed samples
    clk_p : float
        Clock time of the times array (seconds in 1 clock of times, ie the unit of times)
    tail_min : int
        all samples < `tail_min` are discarded (`tail_min` must be >= 0).
    offset : float, optional
        offset for computing the CDF. default is 0.5
    auto_threshold : bool
        If True, set true tail_min threshold based on initial calculation.
        This threshold is set as :math:`$thresh_{final} = F_{bg} bg_{init}$`
        where :math:`F_{bg}` is the ``F_bg`` argument (default of 2.0), and
        :math:`bg_{init}` is the background rate computed using ``tail_min``.
        If False, use the initial calculation.
    F_bg : float
        Only used when auto_threshold is True. Multiplier of background rate
        to compute the minmum interphoton time to include in final MLE computation.
        
    Returns
    -------
    Lambda : float
        background photon rate
        
    """
    tlmin = int(tail_min / clk_p)
    deltaT = np.diff(times)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if auto_threshold:
            x_cdf, y_cdf = get_ecdf(deltaT[deltaT>=tlmin] - tlmin, offset)
            decr_line = np.log(1-y_cdf)
            tlmin = F_bg/linregress(x_cdf, decr_line).slope
            if np.isnan(tlmin):
                return np.nan
        deltaT = deltaT[deltaT>=tlmin] - tlmin
        x_cdf, y_cdf = get_ecdf(deltaT[deltaT>=tlmin] - tlmin, offset)
        decr_line = np.log(1-y_cdf)
        out = -linregress(x_cdf, decr_line).slope / clk_p
    return out


#: ParamDefs for exp_cdffit type BG Param
_bg_cdf_paramdefs = ( 
    _bg_mle_paramdefs[0],
    ParamDef('offset', TV_float(mn=0.0), default=0.5)
    ) + _bg_mle_paramdefs[1:]


def _make_bg_cdf_paramdefs(params:dict)->tuple[ParamDef,...]:
    """Generate append_params tuple for :func:`exp_cdffit`"""
    return  _bg_cdf_paramdefs if params.get('auto_threshold', False) else _bg_cdf_paramdefs[:-1]


register_bg_func(exp_cdffit, _bg_tailmin_preprocess, _make_bg_cdf_paramdefs,
                 _bg_tailmin_postvalidate, _bg_tailmin_stream_conversion)


def expon_fit_hist(s:np.ndarray[np.int64], bins:float|np.ndarray[np.int64], 
                   s_min:float=0.0, weights:Literal['none', 'hist_counts','inv_hist_counts']='none', 
                   offset:float=0.5)->float:
    """
    Fit an exponential model to the hisogram of ``s``, using least squares.

    Parameters
    ----------
    s : np.ndarray[np.int64]
        array of exponetially-distributed samples.
    bins : float|np.ndarray[np.int64]
        Bins of histogram used to fit data.
    s_min : float, optional
        Minimum sample value to count. The default is 0.0.
    weights : {'none', 'hist_counts','inv_hist_counts'},  optional
        One of the following:
        
            -  ``'none'`` : no weights is applied.
            -  ``'hist_counts'`` each bin has a weight equal to its counts
            -  ``'inv_hist_counts'`` the weight is the inverse of the counts.
            
        The default is 'none'.
        
    offset : Float, optional
        Offset for computing the CDF. See :func:`get_ecdf`. The default is 0.5.

    Raises
    ------
    ValueError
        Bad option for weights.

    Returns
    -------
    lam : float
        photon rate.

    """
    if s_min > 0: 
        s = s[s >= s_min] - s_min
    if s.size < 10:
        return np.nan
    
    counts, bins = np.histogram(s, bins=bins, density=True)
    
    x = bins[:-1] + 0.5*(bins[1] - bins[0])  # bin center position
    y = counts
    x = x[y > 0]
    y = y[y > 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if weights in ('none', None):
            w = np.ones(y.size)
        elif weights == 'hist_counts':
            w = np.sqrt(y*s.size*(bins[1]-bins[0]))
        elif weights == 'inv_hist_counts':
            w = np.sqrt(1./y*s.size*(bins[1]-bins[0]))
        else:
            raise ValueError('Weighting scheme not valid (use: None, or '
                             '"hist_counts")')
        exp_fun = lambda x, rate: rate*np.exp(-x*rate)
        err_fun = lambda rate, x, y, w: (exp_fun(x, rate) - y)*w
        res, _ = leastsq(err_fun, x0=1.0/s.mean(), args=(x, y, w))
        lam = res[0] # convert to int
    return lam


def exp_histfit(times:np.ndarray[np.int64], clk_p:float, tail_min:float=500e-6, binw=50e-6, 
                 weights:str='hist_counts', auto_threshold:bool=False, F_bg:float=2.0):
    """
    Compute background rate with WLS histogram fit of waiting-times.

    Compute the background rate, selecting waiting-times (delays) larger
    than a minimum threshold.

    This function performs a Weighed Least Squares (WLS) fit of the
    histogram of waiting times to an exponential decay.
    
    This function is primarily used in the ``func`` parameter of :class:`BG` tables.

    Parameters
    ----------    
    times : np.ndarray[np.int64] 
        timestamps array from which to extract the background
    clk_p : float
        Clock time of the times array (seconds in 1 clock of times, ie the unit of times)
    tail_min : float
        minimum waiting-time in seconds
    binw : float
        bin width for waiting times, in seconds.
    clk_p : float
        clock period for timestamps in `times`
    weights : str
        one of:
            
            -  ``'none'`` : no weights is applied.
            -  ``'hist_counts'`` each bin has a weight equal to its counts
            -  ``'inv_hist_counts'`` the weight is the inverse of the counts.
    
    auto_threshold : bool
        If True, set true tail_min threshold based on initial calculation.
        This threshold is set as :math:`$thresh_{final} = F_{bg} bg_{init}$`
        where :math:`F_{bg}` is the ``F_bg`` argument (default of 2.0), and
        :math:`bg_{init}` is the background rate computed using ``tail_min``.
        If False, use the initial calculation.
    F_bg : float
        Only used when auto_threshold is True. Multiplier of background rate
        to compute the minmum interphoton time to include in final MLE computation.
    
    Returns
    -------
    bg : float
        Estimated background rate in cps.

    """
    if not times.size:
        return np.nan
    deltaT = np.diff(times)
    tail_min = tail_min/clk_p
    binw_clk = binw/clk_p
    bins = np.arange(0, deltaT.max() - tail_min + 1, binw_clk)
    if auto_threshold:
        lamtemp = expon_fit_hist(deltaT, bins=bins, s_min=tail_min, weights=weights)
        tail_min = F_bg / lamtemp
    lam = expon_fit_hist(deltaT, bins=bins, s_min=tail_min, weights=weights)
    return lam/clk_p


#: ParamDefs for exp_histfit
_bg_hist_paramdefs = (_bg_mle_paramdefs[0], 
    ParamDef('binw', TV_float(mn=0.0), default=50e-6),
    ParamDef('weights', TV_str(isin=('none', 'hist_counts', 'inv_hist_counts')), default='hist_counts')
    ) + _bg_mle_paramdefs[1:]


def _make_bg_histfit_ParamDefs(param:dict)->tuple[ParamDef,...]:
    return _bg_hist_paramdefs if param.get('auto_threshold', False) else _bg_hist_paramdefs[:-1]


register_bg_func(exp_histfit, _bg_tailmin_preprocess, _make_bg_histfit_ParamDefs,
                 _bg_tailmin_postvalidate, _bg_tailmin_stream_conversion)


def get_residuals(deltaT:np.ndarray, Lambda:float, offset:float=0.5)->tuple[np.ndarray, np.ndarray]:
    """
    Returns residuals of sample ``deltaT`` CDF vs an exponential CDF.

    Parameters
    ----------
    deltaT : np.ndarray
        sample of exponentially distributed waiting times
    Lambda : float
        computed rate of the exponential distribution to use as reference
    offset : float
        Offset to add to the empirical CDF. See :func:`get_ecdf` for details.
        The default is 0.5. 

    Returns
    -------
    residuals : np.ndarray
        residuals of empirical CDF compared with analytical
        CDF with rate ``Lambda``.
    
    x : np.ndarray
        x values of residuals
    """
    x, y = get_ecdf(deltaT, offset=offset)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ye = expon.cdf(x, scale=1.0/Lambda)
    residuals = y - ye
    return residuals, x


def param_to_ParamDef(param:inspect.Parameter)->ParamDef:
    """
    Convert a ``inspect.Paramater`` object into a ParamDef.
    Useful when a :class:`smfbursts.datamodel.tables.Param` parameter argument
    is a function, and the parameter should have additional arguments based on
    the signature of said function.

    Parameters
    ----------
    param : inspect.Parameter
        Parameter to convert into ParamDef, gets name and trys to convert type
        to TypeValidator, and checks if has default.

    Returns
    -------
    ParamDef
        Rendered :class:`ParamDef`.

    """
    pdict = dict(name=param.name)
    if param.annotation is not inspect._empty:
        pdict['type_validator'] = TypeValidator.convert_type(param.annotation)
    if param.default is not inspect._empty:
        pdict['default'] = param.default
    return ParamDef(**pdict)


def _append_param_bg_func(params:dict)->tuple[ParamDef,...]:
    """Create ParamDefs from registered bg function"""
    func = params.get('func', BG.param_defs[1].default)
    _, validator, _, _ = get_pycode_subval('BG_func', func)
    # if validator is None:
    #     params = tuple(inspect.signature(func).parameters.values())
    #     params = params[1:] if params[0].kind == params[0].VAR_POSITIONAL else params[2:]
    #     return tuple(param_to_ParamDef(param) for param in params)
    return validator(params)


class BG(ChildPhotonTable):
    r"""
    Background rate estimation. Calls ``func(times, clk_p, **params)``
    
    Params
    ------
        compute_stream : str
            How background for multi-stream ph_sel columns is computed.
            Must be one of:
            
                - ``'single'`` compund ph_sel columns always computed as sum of single streams
                - ``'single_all'`` like single, but if ph_sel is *all*, then compute stream separately
                - ``'any'`` all streams computed separately
        
        
        func : Callable[[np.ndarray[np.int64], ...], float]
            Callable accepting array of times and outputing background.
            The most commonly used function is :func:`exp_mlefit`.
        
        Additional params defined by ``func``.
        The most common :func:`exp_mlefit` has the following parameters:
        
            tail_min : float
                The minimum time (in seconds) 
            auto_threshold : bool
                Whether to set threshold using
    
    
    Parents
    -------
        base : Periods
            The time periods for background calculation.
        
    Columns
    -------
        bg : float, (ph_sel:Ph_sel, )
            background rate for (cnts*s\ :sup:`-1`) ``ph_sel``
        err_KS : float, (ph_sel:Ph_sel, )
            Kolmogorov-Smirnov error metric, computes the error as the max of 
            deviation of the empirical CDF from the fitted CDF.
        err_CM : float, (ph_sel:Ph_sel, )
            Crames-von Mises error metric. 
            Computes :math:`\int_{-\infty}^{\infty} \left[] L_{n} - L_{*} \right]^{2}`.
            Using trapezoid rule for numerical integration.
        rangecounts : float (param:Param[BasePhotonTable], ph_sel:Ph_sel, starttype:str, stoptype:str)
            Expected number of photons in ranges from param for stream ph_sel, using
            duration defined by starttype and stoptype. This is computed as
            ``origin.get_table(param)['dur', starttype, stoptype]*self.origin(self.param)[ph_sel]``
    
    """
    #: :meta private:
    param_defs = (
        ParamDef('compute_stream', TV_str(isin=('single', 'single_all', 'any')), default='single_all'),
        ParamDef('func', TV_PyCode, default=exp_mlefit, append_params=_append_param_bg_func),
                  )
    #: :meta private:
    parent_defs = (
        ParentDef(name='base', table_type=Periods, is_base=True), 
                   )
    #: :meta private:
    column_defs = (
        ColumnDef('bg', (PhSel,), 0, 'some',  get_func='_get_bg', dtype=np.float64,
                  title_func='_get_bg_title',
                  unit=r'(cnts\:s^{-1})', index_unit='cnts s^-1', title_is_tex=True),
        ColumnDef('err_KS', (PhSel, ), 0, 'some', get_func='_get_err_KS', dtype=np.float64, 
                  title_func='_err_KS_title', index_func='_err_KS_index'),
        ColumnDef('err_CM', (PhSel, ), 0, 'some', get_func='_get_err_CM', dtype=np.float64, 
                  title_func='_err_CM_title', index_func='_err_CM_index'),
        ColumnDef('tail_min', (PhSel, ), 0, 'never', get_func='_get_tail_min', 
                  dtype=np.float64, check_func='_check_tail_min', title='tail min', unit='(s)'),
        ColumnDef('rangecounts', (PhSel, str, str), 0, iter_func='_iter_rangecounts',
                  dtype=np.float64, reg_func='_regularizecolumn_rangecounts', mapto=BasePhotonTable,
                  title='bg photons'),
                   )
    
    @classmethod
    def param_preprocess(cls, params, parents):
        params = params.asdict if isinstance(params, tupledict) else dict(params)
        params.setdefault('compute_stream', 'single_all')
        preprocess, _, _ , _ = get_pycode_subval('BG_func', params.get('func',exp_mlefit))
        return preprocess(params, parents)
    
    @classmethod
    def validate_param(cls, param:Param)->None:
        _, _, validate, _ = get_pycode_subval('BG_func', param.params['func'])
        validate(param)

    def _compute_stream(self, phsel:PhSel)->bool:
        """Determine if stream should be computed or split, based on 'compute_stream' param"""
        detdef = self.origin.detdef
        stream_id = detdef.get_stream_ids(phsel)
        cstr = self.param.params['compute_stream']
        if cstr == 'any':
            return True
        elif cstr == 'single':
            return stream_id.size == 1
        return stream_id.size == 1 or stream_id.size == detdef.size
    
    def _get_tail_min(self, phsel:PhSel)->np.ndarray[np.double]:
        """Getter function for determining tail-min threshold. Only useful when 'auto_threshold' is Ture"""
        phsel = phsel.render_positive(self.origin.detdef, convert_all=True)
        params = self.param.params.asdict
        func = params.pop('func')
        _, _, _, stream_proc = get_pycode_subval('BG_func', func)
        stream_id = self.origin.detdef.get_stream_ids(phsel)
        params = stream_proc(self.param.params, stream_id)
        tail_min = params['tail_min']
        if not self.param.params['auto_threshold']:
            return np.repeat(tail_min, self.size)
        out = np.empty(self.size, dtype=np.double)
        params['auto_threshold'] = False
        F_bg = params.pop('F_bg')
        for i, ph_times in enumerate(self.parents['base'].iter_column('ph_times', phsel)):
            out[i] = F_bg / func(ph_times, self.origin.clk_p, **params)
        return out

    @classmethod
    def _check_tail_min(cls, column:Column):
        """
        Column existence check func for tail_min, 
        will prevent creating column of BG param that lacks 'auto_threshold' in param.
        """
        if 'tail_min' not in column.param.params:
            raise ValueError("tail_min column only specified for bg functions that include tail_min argument")
        params = column.param.params
        if params['compute_stream'] != 'any':
            detdef = column.param.detdef
            stream_ids = detdef.get_stream_ids(column.keytup[0])
            if stream_ids.size != 1:
                if stream_ids.size != detdef.size or params['compute_stream'] != 'single_all':
                    raise ValueError("Cannot get non-all multi-stream tail_min for single stream bg column")

    @cite('IngargiolaPLOSOne2016', purpose='background analysis with smfBursts')
    def _get_bg(self, phsel:PhSel):
        """Getter function for bg column"""
        phsel = phsel.render_positive(self.origin.detdef, convert_all=True) # ensures consistent representation in DiskDict
        if self._compute_stream(phsel):
            out = self._calc_bg(phsel)
            self._add_column('bg', (phsel,), out)
        else:
            stream_id = self.origin.detdef.get_stream_ids(phsel)
            out = sum(self['bg', self.origin.detdef.stream_ids_to_PhSel(st_id)] for st_id in stream_id)
        return out

    @classmethod
    def _get_bg_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """Title getter function for bg column"""
        title = _title_sels('bg', origin, col.keytup[0])[0]
        title = _title_unit_append(title, 'cnts s^{-1}', include_unit)
        return f'${title}$'

    def _calc_bg(self, phsel:PhSel)->np.ndarray[np.float64]:
        """Compute bg column if cannot compute as sum from existing columns"""
        periods = self.parents['base']
        out = np.empty(self.size, dtype=np.float64)
        params = self.param.params
        func = params['func']
        _, _, _, stream_proc = get_pycode_subval('BG_func', func)
        stream_id = self.origin.detdef.get_stream_ids(phsel)
        params = stream_proc(params, stream_id)
        for i, ph_times in enumerate(periods.iter_column('ph_times', phsel)):
            out[i] = func(ph_times, self.origin.clk_p, **params)
        return out

    def _get_err_KS(self, phsel):
        """Getter function for err_KS column (Kolmogrov-Smirnov error)"""
        phsel = phsel.render_positive(self.origin.detdef, convert_all=True) # ensures consistent representation in DiskDict
        if self._compute_stream(phsel):
            out = self._calc_err_KS(phsel)
            self._add_column('err_KS', (phsel, ), out)
        else:
            stream_id = self.origin.detdef.get_stream_ids(phsel)
            out = sum(self['err_KS', self.origin.detdef.stream_ids_to_Ph_sel(st_id)] for st_id in stream_id)
        return out

    @classmethod
    def _err_KS_title(cls, col:Column, include_unit:bool=False, origin:DataSet=None)->str:
        """Title getter function for err_KS column"""
        title = _title_sels('bg', origin, col.keytup[0])[0]
        return fr'$KS error:\: D({title})$'

    @classmethod
    def _err_KS_index(cls, col:Column, include_unit:bool=False, origin:DataSet=None)->str:
        """Index name getter function for err_KS column"""
        return f'KS err BG {str(col.keytup[0])}'

    def _calc_err_KS(self, phsel:PhSel)->np.ndarray[np.float64]:
        """Compute err_KS column if cannot compute as sum from existing columns"""
        phsel = phsel.render_positive(self.origin.detdef, convert_all=True)
        _, _, _, stream_proc = get_pycode_subval('BG_func', self.param.params['func'])
        params = stream_proc(self.param.params, self.origin.detdef.get_stream_ids(phsel))
        tail_min = params['tail_min']/self.origin.clk_p
        offset = self.param.params.get('offset', 0.5)
        out = np.empty(self.size, dtype=np.float64)
        for i, (times, bg) in enumerate(zip(self.parents['base'].iter_column('ph_times', phsel), self.iter_column('bg', phsel))):
            s = np.diff(times) - tail_min
            out[i] = np.abs(get_residuals(s[s >= 0], bg*self.origin.clk_p, offset)[0]).max()
        return out

    @classmethod
    def _err_CM_title(cls, col:Column, include_unit:bool=False, origin:DataSet=None)->str:
        """Title getter function for err_CM column"""
        title = _title_sels('bg', origin, col.keytup[0])[0]
        return fr'$CM error:\: T({title})$'

    @classmethod
    def _err_CM_index(cls, col:Column, include_unit:bool=False, origin:DataSet=None)->str:
        """Index name getter function for err_CM column"""
        return f'CM err BG {str(col.keytup[0])}'

    def _get_err_CM(self, phsel:PhSel):
        """TGetter function for err_CM column (Cramer von Misses error)"""
        phsel = phsel.render_positive(self.origin.detdef, convert_all=True) # ensures consistent representation in DiskDict
        if self._compute_stream(phsel):
            out = self._calc_err_CM(phsel)
            self._add_column('err_CM', (phsel, ), out)
        else:
            stream_id = self.origin.setup.detdef.get_stream_ids(phsel)
            out = sum(self['err_CM', self.origin.setup.detdef.stream_ids_to_Ph_sel(st_id)] for st_id in stream_id)
        return out

    def _calc_err_CM(self, phsel:PhSel)->np.ndarray[np.float64]:
        """Compute err_CM column if cannot compute as sum from existing columns"""
        phsel = phsel.render_positive(self.origin.detdef, convert_all=True)
        _, _, _, stream_proc = get_pycode_subval('BG_func', self.param.params['func'])
        params = stream_proc(self.param.params, self.origin.detdef.get_stream_ids(phsel))
        tail_min = params['tail_min']/self.origin.clk_p
        offset = self.param.params.get('offset', 0.5)
        out = np.empty(self.size, dtype=np.float64)
        for i, (times, bg) in enumerate(zip(self.parents['base'].iter_column('ph_times', phsel), 
                                            self.iter_column('bg', phsel))):
            s = np.diff(times) - tail_min
            resid, x_resid = get_residuals(s[s >= 0], bg*self.origin.clk_p, offset)
            out[i] = np.trapezoid(resid**2, x=x_resid)
        return out

    @classmethod
    def _regularizecolumn_rangecounts(cls, *args)->tuple[PhSel, str, str]:
        """Column regularization function for mapped column range-counts"""
        if len(args) < 2:
            raise ValueError("no defaults for destination param or Ph_sel, must specify")
        param, ph_sel, startstoptype = args[0], args[1], args[2:]
        starttype, stoptype = _regularize_column_startstop(*startstoptype)
        return param, ph_sel, starttype, stoptype

    def _iter_rangecounts(self, param:Param, ph_sel:PhSel, starttype:ColKeyStart, stoptype:ColKeyStop)->Iterator[float]:
        """Iter function for rangecounts mapped column"""
        dest_table = self.origin.get_table(param)
        period_iter =  zip(self.iter_column('bg', ph_sel),
                           self.parents['base'].iter_column('periods',0),
                           self.parents['base'].iter_column('periods',1))
        bg, pstart, pstop = next(period_iter)
        for dstart, dstop in zip(dest_table.iter_column(starttype), dest_table.iter_column(stoptype)):
            bgc = 0.0 # bgc serves as an accumulatro
            # while loop for all periods that end before dstop
            while pstop < dstart:
                try:
                    bg, pstart, pstop = next(period_iter)
                except StopIteration:
                    bgc = np.nan
                    pstop = np.inf # causes loop to break, avoids break inside except
            if np.isnan(bgc):
                yield bgc
                continue
            cstart = dstart
            while dstop > pstop:
                bgc += bg*(pstop-pstart)*self.origin.clk_p
                cstart = pstop
                try:
                    bg, pstart, pstop = next(period_iter)
                except StopIteration:
                    bgc = np.nan
                    pstop = np.inf # causes loop to break, avoids break inside except
            # final addition add time from beginning of current section to end
            if not np.isnan(bgc):
                bgc += bg*(dstop-cstart)*self.origin.clk_p
            yield bgc

    @classmethod
    def _get_rangecounts_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """Title getter function for rangecounts column"""
        title = _title_sels('_{n}bg', origin, col.keytup[1])[0]
        title = _title_startstop_append(title, col.keytup[2], col.keytup[3])
        title = _title_unit_append(title, 'photons', include_unit)
        return f'${title}$'


def make_bg_param(data:PhotonData, tail_min:float=500e-6, period:float=60.0, 
            func:BGFuncType=exp_mlefit, **kwargs)->Param:
    """
    Make a background computation :class:`Param`. (based on :class:`BG`)

    Parameters
    ----------
    data : PhotonData
        Data for which gate should be created.
    tail_min : float, optional
        Minimum photon separation (in seconds) to consider in bg computation. 
        The default is 500e-6.
    period : float, optional
        Size (in seconds) of 1 backgroun assesment periods. The default is 60.0.
    func : Callable[[np.ndarray[np.int64],float,...], float], optional
        Function used to comptute background. The default is exp_mlefit.
    **kwargs : Hashable
        Any additional arguments to be passed to func for BG computation.

    Returns
    -------
    Param
        :class:`Param` based on :class:`BG` specifying background computation.

    """
    prd = Param(Periods, {'period':period, 'detdef':data.detdef})
    kwargs.update({'func':func, 'tail_min':tail_min})
    return Param(BG, kwargs, {'base':prd})


def get_bg_table(data:PhotonData, tail_min:float=500e-6, period:float=60.0, 
                 func:BGFuncType=exp_mlefit, **kwargs)->BG:
    """
    Get a :class:`BG` table from data.

    Parameters
    ----------
    data : PhotonData
        Data for which gate should be created.
    tail_min : float, optional
        Minimum photon separation (in seconds) to consider in bg computation. 
        The default is 500e-6.
    period : float, optional
        Size (in seconds) of 1 backgroun assesment periods. The default is 60.0.
    func : Callable[[np.ndarray[np.int64],float,...], float], optional
        Function used to comptute background. The default is exp_mlefit.
    **kwargs : Hashable
        Any additional arguments to be passed to func for BG computation.

    Returns
    -------
    Param
        :class:`BG` table computing background of data.

    """
    return data.get_table(make_bg_param(data, tail_min, period, func, **kwargs))