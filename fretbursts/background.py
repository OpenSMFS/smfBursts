#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created : 9/10/20025
# Author: Paul David Harris
# email: harrip@gmail.com
"""
Module for assesment of background rates in data.

Defines base and child tables :class:`Periods` and :class:BG` respectively for
division of data in to consecutive periods (for background assemsment) and 
computation of background, respectively.

Additional background functions are defined.
"""
from typing import Union, ClassVar
from collections.abc import Hashable, Callable, Sequence, Iterator
import inspect
import warnings

import numpy as np
from scipy.stats import linregress, expon
from scipy.optimize import leastsq

from .datamodel.utils import tupledict
from .datamodel.immutabledata import (TypeValidator, TV_float, TV_str, TV_bool, 
                                      TV_PyCode, register_PyCode, get_pycode_subval)
from .datamodel.tables import ParamDef, ParentDef, ColumnDef, Param, Column, DataSet, as_paramdict
from .datamodel.citations import cite
from .ph_sel import PhSel, DetDef, TV_DetDef
from .photondata import (BasePhotonTable, ChildPhotonTable, PhotonData, 
                         _normalize_column_startstop, _title_sels, 
                         _title_startstop_append, _title_unit_append,
                         make_base_column_defs)

import fretbursts.cfuncs as fbc


def _periods_title_func(col:Column, include_unit:bool=False)->str:
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
    start_at : str
        One of ``'time_min'``, ``'zero'``, ``'under'``, or ``'over'``. Defines when 
        first period starts relative to first photon in data.
        
        Options::
            
            - ``'time_min'`` start of first period is first photon in data
            - ``'zero'`` start of first period is time = 0
            - ``'under'`` start of first period is the time that is a integer multiple of 
              period, and less than (greatest possible) the time of the first photon
            - ``'over'`` similar to under, the least possible integer multiple of 
              period greater than the time of the first photon
        
        ``start_at`` must be specified with ``stop_at```, and these are exclusive of 
        specifying ``start`` and ``stop``. Default (if ``start`` is not defined)
        is ``'time_min'``
    stop_at : str
        One of ``'under'`` or ``'over'``. Defines stop time of last period.
        
        Options::
            
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
        
    And all columns in `basephotoncolumns`_ for full list of columns.
    
    """
    row_name:ClassVar[str] = "Periods"
    _origin: PhotonData
    
    param_defs = (
        ParamDef('period', TV_float(mn=0.0), default=60.0),
        ParamDef('start_at', TV_str(isin=('time_min', 'zero', 'under', 'over')), required=False),
        ParamDef('stop_at', TV_str(isin=('under', 'over')), required=False), 
        ParamDef('start', TV_float, required=False), 
        ParamDef('stop', TV_float, required=False),
        ParamDef('detdef', TV_DetDef, required=True)
                  )
    parent_defs = tuple()
    column_defs = (
        ColumnDef('periods', tuple(), 1, 'all', dtype=np.int64, 
                  title_func='_get_periods_title', index_func='_get_periods_index', unit='clk_p'),
        ColumnDef('start', tuple(), 0, remap='_replace_column_startstop'),
        ColumnDef('stop', tuple(), 0, remap='_replace_column_startstop')
                  ) + make_base_column_defs(skip=('start', 'stop'))
    # column_defs = (
    #     ColumnDef('periods', tuple(), 1, 'all', dtype=np.int64, 
    #               title_func='_get_periods_title', index_func='_get_periods_index', unit='clk_p'),
    #     ColumnDef('istart', tuple(), 0, 'all', dtype=np.int64, title='istart', unit='clk_p'), 
    #     ColumnDef('istop', tuple(), 0, 'all', dtype=np.int64, title='istop', unit='clk_p'),
    #     ColumnDef('start', tuple(), 0, remap='_replace_column_startstop'),
    #     ColumnDef('stop', tuple(), 0, remap='_replace_column_startstop')
    #                ) + basetimecoldefs
    
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
        istart, istop = fbc.index_ranges(self.origin.times, periods[:-1], periods[1:])
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
        return 'periods', keys, 0 if col == 'start' else 1
    
    @classmethod
    def _detdef(cls, param:Param)->DetDef:
        return param.params['detdef']
    
    @classmethod
    def _get_periods_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        if 'offset' not in col:
            out = 'periods'
        else:
            out = 'stop' if col.offset else 'start'
        if include_unit:
            out += ' clk_p'
        return out
    
    @classmethod
    def _get_periods_index(cls, col:Column, include_unit:bool=False)->str:
        return cls._periods_title_func(col, include_unit)
    

#########################################################
### Functions for computing BackGround
#########################################################
BGFuncType = Callable[[np.ndarray[np.int64],float,...], float]


def register_bg_func(func:BGFuncType, 
                     param_validator:Callable[[dict,],tuple[ParamDef,...]]=None)->None:
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
    # Check param validator
    if param_validator is not None:
        if not callable(param_validator):
            raise TypeError("param_validator must be callable accepting single dict and returning tuple of ParamDefs")
            val_params = tuple(p for p in inspect.signature(param_validator))
            if sum(p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) or p.default is not inspect._empty
                   for p in val_params) > 1:
                raise ValueError("param_validator required too many positional ")
            if any(p.kind == p.KEYWORD_ONLY for p in val_params):
                raise ValueError("param_validator cannot have keyword-only arguments")
    register_PyCode(func, 'BG_func', param_validator)
            


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



def exp_mlefit(times:np.ndarray[np.int64], clk_p:float, tail_min:float=500e-6, 
               auto_threshold:bool=False, F_bg:float=2.0)->float:
    """
    Fit sample ``times`` to an exponential distribution using the ML estimator.

    This function computes the rate (Lambda) using the maximum likelihood (ML)
    estimator of the mean waiting-time (Tau), that for an exponentially
    distributed sample is the sample-mean.

    Parameters
    ----------
        times : np.ndarray[np.int64]
            array of exponetially-distributed samples
        tail_min : float
            all samples < `tail_min` are discarded (`tail_min` must be >= 0).
        offset : float, optional
            offset for computing the CDF. **Ignored in this function** default is 0.5
        
    Returns:
        Lambda : float
            photon rate
        
    """
    deltaT = np.diff(times)
    tlmin = int(tail_min/clk_p)
    if auto_threshold:
        tlmin = F_bg * np.mean(deltaT[deltaT>=tlmin]-tlmin)
        if np.isnan(tlmin):
            return np.nan
        tlmin = int(tlmin)
    return 1.0/np.mean(deltaT[deltaT>=tlmin]-tlmin) / clk_p

#: ParamDefs for exp_mlefit type BG Param
_bg_mle_paramdefs = (
    ParamDef('tail_min', TV_float(mn=0.0), default=500e-6, unit="s"),
    ParamDef('auto_threshold', TV_bool, default=False),
    ParamDef('F_bg', TV_float(mn=0.0), default=2.0),
    )


def _make_bg_mle_paramdefs(params:dict)->tuple[ParamDef,...]:
    """Generate append_params tuple for :func:`exp_mlefit`"""
    return  _bg_mle_paramdefs if params.get('auto_threshold', False) else _bg_mle_paramdefs[:-1]
    

register_bg_func(exp_mlefit, _make_bg_mle_paramdefs)


def exp_cdffit(times:np.ndarray, clk_p:float, tail_min:float=500e-6, offset:float=0.5,
               auto_threshold:bool=False, F_bg:float=2.0):
    """
    Fit of an exponential model to the empirical CDF of `times`.

    This function computes the rate (Lambda) fitting a line (linear
    regression) to the log of the empirical CDF.

    Parameters
    ----------
        times : np.ndarray[np.int64]
            array of exponetially-distributed samples
        tail_min : int
            all samples < `tail_min` are discarded (`tail_min` must be >= 0).
        offset : float, optional
            offset for computing the CDF. default is 0.5
        
    Returns:
        Lambda : float
            photon rate
        
    """
    tlmin = int(tail_min / clk_p)
    deltaT = np.diff(times)
    if auto_threshold:
        x_cdf, y_cdf = get_ecdf(deltaT[deltaT>=tlmin] - tlmin, offset)
        decr_line = np.log(1-y_cdf)
        tlmin = F_bg/linregress(x_cdf, decr_line).slope
        if np.isnan(tlmin):
            return np.nan
    deltaT = deltaT[deltaT>=tlmin] - tlmin
    x_cdf, y_cdf = get_ecdf(deltaT[deltaT>=tlmin] - tlmin, offset)
    decr_line = np.log(1-y_cdf)
    return linregress(x_cdf, decr_line).slope / clk_p

#: ParamDefs for exp_cdffit type BG Param
_bg_cdf_paramdefs = ( 
    _bg_mle_paramdefs[0],
    ParamDef('offset', TV_float(mn=0.0), default=0.5)
    ) + _bg_mle_paramdefs[1:]


def _make_bg_cdf_paramdefs(params:dict)->tuple[ParamDef,...]:
    """Generate append_params tuple for :func:`exp_cdffit`"""
    return  _bg_cdf_paramdefs if params.get('auto_threshold', False) else _bg_cdf_paramdefs[:-1]


register_bg_func(exp_cdffit, _make_bg_cdf_paramdefs)


def expon_fit_hist(s, bins, s_min:float=0.0, weights:str='none', offset=0.5):
    """Fit of an exponential model to the histogram of `s` using least squares.

    Parameters
    ----------
    s : np.ndarray[np.int64]
        array of exponetially-distributed samples
    bins : Union[float, np.ndarray[np.int64]]
        if float is the bin width, otherwise is the
        array of bin edges (passed to `numpy.histogram`)
    s_min : float
        all samples < `s_min` are discarded
        (`s_min` must be >= 0).
    weights : str
        one of::
            -  ``'none'`` : no weights is applied.
            -  ``'hist_counts'`` each bin has a weight equal to its counts
            -  ``'inv_hist_counts'`` the weight is the inverse of the counts.
    offset :  float 
        offset for computing the CDF. See :func:`get_ecdf`.
    
    Returns
   --------
        A 4-tuple of the fitted rate (1/life-time), residuals array,
        residuals x-axis array, sample size after threshold.
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

    if weights == 'none':
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

    res = leastsq(err_fun, x0=1./(s.mean()), args=(x, y, w))
    lam = res[0]

    return lam


def exp_histfit(times:np.ndarray[np.int64], clk_p:float, tail_min:float=500e-6, binw=50e-6, 
                 weights:str='hist_counts', auto_threshold:bool=False, F_bg:float=2.0):
    """Compute background rate with WLS histogram fit of waiting-times.

    Compute the background rate, selecting waiting-times (delays) larger
    than a minimum threshold.

    This function performs a Weighed Least Squares (WLS) fit of the
    histogram of waiting times to an exponential decay.

    Parameters
    ----------    
    times : np.ndarray[np.int64] 
        timestamps array from which to extract the background
    tail_min : float
        minimum waiting-time in seconds
    binw : float
        bin width for waiting times, in seconds.
    clk_p : float
        clock period for timestamps in `times`
    weights : str
        one of::
            
            -  ``'none'`` : no weights is applied.
            -  ``'hist_counts'`` each bin has a weight equal to its counts
            -  ``'inv_hist_counts'`` the weight is the inverse of the counts.
    
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
        lamtemp = expon_fit_hist(deltaT, bins=bins, s_min=tail_min, weights=weights)[0]
        tail_min = F_bg / lamtemp
    res = expon_fit_hist(deltaT, bins=bins, s_min=tail_min, weights=weights)

    lam, residuals, x_residuals, s_size = res
    lam /= clk_p
    return lam


#: ParamDefs for exp_histfit
_bg_hist_paramdefs = (_bg_mle_paramdefs[0], 
    ParamDef('binw', TV_float(mn=0.0), default=50e-6),
    ParamDef('weights', TV_str(isin=('none', 'hist_counts', 'inv_hist_counts')), default='hist_counts')
    ) + _bg_mle_paramdefs[1:]


def _make_bg_histfit_ParamDefs(param:dict)->tuple[ParamDef,...]:
    return _bg_hist_paramdefs if param.get('auto_threshold', False) else _bg_hist_paramdefs[:-1]


register_bg_func(exp_histfit, _make_bg_histfit_ParamDefs)


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
    ye = expon.cdf(x, scale=1.0/Lambda)
    residuals = y - ye
    return residuals, x


def param_to_ParamDef(param:inspect.Parameter)->ParamDef:
    """
    Convert a ``inspect.Paramater`` object into a ParamDef.
    Useful when a :class:`fretbursts.datamodel.tables.Param` parameter argument
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
        Rendered ParamDef.

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
    validator = get_pycode_subval('BG_func', func)
    if validator is None:
        params = tuple(inspect.signature(func))
        params = params[1:] if params[0].kind == params[0].VAR_POSITIONAL else params[2:]
        return tuple(param_to_ParamDef(param) for param in params)
    return validator(params)


class BG(ChildPhotonTable):
    r"""
    Background rate estimation. Calls ``func(times, clk_p, **params)``
    
    Params
    ------
    compute_stream : str
        How background for multi-stream ph_sel columns is computed.
        Must be one of::
            
            - ``'single'`` compund ph_sel columns always computed as sum of single streams
            - ``'single_all'`` like single, but if ph_sel is *all*, then compute stream separately
            - ``'any'`` all streams computed separately
    
    
    func : Callable[[np.ndarray[np.int64], ...], float]
        Callable accepting array of times and outputing background.
    
    Additional params defined by ``func``
    
    
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
    param_defs = (
        ParamDef('compute_stream', TV_str(isin=('single', 'single_all', 'any')), default='single_all'),
        ParamDef('func', TV_PyCode, default=exp_mlefit, append_params=_append_param_bg_func),
                  )
    parent_defs = (
        ParentDef(name='base', table_type=Periods, is_base=True), 
                   )
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
                  dtype=np.float64, norm_func='_normalizecolumn_rangecounts', mapto=BasePhotonTable,
                  title='bg photons'),
                   )
    
    def __init_columns__(self):
        pass
    
    def _compute_stream(self, stream_id:np.ndarray[np.int8])->bool:
        """Determine if stream should be computed or split, based on 'compute_stream' param"""
        cstr = self.param.params['compute_stream']
        return stream_id.size == 1 or cstr == 'any' or (cstr == 'all' and stream_id.size == self.origin.setup.detdef.size)
    
    def _get_tail_min(self, ph_sel:PhSel)->np.ndarray[np.double]:
        if not self.param.params['auto_threshold']:
            return self.param.params['tail_min']
        stream_id = self.origin.setup.detdef.get_stream_ids(ph_sel)
        if not self._compute_stream(stream_id):
            warnings.warn("getting tail_min for compound stream, this tail_min is not used to compute background")
        periods = self.parents['base']
        out = np.empty(self.size, dtype=np.double)
        times = self.origin.times
        mask = np.isin(self.origin.dets, stream_id)
        kwargs = self.param.params.asdict
        for key in ('compute_stream', 'func'):
            kwargs.pop(key)
        kwargs['auto_threshold'] = False
        for i, (start, stop) in enumerate(zip(periods['istart',], periods['istop',])):
            out[i] = self.param.params['F_bg'] / self.param.params['func'](times[start:stop][mask[start:stop]], self.origin.clk_p, **kwargs)
        return out
    
    @classmethod
    def _check_tail_min(cls, param:Param):
        if 'tail_min' not in param.params:
            raise ValueError("tail_min column only specified for bg functions that include tail_min argument")
    
    @cite('IngargiolaPLOSOne2016', purpose='background analysis with FRETBursts')
    def _get_bg(self, ph_sel:PhSel):
        ph_sel = ph_sel.render_positive(self.origin.setup.detdef, convert_all=True) # ensures consistent representation in DiskDict
        if ('bg', ph_sel) in self._cache:
            return self._cache['bg', ph_sel]
        stream_id = self.origin.setup.detdef.get_stream_ids(ph_sel)
        if self._compute_stream(stream_id):
            out = self._calc_bg(stream_id)
            self._add_column('bg', (ph_sel,), out)
        else:
            out = sum(self['bg', self.origin.setup.detdef.stream_ids_to_PhSel(st_id)] for st_id in stream_id)
        return out
    
    @classmethod
    def _get_bg_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        title = _title_sels('bg', origin, col.keytup[0])[0]
        title = _title_unit_append(title, 'cnts s^{-1}', include_unit)
        return f'${title}$'
    
    def _calc_bg(self, stream_id:np.ndarray[np.uint8])->np.ndarray[np.float64]:
        times = self.origin.times
        mask = np.isin(self.origin.dets, stream_id)
        periods = self.parents['base']
        out = np.empty(self.size, dtype=np.float64)
        params = self.param.params.asdict
        params.pop('compute_stream')
        func = params.pop('func')
        for i, (start, stop) in enumerate(zip(periods['istart',], periods['istop',])):
            out[i] = func(times[start:stop][mask[start:stop]], self.origin.clk_p, **params)
        return out
    
    def _get_err_KS(self, ph_sel):
        ph_sel = ph_sel.render_positive(self.origin.detdef, convert_all=True) # ensures consistent representation in DiskDict
        if ('err_KS', ph_sel) in self._cache:
            return self._cache['err_KS', ph_sel]
        stream_id = self.origin.setup.detdef.get_stream_ids(ph_sel)
        if self._compute_stream(stream_id):
            out = self._calc_err_KS(ph_sel)
            self._add_column('err_KS', (ph_sel, ), out)
        else:
            out = sum(self['err_KS', self.origin.setup.detdef.stream_ids_to_Ph_sel(st_id)] for st_id in stream_id)
        return out
    
    @classmethod
    def _err_KS_title(cls, col:Column, include_unit:bool=False, origin:DataSet=None)->str:
        title = _title_sels('bg', origin, col.keytup[0])[0]
        return f'$KS error:\: D({title})$'
    
    @classmethod
    def _err_KS_index(cls, col:Column, include_unit:bool=False, origin:DataSet=None)->str:
        return f'KS err BG {str(col.keytup[0])}'
    
    def _calc_err_KS(self, ph_sel:PhSel)->np.ndarray[np.float64]:
        tail_min = self.param.params['tail_min']/self.origin.clk_p
        offset = self.param.params.get('offset', 0.5)
        out = np.empty(self.size, dtype=np.float64)
        for i, (times, bg) in enumerate(zip(self.parents['base'].iter_column('ph_times', ph_sel), self.iter_column('bg', ph_sel))):
            s = np.diff(times) - tail_min
            out[i] = np.abs(get_residuals(s[s >= 0], bg*self.origin.clk_p, offset)[0]).max()
        return out
    
    @classmethod
    def _err_CM_title(cls, col:Column, include_unit:bool=False, origin:DataSet=None)->str:
        title = _title_sels('bg', origin, col.keytup[0])[0]
        return f'$CM error:\: T({title})$'
    
    @classmethod
    def _err_CM_index(cls, col:Column, include_unit:bool=False, origin:DataSet=None)->str:
        return f'CM err BG {str(col.keytup[0])}'
    
    def _get_err_CM(self, ph_sel):
        ph_sel = ph_sel.render_positive(self.origin.detdef, convert_all=True) # ensures consistent representation in DiskDict
        if ('err_CM', ph_sel) in self._cache:
            return self._cache['err_CM', ph_sel]
        stream_id = self.origin.setup.detdef.get_stream_ids(ph_sel)
        if self._compute_stream(stream_id):
            out = self._calc_err_CM(ph_sel)
            self._add_column('err_CM', (ph_sel, ), out)
        else:
            out = sum(self['err_CM', self.origin.setup.detdef.stream_ids_to_Ph_sel(st_id)] for st_id in stream_id)
        return out
    
    def _calc_err_CM(self, ph_sel:PhSel)->np.ndarray[np.float64]:
        tail_min = int(self.param.params['tail_min']/self.origin.clk_p)
        offset = self.param.params.get('offset', 0.5)
        out = np.empty(self.size, dtype=np.float64)
        for i, (times, bg) in enumerate(zip(self.parents['base'].iter_column('ph_times', ph_sel),self.iter_column('bg', ph_sel))):
            s = np.diff(times) - tail_min
            resid, x_resid = get_residuals(s[s >= 0], bg*self.origin.clk_p, offset)
            out[i] = np.trapz(resid**2, x=x_resid)
        return out
    
    @classmethod
    def _normalizecolumn_rangecounts(cls, *args)->tuple[PhSel, str, str]:
        if len(args) < 2:
            raise ValueError("no defaults for destination param or Ph_sel, must specify")
        param, ph_sel, startstoptype = args[0], args[1], args[2:]
        starttype, stoptype = _normalize_column_startstop(*startstoptype)
        return param, ph_sel, starttype, stoptype
            
    def _iter_rangecounts(self, param:Param, ph_sel:PhSel, starttype:str, stoptype:str)->Iterator[float]:
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
        title = _title_sels('_{n}bg', origin, col.keytup[1])[0]
        title = _title_startstop_append(title, col.keytup[2], col.keytup[3])
        title = _title_unit_append(title, 'photons', include_unit)
        return f'${title}$'


def make_bg_param(data:PhotonData, tail_min:float=500e-6, period:float=60.0, 
            func:BGFuncType=exp_mlefit, **kwargs)->Param:
    prd = Param(Periods, {'period':period, 'detdef':data.detdef})
    args = {'func':exp_mlefit, 'tail_min':tail_min}
    args.update(kwargs)
    return Param(BG, kwargs, {'base':prd})


def get_bg_table(data:PhotonData, tail_min:float=500e-6, period:float=60.0, 
                 func:BGFuncType=exp_mlefit, **kwargs)->BG:
    return data.get_table(make_bg_param(data, tail_min, period, func, **kwargs))