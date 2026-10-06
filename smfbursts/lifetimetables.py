#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for evaluation of parameters related to fluoresence lifetime decays.

This module is also accessible as ``smf.lt``.


.. |minimize| replace:: `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`__
.. |optimizeresult| replace:: `OptimizeResult <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html>`__
.. |Digman| replace:: `Digman et. al. 2008 <https://doi.org/10.1529/biophysj.107.120154>`__
"""
from typing import Any, TypedDict
from collections.abc import Callable, Sequence
from numbers import Real

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, OptimizeResult

from .datamodel.utils import arr_slc
from .datamodel.immutabledata import TV_bool, TV_ndarray, tupledict
from .datamodel.tables import (
    ParamDef, ParentDef, ColumnDef, Param, Column, GateGroup, as_paramdict, parammethod
    )
from .cite import cite

from .photondata import (
    PhotonData, PhotonDataList, PhotonDataS, BasePhotonTable, ChildPhotonTable,
    _title_sels, _validate_lifetime, TV_irfstyle, IRFStyle, _get_nmstyle_info
    )
from .ph_sel import PhSel


def _phasor_trig(origin:PhotonData, phsel:PhSel, style:IRFStyle, omega:float, 
                 func:Callable[[np.ndarray[np.float64]],np.ndarray[np.float64]]
                 )->tuple[int,int,np.ndarray[np.float64]]:
    ex_start, ex_stop, tcspc_unit, thresh = _get_nmstyle_info(origin, phsel, style)
    trig = func((np.arange(ex_start, ex_stop,1)-thresh)*tcspc_unit*omega)
    thresh = np.ceil(thresh, casting='unsafe', dtype=np.int64)
    return ex_start, thresh, trig


def _phasor_trigs(origin:PhotonData, phsel:PhSel, style:IRFStyle, omegas:np.ndarray[np.float64], 
                  func:Callable[[np.ndarray[np.float64]],np.ndarray[np.float64]])->tuple[tuple[PhSel,...],tuple[int,...],tuple[int,...],tuple[float,...]]:
    stream_ids = origin.detdef.get_stream_ids(phsel)
    sels = tuple(origin.detdef.stream_ids_to_PhSel(i) for i in stream_ids)
    ex_starts, threshs, trigs = zip(*(_phasor_trig(origin, sel, style, omega, func) 
                                      for sel, omega in zip(sels, omegas)))
    return sels, ex_starts, threshs, trigs


def _phasor_prod_exclude(trigs:np.ndarray[np.float64], nhs:np.ndarray[np.int64], 
                         threshs:np.ndarray[np.int64], ex_starts:np.ndarray[np.int64])->np.float64:
    nhs_ = [nh[nh >= thresh] - ex_start for nh, thresh, ex_start in zip(nhs, threshs, ex_starts)]
    num = sum(trig[nh].sum() if np.size else 0.0 for trig, nh in zip(trigs, nhs_))
    dem = sum(nh.size for nh in nhs_)
    return num / dem if dem else np.nan


def _phasor_prod_all(trigs:np.ndarray[np.float64], nhs:np.ndarray[np.int64], 
                     threshs:np.ndarray[np.int64], ex_starts:np.ndarray[np.int64])->np.float64:
    num = sum(trig[nh-ex_start].sum() if nh.size else 0.0 for trig, nh, ex_start in zip(trigs, nhs, ex_starts)) 
    dem = sum(nh.size for nh in nhs)
    return num / dem if dem else np.nan


class Phasor(ChildPhotonTable):
    r"""
    Phasor representation for pulsed excitation data (|Digman|).
    
    Note that this class is made a top-level class, ie can be accessed as
    ``smfbursts.PhotonData``.
    
    Computes the phasor of the nanotimes in each time period (usually burst).
    
    .. math::
        
        g = \displaystyle\int_{t_{start}}^{t_{end}}{I(t)\cos(\omega t)dt} \equiv \sum_{i=1}^{N}{t_{i}\cos(\omega t_{i})} \
        
        s = \displaystyle\int_{t_{start}}^{t_{end}}{I(t)\sin(\omega t)dt} \equiv \sum_{i=1}^{N}{t_{i}\sin(\omega t_{i})}
        
    Where :math:`\omega` is the anglar frequency, this can be specifed aribrarily
    in the param, or using the laser repetition rate 
    (as originally proposed in |Digman|), and :math:`t_{start}` is
    either the start of the excitation period, or time 0, 
    depending on option in ``exclude`` parameter,
    and :math`t_{end}` is the end of the excitation window.
    
    All times shifted by the option in the parameter ``start``, which sets the
    time in the excitation window which is treated as time 0.
    
    
    Params
    ------
        omega : np.ndarray[np.float64]
            angular frequence to use in computation of :math:`g` and :math:`s`
            factors, 1 element per excitation channel. If specified as single
            value, will automatically be exanded to use same value for each
            excitation channel. Values of :math:`0` and :math:`-1` reserved for
            automatic computation based on alternation periods, according to
            :math:`\omega = 2\pi / T` where if :math:`-1`, :math:`T` is the 
            duration of the excitation window of the given excitation,
            (difference between start and stop times of the given excitation window). 
            While if :math:`0`, then :math:`T` is the laser repetition rate
            (full cycle of PIE).
            
            The default is 0
        
        start : {'thresh', 'mean', 'max'}
            How to set :math:`t_{0}` of nanotimes
            
            - 'thresh' : use the irf threshold
            - 'mean' : use the mean of the irf distribution
            - 'max' : use the time of the maximum of the irf distribution
            
            The defaul is 'mean'
        
        exclude : bool
            If :code:`True` then exclude all nanotimes before start value, thus
            :math:`t_{start} = 0`.
            if :code:`False` then include photons from entire excitation windown,
            thus :math:`t_{start} < 0`.
            
            The default is True
    
    Parents
    -------
        base : BasePhotonTable
            The table from which to get the bursts, this is the |Pbaseparam| of
            the table.
    
    Columns
    -------
        phasor_g : float (phsel:PhSel irfstyle:{'thresh', 'mean', 'max'})
            The :math:`g` phasor value of the given photon stream. 
            irfstyle key defaults to 'mean'.
        phasor_g : float (phsel:PhSel irfstyle:{'thresh', 'mean', 'max'})
            The :math:`s` phasor value of the given photon stream. 
            irfstyle key defaults to 'mean'.
            
    """
    _irf_style_map = {'thresh':'t', 'mean':'c', 'max':'m'}
    #: :meta private:
    param_defs = (
        ParamDef('omega', TV_ndarray(dtype='f8', dims=arr_slc[:]), unit='rad s^{-1}'),
        ParamDef('start', TV_irfstyle, default='mean'),
        ParamDef('exclude', TV_bool, default=True)
        )
    #: :meta private:
    parent_defs = (ParentDef('base', BasePhotonTable, is_base=True), )
    #: :meta private:
    column_defs = (
        ColumnDef('phasor_g', (PhSel, ), 0, 'user', 
                  iter_func='_iter_phasor_g', title_func='_get_phasor_g_title'), 
        ColumnDef('phasor_s', (PhSel, ), 0, 'user', 
                  iter_func='_iter_phasor_s', title_func='_get_phasor_s_title')
        )

    @cite("DigmanBiophysJ2008", purpose="Phasor analysis")
    def __init_columns__(self):
        pass
    
    @classmethod
    def param_preprocess(cls, param:Sequence[tuple[str,Any]]|tupledict, parents:dict[str:Param])->tuple[dict,dict]:
        """:meta private: preprocess converts period input to angular frequency"""
        param = as_paramdict(param, tuple(pdef.name for pdef in cls.param_defs) + ('period',))
        parents = as_paramdict(parents, tuple(pdef.name for pdef in cls.parent_defs))
        if 'period' in param.keys():
            if 'omega' in param.keys():
                raise ValueError("Cannot specify time constant as both period and omega")
            param['omega'] = 2*np.pi/param.pop('period')
        param.setdefault('omega', 0.0)
        if np.size(param['omega']) == 1:
            param['omega'] = np.repeat(param['omega'], parents['base'].detdef.ex).astype(np.float64)
        return param, parents
    
    @classmethod
    def validate_param(cls, param:Param):
        """:meta private: Validate a Phasor parameter"""
        if param.detdef.ex != param.params['omega'].size:
            raise ValueError(f"Omega must be same size as number of excitations, got {param.params['omega'].size}, expected {param.detdef.ex}")

    @parammethod(origin_as_kw=True)
    def omega_vals(cls, param:Param, phsel:PhSel, origin:PhotonDataS=None)->np.ndarray[np.float64]:
        """
        Parammethod which gets the values of omega (angular frequency) for a given
        |PhSel|, supply the origin when the parameter uses automatic determination
        of values of omega.

        Parameters
        ----------
        param : Param
            Phasor based parameter.
        phsel : PhSel
            Desired stream(s) to retrieve omega values.
        origin : PhotonDataS, optional
            Source of data, needed when paramter uses automatic computation of
            omega values. The default is None.

        Raises
        ------
        ValueError
            Cannot determine omega values because origin not supplyed.

        Returns
        -------
        np.ndarray[np.float64]
            omega values (rad per second) for phasor in each stream id in the
            input phsel.

        """
        if isinstance(origin, PhotonDataList):
            return (cls.omega_vals(param, phsel, origin=data) for data in origin.datas)
        stream_ids = param.detdef.get_stream_ids(phsel)
        omega = np.empty(stream_ids.size)
        for i, sid in enumerate(stream_ids):
            ex = sid // param.detdef.ex_stride
            if param.params['omega'][ex] > 0.0:
                omega[i] = param.params['omega'][ex]
                continue
            if origin is None:
                raise ValueError("Determination of automatic omega requires suppyling origin")
            if param.params['omega'][ex] == 0.0:
                omega[i] = 2*np.pi/origin.setup.tcspc_range
                continue
            tcspc_unit = origin.setup.tcspc_unit[ex]
            ex_range_size = np.diff(origin.setup.ex_ranges[ex][0,:])[0]
            omega[i] = 2*np.pi/ ex_range_size / tcspc_unit
        return omega
    
    @classmethod
    def _get_phasor_title(cls, title:str, col:Column, origin:PhotonData=None):
        """Function creates title for any phasor column"""
        sub = cls._irf_style_map[col.param.params['start']]
        sub = '' if sub == 'mean' else sub
        sub += '' if col.param.params['exclude'] else r'\:full'
        ttl = r'_{%s}%s' % (sub, title)
        title = _title_sels(ttl, origin, col.keytup[0])[0]
        return f'${title}$'

    def _iter_phasor_any(self, phsel:PhSel, func:Callable[[float],float]):
        """Iterator base for phasor, func should be sin or cos function"""
        _validate_lifetime(phsel, self.detdef)
        omegas = self.omega_vals(phsel)
        sels, ex_starts, threshs, trigs = _phasor_trigs(self.origin, phsel, self.param.params['start'], omegas, func)
        prod_func = _phasor_prod_exclude if self.param.params['exclude'] else _phasor_prod_all
        for nhs in zip(*(self.parents['base'].iter_column('ph_nanos', sel) for sel in sels)):
            yield prod_func(trigs, nhs, threshs, ex_starts)

    def _iter_phasor_g(self, phsel:PhSel)->np.ndarray[np.float64]:
        """Iter func for g phasor"""
        yield from self._iter_phasor_any(phsel, np.cos)
        
    @classmethod
    def _get_phasor_g_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """title func for g"""
        return cls._get_phasor_title('g', col, origin)

    def _iter_phasor_s(self, phsel:PhSel)->np.ndarray[np.float64]:
        """iter func for s phasor"""
        yield from self._iter_phasor_any(phsel, np.sin)
        
    @classmethod
    def _get_phasor_s_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """title func for s column"""
        return cls._get_phasor_title('s', col, origin)


def _norm_amp(amps:np.ndarray)->np.ndarray:
    """Extend array by 1, and fill last with 1 - sum(amps)"""
    namps = np.empty(amps.size+1)
    namps[:-1] = amps
    namps[-1] = 1 - amps.sum()
    return namps


def multi_exp(taus:np.ndarray, amps:np.ndarray, t:np.ndarray)->np.ndarray:
    """
    Generate a multi-exponentail decay with lifetimes ``taus``,
    amplitudes ``amps``, evealuated a time points ``t``. ``taus`` and ``amps``
    should be the same size, an integral from 0 to infinity will be evaluate
    to the sum of the values in amps. If the sum is 1, this is the pdf of the
    a multi-exponential.
    
    Evaluates the following
    
    .. math::
        
        f(t) = \displaystyle\sum_{i=1}^{N}{a_{i}\frac{e^{-t/\tau_{i}}}{\tau_{i}}}

    Parameters
    ----------
    taus : np.ndarray
        Lifetimes of exponential decays.
    amps : np.ndarray
        Amplitudes fo exponential decays.
    t : np.ndarray
        Times at which to evaluate the decay.

    Returns
    -------
    np.ndarray
        PDF (if amps sums to 1) of multi-exponential decay evaluated at time
        point t.

    """
    return np.sum(amps*np.exp(-t[...,np.newaxis]/taus) / taus, axis=-1)


def convolve_irf(decay:np.ndarray, irf:np.ndarray)->np.ndarray:
    """
    Convolution function (wraps numpy convolve), which differs in that it returns
    the array that starst at the "full" correlation (edge effects visible), and
    ends at the end of the first input array. This allows a decay to be convolved
    and retruns an identically sized array, and the position of peaks in the
    irf remain unchagned.

    Parameters
    ----------
    decay : np.ndarray
        Decay to convovle with IRF, output matches size of this array.
    irf : np.ndarray
        IRF to convovle with decay, it is best for it to have same size as decay,
        however this is not required.

    Returns
    -------
    np.ndarray
        convolved decay.

    """
    return np.convolve(decay, irf, mode='full')[:decay.size]


def fldecay(taus:np.ndarray, amps:np.ndarray, t:np.ndarray, irf:np.ndarray)->np.ndarray:
    """
    Compute a multi-exponential decay convovled with irf, 
    often used to represent a fluoresence decay.
    
    This simply convolves the multi-expoenential with lifetimes ``taus``,
    and amplitudes ``amps`` with the ``irf``.

    Parameters
    ----------
    taus : np.ndarray
        Lifetimes of exponential decays.
    amps : np.ndarray
        Amplitudes fo exponential decays.
    t : np.ndarray
        Times at which to evaluate the decay.
    irf : np.ndarray
        Instrument response function, assuemed to be measured at timepoints in
        t.

    Returns
    -------
    np.ndarray
        Convovled multi-exponential decay.

    """
    decay = multi_exp(taus, amps, t)
    return convolve_irf(decay, irf)


def fldecay_pmf(taus:np.ndarray, amps:np.ndarray, t:np.ndarray, irf:np.ndarray)->np.ndarray:
    """
    Compute a probability mass function of multi-exponential decay convovled with irf, 
    often used to represent a fluoresence decay.
    
    This simply convolves the multi-expoenential with lifetimes ``taus``,
    and amplitudes ``amps`` with the ``irf``, and then normalizes so sum of values
    is 1.

    Parameters
    ----------
    taus : np.ndarray
        Lifetimes of exponential decays.
    amps : np.ndarray
        Amplitudes fo exponential decays.
    t : np.ndarray
        Times at which to evaluate the decay.
    irf : np.ndarray
        Instrument response function, assuemed to be measured at timepoints in
        t.

    Returns
    -------
    np.ndarray
        Convovled multi-exponential decay.

    """
    decay = fldecay(taus, amps, t, irf)
    return decay / decay.sum()


def fldecay_bg(taus:np.ndarray, amps:np.ndarray, bg:float, t:np.ndarray, irf:np.ndarray, **kwargs)->np.ndarray:
    """
    Compute probability mass function of a fluoresence decay with background
    fraction.
    
    Convovles multi-exponential decay of lifetimes ``taus`` and amplitudes ``amps``
    with IRF, add constant background, and ensure sum of values in output is 1.

    Parameters
    ----------
    taus : np.ndarray
        Lifetimes of exponential decays.
    amps : np.ndarray
        Amplitudes fo exponential decays, should be same size as taus.
    bg : float
        Fraction of Decay arising from flat background.
    t : np.ndarray
        Times at which to evaluate the decay.
    irf : np.ndarray
        Instrument response function, assuemed to be measured at timepoints in
        t.

    Returns
    -------
    np.ndarray
        PMF of background adjusted convovled multi-exponential decay.

    """
    decay = fldecay_pmf(taus, amps, t, irf)
    return (1-bg)*decay + bg/decay.size


def fldecay_bg_norm(taus:np.ndarray, amps:np.ndarray, bg:float, t:np.ndarray, irf:np.ndarray, **kwargs)->np.ndarray:
    """
    Compute probability mass function of a fluoresence decay with background
    fraction.
    
    Convovles multi-exponential decay of lifetimes ``taus`` and amplitudes ``amps``
    with IRF, add constant background, and ensure sum of values in output is 1.

    Parameters
    ----------
    taus : np.ndarray
        Lifetimes of exponential decays.
    amps : np.ndarray
        Amplitudes fo exponential decays, should have 1 fewer elements as taus.
        Evaluates assuming final amplitude is such that sum of amplitudes is 1.
    bg : float
        Fraction of Decay arising from flat background.
    t : np.ndarray
        Times at which to evaluate the decay.
    irf : np.ndarray
        Instrument response function, assuemed to be measured at timepoints in
        t.

    Returns
    -------
    np.ndarray
        PMF of background adjusted convovled multi-exponential decay.

    """
    namps = _norm_amp(amps)
    return fldecay_bg(taus, namps, bg, t, irf)

#: Type hint defintion for functions that can serve to simulate an IRF
IRFFunc = Callable[[np.ndarray,float,...],np.ndarray]


def fldecay_firf_bg(taus:np.ndarray, amps:np.ndarray, bg:float, t:np.ndarray, irf_func:IRFFunc, irf_params:np.ndarray, **kwargs)->np.ndarray:
    """
    Compute probability mass function of a fluoresence decay with background
    fraction, IRF computed based on function and parameters (simulated IRF).
    amps should be same size as taus.
    
    Convovles multi-exponential decay of lifetimes ``taus`` and amplitudes ``amps``
    with IRF, add constant background, and ensure sum of values in output is 1.


    Parameters
    ----------
    taus : np.ndarray
        Lifetimes of exponential decays.
    amps : np.ndarray
        Amplitudes fo exponential decays, should have same size as taus.
    bg : float
        Fraction of Decay arising from flat background.
    t : np.ndarray
        Times at which to evaluate the decay.
    irf_func : IRFFunc
        Function for computing IRF, is called as ``func(t, *irf_params)``.
    irf_params : np.ndarray
        Parameters used to compute IRF.

    Returns
    -------
    np.ndarray
        PMF of background adjusted convovled multi-exponential decay.

    """
    irf = irf_func(t, *irf_params)
    return fldecay_bg(taus, amps, bg, t, irf)


def fldecay_firf_bg_norm(taus:np.ndarray, amps:np.ndarray, bg:float, t:np.ndarray, irf_func:IRFFunc, irf_params:np.ndarray, **kwargs)->np.ndarray:
    """
    Compute probability mass function of a fluoresence decay with background
    fraction.
    Assumes normalized amps, so amps should be of size 1 less than taus
    
    Convovles multi-exponential decay of lifetimes ``taus`` and amplitudes ``amps``
    with IRF, add constant background, and ensure sum of values in output is 1.


    Parameters
    ----------
    taus : np.ndarray
        Lifetimes of exponential decays.
    amps : np.ndarray
        Amplitudes fo exponential decays, should have 1 fewer elements as taus.
        Evaluates assuming final amplitude is such that sum of amplitudes is 1.
    bg : float
        Fraction of Decay arising from flat background.
    t : np.ndarray
        Times at which to evaluate the decay.
    irf_func : IRFFunc
        Function for computing IRF, is called as ``func(t, *irf_params)``.
    irf_params : np.ndarray
        Parameters used to compute IRF.

    Returns
    -------
    np.ndarray
        PMF of background adjusted convovled multi-exponential decay.

    """
    namps = _norm_amp(amps)
    return fldecay_firf_bg(taus, namps, bg, t, irf_func, irf_params)


def mle_fldecay_bg(params:np.ndarray, t:np.ndarray, counts:np.ndarray[np.int64], irf:np.ndarray)->np.ndarray:
    """
    Compute negative of MLE(so can be used with |minimize| function) of 
    photon counts in ``counts`` with ``t`` specifying the time of the TCSPC bins,
    of a multi-exponential decays defined by params, and having an irf of ``irf``.
    
    Params values are alternating tau/amp values, with amps assuming to be normalized,
    so last amp is skipped, then final value specifies background fraction.
    
    use :func:`pack_fldecay_params` to generate this array from values of
    ``taus``, ``amps`` and ``bg``

    Parameters
    ----------
    params : np.ndarray
        Packed array defiing lifetimes, amplitudes and background fraction of decay.
    t : np.ndarray
        Times of bins in counts.
    counts : np.ndarray[np.int64]
        arary of counts of photons in time bins.
    irf : np.ndarray
        IRF of laser pulse.

    Returns
    -------
    float
        Negative of MLE of data

    """
    taus, amps, bg = params[:-1:2], params[1:-2:2], params[-1]
    return -np.sum(np.log(fldecay_bg_norm(taus, amps, bg, t, irf))*counts)


def mle_fldecay_firf_bg(params:np.ndarray, t:np.ndarray, counts:np.ndarray, irf_func:IRFFunc, nparams_irf:int)->float:
    """
    Compute negative of MLE(so can be used with |minimize| function) of 
    photon counts in ``counts`` with ``t`` specifying the time of the TCSPC bins,
    of a multi-exponential decays defined by params, and a simulated irf, 
    computed by irf_func.
    
    Params values are alternating tau/amp values, with amps assuming to be normalized,
    so last amp is skipped, the next value is the bg fraction followed by the
    parameters for the irf.
    
    use :func:`pack_fldecay_params` to generate this array from values of
    ``taus``, ``amps`` and ``bg``

    Parameters
    ----------
    params : np.ndarray
        Packed array defiing lifetimes, amplitudes and background fraction of decay.
    t : np.ndarray
        Times of bins in counts.
    counts : np.ndarray[np.int64]
        arary of counts of photons in time bins.
    irf_func : IRFFunc
        Function generating simulated IRF, called as 
        ``irf_func(t, *params[-nparams_irf:])``.
    nparams_irf : int
        Number of parameters handed to irf_func.

    Returns
    -------
    float
        Negative of MLE of data

    """
    taus, amps, bg = params[:-1-nparams_irf:2], params[1:-2-nparams_irf:2], params[-1-nparams_irf]
    irf_params = params[-nparams_irf:]
    return -np.sum(np.log(fldecay_firf_bg_norm(taus, amps, bg, t, irf_func, irf_params))*counts)


class FLDecParams(TypedDict):
    """
    Return value of :func:`unpack_fldecay_params` and :func:`fit_fldecay`
    """
    taus:np.ndarray[np.float64]
    amps:np.ndarray[np.float64]
    bg:np.float64|float
    irf_func:IRFFunc
    irf_params:np.ndarray[np.float64]
    result:OptimizeResult


def pack_fldecay_params(taus:np.ndarray, amps:np.ndarray, bg:float, irf_params:np.ndarray=None, 
                        **kwargs:Any)->np.ndarray:
    """
    Pack lifetime definitions into params array used by fldecay functions.

    Parameters
    ----------
    taus : np.ndarray
        Lifetimes of multi-exponential decay.
    amps : np.ndarray
        Amplitues of lifetimes of multi-exponential decay.
    bg : float
        Fraction of signal from background.
    irf_params : np.ndarray, optional
        Parameters defining the IRF, if None, assume using measured IRF. The default is None.
    **kwargs : Any
        Additional kwargs passed, this prevents errors when using this function
        as ``pack_fldecay_params(**fit_result)`` where ``fit_result`` is the
        output dictionary of :func:`fit_fldecay`.

    Raises
    ------
    ValueError
        Mismatch between number lifetimes and amplitudes.

    Returns
    -------
    params : np.ndarray
        packed array of parameters defining fluoresence decay.

    """
    taus, amps = np.atleast_1d(taus, amps)
    if taus.size - amps.size not in (0, 1):
        raise ValueError(f"Inconsistent number of decays between taus ({taus.size}) and amps ({amps.size}), amps may be either the same size as taus, or 1 fewer (prefered)")
    amps = amps[:-1] / amps.sum() if amps.size == taus.size else amps
    irf_params = np.array([]) if irf_params is None else np.atleast_1d(irf_params)
    nirf = irf_params.size
    params = np.empty(2*taus.size+nirf)
    params[:-1-nirf:2] = taus
    params[1:-2-nirf:2] = amps
    params[-1-nirf] = bg
    if nirf:
        params[-nirf:] = irf_params
    return params


def pack_fldecay_params_bounds(taus:np.ndarray, amps:np.ndarray, bg:float, irf_params:np.ndarray=None, 
                        tau_bounds:None|np.ndarray=None, amp_bounds:None|np.ndarray=None, 
                        bg_bound:None|float=None, irf_param_bounds:None|np.ndarray=None,
                        **kwargs:Any)->tuple[np.ndarray,np.ndarray]:
    """
    Pack lifetime definitions into params array and bounds array for use
    in optimization

    Parameters
    ----------
    taus : np.ndarray
        Lifetimes of multi-exponential decay.
    amps : np.ndarray
        Amplitues of lifetimes of multi-exponential decay.
    bg : float
        Fraction of signal from background.
    irf_params : np.ndarray, optional
        Parameters defining the IRF, if None, assume using measured IRF. The default is None.
    tau_bounds : None|np.ndarray, optional
        Nx2 bounds array for lifetimes. If specify as single value,
        set bounds from 0 to tau_bounds as maximun in each lifetime.
        The default is None.
    amp_bounds : None|np.ndarray, optional
        Bounds for ampltidues. The default is None.
    bg_bound : None|float, optional
        Maximum fraction of background allowed in optimization. The default is None.
    irf_param_bounds : None|np.ndarray, optional
        Nx2 bounds array for irf params. The default is None.
    **kwargs : Any
        Additional kwargs passed, this prevents errors when using this function
        as ``pack_fldecay_params(**fit_result)`` where ``fit_result`` is the
        output dictionary of :func:`fit_fldecay`.

    Raises
    ------
    ValueError
        Mismatch between number lifetimes and amplitudes.

    Returns
    -------
    params : np.ndarray
        packed array of parameters defining fluoresence decay.
    bounds : np.ndarray
        Nx2 bounds array for MLE minimization of params.

    """
    taus, amps = np.atleast_1d(taus, amps)
    if taus.size - amps.size not in (0, 1):
        raise ValueError(f"Inconsistent number of decays between taus ({taus.size}) and amps ({amps.size}), amps may be either the same size as taus, or 1 fewer (prefered)")
    amps = amps[:-1] / amps.sum() if amps.size == taus.size else amps
    irf_params = np.array([]) if irf_params is None else np.atleast_1d(irf_params)
    nirf = irf_params.size
    params = np.empty(2*taus.size+nirf)
    bounds = np.empty((params.size, 2))
    params[:-1-nirf:2] = taus
    tau_bounds = np.inf if tau_bounds is None else tau_bounds
    if tau_bounds is None or isinstance(tau_bounds, Real):
        bounds[:-1-nirf:2,0] = 0.0
        bounds[:-1-nirf:2,1] = np.inf if tau_bounds is None else tau_bounds
    else:
        bounds[:-1-nirf:2,:] = np.asarray(tau_bounds)
    params[1:-2-nirf:2] = amps
    if amp_bounds is None or isinstance(amp_bounds, Real):
        bounds[1:-2-nirf:2,0] = 0.0
        bounds[1:-2-nirf:2,1] = np.inf if amp_bounds is None else amp_bounds
    else:
        bounds[1:-2-nirf:2,:] = np.asarray(amp_bounds)
    params[-1-nirf] = bg
    bg_bound = np.inf if bg_bound is None else bg_bound
    if isinstance(bg_bound, Real):
        bounds[-1-nirf,0] = 0.0
        bounds[-1-nirf,1] = bg_bound
    else:
        bounds[-1-nirf,:] = bg_bound
    if nirf:
        params[-nirf:] = irf_params
        bounds[-nirf:,:] = np.array([[-np.inf, np.inf]]) if irf_param_bounds is None else irf_param_bounds
    return params, bounds


def _est_bg_tau(nhist:np.ndarray, t:np.ndarray)->tuple[float,float,float]:
    """Compute estimate of location of IRF, background fraction, and mean lifetime"""
    t_center = t[np.argmax(nhist)]
    times = t - t_center
    bg = min(nhist[:10].sum(), nhist[-10:].sum()) / 10
    nhst = nhist - bg
    return t_center, bg / nhst.size, np.sum(nhst*times) / nhst.sum()
    

def unpack_fldecay_params(params:np.ndarray[np.float64], nirf_params:None|int=None)->FLDecParams:
    """
    Unpack params array of fldecay-type funciton into dictionary of lifetimes,
    amplitudes and background fraction ("taus", "amps", "bg" respectively), 
    and optionally the IRF parameters ("irf_params").

    Parameters
    ----------
    params : np.ndarray[np.float64]
        parameters array.
    nirf_params : None|int, optional
        Number of parameters in IRF, if None, assume used experimental IRF. 
        The default is None.

    Returns
    -------
    FLDecParams (dict)
        Dictionary of parameter arrays of decay.
        Contains the following keys:
        
            - "taus" 1d numpy array of lifetimes
            - "amps" 1d numpy array 1 smaller than taus, normalized amplitudes of
              each lifetime.
            - "bg" float background fraction
            - "irf_params" 1d numpy array, parameterd defined by irf

    """
    nirf = int(nirf_params) if nirf_params else 0
    flpdict = dict(taus=params[:-1-nirf:2], amps=params[1:-2-nirf:2], bg=params[-1-nirf])
    if nirf_params:
        flpdict['irf_params'] = params[-nirf:]
    return flpdict


def fit_fldecay(data:PhotonDataS, phsel:PhSel, gate:None|Param|GateGroup=None,
                ndec:int=None, decay_range:None|slice|tuple[int,int]=None, 
                irf:bool|np.ndarray=None, 
                tau_init:None|np.ndarray=None, amp_init:None|np.ndarray=None, bg_init:None|float=None,
                irf_func:IRFFunc=norm.pdf, irf_init:None|np.ndarray=None, 
                nparams_irf:None|int=2, irf_pos_idx:None|int=0, irf_width_idx:None|int=1,
                tau_bounds:None|np.ndarray=None, amp_bounds:None|np.ndarray=None, 
                bg_bound:None|float=None, irf_param_bounds:None|np.ndarray=None,
                auto_tau_bounds:bool=True, **kwargs)->FLDecParams:
    """
    Fit the fluoresence decay extracted from ``data``, using only photons in
    the time ranges in ``param`` and of the stream defined by ``phsel`` to a
    multi-exponential function.
    
    Initial values can be set automatically by leaving respective kwargs blank,
    or specidied manually using kwargs.

    Parameters
    ----------
    data : PhotonDataS
        Data object form which to extract the decay.
    phsel : PhSel
        Photon stream of desired decay.
    gate : None | Param | GateGroup, optonal
        |GateGroup| or |Param| defining time ranges over which to derive
        TCSPC decay. If None, use all photons in data. The default is None.
    ndec : int, optional
        Number of exponential decays, use only if tau_init is not specified. 
        The default is None.
    decay_range : None|slice|tuple[int,int], optional
        Range of TCSPC bins in excitation range over which to fit the dacay. 
        The default is None.
    irf : None|bool|np.ndarray, optional
        If a numpy array, this is used directly as the IRF, if False, then 
        use a simulated IRF (function can be supplied with the `irf_func`` argument),
        if True, the take IRF from ``data.irf[phsel]``, if None, the use IRF in
        ``data.irf[phsel]`` if it exists, otherwise use a simulated IRF.
        The default is None.
    tau_init : None|np.ndarray, optional
        Initial guess for lifetimes. The default is None.
    amp_init : None|np.ndarray, optional
        Initial guess for amplitudes. The default is None.
    bg_init : None|float, optional
        Initial guess for background fraction. The default is None.
    irf_func : IRFFunc, optional
        Only used if irf is None, the function used to evaluate the simulated IRF. 
        The default is norm.pdf.
    irf_init : None|np.ndarray, optional
        Initial guess for IRF. The default is None.
    nparams_irf : None|int, optional
        Number of params used in IRF function (size of irf_init). The default is 2.
    irf_pos_idx : None|int, optional
        Which index in irf_init corresponds to the position of IRF. 
        If not None, this will override this index in irf_init.
        The default is 0.
    irf_width_idx : None|int, optional
        Which index in irf_init corresponds to the position of IRF. 
        If not None, this will override this index in irf_init.
        The default is 1.
    tau_bounds : None|np.ndarray, optional
        Nx2 bounds array for lifetimes. If specify as single value,
        set bounds from 0 to tau_bounds as maximun in each lifetime.
        The default is None.
    amp_bounds : None|np.ndarray, optional
        Bounds for ampltidues. The default is None.
    bg_bound : None|float, optional
        Maximum fraction of background allowed in optimization. The default is None.
    irf_param_bounds : None|np.ndarray, optional
        Nx2 bounds array for irf params. The default is None.
    auto_tau_bounds : bool, optional
        Whether or not to set bounds on taus so that the cannot exceed the
        duration of the excitation period. The default is True.
    **kwargs : Any
        Additional kwargs handed to |minimize|.

    Raises
    ------
    ValueError
        Invalid number of elements in one or more inputs.

    Returns
    -------
    FLDecParams (dict)
        Dictionary of decay parameter arrays.
        Contains the following keys:
        
            - "taus" 1d numpy array of lifetimes
            - "amps" 1d numpy array 1 smaller than taus, normalized amplitudes of
              each lifetime.
            - "bg" float background fraction
            - "irf_params" (conditional) 1d numpy array, parameterd defined by irf 
            - "result" |optimizeresult| of the fitting, the preceeding params
              are all extracted from the ``x`` attribute of this object using
              the :func:`unpack_fldecay_params`
            - "irf_func" (conditional) function used to compute simulated IRF

    """
    phsel = phsel.render_positive(data.detdef, convert_all=True)
    if len(phsel.ex.elements) != 1:
        raise ValueError("can only compute lifetime of single excitation ")
    t, nhist = data.get_tcspc_decay(phsel, gate)
    t_center, bg_est, tau_est = _est_bg_tau(nhist, t)
    dt = t[-1] - t[0]
    if decay_range is not None and not isinstance(decay_range, slice):
        decay_range = decay_range if isinstance(decay_range, (np.ndarray, Sequence)) else (decay_range, )
        decay_range = slice(*decay_range)
        nhist, t = nhist[decay_range], t=[decay_range]
    ##########################
    #### Set init arrayss ####
    ##########################
    tau_init = None if tau_init is None else np.asarray(tau_init)
    amp_init = None if amp_init is None else np.asarray(amp_init)
    if tau_init is None and amp_init is None:
        ndec = 1 if ndec is None else ndec
    elif tau_init is not None:
        ndec = tau_init.size
        if amp_init is not None:
            if tau_init.size - amp_init.size == 0:
                amp_init = amp_init[:-1] / amp_init.sum()
            elif tau_init.size - amp_init.size != 1:
                raise ValueError("Inconsistent number of decays between tau_int " + 
                                 f"({tau_init.size}) and amps ({amp_init.size}), " +
                                 "amps may be either the same size as taus, or 1 fewer (prefered)")
    else:
        ndec = amp_init.size + 1
    if tau_init is None:
        tau_init = np.logspace(np.log(tau_est/1.5), np.log(min(tau_est*1.5, dt*0.5)), ndec, base=np.e) if ndec != 1 else np.array([tau_est])
    if amp_init is None:
        amp_init = np.ones(ndec - 1) / ndec
    bg_init = bg_est if bg_init is None else bg_init
    if irf is None:
        d = data if isinstance(data, PhotonData) else data.datas[0]
        irf = phsel in d.irf
    if irf is False:
        irf_init = np.zeros(nparams_irf) if irf_init is None else np.asarray(irf_init)
        if irf_pos_idx is not None and irf_pos_idx is not False:
            irf_init[irf_pos_idx] = t_center
        if irf_width_idx is not None and irf_width_idx is not False:
            irf_init[irf_width_idx] = tau_est / 10
    else:
        irf_init = np.array([])
        if irf is True:
            irf = data.irf[phsel] if isinstance(data, PhotonData) else data.datas[0].irf[phsel]
    if auto_tau_bounds and tau_bounds is None:
        tau_bounds = dt
    ################################
    ### build inputs to minimize ###
    ################################
    params, bounds = pack_fldecay_params_bounds(
        tau_init, amp_init, bg_init, irf_params=irf_init, tau_bounds=tau_bounds,
        amp_bounds=amp_bounds, bg_bound=bg_bound, irf_param_bounds=irf_param_bounds)
    if any(b is not None for b in (tau_bounds, amp_bounds, bg_bound, irf_param_bounds)):
        kwargs.setdefault('bounds', bounds)
    func = mle_fldecay_firf_bg if irf is False else mle_fldecay_bg
    args = (t, nhist)
    args += (irf_func, irf_init.size) if irf is False else (irf, )
    ###########################################################################
    ########################## Perform Optimization  ##########################
    ###########################################################################
    res = minimize(func, params, args=args, **kwargs)
    out = unpack_fldecay_params(res.x, nirf_params=irf_init.size)
    out['result'] = res
    if irf is False:
        out['irf_func'] = norm.pdf
    return out