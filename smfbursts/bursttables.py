#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created : 29/05/202
# Author: Paul David Harris
# email: harrip@gmail.com
r"""
The ``smfbursts.bursttables`` module defines |BasePhotonTable| classes for bursts.

The :class:`Bursts` class defines bursts,
while :class:`BurstOvlp` defines ranges based on logical operations of time ranges.

:class:`Bursts` relies of burst search functions, defined by it's "func" parameter.
These must be registered functions defined using 
the :func:`register_burstsearch_func` function.

Two such functions are already defined\:

#. :func:`burstsearch_mwindowF_bg` for standard sliding-window burst search
#. :func:`burstsearch_mwindowP_bg` for Poisson-based burst threshold (discouraged)

.. |tupledict| replace:: :class:`tupledict <smfbursts.datamodel.utils.tupledict>`
.. |ParamDef| replace:: :class:`Param <smfbursts.datamodel.tables.ParamDef>`
.. |ParentDef| replace:: :class:`Param <smfbursts.datamodel.tables.ParentDef>`
.. |Param| replace:: :class:`Param <smfbursts.datamodel.tables.Param>`
.. |Table| replace:: :class:`Table <smfbursts.datamodel.tables.Table>`
.. |paramproperty| replace:: :class:`smfbursts.datamodel.tables.paramproperty`
.. |tableproperty| replace:: :class:`smfbursts.datamodel.tables.tableproperty`
.. |PhotonTable| replace:: :class:`PhotonTable <smfbursts.photondata.PhotonTable>`
.. |BasePhotonTable| replace:: :class:`BasePhotonTable <smfbursts.photondata.BasePhotonTable>`
.. |DetDef| replace:: :class:`DetDef <smfbursts.ph_sel.DetDef>`
.. |PhSel| replace:: :class:`PhSel <smfbursts.ph_sel.PhSel>`
.. |Periods| replace:: :class:`Periods <smfbursts.background.Periods>`
.. |BG| replace:: :class:`BG <smfbursts.background.BG>`
.. |Yang| replace:: `Zhang, K. & Yang 2005. <https://doi.org/10.1021/jp0546047>`__

"""

from typing import Any, ClassVar
from collections.abc import Iterator, Sequence, Callable, Hashable

from itertools import chain
from numbers import Real

import numpy as np
from scipy.stats import erlang

from .datamodel.utils import tupledict, arr_slc, broadcast_truthtable
from .datamodel.immutabledata import (
    register_PyCode, get_pycode_subval, TV_float, TV_int, TV_ndarray, TV_PyCode
                                      )
from .datamodel.tables import (
    ParamDef, ParentDef, ColumnDef, Param, Column, Table, TableConstructionError, 
    as_paramdict, paramproperty, tableproperty, _TT_ft
    )
from .cite import cite
from .photondata import (
    PhotonData, PhotonTable, BasePhotonTable,
    _regularize_column_startstop, _regularize_ph_sel, 
    _title_sels, _title_startstop_append, _title_unit_append, _pol_ps,
    make_base_column_defs, ColKeyStart, ColKeyStop
    )
from .backgroundtables import Periods, BG
from .ph_sel import PhSel, PhStream, DetDef, TV_PhSel, sort_phsels, phsel_all

import smfbursts.cfuncs as smc


from smfbursts import rcParams


BurstSearchFunc = Callable[[PhotonData,dict[str,Any],dict[str,PhotonTable]],tuple[np.ndarray[np.int64],np.ndarray[np.int64]]]

_mburstsearch = cite('slidingwindowsearch', purpose='sliding window burst search')(smc.burstsearch)
_cpburstsearch = cite('ZhangJPCB2005', purpose='change point burst search')(smc.cpburstsearch)


def _bsvalidatepass(*args, **kwargs):
    """Dummy validate function for Bursts validate registered functions"""
    pass


def register_burstsearch_func(func:BurstSearchFunc,
                              param_defs:Sequence[ParamDef], 
                              parent_defs:Sequence[ParentDef],
                              detdeffunc:Callable[[Param],DetDef],
                              validate:Callable[[Param],None]=None)->None:
    r"""
    Register a new function for performing a burst search to be used as a
    ``func`` param in :class:`Bursts` based |Param|.
    The burst search function must have the signature
    ``burstsearch(origin:PhotonData, params:dict[str:Hashable], parents:dict[str:smf.Param])``
    and return a tuple of numpy int64 arrays, which are the start and stop times
    of bursts.
    
    Must also supply a tuples of |ParamDef| and |ParentDef| needed for burst
    searches using the given function.
    
    Also required is a function that retrieves/computes the |DetDef| of a
    :class:`Bursts` based |Param| that uses the given function.
    
    A validate function may also be supplied, called in the post-init of creating
    a |Param| based on :class:`Bursts` with the given function, to prevent creation
    of |Param|\ s with invalid values.

    Parameters
    ----------
    func : BurstSearchFunc
        Burst search function to register.
    param_defs : Sequence[ParamDef]
        Tuple of |ParamDef| for :class:`Bursts` based |Param| with the given
        function.
    parent_defs : Sequence[ParentDef]
        Tuple of |ParentDef| for :class:`Bursts` based |Param| with the given
        function.
    detdeffunc : Callable[[Param],DetDef]
        Function that takes :class:`Bursts` based |Param|, with the given function
        and returns the |DetDef| of the |Param| (used by the detdef paramproperty).
    validate : Callable[[Param],None], optional
        Function that takes a :class:`Bursts` based |Param| before final createion
        with the given burst search function, and raises an error if the |Param|
        contains invalid values.
        The default is None.

    """
    validate = _bsvalidatepass if validate is None else validate
    param_defs = (param_defs, ) if isinstance(param_defs, ParamDef) else tuple(param_defs)
    parent_defs = (parent_defs, ) if isinstance(parent_defs, ParentDef) else tuple(parent_defs)
    subval = dict(params=param_defs, parents=parent_defs, validate=validate, detdef=detdeffunc)
    register_PyCode(func, subtype="BurstSearchFunc", subval=subval)


def get_burstsearch_params(origin:PhotonData, params:tupledict, parents:tupledict
                           )->tuple[np.ndarray[np.int64],np.ndarray[np.uint8],np.ndarray[np.float64],np.ndarray[np.int64],np.ndarray[np.int64]]:
    r"""
    Convenience function to get arrays for a burst search using
    :func:`smfbursts.cfuncs.burstsearch`.
    
    This function takes the origin dataset, and params and parents tupledicts
    of a :class:`Bursts` |Param| and returns the first 5 expected arrays
    (times, dets, periods, bg, det_ids) that are used by
    :func:`smfbursts.cfuncs.burstsearch` 
    (in that order, which is the order used by the final function).
    
    This function assumes that params contains a 'stream' key which is a |PhSel|
    and a 'bg' parent, which is a |BG| based |Param| to define the relevant
    background and det_ids arrays.
    
    
    .. note::
        
        This function is used by :func:`burstsearch_mwindowF_bg` and
        :func:`burstsearch_mwindowP_bg`


    Parameters
    ----------
    origin : PhotonData
        Origin data of burst search.
    params : tupledict
        params tupledict for the bursts search, must contain 'stream' key, which
        must be a |PhSel|.
    parents : tupledict
        parents tupledict for the burst search, must contain 'bg' key which is
        a |BG| based |Param|.

    Returns
    -------
    times : np.ndarray[np.int64]
        Times array of origin.
    dets : np.ndarray[np.uint8]
        detectors array or origin.
    periods : np.ndarray[np.int64]
        periods delimiters of periods in bg.
    bg : np.ndarray[np.float564]
        background rates for each period of 'bg' key of parents.
    det_ids : np.ndarray[np.uint8]
        det_ids of stream.

    """
    times, dets = origin.times, origin.dets
    bg = origin.get_table(parents['bg'])['bg', params['stream']]
    periods = origin.get_table(parents['bg'].parents['base'])['periods']
    det_ids = origin.detdef.get_stream_ids(params['stream'])
    return times, dets, periods, bg, det_ids


def burstsearch_mwindowF_bg(origin:PhotonData, params:tupledict, parents:tupledict)->tuple[np.ndarray[np.int64],np.ndarray[np.int64]]:
    r"""
    This function is primarily used as the 'func' parameter of :class:`Bursts`
    based |Param|.
    
    Finds windows of :math:`m` or more consecutive photons where the instanteous
    rate is always
    
    .. math::
        
        F \leq \frac{m-1-c}{\Delta_m t_i}
    
    Such parameters will have the following\:
        
        **Params**
        
            - 'stream' (|PhSel|) photon selection on which to perform the burst search.
              The default is :code:`PhSel('all')`.
            - 'm' (int) size of sliding window. The default is 10.
            - 'F' (float) multiple of background for sliding window 
              to be considered to be in a burst. The default is 6.0.
            - 'c' (float) correction factor for count-rate vs time lags. The default is -1.0.
            - 'fuse' (float) separation on which to fuse bursts, ie if burst ranges
              are separeted by less than fuse, output will join them together.
              If 0.0 will only fuse bursts that have overlap, if -1.0 no fuse
              will be performed. The default is 0.0.
        
        
        **Parents**
        
            - 'bg' (|BG|) background definition param
        

    Parameters
    ----------
    origin : PhotonData
        origin data on which to perform the burst search.
    params : tupledict
        |tupledict| of params of |Param|.
    parents : tupledict
        |tupledict| of parents of |Param|.

    Returns
    -------
    starts : np.ndarray[np.int64]
        Start times of bursts.
    
    stops : np.ndarray[np.int64]
        Stop times of bursts.

    """
    times, dets, periods, bg, det_ids = get_burstsearch_params(origin, params, parents)
    return _mburstsearch(times, dets, periods, bg, origin.clk_p, det_ids, 
                         m=params['m'], F=params['F'], c=params['c'], 
                         fuse=params['fuse'], bg_is_thresh=False,
                         alloc_size=rcParams['core.alloc_size'], ncore=rcParams['core.ncore'])


def burstsearch_mwindowP_bg(origin:PhotonData, params:tupledict, parents:tupledict)->tuple[np.ndarray[np.int64],np.ndarray[np.int64]]:
    r"""
    This function is primarily used as the 'func' parameter of :class:`Bursts`
    based |Param|. 
    
    This function defines the minimum rate to be in a burst
    as a probability that the given window is not from Poissonian background.
    Since the distribution of photons is typically pseudo-Poissonian, the use
    of this function is discrourage.
    
    Finds windows of :math:`m` or more consecutive photons where the instanteous
    rate is always
    
    .. math::
        
        Erlang \leq \frac{m-1-c}{\Delta_m t_i}
    
    Such parameters will have the following\:
        
        **Params**
        
            - 'stream' (|PhSel|) photon selection on which to perform the burst search.
              The default is :code:`PhSel('all')`.
            - 'm' (int) size of sliding window. The default is 10.
            - 'P' (float) Probability that photon rate is not from background,
              assuming the background rate is Poissonian. The default is 0.9
            - 'fuse' (float) Maximum separation between bursts to fuse (in seconds), 
              ie if burst ranges are separeted by less than fuse, output will join 
              them together. If 0.0 will only fuse bursts that have overlap, 
              if -1.0 no fuse will be performed. The default is 0.0.
        
        
        **Parents**
        
            - 'bg' (|BG|) background definition param
        

    Parameters
    ----------
    origin : PhotonData
        origin data on which to perform the burst search.
    params : tupledict
        |tupledict| of params of |Param|.
    parents : tupledict
        |tupledict| of parents of |Param|.

    Returns
    -------
    starts : np.ndarray[np.int64]
        Start times of bursts.
    
    stops : np.ndarray[np.int64]
        Stop times of bursts.

    """
    times, dets, periods, bg, det_ids = get_burstsearch_params(origin, params, parents)
    thresh = erlang.ppf(params['P'], params['m'], scale=1.0/bg) / origin.clk_p
    return _mburstsearch(times, dets, periods, thresh, origin.clk_p, det_ids,
                         m=params['m'], bg_is_thresh=True, 
                         alloc_size=rcParams['core.alloc_size'], ncore=rcParams['core.ncore'])


def burstsearch_changepoint_maxrate(origin:PhotonData, params:tupledict, parents:tupledict)->tuple[np.ndarray[np.int64],np.ndarray[np.int64]]:
    r"""
    This function is primarily used as the 'func' parameter of :class:`Bursts`
    based |Param|.
    
    Bursts are defined according to |Yang|, which defines bursts based on 
    
    - the background rate, defined in the parent *bg*
    - the molecular brighness, defined by the max photon rate using a sliding
      window, the size defined by the param *m*
    - :math:`\alpha` (the parameter *alpha*), the likelihood of a false positive
      detection.
    - :math:`\beta` (the parameter *beta*), the likelihood of a false negative
      detection
    
    The algorithm uses a sequential hypothesis test, where progressively larger
    time windows are assessed until the window can be assigned, according to the
    thresholds defined in :math:`\alpha` and :math:`\beta` of probability of 
    incorrect assigment, to either background or burst.
    
    .. note::
        
        This is a "fallback" to the :func:`burstsearch_changepoint_constantsbr`
        As calculating molecular brightness based on max rate is less than ideal.
        With :func:`burstsearch_changepoint_constantsbr`, one can set the signal
        to background ratio based on a second experiment, or using more advanced
        algorithms based on FCS from the data.
    
    Such parameters will have the following\:
        
        **Params**
        
            - 'stream' (|PhSel|) photon selection on which to perform the burst search.
              The default is :code:`PhSel('all')`.
            - 'm' (int) size of sliding window. The default is 30.
            - 'alpha' (float) Probability of false positive (type I) error.
              The default is 0.0001
            - 'beta' (float) Probability of false negative (type II) error.
              The default is 0.01
        
        
        **Parents**
        
            - 'bg' (|BG|) background definition param
    

    Parameters
    ----------
    origin : PhotonData
        origin data on which to perform the burst search.
    params : tupledict
        |tupledict| of params of |Param|.
    parents : tupledict
        |tupledict| of parents of |Param|.

    Returns
    -------
    starts : np.ndarray[np.int64]
        Start times of bursts.
    
    stops : np.ndarray[np.int64]
        Stop times of bursts.

    """
    times, dets, periods, bg, det_ids = get_burstsearch_params(origin, params, parents)
    mrate = origin.get_table(parents['bg'].base_param)['max_rate', params['stream'], params['m']]
    sbr = mrate / bg
    alpha, beta, fuse = params['alpha'], params['beta'], params['fuse']
    return _cpburstsearch(times, dets, periods, bg, sbr, clk_p=origin.clk_p,
                          alpha=alpha, beta=beta, det_ids=det_ids, fuse=fuse,
                          alloc_size=rcParams['core.alloc_size'], ncore=rcParams['core.ncore'])


def burstsearch_changepoint_constantsbr(origin:PhotonData, params:tupledict, parents:tupledict)->tuple[np.ndarray[np.int64],np.ndarray[np.int64]]:
    r"""
    This function is primarily used as the 'func' parameter of :class:`Bursts`
    based |Param|.
    
    Bursts are defined according to |Yang|, which defines bursts based on 
    
    - the background rate, defined in the parent *bg*
    - the molecular brighness, set as the parameter *sbr* multiplied by the parent *bg*
    - :math:`\alpha` (the parameter *alpha*), the likelihood of a false positive
      detection.
    - :math:`\beta` (the parameter *beta*), the likelihood of a false negative
      detection
    
    The algorithm uses a sequential hypothesis test, where progressively larger
    time windows are assessed until the window can be assigned, according to the
    thresholds defined in :math:`\alpha` and :math:`\beta` of probability of 
    incorrect assigment, to either background or burst.
    
    .. note::
        
        20 is a good "default" value for the sbr if it is unknown. However,
        it is more ideal to determing this with FCS or separate experiments.
    
    Such parameters will have the following\:
        
        **Params**
        
            - 'stream' (|PhSel|) photon selection on which to perform the burst search.
              The default is :code:`PhSel('all')`.
            - 'sbr' (int) signal to background ratio. The default is 20.
            - 'alpha' (float) Probability of false positive (type I) error.
              The default is 0.0001
            - 'beta' (float) Probability of false negative (type II) error.
              The default is 0.01
        
        
        **Parents**
        
            - 'bg' (|BG|) background definition param
    

    Parameters
    ----------
    origin : PhotonData
        origin data on which to perform the burst search.
    params : tupledict
        |tupledict| of params of |Param|.
    parents : tupledict
        |tupledict| of parents of |Param|.

    Returns
    -------
    starts : np.ndarray[np.int64]
        Start times of bursts.
    
    stops : np.ndarray[np.int64]
        Stop times of bursts.

    """
    times, dets, periods, bg, det_ids = get_burstsearch_params(origin, params, parents)
    sbr = np.ones(bg.shape, dtype=np.float64)*params['sbr']
    alpha, beta, fuse = params['alpha'], params['beta'], params['fuse']
    return _cpburstsearch(times, dets, periods, bg, sbr, clk_p=origin.clk_p,
                          alpha=alpha, beta=beta, det_ids=det_ids, fuse=fuse,
                          alloc_size=rcParams['core.alloc_size'], ncore=rcParams['core.ncore'])


def _fuse_validate(val:Real, *args, **kwargs)->float:
    """Validate function for fuse params, allows non-negtive values and -1.0"""
    if val == -1.0:
        return float(val)
    if val < 0.0:
        raise ValueError("Fuse must be non-negative or 0")
    return float(val)


def _get_detdef_burstsearch_bg(param:Param)->DetDef:
    """
    Function to get |DetDef| from burstsearch_mwindow[]_bg params, used as 
    detdeffunc argument of register_burstsearchfunc
    """
    return param.parents['bg'].detdef


_burstsearchbg_parents = (ParentDef('bg', BG), )

register_burstsearch_func(burstsearch_mwindowF_bg,
                          (ParamDef('stream', TV_PhSel, default=phsel_all), 
                           ParamDef('m', TV_int(mn=2), default=10), 
                           ParamDef('F', TV_float(mn=1.0), default=6.0), 
                           ParamDef('c', TV_float, default=-1.0),
                           ParamDef('fuse', TV_float(validate=_fuse_validate), default=0.0)
                           ),
                          _burstsearchbg_parents, _get_detdef_burstsearch_bg
                          )
    
register_burstsearch_func(burstsearch_mwindowP_bg,
                          (ParamDef('stream', TV_PhSel, default=phsel_all), 
                           ParamDef('m', TV_int(mn=2), default=10), 
                           ParamDef('P', TV_float(mn=0.0, mx=1.0), default=0.1), 
                           ParamDef('fuse', TV_float(validate=_fuse_validate), default=0.0)
                           ),
                          _burstsearchbg_parents, _get_detdef_burstsearch_bg
                          )

register_burstsearch_func(burstsearch_changepoint_maxrate,
                          (ParamDef('stream', TV_PhSel, default=phsel_all), 
                           ParamDef('m', TV_int(mn=2), default=30), 
                           ParamDef('alpha', TV_float(mn=1e-16, mx=1.0-1e-16), default=1e-4), 
                           ParamDef('beta', TV_float(mn=1e-16, mx=1.0-1e-16), default=1e-2), 
                           ParamDef('fuse', TV_float(mn=0.0), default=0.0)
                           ),
                          _burstsearchbg_parents, _get_detdef_burstsearch_bg
                          )

register_burstsearch_func(burstsearch_changepoint_constantsbr,
                          (ParamDef('stream', TV_PhSel, default=phsel_all), 
                           ParamDef('sbr', TV_float(mn=1.0), default=20.0), 
                           ParamDef('alpha', TV_float(mn=1e-16, mx=1.0-1e-16), default=1e-4), 
                           ParamDef('beta', TV_float(mn=1e-16, mx=1.0-1e-16), default=1e-2), 
                           ParamDef('fuse', TV_float(mn=0.0), default=0.0)
                           ),
                          _burstsearchbg_parents, _get_detdef_burstsearch_bg
                          )


def _append_param_bursts(params:tupledict)->tuple[ParamDef,...]:
    """append_params function for :class:`Bursts` 'func' param"""
    return get_pycode_subval('BurstSearchFunc', params['func'])['params']


def _append_parent_bursts(params:tupledict)->tuple[ParentDef,...]:
    """append_parents function for :class:`Bursts` 'func' param"""
    return get_pycode_subval('BurstSearchFunc', params['func'])['parents']


class Bursts(BasePhotonTable):
    r"""
    |BasePhotonTable| of single burst search, method of burst search defined
    by the 'func' parameter. Additional params and parents are defined by the
    function.
    
    smfBursts comes with 2 registered burst search functions\:
    
    #. :func:`burstsearch_mwindowF_bg` sets threshold as multiple of background 
       (encouraged).
    #. :func:`burstsearch_mwindowP_bg` sets threshodl by Poission probability 
       above background (discouraged.)
    
    Additional fucntions may be defined using the :func:`register_burstsearch_func`
    function.
    
    Params
    ------
        func : Callable[[PhotonData, tupledict, tupledict], tuple[np.ndarray[np.int64], np.ndarray[np.int64]]]
            Function defining the burst search, all additional params and parents
            are defined by this function. Function is called as 
            ``func(origin, param.params, param.parents)``


    When func = :func:`burstsearch_mwindowF_bg` it will have the following params\:
        
        stream : |PhSel|
            Photon selection over which to search for bursts.
            The default is :code:`PhSel('all')`.
        m : int
            Size of the sliding window. The default is 10.
        F : float
            Multiple of background rate to be considered in a burst. The default is 6.0.
        c : float
            Correction factor for computing background rate.
        fuse : float
            Maximum separation between bursts to fuse (in seconds), 
            ie if burst ranges are separeted by less than fuse, output will join 
            them together. If 0.0 will only fuse bursts that have overlap, 
            if -1.0 no fuse will be performed. The default is 0.0.


    When func = :func:`burstsearch_mwindowP_bg` (discouraged) it will have the 
    following params\:
        
        stream : |PhSel|
            Photon selection over which to search for bursts.
            The default is :code:`PhSel('all')`.
        m : int
            Size of the sliding window. The default is 10.
        P : float
            Probability that photon rate is not from background, assuming the 
            background rate is Poissonian. The default is 0.9
        fuse : float
            Maximum separation between bursts to fuse (in seconds), 
            ie if burst ranges are separeted by less than fuse, output will join 
            them together. If 0.0 will only fuse bursts that have overlap, 
            if -1.0 no fuse will be performed. The default is 0.0.

    
    Note that the 'stream', 'm' and 'fuse' params are common to both.
    
    Parents
    -------
    All parents determiend by the 'func' param. For both 
    :func:`burstsearch_mwindowF_bg` and :func:`burstsearch_mwindowP_bg` there
    is 1 parent\:
        
        bg : |BG|
            |BG| based |Param| defining background count rate
    
    
    Columns
    -------
    Uses |BasePhotonTable| columns.
    See :any:`basephotoncolumns` for full list of columns.
    
    """
    #: :meta private:
    row_name:ClassVar[str] = "Bursts"
    _origin: PhotonData
    
    #: :meta private:
    param_defs = (
        ParamDef('func', TV_PyCode(subval='BurstSearchFunc'), default=burstsearch_mwindowF_bg,
                 append_params=_append_param_bursts, append_parents=_append_parent_bursts),
        )
    parent_defs = tuple() #: :meta private:
    column_defs = make_base_column_defs() #: :meta private:

    @cite('NirJPCB2006', purpose='Dual Channel Burst Search')
    def __init_columns__(self):
        starts, stops = self._compute_startstop
        istarts, istops = smc.index_ranges(self.origin.times, starts, stops)
        self._add_column('start', tuple(), starts)
        self._add_column('stop', tuple(), stops)
        self._add_column('istart', tuple(), istarts)
        self._add_column('istop', tuple(), istops)

    @classmethod
    def param_preprocess(cls, params:tupledict|dict, parents:tupledict|dict):
        """Param preprocess function, resorts params and parents if specified
        as kwargs based on func :meta private:"""
        params = params.asdict if isinstance(params, tupledict) else params
        parents = parents.asdict if isinstance(parents, tupledict) else parents
        if 'func' not in params:
            params = params.asdict if isinstance(params, tupledict) else params
            for pdef in cls.param_defs:
                if pdef.name == 'func':
                    if 'default' not in pdef:
                        raise ValueError(f"{cls.__name__} has not default burst search function set, must specify func")
                    params['func'] = pdef.default
                    break
            else:
                raise TableConstructionError("No param_def specifying ")
        parent_defs = get_pycode_subval('BurstSearchFunc', params['func'])['parents']
        for pdef in parent_defs:
            if pdef.name in params:
                if pdef.name in parents:
                    raise ValueError(f"parent {pdef.name} specified twice")
                parents[pdef.name] = params.pop(pdef.name)
        return params, parents

    @classmethod
    def validate_param(cls, param:Param)->None:
        """Validator, checks params based on 'func' validator :meta private:"""
        get_pycode_subval('BurstSearchFunc', param.params['func'])['validate'](param)

    @tableproperty
    def _compute_startstop(cls, param:Param, origin:PhotonData)->tuple[np.ndarray[np.int64],np.ndarray[np.int64]]:
        """
        |tableproperty| that performs burst search and returns starts/stops.
        
        
        Call from table::
            
            starts, stops = table._compute_startstop
        
        
        Call from |Param|::
            
            starts, stops = param._compute_startstop(data)
        
        
        .. note::
            
            Calling from |Param| will **not** result in the creation of a table,
            this allows tables like :class:`BurstOvlp` with parents of this
            type of table to get starts/stops without having to generate and
            cache the table (saves memeory).

        Parameters
        ----------
        param : Param
            |Param| defining burst search parameters.
        origin : PhotonData
            data.

        Returns
        -------
        starts : np.ndarray[np.int64]
            Start times of bursts.
        stops : np.ndarray[np.int64]
            Stop times of bursts.

        """
        paramdict = param.params.asdict
        func = paramdict.pop('func')
        return func(origin, paramdict, param.parents)

    @paramproperty
    def detdef(cls, param:Param)->DetDef:
        """|DetDef| of |Param|. This is a |paramproperty|"""
        return get_pycode_subval('BurstSearchFunc', param.params['func'])['detdef'](param)


def _size_burstovlp_bases(params:tupledict)->int:
    """Size fucntion for bases parent of BurstOvlp"""
    return params['truthtable'].ndim


def _func_key(func:Callable)->tuple[str,str]:
    """Key function for ordering functions alphabetically by name"""
    return func.__name__, func.__module__


def _iter_baseparents(param:Param)->Iterator[Param]:
    """Iterate over all origin_params defined in param"""
    param = param.base_param
    yield param
    for prm in param.parents.values():
        if isinstance(prm, Param):
            yield prm.base_param
            yield from _iter_baseparents(prm)
        else:
            for p in prm:
                yield p
                yield from _iter_baseparents(p)


def _get_prd_minmax(origin, param)->tuple[int,int]:
    """Get min and max time of period Param based on origin data"""
    periods = origin.get_table(param)['periods']
    return periods[0], periods[-1]


def _find_period_minmax(origin:PhotonData, param:Param)->tuple[int, int]:
    """From a given Param, find the min and max time of any |Periods| parents"""
    mns, mxs = zip(*(_get_prd_minmax(origin, prm) for prm in _iter_baseparents(param) if prm.tp == Periods))
    mn = min(mns) if len(mns) != 0 else origin.times[0]
    mx = max(mxs) if len(mxs) != 0 else origin.times[-1]+1
    return mn, mx


def _expand_tt_bursts(base:Param)->tuple[tuple[Param,...],np.ndarray[np.bool_]]:
    """Expand function for converting base photon table to tuple of bases and truthtable"""
    if issubclass(base.tp, BurstOvlp):
        return base.parents['bases'], base.params['truthtable']
    return (base, ), _TT_ft


class BurstOvlp(BasePhotonTable):
    r"""
    Burst Gate/overlap. Find the logical overlap between multiple definitions
    of time ranges. Uses :func:`smfbursts.cfuncs.burstgate` to perform the
    gating of start/stop times of the underlying bases.
    
    This is useful for defining a dual-channel-burst-search
    (see `Nir 2006 <https://doi.org/10.1021/jp063483n>`_).
    But extends the "and" gate to any arbitrary number and logical operation
    using a truthtable.
    
    Params
    ------
        fuse : float
            Maximum separation between bursts to fuse (in seconds), 
            ie if burst ranges are separeted by less than fuse, output will join 
            them together. If 0.0 will only fuse bursts that have overlap, 
            if -1.0 no fuse will be performed. 
            This is performed after :func:`smfbursts.cfuncs.burstgate` has been
            called.
            The default is 0.0.
        truthtable : np.ndarray[np.bool\_] | {'and', 'or', 'invand', 'invor', 'single', 'inv'}
            Boolean array, with number of dimensions as number of |Param| in 
            parent 'bases'. All dimensions size 2. 
            Certain strings can be passed during creation, which will be converted
            to the appropriate truthtable, matching the size bases.
            These are\:
                
                - 'and' an "all-and" gate, the only true value is where all 
                  bases are "in-burst", ie all bases must be in burst for range
                  to be in output
                - 'or' an "all-or" gate, the only false value is where all bases
                  are not "in-burst", ie a time range is in the output if it is
                  in any bases.
                - 'invand' the exact inverse of 'and', all times are in range
                  unless it is in all input ranges
                - 'invor' the exact inverse or 'or', only when a time is not in
                  any of the bases is the time range in the output
                - 'single' only for 1-base instances, reflects the input.
                  This should only be used with fuse!=0.0, as it will only fuse
                  bursts.
                - 'inv' only for 1-base instances, invert the base.
            
            The default is 'all'
            
            
    Parents
    -------
        bases : tuple[Param[BasePhotonTable],...]
            Usually these are :class:`Bursts` based |Param|, but the can be any
            |BasePhotonTable|, specified as a sequence, with the same length
            as the number of dimensions in the param 'truthtable'
        
        
    Columns
    -------
    Uses |BasePhotonTable| columns.
    See :any:`basephotoncolumns` for full list of columns.
    
    """
    #: :meta private:
    row_name:ClassVar[str] = "Bursts"
    _origin: PhotonData

    #: :meta private:
    param_defs = (
        ParamDef('fuse', TV_float(mn=0.0), default=0.0),
        ParamDef('truthtable', TV_ndarray(dtype=np.dtype('|b1'), square=True, dims=arr_slc[2,...])),
        )
    #: :meta private:
    parent_defs = (
        ParentDef('bases', BasePhotonTable, size_func=_size_burstovlp_bases),
        )
    #: :meta private:
    column_defs = make_base_column_defs()

    def __init_columns__(self):
        starts, stops = list(), list()
        for base in self.param.parents['bases']:
            if not self.origin.has_table(base) and hasattr(base, '_compute_startstop'):
                start_temp, stop_temp = base._compute_startstop(self.origin)
                starts.append(start_temp)
                stops.append(stop_temp)
        starttime = min(chain([self.origin.times[0]], *(start[:1] for start in starts)))
        stoptime = max(chain([self.origin.times[0]], *(stop[-1:] for stop in stops)))
        starts, stops = smc.burstgate(starts, stops, self.param.params['truthtable'], 
                                      starttime=starttime, stoptime=stoptime, 
                                      alloc_size=rcParams['core.alloc_size'])
        if self.param.params['fuse'] != 0.0:
            starts, stops = smc.fusebursts(starts, stops, int(self.param.params['fuse']/self.origin.clk_p))
        istarts, istops = smc.index_ranges(self.origin.times, starts, stops)
        self._add_column('start', tuple(), starts)
        self._add_column('stop', tuple(), stops)
        self._add_column('istart', tuple(), istarts)
        self._add_column('istop', tuple(), istops)
    
    @paramproperty
    def detdef(cls, param:Param)->DetDef:
        """|DetDef| of the |PhotonTable|/|Param|. This is a |paramproperty|."""
        return param.parents['bases'][0].detdef
    
    @classmethod
    def param_preprocess(cls, params, parents)->tuple[dict[str:Hashable],dict[str:Param]]:
        """
        Preprocess function, casts truthtable from string, simplifies truthtable
        if duplicate parents exist, and normalizes order of parents
        :meta private:
        """
        params = as_paramdict(params, tuple(pdef.name for pdef in cls.param_defs))
        parents = as_paramdict(parents, tuple(pdef.name for pdef in cls.parent_defs))
        bases = (parents['bases'], ) if isinstance(parents['bases'], Param) else parents['bases']
        truthtable = params.get('truthtable', 'single' if len(bases) ==1 else 'all')
        if isinstance(params['truthtable'], str):
            # check valid truthtable string
            if truthtable not in ('and', 'or', 'invand', 'invor', 'single', 'inv'):
                raise ValueError("Unrecognized truthtable string %s"%params['truthtable'])
            size = len(bases)
            if truthtable in ('single', 'inv'):
                if size != 1:
                    raise ValueError("Single burst searches must be 'single', or 'inv', not %s"%params['truthtable'])
            elif size == 1:
                raise ValueError("Cannot specify turthtable as %s with multiple ranges"%truthtable)
            inv = 'inv' in truthtable
            # created arrays for non-inverted values
            if 'or' in truthtable:
                truthtable = np.ones(tuple(2 for _ in range(size)), dtype=np.bool_)
                truthtable[tuple(0 for _ in range(size))] = False
            else:
                truthtable = np.zeros(tuple(2 for _ in range(size)), dtype=np.bool_)
                truthtable[tuple(1 for _ in range(size))] = True
            # invert if needed
            if inv:
                truthtable = ~truthtable
        # broadcast any BurstOvlp 
        truthtable, bases = broadcast_truthtable(truthtable, bases, 
                                                 _expand_tt_bursts,
                                                 sort_key=hash, dtype=np.bool_)
        parents['bases'] = bases
        params['truthtable'] = truthtable
        return params, parents
    