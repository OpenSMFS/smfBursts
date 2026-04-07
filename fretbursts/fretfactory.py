#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The fretfacory module provides a set of "helper" factory functions, which create
dictionaries of associated |Param| and |Column| objects.
These are designed to be the most commonly used params of a set.

The naming convention of each key is that |Param| objects start with a lowercase 
letter, while |Column| objects start with a capital letter.

For instance, the function :func:`make_burst_search` returns a dictionary
which contains the keys "nphbg" and "NphDD_bg", the former is the |Param|
of background corrected indensities, while the latter is the column for donor
excitation-donor emission.

.. |Param| replace:: :class:`Param <fretbursts.datamodel.tables.Param>`
.. |Column| replace:: :class:`Column <fretbursts.datamodel.tables.Column>`
.. |Coparam| replace:: :attr:`Column.origin_param <fretbursts.datamodel.tables.Column.origin_param>`
.. |Cbparam| replace:: :attr:`Column.base_param <fretbursts.datamodel.tables.Column.base_param>`
"""
from collections.abc import Sequence, Callable, Hashable
import warnings
from typing import Any, Literal
from numbers import Integral

import numpy as np

from .datamodel.utils import SequenceDefaults, _dict_update
from .datamodel.tables import Param, Column, GateGroup, _column_sort

from .ph_sel import PhSel
from .photondata import (
    PhotonDataS, 
    )
from .background import (
    Periods, BG, exp_mlefit, BGFuncType
    )
from .bursttables import (
    Bursts, NphBG, Ratios
    )


def make_bg(data:PhotonDataS, period:float=60.0, tail_min:float=500e-6, func:BGFuncType=exp_mlefit, 
            auto_threshold:bool=False, F_bg:float=2.0, 
            start_at:Literal['time_min','zero','under','over']='time_min', 
            stop_at:Literal['under', 'over']='over', 
            **kwargs)->dict[str:Param|Column]:
    """
    Create dictionary of stardard background analysis |Param| and |Column| s.

    Parameters
    ----------
    data : PhotonDataS
        Data on which to start building a dictionary.
    period : float, optional
        Size of single background period in seconds. The default is 60.0.
    tail_min : float, optional
        Minimum time separation between consecutive photons in seconds to consider
        in computing the background rate. The default is 500e-6.
    func : BGFuncType, optional
        Function used to compute background. The default is exp_mlefit.
    auto_threshold : bool, optional
        Whether to compute ideal threshold from tail_min (True) or use tail_min
        threshold directly (False). The default is False.
    F_bg : bool, optional
        Only used when autothreshold is :code:`True`. The default is 2.0.
    **kwargs : Any
        Additional kwargs included in background |Param| creation params.

    Returns
    -------
    dict[str:Param|Column]
        Dictionary of |Param| and |Column| objects defining background periods.

    """
    prd = Param(Periods, params=dict(start_at='time_min', stop_at='over', 
                                        period=period, detdef=data.detdef))
    bg_params = dict(compute_stream='single_all', func=func, tail_min=tail_min, 
                     auto_threshold=auto_threshold, **kwargs)
    if auto_threshold:
        bg_params['F_bg'] = F_bg
    bg = Param(BG, bg_params, {'base':prd})
    out = {'periods':prd, 'bg':bg}
    out['BgDD'] = Column(bg, 'bg', PhSel('0ex0em'))
    out['BgDA'] = Column(bg, 'bg', PhSel('0ex1em'))
    if data.detdef.ex == 1:
        out['BgAll'] = Column(bg, 'bg', PhSel('all'))
    else:
        out['BgAll'] = Column(bg, 'bg', PhSel('0ex_1ex1em'))
        out['BgAA'] = Column(bg, 'bg', PhSel('1ex1em'))
    return out


def _set_skip(dct:dict, key:str, skip:Sequence[str], make:Callable, *args, **kwargs):
    """set key in dct unless in skip, using make(*arg, **kwargs)"""
    if key not in skip:
        dct[key] = make(*args, **kwargs)
        return dct[key]
    return None


def _infer_bg(param:Param)->None|Param:
    """Attempt to get the BG Param object used by a Param"""
    bg = param.parents.get('bg', None)
    if bg is not None:
        return bg if isinstance(bg, Param) else bg[0]
    for parent in param.parents.values():
        bg = _infer_bg(parent)
        if bg is not None:
            return bg
    return None


def make_fret_from_base(base:Param, bg:Param=None, skip:Sequence[str]=None, 
                        nbva:int=10, update:dict=None, **kwargs)->dict[str:Param|Column]:
    """
    Create default set of |Param| and |Column| objects from an initial BasePhotonTable
    based |Param| .

    Parameters
    ----------
    base : Param
        Initial param defining time ranges on which to generate columns.
    bg : Param, optional
        DESCRIPTION. The default is None.
    skip : Sequence[str], optional
        Keys to skip. The default is None.
    nbva : int, optional
        Default size of bva chunck for bva columns. The default is 10.
    update : dict, optional
        Dictionary to update in place. The default is None.
    **kwargs : Any
        Kwargs handed to param when creating Ratios |Param|.

    Returns
    -------
    dict[str:Param|Column]
        Dictionary of default |Param| and |Column| objects.

    """
    out = dict()
    skip = tuple() if skip is None else skip
    nbva = (nbva, ) if isinstance(nbva, Integral) else nbva
    detdef = base.tp._detdef(base)
    _set_skip(out, 'Dur', skip, Column, base, 'dur')
    _set_skip(out, 'E_raw', skip, Column, base, 'E_raw')
    _set_skip(out, 'NphDD_raw', skip, Column, base, 'nph_raw', (PhSel('0ex0em'), ))
    _set_skip(out, 'NphDA_raw', skip, Column, base, 'nph_raw', (PhSel('0ex1em'), ))
    _set_skip(out, 'NphAll_raw', skip, Column, base, 'nph_raw', (PhSel('all'), ))
    for n in nbva:
        _set_skip(out, f'BVA{n}', skip, Column, base, 'bva', (PhSel('0ex1em'), PhSel('0ex'), n))
    bg = _infer_bg(base) if bg is None else bg
    nphbb, ratios = None, None
    if bg is not None:
        nphbg = _set_skip(out, 'nphbg', skip, Param, NphBG, {'single':True}, {'base':base, 'bg':bg})
    if nphbg is not None and kwargs:
        ratios = _set_skip(out, 'ratios', skip, Param, Ratios, kwargs, {'nph':nphbg})
    if nphbg is not None:
        _set_skip(out, 'E_bg', skip, Column, nphbg, 'E_bg')
        _set_skip(out, 'NphDD_bg', skip, Column, nphbg, 'nph_bg', PhSel('0ex0em'))
        _set_skip(out, 'NphDA_bg', skip, Column, nphbg, 'nph_bg', (PhSel('0ex1em'), ))
        _set_skip(out, 'NphAll_bg', skip, Column, nphbg, 'nph_bg', (PhSel('all'), ))
    if ratios is not None:
        _set_skip(out, 'E', skip, Column, ratios, 'E')
        _set_skip(out, 'NphDD_c', skip, Column, ratios, 'nph_c', (PhSel('0ex0em'), ))
        _set_skip(out, 'NphDA_c', skip, Column, ratios, 'nph_c', (PhSel('0ex1em'), ))
        _set_skip(out, 'NphAll_c', skip, Column, ratios, 'nph_c', (PhSel('all'), ))
    if detdef.ex == 2:
        _set_skip(out, 'S_raw', skip, Column, base, 'S_raw')
        _set_skip(out, 'NphAA_raw', skip, Column, base, 'nph_raw', (PhSel('1ex1em'), ))
        _set_skip(out, 'NphDex_raw', skip, Column, base, 'nph_raw', (PhSel('0ex'), ))
        _set_skip(out, 'NphActive_raw', skip, Column, base, 'nph_raw', (PhSel('0ex_1ex1em'), ))
        if nphbg is not None:
            _set_skip(out, 'S_bg', skip, Column, nphbg, 'S_bg')
            _set_skip(out, 'NphAA_bg', skip, Column, nphbg, 'nph_bg', (PhSel('1ex1em'), ))
            _set_skip(out, 'NphDex_bg', skip, Column, nphbg, 'nph_bg', (PhSel('0ex'), ))
            _set_skip(out, 'NphActive_bg', skip, Column, nphbg, 'nph_bg', (PhSel('0ex_1ex1em'), ))
        if ratios is not None:
            _set_skip(out, 'S', skip, Column, ratios, 'S')
            _set_skip(out, 'NphAA_c', skip, Column, ratios, 'nph_c', (PhSel('1ex1em'), ))
            _set_skip(out, 'NphDex_c', skip, Column, ratios, 'nph_c', (PhSel('0ex'), ))
            _set_skip(out, 'NphActive_c', skip, Column, ratios, 'nph_c', (PhSel('0ex_1ex1em'), ))
    if update is not None:
        update.update(out)
        out = update
    return out


def make_burst_search(bg:Param|Sequence[Param], m:int|np.ndarray[np.int64]=10, 
                      F:float|np.ndarray[np.float64]=6.0, 
                      streams:PhSel|Sequence[PhSel]='auto', skip:Sequence[str]=None,
                      alpha:float=None, delta:float=None, gamma:float=None, beta:float=None, 
                      dir_ex:float=None, lk:float=None, corr_mat:np.ndarray[np.float64]=None,
                      **kwargs)->dict[str:Param|Column]:
    """
    Create a dictionary with standard burst search and background correction
    |Param| s and |Column| s.


    Parameters
    ----------
    bg : Param|Sequence[Param]
        Background |Param| (s) used to compute thresholds for each burst search 
        must be either single |Param| or of same length as streams.
    m : int|np.ndarray[np.int64], optional
        Size of sliding window in sliding window burst search. The default is 10.
    F : float|np.ndarray[np.float64], optional
        Number of times above background for a sliding window to be considered
        in a burst. Must be either single int or same length as streams. 
        The default is 6.0.
    streams : PhSel|Sequence[PhSel], optional
        The photon streams used in individual burst searches that are then gated
        together for final burst definition. If 'auto', use 'all' for
        single excitatoin or '0ex_1ex1em' for 2 excitation data. The default is 'auto'.
    skip : Sequence[str], optional
        List of keys for columns to skip. The default is None.
    alpha : float, optional
        Correction for leakage from donor into acceptor channel. The default is None.
    delta : float, optional
        Correction for direct excitation. The default is None.
    gamma : float, optional
        Gamma correction factor for donor/acceptor emmission sensitivity. The default is None.
    beta : float, optional
        beta correction factor for donor/acceptor excitation sensitivity. The default is None.
    dir_ex : float, optional
        Direct excitation correction, equivalent to delta. The default is None.
    lk : float, optional
        Leakage correction, equivalent to alpha. The default is None.
    corr_mat : np.ndarray[np.float64], optional
        Over-rides alpha, delta, gamma, beta, matrix of correction factors. 
        Should be used if There are channels beyond D/Aex D/Aem.
        The default is None.
    **kwargs : Any
        Additional params in params of Bursts |Param|.

    Returns
    -------
    dict[str:Param|Column]
        Dictionary of |Param| and |Column| objects from burst search.

    """
    detdef = bg.tp._detdef(bg)
    if streams == 'auto':
        streams = PhSel('all') if detdef.ex == 1 else PhSel('0ex_1ex1em')
    burst_params = {'m':m, 'F':F, 'streams':streams}
    burst_params.update(kwargs)
    bursts = Param(Bursts, burst_params, {'bg':bg})
    rkwargs = dict(alpha=alpha, delta=delta, gamma=gamma, beta=beta, 
                   dir_ex=dir_ex, lk=lk, corr_mat=corr_mat)
    rkwargs = {k:v for k, v in rkwargs.items() if v is not None}
    out = make_fret_from_base(bursts, bg=bg, **rkwargs)
    out['bursts'] = bursts
    return out


def make_correction_facors(nphbg:Param=None, alpha:float=None, delta:float=None,
                           gamma:float=1.0, beta:float=1.0, lk:float=0.0, 
                           dir_ex:float=0.0, update:dict=None)->dict[str:Param|Column]:
    """
    Create dictionary of standard fully corrected 
    |Param| and |Column| s. Can specify dictionary to add
    these values to inplace through the update kwarg

    Parameters
    ----------
    nphbg : Param, optional
        DESCRIPTION. The default is None.
    alpha : float, optional
        Leakage correction factor. The default is None.
    delta : float, optional
        Direct excitation correction factor. The default is None.
    gamma : float, optional
        Gamma correction factor. The default is 1.0.
    beta : float, optional
        Beta correction factor. The default is 1.0.
    lk : float, optional
        Leakage correction factor, equivalent to alpha. The default is 0.0.
    dir_ex : float, optional
        Direct excitation factor, equivalent to delta. The default is 0.0.
    update : dict, optional
        If specified, the dictionary will be updated inplace. The default is None.

    Raises
    ------
    TypeError
        Cannot infer NphBg |Param| to base correction factors on.

    Returns
    -------
    dict[str:Param|Column]
        Dictioanry of fully corrected burst |Param| and |Column| objects.

    """
    update = dict() if update is None else update
    if nphbg is None and 'nphbg' not in update:
        raise TypeError("Must specify nphbg either as fist argument or as key in update")
    nphbg = update['nphbg'] if nphbg is None else nphbg
    detdef = nphbg.tp._detdef(nphbg)
    alpha = lk if alpha is None else alpha
    delta = dir_ex if delta is None else delta
    if alpha == 0.0 and delta == 0.0 and gamma == 0.0 and beta == 0.0:
        warnings.warn("Creating Ratios with all trivial correction factors, values identical to NphBG")
    ratios = Param(Ratios, dict(alpha=alpha, delta=delta, gamma=gamma, beta=beta), {'nph':nphbg})
    out = {'ratios':ratios}
    out['NphDD_c'] = Column(ratios, 'nph_c', PhSel('0ex0em'))
    out['NphDA_c'] = Column(ratios, 'nph_c', PhSel('0ex1em'))
    out['NphAll_c'] = Column(ratios, 'nph_c', PhSel('all'))
    out['E'] = Column(ratios, 'E')
    if detdef.ex != 1:
        out['NphDex_c'] = Column(ratios, 'nph_c', PhSel('0ex'))
        out['NphAA_c'] = Column(ratios, 'nph_c', PhSel('1ex1em'))
        out['NphActive_c'] = Column(ratios, 'nph_c', PhSel('0ex_1ex1em'))
        out['S'] = Column(ratios, 'S')
    update.update(out)
    return out


def _regateable(v:Any, gate:GateGroup)->bool:
    """Check if v can be regated to gate"""
    return isinstance(v, (Param, Column)) and gate.origin_param == v.origin_param


def apply_gate(dct:dict[str:Param|Column], gate:GateGroup)->dict[str:Param|Column]:
    """
    Return a copy of the dictionary ``dct`` where all :class:`Param` and :class:`Column`
    values in the dictonary which share the same :attr:`Param.origin_param` /
    :attr:`Column.origin_param` with ``gate`` have been reates to ``gate``.

    Parameters
    ----------
    dct : dict[str:Param|Column]
        Dictionary of :class:`Column` and :class:`Param` objects.
    gate : GateGroup
        Gate to apply to all values in dct (to which the gate can be applied).

    Returns
    -------
    dict
        Regated dictionary.

    """
    return {k:v.regate(gate) if _regateable(v, gate) else v for k, v in dct.items()}


def get_columns(dct:dict[Any], gate:Column|Param|GateGroup,
                key:Callable[[Column],Hashable]=None)->list[Column]:
    """
    Retrieve a list of columns from the values of a dictionary that share the same
    |Coparam| all regated to also share the same |Cbparam|
    The `matches` argument defines the origin_param to match in columns, and the
    base_gate to which all columns will be re-gated
    
    
    This is useful to retrieve from a factory-function created dictionary, all
    columns that can be used to create a 
    `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_ .

    Parameters
    ----------
    dct : dict[Any]
        Dictionary from which to retrieve only columns with matching |Coparam| values.
    gate : GateGroup|Column|Param
        Object to match |Coparam| and define |Cbparam|
        of output columns
    key : None|True|Callable[[Column],Hashable], optional
        Key function to sort Columns by. If None or False will not perform any 
        sorting. If True, will use same sort function as used to sort columns
        in :attr:`Gate.columns <fretbursts.datamodel.tables.Gate.columns>` , 
        finally, if a callable, then it should be 
        usable as argument to `key` in the :code:`sorted` function. 
        The default is None.

    Returns
    -------
    list[Column]
        List of :class:`Column` in dct that can be regated to gate.

    """
    out = [v.regate(gate.base_gate) for v in dct.values() 
           if isinstance(v, Column) and _regateable(v, gate)]
    out = [v for i, v in enumerate(out) if v not in out[:i]] # remove duplicates
    if key is not None and key is not False:
        key = _column_sort if key is True else key
        out = sorted(out, key=key)
    return out

blue = '#0055d4'
green = '#2ca02c'
red = '#e74c3c'  # '#E41A1C'
purple = '#9b59b6'

_base_ALEX_kwargs = ({'color':blue},{'color':green},{'color':red},{'color':purple})
_histbar_kwargs = {'facecolor':'#74a9cf', 'edgecolor':'k', 'alpha':1, 'linewidth':0.15}
_subkdeline_kwargs = {'kdeplot_kwargs':{'color':'k','linewidth':1.0, 'alpha':0.6}}
_kdehistbar_kwargs = _dict_update(_histbar_kwargs, _subkdeline_kwargs)
_ratio_bins = np.linspace(-0.2, 1.2, 71)
_raw_ratio_bins = np.linspace(-0.0, 1.0, 51)

#: Default PhSel sequences for ALEX/PIE measurements
#: Conains the following keys\:
#:
#: - "bursts" - default kwargs for :func:`fretbursts.datamodel.plot.scatter` when plotting bursts
#: - "hexbin" - default kwargs for :func:`fretbursts.datamodel.plot.hexbin` when plotting bursts
#: - "histbar"- default kwargs for :func:`fretbursts.datamodel.plot.hist_bar` when plotting bursts
#: - "ratio_bins"- default bins for histograms of bg/fully corrected ratiometric burst parameters like E and S
#: - "ratio_raw_bins"- default bins for histograms of raw corrected ratiometric burst parameters like E_raw and S_raw
#: - "kdeover"- default kwargs to use with :func:`fretbursts.datamodel.plot.hist_kdeoverlay` when plotting bursts
#: - "streams"- default :class:`fretbursts.ph_sel.PhSel`\s for streams of ALEX parameters
#: - "stream_labels"- default D/A ex/em names for streams, parallels "streams"
#: - "stream_zorder"- default zorder for stacking streams over each other, parallels "streams"
#: - "stream_colors"- default colors for each stream, parallels "streams"
ALEXdefaults = SequenceDefaults(
    bursts = {'s':2.0},
    hexbin={'gridsize':40, 'extent':(-0.2,1.2,-0.2,1.2), 'mincnt':1, 
            'edgecolor':'none', 'linewidth':0.2},
    histbar=_histbar_kwargs, 
    ratio_bins=_ratio_bins, 
    raw_ratio_bins=_raw_ratio_bins,
    kdeover=_kdehistbar_kwargs,
    streams=(PhSel('all'), PhSel('0ex0em'), PhSel('0ex1em'), PhSel('1ex1em')),
    stream_labels=('All', 'DexDem', 'DexAem', 'AexAem'),
    stream_zorder = (0.9, 0.8, 0.7, 0.6),
    stream_colors=_base_ALEX_kwargs, 
    )


#: Default PhSel sequences for single excitation measurements
MonoExdefaults = SequenceDefaults(
    bursts={'s':2.0},
    histbar=_histbar_kwargs, ratio_bins=_ratio_bins, raw_ratio_bins=_raw_ratio_bins,
    kdeover=_kdehistbar_kwargs,
    streams=(PhSel('all'), PhSel('0em'), PhSel('1em')),
    stream_colors=_base_ALEX_kwargs[:-1], 
    )