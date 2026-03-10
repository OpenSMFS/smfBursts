#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The fretfacory module provides a set of "helper" factory functions, which create
dictionaries of associated :class:`Param` and :class:`Column` objects.
These are designed to be the most commonly used params of a set.

The naming convention of each key is that :class:`Param` objects start with a 
lowercase letter,
while :class:`Column` objects start with a capital letter.

For instance, the function :func:`make_burst_search` returns a dictionary
which contains the keys "nphbg" and "NphDD_bg", the former is the :class:`Param`
of background corrected indensities, while the latter is the column for donor
excitation-donor emission.
"""
import warnings
from typing import Any

from .datamodel.tables import Param, Column, GateGroup

from .ph_sel import PhSel
from .photondata import (
    PhotonDataS, 
    )
from .background import (
    Periods, BG, exp_mlefit
    )
from .bursttables import (
    Bursts, NphBG, Ratios
    )


def make_bg(data:PhotonDataS, period=60.0, tail_min=500e-6, func=exp_mlefit, 
            auto_threshold=False, F_bg=2.0, **kwargs)->dict[str:Param]:
    """
    Create dictionary of stardard background analysis :class:`Param` s.

    Parameters
    ----------
    data : PhotonDataS
        DESCRIPTION.
    period : TYPE, optional
        DESCRIPTION. The default is 60.0.
    tail_min : TYPE, optional
        DESCRIPTION. The default is 500e-6.
    func : TYPE, optional
        DESCRIPTION. The default is exp_mlefit.
    auto_threshold : TYPE, optional
        DESCRIPTION. The default is False.
    F_bg : TYPE, optional
        DESCRIPTION. The default is 2.0.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    [str:Param]
        DESCRIPTION.

    """
    prd = Param(Periods, params=dict(start_at='time_min', stop_at='over', 
                                        period=period, detdef=data.detdef))
    bg_params = dict(compute_stream='single_all', func=func, tail_min=tail_min, auto_threshold=auto_threshold)
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


def make_burst_search(bg:Param=None, m=10, F=6.0, streams='auto', **kwargs)->dict[str:Param|Column]:
    """
    Create a dictionary with standard burst search and background correction
    :class:`Param` s and :class:`Column` s.

    Parameters
    ----------
    bg : Param, optional
        DESCRIPTION. The default is None.
    m : TYPE, optional
        DESCRIPTION. The default is 10.
    F : TYPE, optional
        DESCRIPTION. The default is 6.0.
    streams : TYPE, optional
        DESCRIPTION. The default is 'auto'.
    **kwargs : Any
        Additional kwargs incorporated into Bursts params dict.

    Returns
    -------
    dict[str:Param|Column]
        DESCRIPTION.

    """
    detdef = bg.tp._detdef(bg)
    if streams == 'auto':
        streams = PhSel('all') if detdef.ex == 1 else PhSel('0ex_1ex1em')
    burst_params = {'m':m, 'F':F, 'streams':streams}
    burst_params.update(kwargs)
    bursts = Param(Bursts, burst_params, {'bg':bg})
    nphbg = Param(NphBG, {'single':True}, {'base':bursts, 'bg':bg})
    out = dict(bursts=bursts, nphbg=nphbg)
    out['NphDD_raw'] = Column(bursts, 'nph_raw', PhSel('0ex0em'))
    out['NphDD_bg'] = Column(nphbg, 'nph_bg', PhSel('0ex0em'))
    out['NphDA_raw'] = Column(bursts, 'nph_raw', PhSel('0ex1em'))
    out['NphDA_bg'] = Column(nphbg, 'nph_bg', PhSel('0ex1em'))
    out['NphAll_raw'] = Column(bursts, 'nph_raw', PhSel('all'))
    out['NphAll_bg'] = Column(nphbg, 'nph_bg', PhSel('all'))
    out['E_bg'] = Column(bursts, 'E_raw')
    out['E_bg'] = Column(nphbg, 'E_bg')
    out['BVA10'] = Column(bursts, 'bva', (PhSel('0ex1em'), PhSel('0ex'), 10))
    if detdef.ex == 1:
        out['NphAll_raw'] = Column(bursts, 'nph_raw', PhSel('all'))
        out['NphAll_bg'] = Column(nphbg, 'nph_bg', PhSel('0ex'))
    else:
        out['NphDex_raw'] = Column(bursts, 'nph_raw', PhSel('0ex'))
        out['NphDex_bg'] = Column(nphbg, 'nph_bg', PhSel('0ex'))
        out['NphAA_raw'] = Column(bursts, 'nph_raw', PhSel('1ex1em'))
        out['NphAA_bg'] = Column(nphbg, 'nph_bg', PhSel('1ex1em'))
        out['NphActive_raw'] = Column(bursts, 'nph_raw', PhSel('0ex_1ex1em'))
        out['NphActive_bg'] = Column(nphbg, 'nph_bg', PhSel('0ex_1ex1em'))
        out['S_raw'] = Column(bursts, 'S_raw')
        out['S_bg'] = Column(nphbg, 'S_bg')
    return out


def make_correction_facors(nphbg:Param=None, alpha:float=None, delta:float=None,
                           gamma:float=1.0, beta:float=1.0, lk:float=0.0, 
                           dir_ex:float=0.0, update=None)->dict[str:Param|Column]:
    """
    Create dictionary of standard fully corrected 
    :class:`Param` and :class:`Column` s. Can specify dictionary to add
    these values to inplace through the update kwarg

    Parameters
    ----------
    nphbg : Param, optional
        DESCRIPTION. The default is None.
    alpha : float, optional
        DESCRIPTION. The default is None.
    delta : float, optional
        DESCRIPTION. The default is None.
    gamma : float, optional
        DESCRIPTION. The default is 1.0.
    beta : float, optional
        DESCRIPTION. The default is 1.0.
    lk : float, optional
        DESCRIPTION. The default is 0.0.
    dir_ex : float, optional
        DESCRIPTION. The default is 0.0.
    update : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    dict[str:Param|Column]
        DESCRIPTION.

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


class SequenceDefaults:
    """
    Special defaults object type, each input kwarg is assumed to be a sequence
    of values.
    Input kwargs become attributes, but when accessed, the value is always a copy
    of the original, this way modifications can take place "inplace" when
    using an attribute as a function input etc.
    """
    def __init__(self, **kwargs):
        self._kwargs = kwargs
    
    def __getattr__(self, name):
        if name in self._kwargs:
            return tuple(v.copy() if isinstance(v, (dict, list)) else v for v in self._kwargs[name])
        raise AttributeError(f"no attribute {name}")


blue = '#0055d4'
green = '#2ca02c'
red = '#e74c3c'  # '#E41A1C'
purple = '#9b59b6'

_base_ALEX_kwargs = ({'color':blue},{'color':green},{'color':red},{'color':purple})

#: Default PhSel sequence for ALEX/PIE measurements
ALEXdefaults = SequenceDefaults(
    streams=(PhSel('all'), PhSel('0ex0em'), PhSel('0ex1em'), PhSel('1ex1em')),
    labels=('All', 'DexDem', 'DexAem', 'AexAem'),
    plot_kwargs=_base_ALEX_kwargs, 
    scatter_kwargs=_base_ALEX_kwargs,
    bar_kwargs=_base_ALEX_kwargs
    )


#: Default PhSel sequence for single excitation measurements
MonoExdefaults = SequenceDefaults(
    streams = (PhSel('all'), PhSel('0em'), PhSel('1em')),
    plot_kwargs=_base_ALEX_kwargs[:-1], 
    scatter_kwargs=_base_ALEX_kwargs[:-1],
    bar_kwargs=_base_ALEX_kwargs[:-1]
    )