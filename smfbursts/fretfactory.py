#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
The fretfacory module provides a set of "helper" factory functions, which create
dictionaries of associated |Param| and |Column| objects.
These are designed to be the most commonly used params of a set.

The naming convention of each key is that |Param| objects start with a lowercase 
letter, while |Column| objects start with a capital letter.

For instance, the function :func:`make_burst_search` returns a dictionary
which contains the keys "nphbg" and "NphDD_bg", the former is the |Param|
of background corrected indensities, while the latter is the column for donor
excitation-donor emission.


.. |Param| replace:: :class:`Param <smfbursts.datamodel.tables.Param>`
.. |Column| replace:: :class:`Column <smfbursts.datamodel.tables.Column>`
.. |Coparam| replace:: :attr:`Column.origin_param <smfbursts.datamodel.tables.Column.origin_param>`
.. |Cbparam| replace:: :attr:`Column.base_param <smfbursts.datamodel.tables.Column.base_param>`
.. |BasePhotonTable| replace:: :class:`photondata.BasePhotonTable <smfbursts.photondata.BasePhotonTable>`
.. |Periods| replace:: :class:`Periods <smfbursts.background.Periods>`
.. |BG| replace:: :class:`BG <smfbursts.background.BG>`
.. |Bursts| replace:: :class:`Bursts <smfbursts.bursttables.Bursts>`
.. |BurstOvlp| replace:: :class:`BurstOvlp <smfbursts.bursttables.BurstOvlp>`
.. |NphBG| replace:: `NphBG <smfbursts.bursttables.NphBG>`
.. |Ratios| replace:: :class:`Ratios <smfbursts.bursttables.Ratios>`
.. |DetDef| replace:: :class:`DetDef <smfbursts.ph_sel.DetDef>`
.. |leastsquare| replace:: `scipy.optimize.least_squares <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>`__
.. |minimize| replace:: `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`__
.. |optimizeresult| replace:: `OptimizeResult <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html>`__
.. |linregress| replace:: `scipy.stats.linregress <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html>`__
.. |hellenkamp| replace:: `*Hellenkamp et. al.* <https://doi.org/10.1038/s41592-018-0085-0>`__

"""
from collections.abc import Sequence, Callable, Hashable, Iterator
import warnings
from typing import Any, Literal
from numbers import Integral
from itertools import repeat

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.stats import linregress
from scipy.stats._stats_py import LinregressResult

from .datamodel.utils import SequenceDefaults, _dict_update
from .datamodel.tables import Param, Column, GateGroup, _column_sort
from .datamodel.multifit import lsq_anyfit, MinFunc
from .cite import cite

from .ph_sel import PhSel, PhStream, DetDef
from .photondata import (
    PhotonDataS, PhotonData
    )
from .backgroundtables import (
    Periods, BG, exp_mlefit, BGFuncType
    )
from .bursttables import Bursts, BurstOvlp
from .childphotontables import NphBG, Ratios


def _as_phsel(val:str|PhStream|PhSel)->PhSel:
    """Convert val to |PhSel| if possible"""
    if isinstance(val, (PhStream, str)):
        return PhSel(val)
    return val


def _match_size(val:Any, size:int|None, name:str)->tuple[Sequence|Iterator,int|None]:
    """Make val match the size size, if size is None, assume size 1, and return expected size"""
    if isinstance(val, (Sequence,np.ndarray)) and not isinstance(val,str):
        if size is not None and size != len(val):
            raise ValueError(f"Size of {name} does not match the expected size ({size})")
        if len(val) == 1:
            return repeat(val[0]), size
        return val, len(val)
    return repeat(val), size


def make_bg(detdef:DetDef|PhotonDataS, period:float=60.0, tail_min:float=500e-6, func:BGFuncType=exp_mlefit, 
            auto_threshold:bool=False, F_bg:float=2.0, 
            start_at:Literal['time_min','zero','under','over']='time_min', 
            stop_at:Literal['under', 'over']='over', 
            **kwargs)->dict[str:Param|Column]:
    r"""
    Create dictionary of stardard background analysis |Param| and |Column| s.
    
    Keys in output dictionary\:
    
    - 'periods' |Param| of type |Periods|
    - 'bg' |Param| of type |BG|
    - 'BgDD' |Column| of background rate in DexDem (0ex0em) channel
    - 'BgDA' |Column| of background rate in DexAem (0ex1em) channel
    - 'BgAA' |Column| of background rate in AexAem (1ex1em) channel
    - 'BgAll' |Column| of background rate in Dex + AexAem (0ex_1ex1em) channels

    Parameters
    ----------
    detdef : DetDef | PhotonDataS
        |DetDef| of, or Data on which to start building a dictionary. The function
        tests if the detdef is a |DetDef|, if it is not, will access the 
        ``detdef.detdef`` property to get the |DetDef| of the data.
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
    detdef = detdef if isinstance(detdef, DetDef) else detdef.detdef
    prd = Param(Periods, params=dict(start_at='time_min', stop_at='over',
                                        period=period, detdef=detdef))
    bg_params = dict(compute_stream='single_all', func=func, tail_min=tail_min,
                     auto_threshold=auto_threshold, **kwargs)
    if auto_threshold:
        bg_params['F_bg'] = F_bg
    bg = Param(BG, bg_params, {'base':prd})
    out = {'periods':prd, 'bg':bg}
    out['BgDD'] = Column(bg, 'bg', PhSel('0ex0em'))
    out['BgDA'] = Column(bg, 'bg', PhSel('0ex1em'))
    if detdef.ex == 1:
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
        parents = (parent, ) if isinstance(parent, Param) else parent
        for p in parents:
            bg = _infer_bg(p)
            if bg is not None:
                return bg
    return None


def make_fret_from_base(base:Param, bg:Param=None, skip:Sequence[str]=None, 
                        nbva:int|Sequence[int]=5, update:dict=None, **kwargs)->dict[str:Param|Column]:
    r"""
    Create default set of |Param| and |Column| objects from an initial BasePhotonTable
    based |Param|.
    
    .. _fretfackeys:

    Standard names for keys in fret dictionary\:

    - 'nphbg' |Param| of type |NphBG|, with parent "base" set by ``base`` param, 
      and parent "bg" set by ``bg`` param (if not specified, this key is skipped)
    - 'ratios' |Param| of type |Ratios| with 
    - 'Dur' |Column| of duration of bursts 
      (uses default start/stop types of input argument ``base``)
    - 'NphDD_raw' |Column| of raw photon counts in DexDem (0ex0em) channel
      based on input argument ``base``
    - 'NphDD_bg' |Column| of background corrected photon counts in DexDem (0ex0em) 
      channel, based on 'nphbg' key
      (uses default start/stop types of input argument ``base``)
    - 'NphDD_c' |Column| of bg and correction factor corrected photon counts in 
      DexDem (0ex0em) channel, based on 'ratios' key
      (uses default start/stop types of input argument ``base``)
    - 'NphDA_raw' |Column| of raw photon counts in DexAem (0ex1em) channel
      based on input argument ``base``
    - 'NphDA_bg' |Column| of background corrected photon counts in DexAem (0ex1em) 
      channel, based on 'nphbg' key
      (uses default start/stop types of input argument ``base``)
    - 'NphDA_c' |Column| of bg and correction factor corrected photon counts in 
      DexAem (0ex1em) channel, based on 'ratios' key
      (uses default start/stop types of input argument ``base``)
    - 'NphAA_raw' |Column| of raw photon counts in AexAem (1ex1em) channel
      based on input argument ``base``
    - 'NphAA_bg' |Column| of background corrected photon counts in AexAem (1ex1em) 
      channel, based on 'nphbg' key
      (uses default start/stop types of input argument ``base``)
    - 'NphAA_c' |Column| of bg and correction factor corrected photon counts in 
      AexAem (1ex1em) channel, based on 'ratios' key
      (uses default start/stop types of input argument ``base``)
    - 'NphDex_raw' |Column| of raw photon counts in Dex (0ex) channels (donor excitation)
      based on input argument ``base``
    - 'NphDex_bg' |Column| of background corrected photon counts in Dex (0ex) 
      channels (donor excitation), based on 'nphbg' key
      (uses default start/stop types of input argument ``base``)
    - 'NphDex_c' |Column| of bg and correction factor corrected photon counts in 
      Dex (0ex) channels (donor excitation), based on 'nphbg' key
      (uses default start/stop types of input argument ``base``)
    - 'NphActive_raw' |Column| of raw photon counts in all active channels 
      (Dex + AexAem, ie 0ex_1ex1em), not included in single excitation experiments
      based on input argument ``base``
    - 'NphActive_bg' |Column| of background corrected photon counts in all active channels
      based on 'nphbg' key
      (uses default start/stop types of input argument ``base``)
    - 'NphActive_c' |Column| of bg and correction factor corrected  photon counts 
      in all active channels, based on 'ratios' key
      (uses default start/stop types of input argument ``base``)
    - 'NphAll_raw' |Column| of raw photon counts in all channels (included non-active)
      based on input argument ``base``
    - 'NphAll_bg' |Column| of background corrected photon counts in all channels 
      (included non-active) based on the 'nphbg' key
      (uses default start/stop types of input argument ``base``)
    - 'NphAll_c' |Column| of bg and correction factor corrected photon counts 
      in all channels (included non-active), based on the 'ratios' key
      (uses default start/stop types of input argument ``base``)
    - 'E_raw' |Column| of ratio of raw photon counts between DexAem and Dex
      (0ex1em and 0ex), based on input argument ``base``
    - 'E_bg' |Column| of ratio of background corrected photon counts between 
      DexAem and Dex (0ex1em and 0ex), based on the 'nphbg' key
      (uses default start/stop types of input argument ``base``)
    - 'E' |Column| of ratio of bg and correction factor corrected photon counts 
      between DexAem and Dex (0ex1em and 0ex), based on the 'ratios' key
      (uses default start/stop types of input argument ``base``)
    - 'S_raw' |Column| of ratio of raw photon counts between Dex and Dex_AexAem
      (0ex and 0ex_1ex1em), based on input argument ``base``
    - 'S_bg' |Column| of ratio of background corrected photon counts between 
      Dex and Dex_AexAem (0ex and 0ex_1ex1em), based on the 'nphbg' key
      (uses default start/stop types of input argument ``base``)
    - 'S' |Column| of ratio of bg and correction factor corrected photon counts 
      between Dex and Dex_AexAem (0ex and 0ex_1ex1em), based on the 'ratios' key
      (uses default start/stop types of input argument ``base``)
    
    
    Parameters
    ----------
    base : Param
        Initial param defining time ranges on which to generate columns.
        Must be a |BasePhotonTable| based |Param|, it is typically either
        a |Bursts| or |BurstOvlp| based |Param|.
    bg : Param, optional
        A |BG| based |Param|, the background definition for the output |Param|
        and |Column| objects requiring background correction. If None, then the
        function will search the parents of base for a |BG| based |Param|, and
        the first one found will be used as the background correction parameter. 
        The default is None.
    skip : Sequence[str], optional
        Keys to skip, use to avoid creating |Column|/|Param| that are invalid
        for the given dataset. The default is None.
    nbva : int | Sequence[int], optional
        Default size of bva chunck for bva columns. The default is 5.
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
    detdef = base.detdef
    _set_skip(out, 'Dur', skip, Column, base, 'dur')
    _set_skip(out, 'E_raw', skip, Column, base, 'E_raw')
    _set_skip(out, 'NphDD_raw', skip, Column, base, 'nph_raw', (PhSel('0ex0em'), ))
    _set_skip(out, 'NphDA_raw', skip, Column, base, 'nph_raw', (PhSel('0ex1em'), ))
    _set_skip(out, 'NphAll_raw', skip, Column, base, 'nph_raw', (PhSel('all'), ))
    for n in nbva:
        _set_skip(out, f'BVA{n}', skip, Column, base, 'bva', (PhSel('0ex1em'), PhSel('0ex'), n))
    bg = _infer_bg(base) if bg is None else bg
    nphbg, ratios = None, None
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


def make_burst_search(bg:Param|Sequence[Param], m:int|Sequence[int]=10, 
                      F:float|Sequence[float]=6.0, fuse:float=0.0,
                      streams:PhSel|Sequence[PhSel]='auto', 
                      truthtable:np.ndarray[np.bool_]|Literal['auto','and','or','invand','invor','single','inv']='auto',
                      c:float|Sequence[np.float64]=-1.0, 
                      skip:Sequence[str]=None,
                      alpha:float=None, delta:float=None, gamma:float=None, beta:float=None, 
                      dir_ex:float=None, lk:float=None, corr_mat:np.ndarray[np.float64]=None,
                      nbva:int|Sequence[int]=None, update:dict=None,
                      **kwargs)->dict[str:Param|Column]:
    r"""
    Create a dictionary with standard burst search and background correction
    |Param| s and |Column| s.
    
    Key of base param\:
    
    - 'bursts'
    
    The remaining keys in the dictionary are made by :func:`make_fret_from_base`
    and therefore the keys are those of fretfackeys_.


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
    fuse: float
        The maximum separation between windows (in seconds) to fuse bursts.
        The default is 0.0.
    streams : PhSel|Sequence[PhSel], optional
        The photon streams used in individual burst searches that are then gated
        together for final burst definition. If 'auto', use 'all' for
        single excitatoin or '0ex_1ex1em' for 2 excitation data. The default is 'auto'.
    truthtable : np.ndarray[np.bool\_] | {'auto', 'and', 'or', 'invand', 'invor', 'single', 'inv'}
        Truthtable defining logical operation combining burst searches
    c : float | np.ndarray[np.float64]
        correction factor for photon rate of sliding window.
        :math:`\tau^{m}_{i} = \frac{m - 1 - c}{\Delta^{m}t_{i}}`.
        **Note that this factor is rarely changed** from -1.0, ie the rate is
        simply :math:`\frac{m}{\Delta^{m}t_{i}}`.
        The default is -1.0
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
    nbva : int | Sequence[int], optional
        Default size of bva chunck for bva columns. The default is 5.
    update : dict
        Dictionary to update (inplace) with new values. The default is None.
    **kwargs : Any
        Additional params in params of Bursts |Param|.

    Returns
    -------
    dict[str:Param|Column]
        Dictionary of |Param| and |Column| objects from burst search.

    """
    detdef = bg.detdef
    if streams == 'auto':
        streams = PhSel('0ex_1ex1em') if detdef.ex == 2 else PhSel('all')
    if isinstance(streams, (PhSel, PhStream, str)):
        streams = (_as_phsel(streams), )
    else:
        streams = tuple(_as_phsel(stream) for stream in streams)
    streams, size = _match_size(streams, None, 'streams')
    m, size = _match_size(m, size, 'm')
    F, size = _match_size(F, size, 'F')
    c, size = _match_size(c, size, 'c')
    if size is None:
        streams = (next(streams), )
        size = 1
    sfuse = 0.0 if size != 1 else fuse
    parents = {'bg':bg}
    param_iter = ({'stream':stream, 'm':mm, 'F':FF, 'c':cc, 'fuse':sfuse} 
                  for stream, mm, FF, cc in zip(streams, m, F, c))
    bursts = tuple(Param(Bursts, param, parents) for param in param_iter)
    if size == 1 and isinstance(truthtable, str) and truthtable == 'auto':
        bursts = bursts[0]
    else:
        if truthtable == 'auto':
            truthtable = 'and'
        bursts = Param(BurstOvlp, {'fuse':fuse, 'truthtable':truthtable}, {'bases':bursts})
    rkwargs = dict(alpha=alpha, delta=delta, gamma=gamma, beta=beta, nbva=nbva,
                   dir_ex=dir_ex, lk=lk, corr_mat=corr_mat, update=update)
    rkwargs = {k:v for k, v in rkwargs.items() if v is not None}
    out = make_fret_from_base(bursts, bg=bg, **rkwargs)
    out['bursts'] = bursts
    return out


def make_correction_factors(nphbg:Param=None, alpha:float=None, delta:float=None,
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
    detdef = nphbg.detdef
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


def apply_gate(dct:dict[str:Param|Column], gate:GateGroup, inplace:bool=True)->dict[str:Param|Column]:
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
    out = dct if inplace else dict()
    for key, val in dct.items():
        if not _regateable(val, gate):
            continue
        out[key] = val.regate(gate)
    return out


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
        in :attr:`Gate.columns <smfbursts.datamodel.tables.Gate.columns>` , 
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


@cite('HellenkampNatMeth2018', purpose="Computation of alpha/delta correction factors mean of population (using alpha delta gamma beta formalism)")
def crosstalk_correction(data:PhotonDataS, col:Column, gate:GateGroup=None)->float:
    r"""
    Computes the correction factor for a cross-talk correction factor according
    to the equation :math:`c = \frac{\langle R \rangle}{1 - \langle R \rangle}`.
    Where :math:`c` is the correction factor 
    (:math:`\alpha`/leakage, or :math:`\delta`/direct excitation)
    and :math:`R` is the column of the appropriate ratiometric parameter
    (``E_bg`` for :math:`\alpha` and ``S_bg`` for :math:`\delta`)

    Parameters
    ----------
    data : PhotonDataS
        Source of burst data.
    col : Column
        Column used to compute correction factor, should either be a ``E_bg``
        or ``S_bg`` column.
    gate : GateGroup, optional
        Gate to select the approprite single emitter population. If None, assume
        that ``col`` already has appropriate gate. The default is None.

    Returns
    -------
    float
        Value of correction factor.

    """
    col = col if gate is None else col.regate(gate)
    m = data.get_column(col) if isinstance(data, PhotonData) else data.concatenate_column(col)
    m = np.mean(m)
    return m / (1 - m)


@cite('HellenkampNatMeth2018', purpose="Computation of gamma/beta corrections from fit of multiple values (using alpha delta gamma beta formalism)")
def gamma_beta_twopopvals(EA:float, SA:float, EB:float, SB:float)->tuple[float,float]:
    r"""
    Compute :math:`\gamma` and :math:`\beta` values from the cross-talk corrected
    :math:`^{iii}E_{app}` and :math:`^{iii}S_{app}` values of two sub-populations
    
    Computes the slope :math:`a` and intercept :math:`b` of the inverse of
    :math:`^{iii}E_{app}` and :math:`^{iii}S_{app}` of the two populations,
    Computes gamma as :math:`\gamma = (a-1)/(a+b+1)`, and beta as
    :math:`\beta = a + b + 1`.

    Parameters
    ----------
    EA : float
        :math:`^{iii}E_{app}` value of first population.
    SA : float
        :math:`^{iii}S_{app}` value of first population.
    EB : float
        :math:`^{iii}E_{app}` value of second population.
    SB : float
        :math:`^{iii}S_{app}` value of second population.

    Returns
    -------
    gamma : float
        Computed gamma value.
    beta : float
        Computed beta value.
    """
    ea, eb, sa, sb = 1/EA, 1/EB, 1/SA, 1/SB
    a = (sb - sa) / (ea - eb)
    b = a*ea - sa
    beta = a + b - 1
    gamma = (a-1) / beta
    return gamma, beta


@cite('HellenkampNatMeth2018', purpose="Computation of gamma/beta corrections from fit of multiple values (using alpha delta gamma beta formalism)")
def gamma_beta_twopop(data:PhotonDataS, colEapp:Column, colSapp:Column, 
                      gateA:GateGroup, gateB:GateGroup)->tuple[float,float]:
    r"""
    Compute :math:`\gamma` and :math:`\beta` values from the mean 
    :math:`^{iii}E_{app}` and :math:`^{iii}S_{app}` values of the sub-populations
    defined by ``gateA`` and `gateB``.
    
    Computes the slope :math:`a` and intercept :math:`b` of the inverse of
    :math:`^{iii}E_{app}` and :math:`^{iii}S_{app}` of the two populations,
    Computes gamma as :math:`\gamma = (a-1)/(a+b+1)`, and beta as
    :math:`\beta = a + b + 1`.

    Parameters
    ----------
    data : PhotonDataS
        Source data.
    colEapp : Column
        Column defining the cross-talk corrected transfer efficiency.
        Generally this means a |Column| with source param of a |Param| based
        on :class:`smfbursts.bursttables.Ratios` with alpha and delta set and 
        gamma, beta = 1.0. Should be a ``E`` column.
    colSapp : Column
        Column defining the cross-talk corrected stoichiometry.
        Generally this means a |Column| with source param of a |Param| based
        on :class:`smfbursts.bursttables.Ratios` with alpha and delta set and 
        gamma, beta = 1.0. Should be a ``S`` column.
    gateA : GateGroup
        Gate for first fret population.
    gateB : GateGroup
        Gate for second fret population.

    Returns
    -------
    gamma : float
        Computed gamma value.
    beta : float
        Computed beta value.

    """
    getfunc = data.get_column if isinstance(data, PhotonData) else data.concatenate_column
    EA = np.mean(getfunc(colEapp, gateA))
    SA = np.mean(getfunc(colSapp, gateA))
    EB = np.mean(getfunc(colEapp, gateB))
    SB = np.mean(getfunc(colSapp, gateB))
    return gamma_beta_twopopvals(EA, SA, EB, SB)


def _ab_to_gamma_beta(a:float, b:float)->tuple[float,float]:
    r"""
    Conversion from slope-intercept of inverse :math:`S^{-1} = a*E+b` 
    to gamma beta according to |hellenkamp| :math:`\beta = a + b + 1`
    :math:`\gamma = (b - 1) / (a + b -1)`
    """
    beta = a + b - 1
    gamma = (b - 1) / beta
    return gamma, beta


GBfit = Callable[[np.ndarray[np.float64],np.ndarray[np.float64]],tuple[float,float,OptimizeResult|LinregressResult]]


@cite('HellenkampNatMeth2018', purpose="Computation of gamma/beta corrections from fit of multiple values (using alpha delta gamma beta formalism)")
def gamma_beta_linregressvals(E:np.ndarray[np.float64], S:np.ndarray[np.float64],
                              **kwargs:Any)->tuple[float,float,LinregressResult]:
    r"""
    Compute :math:`\gamma` and :math:`\beta` values from slope (:math:`a`) and 
    intercept (:math:`b`) of cross-talk corrected proximity ratio and
    inverse of cross-talk corrected stoichometry, with slope and intercept 
    computed using linear regression (uses |linregress|).
    
    This function implements the linear fitting method from |hellenkamp|.
    
    :math:`\gamma = (b - 1) / (a + b - 1)`
    
    :math:`\beta = a + b - 1`

    Parameters
    ----------
    E : np.ndarray[np.float64]
        Array of values of cross-talk corrected proximity ratio :math:`^{iii}E_{app}`.
    S : np.ndarray[np.float64]
        Array of values of cross-talk corrected stoichiometry :math:`^{iii}S_{app}`.
    **kwargs : Any
        Keyword arguments passed to |linregress|.

    Returns
    -------
    gamma : float
        Computed value of :math:`\gamma`
    beta : float
        Computed value of :math:`beta`
    res : LinregressResult
        Output of |linregress| note that this provides slope (:math:`a`) and 
        intercept (:math:`b`) values, before conversion to :math:`\gamma` and
        :math:`\beta`.

    """
    e, s = np.asarray(E), 1/ np.asarray(S)
    res = linregress(e, s, **kwargs)
    a, b = res.slope, res.intercept
    gamma, beta = _ab_to_gamma_beta(a, b)
    return gamma, beta, res


def _linfit(x, E):
    """Linear equation fit function"""
    return x[0]*E + x[1]


@cite('HellenkampNatMeth2018', purpose="Computation of gamma/beta corrections from fit of multiple values (using alpha delta gamma beta formalism)")
def gamma_beta_fitlinearvals(E:np.ndarray[np.float64], S:np.ndarray[np.float64], 
                             min_func:MinFunc=lsq_anyfit, **kwargs)->tuple[float,float,OptimizeResult]:
    r"""
    Compute :math:`\gamma` and :math:`\beta` values from slope (:math:`a`) and 
    intercept (:math:`b`) of inverses of cross-talk corrected proximity ratio and
    stoichometry, with slope and intercept computed using fitting algorithm
    of ``min_func``.
    
    This function implements the linear fitting method from |hellenkamp|.
    
    :math:`\gamma = (a - 1) / (a + b - 1)`
    
    :math:`\beta = a + b - 1`


    Parameters
    ----------
    E : np.ndarray[np.float64]
        Array of values of cross-talk corrected proximity ratio :math:`^{iii}E_{app}`.
    S : np.ndarray[np.float64]
        Array of values of cross-talk corrected stoichiometry :math:`^{iii}S_{app}`.
    min_func : MinFunc, optional
         Minimization algorithm, should have signature of a 
         :attr:`smfbursts.datamodel.multifit.MinFunc`.
         Typically is one of 
         - :func:`smfbursts.datamodel.multifit.lsq_anyfit`
         - :func:`smfbursts.datamodel.multifit.min_anyfit`
         The default is lsq_anyfit.
    **kwargs : Any
        Keyword arguments passed to ``min_func``.

    Returns
    -------
    gamma : float
        Computed value of :math:`\gamma`.
    beta : float
        Computed value of :math:`beta`.
    res : OptimizeResult
        Result returned by fitting algorithm. Note that this provides 
        slope (:math:`a`) and intercept (:math:`b`) values, 
        before conversion to :math:`\gamma` and :math:`\beta`.
    
    """
    e, s = np.asarray(E), 1/np.asarray(S)
    res = min_func(np.array([1.0, 1.0]), args=(e, s, _linfit), **kwargs)
    a, b = res.x
    gamma, beta = _ab_to_gamma_beta(a, b)
    return gamma, beta, res


def _gamma_beta_relation(x:np.ndarray[np.float64], E:np.ndarray[np.float64]):
    """Equation 17 from hellenkamp_ evaluator for use with min_anyfit or lsq_anyfit"""
    gamma, beta = x
    return 1/(1+gamma*beta+(1-gamma)*beta*E)


@cite('HellenkampNatMeth2018', purpose="Computation of gamma/beta corrections from fit of multiple values (using alpha delta gamma beta formalism)")
def gamma_beta_fitdirectvals(E:np.ndarray[np.float64], S:np.ndarray[np.float64], 
                       min_func=lsq_anyfit, **kwargs)->tuple[float,float,OptimizeResult]:
    r"""
    Compute :math:`\gamma` and :math:`\beta` values by direct fitting of values
    of ``E`` and ``S`` (should be :math:`^{iii}E_{app}` and :math:`^{iii}S_{app}`)
    to equation 17 of |hellenkamp|.
    
    Fits :math:`^{iii}S_{app} = (1 +\gamma\beta+(1-\gamma)\beta ^{iii}E_{app})^{-1}`

    Parameters
    ----------
    E : np.ndarray[np.float64]
        Array of values of cross-talk corrected proximity ratio :math:`^{iii}E_{app}`.
    S : np.ndarray[np.float64]
        Array of values of cross-talk corrected stoichiometry :math:`^{iii}S_{app}`.
    min_func : MinFunc, optional
         Minimization algorithm, should have signature of a 
         :attr:`smfbursts.datamodel.multifit.MinFunc`.
         Typically is one of 
         - :func:`smfbursts.datamodel.multifit.lsq_anyfit`
         - :func:`smfbursts.datamodel.multifit.min_anyfit`
         The default is lsq_anyfit.
    **kwargs : Any
        Keyword arguments passed to ``min_func``.

    Returns
    -------
    gamma : float
        Computed value of :math:`\gamma`.
    beta : float
        Computed value of :math:`beta`.
    res : OptimizeResult
        Result returned by fitting algorithm.
    
    """
    E, S = np.asarray(E), np.asarray(S)
    res = min_func(np.array([1.0, 1.0]), args=(E, S, _gamma_beta_relation), **kwargs)
    gamma, beta = res.x
    return gamma, beta, res


@cite('HellenkampNatMeth2018', purpose="Computation of gamma/beta corrections from fit of multiple values (using alpha delta gamma beta formalism)")
def gamma_beta_pops(data:PhotonDataS, colEapp:Column, colSapp:Column, 
                        *args:GateGroup, fit_func:GBfit=gamma_beta_linregressvals, 
                        return_fitres:bool=False, **kwargs:Any
                        )->tuple[float,float]|tuple[float,float,OptimizeResult|LinregressResult]:
    r"""
    Compute the gamma and beta factors based on a linear fit of the E and S
    values of N distributions. Note this function is specifically tailored to
    ALEX/PIE measurements. Note that this function does not check that ``colEapp``
    or ``colSapp`` or any of the gates are reasonable.
    
    The input to ``colEapp`` should be a column of the cross-talk corrected FRET
    efficiency (:math:`^{iii}E_{app}`) 
    (``E`` |Column| from |Param| based on  :class:`smfbursts.bursttables.Ratios`).
    (the alpha and delta parameters should be set, gamma and beta = 1.0)
    
    The input to ``colSapp`` should be a column of the cross-talk corrected FRET
    efficiency (:math:`^{iii}S_{app}`) 
    (``S`` |Column| with the same source param as ``colEapp``.
     
     Returns gamma and beta values as tuple, can also optionally return |optimizeresult|
     after gamma and beta, allowing characterization of error etc. 
    

    Parameters
    ----------
    data : PhotonDataS
        Source data.
    colEapp : Column
        Column defining the cross-talk corrected transfer efficiency.
        Generally this means a |Column| with source param of a |Param| based
        on :class:`smfbursts.bursttables.Ratios` with alpha and delta set and 
        gamma, beta = 1.0. Should be a ``E`` column.
    colSapp : Column
        Column defining the cross-talk corrected stoichiometry.
        Generally this means a |Column| with source param of a |Param| based
        on :class:`smfbursts.bursttables.Ratios` with alpha and delta set and 
        gamma, beta = 1.0. Should be a ``S`` column.
    *args : GateGroup
        Gates for each FRET active population, must supply at least 2.
    fit_func : Callable, optional
        Function used to fit the gamma and beta values to ``colEapp`` and ``colSapp``.
        Typically one of
        - :func:`gamma_beta_linregressvals`
        - :func:`gamma_beta_fitdirectvals`
        - :func:`gamma_beta_fitlinearvals`
        The above functions all fit equation 17 of |hellenkamp| in some way.
        They all must return a 3 tuple of ``(gamma, beta, res)`` where ``res``
        is the full result object 
        (|optimizeresult| or ``LinregressResult`` for the standard functions).
        ``fit_func`` is called as ``fit_func(E, S, **kwargs)`` where ``E`` and
        ``S`` are the mean values of the populations specified by the input
        gates (\*args) in ``colEapp`` and ``colSapp`` respectively.
        The default is gamma_beta_linregressvals.
    return_optimizerresult : bool, optional
        If :code:`True` return the result after the gamma and beta values in 
        returned tuple. The default is False.
    **kwargs : Any
        kwargs handed to ``fit_func``.

    Returns
    -------
    gamma : float
        Best fit value for gamma factor.
    beta : float
        Best fit value for beta factor.
    res : OptimizerResult | LinregressResult, optional
        Only returned if ``return_optimizerresult = True`` the |optimizeresult|
        returned by the optimization function ``fit_func``

    """
    if len(args) < 2:
        raise ValueError("must specify at least 2 FRET populations")
    getfunc = data.get_column if isinstance(data, PhotonData) else data.concatenate_column
    E = [np.mean(getfunc(colEapp, g)) for g in args]
    S = [np.mean(getfunc(colSapp, g)) for g in args]
    out = fit_func(E, S, **kwargs)
    if not return_fitres:
        out = out[:-1]
    return out


@cite('HellenkampNatMeth2018', purpose="Computation of gamma/beta corrections from fit of multiple values (using alpha delta gamma beta formalism)")
def gamma_beta_bursts(data:PhotonDataS, colEapp:Column, colSapp:Column, 
                      gate:GateGroup=None, fit_func:GBfit=gamma_beta_linregressvals, 
                      return_fitres:bool=False, **kwargs:Any
                      )->tuple[float,float]|tuple[float,float,OptimizeResult|LinregressResult]:
    r"""
    Compute the gamma and beta factors based on a linear fit of the E and S
    values of all FRET active bursts. Note this function is specifically tailored to
    ALEX/PIE measurements. Note that this function does not check that ``colEapp``
    or ``colSapp`` or any of the gates are reasonable.
    
    The input to ``colEapp`` should be a column of the cross-talk corrected FRET
    efficiency (:math:`^{iii}E_{app}`) 
    (``E`` |Column| from |Param| based on  :class:`smfbursts.bursttables.Ratios`).
    (the alpha and delta parameters should be set, gamma and beta = 1.0)
    
    The input to ``colSapp`` should be a column of the cross-talk corrected FRET
    efficiency (:math:`^{iii}S_{app}`) 
    (``S`` |Column| with the same source param as ``colEapp``.
     
     Returns gamma and beta values as tuple, can also optionally return |optimizeresult|
     after gamma and beta, allowing characterization of error etc. 
    

    Parameters
    ----------
    data : PhotonDataS
        Source data.
    colEapp : Column
        Column defining the cross-talk corrected transfer efficiency.
        Generally this means a |Column| with source param of a |Param| based
        on :class:`smfbursts.bursttables.Ratios` with alpha and delta set and 
        gamma, beta = 1.0. Should be a ``E`` column.
    colSapp : Column
        Column defining the cross-talk corrected stoichiometry.
        Generally this means a |Column| with source param of a |Param| based
        on :class:`smfbursts.bursttables.Ratios` with alpha and delta set and 
        gamma, beta = 1.0. Should be a ``S`` column.
    gate : GateGroup
        Gate for the FRET population, if None, & gate of colEapp and colSapp.
    fit_func : Callable, optional
        Function used to fit the gamma and beta values to ``colEapp`` and ``colSapp``.
        Typically one of
        - :func:`gamma_beta_linregressvals`
        - :func:`gamma_beta_fitdirectvals`
        - :func:`gamma_beta_fitlinearvals`
        The above functions all fit equation 17 of |hellenkamp| in some way.
        They all must return a 3 tuple of ``(gamma, beta, res)`` where ``res``
        is the full result object 
        (|optimizeresult| or ``LinregressResult`` for the standard functions).
        ``fit_func`` is called as ``fit_func(E, S, **kwargs)`` where ``E`` and
        ``S`` are the values of the columns ``colEapp`` and ``colSapp`` respectively.
        The default is gamma_beta_linregressvals.
    return_fitres : bool, optional
        If :code:`True` return the result after the gamma and beta values in 
        returned tuple. The default is False.
    **kwargs : Any
        kwargs handed to ``fit_func``.

    Returns
    -------
    gamma : float
        Best fit value for gamma factor.
    beta : float
        Best fit value for beta factor.
    res : OptimizerResult | LinregressResult, optional
        Only returned if ``return_optimizerresult = True`` the |optimizeresult|
        returned by the optimization function ``fit_func``

    """
    getfunc = data.get_column if isinstance(data, PhotonData) else data.concatenate_column
    gate = colEapp.base_gate & colSapp.base_gate if gate is None else gate
    colEapp = colEapp.regate(gate)
    colSapp = colSapp.regate(gate)
    E = getfunc(colEapp)
    S = getfunc(colSapp)
    out = fit_func(E, S, **kwargs)
    if not return_fitres:
        out = out[:-1]
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
#: - "scatter" - default kwargs for :func:`smfbursts.datamodel.plot.scatter` when plotting bursts
#: - "hexbin" - default kwargs for :func:`smfbursts.datamodel.plot.hexbin` when plotting bursts
#: - "hexbinraw" - default kwargs for :func:`smfbursts.datamodel.plot.hexbin` when plotting bursts with raw, uncorrected values
#: - "histbar" - default kwargs for :func:`smfbursts.datamodel.plot.hist_bar` when plotting bursts
#: - "ratio_bins"- default bins for histograms of bg/fully corrected ratiometric burst parameters like E and S
#: - "raw_ratio_bins"- default bins for histograms of raw corrected ratiometric burst parameters like E_raw and S_raw
#: - "kdeover"- default kwargs to use with :func:`smfbursts.datamodel.plot.hist_kdeoverlay` when plotting bursts
#: - "streams"- default :class:`smfbursts.ph_sel.PhSel`\s for streams of ALEX parameters
#: - "stream_labels"- default D/A ex/em names for streams, parallels "streams"
#: - "stream_zorder"- default zorder for stacking streams over each other, parallels "streams"
#: - "stream_colors"- default colors for each stream, parallels "streams"
#: 
ALEXdefaults = SequenceDefaults(
    scatter={'s':2.0},
    hexbin={'gridsize':40, 'extent':(-0.2,1.2,-0.2,1.2), 'mincnt':1, 
            'edgecolor':'none', 'linewidth':0.2},
    hexbinraw={'gridsize':40, 'extent':(0.0,1.0,0.0,1.0), 'mincnt':1, 
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
#: 
#: Contains the following keys\:
#: 
#: - "bursts" - default kwargs for scatter functions of bursts
#: - "histbar" - default kwargs for :func:`smfbursts.datamodel.plot.hist_bar` when plotting bursts
#: - "ratio_bins"- default bins for histograms of bg/fully corrected ratiometric burst parameters like E and S
#: - "raw_ratio_bins"- default bins for histograms of raw corrected ratiometric burst parameters like E_raw and S_raw
#: - "kdeover"- default kwargs to use with :func:`smfbursts.datamodel.plot.hist_kdeoverlay` when plotting bursts
#: - "streams"- default :class:`smfbursts.ph_sel.PhSel`\s for streams of ALEX parameters
#: - "stream_labels"- default D/A ex/em names for streams, parallels "streams"
#: 
MonoExdefaults = SequenceDefaults(
    bursts={'s':2.0},
    histbar=_histbar_kwargs, ratio_bins=_ratio_bins, raw_ratio_bins=_raw_ratio_bins,
    kdeover=_kdehistbar_kwargs,
    streams=(PhSel('all'), PhSel('0em'), PhSel('1em')),
    stream_colors=_base_ALEX_kwargs[:-1], 
    stream_labels=('All', 'Dem', 'Aem',),
    )