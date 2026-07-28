#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created : 9/10/20025
# Author: Paul David Harris
# email: harrip@gmail.com
"""
The ``smfbursts.childtables`` module defines the main |ChildPhotonTable| used
in burst based analysis. These tables can have as a |Pbaseparam| any
|BasePhotonTable|, but typically they will be either |Bursts| or |BurstOvlp|
based |Param|


.. |Param| replace:: :class:`Param <smfbursts.datamodel.tables.Param>`
.. |Pbaseparam| replace:: :attr:`base_parent <smfbursts.datamodel.tables.Param.base_param`
.. |BasePhotonTable| replace:: :class:`BasePhotonTable <smfbursts.photondata.BasePhotonTable>`
.. |ChildPhotonTable| replace:: :class:`ChildPhotonTable <smfbursts.photondata.ChildPhotonTable>`
.. |Bursts| replace:: :class:`Bursts <smfbursts.bursttables.Bursts>`
.. |BurstOvlp| replace:: :class:`BurstOvlp <smfbursts.bursttables.BurstOvlp>`
"""
from typing import Any, ClassVar, Literal
from collections.abc import Iterator, Sequence, Callable
from functools import partial
from warnings import warn
from itertools import chain, repeat, permutations
from numbers import Real

import numpy as np

from .datamodel.utils import tupledict, arr_slc
from .datamodel.immutabledata import (
    register_PyCode, get_pycode_subval,
    TV_bool, TV_float, TV_int, TV_str, TV_ndarray, TV_PyCode, TV_tuple
                                      )
from .datamodel.diskdict import DiskDict
from .datamodel.tables import ParamDef, ParentDef, ColumnDef, Param, Column, as_paramdict, paramproperty
from .cite import cite, add_citation
from .photondata import (
    PhSpec, PhotonData, PhotonTable, BasePhotonTable, ChildPhotonTable, BasePhotonTableLike, 
    _regularize_column_startstop, _regularize_ph_sel, 
    _title_sels, _title_startstop_append, _title_unit_append, _pol_ps,
    make_base_column_defs, ColKeyStart, ColKeyStop
    )
from .backgroundtables import BG
from .ph_sel import PhSel, PhStream, DetDef, TV_PhSel, sort_phsels, phsel_all

import smfbursts.cfuncs as smc


_alloc_size:int = 512


def _get_nph_title(col:Column, name:str, include_unit:bool, origin:PhotonData)->str:
    """Sub-function to format title of nph-like columns, with "core" name of ``name`` """
    title = _title_sels('_{%s}n'%name, origin, col.keytup[0])[0]
    title = _title_startstop_append(title, col.keytup[1], col.keytup[2])
    title = _title_unit_append(title, 'cnts', include_unit)
    return f'${title}$'


def _calc_brightness(table:PhotonTable, base:BasePhotonTable, nph_name:str, phsel:PhSel, 
                     starttype:ColKeyStart, stoptype:ColKeyStop)->np.ndarray[np.double]:
    """
    Brightness calcualtion function, table and base are Tables for getting approprite column,
    nph_name is name of nph_column, broadcasts starttype and stoptype so consistent.
    """
    return table[nph_name, phsel, starttype, stoptype] / base['dur', starttype, stoptype]


def _get_brightness_title(col:Column, name:str, include_unit:bool, origin:PhotonData)->str:
    """Getter function for brightness title"""
    title = _title_sels(name, origin, col.keytup[0])[0]
    title = _title_startstop_append(title, col.keytup[1], col.keytup[2])
    title = _title_unit_append(title, 'cnts s^{-1}', include_unit)
    return f'${title}$'


def _iter_ratio(table:PhotonTable, nph_name:str, phsel_num:PhSel, phsel_dem:PhSel, 
                starttype:ColKeyStart, stoptype:ColKeyStop)->float:
    """General iterator function for ratio_[]"""
    for n, d in zip(table.iter_column(nph_name, phsel_num, phsel_dem, starttype, stoptype),
                    table.iter_column(nph_name, phsel_num, phsel_dem, starttype, stoptype)):
        yield n / d

def _calc_ratio(table:PhotonTable, nph_name:str, phsel_num:PhSel, phsel_dem:PhSel, starttype:str, stoptype:str):
    """General getter function for ratio_[]"""
    with np.errstate(divide='ignore'):
        out = table[nph_name, phsel_num, starttype, stoptype] / table[nph_name, phsel_dem, starttype, stoptype]
    return out


def _get_ratio_title(col:Column, name:str, include_unit:bool, origin:PhotonData)->str:
    """General title getter function for ratio_[]"""
    title = '%s/%s' %  _title_sels(name, origin, *col.keytup[:2])
    title = _title_startstop_append(title, col.keytup[2], col.keytup[3])
    return f'${title}$'


def _iter_anisotropy(table:PhotonTable, nph_name:str, phsel_p:PhSel, phsel_s:PhSel, 
                     starttype:ColKeyStart, stoptype:ColKeyStop)->np.ndarray[np.double]:
    """General iterator function for anisotropy_[]"""
    for p, s in zip(table.iter_column(nph_name, phsel_p, starttype, stoptype),
                    table.iter_column(nph_name, phsel_p, starttype, stoptype)):
        return (p-s)/(p+2*s)


def _calc_anisotropy(table:PhotonTable, nph_name:str, phsel_p:PhSel, phsel_s:PhSel, 
                     starttype:ColKeyStart, stoptype:ColKeyStop):
    """General igetter function for anisotropy_[]"""
    p = table[nph_name, phsel_p, starttype, stoptype]
    s = table[nph_name, phsel_s, starttype, stoptype]
    with np.errstate(divide='ignore'):
        out = (p-s)/(p+2*s)
    return out


def _get_anisotropy_title(col:Column, name:str, include_unit:bool=False, origin:PhotonData=None)->str:
    """General title getter for anisotropy_[] column"""
    kw = {'name':name}
    par, perp, start, stop = col.keytup
    fuse = par | perp
    overlap = par | perp
    detdef = None
    if origin is not None:
        kw.update(detdef=origin.detdef, stream_names=origin.get_stream_names())
        fuse = fuse.render_positive(origin.detdef, convert_all=True)
        overlap = overlap.render_positive(origin.detdef, convert_all=True)
        detdef = origin.detdef
    if not overlap and _pol_ps(fuse, detdef, None if origin is None else origin.setup):
        kw['name'] = 'r'
        title = fuse.tex_str(kw)
    else:
        title = rf'anis({par.tex_str(**kw)},\: {perp.tex_str(**kw)})'
    return _title_startstop_append(title, start, stop)


def _get_nmunits(setup:PhSpec, sid:int, ex_stride:int, irf:DiskDict)->tuple[float,float,float]:
    """
    Compute the tcspc_unit, irf_mean, bg_mean of a given stream (sid)

    Parameters
    ----------
    setup : PhSpec
        Setup Spec of data.
    sid : int
        Single detector ID.
    ex_stride : int
        DetDef.ex_stride.
    irf : DiskDict
        IRF dict of choice (either thresh or irf).

    Raises
    ------
    ValueError
        ex ranges specifies broken excitation range.

    Returns
    -------
    tcspc_unit : float
        TCSPC unit of channel sid.
    irf_mean : float
        Mean time of IRF for given channel, this is the value to shift each nanotime
        so that nanomean computes correctly.
    bg_mean : float
        Expected nanomean of background, 
        where start of excitation period is ``-irf_mean``.

    """
    tcspc_unit = setup.tcspc_unit[sid%ex_stride]
    ex_range = setup.ex_ranges[sid%ex_stride]
    if ex_range.shape[0] != 1:
        raise ValueError("can only compute nanomean of contiguous time range, split excitation ranges not allowed")
    irf_c = irf[setup.detdef.stream_ids_to_PhSel(sid)]
    if isinstance(irf_c, Real):
        irf_mean = irf_c
    else:
        irf_mean = np.sum(np.arange(irf_c.size)*irf_c) / irf_c.sum() + ex_range[0,0]
    bg_mean = np.diff(ex_range[0])[0] / 2 - irf_mean +  ex_range[0,0]
    return tcspc_unit, irf_mean, bg_mean


def _extract_nmarrays(stream_ids:np.ndarray[np.uint8], setup:PhSpec, irf:DiskDict
                      )->list[np.ndarray[np.float64],np.ndarray[np.float64],np.ndarray[np.float64]]:
    """
    Get the necessary nanomean bg arrays from stream ids.

    Parameters
    ----------
    stream_ids : np.ndarray[np.uint8]
        Array of stream_ids of PhSel.
    setup : PhSpec
        Setup spec of origin.
    irf : DiskDict
        Choose IRF type, either .

    Returns
    -------
    tcspc_unit : np.ndarray[np.float64]
        TCSPC unit of each detector id in stream_ids
    irf_mean : np.ndarray[np.float64]
        Expeceted mean of IRF (in TCSPC units) of each detector id in stream_ids.
    bg_mean : np.ndarray[np.float64]
        Expected mean of background (in TCSPC unit, shifted by irf_mean) fo 
        each detector id in stream_ids.
    """
    ex_stride = setup.detdef.ex_stride
    return list(map(np.array, zip(*(_get_nmunits(setup, sid, ex_stride, irf) for sid in stream_ids))))


class NphBG(ChildPhotonTable):
    r"""
    Table for background corrected photon counts.
    No corrections for cross-talk and/or detection efficiencies.
    
    Params
    ------
        single : bool
            If :code:`True`, compute only single streams, colums of compund 
            :class:`PhSel <smfbursts.ph_sel.PhSel>` computed as sum of streams. 
            Default is True.
    
    Parents
    -------
        base : Param[BasePhotonTable]
            Usually a :class:`Burst`, the time ranges over which rows are computed
        bg : Param[BG]
            The background counts to use.
     
    Columns
    -------
        nph_bg : float, (ph_sel:PhSel, starttype:str, stoptype:str)
            Background adjusted photon counts in ph_sel, starttype and stoptype
            define what start/stop values to use for computing burst duration and
            therefore background photon counts
        sbr : float, (ph_sel:PhSel, starttype:{'istarttime', 'start'}, stoptype:{'istoptime', 'stop'})
            signal to background ratio, nph_raw / bg counts
        brightness_bg : float, (ph_sel:PhSel, starttype:{'istarttime', 'start'}, stoptype:{'istoptime', 'stop'})
            counts per second in ph_sel, with background rate subtracted
        ratio_bg : float, (num_ph_sel:PhSel, dem_ph_sel:PhSel, starttype:{'istarttime', 'start'}, stoptype:{'istoptime', 'stop'})
            ratio of num_ph_sel to dem_ph_sel background adjusted counts.
        nanomean_bg : float, (phsel:PhSel, mean:{'irf','thresh'}, starttype:str, stoptype:str)
            Mean nanotime with correction for background counts. Uses the equation
            
            .. math:
                
                \tau = \frac{\sum_{i=1}^{N}{t_{i}} - n_{bg}*\bar{t_{bg}}}{N-n_{bg}}
        
        
            where :math:`N` is the total number of photons, :math:`t_{i}` is the
            nanotime of the :math:`i^{th}` photon in the burst, with time 0 set
            by the choice of mean, if ``'irf'`` then set time 0 as mean of IRF,
            if ``'thresh'``, set time 0 as ``irf_thresh``. :math:`n_{bg}` is the
            estimated number of photons in the burst 
            (using ``rangecounts`` column of :class:`BG <smfbursts.background.BG>`)
            and :math:`\bar{t_{bg}}` is the expected mean of the background, assuming
            background is equally likely across all TCSPC bins in the excitation range.
            This mean is set using the same time scale as :math:`t_{i}`.
    
    Remapped Columns
    ----------------
        E_bg : float, (starttype:{'istarttime', 'start'}, stoptype:{'istoptime', 'stop'})
            Convenience column returns presumed FRET efficiency
            Remaped column of ratio_bg, give ratio of PhSel('0ex1em') to PhSel('0ex')
        S_bg : float, (starttype:{'istarttime', 'start'}, stoptype:{'istoptime', 'stop'})
            Convenience column return presumed Stoichiometry
            Remaped column of ratio_bg, give ratio of PhSel('1ex1em') to 
            PhSel('0ex_1ex1em')     
    
    """
    #: :meta private:
    param_defs = (
        ParamDef('single', TV_bool, default=True),
                  )
    #: :meta private:
    parent_defs = (
        ParentDef('base', BasePhotonTableLike, is_base=True),
        ParentDef('bg', BG, is_base=False),
                   )
    #: :meta private:
    column_defs = (
        ColumnDef('nph_bg', (PhSel,TV_str, TV_str), 0, 'some', 
                  get_func='_get_nph_bg', iter_func='_iter_nph_bg',
                  reg_func='_regularizecolumn_nph_bg_sbr', title_func='_get_nph_bg_title',
                  unit='cnts s^{-1}', index_unit='cnts s-1', title_is_tex=True),
        ColumnDef('sbr', (PhSel, TV_str, TV_str), 0, 'user', 
                  get_func='_get_sbr', iter_func='_iter_sbr', 
                  reg_func='_regularizecolumn_nph_bg_sbr',
                  title_func='_get_sbr_title', title_is_tex=True),
        ColumnDef('brightness_bg', (PhSel, TV_str, TV_str), 0, 'never', 
                  get_func='_get_brightness_bg', reg_func='_regularizecolumn_brightness_bg',
                  title_func='_get_brightness_bg_title', unit='cnts s^{-1}',
                  index_unit='cnts s-1', title_is_tex=True),
        ColumnDef('ratio_bg', (PhSel, PhSel, TV_str, TV_str), 0, 'never', 
                  get_func='_get_ratio_bg', iter_func='_iter_ratio_bg',
                  reg_func='_regularizecolumn_ratio_bg', title_func='_get_ratio_bg_title',
                  title_is_tex=True),
        ColumnDef('anisotropy_bg', (PhSel, PhSel, TV_str, TV_str), 0, 'never', 
                  get_func='_get_anisotropy_bg', iter_func='_iter_anisotropy_bg',
                  reg_func='_regularizecolumn_ratio_bg', title_func='_get_anisotropy_bg_title',
                  title_is_tex=True),
        ColumnDef('nanomean_bg', (PhSel, TV_str(isin=('irf', 'thresh')), TV_str, TV_str), 0, 'user', 
                  iter_func='_iter_nanomean_bg', reg_func='_regularizecolumn_nanomean_bg',
                  title_func='_get_nanomean_bg_title', unit='s'),
        ColumnDef('E_bg', (TV_str, TV_str), 0, remap='_replace_E_bg', reg_func='_regularizecolumn_ES_bg'),
        ColumnDef('S_bg', (TV_str, TV_str), 0, remap='_replace_S_bg', reg_func='_regularizecolumn_ES_bg'),
                   )

    def __init_columns__(self):
        pass

    @classmethod
    def _regularizecolumn_nph_bg_sbr(cls, source_param:Param, *args):
        """Column regularization for nph_bg and sbr columns"""
        return args[0:1] +  _regularize_column_startstop(source_param, *args[1:])

    def _iter_nph_bg(self, phsel:PhSel, starttype:str, stoptype:str)->Iterator[float]:
        """Iter function for nph_bg column"""
        nph_iter = (nph - bg for nph, bg in zip(self.parents['base'].iter_column('nph_raw', phsel),
                                                        self.parents['bg'].iter_column('rangecounts', 
                                                                                       self.parents['base'].param, 
                                                                                       phsel, starttype, stoptype)))
        stream_ids = self.origin.detdef.get_stream_ids(phsel)
        if not self.param.params['single'] or len(stream_ids) == 1:
            out = list()
            for nph in nph_iter:
                out.append(nph)
                yield nph
            self._add_column('nph_bg', (phsel, starttype, stoptype), np.array(out, dtype=np.float64))
        else:
            yield from nph_iter

    def _get_nph_bg(self, phsel:PhSel, starttype:str, stoptype:str)->np.ndarray[np.float64]:
        """Getter function for nph_bg"""
        nph = self.parents['base']['nph_raw', phsel]
        bg = self.parents['bg']['rangecounts', self.param.parents['base'], phsel, starttype, stoptype]
        out = nph - bg
        stream_ids = self.origin.detdef.get_stream_ids(phsel)
        if not self.param.params['single'] or len(stream_ids) == 1:
            self._add_column('nph_bg', (phsel, starttype, stoptype), out)
        return out

    @classmethod
    def _get_nph_bg_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """Title getter function for nph_bg"""
        return _get_nph_title(col, '^{ii}I', include_unit, origin)

    @classmethod
    def _regularizecolumn_brightness_bg(cls, source_param:Param, *args):
        """Column regularization function for brightness_bg function"""
        return args[0:1] +  cls._regularize_column_startstop(source_param, *args[1:])

    def _get_brightness_bg(self, phsel:PhSel, starttype:str, stoptype:str)->np.ndarray[np.double]:
        """Getter function for brightness_bg column"""
        return _calc_brightness(self, self.parents['base'], 'nph_bg', phsel, starttype, stoptype)

    @classmethod
    def _get_brightness_bg_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """Title getter function for brightness_bg column"""
        return _get_brightness_title(col, '_{bg}br', include_unit, origin)

    @classmethod
    def _regularizecolumn_ratio_bg(cls, source_param:Param, *args):
        """Column regularization function for ratio_bg column"""
        return args[0:2] +  cls._regularize_column_startstop(source_param, *args[2:])

    def _iter_ratio_bg(self, num_phsel:PhSel, dem_phsel:PhSel, 
                         starttype:ColKeyStart, stoptype:ColKeyStop)->Iterator[float]:
        """Iter function for ratio_bg column"""
        yield from _iter_ratio(self, 'nph_bg', num_phsel, dem_phsel, starttype, stoptype)

    def _get_ratio_bg(self, num_phsel:PhSel, dem_phsel:PhSel, 
                         starttype:ColKeyStart, stoptype:ColKeyStop)->Sequence[float]:
        """Getter function for ratio_bg column"""
        
        return _calc_ratio(self, 'nph_bg', num_phsel, dem_phsel, starttype, stoptype)

    @classmethod
    def _get_ratio_bg_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """Title getter function for ratio_bg column"""
        title = '%s/%s' %  _title_sels('^{ii}I', origin, *col.keytup[:2])
        title = _title_startstop_append(title, col.keytup[2], col.keytup[3])
        return f'${title}$'

    def _iter_anisotropy_bg(self, phsel_p:PhSel, phsel_s:PhSel, starttype:str, stoptype:str)->Iterator[float]:
        """Iter function for anisotropy_bg column"""
        yield from _iter_anisotropy(self, 'nph_bg', phsel_p, phsel_s, starttype, stoptype)

    def _get_anisotropy_bg(self, phsel_p:PhSel, phsel_s:PhSel, starttype:str, stoptype:str)->np.ndarray[np.float64]:
        """Getter function for anisotropy_bg column"""
        return _calc_anisotropy(self, 'nph_bg', phsel_p, phsel_s, starttype, stoptype)

    @classmethod
    def _get_anisotropy_bg_title(cls, col:Column, include_unit:Real=False, origin:PhotonData=None)->str:
        """Title getter function for anisotropy_raw column"""
        kw = {'name':'^{ii}I'}
        par, perp, start, stop = col.keytup
        fuse = par | perp
        overlap = par | perp
        detdef = None
        if origin is not None:
            kw.update(detdef=origin.detdef, stream_names=origin.get_stream_names())
            fuse = fuse.render_positive(origin.detdef, convert_all=True)
            overlap = overlap.render_positive(origin.detdef, convert_all=True)
            detdef = origin.detdef
        if not overlap and _pol_ps(fuse, detdef, None if origin is None else origin.setup):
            kw['name'] = 'r'
            title = fuse.tex_str(kw)
        else:
            title = rf'anis({par.tex_str(**kw)},\: {perp.tex_str(**kw)})'
        return title

    @classmethod
    def _replace_E_bg(cls, col:str, keytup:tuple[str,str])->tuple:
        """Column re-mapping function for E_bg"""
        return 'ratio_bg', (PhSel('0ex1em'), PhSel('0ex'),)+keytup, {'title':'^{ii}E_{app}'}

    @classmethod
    def _replace_S_bg(cls, col:str, keytup:tuple[str,str])->tuple:
        """Column re-mapping function for S_bg"""
        return 'ratio_bg', (PhSel('0ex'), PhSel('0ex_1ex1em'),)+keytup, {'title':'^{ii}S_{app}'}

    @classmethod
    def _regularizecolumn_ES_bg(cls, source_param:Param, *args:str)->tuple[str, str]:
        """Mapped Column regularization function fro E/S_bg"""
        return cls._regularize_column_startstop(source_param, *args)

    def _iter_sbr(self, phsel:PhSel, starttype:ColKeyStart, stoptype:ColKeyStop)->float:
        """Iter function for sbr column"""
        for nph, bg in zip(self.parents['base'].iter_column('nph_raw', phsel),
                           self.parents['bg'].iter_column('rangecounts', 
                                                          self.parents['base'].param, 
                                                          phsel, starttype, stoptype)):
            yield nph / bg

    def _get_sbr(self, phsel:PhSel, starttype:ColKeyStart, stoptype:ColKeyStop)->float:
        """Getter function for sbr column"""
        nph = self.parents['base']['nph_raw', phsel]
        bg = self.parents['bg']['rangecounts', self.param.parents['base'], phsel, starttype, stoptype]
        return nph / bg

    @classmethod
    def _get_sbr_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """Title getter for sbr column"""
        title = _title_sels('sbr', origin, col.keytup[0])[0]
        title = _title_startstop_append(title, col.keytup[1], col.keytup[2])
        return f'${title}$'
    
    def _iter_nanomean_bg(self, phsel:PhSel, mean:Literal['irf','thresh'], starttype:str, stoptype:str)->float:
        """Iter func for nanomean corrected for bg"""
        phsel = phsel.render_positive(self.origin.detdef)
        stream_ids = self.origin.detdef.get_stream_ids(phsel)
        phsels = tuple(self.origin.detdef.stream_ids_to_PhSel(sid) for sid in stream_ids)
        irf = self.origin.irf_thresh if mean == 'thresh' else self.origin.irf
        tcspc_units, irf_means, bg_means = _extract_nmarrays(stream_ids, self.origin.setup, irf)
        base, bg = self.parents['base'], self.parents['bg']
        if stream_ids.size == 1:
            tcspc_unit = tcspc_units[0]
            irf_mean = irf_means[0]
            bg_mean = bg_means[0]
            for nanos, bgcnt in zip(base.iter_column('ph_nanos', phsel), 
                                    bg.iter_column('rangecounts', base.param, 
                                                   phsel, starttype, stoptype)):
                nanosum = np.sum(nanos, dtype=np.float64)-nanos.size*irf_mean - bgcnt*bg_mean
                nanocnts = nanos.size - bgcnt
                yield tcspc_unit*nanosum/nanocnts
        else:
            for nanos, dets, *bgcnts in zip(base.iter_column('ph_nanos', phsel), 
                                            base.iter_column('ph_dets', phsel),
                                            *(bg.iter_column('rangecounts', base.param, 
                                                             sel, starttype, stoptype) 
                                              for sel in phsels)):
                nanosum, nanocnts = 0.0, 0.0
                for i, bgcnt in enumerate(bgcnts):
                    mask = dets == stream_ids[i]
                    mask_size = mask.sum()
                    nanosum += np.sum(nanos[mask], dtype=np.float64)-mask_size*irf_means[i] - bgcnt*bg_means[i]
                    nanocnts += mask_size - bgcnt
                yield tcspc_unit*nanosum/nanocnts

    @classmethod
    def _get_nanomean_bg_title(cls, col:Column, include_unit:Real|bool=False, origin:PhotonData=None)->str:
        """Nanomean corrected for background title func"""
        title = _title_sels(r'\bar{_{bg}\tau}', origin, col.keytup[0])[0]
        title = _title_unit_append(title, 's', include_unit)
        return f'${title}$'
    
    @classmethod
    def _regularizecolumn_nanomean_bg(cls, source_param:Param, *args)->tuple[PhSel,Literal['irf','thresh'],str,str]:
        phsel = [arg for arg in args if isinstance(arg, (PhSel, PhStream))]
        if len(phsel) != 1:
            raise ValueError("Only 1 PhSel may be specified in nanomean_bg column")
        phsel = phsel[0] if isinstance(phsel[0], PhSel) else PhSel(phsel[0])
        mean = [arg for arg in args if arg in ('irf', 'thresh')]
        if len(mean) > 1:
            raise ValueError("multiple definitions for mean type in nanomean_bg column")
        mean = mean[0] if mean else 'irf'
        startstop = tuple(arg for arg in args if not isinstance(arg, (PhSel, PhStream)) and arg not in ('irf', 'thresh'))
        starttype, stoptype = cls._regularize_column_startstop(source_param, *startstop)
        return phsel, mean, starttype, stoptype
        

def _index_broadcast_2dto2d(ndim:int, nmat:int, i:int)->tuple[slice|np.newaxis,...]:
    """
    Create slice/newaxis tuple to broadcast a the i-th ndim array of nmat
    into a ndim*nmat dimensional array.

    Parameters
    ----------
    ndim : int
        Number of dims in each matrix.
    nmat : int
        Number of matrices being combines.
    i : int
        place in sequence.

    Returns
    -------
    tuple[slice|np.newaxis,...]
        tuple to reshape array to allow broadcasting to final array.

    """
    return tuple(chain(*repeat(tuple(slice(None) if i == j else np.newaxis for j in range(nmat)), ndim)))


def _broadcast_2dto2d(*args:np.ndarray)->np.ndarray:
    """
    Broadcast arbitrary number of nd-arrays into a single nd-array.

    Parameters
    ----------
    *args : np.ndarray
        nd-array (all arrays should have same ndim).

    Raises
    ------
    ValueError
        Arrays have different number of dimensions.

    Returns
    -------
    np.ndarray
        Broadcasted array.

    """
    if any(args[0].ndim != arg.ndim for arg in args[1:]):
        raise ValueError("all arrays must have same number of dimensions")
    out = np.ones(tuple(chain(*zip(*(arg.shape for arg in args)))), dtype=args[0].dtype)
    nmat = len(args)
    for i, arg in enumerate(args):
        out *= arg[_index_broadcast_2dto2d(arg.ndim, nmat, i)]
    return out.reshape(tuple(np.prod([arg.shape[i] for arg in args]) for i in range(args[0].ndim)))


class Ratios(ChildPhotonTable):
    r"""
    Table for fully correct ratios between different photon streams.
    
    Params
    ------
        corr_mat : np.ndarray[np.double]
            correction matrix used to compute corrected streams
            :math:`\mathbf{M}\vec{^{bg}n}` where :math:`\vec{^{bg}n}` is the
            background correcte intensity of each stream.

    Remapped Params
    ---------------
        scheme : str
            One of '1ex', 'ALEX', 'PAM'. Default is 'ALEX'
        alpha : float
            leakage factor- remaps to lk. Default is 0.0.
        lk : float
            leakage factor. Default is 0.0.
        delta : float
            direct excitation factor, remaps to dir_ex. Default is 0.0.
        dir_ex : float
            direct excitation factor. Default is 0.0.
        gamma : float
            gamma correction factor for donor/acceptor emmission sensitivity.
            Default is 1.0.
        beta : float
            beta correction factor for donor/acceptor excitation sensitivity.
            Default is 1.0.
    
    Parents
    -------
        nph : Nph_bg
            Defines the base of the Ratios table, and background corrected values
    
    Columns
    -------
        nph_c : float, (ph_sel:PhSel, starttype:str, stoptype:str)
            Corrected (according to correction factors and background) number of photons
            in ph_sel.
        brightness_c : float, (ph_sel:PhSel, starttype:str, stoptype:str)
            counts per second in given stream, with all correction factors applied
        ratio_c : float, (num_ph_sel:PhSel, dem_ph_sel:PhSel, starttype:str, stoptype:str)
            ratio of num_ph_sel to dem_ph_sel, with all correction factors applied
    
    Re-mapped Columns
    -----------------
        E : (starttype:str, stoptype:str)
            Convenience column returns presumed FRET efficiency
            Remaped column of ratio_c, give ratio of PhSel('0ex1em') to PhSel('0ex')
        S : (starttype:str, stoptype:str)
            Convenience column return presumed Stoichiometry
            Remaped column of ratio_c, give ratio of PhSel('1ex1em') to 
            PhSel('0ex_1ex1em')

    """
    #: :meta private:
    param_defs = (
        ParamDef('corr_mat', TV_ndarray(square=True, dims=arr_slc[:,:])),
                  )
    #: :meta private:
    parent_defs = (
        ParentDef('nph', NphBG, is_base=True),
                   )
    #: :meta private:
    column_defs = (
        ColumnDef('nph_c', (PhSel, TV_str, TV_str), 0, 'never', get_func='_get_nph_c', 
                  reg_func='_regularizecolumn_nph_c', title_func='_get_nph_c_title',
                  unit='cnts s^{-1}', index_unit='cnts s-1', title_is_tex=True),
        ColumnDef('brightness_c', (PhSel, TV_str, TV_str), 0, 'never', get_func='_get_brightness_c', 
                  reg_func='_regularizecolumn_brightness_c', title_func='_get_brightness_c_title',
                  unit='cnts s^{-1}', index_unit='cnts s-1', title_is_tex=True),
        ColumnDef('ratio_c', (PhSel, PhSel, TV_str, TV_str), 0, 'user', get_func='_get_ratio_c', 
                  reg_func='_regularizecolumn_ratio_c', title_func='_get_ratio_c_title'),
        ColumnDef('anisotropy_c', (PhSel, PhSel, TV_str, TV_str), 0, 'user', 
                  get_func='_get_anisotropy_c', reg_func='_regularizecolumn_anisotropy_c',
                  title_func='_get_anisotropy_c_title'),
        ColumnDef('E', (TV_str, TV_str), 0, remap='_replace_E', reg_func='_regularizecolumn_ES'),
        ColumnDef('S', (TV_str, TV_str), 0, remap='_replace_S', reg_func='_regularizecolumn_ES'),
                   )
    _fret_factors = ('alpha', 'lk', 'gamma', 'delta', 'dir_ex', 'beta', 
                     'scheme', 'npol', 'nsplit', 'matchstreams')
    _alternating_factors = ('dir_ex', 'beta')
    # scheme can be '1ex', 'ALEX', or 'PAM' 
    # all more complicated schemes require directly specifying corr_mat
    # matchstreams specifies how to spread over pol and split parameters. 
    # if True, then correction factor applies only to "matched" stream,
    # ie lk correction factor (DexDem->DexAem) applies 
    # (DexDemPpol->DexAemPpol) and (DexDemSpol->DexAemSpol), 
    # while (DexDemSpol->DexAemPpol) and (DexDemPpol->DexAemSpol) are 0
    # if false, then apply correction from DexDemPpol+DexDemSpol and apply average
    # to both DexAemPpol and DexAemSpol
    
    def __init_columns__(self):
        pass
    
    @classmethod
    def param_preprocess(cls, param:Sequence[tuple[str,Any]]|tupledict, parents:dict[str:Param])->tuple[dict,dict]:
        """
        Preprocess, not called by user. Sorts inputs using different formats
        and converts into consistent corr_mat approach for correction factors
        :meta private:
        """
        param = as_paramdict(param, tuple(pdef.name for pdef in cls.param_defs)+cls._fret_factors)
        if isinstance(parents, Param):
            parents = {'nph':parents}
        elif isinstance(parents, tupledict):
            parents = parents.asdict
        if 'corr_mat' in param:
            if len(param) != 1:
                raise ValueError("Specifying corr_mat not compatible with building from other factors")
            return param, parents
        if any(cfactor in param for cfactor in ('alpha', 'delta', 'lk', 'dir_ex', 'gamma', 'beta')):
            add_citation('HellenkampNatMeth2018', purpose='Use of alpha/delta/gamma/beta formalism')
        scheme = param.get('scheme', 'ALEX')
        matchstreams = param.get('matchstreams', True) # match streams defines how split/pol leakage/direx/beta/gamma are broadcast
        npol = param.get('npol', 1)
        nsplit = param.get('nsplit', 1)
        corr_mat = np.eye(2 if scheme == '1ex' else 4)
        lk = param.get('alpha', param.get('lk', 0.0))
        dir_ex = param.get('delta', param.get('dir_ex', 0.0))
        gamma, beta = param.get('gamma', 1.0), param.get('beta', 1.0)
        if scheme == 'ALEX':
            corr_mat[0,0] = gamma
            corr_mat[1,0] = -lk*gamma
            corr_mat[3,3] = 1/beta
            corr_mat[1,3] = -dir_ex
        elif scheme == '1ex':
            if any((err:=cf) in param for cf in cls._alternating_factors):
                raise ValueError(f"'{err}' only applies to setups with alternating excitation")
            corr_mat[0,0] = gamma
            corr_mat[1,0] = -lk
        elif scheme == 'PAX':
            corr_mat[0,0] = gamma
            corr_mat[1,0] = -lk
            corr_mat[3,3] = 1/beta
            corr_mat[3,1] = -1/beta
            corr_mat[1,1] += dir_ex
            corr_mat[1,3] = -dir_ex
        else:
            raise ValueError(f"scheme must be '1ex', 'ALEX' or 'PAX'. scheme of '{scheme}' is invalid")
        if npol == 1 and nsplit == 1:
            return dict(corr_mat=corr_mat), parents
        nschan = npol*nsplit
        if matchstreams:
            new_corr_mat = _broadcast_2dto2d(corr_mat, np.eye(nschan))
        else:
            mask = _broadcast_2dto2d(np.eye(corr_mat.shape[0], dtype=np.bool_), np.ones((nschan, nschan), dtype=np.bool_))
            diag_mat = _broadcast_2dto2d(corr_mat, np.eye(nschan))
            new_corr_mat = _broadcast_2dto2d(corr_mat, np.ones((nschan, nschan))/nschan)
            new_corr_mat[mask] = diag_mat[mask]
        return dict(corr_mat=new_corr_mat), parents

    @classmethod
    def validate_param(cls, param:Param):
        """Not usually called by user- validate a Ratios :class:`Param` :meta private:"""
        if param.detdef.size != param.params['corr_mat'].shape[0]:
            raise ValueError("corr_mat must have both dimensions of size equal to the number of streams in detdef")

    @classmethod
    def _regularizecolumn_nph_c(cls, source_param:Param, *args):
        """Column regularization function for nph_c column"""
        return args[0:1] +  cls._regularize_column_startstop(source_param, *args[1:])

    def _get_nph_c(self, phsel:PhSel, starttype:str, stoptype:str)->Iterator[float]:
        stream_ids = self.origin.detdef.get_stream_ids(phsel)
        if stream_ids.size == 1:
            corr_row = self.param.params['corr_mat'][stream_ids[0], :]
            mask = corr_row != 0.0
            source_ids = np.argwhere(mask)
            corr_row = corr_row[mask]
            smat = np.array([self.parents['nph']['nph_bg', 
                                                    self.origin.detdef.stream_ids_to_PhSel(i), 
                                                    starttype, stoptype] 
                             for i in source_ids]).T
            return np.sum(corr_row*smat, axis=1)
        else:
            return np.sum([self['nph_c',self.origin.detdef.stream_ids_to_PhSel(i), starttype, stoptype] 
                           for i in stream_ids], axis=0)

    @classmethod
    def _get_nph_c_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """Title getter function for nph_c column"""
        return _get_nph_title(col, 'F', include_unit, origin)

    @classmethod
    def _regularizecolumn_brightness_c(cls, source_param:Param, *args):
        """Column regularization function for brightness_c column"""
        return args[0:1] + cls._regularize_column_startstop(source_param, *args[1:])

    def _get_brightness_c(self, phsel:PhSel, starttype:str, stoptype:str)->np.ndarray[np.double]:
        """Getter function for brightness_c column"""
        return _calc_brightness(self, self.parents['nph'].parents['base'], 'nph_c', phsel, starttype, stoptype)

    @classmethod
    def _get_brightness_c_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """Title getter function for brightness_c column"""
        return _get_brightness_title(col, '_{c}br', include_unit, origin)

    @classmethod
    def _regularizecolumn_ratio_c(cls, source_param:Param, *args):
        """Column regularization function for ratio_c column"""
        return args[0:2] +  cls._regularize_column_startstop(source_param, *args[2:])

    def _get_ratio_c(self, num_phsel:PhSel, dem_phsel:PhSel, starttype:str, stoptype:str)->np.ndarray[np.float64]:
        """Getter function for ratio_c column"""
        return _calc_ratio(self, 'nph_c', num_phsel, dem_phsel, starttype, stoptype)

    @classmethod
    def _get_ratio_c_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """Title getter function for ratio_c"""
        return _get_ratio_title(col, 'F', include_unit, origin)

    @classmethod
    def _regularizecolumn_anisotropy_c(cls, source_param:Param, *args):
        """Column regularization function for anisotropy_c column"""
        return args[0:2] +  cls._regularize_column_startstop(source_param, *args[2:])

    def _get_anisotropy_c(self, phsel_p:PhSel, phsel_s:PhSel, starttype:str, stoptype:str)->np.ndarray[np.float64]:
        """Getter function for anisotropy_c column"""
        return _calc_anisotropy(self, 'nph_c', phsel_p, phsel_s, starttype, stoptype)

    @classmethod
    def _get_anisotropy_c_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """Title getter function for anisotropy_c column"""
        return _get_anisotropy_title(col, 'F', include_unit, origin)

    @classmethod
    def _regularizecolumn_ES(cls, source_param:Param, *args:str)->tuple[str, str]:
        """Column regularization function for re-mapped columns E/S"""
        return cls._regularize_column_startstop(source_param, *args)

    @classmethod
    def _replace_E(cls, col:str, keytup:tuple[str,str])->tuple:
        """Column re-mapping function for E column, maps to ratio_c"""
        return 'ratio_c', (PhSel('0ex1em'), PhSel('0ex'),)+keytup, {'title':'E'}

    @classmethod
    def _replace_S(cls, col:str, keytup:tuple[str,str])->tuple:
        """Column re-mapping function for S column, maps to ratio_c"""
        return 'ratio_c', (PhSel('0ex'), PhSel('0ex_1ex1em'),)+keytup, {'title':'S'}

    def dither(self, phsel:PhSel, lsb:float=2.0, generator:np.random.Generator=None)->np.ndarray[np.double]:
        """
        Dither number of photons in phsel. Returns dithered array

        Parameters
        ----------
        phsel : PhSel
            photon stream to dither.
        lsb : float, optional
            Magnitued of dithering. Evaluated as follows:
            :math:`0.5 rnd n` where :math:`rnd` is random number in inteval ``[-1,1]``
            and :math:`n` is number of photons in stream.
            The default is 2.0.
        generator : np.random.Generator, optional
            Random number generator to use. The default is None.

        Returns
        -------
        np.ndarray[np.double]
            Version of nph_c dithered with lsb.

        """
        if not isinstance(generator, np.random.Generator):
            generator = np.random.default_rng(generator)
        return self['nph_c', phsel] + lsb*(generator.random(self.size) - 0.5)

    def dither_ratio(self, phsel_num:PhSel, phsel_dem:PhSel, 
                     lsb_num:float=2.0, lsb_dem:float=2.0,
                     generator:np.random.Generator=None)->np.ndarray[np.double]:
        """
        Dithered ratio of photons in phsel_num to phsel_dem


        Parameters
        ----------
        phsel_num : PhSel
            photon stream of numerator of ratio.
        phsel_dem : PhSel
            photon stream of denomenator of ratio.
        lsb_num : float, optional
            Magnitued of dithering in numerator stream. Evaluated as follows:
            :math:`0.5 rnd n` where :math:`rnd` is random number in inteval ``[-1,1]``
            and :math:`n` is number of photons in stream.The default is 2.0.
        lsb_dem : float, optional
            Magnitude of dithering in denominator stream, same convention as lsb_num. 
            The default is 2.0.
        generator : np.random.Generator, optional
            Random number generator to use. The default is None.

        Returns
        -------
        np.ndarray[np.double]
            Dithered ratio array.

        """
        if not isinstance(generator, np.random.Generator):
            generator = np.random.default_rng(generator)
        return self.dither(phsel_num, lsb_num, generator) / self.dither(phsel_num, lsb_num, generator)


###############################################################################
######################### Experimental CDE functions  #########################
###############################################################################
def register_2cde_func(func:Callable[[int,int,float],float], shortcut:Callable=None)->None:
    """
    Add new function for computing KDE function to TypeValidator code.

    Parameters
    ----------
    func : Callable[[int,int,float],float]
        function of signature ``func(tloc:ing, tphoton:int, tau:float)->float``
        Where tloc is time where KDE is evaluated, tphoton is time of a photon,
        and tau is a floating point constant, must return float.
    shortcut : Callable, optional
        If faster method available, that takes signature
        ``func(times:np.ndarray, tau:float, lim:float=None, locs:np.ndarray=None)->np.ndarray``. 
        The default is None.

    """
    if shortcut is not None:
        register_PyCode(func, 'KDE_func', shortcut)
    else:
        register_PyCode(func, 'KDE_func', partial(smc.kde_photons_user, func=func))

    
def laplace_kde_2cde(tl:int, tp:int, tau:float)->float:
    """
    Kernel function of laplace KDE

    Parameters
    ----------
    tl : int
        Location (time) where to evaluate laplace KDE.
    tp : int
        Time of single photon in laplace KDE.
    tau : float
        Time constant of laplace.

    Returns
    -------
    float
        Contribution of tp to tl based on laplace KDE with tau.

    """
    return np.exp(-abs(tl-tp) / tau)

laplace_kde_2cde.name = r'\mathcal{L}'
laplace_kde_2cde.factor = 5.0
register_2cde_func(laplace_kde_2cde, shortcut=0)


def gaussian_kde_2cde(tl:int, tp:int, tau:float)->float:
    """
    Kernel function of gaussian KDE

    Parameters
    ----------
    tl : int
        Location (time) where to evaluate gaussina KDE.
    tp : int
        Time of single photon in gaussin KDE.
    tau : float
        Time constant of laplace.

    Returns
    -------
    float
        Contribution of tp to tl based on gaussian KDE with tau.

    """
    return np.exp(-(tl-tp)**2 / (2*(tau**2)))

gaussian_kde_2cde.name = r'\mathcal{L}'
gaussian_kde_2cde.factor = 3.0
register_2cde_func(gaussian_kde_2cde, shortcut=1)


class KDE(ChildPhotonTable):
    r"""
    Implementation of FRET and ALEX 2CDE methods from Tomov_.
    
    This method estimates the probability that there is variance in the expected
    emission probabilities as a molecule transits the confocal value
    (within burst dynamics) by comparing the ratio of kernel density estimator 
    values of one stream vs another.
    
    The basic equation of a KDE is
    
    .. math::
        
        KDE_{X_{i}}^{Y} \left(t_{(CHX)_{i}}, t_{\{CHY\}} \right) = 
        \sum_j^{N_{CHY}} \exp \left( - \frac{\lvert t_{(CHX)_i} - t_{(CHY)_j} \rvert}{\tau}\right)
    
    
    where :math:`X are the points where the KDE is esimated, and :math:`Y` are
    the points contributing to the KDE. Tomov_ set both of these to arrival times
    of particular streams. When the streams of :math:`X` and :math:`Y` are the
    same, they introduced a modified KDE, which does not count the photon at
    the location 
    
    .. math::
        
        nbKDE_{X_i}^X \left(t_{\{CHX\}} \right) = \left(1 + \frac{2}{N_{CHX}} \right) \cdot
        \sum_{j, \;j\ne i}^{N_{CHX}} \exp \left( - \frac{\lvert t_{(CHX)_i} - t_{(CHX)_j} \rvert}{\tau}\right)
    
    .. note::
        
        The main text of Tomov_ describes :math:`N_{CHX}` ambiguously as the 
        number of photons in channel :math:`X`, but fails to define if this is
        with a burst, or fixed time range. Several interpretations are possible,
        and the paper actively admits that they arrived at the :math:`1+2/N_{CHX}`
        correction factor through trial and error. They provide in the supplementary
        material screenshots of their labview code, which would suggest that
        :math:`N_{CHX}` is the number of photons in a given burst. Therefore
        this is what is impolemented here.
    
    
    The KDE values are used to define the following ratiometric values
    
    .. math::
        
        (E)_D = \frac{1}{N_{CHD}} \sum_{i=1}^{N_{CHD}} \frac{KDE_{Di}^A}{KDE_{Di}^A + nbKDE_{Di}^D}
    
    
    and 
    
    .. math::
        
        (1 - E)_A = \frac{1}{N_{CHA}} \sum_{i=1}^{N_{CHA}} \frac{KDE_{Ai}^D}{KDE_{Ai}^D + nbKDE_{Ai}^A}
        
    
    Which generate the :math:`FRET-2CDE` parameter (``fret`` column)
    
    .. math::
        
        FRET-2CDE \left( t_{CHD}, t_{CHA} \right) = 110 - 100 \cdot \left[ (E)_D + (1 - E)_A \right]
    
    and, for assesing blinking
    
    .. math::
        
        BR_{D_{EX}} = \frac{1}{N_{CHA_{EX}}} 
        \sum_{i=1}^{N_{CHD_{EX}}}{\frac{KDE_{D_{EX^{i}}}^{A_EX}}{KDE_{D_{EX^{i}}}^{D_EX}}}
    
        
    and 
    
    .. math::
        
        BR_{A_{EX}} = \frac{1}{N_{CHD_{EX}}} 
        \sum_{j=1}^{N_{CHA_{EX}}}{\frac{KDE_{A_{EX^{j}}}^{D_EX}}{KDE_{A_{EX^{j}}}^{A_EX}}}
        
    
    which generates the :math:`ALEX-2CDE` parameter (``alex`` column)
    
    .. math::
        
        ALEX-2CDE \left( t_{CHD}, t_{CHA} \right) = 
        100 - 50 \left[ BR_{D_{EX}} + BR_{A_{EX}} \right]
    
    
    Params
    ------
        kernel : Callable[[int, int, float], float]
            Kenel function (funtion inside sumation of :math:`KDE` or :math:`nbKDE`)
            Must take 3 arguments, in order time where KDE is being evaluated,
            then the time of a photon generating the kernel, and finally a float,
            the time constant of the kernel.
        tau : float
            Lifetime (:math:`\tau`) of kernel, in seconds
        thresh : float
            Maximum time range to evaluate photons in kernel, in seconds.
    
    Parents
    -------
        base : BasePhotonTable
            The time ranges to evaluate KDE values
    
    Columns
    -------
        fret : float (phsel_d:PhSel, phsel_aPhSel)
            Evaluate :math:`FRET-2CDE` where :math:`t_{CHD}` is ``phsel_d``, and
            :math:`t_{CHA}` is ``phsel_a``
        
        Alex : float (phsel_d:PhSel, phsel_aPhSel)
            Evaluate :math:`ALEX-2CDE` where :math:`t_{CHD_{EX}}` is ``phsel_d``, and
            :math:`t_{CHA_{EX}}` is ``phsel_a``
    
    
    .. _Tomov: `Tomov 2012 <https://doi.org/10.1016/j.bpj.2011.11.4025>`__
    
    """
    #: :meta private:
    param_defs = (
        ParamDef('kernel', TV_PyCode, default=laplace_kde_2cde),
        ParamDef('tau', TV_float(mn=0.0), default=5e-4),
        ParamDef('thresh', TV_float(mn=0.0))
        )
    #: :meta private:
    parent_defs = (ParentDef('base', BasePhotonTableLike, is_base=True), )
    #: :meta private:
    column_defs = (
        ColumnDef('fret', (PhSel, PhSel), 0, 'user', iter_func='_iter_fret', 
                  reg_func='_regularizecolumn_fret', title_func='_get_fret_title'),
        ColumnDef('alex', (PhSel, PhSel), 0, 'user', iter_func='_iter_alex', 
                  reg_func='_regularizecolumn_alex', title_func='_get_alex_title')
        )
    
    def __init_columns__(self):
        pass
    
    @classmethod
    def param_preprocess(cls, param:Sequence[tuple[str,Any]]|tupledict, parents:dict[str:Param])->tuple[dict,dict]:
        """
        Preprocess, not called by user. Sorts inputs using different formats
        and converts into consistent corr_mat approach for correction factors
        :meta private:
        """
        param = as_paramdict(param, tuple(pdef.name for pdef in cls.param_defs))
        if isinstance(parents, Param):
            parents = {'base':parents}
        elif isinstance(parents, tupledict):
            parents = parents.asdict
        param.setdefault('kernel', laplace_kde_2cde)
        param.setdefault('tau', 5e-4)
        if 'thresh' not in param:
            factor = param['kernel'].factor if hasattr(param['kernel'], 'factor') else 5.0
            param['thresh'] = factor * param['tau']
        return param, parents
        

    @paramproperty
    def kde_func(cls, param:Param)->Callable[[np.ndarray[np.int64],float,np.ndarray[np.float64]],np.ndarray[np.float64]]:
        func = get_pycode_subval('KDE_func', param.params['kernel'], param.params['kernel'])
        return partial(smc.kde_photons, func=func)

    def _index_iter(self, phsel_a:PhSel, phsel_b:PhSel, drop_self:bool
                    )->tuple[np.ndarray[np.float64],np.ndarray[np.float64],np.ndarray[np.float64],np.ndarray[np.float64]]:
        sela = self.origin.detdef.get_stream_ids(phsel_a)
        selb = self.origin.detdef.get_stream_ids(phsel_b)
        mask_a = np.isin(self.origin.dets, sela)
        times_a = self.origin.times[mask_a]
        mask_b = np.isin(self.origin.dets, selb)
        times_b = self.origin.times[mask_b]
        func = self.kde_func
        tau = self.param.params['tau'] / self.origin.clk_p
        kde_aa = func(times_a, tau, drop_self=drop_self)
        kde_ab = func(times_a, tau, times_b)
        kde_ba = func(times_b, tau, times_a)
        kde_bb = func(times_b, tau, drop_self=drop_self)
        idx_a = np.cumsum(mask_a)
        idx_b = np.cumsum(mask_b)
        for istart, istop in zip(self.parents['base']['istart'], self.parents['base']['istop']):
            slc = slice(istart, istop)
            ma = idx_a[slc][mask_a[slc]]
            mb = idx_b[slc][mask_b[slc]]
            yield kde_aa[ma], kde_ab[mb], kde_ba[ma], kde_bb[mb]

    @classmethod    
    def _get_kde_title(cls, title:str, col:Column, origin:PhotonData=None):
        if hasattr(col.param.params['kernel'], 'name'):
            title += '_{%s}' % col.param.params['kernel'].name
        else:
            title += '_{%s}' % col.param.params['kernel'].__name__
        title = '%s(%s/%s)' % ((title, ) + _title_sels('t', origin, *col.keytup))
        return f'${title}$'
    
    @classmethod
    def _get_fret_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        return cls._get_kde_title('FRET-2CDE', col, origin=origin)

    @classmethod
    def _regularizecolumn_fret(cls, source_param:Param, *args):
        phsel_d, phsel_a, = args[0:1], args[1:2]
        phsel_d = PhSel('0ex0em') if len(phsel_d) == 0 else phsel_d[0]
        phsel_a = PhSel('0ex1em') if len(phsel_a) == 0 else phsel_a[0]
        phsel_d, phsel_a = sort_phsels((phsel_d, phsel_a))
        return phsel_d, phsel_a
    
    @cite('TomovBioPhysJ2012', purpose='FRET 2CDE')
    def _iter_fret(self, phsel_d:PhSel, phsel_a:PhSel)->float:
        for kde_dd, kde_da, kde_ad, kde_aa in self._index_iter(phsel_d, phsel_a, True):
            if kde_dd.size + kde_aa.size == 0:
                yield np.nan
                continue
            with np.errstate(invalid='ignore', divide='ignore'):
                if kde_dd.size != 0:
                    e_d = np.mean(kde_ad / (kde_ad + (1.0 + 2.0/kde_dd.size)*kde_dd))
                else:
                    e_d = 0.0
                if kde_aa.size != 0:
                    e_a = np.mean(kde_da / (kde_da + (1.0 + 2.0/kde_aa.size)*kde_aa))
                else:
                    e_a = 0.0
            yield 110.0 - 100.0*(e_d+e_a)
    
    @classmethod
    def _regularizecolumn_alex(cls, source_param:Param, *args):
        phsel_a, phsel_d, = args[0:1], args[1:2]
        phsel_a = PhSel('1ex1em') if len(phsel_a) == 0 else phsel_a[0]
        phsel_d = PhSel('0ex') if len(phsel_d) == 0 else phsel_d[0]
        phsel_a, phsel_d = sort_phsels((phsel_a, phsel_d))
        return phsel_a, phsel_d

    @classmethod
    def _get_alex_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        return cls._get_kde_title('ALEX-2CDE', col, origin=origin)


    @cite('TomovBioPhysJ2012', purpose='ALEX 2CDE')
    def _iter_alex(self, phsel_d:PhSel, phsel_a:PhSel)->float:
        for kde_dd, kde_da, kde_ad, kde_aa in self._index_iter(phsel_d, phsel_a, False):
            if kde_dd.size == 0 or kde_aa.size == 0:
                yield np.nan
                continue
            with np.errstate(invalid='ignore', divide='ignore'):
                br_d = kde_ad / kde_dd / kde_aa.size
                br_a = kde_da / kde_aa / kde_dd.size
            yield 100.0 - 50.0*(br_d+br_a)

