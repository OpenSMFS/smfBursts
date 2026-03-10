#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created : 9/10/20025
# Author: Paul David Harris
# email: harrip@gmail.com
"""
Module defines base :class:`Burst` table that represents bursts data based on
sinlge or multi-channel burst search, and child-tables to compute burst-based
parameters.
"""
import os
from typing import Any, ClassVar
from collections.abc import Iterator, Sequence, Callable
from functools import partial
from warnings import warn
from itertools import chain, repeat, permutations
from numbers import Real

import numpy as np

from .datamodel.utils import tupledict, arr_slc
from .datamodel.immutabledata import (
    register_PyCode, get_pycode_subval, TV_bool, TV_float, TV_ndarray, TV_PyCode,
    TV_tuple
                                      )
from .datamodel.tables import ParamDef, ParentDef, ColumnDef, Param, Column, as_paramdict
from .datamodel.citations import cite, add_citation
from .photondata import (
    PhotonData, PhotonTable, BasePhotonTable, ChildPhotonTable, BasePhotonTableLike, 
    _regularize_column_startstop, _regularize_ph_sel, 
    _title_sels, _title_startstop_append, _title_unit_append, _pol_ps,
    TV_str_start, TV_str_stop, make_base_column_defs,
    ColKeyStart, ColKeyStop
    )
from .background import BG
from .ph_sel import PhSel, DetDef, TV_PhSel, sort_phsels, phsel_all
from .poisson_threshold import find_optimal_T_bga

import fretbursts.cfuncs as fbc


_alloc_size:int = 512


def _get_nph_title(col:Column, name:str, include_unit:bool, origin:PhotonData)->str:
    """Sub-function to format title of nph-like columns, with "core" name of ``name`` """
    title = _title_sels('_{bg}n', origin, col.keytup[0])[0]
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


class Bursts(BasePhotonTable):
    r"""
    Table of bursts, uses sliding window burst search on arbitrary number of
    streams, and joins based on arbitrary truthatble.
    
    Params
    ------
        streams : tuple[PhSel,...]
            tuple of :class:`fretbursts.ph_sel.PhSel` objects, 1 per burst search.
            Can specify single PhSel and will automatically convert to 1-tuple.
        m : int | np.ndarray[np.int64]
            size of sliding window burst search(s). Must match size of streams.
            If specified as int, will automatically convert to streams size.
            Default is 10
        F : float | np.ndarray[np.float64]
            Threshold for burst-search (F*bg). Like `m`, match size of streams,
            and autmatically converted to correct size repeating array of value.
            Default is 6.0.
        c : float | np.ndarray[np.float64]
            Correction factor used in the rate vs time-lags relation. Like `m`,
            must match size of streams, if specified as float, will automatically
            to correct size repeating array of value. Default is -1.0.
        truthtable : np.ndarray[np.bool\_]
            `(2,)*len(streams)` shape boolean array specifying under which combinations
            of in-burst per stream to consider as part of actual burst.
            Default to 'and-gate' ie only when all streams are in-burst is the system
            considered in-burst
        fuse : float
            If separation between bursts in burst-search-gate is less than `fuse`
            (in seconds), merge bursts into one. If -1.0 no merge operation is performed,
            Therefore overlapping bursts are possible, if 0.0 overlapping bursts are
            merged, but separation between non-overlapping burts can be 0.
            Default is 0.0
        asP : np.ndarray[np.bool\_]
            If ``True`` threshold for burst detection (`F`) expressed as a
            probability that a detected bursts is not due to a Poisson
            background. Since background process is usually non-Poissonian, 
            setting asP to True is discourange. 
            Like `m`, must be same size as `streams`, and if specified as bool, 
            will automatically be converted to correct size array of all specified
            value. Default is False.
    
    Parents
    -------
        bg : tuple[Param[BG],...]
            Background computation to use, must be same length as `streams`, and like
            `m` if specified as single, automatically expanded to correct size array
            of same BG.
    
    Columns
    -------
    Uses :class:`fretbursts.photondata.BasePhotonTable` columns.
    See :any:`basephotoncolumns` for full list of columns.
    
    """
    row_name:ClassVar[str] = "Bursts"
    origin: PhotonData
    #: :meta private:
    param_defs = (
        ParamDef('streams', TV_tuple(typedefs=TV_PhSel), default=(phsel_all,)),
        ParamDef("m", TV_ndarray(mn=2, dtype=np.dtype('<i8'), dims=arr_slc[:])),
        ParamDef("F", TV_ndarray(mn=0.0, dtype=np.dtype('<f8'), dims=arr_slc[:])),
        ParamDef("c", TV_ndarray(dtype=np.dtype('<f8'), dims=arr_slc[:])),
        ParamDef('truthtable', TV_ndarray(dtype=np.dtype('|b1'), square=True, dims=arr_slc[2,...])),
        ParamDef('fuse', TV_float, default=0.0), # positive or -1.0 negative allowed.
        ParamDef('asP', TV_ndarray(dtype=np.bool_, dims=arr_slc[:])),
        # -1.0 no fuse in burst search, no fuse afterwards
        # only fuse in bursts search (equivalend to burst fuse 0)
                  )
    #: :meta private:
    parent_defs = (
        ParentDef('bg', BG, is_base=False, size_func='_param_nstreams'), 
                   )
    #: :meta private:
    column_defs = make_base_column_defs()

    @cite('slidingwindowsearch', purpose='sliding window burst search')
    def __init_columns__(self):
        bg_table = self.parents['bg']
        ddef = self.origin.setup.detdef
        ms = self.param.params['m']
        Fs = self.param.params['F']
        cs = self.param.params['c']
        asPs = self.param.params['asP']
        phsels = self.param.params['streams']
        starts, stops = list(), list()
        post_fuse = self.param.params['fuse'] > 0.0
        search_fused = self.param.params['fuse'] != -1.0
        max_sep = int(self.param.params['fuse']/self.origin.clk_p)
        starttime, stoptime = np.inf, 0.0
        for m, F, c, asP, phsel, bg in zip(ms,Fs, cs, asPs, phsels, bg_table):
            d_id = ddef.get_stream_ids(phsel)
            periods = bg.parents['base']['periods']
            if periods[0] < starttime:
                starttime = periods[0]
            if periods[-1] > stoptime:
                stoptime = periods[-1]
            if asP:
                bg = find_optimal_T_bga(bg, m, F) / self.origin.clk_p
            start, stop = fbc.burstsearch(self.origin.times, self.origin.dets,
                                            periods, bg['bg', phsel],
                                            self.origin.clk_p, d_id, m=m, F=F, c=c,
                                            fuse=search_fused, bg_is_thresh=asP,
                                            alloc_size=_alloc_size, ncore=os.cpu_count())
            starts.append(start)
            stops.append(stop)
        if ms.size != 1 or self.param.params['truthtable'][0] == True:
            add_citation('NirJPCB2006', purpose='Dual Channel Burst Search')
            starts, stops = fbc.burstgate(starts, stops, self.param.params['truthtable'], 
                                          starttime=starttime, stoptime=stoptime)
        else:
            starts, stops = starts[0], stops[0]
        if post_fuse:
            starts, stops = fbc.fusebursts(starts, stops, max_sep)
        self._add_column('start', tuple(), starts)
        self._add_column('stop', tuple(), stops)
        istart, istop = fbc.index_ranges(self.origin.times, starts, stops)
        self._add_column('istart', tuple(), istart)
        self._add_column('istop', tuple(), istop)

    @classmethod
    def _cast_param_array(cls, params:dict, nchan:int, name:str, default:Any, dtype:np.dtype)->np.ndarray:
        """Caster function to coercy array in param to correct dtype and shape"""
        arr = np.atleast_1d(params.get(name, default)).astype(dtype)
        if arr.size == 1:
            return np.repeat(arr, nchan)
        elif arr.size == nchan:
            return arr
        raise ValueError(f"{name} has {arr.size} elements, but {nchan} streams specified")

    @classmethod
    def param_preprocess(cls, params:Sequence[tuple[str,Any]]|tupledict, parents:dict[str:Param])->tuple[dict, dict]:
        """Regularize params, relicating arrays, filling defaults etc for burst search"""
        params = as_paramdict(params, tuple(pdef.name for pdef in cls.param_defs))
        # Step 1 of processing burst param: determine and sort number of streams
        streams = params.get('streams', phsel_all)
        streams = (streams, ) if isinstance(streams, PhSel) else streams
        if 'bg' not in parents:
            raise ValueError("must specify bg in parents")
        bg = parents['bg']
        bg = (bg, ) if isinstance(bg, Param) else bg
        detdef = bg[0].tp._detdef(bg[0]) # get detdef so proper conversion of PhSels to posiive can take place
        streams = _regularize_ph_sel(streams, detdef, convert_all=True) # PhSels positively defined
        nstream = len(streams)
        if len(bg) == 1:
            bg = bg * nstream
        elif len(bg) != nstream:
            raise ValueError(f"incorrect size of bg tuple, expected {nstream}, got {len(bg)}")
        m = cls._cast_param_array(params, nstream, 'm', 10, '<i8')
        F = cls._cast_param_array(params, nstream, 'F', 6.0, '<f8')
        c = cls._cast_param_array(params, nstream, 'c', -1.0, '<f8')
        asP = cls._cast_param_array(params, nstream, 'asP', False, '|b1')
        c[asP] = 0.0
        truthtable = params.get('truthtable', 'and')
        if isinstance(truthtable, str):
            if truthtable == 'and':
                truthtable = np.zeros([2 for _ in range(nstream)], dtype=np.dtype('|b1'))
                truthtable[tuple(1 for _ in range(nstream))] = True
            elif truthtable == 'or':
                truthtable = np.ones([2 for _ in range(nstream)], dtype=np.dtype('|b1'))
                truthtable[tuple(0 for _ in range(nstream))] = False
            else:
                raise ValueError("string specification of truthtable can only be 'and' or 'or', '%s' is invalid"%params['truthtable'])
        else:
            truthtable = np.asarray(truthtable).astype('|b1')
            if truthtable.ndim != nstream:
                raise ValueError("truthtable has incorrect number of dimensions")
            if any(s != 2 for s in params['truthtable'].shape):
                raise ValueError("truthtable must have size 2 along all dimensions")
        # check for duplicate stream specifications
        cont = True
        idx, slc = np.arange(2), slice(None)
        while cont:
            for i, j in permutations(range(m.size), 2):
                if all(v[i] == v[j] for v in (streams, m, F, c, asP)):
                    streams = tuple(stream for k, stream in enumerate(streams) if k != j)
                    mask = np.ones(m.size, dtype=np.bool_)
                    mask[j] = False
                    m = m[mask]
                    F = F[mask]
                    c = c[mask]
                    asP = asP[mask]
                    truthtable = truthtable[tuple(idx if k in (i,j) else slc for k in range(truthtable.ndim))]
                    break
            cont = False
        nchan = m.size
        streams, order = sort_phsels(detdef, streams, return_index=True) # ensure canonical order
        truthtable = np.moveaxis(truthtable, order, np.arange(nchan))
        # Final filling of params keys
        params['streams'] = streams
        params['m'] = m[order]
        params['F'] = F[order]
        params['c'] = c[order]
        params['asP'] = asP[order]
        params['truthtable'] = truthtable
        parents['bg'] = tuple(bg[i] for i in order)
        return params, parents

    @classmethod
    def validate_param(cls, param:Param)->None:
        """
        Check param is valid Bursts Param, used almost exclusively internally.
        User will rarely use this method :meta private:
        """
        nchan = len(param.params['streams'])
        if nchan != param.params['m'].size:
            raise ValueError(f"size of m must be {nchan}")
        if nchan != param.params['F'].size:
            raise ValueError(f"size of F must be {nchan}")
        if nchan != param.params['c'].size:
            raise ValueError(f"size of c must be {nchan}")
        if np.any(param.params['c'][param.params['asP']] != 0.0):
            raise ValueError("poisson threshhold burstsearches must have c=0.0")
        if np.any(param.params['F'][param.params['asP']] >= 1.0):
            raise ValueError("F must be < 1.0 for poisson threshold burstsearches")
        if np.any(param.params['F'][~param.params['asP']] <= 1.0):
            warn("F < 1.0 for constant threshold burstsearches will encompase most photons")

    @classmethod
    def _param_nstreams(cls, params:tupledict)->int:
        """Size func to set numbr of BG parents"""
        return len(params['streams'])

    @classmethod
    def _detdef(cls, param:Param)->DetDef:
        """Determine :class:`DetDef` of :class:`Param` based on Bursts"""
        return param.parents['bg'][0].tp._detdef(param.parents['bg'][0])


class NphBG(ChildPhotonTable):
    r"""
    Table for background corrected photon counts.
    No corrections for cross-talk and/or detection efficiencies.
    
    Params
    ------
        single : bool
            If true, compute only single streams, colums of compund PhSel computed
            as sum of streams.
    
    Parents
    -------
        base : Param[BasePhotonTable]
            Usually a :class:`Burst`, the time ranges over which rows are computed
        bg : Param[BG]
            The background counts to use.
     
    Columns
    -------
        nph_bg : (ph_sel:Ph_sel, starttype:str, stoptype:str)
            Background adjusted photon counts in ph_sel, starttype and stoptype
            define what start/stop values to use for computing burst duration and
            therefore background photon counts
        sbr : (ph_sel:PhSel, starttype:{'istarttime', 'start'}, stoptype:{'istoptime', 'stop'})
            signal to background ratio, nph_raw / bg counts
        brightness_bg : (ph_sel:PhSel, starttype:{'istarttime', 'start'}, stoptype:{'istoptime', 'stop'})
            counts per second in ph_sel, with background rate subtracted
        ratio_bg : (num_ph_sel:Ph_sel, dem_ph_sel:Ph_sel, starttype:{'istarttime', 'start'}, stoptype:{'istoptime', 'stop'})
            ratio of num_ph_sel to dem_ph_sel background adjusted counts.
    
    Remapped Columns
    ----------------
        E_bg : (starttype:{'istarttime', 'start'}, stoptype:{'istoptime', 'stop'})
            Convenience column returns presumed FRET efficiency
            Remaped column of ratio_bg, give ratio of PhSel('0ex1em') to PhSel('0ex')
        S_bg : (starttype:{'istarttime', 'start'}, stoptype:{'istoptime', 'stop'})
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
        ColumnDef('nph_bg', (PhSel,TV_str_start, TV_str_stop), 0, 'some', 
                  get_func='_get_nph_bg', iter_func='_iter_nph_bg',
                  reg_func='_regularizecolumn_nph_bg_sbr', title_func='_get_nph_bg_title',
                  unit='cnts s^{-1}', index_unit='cnts s-1', title_is_tex=True),
        ColumnDef('sbr', (PhSel, TV_str_start, TV_str_stop), 0, 'user', 
                  get_func='_get_sbr', iter_func='_iter_sbr', 
                  reg_func='_regularizecolumn_nph_bg_sbr',
                  title_func='_get_sbr_title', title_is_tex=True),
        ColumnDef('brightness_bg', (PhSel, TV_str_start, TV_str_stop), 0, 'never', 
                  get_func='_get_brightness_bg', reg_func='_regularizecolumn_brightness_bg',
                  title_func='_get_brightness_bg_title', unit='cnts s^{-1}',
                  index_unit='cnts s-1', title_is_tex=True),
        ColumnDef('ratio_bg', (PhSel, PhSel, TV_str_start, TV_str_stop), 0, 'never', 
                  get_func='_get_ratio_bg', iter_func='_iter_ratio_bg',
                  reg_func='_regularizecolumn_ratio_bg', title_func='_get_ratio_bg_title',
                  title_is_tex=True),
        ColumnDef('anisotropy_bg', (PhSel, PhSel, TV_str_start, TV_str_stop), 0, 'never', 
                  get_func='_get_anisotropy_bg', iter_func='_iter_anisotropy_bg',
                  reg_func='_regularizecolumn_ratio_bg', title_func='_get_anisotropy_bg_title',
                  title_is_tex=True),
        ColumnDef('E_bg', (TV_str_start, TV_str_stop), 0, remap='_replace_E_bg', reg_func='_regularizecolumn_ES_bg'),
        ColumnDef('S_bg', (TV_str_start, TV_str_stop), 0, remap='_replace_S_bg', reg_func='_regularizecolumn_ES_bg'),
                   )

    def __init_columns__(self):
        pass

    @classmethod
    def _regularizecolumn_nph_bg_sbr(cls, *args):
        """Column regularization for nph_bg and sbr columns"""
        return args[0:1] +  _regularize_column_startstop(*args[1:])

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
    def _regularizecolumn_brightness_bg(cls, *args):
        """Column regularization function for brightness_bg function"""
        return args[0:1] +  _regularize_column_startstop(*args[1:])

    def _get_brightness_bg(self, phsel:PhSel, starttype:str, stoptype:str)->np.ndarray[np.double]:
        """Getter function for brightness_bg column"""
        return _calc_brightness(self, self.parents['base'], 'nph_bg', phsel, starttype, stoptype)

    @classmethod
    def _get_brightness_bg_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """Title getter function for brightness_bg column"""
        return _get_brightness_title(col, '_{bg}br', include_unit, origin)

    @classmethod
    def _regularizecolumn_ratio_bg(cls, *args):
        """Column regularization function for ratio_bg column"""
        return args[0:2] +  _regularize_column_startstop(*args[2:])

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
    def _regularizecolumn_ES_bg(cls, *args:str)->tuple[str, str]:
        """Mapped Column regularization function fro E/S_bg"""
        return _regularize_column_startstop(*args)
    
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
        nph_c : (ph_sel:PhSel, starttype:str, stoptype:str)
            Corrected (according to correction factors and background) number of photons
            in ph_sel.
        brightness_c : (ph_sel:PhSel, starttype:str, stoptype:str)
            counts per second in given stream, with all correction factors applied
        ratio_c : (num_ph_sel:PhSel, dem_ph_sel:PhSel, starttype:str, stoptype:str)
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
        ColumnDef('nph_c', (PhSel, TV_str_start, TV_str_stop), 0, 'never', get_func='_get_nph_c', 
                  reg_func='_regularizecolumn_nph_c', title_func='_get_nph_c_title',
                  unit='cnts s^{-1}', index_unit='cnts s-1', title_is_tex=True),
        ColumnDef('brightness_c', (PhSel, TV_str_start, TV_str_stop), 0, 'never', get_func='_get_brightness_c', 
                  reg_func='_regularizecolumn_brightness_c', title_func='_get_brightness_c_title',
                  unit='cnts s^{-1}', index_unit='cnts s-1', title_is_tex=True),
        ColumnDef('ratio_c', (PhSel, PhSel, TV_str_start, TV_str_stop), 0, 'user', get_func='_get_ratio_c', 
                  reg_func='_regularizecolumn_ratio_c', title_func='_get_ratio_c_title'),
        ColumnDef('anisotropy_c', (PhSel, PhSel, str, str), 0, 'user', 
                  get_func='_get_anisotropy_c', reg_func='_regularizecolumn_anisotropy_c',
                  title_func='_get_anisotropy_c_title'),
        ColumnDef('E', (TV_str_start, TV_str_stop), 0, remap='_replace_E', reg_func='_regularizecolumn_ES'),
        ColumnDef('S', (TV_str_start, TV_str_stop), 0, remap='_replace_S', reg_func='_regularizecolumn_ES'),
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
        scheme = param.get('scheme', 'ALEX')
        matchstreams = param.get('matchstreams', True) # match streams defines how split/pol leakage/direx/beta/gamma are broadcast
        npol = param.get('npol', 1)
        nsplit = param.get('nsplit', 1)
        corr_mat = np.eye(2 if scheme == '1ex' else 4)
        lk = param.get('alpha', param.get('lk', 0.0))
        dir_ex = param.get('delta', ('dir_ex', 0.0))
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
        detdef = cls._detdef(param)
        if detdef.size != param.params['corr_mat'].shape[0]:
            raise ValueError("corr_mat must have both dimensions of size equal to the number of streams in detdef")
    
    @classmethod
    def _regularizecolumn_nph_c(cls, *args):
        """Column regularization function for nph_c column"""
        return args[0:1] +  _regularize_column_startstop(*args[1:])

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
    def _regularizecolumn_brightness_c(cls, *args):
        """Column regularization function for brightness_c column"""
        return args[0:1] + _regularize_column_startstop(*args[1:])

    def _get_brightness_c(self, phsel:PhSel, starttype:str, stoptype:str)->np.ndarray[np.double]:
        """Getter function for brightness_c column"""
        return _calc_brightness(self, self.parents['nph'].parents['base'], 'nph_c', phsel, starttype, stoptype)

    @classmethod
    def _get_brightness_c_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """Title getter function for brightness_c column"""
        return _get_brightness_title(col, '_{c}br', include_unit, origin)

    @classmethod
    def _regularizecolumn_ratio_c(cls, *args):
        """Column regularization function for ratio_c column"""
        return args[0:2] +  _regularize_column_startstop(*args[2:])

    def _get_ratio_c(self, num_phsel:PhSel, dem_phsel:PhSel, starttype:str, stoptype:str)->np.ndarray[np.float64]:
        """Getter function for ratio_c column"""
        return _calc_ratio(self, 'nph_c', num_phsel, dem_phsel, starttype, stoptype)

    @classmethod
    def _get_ratio_c_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """Title getter function for ratio_c"""
        return _get_ratio_title(col, 'F', include_unit, origin)

    @classmethod
    def _regularizecolumn_anisotropy_c(cls, *args):
        """Column regularization function for anisotropy_c column"""
        return args[0:2] +  _regularize_column_startstop(*args[2:])

    def _get_anisotropy_c(self, phsel_p:PhSel, phsel_s:PhSel, starttype:str, stoptype:str)->np.ndarray[np.float64]:
        """Getter function for anisotropy_c column"""
        return _calc_anisotropy(self, 'nph_c', phsel_p, phsel_s, starttype, stoptype)

    @classmethod
    def _get_anisotropy_c_title(cls, col:Column, include_unit:bool=False, origin:PhotonData=None)->str:
        """Title getter function for anisotropy_c column"""
        return _get_anisotropy_title(col, 'F', include_unit, origin)

    @classmethod
    def _regularizecolumn_ES(cls, *args:str)->tuple[str, str]:
        """Column regularization function for re-mapped columns E/S"""
        return _regularize_column_startstop(*args)

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
######################## Conveniece creation functions ########################
###############################################################################
def make_burst_search(bg:Param, m:int, F:float, stream:tuple[PhSel]=PhSel('0ex_1ex1em'))->Param:
    """
    Make a standard burst search :class:`Param`, makes ACBS burst search.

    Parameters
    ----------
    bg : Param
        Background :class:`Param` to use to set burst SNR thresholds.
    m : int
        Size of sliding window.
    F : float
        Minimum SNR to be considered in a burst.
    stream : tuple[PhSel], optional
        Photon stream on which to perform burst search. The default is PhSel('0ex_1ex1em').

    Returns
    -------
    Param
        ACBS burst search :class:`Param`. (Based on :class:`Bursts`)

    """
    return Param(Bursts, {'streams':(stream, ), 'm':m, 'F':F}, {'bg':bg})
    

def make_dcbs_burst_search(bg:Param, m:int, F:float, 
                           streamA:PhSel=PhSel('0ex'), streamB:PhSel=PhSel('1ex1em'))->Param:
    """
    Make Dual-Channel-Burst-Search :class:`Param`.

    Parameters
    ----------
    bg : Param
        Background :class:`Param` to use to set burst SNR thresholds.
    m : int
        Size of sliding window.
    F : float
        Minimum SNR to be considered in a burst.
    streamA : PhSel, optional
        First photon stream on which to perform burst search. The default is PhSel('0ex').
    streamB : PhSel, optional
        Second photon stream on which to perform burst search. The default is PhSel('1ex1em').

    Returns
    -------
    Param
        DCBS :class:`Param`. (Based on :class:`Bursts`)

    """
    return Param(Bursts, {'streams':(streamA, streamB), 'm':m, 'F':F}, {'bg':bg})
    

def make_correction_factors(bursts:Param, alpha:float=None, sigma:float=None,
                            gamma:float=1.0, beta:float=1.0, 
                            lk:float=0.0, dir_ex:float=0.0)->tuple[Param, Param]:
    """
    Make backgrouch corrected and cross-talk corrected :class:`Param` from burst
    search and information on correction factors.

    Parameters
    ----------
    bursts : Param
        Burst :class:`Param` to which to apply correction factors.
    alpha : float, optional
        Leakage factor, takes precedence over equivalent lk kwarg. 
        The default is None.
    sigma : float, optional
        Direct excitation factor, takes precedence over equivalent dir_ex kwarg. 
        The default is None.
    gamma : float, optional
        Correction coefficient for Donor Donor emission. The default is 1.0.
    beta : float, optional
        Correction coefficent for Acceptor Acceptor emission. The default is 1.0.
    lk : float, optional
        Leakage factor, overwritten by alpha. The default is 0.0.
    dir_ex : float, optional
        Direct excitation factor, overwritten by sigma. The default is 0.0.

    Returns
    -------
    nph : Param
        Background corrected :class:`Param` (based on :class:`NphBG`).
    ratio : Param
        Fully corrected stream intensity/ratio :class:`Param` 
        (based on :class:`Ratos`)

    """
    nph = Param(NphBG, {'single':True}, {'base':bursts, 'bg':bursts.parents['bg'][0]})
    alpha = lk if alpha is None else alpha
    sigma = dir_ex if sigma is None else sigma
    ratios = Param(Ratios, params={'gamma':gamma, 'beta':beta, 'alpha':alpha, 'sigma':sigma}, parents={'nph':nph})
    return nph, ratios
    

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
        register_PyCode(func, 'KDE_func', partial(fbc.kde_photons_user, func=func))

    
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

register_2cde_func(laplace_kde_2cde, shortcut=partial(fbc.kde_photons, func=0))


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

register_2cde_func(gaussian_kde_2cde, shortcut=partial(fbc.kde_photons, func=1))

class KDE(ChildPhotonTable):
    """
    This is still untested
    
    .. note::
        
        The original paper contains some ambiguities, and supplementary original
        labview screeen-shots do not clarify. Thus guranteed replication of
        publisehd 2CDE method is not possible.
        Use of 2CDE is generally discouraged.
        
    
    """
    param_defs = (
        ParamDef('kernel', TV_PyCode, default=laplace_kde_2cde),
        ParamDef('tau', TV_float(mn=0.0), default=5e-5),
        ParamDef('thresh', TV_float(mn=0.0), default=5.0)
        )
    parent_defs = (ParentDef('base', BasePhotonTableLike, is_base=True), )
    column_defs = (
        ColumnDef('fret', (PhSel, PhSel), 0, 'user', iter_func='_iter_fret2cde', reg_func='_regularizecolumn_2cde'),
        ColumnDef('alex', (PhSel, PhSel), 0, 'user', iter_func='_iter_alex2cde', reg_func='_regularizecolumn_2cde')
        )
    
    @classmethod
    def _regularizecolumn_2cde(cls, *args):
        phsel_d, phsel_a, = args[0:1], args[1:2]
        phsel_d = PhSel('0ex0em') if len(phsel_d) == 0 else phsel_d[0]
        phsel_a = PhSel('0ex1em') if len(phsel_a) == 0 else phsel_a[0]
        return phsel_d, phsel_a
    
    @cite('TorellaBioPhyJ2011', purpose='FRET 2CDE')
    def _iter_fret2cde(self, phsel_d:PhSel, phsel_a:PhSel)->float:
        func = self.param.params['kernel']
        tau = self.param.params['tau']
        thresh = self.param.params['thresh']
        stid_d = self.origin.detdef.get_stream_ids(phsel_d)
        stid_a = self.origin.detdef.get_stream_ids(phsel_a)
        if np.intersect1d(stid_d, stid_a).size != 0:
            raise ValueError("donor and acceptor FRET 2cde streams cannot overlap")
        kdefunc = (func)
        times_d = self.origin.times[np.isin(self.origin.dets, stid_d)]
        times_a = self.origin.times[np.isin(self.origin.dets, stid_a)]
        kde_dd = kdefunc(times_d, tau, lim=thresh) - 1.0
        kde_da = kdefunc(times_a, tau, locs=times_d, lim=thresh)
        kde_aa = kdefunc(times_a, tau, lim=thresh) - 1.0
        kde_ad = kdefunc(times_a, tau, locs=times_d, lim=thresh)
        prev_d, prev_a = 0, 0
        for start, stop in zip(self['start',], self['stop',]):
            istart_d, istop_d = fbc.index_range(times_d, start, stop, prev_d)
            istart_a, istop_a = fbc.index_range(times_a, start, stop, prev_a)
            prev_d, prev_a = istart_d, istart_a
            if istart_d == istop_d or istart_a == istop_d:
                yield np.nan
                continue
            e_d = kde_da[istart_d:istop_d] / (kde_da[istart_d:istop_d]+(1+2/(istop_d-istart_d))*kde_dd)
            e_a = kde_ad[istart_a:istop_a] / (kde_ad[istart_a:istop_a]+(1+2/(istop_a-istart_a))*kde_aa)
            yield 110 - 100*(np.mean(e_d)+np.mean(e_a))
    
    @cite('TorellaBioPhyJ2011', purpose='ALEX 2CDE')
    def _iter_alex2cde(self, phsel_d:PhSel, phsel_a:PhSel)->float:
        func = self.param.params['kernel']
        tau = self.param.params['tau']
        thresh = self.param.params['thresh']
        stid_d = self.origin.detdef.get_stream_ids(phsel_d)
        stid_a = self.origin.detdef.get_stream_ids(phsel_a)
        if np.intersect1d(stid_d, stid_a).size != 0:
            raise ValueError("donor and acceptor FRET 2cde streams cannot overlap")
        kdefunc = get_pycode_subval('KDE_func', func, func)
        times_d = self.origin.times[np.isin(self.origin.dets, stid_d)]
        times_a = self.origin.times[np.isin(self.origin.dets, stid_a)]
        kde_dd = kdefunc(times_d, tau, lim=thresh)
        kde_da = kdefunc(times_a, tau, locs=times_d, lim=thresh)
        kde_aa = kdefunc(times_a, tau, lim=thresh)
        kde_ad = kdefunc(times_a, tau, locs=times_d, lim=thresh)
        prev_d, prev_a = 0, 0
        for start, stop in zip(self['start',], self['stop',]):
            istart_d, istop_d = fbc.index_range(times_d, start, stop, prev_d)
            istart_a, istop_a = fbc.index_range(times_a, start, stop, prev_a)
            prev_d, prev_a = istart_d, istart_a
            if istart_d == istop_d or istart_a == istop_d:
                yield np.nan
                continue
            b_d = kde_da[istart_d:istop_d] / kde_dd
            b_a = kde_ad[istart_a:istop_a] / kde_aa
            yield 100 - 50*(np.mean(b_d)+np.mean(b_a))
