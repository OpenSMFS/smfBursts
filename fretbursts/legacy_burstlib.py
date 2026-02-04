#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created: 01/09/2025
# Author: Paul David Harris
"""
Wrapper that immitates the old FRETBursts Data objects and methods
"""
import os
from collections.abc import Sequence, Iterator, Callable
import hashlib
import numbers

import numpy as np
import pandas as pd
from scipy.stats import norm

from .utils.misc import tupledict, MutDict, pprint, clk_to_s
from .datamodel.tables import Param, Column, Gate, GateGroup
from .datamodel.gates import gte_gate, gt_gate
from .ph_sel import Ph_sel, mask_detarray, DetDef
from .read_photonHDF5 import PhEventsRawDiskDict, PhotonHDF5Data
from .photondata import PhotonData, PhotonDataList, Periods, BG, Burst, Nph_bg, Ratios, PhSpec
from .poisson_threshold import find_optimal_T_bga

from . import fret_fit

from .fit.gaussian_fitting import gaussian_fit_hist, gaussian_fit_cdf
from .fit.gaussian_fitting import two_gaussian_fit_hist, two_gaussian_fit_hist_min
from .fit.gaussian_fitting import two_gaussian_fit_hist_min_ab, two_gaussian_fit_EM
from .fit.gaussian_fitting import two_gauss_mix_pdf, two_gauss_mix_ab
from . import select_bursts

class ProxyDict:
    def __init__(self, proxy, prekey, postkey, keys):
        self.__proxy = proxy
        self.__prekey = prekey
        self.__postkey = postkey
        self.__keys = keys
    
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = self.__prekey + (key, )
        key = self.__prekey + key + self.__postkey
        return tuple(prox[key] for prox in self.__proxy)
    
    def keys(self):
        return self.__keys
    
    def values(self):
        for key in self.__keys:
            yield self[key]
    
    def items(self):
        for key in self.__keys:
            yield key, self[key]


class BurstProxy:
    def __init__(self, burst:Burst):
        self._burst = burst
    
    @property
    def start(self):
        return self._burst['istarttime',]
    
    @property
    def stop(self):
        return self._burst['istoptime',]
    
    @property
    def istart(self):
        return self._burst['istart',]
    
    @property
    def istop(self):
        return self._burst['istop',] - 1
    
    @property
    def width(self):
        return self.stop - self.start
    
    @property
    def counts(self):
        return self._burst['nph', Ph_sel('all')]
    
    @property
    def ph_rate(self):
        return self.counts / self.width
    
    @property
    def separation(self):
        """Separation between nearby bursts"""
        return self.start[1:] - self.stop[:-1]
    
    @property
    def dataframe(self):
        """A `pandas.DataFrame` containing burst data, one row per burst.
        """
        return pd.DataFrame(dict(istart=self.istart, istop=self.istop, 
                                 start=self.start, stop=self.stop))
    
    @property
    def size(self):
        return self.istart.size
    
    @property
    def num_bursts(self):
        return self.istart.size

    def __repr__(self):
        return self.dataframe.__repr__()

    def _repr_html_(self):
        return self.dataframe._repr_html_()
    
    def __getitem__(self, key):
        return self._burst[key]


def iter_bursts_start_stop(bursts):
    """Iterate over (start, stop) indexes to slice photons for each burst.
    """
    arr_istart = bursts.istart
    arr_istop = bursts.istop + 1
    for istart, istop in zip(arr_istart, arr_istop):
        yield istart, istop
        
        
def iter_bursts_ph(ph_data, bursts, mask=None, compact=False,
                   alex_period=None, excitation_width=None):
    """Iterator over arrays of photon-data for each burst.

    Arguments:
        ph_data (1D array): array of photon-data (timestamps, nanotimes).
        bursts (Bursts object): bursts computed from `ph`.
        mask (boolean mask or None): if not None, is a boolean mask
            to select photons in `ph_data` (for example Donor-ch photons).
        compact (bool): if True, a photon selection of only one excitation
            period is required and the timestamps are "compacted" by
            removing the "gaps" between each excitation period.
        alex_period (scalar): period of alternation in timestamp units.
            Used only when compact is True to "compact" the timestamps.
        excitation_width (float): fraction of `alex_period` covered by
            current photon selection. Used only when compact is True to
            "compact" the timestamps.

    Yields an array with a selection of "photons" for each burst.
    """
    if isinstance(mask, slice) and mask == slice(None):
        mask = None
    if compact:
        assert alex_period is not None
        assert excitation_width is not None
        assert mask is not None
    for start, stop in iter_bursts_start_stop(bursts):
        ph = ph_data[start:stop]
        if mask is not None:
            ph = ph[mask[start:stop]]
        if compact:
            ph = _ph_times_compact(ph, alex_period, excitation_width)
        yield ph


def burst_ph_stats(ph_data, bursts, func=np.mean, func_kw=None, **kwargs):
    """Reduce burst photons (timestamps, nanotimes) to a scalar using `func`.

    Arguments
        ph_data (1D array): array of photon-data (timestamps, nanotimes).
        bursts (Bursts object): bursts computed from `ph`.
        func (callable): function that takes the burst photon timestamps
            as first argument and returns a scalar.
        func_kw (callable): additional arguments in `func` beyond photon-data.
        **kwargs: additional arguments passed to :func:`iter_bursts_ph`.

    Return
        Array one element per burst.
    """
    if func_kw is None:
        func_kw = {}
    burst_stats = []
    for burst_ph in iter_bursts_ph(ph_data, bursts, **kwargs):
        burst_stats.append(func(burst_ph, **func_kw))
    return np.asarray(burst_stats, dtype=np.float64)  # NOTE: asfarray converts None to nan


def _ph_times_compact(ph_times_sel, alex_period, excitation_width):
    """Compact ph_times inplace by removing gaps between alternation periods.

    Arguments:
        ph_times_sel (array): array of timestamps from one alternation period.
        alex_period (scalar): period of alternation in timestamp units.
        excitation_width (float): fraction of `alex_period` covered by
            current photon selection.

    Returns nothing, ph_times is modified in-place.
    """
    # The formula is
    #   gaps = (ph_times_sel // alex_period)*excitation_width
    #   ph_times_sel = ph_times_sel - gaps
    # As a memory optimization the `-gaps` array is reused inplace
    times_minusgaps = (ph_times_sel // alex_period) * (-1 * excitation_width)
    # The formula is ph_times_sel = ph_times_sel - "gaps"
    times_minusgaps += ph_times_sel
    return times_minusgaps


def _excitation_width(excitation_range, alex_period):
    """Returns duration of alternation period outside selected excitation.
    """
    if excitation_range[1] > excitation_range[0]:
        return alex_period - excitation_range[1] + excitation_range[0]
    elif excitation_range[1] < excitation_range[0]:
        return excitation_range[0] - excitation_range[1]

def burst_stats(mburst, clk_p):
    """Compute average duration, size and burst-delay for bursts in mburst.
    """
    nans = [np.nan, np.nan]
    width_stats = np.array([[b.width.mean(), b.width.std()]
                            if b.num_bursts > 0 else nans for b in mburst]).T
    height_stats = np.array([[b.counts.mean(), b.counts.std()]
                             if b.num_bursts > 0 else nans for b in mburst]).T
    mean_burst_delay = np.array([b.separation.mean() if b.num_bursts > 0
                                 else np.nan for b in mburst])
    return (clk_to_s(width_stats, clk_p) * 1e3, height_stats,
            clk_to_s(mean_burst_delay, clk_p))


def print_burst_stats(d):
    """Print some bursts statistics."""
    nch = len(d.mburst)
    width_ms, height, delays = burst_stats(d.mburst, d.clk_p)
    s = "\nNUMBER OF BURSTS: m = %d, L = %d" % (d.m, d.L)
    s += "\nPixel:          "+"%7d "*nch % tuple(range(1, nch+1))
    s += "\n#:              "+"%7d "*nch % tuple([b.num_bursts for b in d.mburst])
    s += "\nT (us) [BS par] "+"%7d "*nch % tuple(np.array(d.T)*1e6)
    s += "\nBG Rat T (cps): "+"%7d "*nch % tuple(d.bg_mean[Ph_sel('all')])
    s += "\nBG Rat D (cps): "+"%7d "*nch % tuple(d.bg_mean[Ph_sel(Dex='Dem')])
    s += "\nBG Rat A (cps): "+"%7d "*nch % tuple(d.bg_mean[Ph_sel(Dex='Aem')])
    s += "\n\nBURST WIDTH STATS"
    s += "\nPixel:          "+"%7d "*nch % tuple(range(1, nch+1))
    s += "\nMean (ms):      "+"%7.3f "*nch % tuple(width_ms[0, :])
    s += "\nStd.dev (ms):   "+"%7.3f "*nch % tuple(width_ms[1, :])
    s += "\n\nBURST SIZE STATS"
    s += "\nPixel:          "+"%7d "*nch % tuple(range(1, nch+1))
    s += "\nMean (# ph):    "+"%7.2f "*nch % tuple(height[0, :])
    s += "\nStd.dev (# ph): "+"%7.2f "*nch % tuple(height[1, :])
    s += "\n\nBURST MEAN DELAY"
    s += "\nPixel:          "+"%7d "*nch % tuple(range(1, nch+1))
    s += "\nDelay (s):      "+"%7.3f "*nch % tuple(delays)
    return s


def _ex_scheme(setup:PhSpec)->str:
    if setup['alex_type'] == 'nano':
        return 'ALEX'
    if setup['alex_type'] == 'none':
        return '1ex'
    return 'ALEX' if np.all(setup['alternated']) else 'PAX'


class Data:
    """
    Container for all the information (timestamps, bursts) of a dataset.

    Data() contains all the information of a dataset (name, timestamps, bursts,
    correction factors) and provides several methods to perform analysis
    (background estimation, burst search, FRET fitting, etc...).

    When loading a measurement file a Data() object is created by one
    of the loader functions in `loaders.py`. Data() objects can be also
    created with :meth:`Data.copy`, :meth:`Data.fuse_bursts()` or
    :meth:`Data.select_bursts`.

    To add or delete data-attributes use `.add()` or `.delete()` methods.
    All the standard data-attributes are listed below.

    Note:
        Attributes of type "*list*" contain one element per channel.
        Each element, in turn, can be an array. For example `.ph_times_m[i]`
        is the array of timestamps for channel `i`; or `.nd[i]` is the array
        of donor counts in each burst for channel `i`.

    **Measurement attributes**

    Attributes:
        fname (string): measurements file name
        nch (int): number of channels
        clk_p (float): clock period in seconds for timestamps in `ph_times_m`
        ph_times_m (list): list of timestamp arrays (int64). Each array
            contains all the timestamps (donor+acceptor) in one channel.
        A_em (list): list of boolean arrays marking acceptor timestamps. Each
            array is a boolean mask for the corresponding ph_times_m array.
        leakage (float or array of floats): leakage (or bleed-through) fraction.
            May be scalar or same size as nch.
        gamma (float or array of floats): gamma factor.
            May be scalar or same size as nch.
        D_em (list of boolean arrays):  **[ALEX-only]**
            boolean mask for `.ph_times_m[i]` for donor emission
        D_ex, A_ex (list of boolean arrays):  **[ALEX-only]**
            boolean mask for `.ph_times_m[i]` during donor or acceptor
            excitation
        D_ON, A_ON (2-element tuples of int ): **[ALEX-only]**
            start-end values for donor and acceptor excitation selection.
        alex_period (int): **[ALEX-only]**
            duration of the alternation period in clock cycles.

    **Background Attributes**

    The background is computed with :meth:`Data.calc_bg`
    and is estimated in chunks of equal duration called *background periods*.
    Estimations are performed in each spot and photon stream.
    The following attributes contain the estimated background rate.

    Attributes:
        bg (dict): background rates for the different photon streams,
            channels and background periods. Keys are `Ph_sel` objects
            and values are lists (one element per channel) of arrays (one
            element per background period) of background rates.
        bg_mean (dict): mean background rates across the entire measurement
            for the different photon streams and channels. Keys are `Ph_sel`
            objects and values are lists (one element per channel) of
            background rates.
        nperiods (array): number of periods in which timestamps are split for
            background calculation, given per channel
            **NOTE: this is changed from previous versions, to support grouped experiments**
        bg_fun (function): function used to compute the background rates
        Lim (list): each element of this list is a list of index pairs for
            `.ph_times_m[i]` for **first** and **last** photon in each period.
        Ph_p (list): each element in this list is a list of timestamps pairs
            for **first** and **last** photon of each period.
        bg_ph_sel (Ph_sel object): photon selection used by Lim and Ph_p.
            See :mod:`fretbursts.ph_sel` for details.
        Th_us (dict): thresholds in us used to select the tail of the
            interphoton delay distribution. Keys are `Ph_sel` objects
            and values are lists (one element per channel) of arrays (one
            element per background period).

    Additionlly, there are a few deprecated attributes (`bg_dd`, `bg_ad`,
    `bg_da`, `bg_aa`, `rate_dd`, `rate_ad`, `rate_da`, `rate_aa` and `rate_m`)
    which will be removed in a future version.
    Please use :attr:`Data.bg` and :attr:`Data.bg_mean` instead.

    **Burst search parameters (user input)**

    These are the parameters used to perform the burst search
    (see :meth:`burst_search`).

    Attributes:
        ph_sel (Ph_sel object): photon selection used for burst search.
            See :mod:`fretbursts.ph_sel` for details.
        m (int): number of consecutive timestamps used to compute the
            local rate during burst search
        L (int): min. number of photons for a burst to be identified and saved
        P (float, probability): valid values [0..1].
            Probability that a burst-start is due to a Poisson background.
            The employed Poisson rate is the one computed by `.calc_bg()`.
        F (float): `(F * background_rate)` is the minimum rate for burst-start

    **Burst search data (available after burst search)**

    When not specified, parameters marked as (list of arrays) contains arrays
    with one element per bursts. `mburst` arrays contain one "row" per burst.
    `TT` arrays contain one element per `period` (see above: background
    attributes).

    Attributes:
        mburst (list of Bursts objects): list Bursts() one element per channel.
            See :class:`fretbursts.phtools.burstsearch.Bursts`.

        TT (list of arrays): list of arrays of *T* values (in sec.). A *T*
            value is the maximum delay between `m` photons to have a
            burst-start. Each channels has an array of *T* values, one for
            each background "period" (see above).
        T (array): per-channel mean of `TT`

        nd, na (list of arrays): number of donor or acceptor photons during
            donor excitation in each burst
        nt (list of arrays): total number photons (nd+na+naa)
        naa (list of arrays): number of acceptor photons in each burst
            during acceptor excitation **[ALEX only]**
        nar (list of arrays): **Deprecated** number of acceptor photons in each burst
            during donor excitation, not corrected for D-leakage and
            A-direct-excitation. **[PAX only]**
        bp (list of arrays): time period for each burst. Same shape as `nd`.
            This is needed to identify the background rate for each burst.
        bg_bs (list): background rates used for threshold computation in burst
            search (is a reference to `bg`, `bg_dd` or `bg_ad`).

        fuse (None or float): if not None, the burst separation in ms below
            which bursts have been fused (see `.fuse_bursts()`).

        E (list): FRET efficiency value for each burst:
                    E = na/(na + gamma*nd).
        S (list): stoichiometry value for each burst:
                    S = (gamma*nd + na) /(gamma*nd + na + naa)
    """
    #### Use as cataloge of actually stored parameters ########################
    _metadata:dict
    _rawevents: tuple[PhEventsRawDiskDict,...]
    _data: tuple[PhotonData,...]
    _period_params: tuple[Param,...]
    _period_tables: tuple[Periods,...]
    _bg_param: tuple[Param,...]
    _bg_tables: tuple[BG,...]
    _bg_err_type: str
    _burst_params: tuple[Param,...]
    _burst_tables: tuple[Burst,...]
    _nph_params: tuple[Param,...]
    _nph_tables: tuple[Nph_bg,...]
    _max_rate_m: int
    _max_rate_ph_sel: Ph_sel
    _sbr_ph_sel: Ph_sel
    _corr_params:tuple[Param,...]
    _corr_tables: tuple[Ratios,...]    
    _dither_ndd:tuple[np.ndarray[np.double],...]
    _dither_nda:tuple[np.ndarray[np.double],...]
    _dither_nad:tuple[np.ndarray[np.double],...]
    _dither_naa:tuple[np.ndarray[np.double],...]
    ###########################################################################
    _copyattrs = ('_metadata', '_rawdata', '_data', 
                  '_period_params', '_period_tables', '_bg_params', '_bg_tables',
                  '_bg_err_type', 
                  '_burst_params', '_burst_tables', '_nph_params', '_nph_tables', 
                  '_corr_params', '_corr_tables', 
                  '_max_rate_m', '_max_rate_ph_sel', '_sbr_ph_sel'
                  '_dither_ndd', '_dither_nda', '_dither_nad', '_dither_naa')
    _burstattrs = ('_burst_params', '_burst_tables', '_nph_params', '_nph_tables', 
                   '_corr_params', '_corr_tables', 
                   '_dither_ndd', '_dither_nda', '_dither_nad', '_dither_naa')
    _ditherattrs = ('_dither_ndd', '_dither_nda', '_dinter_nad', '_dither_naa')
    # Attribute names containing per-photon data.
    # Each attribute is a list (1 element per ch) of arrays (1 element
    # per photon).
    ################# these have been addressed as properties #################
    ph_fields = ['ph_times_m', 'nanotimes', 'particles',
                 'ph_times_t', 'nanotimes_t', 'particles_t', 'det_t',
                 'A_em', 'D_em', 'A_ex', 'D_ex']
    ###########################################################################
    # Attribute names containing background data.
    # The attribute `bg` is a dict with photon-selections as keys and
    # list of arrays as values. Each list contains one element per channel and
    # each array one element per background period.
    # The attributes `.Lim`  and `.Ph_p` are lists with one element per channel.
    # Each element is a lists-of-tuples (one tuple per background period).
    # These attributes do not exist before computing the background.
    ################# these have been addressed as properties #################
    bg_fields = ['bg', 'Lim', 'Ph_p']
    ###########################################################################

    # Attribute names containing per-burst data.
    # Each attribute is a list (1 element per ch) of arrays (1 element
    # per burst).
    # They do not necessarly exist. For example 'naa' exists only for ALEX
    # data. Also none of them exist before performing a burst search.
    ##### these have been addressed as properties (except deprecated nar) #####
    burst_fields = ('E', 'S', 'mburst', 'nd', 'na', 'nt', 'bp', 'nda', 'naa',
                    'max_rate', 'sbr')
    ###########################################################################
    # still to implement: nd, na, nt, nda, naa, max_rate, sbr, nar
    # Quantities (scalars or arrays) defining the current set of bursts
    ################# these have been addressed as properties #################
    burst_metadata = ['m', 'L', 'T', 'TT', 'F', 'FF', 'P', 'PP', 'rate_th',
                      'bg_bs', 'ph_sel', 'bg_corrected', 'leakage_corrected',
                      'dir_ex_corrected', 'dithering', 'fuse', 'lsb']
    ###########################################################################
    
    @classmethod
    def new_raw(cls, rawdata:PhotonHDF5Data, 
                 leakage:float=None, dir_ex:float=None, gamma:float=None, beta:float=None,
                 ALEX=True, PAX=False, **kwargs):
        obj = object.__new__(cls)
        obj._metadata = dict(ALEX=ALEX, 
                              leakage = 0.0 if leakage is None else float(leakage),
                              gamma = 1.0 if gamma is None else float(gamma),
                              dir_ex = 0.0 if dir_ex is None else float(dir_ex), 
                              beta = 1.0 if beta is None else float(beta), 
                              **kwargs)
        obj = object.__new__(cls)
        obj._rawdata = rawdata
        kw = dict(leakage=leakage, dir_ex=dir_ex, gamma=gamma, beta=beta, ALEX=ALEX, PAX=PAX)
        kwargs.update({k:v for k, v in kw.items() if v is not None})
        obj._metadata = kwargs
    
    @property
    def nch(self)->int:
        return len(self._data) if hasattr(self, '_datas') else len(self._rawdata.photon_data)
    
    @property
    def D_ON(self):
        if hasattr(self, '_data'):
            return self._data[0].setup.ex_ranges[0][0]
        if 'alex_excitation_period1' not in self._rawdata.photon_data[0].meas_specs:
            raise ValueError("alex_excitation_period2 not specified")
        return self._rawdata.photon_data[0].meas_specs['alex_excitation_period1']
        
    @D_ON.setter
    def D_ON(self, val):
        if hasattr(self, '_data'):
            raise AttributeError("after sorting photons by excitation, cannot modify D_ON")
        for photon_data in self._rawdata.photon_data:
            photon_data.meas_specs['alex_excitation_period1'] = val
        

    @property
    def A_ON(self):
        if hasattr(self, '_data'):
            return self._data[0].setup.ex_ranges[1][0]
        if 'alex_excitation_period2' not in self._rawdata.photon_data[0].meas_specs:
            raise ValueError("alex_excitation_period2 not specified")
        return self._rawsetup['meas_specs']['alex_excitation_period2']
    
    @A_ON.setter
    def A_ON(self, val):
        if hasattr(self, '_data'):
            raise AttributeError("after sorting photons by excitation, cannot modify A_ON")
        for photon_data in self._rawdata.photon_data:
            photon_data.meas_specs['alex_excitation_period2'] = val
        
    @property
    def _detdef(self)->DetDef:
        if not hasattr(self, "_data"):
            raise AttributeError("must sort photons first")
        return self._data[0].detdef
    
    @property
    def ph_streams(self):
        if not hasattr(self, '_detdef'):
            raise AttributeError("cannot define ph_streams until photons are sorted")
        return [Ph_sel('all')] + [self._detdef.stream_ids_to_Ph_sel(i) for i in range(self._detdef.size)]
    
    @property
    def ph_times_t(self)->tuple[np.ndarray[np.int64],...]:
        if not hasattr(self, '_rawdata'):
            raise AttributeError("photons sorted, ph_times_t removed")
        return tuple(rd.times for rd in self.rawdata.photon_data)
    
    @property
    def ph_times_m(self)->tuple[np.ndarray[np.int64]]:
        if not hasattr(self, '_data'):
            raise AttributeError("photons not sorted, ph_times_m not created")
        return tuple(data.times for data in self._data)
    
    @property
    def det_t(self):
        if not hasattr(self, '_rawdata'):
            raise AttributeError("photons sorted, det_t removed")
        return tuple(rd.dets for rd in self.rawdata.photon_data)
    
    @property
    def det_m(self):
        if not hasattr(self, '_data'):
            raise AttributeError("photons not sorted, det_m not created")
        return tuple(data.dets for data in self._data)
    
    @property
    def nanotimes_t(self):
        if not hasattr(self, '_rawdata'):
            raise AttributeError("photons sorted, nanotimes_t removed")
        return tuple(rd.nanos for rd in self.rawdata.photon_data)
    
    @property
    def nanotimes(self):
        if not hasattr(self, '_data'):
            raise AttributeError("photons not sorted, nanotimaes not created")
        return tuple(data.nanos for data in self._data)
    
    @property
    def particles_t(self):
        if not hasattr(self, '_rawdata'):
            raise AttributeError("photons sorted, ph_times_t removed")
        return tuple(rd.particles for rd in self.rawdata.photon_data)
    
    @property
    def particles(self):
        if not hasattr(self, '_data'):
            raise AttributeError("photons not sorted, particles not created")
        return tuple(data.particles for data in self._data)
    
    @property
    def Lim(self):
        """istart and istop time indexes (of ph_times_m) of each burst period"""
        if not hasattr(self, '_bg_params'):
            raise AttributeError("must perform calc_bg first")
        return tuple(np.vstack([bg['istart',], bg['istop',]-1]).T for bg in self._bg_tables)
    
    @property
    def Ph_p(self):
        if not hasattr(self, '_period_params'):
            raise AttributeError("must perform calc_bg first")
        return tuple(np.vstack([period['istarttime'], period['istoptime']]) 
                     for period in self._period_tables)
        
    @property
    def bp(self):
        """index of bg period of each burst"""
        if not hasattr(self, '_burst_params'):
            raise AttributeError("must perform calc_bg first")
        out = list()
        for burst, per in zip(self._burst_tables):
            temp = np.empty(burst.size, dtype=np.int64)
            piter = enumerate(burst.parents['bg'].parents['base'].iter_column('stop'))
            cp, pstop = next(piter)
            for i, start in enumerate(burst['start']):
                while start > pstop:
                    pcp, stop = next(piter)
                temp[i] = cp
            out.append(out)
        return out
    
    @property
    def bg(self):
        """dictionary of background rates per bg period"""
        if not hasattr(self, '_bg_params'):
            raise AttributeError("must perform calc_bg first")
        return ProxyDict(self._bg_tables, ('bg', ), tuple(), self.ph_streams)
    
    @property
    def bg_err(self):
        """Error metric of bg calculation"""
        if not hasattr(self, '_bg_params'):
            raise AttributeError("must perform calc_bg first")
        return ProxyDict(self._bg_tables, (self._bg_err_type, ), tuple(), self.ph_streams)
    
    @property
    def ph_sel(self)->Ph_sel:
        """Photon selection of burst search (in case of dual-channel, returns the union of photon selections)"""
        if not hasattr(self, '_burst_params'):
            raise AttributeError("must perform burst search first")
        out = self._burst_params[0].params['channels'][0]
        for ph_sel in self._burst_params[0].params['channels'][1:]:
            out |= ph_sel
        return out
    
    @property
    def bg_bs(self):
        """bg rates used for burst search"""
        if not hasattr(self, '_burst_params'):
            raise AttributeError("must perform burst search first")
        return self.bg[self.ph_sel]
    
    @property
    def nperiods(self):
        return np.array([prd.size for prd in self._period_tables])        

    @property
    def bg_mean(self):
        if not hasattr(self, '_bg_tables'):
            raise RuntimeError('No background found, compute it first.')
        if not hasattr(self, '_bg_mean'):
            self._bg_mean = {ph_sel: [np.mean(bg_ch) for bg_ch in self.bg[ph_sel]]
                             for ph_sel in self.ph_streams}
        return self._bg_mean
    
    @property
    def fuse(self)->bool:
        if not hasattr(self, '_burst_params'):
            raise AttributeError("must perform burst search first")
        if self._burst_params[0].params['fuse'] == -1.0:
            raise AttributeError("non-fused burst search")
        return self._burst_param[0].params['fuse']
    
    @property
    def m(self)->int:
        """sliding window size (no. of photons) for burst search"""
        if not hasattr(self, '_burst_params'):
            raise AttributeError("must perform burst search first")
        return self._burst_params[0].params['m'][0]
    
    @property
    def c(self)->float:
        if not hasattr(self, '_burst_params'):
            raise AttributeError("must perform burst search first")
        return self._burst_params[0].params['c'][0]
    
    @property
    def F(self)->float:
        """how many times higher than the background rate is 
        the minimum rate used for burst search"""
        if not hasattr(self, '_burst_params'):
            raise AttributeError("must perform burst search first")
        if self._burst_params[0].params['asP'][0]:
            return None
        return self._burst_params[0].params['F'][0]
    
    @property
    def FF(self)->np.ndarray[np.double]:
        """array of size nch of F"""
        if not hasattr(self, '_burst_params'):
            raise AttributeError("must perform burst search first")
        return np.repeat(self.F, self.nch)
    
    @property
    def P(self)->float:
        """threshold for burst detection expressed as a probability that a 
        detected bursts is not due to a Poisson background"""
        if not hasattr(self, '_burst_params'):
            raise AttributeError("must perform burst search first")
        if self._burst_params[0].params['asP'][0]:
            return None
        return self._burst_params[0].params['F'][0]
    
    @property
    def PP(self)->np.ndarray[np.double]:
        """Array of size nch of P"""
        if not hasattr(self, '_burst_params'):
            raise AttributeError("must perform burst search first")
        return np.repeat(self.P, self.nch)
        
    @property
    def TT(self)->tuple[np.ndarray[np.double],...]:
        if not hasattr(self, '_burst_params'):
            raise AttributeError("must perform burst search first")
        if self._burst_params[0].params['asP'][0]:
            return tuple(find_optimal_T_bga(bg, self.m, self.P) for bg in self.bg[self.ph_sel])
        return tuple((self.m - 1 - self.c)/(bg*self.F) for bg in self.bg[self.ph_sel])
    
    @property
    def T(self)->np.ndarray[np.double]:
        """mean threshold (TT) per channel"""
        if not hasattr(self, '_burst_params'):
            raise AttributeError("must perform burst search first")
        return np.array([np.mean(TT) for TT in self.TT])
    
    @property
    def rate_th(self)->tuple[np.ndarray[np.double],...]:
        if not hasattr(self, '_burst_params'):
            raise AttributeError("must perform burst search first")
        return tuple(self.m/TT for TT in self.TT)
    
    @property
    def nd(self):
        if hasattr(self, '_dither_ndd'):
            return self._dither_ndd
        if not hasattr(self, '_nph_tables'):
            raise AttributeError("must perform burst search first")
        if hasattr(self, '_corr_params'):
            return tuple(corr['nph_c', Ph_sel('0ex0em')] for corr in self._corr_tables)
        return tuple(brst['nph_bg', Ph_sel('0ex0em')] for brst in self._nph_tables)
    
    @property
    def na(self):
        if hasattr(self, '_dither_nda'):
            return self._dither_nda
        if not hasattr(self, '_nph_tables'):
            raise AttributeError("must perform burst search first")
        if hasattr(self, '_corr_params'):
            return tuple(corr['nph_c', Ph_sel('0ex1em')] for corr in self._corr_tables)
        return tuple(brst['nph_bg', Ph_sel('0ex1em')] for brst in self._nph_tables)
    
    @property
    def nda(self):
        if hasattr(self, '_dither_nad'):
            return self._dither_nad
        if not hasattr(self, '_nph_tables'):
            raise AttributeError("must perform burst search first")
        if hasattr(self, '_corr_params'):
            return tuple(corr['nph_c', Ph_sel('1ex0em')] for corr in self._corr_tables)
        return tuple(brst['nph_bg', Ph_sel('1ex0em')] for brst in self._nph_tables)
    
    @property
    def naa(self):
        if hasattr(self, '_dither_naa'):
            return self._dither_naa
        if not hasattr(self, '_nph_tables'):
            raise AttributeError("must perform burst search first")
        if hasattr(self, '_corr_params'):
            return tuple(corr['nph_c', Ph_sel('1ex1em')] for corr in self._corr_tables)
        return tuple(brst['nph_bg', Ph_sel('1ex1em')] for brst in self._nph_tables)
    
    @property
    def nt(self):
        if hasattr(self, '_dither_nd'):
            return tuple(sum(ns) for ns in zip(*(getattr(self, attr) 
                                                 for attr in self._ditherattrs 
                                                 if hasattr(self, attr) and attr != '_dither_nad')))
        return tuple(brst['nph_bg', Ph_sel('all') if self._detdef.ex == 1 else Ph_sel('0ex_1ex1em')]
                     for brst in self._nph_tables)
    
    @property
    def _dithered_factors(self):
        ndd, nda = self._dither_ndd, self._dither_nda
        if hasattr(self, '_dither_naa'):
            nad, naa = self._dither_nad, self._dither_naa
        if hasattr(self, '_corr_params'):
            corr_mat = self._corr_params.params['corr_mat']
            cdd = corr_mat[0,0]*ndd + corr_mat[0,1]*nda
            cda = corr_mat[1,0]*ndd + corr_mat[1,1]*nda
            if hasattr(self, '_dither_naa'):
                cad = corr_mat[0,2]*nad + corr_mat[0,3]*naa
                caa = corr_mat[1,2]*nad + corr_mat[1,3]*naa
        else:
            cdd, cda = ndd, nda
            if hasattr(self, '_dither_naa'):
                cad, caa = nad, naa
        out = (cdd, cda)
        if hasattr(self, '_dither_naa'):
            out += (cad, caa)
        return out
    
    @property
    def E(self)->tuple[np.ndarray[np.double],...]:
        if not hasattr(self, '_burst_params'):
            raise AttributeError("must perform burst search first")
        if hasattr(self, '_dither_ndd'):
            ot = self._dithered_factors
            return ot[1] / (ot[0] + ot[1])
        return tuple(corr['ratio_c', Ph_sel('0ex1em'), Ph_sel('0ex[0,1]em'), 'istarttime', 'istoptime'] for corr in self._corr_tables)
        
    @property
    def S(self)->tuple[np.ndarray[np.double],...]:
        if not hasattr(self, '_burst_params'):
            raise AttributeError("must perform burst search first")
        if hasattr(self, '_dither_naa'):
            cdd, cda, cad, caa = self._dithered_factors
            return (cdd+cda)/(cdd+cda+caa)
        return tuple(corr['ratio_c', Ph_sel('0ex'), Ph_sel('0ex_1ex1em'), 'istarttime', 'istoptime'] for corr in self._corr_tables)
    
    @property
    def max_rate_dct(self):
        if not hasattr(self, '_max_rate_m'):
            self._max_rate_m = self.m
        return ProxyDict(self._burst_tables, ('max_rate', ),  (self._max_rate_m, ), self.ph_streams)
    
    @property
    def max_rate(self):
        if not hasattr(self, '_max_rate_m') or not hasattr(self, '_max_rate_ph_sel'):
            raise AttributeError("max_rate not calculated")
        return self.max_rate_dct[self._max_rate_ph_sel]
    
    @property
    def sbr_dct(self):
        return ProxyDict(self._nph_tables, ('sbr', ),  tuple(), self.ph_streams)
    
    @property
    def sbr(self):
        if not hasattr(self, '_sbr_ph_sel'):
            raise AttributeError("sbr not calculated")
        return self.sbr_dct(self._sbr_ph_sel)
    
    @property
    def alternated(self)->bool:
        return self._detdef.ex != 1
    
    @property
    def s(self)->GateGroup:
        """Note: this property now returns gategroup, use is discouraged"""
        if not hasattr(self, "_burst_params"):
            raise AttributeError("burst have not been calculated yet")
        return self._burst_params[0].gategroup
    
    def _assert_compact(self, ph_sel:Ph_sel)->Ph_sel:
        if not self.alternated:
            raise ValueError('Option compact=True requires ALEX data.')
        ph_sel = ph_sel.render_positive(self._detdef)
        if len(ph_sel.ex.elements) != 1:
            msg = ('Option compact=True requires a photon selection \n'
                   'from a single excitation period (either Dex or Aex).')
            raise ValueError(msg)
        return ph_sel

    def _iter_ph_fields(self, *args:str)->Iterator[tuple[np.ndarray,...]]:
        for data in self._data:
            yield tuple(getattr(data, arg) for arg in args)

    def iter_ph_masks(self, ph_sel:Ph_sel=Ph_sel('all'))->Iterator[tuple[np.ndarray[np.bool_],...]]:
        """Iterator returning masks for `ph_sel` photons.

        Arguments:
            ph_sel (Ph_sel object): object defining the photon selection.
                See :mod:`fretbursts.ph_sel` for details.
        """
        for data in self._data:
            yield mask_detarray(data.detdef, ph_sel, data.dets)

    def get_ph_times(self, ich:int=0, ph_sel:Ph_sel=Ph_sel('all'), compact:bool=False):
        """Returns the timestamps array for channel `ich`.

        This method always returns in-memory arrays, even when ph_times_m
        is a disk-backed list of arrays.

        Arguments:
            ph_sel (Ph_sel object): object defining the photon selection.
                See :mod:`fretbursts.ph_sel` for details.
            compact (bool): if True, a photon selection of only one excitation
                period is required and the timestamps are "compacted" by
                removing the "gaps" between each excitation period.
        """
        mask = slice(None) if self._is_allph(ph_sel) else mask_detarray(self._detdef, ph_sel, self._data[ich].dets)
        return self._data[ich].times[mask]
    
    def iter_ph_times(self, ph_sel=Ph_sel('all'), compact=False):
        """Iterator that returns the arrays of timestamps in `.ph_times_m`.

        Arguments:
            Same arguments as :meth:`get_ph_mask` except for `ich`.
        """
        if self._is_allph(ph_sel):
            for data in self._data:
                times = data.times
                if compact:
                    ...
                else:
                    yield times
        else:
            for i, (dets, times) in enumerate(self._iter_ph_fields('dets', 'times')):
                times_masked = times[mask_detarray(self.detdef, ph_sel, dets)]
                if compact:
                    yield _ph_times_compact(times_masked, 
                                            self._data[i].setup['alex_period'], 
                                            self._excitation_width(ph_sel, ich=i))
                else:
                    yield times_masked
    
    @property
    def D_ex(self):
        return tuple(self.iter_ph_masks(Ph_sel('0ex')))
    
    @property
    def A_ex(self):
        return tuple(self.iter_ph_masks(Ph_sel('1ex')))
    
    @property
    def D_em(self):
        return tuple(self.iter_ph_masks(Ph_sel('0em')))
    
    @property
    def A_em(self):
        return tuple(self.iter_ph_masks(Ph_sel('1em')))
    
    def copy(self, mute=False):
        """
        Copy values to new data object
        """
        obj = object.__new__(type(self))
        for attr in self._copyattrs:
            if hasattr(self, attr):
                setattr(obj, attr, getattr(self, attr))
        return obj

    ##
    # Methods for photon timestamps (ph_times_m) access
    #
    def ph_times_hash(self, hash_name='md5', hexdigest=True):
        """Return an hash for the timestamps arrays.
        """
        m = hashlib.new(hash_name)
        for ph in self.iter_ph_times():
            m.update(ph[:])
        if hexdigest:
            return m.hexdigest()
        else:
            return m
    
    def _excitation_width(self, ph_sel:Ph_sel, ich:int=0)->int:
        """Returns duration of alternation period outside selected excitation.
        """
        ph_sel = self._assert_compact(ph_sel)
        excitation_range = self._data[ich].setup['ex_ranges'][list(ph_sel.ex.elements)[0]][0,:]
        return _excitation_width(excitation_range, self._data[ich].setup['alex_period'])

    @property
    def ph_data_sizes(self)->np.ndarray[np.int64]:
        """Array of total number of photons (ph-data) for each channel.
        """
        return np.array([ph.shape[0] for ph in self.ph_times_m])
    
    
    def _is_allph(self, ph_sel:Ph_sel)->bool:
        """Return whether a photon selection `ph_sel` covers all photon."""
        return self._detdef.get_stream_ids(ph_sel).size == self._detdef.size

    def get_ph_mask(self, ich:int=0, ph_sel:Ph_sel=Ph_sel('all'))->np.ndarray[np.bool_]:
        """Returns a mask for `ph_sel` photons in channel `ich`.

        The masks are either boolean arrays or slices (full or empty). In
        both cases they can be used to index the timestamps of the
        corresponding channel.

        Arguments:
            ph_sel (Ph_sel object): object defining the photon selection.
                See :mod:`fretbursts.ph_sel` for details.
        """
        if not isinstance(ich, numbers.Integral):
            raise TypeError(f"channel must be integer value, got {type(ich)}")

        if self._is_allph(ph_sel):
            # save memory as slices are not copied
            return slice(None)
        return mask_detarray(self.detdef, ph_sel, self._data[ich].dets)
    
    def get_A_em(self, ich:int=0)->np.ndarray[np.bool_]:
        """Returns a mask to select photons detected in the acceptor ch."""
        return self._get_ph_mask(ich, Ph_sel('1em'))

    def get_D_em(self, ich=0)->np.ndarray[np.bool_]:
        """Returns a mask to select photons detected in the donor ch."""
        return self._get_ph_mask(ich, Ph_sel('0em'))

    def get_A_ex(self, ich=0)->np.ndarray[np.bool_]:
        """Returns a mask to select photons in acceptor-excitation periods."""
        return self._get_ph_mask(ich, Ph_sel('1ex'))

    def get_D_ex(self, ich=0)->np.ndarray[np.bool_]:
        """Returns a mask to select photons in donor-excitation periods."""
        return self._get_ph_mask(ich, Ph_sel('0ex'))

    def get_D_em_D_ex(self, ich:int=0)->np.ndarray[np.bool_]:
        """Returns a mask of donor photons during donor-excitation."""
        return self._get_ph_mask(ich, Ph_sel('0ex0em'))

    def get_A_em_D_ex(self, ich:int=0)->np.ndarray[np.bool_]:
        """Returns a mask of acceptor photons during donor-excitation."""
        return self._get_ph_mask(ich, Ph_sel('0ex1em'))

    def iter_ph_times_period(self, ich:int=0, ph_sel:Ph_sel=Ph_sel('all')):
        """Iterate through arrays of ph timestamps in each background period.
        """
        yield from self._bg_tables[ich].iter_column('ph_times', ph_sel)
    
    def get_ph_times_period(self, period:int, ich:int=0, ph_sel:Ph_sel=Ph_sel('all'),
                            mask=None):
        """Return the array of ph_times in `period`, `ich` and `ph_sel`.
        """
        return self._bg_tables[ich]['ph_times', ph_sel][period]
    
    ##
    # Methods and properties for burst-data access
    #
    @property
    def num_bursts(self)->np.ndarray[np.int64]:
        """Array of number of bursts in each channel."""
        return np.array([bursts.size for bursts in self._burst_tables])

    @property
    def burst_widths(self)->tuple[np.ndarray[np.double],...]:
        """List of arrays of burst duration in seconds. One array per channel.
        """
        return np.array([bursts.width * self.clk_p for bursts in self.mburst])
    
    # NOTE: this should be done by setting appropriate values in _corr_params, probably deprecate _aex_fraction and _aex_dex_ratio
    @property
    def _aex_dex_ratio(self):
        """Ratio of Aex and Dex period durations."""
        #### DEPRECATED because cannot understand and poorly documented how PAX computation should actually work
        assert self.alternated
        D_ON, A_ON = self.D_ON, self.A_ON
        return (A_ON[1] - A_ON[0]) / (D_ON[1] - D_ON[0])
    
    @staticmethod
    def _burst_sizes_pax_formula(ph_sel=Ph_sel('all'),
                                 naa_aexonly=False, naa_comp=False,
                                 na_comp=False,
                                 gamma=None, beta=None, donor_ref=True):
        """Return a latex expression of the PAX burst size."""
        gamma = None if gamma == 1 else gamma
        beta = None if beta == 1 else beta
        terms = []
        if Ph_sel('0ex0em') in ph_sel:
            terms.append('n_d')
        if Ph_sel('1ex0em') in ph_sel:
            terms.append('n_{da}')
        if len(terms) > 1:
            terms = ['+'.join(terms)]
        if gamma is not None and not donor_ref and len(terms) > 0:
            terms[0] = r'\gamma\left(' + terms[0] + r'\right) '

        if Ph_sel('0ex1em') in ph_sel:
            na_term = 'n_a'
            corr = ''
            if na_comp:
                corr += r'\alpha'
            if gamma is not None and donor_ref:
                corr += r'\gamma'
            if len(corr) > 0:
                corr = corr if len(corr) < 8 else '(%s)' % corr
                na_term += '%s^{-1}' % corr
            terms.append(na_term)
        if Ph_sel('1ex1em') in ph_sel:
            naa_term = 'n_{DA_{ex}A_{em}} '
            if naa_aexonly:
                naa_term += r' - \frac{w_A}{w_D} n_a '
                naa_term = r'\left(' + naa_term + r'\right) '
            corr = ''
            if naa_comp:
                corr += r'\alpha'
            if beta is not None:
                corr += r'\beta'
            if gamma is not None and donor_ref:
                corr += r'\gamma'
            if len(corr) > 0:
                corr = corr if len(corr) < 8 else '(%s)' % corr
                naa_term += '%s^{-1}' % corr
            terms.append(naa_term)
        return ' + '.join(terms)

    def burst_sizes_pax_ich(self, ich=0, ph_sel=Ph_sel('all'),
                            naa_aexonly=False, naa_comp=False, na_comp=False,
                            gamma=1., beta=1., donor_ref=True):
        r"""Return different definitions of PAX burst sizes for channel `ich`.

        There are 4 basic "terms" corresponding to the 4 photon streams:
        `nd`, `na`, `nda`, `naa`. Which term is included is defined by
        the `ph_sel` argument (by default all are included).
        The other arguments specify the various corrections for each term.

        Arguments:
            ich (int): the spot number, only relevant for multi-spot.
                In single-spot data there is only one channel (`ich=0`)
                so this argument may be omitted. Default 0.
            gamma (float): coefficient for gamma correction of burst
                sizes. Default: 1. For more info see explanation above.
            beta (float): beta correction factor used for the DAexAem term.
            donor_ref (bool): True or False select different conventions
                for burst size correction. For details see
                :meth:`fretbursts.burstlib.Data.burst_sizes_ich`.
            ph_sel (Ph_sel object): defines which terms are included in the
                burst size.
            na_comp (bool): If True, multiply the `na` term by `(1 + Wa/Wd)`,
                where Wa and Wd are the D and A alternation durations
                (typically Wa/Wd = 1).
            naa_aexonly (bool): if True, the `naa` term is corrected to
                include only A emission due to A excitation.
                If False, the `naa` term includes all the counts in DAexAem.
                The `naa` term also depends on the `naa_comp` argument.
            naa_comp (bool): If True, multiply the `naa` term
                by `(1 + Wa/Wd)`,
                where Wa and Wd are the D and A alternation durations
                (typically Wa/Wd = 1). The `naa` term also depends on
                the `naa_aexonly` argument.

        Returns
            Array of burst sizes for channel `ich`.

        Examples:

            Burst sizes with all streams and no correction::

                Data.burst_sizes_pax_ich(ph_sel=Ph_sel('all'))

            .. math::

                F_{D_{ex}D_{em}} + F_{DA_{ex}D_{em}} +
                F_{FRET} + F_{DA_{ex}A_{em}}

            Burst sizes with all streams and all corrections::

                Data.burst_sizes_pax_ich(ph_sel=Ph_sel('all'), na_comp=True,
                                         aa_aexonly=True, naa_comp=True)

            .. math::

                \gamma (F_{D_{ex}D_{em}} + F_{DA_{ex}D_{em}}) +
                \left(1 + \frac{W_A}{W_D} \right) \,
                ( F_{FRET} +
                  (F_{DA_{ex}A_{em}} - F_{D_{ex}A_{em}})\,\beta^{-1})

        See also:
            :meth:`Data.burst_sizes_ich`
        """
        assert 'PAX' in self.meas_type
        aex_dex_ratio = self._aex_dex_ratio

        bsize = 0
        if ph_sel & Ph_sel('0ex'):
            bsize += self.nd[ich] * gamma
        if ph_sel & Ph_sel('1ex0em') :
            bsize += self.nda[ich] * gamma

        if ph_sel * Ph_sel('0ex1em'):
            na_term = self.na[ich]
            if na_comp:
                na_term = na_term * (1 + aex_dex_ratio)
            bsize += na_term
        if ph_sel & Ph_sel('1ex1em') :
            naa_term = self.naa[ich].copy()
            if naa_aexonly:
                naa_term -= aex_dex_ratio * self.nar[ich]
            if naa_comp:
                naa_term *= (1 + aex_dex_ratio)
            bsize += naa_term / beta

        if donor_ref:
            bsize /= gamma
        return bsize

    def burst_sizes_ich(self, ich=0, gamma=1., add_naa=False,
                        beta=1., donor_ref=True):
        """Return gamma corrected burst sizes for channel `ich`.

        If `donor_ref == True` (default) the gamma corrected burst size is
        computed according to::

            1)    :math:`nd + na / gamma`

        Otherwise, if `donor_ref == False`, the gamma corrected burst size is::

            2)    :math:`nd * gamma  + na`

        With the definition (1) the corrected burst size is equal to the raw
        burst size for zero-FRET or D-only bursts (that's why is `donor_ref`).
        With the definition (2) the corrected burst size is equal to the raw
        burst size for 100%-FRET bursts.

        In an ALEX measurement, use `add_naa = True` to add counts from
        AexAem stream to the returned burst size. The argument `gamma` and
        `beta` are used to correctly scale `naa` so that it become
        commensurate with the Dex corrected burst size. In particular,
        when using definition (1) (i.e. `donor_ref = True`), the total
        burst size is::

            :math:`(nd + na/gamma) + naa / (beta * gamma)`

        Conversely, when using definition (2) (`donor_ref = False`), the
        total burst size is::

            :math:`(nd * gamma + na) + naa / beta`

        Arguments:
            ich (int): the spot number, only relevant for multi-spot.
                In single-spot data there is only one channel (`ich=0`)
                so this argument may be omitted. Default 0.
            add_naa (boolean): when True, add a term for AexAem photons when
                computing burst size. Default False.
            gamma (float): coefficient for gamma correction of burst
                sizes. Default: 1. For more info see explanation above.
            beta (float): beta correction factor used for the AexAem term
                of the burst size. Default 1. If `add_naa = False` or
                measurement is not ALEX this argument is ignored.
                For more info see explanation above.
            donor_ref (bool): select the convention for burst size correction.
                See details above in the function description.

        Returns
            Array of burst sizes for channel `ich`.
        """
        burst_size = self.nd[ich] * gamma + self.na[ich]

        if add_naa and self.alternated:
            naa = self.naa[ich]
            if 'PAX' in self.meas_type:
                naa = naa - self._aex_dex_ratio * self.nar[ich]
            burst_size += naa / beta
        if donor_ref:
            burst_size /= gamma
        return burst_size

    @property
    def naa_aexonly(self):
        """Returns self.naa - aex_dex_ratio * self.nar. PAX-only.
        """
        WaWd_ratio = self._aex_dex_ratio
        return [naa - WaWd_ratio * nar
                for naa, nar in zip(self.naa, self.nar)]
    
    def burst_sizes(self, gamma=1., add_naa=False, beta=1., donor_ref=True):
        """Return gamma corrected burst sizes for all the channel.

        Compute burst sizes by calling, for each channel,
        :meth:`burst_sizes_ich`.

        See :meth:`burst_sizes_ich` for description of the arguments.

        Returns
            List of arrays of burst sizes, one array per channel.
        """
        kwargs = dict(gamma=gamma, add_naa=add_naa, beta=beta,
                      donor_ref=donor_ref)
        bsize_list = [self.burst_sizes_ich(ich, **kwargs) for ich in
                      range(self.nch)]
        return bsize_list

    def iter_bursts_start_stop(self, ich=0):
        """Iterate over (start, stop) indexes to slice photons for each burst.
        """
        for istart, istop in zip(self.mburst[ich].istart, self.mburst[ich].istop):
            yield istart, istop
    
    def bursts_slice(self, N1=0, N2=-1):
        raise DeprecationWarning("Burst slice is deprecated")
        
    def delete_burst_data(self):
        """Erase all the burst data"""
        for attr in self._burstattrs:
            if hasattr(self, attr):
                delattr(self, attr)
        for name in ('E_fitter', 'S_fitter'):
            if hasattr(self, name):
                delattr(self, name)

    ##
    # Methods for high-level data transformation
    #
    def slice_ph(self, time_s1=0, time_s2=None, s='slice'):
        raise DeprecationWarning("this method is deprecated")
    
    def collapse(self, update_gamma=True, skip_ch=None):
        raise DeprecationWarning("collapsing is deprecated")

    ##
    # Utility methods
    #
    def get_params(self):
        """Returns a plain dict containing only parameters and no arrays.
        This can be used as a summary of data analysis parameters.
        Additional keys `name' and `Names` are added with values
        from `.name` and `.Name()`.
        """
        return self._corr_params[0]
    
    # this method should work as is
    def expand(self, ich=0, alex_naa=False, width=False):
        """Return per-burst D and A sizes (nd, na) and their background counts.

        This method returns for each bursts the corrected signal counts and
        background counts in donor and acceptor channels. Optionally, the
        burst width is also returned.

        Arguments:
            ich (int): channel for the bursts (can be not 0 only in multi-spot)
            alex_naa (bool): if True and self.ALEX, returns burst sizes and
                background also for acceptor photons during accept. excitation
            width (bool): whether return the burst duration (in seconds).

        Returns:
            List of arrays: nd, na, donor bg, acceptor bg.
            If `alex_naa` is True returns: nd, na, naa, bg_d, bg_a, bg_aa.
            If `width` is True returns the bursts duration (in sec.) as last
            element.
        """
        period = self.bp[ich]
        w = self.mburst[ich].width * self.clk_p
        bg_a = self.bg[Ph_sel('0ex1em')][ich][period] * w
        bg_d = self.bg[Ph_sel('0ex0em')][ich][period] * w
        res = [self.nd[ich], self.na[ich]]
        if self.alternated and alex_naa:
            bg_aa = self.bg[Ph_sel(Aex='Aem')][ich][period] * w
            res.extend([self.naa[ich], bg_d, bg_a, bg_aa])
        else:
            res.extend([bg_d, bg_a])
        if width:
            res.append(w)
        return res

    def burst_data_ich(self, ich):
        """Return a dict of burst data for channel `ich`."""
        # this part already works
        bursts = dict()
        if self.num_bursts[ich] == 0:
            return bursts
        bursts['size_raw'] = self.mburst[ich].counts
        bursts['t_start'] = self.mburst[ich].start * self.clk_p
        bursts['t_stop'] = self.mburst[ich].stop * self.clk_p
        bursts['i_start'] = self.mburst[ich].istart
        bursts['i_stop'] = self.mburst[ich].istop

        period = bursts['bg_period'] = self.bp[ich]
        width = self.mburst[ich].width * self.clk_p
        bursts['width_ms'] = width * 1e3
        bursts['bg_ad'] = self.bg[Ph_sel('0ex1em')][ich][period] * width
        bursts['bg_dd'] = self.bg[Ph_sel('0ex0em')][ich][period] * width
        if self.alternated:
            bursts['bg_aa'] = self.bg[Ph_sel('1ex1em')][ich][period] * width
            bursts['bg_da'] = self.bg[Ph_sel('1ex0em')][ich][period] * width

        burst_fields = list(self.burst_fields)
        burst_fields.remove('mburst')
        burst_fields.remove('bp')
        for field in burst_fields:
            if field in self:
                bursts[field] = self[field][ich]
        return bursts

    @property
    def time_max(self):
        """The last recorded time in seconds."""
        if hasattr(self, "_data"):
            return max(d.times[-1] for d in self._data)
        return max(d.times[-1] for d in self._rawevents)

    @property
    def time_min(self):
        """The first recorded time in seconds."""
        if hasattr(self, "_data"):
            return min(d.times[0] for d in self._data)
        return min(d.times[0] for d in self._rawevents)
    
    def ph_in_bursts_mask_ich(self, ich:int=0, ph_sel:Ph_sel=Ph_sel('all'))->np.ndarray[np.bool_]:
        """Return mask of all photons inside bursts for channel `ich`.

        Returns
            Boolean array for photons in channel `ich` and photon
            selection `ph_sel` that are inside any burst.
        """
        data = self._data[ich]
        mask = np.zeros(data.size, dtype=np.bool_)
        for istart, istop in zip(data['istart',], data['istop',]):
            mask[istart:istop] = True
        mask *= np.isin(data.origin.dets, self._detdef.get_stream_ids(ph_sel))
        return mask

    def ph_in_bursts_ich(self, ich=0, ph_sel=Ph_sel('all')):
        """Return timestamps of photons inside bursts for channel `ich`.

        Returns
            Array of photon timestamps in channel `ich` and photon
            selection `ph_sel` that are inside any burst.
        """
        return np.concatenate(self._data[ich]['ph_times', ph_sel])

    ##
    # Background analysis methods
    #
    def calc_bg_cache(self, fun, time_s=60, tail_min_us=500, F_bg=2,
                      error_metrics=None, fit_allph=True,
                      recompute=False):
        """Compute time-dependent background rates for all the channels.

        This version is the cached version of :meth:`calc_bg`.
        This method tries to load the background data from a cache file.
        If a saved background data is not found, it computes
        the background and stores it to disk.

        The arguments are the same as :meth:`calc_bg` with the only addition
        of `recompute` (bool) to force a background recomputation even if
        a cached version is found.

        Form more details on the other arguments see :meth:`calc_bg`.
        """
        self.calc_bg(fun, time_s=time_s, taim_min_us=tail_min_us, F_bg=F_bg,
                     error_metrics=error_metrics, fit_allph=fit_allph)
        for table in self._bg_tables:
            table.save()

    def calc_bg(self, fun:Callable, time_s:float=60.0, tail_min_us:float=500.0, F_bg:float=2.0,
                error_metrics:str=None, fit_allph:bool=True):
        """Compute time-dependent background rates for all the channels.

        Compute background rates for donor, acceptor and both detectors.
        The rates are computed every `time_s` seconds, allowing to
        track possible variations during the measurement.

        Arguments:
            fun (function): function for background estimation (example
                `bg.exp_fit`)
            time_s (float, seconds): compute background each time_s seconds
            tail_min_us (float, tuple or string): min threshold in us for
                photon waiting times to use in background estimation.
                If float is the same threshold for 'all', DD, AD and AA photons
                and for all the channels.
                If a 3 or 4 element tuple, each value is used for 'all', DD, AD
                or AA photons, same value for all the channels.
                If 'auto', the threshold is computed for each stream ('all',
                DD, DA, AA) and for each channel as `bg_F * rate_ml0`.
                `rate_ml0` is an initial estimation of the rate performed using
                :func:`bg.exp_fit` and a fixed threshold (default 250us).
            F_bg (float): when `tail_min_us` is 'auto', is the factor by which
                the initial background estimation if multiplied to compute the
                threshold.
            error_metrics (string): Specifies the error metric to use.
                See :func:`fretbursts.background.exp_fit` for more details.
            fit_allph (bool): if True (default) the background for the
                all-photon is fitted. If False it is computed as the sum of
                backgrounds in all the other streams.

        The background estimation functions are defined in the module
        `background` (conventionally imported as `bg`).

        Example:
            Compute background with `bg.exp_fit` (inter-photon delays MLE
            tail fitting), every 30s, with automatic tail-threshold::

               d.calc_bg(bg.exp_fit, time_s=20, tail_min_us='auto')

        Returns:
            None, all the results are saved in the object itself.
        """
        error_metrics = "KS" if error_metrics is None else error_metrics
        if error_metrics not in ("KS", "CM"):
            raise ValueError("invlaid error_metrics, must be 'KS', 'CM' or None")
        self._bg_err_type = 'err_' + error_metrics
        self._fit_allph = bool(fit_allph)
        self._period_params = tuple(Param(Periods, dict(period=time_s, start_at='time_min', 
                                                  stop_at='over', detdef=data.detdef)) for data in self._data)
        self._period_tables = tuple(data.get_table(period_param) for data, period_param in zip(self._data, self._period_params))
        bg_dct = dict(tail_min_auto=tail_min_us == 'auto', 
                      tail_min=250e-6 if tail_min_us == 'auto' else tail_min_us * 1e-6)
        self._bg_params = tuple(Param(BG, bg_dct, dict(base=period_param)) for period_param in self._period_params)
        self._bg_tables = tuple(data.get_table(bg_param) for data, bg_param in zip(self._data, self._bg_params))
    
    def recompute_bg_lim_ph_p(self, ph_sel, mute=False):
        """
        **Deprecated**
        This method does nothing, Param/Column/Gate system makes obsolete
        
        Recompute self.Lim and self.Ph_p relative to ph selection `ph_sel`
        `ph_sel` is a Ph_sel object selecting the timestamps in which self.Lim
        and self.Ph_p are being computed.
        """
        pass

    ##
    # Burst analysis methods
    #
    def _param_as_mch_array(self, par):
        """Regardless of `par` size, return an arrays with size == nch.
        if `par` is scalar the arrays repeats the calar multiple times
        if `par is a list/array must be of length `nch`.
        """
        assert np.size(par) == 1 or np.size(par) == self.nch
        return np.repeat(par, self.nch) if np.size(par) == 1 else np.asarray(par)

    def bg_from(self, ph_sel):
        """Return the background rates for the specified photon selection.
        """
        return self.bg[ph_sel]
    
    def burst_search(self, L=None, m=10, F=6., P=None, min_rate_cps=None,
                     ph_sel=Ph_sel('all'), compact=False, index_allph=True,
                     c=-1.0, computefret=True, max_rate=False, dither=False,
                     pure_python=False, verbose=False, mute=False, pax=False):
        """Performs a burst search with specified parameters.

        This method performs a sliding-window burst search without
        binning the timestamps. The burst starts when the rate of `m`
        photons is above a minimum rate, and stops when the rate falls below
        the threshold. The result of the burst search is stored in the
        `mburst` attribute (a list of Bursts objects, one per channel)
        containing start/stop times and indexes. By default, after burst
        search, this method computes donor and acceptor counts, it applies
        burst corrections (background, leakage, etc...) and computes
        E (and S in case of ALEX). You can skip these steps by passing
        `computefret=False`.

        The minimum rate can be explicitly specified with the `min_rate_cps`
        argument, or computed as a function of the background rate with the
        `F` argument.

        Parameters:
            m (int): number of consecutive photons used to compute the
                photon rate. Typical values 5-20. Default 10.
            L (int or None): minimum number of photons in burst. If None
                (default) L = m is used.
            F (float): defines how many times higher than the background rate
                is the minimum rate used for burst search
                (`min rate = F * bg. rate`), assuming that `P = None` (default).
                Typical values are 3-9. Default 6.
            P (float): threshold for burst detection expressed as a
                probability that a detected bursts is not due to a Poisson
                background. If not None, `P` overrides `F`. Note that the
                background process is experimentally super-Poisson so this
                probability is not physically very meaningful. Using this
                argument is discouraged.
            min_rate_cps (float or list/array): minimum rate in cps for burst
                start. If not None, it has the precedence over `P` and `F`.
                If non-scalar, contains one rate per each multispot channel.
                Typical values range from 20e3 to 100e3.
            ph_sel (Ph_sel object): defines the "photon selection" (or stream)
                to be used for burst search. Default: all photons.
                See :mod:`fretbursts.ph_sel` for details.
            compact (bool): if True, a photon selection of only one excitation
                period is required and the timestamps are "compacted" by
                removing the "gaps" between each excitation period.
            index_allph: **DEPRECATED, this argument does nothing**
            c (float): correction factor used in the rate vs time-lags relation.
                `c` affects the computation of the burst-search parameter `T`.
                When `F` is not None, `T = (m - 1 - c) / (F * bg_rate)`.
                When using `min_rate_cps`, `T = (m - 1 - c) / min_rate_cps`.
            computefret (bool): if True (default) compute donor and acceptor
                counts, apply corrections (background, leakage, direct
                excitation) and compute E (and S). If False, skip all these
                steps and stop just after the initial burst search.
            max_rate (bool): if True compute the max photon rate inside each
                burst using the same `m` used for burst search. If False
                (default) skip this step.
            dither (bool): if True applies dithering corrections to burst
                counts. Default False. See :meth:`Data.dither`.
            pure_python (bool): **Deprecated, does nothing**
            pax (bool): this has effect only if measurement is PAX.
                In this case, when True computes E using a PAX-enhanced
                formula: ``(2 na) / (2 na + nd + nda)``.
                Otherwise use the usual usALEX formula: ``na / na + nd``.
                Quantities `nd`/`na` are D/A burst counts during D excitation
                period, while `nda` is D emission during A excitation period.

        .. note::
            The background rates are needed, so
            `.calc_bg()` must be called before the burst search.

        Example:
            d.burst_search(m=10, F=6)

        Returns:
            None, all the results are saved in the `Data` object.
        """
        brst_param = dict(channels=(ph_sel, ), m=m, F=F if P is None else P, c=c, asP= P is None, fuse=-1.0)
        self._burst_params = tuple(Param(Burst, brst_param, {'bg':bg_param}) 
                                   for bg_param in self._bg_params)
        self._burst_tables = tuple(data.get_table(burst_param) for data, burst_param 
                                   in zip(self._data, self._burst_params))
        self.mburst = tuple(BurstProxy(burst) for burst in self._burst_tables)
        self._burst_search_postprocess(dither=dither)
    
    def _burst_search_postprocess(self, dither:bool=False):
        self._nph_params = tuple(Param(Nph_bg, {'single':True}, {'base':burst_param}) 
                                 for burst_param in self._burst_params)
        self._nph_tables = tuple(data.get_table(nph_param) for data, nph_param 
                                 in zip(self._data, self._nph_params))
        gamma = np.atleast_1d(self._metadata.get('gamma', 1.0)).astype(np.double)
        if gamma.size == 1:
            gamma = np.repeat(gamma, self.nch)
        elif gamma.size != self.nch:
            raise ValueError("gamma must be either scalar or same size as number of channels")
        self._corr_params = tuple(Param(Ratios, parents={'nph':nph_param},
                                        params=dict(lk=self._metadata.get('leakage', 0.0), 
                                                    dir_ex=self._metadata.get('dir_ex', 0.0),
                                                    gamma=g, beta=self._metadata.get('beta'),
                                                    scheme=_ex_scheme(data.setup))) 
                                  for data, nph_param, g in zip(self._data, self._nph_params, gamma))
        self._corr_tables = tuple(data.get_table(corr_param) for data, corr_param 
                                  in zip(self._data, self._corr_params))
        self.dither()
    
    def calc_ph_num(self, alex_all=False, pure_python=False):
        """
        **DEPRECATED, all values calculated now function as properteis**
        
        Computes number of D, A (and AA) photons in each burst.

        Arguments:
            alex_all (bool): if True and self.ALEX is True, computes also the
                donor channel photons during acceptor excitation (`nda`)
            pure_python (bool): if True, uses the pure python functions even
                when the optimized Cython functions are available.

        Returns:
            Saves `nd`, `na`, `nt` (and eventually `naa`, `nda`) in self.
            Returns None.
        """
        pass

    def fuse_bursts(self, ms=0, process=True, mute=False):
        """Return a new :class:`Data` object with nearby bursts fused together.

        Arguments:
            ms (float): fuse all burst separated by less than `ms` millisecs.
                If < 0 no burst is fused. Note that with ms = 0, overlapping
                bursts are fused.
            process (bool): if True (default), reprocess the burst data in
                the new object applying corrections and computing FRET.
            mute (bool): if True suppress any printed output.

        """
        out = self.copy()
        sep = ms*1e-3
        burst_params = list()
        for bparam in out._burst_params:
            if sep < bparam.params['fuse']:
                bpdict = bparam.params.asdict
                bpdict['fuse'] = sep
                bparam = Param(BG, bpdict, bparam.parents)
            burst_params.append(bparam)
        out._burst_params = tuple(burst_params)
        out._nph_params = tuple(Param(Nph_bg, nph_params.params, {'base':burst_param}) 
                                for nph_params, burst_param in zip(self._nph_params, out._burst_params))
        out._nph_tables = tuple(data.get_table(nph_param) for data, nph_param in zip(out._data, out._nph_params))
        for attr  in ('_dither_ndd', '_dither_nda', '_dither_nad', '_dither_naa', '_corr_params', '_corr_tables'):
            if hasattr(out, attr):
                delattr(out, attr)
        return out
    
    ##
    # Burst selection and filtering
    #
    def select_bursts(self, filter_fun, negate=False, computefret=True,
                      args=None, **kwargs):
        """Return an object with bursts filtered according to `filter_fun`.

        This is the main method to select bursts according to different
        criteria. The selection rule is defined by the selection function
        `filter_fun`. FRETBursts provides a several predefined selection
        functions see :ref:`burst_selection`. New selection
        functions can be defined and passed to this method to implement
        arbitrary selection rules.

        Arguments:
            filter_fun (function): function used for burst selection
            negate (boolean): If True, negates (i.e. take the complementary)
                of the selection returned by `filter_fun`. Default `False`.
            computefret (boolean): If True (default) recompute donor and
                acceptor counts, corrections and FRET quantities (i.e. E, S)
                in the new returned object.
            args (tuple or None): positional arguments for `filter_fun()`

        kwargs:
            Additional keyword arguments passed to `filter_fun()`.

        Returns:
            A new :class:`Data` object containing only the selected bursts.

        Note:
            In order to save RAM, the timestamp arrays (`ph_times_m`)
            of the new Data() points to the same arrays of the original
            Data(). Conversely, all the bursts data (`mburst`, `nd`, `na`,
            etc...) are new distinct objects.
        """
        if args is None:
            args = tuple()
        masks = filter_fun(self, *args, **kwargs)
        if negate:
            masks = tuple(~mask for mask in masks)
        gates = tuple(mask & burst_param.base_gate for mask, burst_param in zip(masks, self._burst_params))
        out = self.copy()
        out._burst_params = tuple(burst_param.regate(gate) for burst_param, gate in zip(out._burst_params, gates))
        out._burst_tables = tuple(data.get_table(burst_param) for data, burst_param in zip(out.data, out._burst_params))
        if hasattr(self, '_nph_param'):
            out._nph_param = tuple(nph_param.regate(gate) for nph_param, gate in zip(out._nph_params, gates))
            out._nph_tables = tuple(data.get_table(nph_param) for data, nph_param, gate in zip(out._data, out._nph_params, gates))
        if hasattr(self, '_corr_param'):
            out._corr_params = tuple(corr_param.reage(gate) for corr_param, gate in zip(out._corr_params, gates))
            out._corr_tables = tuple(data.get_table(corr_param) for data, corr_param in zip(out._data, out._corr_params))
        if hasattr(self, '_dither_ndd'):
            out._dither()
        return out

    def select_bursts_mask(self, filter_fun, negate=False, return_str=False,
                           args=None, **kwargs):
        """Returns mask arrays to select bursts according to `filter_fun`.

        The function `filter_fun` is called to compute the mask arrays for
        each channel.

        This method is useful when you want to apply a selection from one
        object to a second object. Otherwise use :meth:`Data.select_bursts`.

        Arguments:
            filter_fun (function): function used for burst selection
            negate (boolean): If True, negates (i.e. take the complementary)
                of the selection returned by `filter_fun`. Default `False`.
            return_str: if True return, for each channel, a tuple with
                a bool array and a string that can be added to the measurement
                name to indicate the selection. If False returns only
                the bool array. Default False.
            args (tuple or None): positional arguments for `filter_fun()`

        kwargs:
            Additional keyword arguments passed to `filter_fun()`.

        Returns:
            A list of boolean arrays (one per channel) that define the burst
            selection. If `return_str` is True returns a GateGroup object representing
            the full set of Gates applied.

        """
        if args is None:
            args = tuple()
        mask = filter_fun(self, *args, **kwargs)
        if negate:
            mask = ~mask
        gate = mask & self._burst_param.base_gate
        masks = tuple(data.get_gategroup(gate, relative=burst_param.base_bate) 
                      for data, burst_param in zip(self._data, self._burst_params))
        if return_str:
            return masks, gate
        else:
            return masks

    def select_bursts_mask_apply(self, masks, computefret=True, str_sel=''):
        """**Deprecated** for custom masks, use new Gate/GateGroup system.
        """
        raise DeprecationWarning("to apply custom mask, use Gate/GateGroup system instead")

    ##
    # Burst corrections
    #
    def background_correction(self, relax_nt=False, mute=False):
        """**Deprecated** does nothing, all corections applied as soon as specified.
        """
        pass
    
    def leakage_correction(self, mute=False):
        """Apply leakage correction to burst sizes (nd, na,...)
        """
        pass

    def direct_excitation_correction(self, mute=False):
        """**Deprecated** does nothing, all corections applied as soon as specified.
        """
        pass

    def dither(self, lsb:float=2.0, mute:bool=False, generator=None):
        """Add dithering (uniform random noise) to burst counts (nd, na,...).

        The dithering amplitude is the range -0.5*lsb .. 0.5*lsb.
        """
        self._dither_ndd = tuple(corr.dither(Ph_sel('0ex0em'), lsb, generator) for corr in self._corr_tables)
        self._dither_nda = tuple(corr.dither(Ph_sel('0ex1em'), lsb, generator) for corr in self._corr_tables)
        if self._detdef['ex'] != 1:
            self._dither_nad = tuple(corr.dither(Ph_sel('1ex0em'), lsb, generator) for corr in self._corr_tables)
            self._dither_naa = tuple(corr.dither(Ph_sel('1ex1em'), lsb, generator) for corr in self._corr_tables)
    
    def calc_chi_ch(self, E):
        """Calculate the gamma correction prefactor factor `chi_ch` (array).

        Computes `chi_ch`, a channel-dependent prefactor for gamma used
        to correct dispersion of E across channels.

        Returns:
            array of `chi_ch` correction factors (one per spot).
            To apply the correction assign the returned array to `Data.chi_ch`.
            Upon assignment E values for all bursts will be corrected.
        """
        chi_ch = (1 / E.mean() - 1) / (1 / E - 1)
        return chi_ch

    def corrections(self, mute=False):
        """**Deprecated** does nothing, all corections applied as soon as specified.
        """
        pass
    @property
    def leakage(self):
        """Spectral leakage (bleed-through) of D emission in the A channel.
        """
        return -self._corr_params[0].params['corr_mat'][1,0]

    @leakage.setter
    def leakage(self, leakage):
        corr_params = list()
        for corr_param in self._corr_params:
            corr_mat = corr_param.params['corr_mat']
            corr_mat[1,0] = -leakage
            corr_params.append(Param(Ratios, {'corr_mat':corr_mat}, corr_param.parents, corr_param.base_gate))
        self._corr_params = tuple(corr_params)
        self._corr_tables = tuple(data.get_table(corr_param) for data, corr_param in zip(self._data, self._corr_params))

    @property
    def dir_ex(self):
        """Direct excitation correction factor."""
        return -self._corr_params[0].params['corr_mat'][1,3]

    @dir_ex.setter
    def dir_ex(self, direx):
        corr_params = list()
        for corr_param in self._corr_params:
            corr_mat = corr_param.params['corr_mat']
            corr_mat[1,3] = -direx
            corr_params.append(Param(Ratios, {'corr_mat':corr_mat}, corr_param.parents, corr_param.base_gate))
        self._corr_params = tuple(corr_params)
        self._corr_tables = tuple(data.get_table(corr_param) for data, corr_param in zip(self._data, self._corr_params))

    @property
    def beta(self):
        """Beta factor used to correct S (compensates Dex and Aex unbalance).
        """
        return 1/self._corr_params[0].params['corr_mat'][3,3]

    @beta.setter
    def beta(self, beta):
        corr_params = list()
        for corr_param in self._corr_params:
            corr_mat = corr_param.params['corr_mat']
            corr_mat[3,3] = 1/beta
            corr_params.append(Param(Ratios, {'corr_mat':corr_mat}, corr_param.parents, corr_param.base_gate))
        self._corr_params = tuple(corr_params)
        self._corr_tables = tuple(data.get_table(corr_param) for data, corr_param in zip(self._data, self._corr_params))
    
    @property
    def chi_ch(self):
        """Per-channel relative gamma factor. Now identical to gamma"""
        return self.gamma

    @chi_ch.setter
    def chi_ch(self, value):
        self.gamma = value

    @property
    def gamma(self):
        """Gamma correction factor (compensates DexDem and DexAem unbalance).
        """
        return np.array([corr_param.params['corr_mat'][0,0] for corr_param in self._corr_params])

    @gamma.setter
    def gamma(self, gamma):
        gamma = np.asarray(gamma, dtype=np.double)
        if gamma.size != 1 or gamma.size != self.nch:
            raise ValueError("gamma must be scalar or 1D array of the same size as nch")
        gamma = np.repeat(gamma, self.nch) if gamma.size == 1 else gamma
        self._corr_params = tuple(Param(Ratios, dict(lk=self.leakage, dir_ex=self.dir_ex, 
                                                     beta=self.beta, gamma=g, 
                                                     scheme=_ex_scheme(data.detup)), 
                                        parents=corr_param.parents, gate=corr_param.base_gate) 
                                  for g, data, corr_param in zip(gamma, self._data, self._corr_params))
        self._corr_tables = tuple(data.get_table(corr_param) for data, corr_param 
                                  in zip(self._data, self._corr_params))
            

    def get_gamma_array(self):
        """Get the array of gamma factors, one per ch.

        It always returns an array of gamma factors regardless of
        whether `self.gamma` is scalar or array.

        Each element of the returned array is multiplied by `chi_ch`.
        """
        gamma = self.gamma
        G = np.repeat(gamma, self.nch) if np.size(gamma) == 1 else gamma
        G *= self.chi_ch
        return G

    def get_leakage_array(self):
        """Get the array of leakage coefficients, one per ch.

        It always returns an array of leakage coefficients regardless of
        whether `self.leakage` is scalar or array.

        Each element of the returned array is multiplied by `chi_ch`.
        """
        leakage = self.leakage
        Lk = np.r_[[leakage] * self.nch] if np.size(leakage) == 1 else leakage
        Lk *= self.chi_ch
        return Lk

    ##
    # Methods to compute burst quantities: FRET, S, SBR, max_rate, etc ...
    #
    def calc_sbr(self, ph_sel=Ph_sel('all'), gamma=1.):
        """Return Signal-to-Background Ratio (SBR) for each burst.

        Arguments:
            ph_sel (Ph_sel object): object defining the photon selection
                for which to compute the sbr. Changes the photons used for
                burst size and the corresponding background rate. Valid values
                here are Ph_sel('all'), Ph_sel(Dex='Dem'), Ph_sel(Dex='Aem').
                See :mod:`fretbursts.ph_sel` for details.
            gamma (float): **Deprecated, value is ignrored**
        Returns:
            A list of arrays (one per channel) with one value per burst.
            The list is also saved in `sbr` attribute.
        """
        self._sbr_ph_sel = ph_sel
        return self.sbr

    def calc_burst_ph_func(self, func:Callable, func_kw:dict, ph_sel:Ph_sel=Ph_sel('all'),
                           compact:bool=False, ich:int=0):
        """Evaluate a scalar function from photons in each burst.

        This method allow calling an arbitrary function on the photon
        timestamps of each burst. For example if `func` is `np.mean` it
        computes the mean time in each bursts.

        Arguments:
            func (callable): function that takes as first argument an array of
                timestamps for one burst.
            func_kw (callable): additional arguments to be passed  `func`.
            ph_sel (Ph_sel object): object defining the photon selection.
                See :mod:`fretbursts.ph_sel` for details.
            compact (bool): if True, a photon selection of only one excitation
                period is required and the timestamps are "compacted" by
                removing the "gaps" between each excitation period.

        Returns:
            A list (on element per channel) array. The array size is equal to
            the number of bursts in the corresponding channel.
        """
        if compact:
            self._assert_compact(ph_sel)

        kwargs = dict(func=func, func_kw=func_kw, compact=compact)
        if compact:
            kwargs.update(alex_period=self.alex_period,
                          excitation_width=self._excitation_width(ph_sel))

        results_mch = [burst_ph_stats(ph, bursts, mask=mask, **kwargs)
                       for ph, mask, bursts in
                       zip(self.iter_ph_times(),
                           self.iter_ph_masks(ph_sel=ph_sel),
                           self.mburst)]
        return results_mch

    def calc_max_rate(self, m, ph_sel=Ph_sel('all'), compact=False, c=-1.0):
        """Compute the max m-photon rate reached in each burst.

        Arguments:
            m (int): number of timestamps to use to compute the rate.
                As for burst search, typical values are 5-20.
            ph_sel (Ph_sel object): object defining the photon selection.
                See :mod:`fretbursts.ph_sel` for details.
            compact: **No longer used, dummy parameter**
            c (float): **No longer used, dummy parameter, now always -1.0** 
                this parameter was used in the definition of the
                rate estimator which is `(m - 1 - c) / t[last] - t[first]`.
                For more details see :func:`.phtools.phrates.mtuple_rates`.
        """
        self._max_rate_m = m
        self._max_rate_ph_sel = ph_sel
        
    def calc_fret(self, count_ph=False, corrections=True, dither=False,
                  mute=False, pure_python=False, pax=False):
        """Compute FRET (and stoichiometry if ALEX) for each burst.

        This is an high-level functions that can be run after burst search.
        By default, it will count Donor and Acceptor photons, perform
        corrections (background, leakage), and compute gamma-corrected
        FRET efficiencies (and stoichiometry if ALEX).

        Arguments:
            count_ph (bool): if True (default), calls :meth:`calc_ph_num` to
                counts Donor and Acceptor photons in each bursts
            corrections (bool):  if True (default), applies background and
                bleed-through correction to burst data
            dither (bool): whether to apply dithering to burst size.
                Default False.
            mute (bool): whether to mute all the printed output. Default False.
            pure_python (bool): if True, uses the pure python functions even
                when the optimized Cython functions are available.
            pax (bool): this has effect only if measurement is PAX.
                In this case, when True computes E using a PAX-enhanced
                formula: ``(2 na) / (2 na + nd + nda)``.
                Otherwise use the usual usALEX formula: ``na / na + nd``.
                Quantities `nd`/`na` are D/A burst counts during D excitation
                period, while `nda` is D emission during A excitation period.

        Returns:
            None, all the results are saved in the object.
        """
        if dither:
            self.dither()
        for attr in ('ES_binwidth', 'ES_hist', 'E_fitter', 'S_fitter'):
            # E_fitter and S_fitter are only attributes
            # so we cannot use the membership syntax (attr in self)
            if hasattr(self, attr):
                delattr(self, attr)

    ##
    # Methods for measurement info
    #
    def status(self, add="", noname=False):
        """Return a string with burst search, corrections and selection info.
        """
        name = "" if noname else self.name
        s = name
        if 'L' in self:  # burst search has been done
            if 'rate_th' in self:
                s += " BS_%s L%d m%d MR%d" % (self.ph_sel, self.L, self.m,
                                              np.mean(self.rate_th) * 1e-3)
            else:
                P_str = '' if self.P is None else ' P%s' % self.P
                s += " BS_%s L%d m%d F%.1f%s" % \
                     (self.ph_sel, self.L, self.m, np.mean(self.F), P_str)
        s += " G%.3f" % np.mean(self.gamma)
        if 'bg_fun' in self: s += " BG%s" % self.bg_fun.__name__[:-4]
        if 'bg_time_s' in self: s += "-%ds" % self.bg_time_s
        if 'fuse' in self: s += " Fuse%.1fms" % self.fuse
        if 'bg_corrected' in self and self.bg_corrected:
            s += " bg"
        if 'leakage_corrected' in self and self.leakage_corrected:
            s += " Lk%.3f" % np.mean(self.leakage*100)
        if 'dir_ex_corrected' in self and self.dir_ex_corrected:
            s += " dir%.1f" % (self.dir_ex*100)
        if 'dithering' in self and self.dithering:
            s += " Dith%d" % self.lsb
        if 's' in self: s += ' '.join(self.s)
        return s + add

    @property
    def name(self):
        """Measurement name: last subfolder + file name with no extension."""
        if not hasattr(self, '_name'):
            basename = str(os.path.splitext(os.path.basename(self._metadata['fname']))[0])
            name = basename
            last_dir = str(os.path.basename(os.path.dirname(self._metadata['fname'])))
            if len(last_dir) > 0:
                name = '_'.join([last_dir, basename])
            self._name = name
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def Name(self, add=""):
        """Return short filename + status information."""
        n = self.status(add=add)
        return n

    def __repr__(self):
        return self.status()

    def stats(self, string=False):
        """Print common statistics (BG rates, #bursts, mean size, ...)"""
        s = print_burst_stats(self)
        if string:
            return s
        else:
            print(s)

    ##
    # FRET fitting methods
    #
    # TODO: continue working on the fit methods
    def fit_E_m(self, E1=-1, E2=2, weights='size', gamma=1.):
        """Fit E in each channel with the mean using bursts in [E1,E2] range.

        Note:
            This two fitting are equivalent (but the first is much faster)::

                fit_E_m(weights='size')
                fit_E_minimize(kind='E_size', weights='sqrt')

            However `fit_E_minimize()` does not provide a model curve.
        """
        Mask = self.select_bursts_mask(select_bursts.E, E1=E1, E2=E2)

        fit_res, fit_model_F = np.zeros((self.nch, 2)), np.zeros(self.nch)
        for ich, (nd, na, E, mask) in enumerate(zip(
                self.nd, self.na, self.E, Mask)):
            w = fret_fit.get_weights(nd[mask], na[mask],
                                     weights=weights, gamma=gamma)
            # Compute weighted mean
            fit_res[ich, 0] = np.dot(w, E[mask])/w.sum()
            # Compute weighted variance
            fit_res[ich, 1] = np.sqrt(
                np.dot(w, (E[mask] - fit_res[ich, 0])**2)/w.sum())
            fit_model_F[ich] = mask.sum()/mask.size

        fit_model = lambda x, p: norm.pdf(x, p[0], p[1])
        self.fit_E_res = fit_res
        self.fit_E_name = 'Moments'
        self.E_fit = fit_res[:, 0]
        self.fit_E_curve = True
        self.fit_E_E1 = E1
        self.fit_E_E2 = E2
        self.fit_E_model = fit_model
        self.fit_E_model_F = fit_model_F
        self.fit_E_calc_variance()
        return self.E_fit

    def fit_E_ML_poiss(self, E1=-1, E2=2, method=1, **kwargs):
        """ML fit for E modeling size ~ Poisson, using bursts in [E1,E2] range.
        """
        assert method in [1, 2, 3]
        fit_fun = {1: fret_fit.fit_E_poisson_na, 2: fret_fit.fit_E_poisson_nt,
                   3: fret_fit.fit_E_poisson_nd}
        Mask = self.select_bursts_mask(select_bursts.E, E1=E1, E2=E2)
        fit_res = np.zeros(self.nch)
        for ich, mask in zip(range(self.nch), Mask):
            nd, na, bg_d, bg_a = self.expand(ich)
            bg_x = bg_d if method == 3 else bg_a
            fit_res[ich] = fit_fun[method](nd[mask], na[mask],
                                           bg_x[mask], **kwargs)
        self.fit_E_res = fit_res
        self.fit_E_name = 'MLE: na ~ Poisson'
        self.E_fit = fit_res
        self.fit_E_curve = False
        self.fit_E_E1 = E1
        self.fit_E_E2 = E2
        self.fit_E_calc_variance()
        return self.E_fit

    def fit_E_ML_binom(self, E1=-1, E2=2, **kwargs):
        """ML fit for E modeling na ~ Binomial, using bursts in [E1,E2] range.
        """
        Mask = self.select_bursts_mask(select_bursts.E, E1=E1, E2=E2)
        fit_res = np.array([fret_fit.fit_E_binom(_d[mask], _a[mask], **kwargs)
                            for _d, _a, mask in zip(self.nd, self.na, Mask)])
        self.fit_E_res = fit_res
        self.fit_E_name = 'MLE: na ~ Binomial'
        self.E_fit = fit_res
        self.fit_E_curve = False
        self.fit_E_E1 = E1
        self.fit_E_E2 = E2
        self.fit_E_calc_variance()
        return self.E_fit

    def fit_E_minimize(self, kind='slope', E1=-1, E2=2, **kwargs):
        """Fit E using method `kind` ('slope' or 'E_size') and bursts in [E1,E2]
        If `kind` is 'slope' the fit function is fret_fit.fit_E_slope()
        If `kind` is 'E_size' the fit function is fret_fit.fit_E_E_size()
        Additional arguments in `kwargs` are passed to the fit function.
        """
        assert kind in ['slope', 'E_size']
        # Build a dictionary fun_d so we'll call the function fun_d[kind]
        fun_d = dict(slope=fret_fit.fit_E_slope,
                     E_size=fret_fit.fit_E_E_size)
        Mask = self.select_bursts_mask(select_bursts.E, E1=E1, E2=E2)
        fit_res = np.array([fun_d[kind](nd[mask], na[mask], **kwargs)
                            for nd, na, mask in
                            zip(self.nd, self.na, Mask)])
        fit_name = dict(slope='Linear slope fit', E_size='E_size fit')
        self.fit_E_res = fit_res
        self.fit_E_name = fit_name[kind]
        self.E_fit = fit_res
        self.fit_E_curve = False
        self.fit_E_E1 = E1
        self.fit_E_E2 = E2
        self.fit_E_calc_variance()
        return self.E_fit

    def fit_E_two_gauss_EM(self, fit_func=two_gaussian_fit_EM,
                           weights='size', gamma=1., **kwargs):
        """Fit the E population to a Gaussian mixture model using EM method.
        Additional arguments in `kwargs` are passed to the fit_func().
        """
        fit_res = np.zeros((self.nch, 5))
        for ich, (nd, na, E) in enumerate(zip(self.nd, self.na, self.E)):
            w = fret_fit.get_weights(nd, na, weights=weights, gamma=gamma)
            fit_res[ich, :] = fit_func(E, weights=w, **kwargs)
        self.fit_E_res = fit_res
        self.fit_E_name = fit_func.__name__
        self.E_fit = fit_res[:, 2]
        self.fit_E_curve = True
        self.fit_E_model = two_gauss_mix_pdf
        self.fit_E_model_F = np.repeat(1, self.nch)
        return self.E_fit

    def fit_E_generic(self, E1=-1, E2=2, fit_fun=two_gaussian_fit_hist,
                      weights=None, gamma=1., **fit_kwargs):
        """Fit E in each channel with `fit_fun` using burst in [E1,E2] range.
        All the fitting functions are defined in
        :mod:`fretbursts.fit.gaussian_fitting`.

        Parameters:
            weights (string or None): specifies the type of weights
                If not None `weights` will be passed to
                `fret_fit.get_weights()`. `weights` can be not-None only when
                using fit functions that accept weights (the ones ending in
                `_hist` or `_EM`)
            gamma (float): passed to `fret_fit.get_weights()` to compute
                weights

        All the additional arguments are passed to `fit_fun`. For example `p0`
        or `mu_fix` can be passed (see `fit.gaussian_fitting` for details).

        Note:
            Use this method for CDF/PDF or hist fitting.
            For EM fitting use :meth:`fit_E_two_gauss_EM()`.
        """
        if fit_fun.__name__.startswith("gaussian_fit"):
            fit_model = lambda x, p: norm.pdf(x, p[0], p[1])
            if 'mu0' not in fit_kwargs: fit_kwargs.update(mu0=0.5)
            if 'sigma0' not in fit_kwargs: fit_kwargs.update(sigma0=0.3)
            iE, nparam = 0, 2
        elif fit_fun.__name__ == "two_gaussian_fit_hist_min_ab":
            fit_model = two_gauss_mix_ab
            if 'p0' not in fit_kwargs:
                fit_kwargs.update(p0=[0, .05, 0.5, 0.6, 0.1, 0.5])
            iE, nparam = 3, 6
        elif fit_fun.__name__.startswith("two_gaussian_fit"):
            fit_model = two_gauss_mix_pdf
            if 'p0' not in fit_kwargs:
                fit_kwargs.update(p0=[0, .05, 0.6, 0.1, 0.5])
            iE, nparam = 2, 5
        else:
            raise ValueError("Fitting function not recognized.")

        Mask = self.select_bursts_mask(select_bursts.E, E1=E1, E2=E2)

        fit_res, fit_model_F = np.zeros((self.nch, nparam)), np.zeros(self.nch)
        for ich, (nd, na, E, mask) in enumerate(zip(
                self.nd, self.na, self.E, Mask)):
            if '_hist' in fit_fun.__name__ or '_EM' in fit_fun.__name__:
                if weights is None:
                    w = None
                else:
                    w = fret_fit.get_weights(nd[mask], na[mask],
                                             weights=weights, gamma=gamma)
                fit_res[ich, :] = fit_fun(E[mask], weights=w, **fit_kwargs)
            else:
                # Non-histogram fits (PDF/CDF) do not support weights
                fit_res[ich, :] = fit_fun(E[mask], **fit_kwargs)
            fit_model_F[ich] = mask.sum()/mask.size

        # Save enough info to generate a fit plot (see hist_fret in burst_plot)
        self.fit_E_res = fit_res
        self.fit_E_name = fit_fun.__name__
        self.E_fit = fit_res[:, iE]
        self.fit_E_curve = True
        self.fit_E_E1 = E1
        self.fit_E_E2 = E2
        self.fit_E_model = fit_model
        self.fit_E_model_F = fit_model_F
        self.fit_E_weights = weights
        self.fit_E_gamma = gamma
        self.fit_E_kwargs = fit_kwargs
        return self.E_fit

    def fit_from(self, D):
        """Copy fit results from another Data() variable.
        Now that the fit methods accept E1,E1 parameter this probabily useless.
        """
        # NOTE Are 'fit_guess' and 'fit_fix' still used ?
        fit_data = ['fit_E_res', 'fit_E_name', 'E_fit', 'fit_E_curve',
                    'fit_E_E1', 'fit_E_E2=E2', 'fit_E_model',
                    'fit_E_model_F', 'fit_guess', 'fit_fix']
        for name in fit_data:
            if name in D:
                setattr(self, name, D[name])
        # Deal with the normalization to the number of bursts
        self.fit_model_F = np.r_[[old_E.size/new_E.size 
                                  for old_E, new_E in zip(D.E, self.E)]]
    
    def fit_E_calc_variance(self, weights='sqrt', dist='DeltaE',
                            E_fit=None, E1=-1, E2=2):
        """Compute several versions of WEIGHTED std.dev. of the E estimator.
        `weights` are multiplied *BEFORE* squaring the distance/error
        `dist` can be 'DeltaE' or 'SlopeEuclid'

        Note:
            This method is still experimental
        """
        assert dist in ['DeltaE', 'SlopeEuclid']
        if E_fit is None:
            E_fit = self.E_fit
            E1 = self.fit_E_E1 if 'fit_E_E1' in self else -1
            E2 = self.fit_E_E2 if 'fit_E_E2' in self else 2
        else:
            # If E_fit is not None the specified E1,E2 range is used
            if E1 < 0 and E2 > 1:
                pprint('WARN: E1 < 0 and E2 > 1 (wide range of E eff.)\n')
        if np.size(E_fit) == 1 and self.nch > 0:
            E_fit = np.repeat(E_fit, self.nch)
        assert np.size(E_fit) == self.nch

        E_sel = [Ei[(Ei > E1)*(Ei < E2)] for Ei in self.E]
        Mask = self.select_bursts_mask(select_bursts.E, E1=E1, E2=E2)

        E_var = np.zeros(np.nch)
        E_var_bu = np.zeros(self.nch)
        E_var_ph = np.zeros()
        for i, (Ech, nt, mask) in enumerate(zip(E_sel, self.nt, Mask)):
            nt_s = nt[mask]
            nd_s, na_s = self.nd[i][mask], self.na[i][mask]
            w = fret_fit.get_weights(nd_s, na_s, weights=weights)
            info_ph = nt_s.sum()
            info_bu = nt_s.size

            if dist == 'DeltaE':
                distances = (Ech - E_fit[i])
            elif dist == 'SlopeEuclid':
                distances = fret_fit.get_dist_euclid(nd_s, na_s, E_fit[i])

            residuals = distances * w
            var = np.mean(residuals**2)
            var_bu = np.mean(residuals**2)/info_bu
            var_ph = np.mean(residuals**2)/info_ph
            #lvar = np.mean(log(residuals**2))
            #lvar_bu = np.mean(log(residuals**2)) - log(info_bu)
            #lvar_ph = np.mean(log(residuals**2)) - log(info_ph)
            E_var[i], E_var_bu[i], E_var_ph[i] = var, var_bu, var_ph
            assert (-np.isnan(E_var[i])).all() # check there is NO NaN
        self.E_var = E_var
        self.E_var_bu = E_var_bu
        self.E_var_ph = E_var_ph
        return E_var