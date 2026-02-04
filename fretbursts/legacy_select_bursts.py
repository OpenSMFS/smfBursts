#
# FRETBursts - A single-molecule FRET burst analysis toolkit.
#
# Copyright (C) 2014-2016 The Regents of the University of California,
#               Antonino Ingargiola <tritemio@gmail.com>
#
"""

The module :mod:`select_bursts` defines functions to select
bursts according to different criteria.

These functions are usually passed to
:meth:`Data.select_bursts() <fretbursts.burstlib.Data.select_bursts>`.
For example::

    ds = d.select_bursts(select_bursts.E, th1=0.2, th2=0.6)

returns a new object `ds` containing only the bursts of `d` that pass the
specified selection criterium (`E` between 0.2 and 0.6 in this case).

"""

import numpy as np
from scipy import stats

from .datamodel.tables import Gate, GateGroup, Column, Param, GG_all
from .datamodel.gates import make_gt_gate, make_gte_gate, make_lt_gate, make_lte_gate
from .datamodel.gates import make_ellipsoid_inclusive_gate, make_upper_inclusive_percentile_gate
from .utils.misc import clk_to_s as _clk_to_s
from .datamodel.utils import _tuple_kwarg
from .ph_sel import Ph_sel, DetDef
from .legacy_burstlib import Data

## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  BURSTS SELECTION FUNCTIONS
#

def _gate_gt_new(param:Param, mn:float, col:str, keytup=None)->Gate:
    keytup = _tuple_kwarg(keytup)
    col = Column(param, col, keytup)
    return make_gt_gate(col, mn)

def _gate_gte_new(param:Param, mn:float, col:str, keytup=None, offset=None, fill=None)->Gate:
    keytup = _tuple_kwarg(keytup)
    if offset is None:
        col = Column(param, col, keytup)
    elif fill is None:
        col = Column(param, col, keytup, offset=offset)
    else:
        col = Column(param, col, keytup, offset=offset, fill=fill)
    return make_gte_gate(col, mn)


def _range_lte_gte_new(param:Param, mn:float, mx:float, col:str, keytup:tuple=None)->GateGroup:
    if mn > mx:
        raise ValueError('Threshold {col}1 (%.2f) must be <= of {col}2 (%.2f)' % (mn, mx))
    gate = param.base_gate
    if mn != -np.inf:
        gate &= _gate_gte_new(param, mn, col, keytup)
    if mx != np.inf:
        gate &= ~_gate_gt_new(param, mx, col, keytup)
    return gate
    

## Selection on E or S values
def E(data:Data, E1:float=-np.inf, E2:float=np.inf)->GateGroup:
    """Select bursts with E between E1 and E2."""
    return tuple(_range_lte_gte_new(corr_param, E1, E2, 'E') for corr_param in data._corr_params)

def S(data:Data, S1:float=-np.inf, S2:float=np.inf)->GateGroup:
    """Select bursts with S between S1 and S2."""
    return tuple(_range_lte_gte_new(corr_param, S1, S2, 'S') for corr_param in data._corr_params)

def ES(data:Data, E1:float=-np.inf, E2:float=np.inf, S1:float=-np.inf, S2:float=np.inf, rect:bool=True)->GateGroup:
    """Select bursts with E between `E1` and `E2` and S between `S1` and `S2`.

    When `rect` is True the selection is rectangular otherwise is elliptical.

    See also:
        For plotting the ES region selected by (`E1`, `E2`, `S1`, `S2`, `rect`):

        - :func:`fretbursts.burst_plot.plot_ES_selection`
    """
    if rect:
        return ES_rect(data, E1, E2, S1, S2)
    else:
        return ES_ellips(data, E1=E1, E2=E2, S1=S1, S2=S2)

def ES_rect(data:Data, E1:float=-np.inf, E2:float=np.inf, S1:float=-np.inf, S2:float=np.inf)->GateGroup:
    """Select bursts inside the rectangle defined by E1, E2, S1, S2.
    """
    return tuple(_range_lte_gte_new(corr_param, E1, E2, 'E') & _range_lte_gte_new(corr_param, S1, S2, 'S') 
                 for corr_param in data._corr_params)

def ES_ellips(data:Data, E1:float=-1e3, E2:float=1e3, S1:float=-1e3, S2:float=1e3):
    """Select bursts with E-S inside an ellipsis inscribed in E1, E2, S1, S2.
    """
    return tuple(make_ellipsoid_inclusive_gate(Column(corr_param, 'E'), Column(corr_param, 'E'), 
                                               cx=np.mean([E1,E2]), cy=np.mean([S1,S2]), 
                                               w=abs(E2-E1), h=abs(S2-S1)) 
                 for corr_param in data._corr_params)


## Selection on static burst size, width or period
def period(data:Data, bp1:int=0, bp2:int=None):
    """Select bursts from period bp1 to period bp2 (included)."""
    t1 = data._period_tables[0]['start'][bp1]
    gates = (burst_param.base_gate for burst_param in data._burst_params)
    if bp1 is not None:
        gates = (gate&_gate_gte_new(burst_param, t1, 'start') 
                 for gate, burst_param in zip(gates, data._burst_params))
    if bp1 is not None:
        t2 = data._period_tables[0]['stop'][bp2]
        gates = (gate&(~_gate_gt_new(burst_param, t2, 'start')) 
                 for gate, burst_param in zip(gates, data._burst_params))
    return tuple(gates)

def time(data:Data, time_s1:float=0, time_s2:float=np.inf)->GateGroup:
    """Select the burst starting from time_s1 to time_s2 (in seconds)."""
    return tuple(_range_lte_gte_new(burst_param, time_s1*data.clk_p, time_s2*data.clk_p, 'istarttime')
                 for burst_param in data._burst_params)
    

def nd(data:Data, th1:float=20, th2:float=np.inf)->GateGroup:
    """Select bursts with (nd >= th1) and (nd <= th2)."""
    return tuple(_range_lte_gte_new(corr_param, th1, th2, 'nph_c', Ph_sel('0ex0em'))
                 for corr_param in data._corr_params)


def na(data:Data, th1:float=20.0, th2:float=np.inf)->GateGroup:
    """Select bursts with (na >= th1) and (na <= th2)."""
    return tuple(_range_lte_gte_new(corr_param, th1, th2, 'nph_c', Ph_sel('0ex1em'))
                 for corr_param in data._corr_params)


def naa(data:Data, th1:float=20, th2:float=np.inf, gamma=1., beta=1., donor_ref=True,
        naa_comp=False, naa_aexonly=True):
    """Select bursts with (naa >= th1) and (naa <= th2).

    The `naa` quantity are corrected with gamma and beta values.

    Arguments:
        th1, th2 (floats): lower (`th1`) and upper (`th2`) bounds for
            selecting `naa`. By default `th2 = inf` (i.e. no upper limit).
        gamma, beta (floats): **Ignored** legacy param. use corrected values of gamma/beta in data
        donor_ref (bool): **Ignored** legacy param for defining convention, now use FRETBursts universal convention
        na_comp (bool): **Ignored** legacy param for defining convention, now use FRETBursts universal convention
        naa_aexonly (bool): **Ignored** legacy param for defining convention, now use FRETBursts universal convention
        naa_comp (bool): **Ignored** legacy param for defining convention, now use FRETBursts universal convention

    See also:
        - :meth:`fretbursts.burstlib.Data.burst_sizes_pax_ich`.
    """
    return tuple(_range_lte_gte_new(data._corr_param, th1, th2, 'nph_c', Ph_sel('1ex1em'))
                 for corr_param in data._corr_params)
    


def size(data:Data, th1:float=20, th2:float=np.inf, add_naa:bool=False, ph_sel:Ph_sel=None, 
         gamma=1., beta=1., donor_ref=True, naa_aexonly=False, naa_comp=False,
         na_comp=False)->GateGroup:
    """Select bursts with burst sizes (i.e. counts) between `th1` and `th2`.

    The burst size is the number of photon in a burst. By default it
    includes all photons during donor excitation (`Dex`).
    To add *AexAem* photons to the burst size use `add_naa=True`.
    If `ph_sel` is specified use a PAX-specific definition of size
    as defined in :meth:`fretbursts.burstlib.Data.burst_sizes_pax_ich`.

    Arguments:
        d (Data object): the object containing the measurement.
        ich (int): the spot number, only relevant for multi-spot. In
            single-spot data there is only CH-0 so this argument may be
            omitted. Default 0.
        th1, th2 (floats): select bursts with ``th1 <= size <= th2``.
            Default `th2 = inf` (i.e. no upper limit).
        add_naa (boolean): when True, add AexAem photons when computing burst
            burst size. Default False.
        ph_sel (Ph_sel object or None): if specified, the stream on which the
            Gate will be created.
        gamma, beta (floats): 
            **Ignored** legacy param. use corrected values of gamma/beta in data
        donor_ref (bool): **Ignored** legacy param for defining convention, now use FRETBursts universal convention
        na_comp (bool): **Ignored** legacy param for defining convention, now use FRETBursts universal convention
        naa_aexonly (bool): **Ignored** legacy param for defining convention, now use FRETBursts universal convention
        naa_comp (bool): **Ignored** legacy param for defining convention, now use FRETBursts universal convention

    Returns:
        GateGroup for selection

    See also:
        - :meth:`fretbursts.burstlib.Data.burst_sizes_ich`.
        - :meth:`fretbursts.burstlib.Data.burst_sizes_pax_ich`.
    """
    if ph_sel is None:
        ph_sel = Ph_sel('0ex_1ex1em') if add_naa else Ph_sel('0ex')
    return tuple(_range_lte_gte_new(corr_param, th1, th2, 'nph_c', ph_sel)
                 for corr_param in data._corr_params)
    

def width(data:Data, th1:float=0.5, th2:float=np.inf)->GateGroup:
    """Select bursts with (width >= th1) and (width <= th2), in ms."""
    return tuple(_range_lte_gte_new(burst_param, th1, th2, 'dur', ('istarttime', 'istoptime'))
                 for burst_param in data._burst_params)


def sbr(data:Data, th1:float=0, th2:float=np.inf)->GateGroup:
    """Select bursts with SBR between `th1` and `th2`."""
    if 'sbr' not in data:
        data.calc_sbr()
    return tuple(_range_lte_gte_new(burst_param, th1, th2, 'sbr', (data._sbr_ph_sel, 'istarttime', 'istoptime'))
                 for burst_param in data._burst_params)

def peak_phrate(data:Data, th1:float=0, th2:float=np.inf)->GateGroup:
    """Select bursts with peak phtotons rate between th1 and th2 (cps).

    Note that this function requires to compute the peak photon rate
    first using :meth:`fretbursts.burstlib.Data.calc_max_rate`.
    """
    return tuple(_range_lte_gte_new(burst_param, th1, th2, 'max_rate', (data._max_rate_ph_sel, data._max_rate_m))
                 for burst_param in data._burst_params)


def brightness(data:Data, th1:float=0, th2:float=np.inf, add_naa=False, ph_sel:Ph_sel=Ph_sel('all'), gamma=1, beta=1,
               donor_ref=True)->GateGroup:
    """Select bursts with size/width between th1 and th2 (cps).
    """
    return tuple(_range_lte_gte_new(corr_param, th1, th2, 'brightness_c', ph_sel) 
                 for corr_param in data._corr_params)
    

def nda_percentile(data:Data, q:float=50, low:bool=False, ph_sel:Ph_sel=Ph_sel('0ex0em'), gamma=1., add_naa=False)->tuple[GateGroup,...]:
    """Select bursts with SIZE >= q-percentile (or <= if `low` is True)

    `gamma` and `add_naa` are deprecated
    """
    cols = (Column(corr_param, 'nph_bg', (ph_sel, )) for corr_param in data._corr_params)
    return tuple(make_upper_inclusive_percentile_gate(col, q) for col in cols)

def topN_nda(d, ich=0, N=500, gamma=1., add_naa=False):
    """
    **DEPRECATED, used percentile gate instead**
    
    Select the N biggest bursts in the channel.

    `gamma` and `add_naa` are passed to
    :meth:`fretbursts.burstlib.Data.burst_sizes_ich` to compute the burst size.
    """
    raise DeprecationWarning("this function is deprecated, user percentile gate instead")
    
    
def topN_max_rate(d, ich=0, N=500):
    """Select `N` bursts with the highest max burst rate.
    """
    raise DeprecationWarning("this function is deprecated, user percentile gate instead")
    
def topN_sbr(d, ich=0, N=200):
    """Select the top `N` bursts with highest SBR."""
    raise DeprecationWarning("this function is deprecated, user percentile gate instead")


## Selection on burst time (nearby, overlapping or isolated bursts)
def single(data:Data, th:float=1.0)->GateGroup:
    """Select bursts that are at least th millisec apart from the others."""
    th = th*1e-3
    return tuple(_gate_gte_new(burst_param, th, 'sep', 0) & _gate_gte_new(burst_param, th, 'sep', 1)
                 for burst_param in data._burst_params)
    

def consecutive(data:Data, th1:float=0.0, th2:float=np.inf, kind:str='both')->tuple[GateGroup,...]:
    """Select consecutive bursts with th1 <= separation <= th2 (in sec.).

    Arguments:
        kind (string): valid values are 'first' to select the first burst
            of each pair, 'second' to select the second burst of each pair
            and 'both' to select both bursts in each pair.
    """
    assert th1 <= th2, 'th1 (%.2f) must be <= of th2 (%.2f)' % (th1, th2)
    if kind not in ('first', 'second', 'both'):
        raise ValueError("kind must be 'first', 'second', or 'both'")
    gates = (burst_param.base_gate for burst_param in data._burst_params)
    if kind in ('first', 'both'):
        gates = (gate & _range_lte_gte_new(burst_param, th1, th2, 'sep', (0, np.inf)) 
                 for gate, burst_param in zip(gates, data._burst_params))
    if kind in ('second', 'both'):
        gates = (gate & _range_lte_gte_new(burst_param, th1, th2, 'sep', (1, np.inf)) 
                 for gate, burst_param in zip(gates, data._burst_params))
    return tuple(gates)

## Selection on burst size vs BG
def nd_bg(data:Data, F:float=5.0)->tuple[GateGroup,...]:
    """Select bursts with (nd >= bg_dd*F)."""
    return tuple(_gate_gte_new(nph_param, F, 'sbr', Ph_sel('0ex0em')) 
                 for nph_param in data._nph_params)


def na_bg(data:Data, F:float=5.0)->tuple[GateGroup,...]:
    """Select bursts with (na >= bg_ad*F)."""
    return tuple(GateGroup.as_gategroup(_gate_gte_new(nph_param, F, 'sbr', (Ph_sel('0ex1em')))) 
                 for nph_param in data._nph_params)
    
# TOOD: keep working from here
def naa_bg(data:Data, F:float=5.0)->tuple[GateGroup,...]:
    """Select bursts with (naa >= bg_aa*F)."""
    return tuple(GateGroup.as_gategroup(_gate_gte_new(nph_param, F, 'sbr', (Ph_sel('1ex1em')))) 
                 for nph_param in data._nph_params)
    
def nt_bg(data:Data, F:float=5.0)->tuple[GateGroup,...]:
    """Select bursts with (nt >= bg*F)."""
    return tuple(GateGroup.as_gategroup(_gate_gte_new(nph_param, F, 'sbr', (Ph_sel('0ex_1ex1em')))) 
                 for nph_param in data._nph_params)
    
## Selection on burst size vs BG (probabilistic)
def na_bg_p(data:Data, P:float=0.05, F:float=1.0)->tuple[GateGroup,...]:
    """Select bursts w/ AD signal using P{F*BG>=na} < P."""
    raise DeprecationWarning("this function is deprecated, user percentile gate instead")
    
def nd_bg_p(data:Data, P:float=0.05, F:float=1.0)->tuple[GateGroup,...]:
    """Select bursts w/ DD signal using P{F*BG>=nd} < P."""
    raise DeprecationWarning("this function is deprecated, user percentile gate instead")
    
def naa_bg_p(data:Data, P:float=0.05, F:float=1.0)->tuple[GateGroup,...]:
    """Select bursts w/ AA signal using P{F*BG>=naa} < P."""
    raise DeprecationWarning("this function is deprecated, user percentile gate instead")
    
def nt_bg_p(data:Data, P:float=0.05, F:float=1.0)->tuple[GateGroup,...]:
    """Select bursts w/ signal using P{F*BG>=nt} < P."""
    raise DeprecationWarning("this function is deprecated, user percentile gate instead")
    