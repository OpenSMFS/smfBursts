#
# FRETBursts - A single-molecule FRET burst analysis toolkit.
#
# Copyright (C) 2014-2016 The Regents of the University of California,
#               Antonino Ingargiola <tritemio@gmail.com>
#
from sys import version_info as python_version
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version('fretbursts')
except PackageNotFoundError:
    print("Cannot find package version")
    __version__ = 'undefined'
del python_version, version, PackageNotFoundError
# import warnings

## Citation information
_CITATION = """
   FRETBursts: An Open Source Toolkit for Analysis of Freely-Diffusing Single-Molecule FRET
   Ingargiola et al. (2016). http://dx.doi.org/10.1371/journal.pone.0160716 """

_INFO_CITATION = (f' You are running FRETBursts (version {__version__}).\n\n'
                  ' If you use this software please cite the following'
                  f' paper:\n{_CITATION}\n\n')

def citation(bar=True):
    cit = _INFO_CITATION
    if bar:
        cit = ('-' * 62) + '\n' + _INFO_CITATION +  ('-' * 62)
    print(cit)

# data model imports
from .datamodel.utils import ImDict, FixedDict, MutDict, tupledict
from .datamodel.tables import Param, Column, Gate, GateGroup, GG_all, GG_none
from .datamodel import gates as gates
from .datamodel.gates import (make_gte_gate, make_lt_gate, make_ellipsoid_gate,
                              make_inv_ellipsoid_inclusive_gate)
from .datamodel.citations import (register_citation, cite, add_citation, get_citations,
    create_citation_group, set_prefered_style, list_citation_groups, list_tags,
    registered_citations, registered_citation_groups)

from .datamodel import has_matplotlib, has_numba

# fretbursts imports
from ._citations import fretbursts_citations
from .photondata import PhotonData, PhotonDataList
from .background import Periods, BG, make_bg_param
from . import background as bg
from .bursttables import Bursts, NphBG, Ratios, make_burst_search, make_dcbs_burst_search, make_correction_factors
from . import ph_sel
from .ph_sel import PhSel, DetDef
from .import photonHDF5
from . import loadraw

if has_matplotlib:
    from . import plot
    from . import rawplot


add_citation('IngargiolaPLOSOne2016', purpose='FRETBursts citation')
# import of legacy functions
# from .legacy_burstlib import Data
# from . import legacy_background as bg
# from . import legacy_select_bursts as select_bursts

# try:
#     import matplotlib
#     has_matplotlib = True
# except:
#     has_matplotlib = False

# try:
#     import PyQt5
#     has_pyqt = True
# except:
#     has_pyqt = False

# if has_matplotlib and has_pyqt:
#     from .legacy_burst_plot import (
#         # Standalone plots as a function of ch
#             mch_plot_bg, plot_alternation_hist, alex_jointplot,

#             # Single-ch plots used in multi-ch plots through `dplot`
#             timetrace, timetrace_single, ratetrace, ratetrace_single,
#             timetrace_fret, timetrace_bg,
#             hist_width, hist_size, hist_size_all, hist_brightness,
#             hist_fret, hist_burst_data,
#             hist2d_alex, hist_S, hist_sbr, hist_asymmetry,
#             hist_interphoton_single, hist_interphoton,
#             hist_bg_single, hist_bg, hist_ph_delays, hist_mdelays,
#             hist_mrates, hist_burst_phrate, hist_burst_delays,
#             scatter_width_size, scatter_rate_da, scatter_fret_size,
#             scatter_fret_nd_na, scatter_fret_width, scatter_da,
#             scatter_naa_nt, scatter_alex, hexbin_alex,

#             # Wrapper functions that create a plot for each channel
#             dplot, dplot_48ch, dplot_8ch, dplot_1ch,)

citation()
