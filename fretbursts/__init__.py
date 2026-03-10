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
from .datamodel.tables import Param, Column, Gate, MappedGate, GateGroup, GG_all, GG_none
from .datamodel import gates as gates
from .datamodel.gates import (make_geq_gate, make_lt_gate, make_ellipsoid_gate,
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
from .bursttables import (
    Bursts, NphBG, Ratios, make_burst_search, 
    make_dcbs_burst_search, make_correction_factors)
from . import ph_sel
from .ph_sel import PhSel, DetDef
from .import photonHDF5
from . import loadraw
from . import fretfactory

if has_matplotlib:
    from . import plot


add_citation('IngargiolaPLOSOne2016', purpose='FRETBursts citation')
citation()
