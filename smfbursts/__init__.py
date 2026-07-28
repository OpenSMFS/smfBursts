#
# smfBursts - A single-molecule fluorescence burst analysis toolkit.
#
# Copyright (C) 2025-2027 TU Dortmund,
#               Paul David Harris <harripd@gmail.com>
#
from sys import version_info as python_version
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version('smfbursts')
except PackageNotFoundError:
    print("Cannot find package version")
    __version__ = 'undefined'
del python_version, version, PackageNotFoundError

import os

from .datamodel import rcParams
rcParams.update({'core.ncore':os.cpu_count(), 'core.alloc_size':512})


# import warnings

## Citation information
# _CITATION = """
#    FRETBursts: An Open Source Toolkit for Analysis of Freely-Diffusing Single-Molecule FRET
#    Ingargiola et al. (2016). http://dx.doi.org/10.1371/journal.pone.0160716 """

# _INFO_CITATION = (f' You are running smfBursts (version {__version__}).\n\n'
#                   ' If you use this software please cite the following'
#                   f' paper:\n{_CITATION}\n\n')

# def citation(bar=True):
#     cit = _INFO_CITATION
#     if bar:
#         cit = ('-' * 62) + '\n' + _INFO_CITATION +  ('-' * 62)
#     print(cit)

# data model imports
from .datamodel.utils import ImDict, FixedDict, MutDict, tupledict
from .datamodel.immutabledata import encode_msgpack, decode_msgpack
from .datamodel.tables import Param, Column, Gate, MappedGate, GateGroup, GG_all, GG_none
from .datamodel import gates
from .datamodel import multifit
from .datamodel.gates import (
    make_geq_gate, make_lt_gate, make_range_gate, make_ellipsoid_gate, 
    make_inv_ellipsoid_inclusive_gate, make_isin_gate
    )
from .cite import (
    register_citation, cite, add_citation, get_citations, print_citations,
    create_citation_group, set_prefered_style, list_citation_groups, list_tags,
    registered_citations, registered_citation_groups
    )

from .datamodel import has_matplotlib, has_numba

# smfbursts imports
from ._citations import smfbursts_citations
from .photondata import PhotonData, PhotonDataList
from .backgroundtables import Periods, BG, make_bg_param
from . import backgroundtables as bg
from .bursttables import Bursts, BurstOvlp
from .childphotontables import NphBG, Ratios, KDE
from . import ph_sel
from .ph_sel import PhSel, DetDef
from .import photonHDF5
from . import loadraw
from . import fretfactory

if has_matplotlib:
    from . import plot


add_citation('IngargiolaPLOSOne2016', purpose='smfBursts citation')
# citation()
