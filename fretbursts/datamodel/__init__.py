# -*- coding: utf-8 -*-
# Author : Paul David Harris
# email : harripd@gmail.com
# created 20/10/2025
# purpose init file
"""
"""

from . import utils
from . import immutabledata
from . import diskdict
from . import tables
from . import gates

from .utils import ImDict, FixedDict, MutDict, tupledict, _DataLike, _ImDataLike
from .immutabledata import (
    TypeValidator, _ImData, TV_ImData, TV_PyCode, TV_dtype,
    TV_tuple, TV_frozenset, TV_ndarray, TV_tupledict, TV_type, TV_typewithnodename,
    TV_int, TV_float, TV_bool, TV_str, TV_attrstr, TV_attrstr_allow_empty, TV_bytes
    )
from .diskdict import DiskDict, AttrDD, VattrDD, TypedValueDD, NestedDD, SubDiskDict
from .tables import (ParamDef, ColumnDef, ParentDef, GateDef, 
                     Param, Column, Gate, GateGroup, 
                     DataSet, DataSetList, BaseTable, ChildTable
                     )

from .citations import (register_citation, cite, add_citation, get_citations,
    create_citation_group, set_prefered_style, list_citation_groups, list_tags,
    registered_citations, registered_citation_groups)

from .utils import has_numba

if has_numba:
    from .utils import fjit, fnumba


has_matplotlib = None
try:
    import matplotlib as mpl
except:
    has_matplotlib = False
else:
    has_matplotlib = True

if has_matplotlib:
    from . import plotting
