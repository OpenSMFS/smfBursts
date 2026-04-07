#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Fri Feb 28 07:15:41 2025
# @author: paul

"""
The tables module defines the core of the data model of FRETBursts.

Raw data is to be stored in subclasses of :class:`DataSet`, which functions like
an abstract-base-class, and is responsible for managing both the raw data, and
the processed data objects that depend on it 
(:class:`Param`, :class:`Gate`, and :class:`GateGroup`).

.. note::
    
    The class :class:`fretbursts.photondata.PhotonData` is the primary subclass
    of :class:`DataSet` used in FRETBursts, representing a raw photon accquisition 
    from a single confocal excitation spot.


The various processed values, (in FRETBursts things values like E, S, lifetimes etc.) 
are stored in :class:`Table` objects, which contain multiple columns of related parameters. 
In order to make a :class:`Table`, it is first necessary to define the constants 
that are used in whatever calculation is necessary for the values being calcualted.

This is done by specifying a :class:`Param` object. :class:`Param` objects are
the first layer of immutable specifiers that allows FRETBursts to create reproducable
computations. With a :class:`Param` object, and the raw data from a :class:`DataSet`,
is is possible to recompute the exact values from scratch.

:class:`Table` are divided into two sub-types:
    
    #. :class:`BaseTable` define the rows of a given table, and the essential
       columns, if a :class:`Param` has a ``tp``.
    #. :class:`ChildTable` are tables whose rows are defined by one of its parents.
       This allows for additional columns that are dependent on additional parameters
       to be a computed/defined without needing to include them in the :class:`BaseTable`
       params. One of the parents of a :class:`Param` based on a :class:`ChildTable`
       must define it's "base". Either the "base" is a :class:`Param` based on a
       :class:`BaseTable` or if you follow the "base" parents of successive :class:`Param`
       it will terminate in a :class:`Param` based on a :class:`BaseTable`

.. note::
    
    The subclasses :class:`BasePhotonTable <fretbursts.photondata.BasePhotonTable>` 
    and :class:`ChildPhotonTable <fretbursts.photondata.ChildPhotonTable>` of 
    :class:`BaseTable` and :class:`ChildTable` respectively, are the main way in 
    which these two sub-types are used when analyzing single molecule data.


New instances of :class:`Table` are usually not created directly, but through the
managemnt of a :class:`DataSet`, when it is given a set of instructions in a
:class:`Param` object.

:class:`Param` takes a :class:`Table` type as well as other parameters that are
dependant on the subclass of :class:`Table` it is given.

Processed data values, are retrieved from :class:`DataSet` objects using
:class:`Column` objects, which are composed a a :class:`Param`, a :class:`Param`
specific column name, and a set of column specific keys.

Finally, values can be filtered/gated with :class:`GateGroup` objects, which
rely on :class:`Gate` objects to define gate functions, that :class:`GateGroup`
objects combine with logical operations.

"""
from itertools import chain, product, permutations
from collections import Counter
from collections.abc import Callable, Sequence, Hashable, Iterator
import weakref
from weakref import WeakValueDictionary as WVD
from typing import ClassVar, Any, Union
import warnings
import inspect
import re
import os
from numbers import Number, Real
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tables as tb

from .utils import (
    tupledict, FixedDict, ImDict, _echo, arr_slc, _make_sortable, 
    _nested_set, _nested_pop, _ImDataLike, kwarg_like, _FileFinalizer, 
    _GroupFuture, GroupFuture, _masked_iter, _delayed_iter, _indent
    )
from .immutabledata import (
    TypeValidator, _ImData,TV_int, TV_bool, TV_str, TV_attrstr, TV_attrstr_allow_empty, 
    TV_ndarray, TV_tuple, TV_type, TV_dtype, TV_PyCode, TV_tupledict, TV_ImData, 
    register_PyCode, register_type
    )
from .diskdict import DiskDict, MaskedDD


def _check_appendable_param(val:Any)->Callable[[dict],tuple["ParamDef",...]]:
    """
    Function checks value of append_params in ParamDef, and ensures it is
    a function that can be used to append additional parameters
    """
    if not callable(val):
        raise TypeError("append_params must be a callable")
    params = inspect.signature(val).parameters
    if sum(p.kind != p.KEYWORD_ONLY for p in params.values()) < 1:
        raise ValueError("append_params must accept at least 1 potentially positional argument")
    if sum(p.kind == p.POSITIONAL_ONLY for p in params.values()) > 1:
        raise ValueError("append_params can have no more than 1 positional only arguments")
    if sum(p.default is p.empty for p in params.values()) > 1:
        raise ValueError("append_params must be able to accept a call with only one positional argument specified")
    return val


class ParamDef(_ImDataLike):
    """
    Type used for defining a key in the params tupledict of a :class:`Param`
    
    Parameters
    ----------
    name : str
        The key name of the param.
    type_validator : TypeValidator 
        :class:`TypeValidator` object used after param_preprocess to confirm 
        param is of correct type.
    required : bool , optional
        Whether the parameter must be present, default is True.
    default : Hashable, optional
        Default value given to parameter, default is None.
    append_params : Callable, optional
        Fuction that returns tuple of ParamDefs to be added to param_defs based 
        on value of given param. 
    unit : str, optional
        String defining the unit (if any) to give to parameter in description.
        Default is empty string
    
    """
    #: Name of param, ie the key in teh params for this ParamDef
    name: str
    #: :class:`TypeValidator` used to check value of this param
    type_validator: TypeValidator
    #: Whether this param must be in the params dict or is optional
    required: bool
    #: default value if param not specified in param
    default: Any
    #: Function that takes params and returns tupel of :class:`ParamDef` s that
    #: are conditional on other params
    append_params: Callable[[dict],tuple["ParamDef",...]]
    #: string name of unit
    unit: str
    __slots__ = ('name', 'type_validator', 'required', 'default', 'append_params', 'unit')
    _setfuncs = ImDict(name=TV_attrstr.check_val, type_validator=TypeValidator.convert_type, 
                       required=bool, append_params=_check_appendable_param, unit=str)
    _required = frozenset({'name', 'type_validator', 'required'})
    _defaults = ImDict(required=True, type_validator=TypeValidator.check_any, unit='')


class ParentDef(_ImData):
    """
    Class for defining a parent in a :class:`Table`.
    
    Parameters
    ----------
        name : str
            The name given to the parent, used as key in parents tupledict
        table_type : type
            A type object, must be subclass of :class:`Table`, defines what type of
            parents (``Param.tp`` is ``table_type`` or subclass thereof)
        is_base : bool, optional
            Whether the parent defines the base_param of the table or not. The default is False.
        share_base : bool, optional
            If the parent must have the same ``base_param`` as table. If ``is_base``
            is ``True``, then  ``share_base`` must be ``False``. The default is False.
        size_func : str, optional
            string name of classmethod used to determine the correct number of
            params in parent (for array parents only) If set, parent will be tuple
            of params, if not set, parent is single param. Default is empty string.
    
    """
    __slots__ = ('name', 'table_type', 'is_base', 'share_base', 'size_func')
    _typeconversions = ImDict(name=TV_str, 
                              table_type=TV_type,
                              # table_type=TV_type(validator=_check_Table),
                              is_base=TV_bool, share_base=TV_bool, 
                              size_func=TV_attrstr(allow_empty=True))
    _required = frozenset({'name', 'table_type', 'is_base', 'share_base'})
    _defaults = ImDict(is_base=False, share_base=False, size_func='')

    #: name of parent ie the key in parents for this ParentDef
    name: str
    #: the Table subclass requirement of the parent
    table_type: type|tuple[type, ...]
    #: the parameter that defines the base (including subgate)
    is_base: bool
    #: whether or not the base_parent of the parent is the same as the resultant Table
    share_base: bool
    #: empty string if parent not a tuple, takes params, and returns expected size of parent
    size_func: str

    def __post_init__(self):
        if self.is_base and self.share_base:
            raise ValueError("is_base and share_base cannot both be True")

    @property
    def base_like(self)->bool:
        """If the param should have the same base as the base_param of the table"""
        return self.is_base or self.share_base

    @property
    def is_array(self)->bool:
        """If parent is array/tuple of params or single param"""
        return bool(self.size_func)


def _get_baseparent(parent_defs:Sequence[ParentDef])->ParentDef|None:
    """Retrieve the ParentDef that defines the base"""
    for val in parent_defs:
        if val.is_base:
            return val
    return None


def as_paramdict(val:Sequence[Any]|Sequence[tuple[str,Any],...]|dict[str:Any]|tupledict,
                 defs:tuple[str,...])->dict:
    """
    Convert inputs into dictionary with keys defined by defs. Useful in 
    :meth:`Table.param_preprocess`
    to regularlize input before further processing.

    .. note::

        returns dict instead of tupledict so that values can be popped and
        modified. TypeValidator for Param.params will automatically convert
        to tupledict after :meth:`Table.param_preprocess`

    Parameters
    ----------
    val: Sequence
        Sequence of parameter values, or dict of parameter values
    defs : tuple[str,...]
        Sequence of strings defining all valid dictionary keys.

    Returns
    -------
    dict
        Regularlized dict of param values.

    """
    if isinstance(val, tupledict):
        val = dict(val)
    elif isinstance(val, Sequence):
        if all(isinstance(v, Sequence) and len(v) == 2 and isinstance(v[0], str) for v in val):
            val = dict(*val)
        elif len(val) <= len(defs):
            val = {d:v for d, v in zip(defs, val)}
        else:
            raise ValueError("params too long for given defs")
    elif not isinstance(val, dict):
        raise TypeError(f'cannot convert {type(val)} to tupledict')
    if any((err := k) not in defs for k in val.keys()):
        raise ValueError(
            f'{err} is not a valid params key for this type of Table')
    return val


def _check_typeTable(val:type, *args:Any, **kwargs)->type:
    """Checks that tp input of Param is a Table subclass"""
    if not issubclass(val, Table):
        raise TypeError('must be subclass of Column')
    return val


def _proc_TV_param_defs(imdata:"Param", *args)->dict:
    """Pre-process params of Param, get order, types and defaults for each param key"""
    param_defs = imdata['tp'].param_defs
    order = [pdef.name for pdef in param_defs]
    required = set(pdef.name for pdef in param_defs if pdef.required)
    defaults = {pdef.name: pdef.default for pdef in param_defs if hasattr(pdef, 'default')}
    typedefs = {pdef.name: pdef.type_validator for pdef in param_defs}
    kwargs = dict(required=required, order=order, defaults=defaults, 
                  typedefs=typedefs, pdefs=param_defs)
    kwargs['_kwargs'] = kwargs
    return kwargs


def _check_TV_param_defs(val:tupledict, _kwargs:dict=None, **kwargs)->tupledict:
    """Check and regularize Param params so correct ordered tupledict"""
    param_defs = kwargs['pdefs']
    for pdef in param_defs:
        if 'append_params' in pdef and (pdef.name in val or 'default' in pdef or pdef.required):
            append = pdef.append_params(val)
            _kwargs['required'] |= {apdef.name for apdef in append if apdef.required}
            _kwargs['order'] += [apdef.name for apdef in append]
            _kwargs['defaults'].update({apdef.name:apdef.default for apdef in 
                                        append if hasattr(apdef, 'default')})
            _kwargs['typedefs'].update({apdef.name:apdef.type_validator for apdef in append})
    return val


def _proc_TV_parent_defs(imdata:"Param", *args)->dict:
    """Pre-process function for Param parents, set types"""
    return dict(tp=imdata['tp'], parent_defs=imdata['tp'].parent_defs, params=imdata['params'])


def _check_parent_defs(val:tupledict["Param"], tp:type=None, params:tupledict=None, **kwargs)->tupledict["Param"]:
    """Check function for Param parents, ensure correct types and regularize to tupledict of correct order"""
    if not tp.parent_defs:
        if val:
            raise ValueError("parents must be empty for the given Table type")
    base_parentdef = _get_baseparent(tp.parent_defs)
    if base_parentdef is not None:
        if base_parentdef.name not in val:
            raise ValueError(f"missing base param from parents {base_parentdef.name}")
        base_parent = val[base_parentdef.name]
    else:
        base_parent = None
    val = tupledict(*((pdef.name, _verify_parent_param(val.get(pdef.name), tp, pdef, base_parent, params))
                      for pdef in tp.parent_defs))
    return val


def _verify_parent_param(val:Union["Param",tuple["Param",...]], tp:type, parent_def:ParentDef, 
                         base:"Param", params:tupledict)->Union["Param",tuple["Param",...]]:
    """
    Check a single parent is valid in Param.parents dict. 
    Checks both type and if it shoudl be single or tuple
    """
    if val is None:
        raise ValueError(f"must specify {parent_def.name} in parents")
    if parent_def.is_array:
        size = getattr(tp, parent_def.size_func)(params)
        if isinstance(val, Param):
            val = _verify_parent_param_sub(val, parent_def, base)
            return tuple(val for _ in range(size))
        elif len(val) == size:
            return tuple(_verify_parent_param_sub(v, parent_def, base) for v in val)
        raise ValueError(f"Incorrect number of parents for {parent_def.name}, expected {size}, got {len(val)}")
    else:
        return _verify_parent_param_sub(val, parent_def, base)


def _verify_parent_param_sub(subval:"Param", parent_def:ParentDef, base:"Param")->"Param":
    """Sub-parent verification, if parent is single, or single element of array"""
    if parent_def.share_base and subval.base_param != base:
        raise ValueError(
            f"{parent_def.name} must share base param with parent defining the base")
    if not issubclass(subval.tp, parent_def.table_type):
        raise ValueError(
            f"{parent_def.name} must be a param with tp of {parent_def.table_type}, got {subval.tp}")
    return subval


def _proc_paramgates_defs(imdata:"Param", *args)->dict:
    """Data pre-process for :attr:`Param.gategroup` ensures only base params have gategroup"""
    if issubclass(imdata.tp, ChildTable):
        raise TypeError(
            f'cannot specify gate for {imdata.tp} columns (must gate the base_parent param)')
    return dict()


def _check_GateGroup(val:"GateGroup", **kwargs)->"GateGroup":
    """Check function for Param gategroup parameter"""
    if not isinstance(val, (Gate, GateGroup)):
        raise TypeError("gategroup must be instance of GateGroup or Gate")
    val = GateGroup.as_gategroup(val)
    return val


class Param(_ImData):
    """
    Param objects define all the information necessary for creating a given 
    :class:`Table` from a given :class:`DataSet`.
    
    If ``tp`` is a BaseTable, the gate of the param is set by the ``gategroup``
    attribute, and assumed to be an all gate if not specified.
    If ``tp`` is a ChildTable, then the gate is set through the base parent instead.
    
    Parameters
    ----------
    tp : type
        A ``type`` object, which must be a subclass of :class:`Table` that the
        Param defines.
    params : tupledict[str:Hashable]
        The values (constants) used in computing the values, these are things 
        like the sliding window size m, and backgroud threshhold F
    parents : the :class:`Param` objects of whaterver other :class:`Table` 
           that are necessary to compute the :class:`Table`
    gategroup: GateGroup, optional
        This defines which rows to include in the table. The `gategroup` attribute
        can only exist for ``Param`` objects whose ``tp`` attribute is a subclass
        of :class:`BaseTable`, since it is these tables which define the rows.
    name : str, optional
        **Unhashed** This is a user-specified name to give to the current param.
        This name does not contribute to the hash, or ``==`` operations, thus
        two Param objects may have different names, but will have the same hash
        and will evaluate to equal one another.
    flags : list[str], optional
        **Unshahsed** List of string flags, used to indicate expected uses of this param
        
    """
    __slots__ = ('tp', 'params', 'parents', 'gategroup', 'name', 'flags')
    _typeconversions = ImDict(tp=TV_type(validator=_check_typeTable),
                              params=TV_tupledict(
                                  validator=_check_TV_param_defs, data_proc=_proc_TV_param_defs),
                              parents=TV_tupledict(
                                  validator=_check_parent_defs, data_proc=_proc_TV_parent_defs),
                              gategroup=TV_ImData(validator=_check_GateGroup, 
                                                  data_proc=_proc_paramgates_defs))
    _defaults = ImDict(params=lambda: dict(), parents=lambda: dict())
    _required = frozenset({'tp', })
    _hashskip = ('name', 'flags')

    #: subclass of Table associated with param
    tp: "Table"
    #: tuple of parameters values defining a given column
    params: tupledict
    #: tuple of parameters of parent columns
    parents: tupledict
    # : GateGroup, a mapping of Gates to include/exclude from returned table
    gategroup: "GateGroup"
    #: (mutable) "common" name for object
    name: str
    #: (mutable) list of flags that can indicate function or intended use of object
    flags: list[str]

    def __new__(cls, tp:type, params:dict[str,Any]=None, parents:dict[str,Union["Param",Sequence["Param"]]]=None, 
                gategroup:"GateGroup"=None, name:str=None, flags:list[str]=None, **kwargs):
        # first sort out "standard" params/parents inputs
        params = dict() if params is None else params
        params = params.asdict if isinstance(params, tupledict) else params
        parents = dict() if parents is None else parents
        parents = parents.asdict if isinstance(parents, tupledict) else parents
        param_names = tuple(pdef.name for pdef in tp.param_defs)
        parent_names = tuple(pdef.name for pdef in tp.parent_defs)
        # Now take kwargs as nonstandard inputs matching them to params or parents
        kkeys = list(kwargs.keys())
        for key in kkeys:
            if key in param_names and key not in params:
                params[key] = kwargs[key]
            elif key in parent_names:
                if key in parents:
                    raise ValueError(f"{key} already specified in parents")
                parents[key] = kwargs[key]
            elif key in params:
                raise ValueError(f"{key} already specified in params")
            else:
                params[key] = kwargs[key]
        params, parents = tp._param_preprocess(params, parents)
        return super().__new__(cls, tp, params, parents, **{k:v for k, v in 
                                                            zip(('gategroup', 'name', 'flags'), 
                                                                (gategroup, name, flags)) 
                                                            if v is not None})

    def __post_init__(self):
        if issubclass(self.tp, BaseTable) and 'gategroup' in self and self.gategroup.nogate and self.gategroup.truthtable:
            super(_ImData, self).__delattr__('gategroup')
        if issubclass(self.tp, ChildTable):
            base = self.base_param
            if not issubclass(base.tp, BaseTable):
                raise TableConstructionError(
                    "ChildTable must parent 'base' must be a BaseTable")
        self.tp._validate_param(self)

    def __getattr__(self, attr):
        if attr in self.params:
            return self.params[attr]
        if attr in self.parents:
            return self.parents[attr]
        for tp in self.tp.__mro__:
            if hasattr(tp, '_parammethods') and attr in tp._parammethods:
                return getattr(self.tp, attr)(self)
        raise AttributeError(f"Param[{self.tp.__name__}] has no param or parent {attr}")

    def degate(self)->"Param":
        """Remove gategroup from param/base_param of param"""
        if issubclass(self.tp, BaseTable):
            return self._replace_fields(pop=('gategroup',))
        parents = tupledict(
            *(((k, _parent_degate(v)) if _get_parentdef(self.tp.parent_defs, k).base_like else (k, v))
              for k, v in self.parents.items()))
        return self._replace_fields(fields={'parents': parents})

    def regate(self, gate:"GateGroup")->"Param":
        """
        Change gategroup of param or base_param of param

        Parameters
        ----------
        gate : GateGroup
            New gate for returned param

        Raises
        ------
        ValueError
            Gate is invalid (does not share the same ``origin_param`` ) with param.

        Returns
        -------
        Param
            New :class:`Param` with new gate.

        """
        gate = GateGroup.as_gategroup(gate)
        # catch all/none cases
        if gate.nogate:
            if 'param' not in gate or gate.param == self.origin_param:
                if gate:
                    return self.degate()
                if 'param' not in gate:
                    gate = GateGroup(gate.truthtable, param=self.origin_param)
            else:
                raise ValueError("origin_param of gate does not match param")
        if issubclass(self.tp, BaseTable):
            return self._replace_fields(fields={'gategroup': gate})
        parents = tupledict(
            *((k, v.regate(gate) if _get_parentdef(self.tp.parent_defs, k).is_base else v) 
              for k, v in self.parents.items()))
        return self._replace_fields(fields={'parents': parents})

    @classmethod
    def param_comp(cls, subparam:"Param", superparam:"Param")->int:
        """
        Determine the overlap code of the base_gate of two parameters.
        Shortcut for :meth:`GateGroup.overlap` when using :class:`Param` objects.
        If subparam and superparam are of different type or mismatched origin_param,
        return -1.

        Parameters
        ----------
        subparam : Param
            First parameter to compare, considdered the "inner" parameter.
        superparam : Param
            Secont parameter to compare, considdered the "outer" parameter.

        Raises
        ------
        TypeError
            One ro both arguments not a :class:`Param`

        Returns
        -------
        int
            Overlap code between gates of both params, -1 if params are for
            different base tables.

        """
        if not isinstance(subparam, Param) or not isinstance(superparam, Param):
            raise TypeError('param_comp arguments must both be Param objects')
        if subparam.degate() == superparam.degate():
            return GateGroup.overlap(subparam.base_gate, superparam.base_gate) & 0b0010
        return -1

    @property
    def base_param(self)->"Param":
        """
        The :class:`Param` defining the rows of the current param. 
        Has ``tp`` which is subclass of :class:`BaseTable`
        """
        if issubclass(self.tp, BaseTable):
            return self
        return self.parents[_get_baseparent(self.tp.parent_defs).name].base_param

    @property
    def base_gate(self)->"GateGroup":
        """
        The gate of the current param, defines which rows of the origin_param 
        table to include
        """
        base_param = self.base_param
        if "gategroup" in base_param:
            return base_param.gategroup
        return GateGroup(truthtable=_TT_all, gates=tuple(), param=base_param)

    @property
    def parent_param(self)->"Param":
        """
        :class:`Param` used for computing the gate of :attr:`Param.base_param`.

        .. note::

            This is usually the same as origin_param, and is only different if
            the gate is non-atomic
        """
        return self.base_param.regate(self.base_param.base_gate.parent_gate)

    @property
    def parent_gate(self)->"GateGroup":
        """
        The parent_gate of the gate of this param.

        .. note::

            For atomic gates, this is always GG_all, it is not GG_all only if the
            gate is non-atomic.
        """
        return self.base_param.base_gate.parent_gate

    @property
    def max_gate(self)->"GateGroup":
        """The largest reasonable gate for the param, only used to check reasonable param"""
        return self.tp.max_gate_from_param(self)

    @property
    def origin_param(self) -> "Param":
        """The Param defining all rows in the table, disregarding all gates"""
        return self.base_param.degate()

    @property
    def description(self)->str:
        """Human readable description of parameters of self"""
        return self.tp.get_param_description(self)

    def __str__(self):
        return f'Param of {self.tp.__module__}.{self.tp.__name__} at 0x{id(self):x}'
    
    @property
    def _sort_tuple(self)->tuple[Hashable,...]:
        return (f'{self.tp.__module__}.{self.tp.__name__}', 
                _make_sortable(self.parents), _make_sortable(self.params))
        


def _param_validator(val:Param, table_type:type|Sequence[type]=None, **kwargs)->Param:
    """
    Validate :class:`Param` is of valid tp

    Parameters
    ----------
    val : Param
        :class:`Param` object to be checked.
    table_type : type|Sequence[type], optional
        If supplied, valid type(s) for :attr:`Param.tp`. The default is None.
    **kwargs : Any
        Ignored.

    Raises
    ------
    ValueError
        val of incorrect type.

    Returns
    -------
    Param
        echo of val.

    """
    if table_type is not None and not issubclass(val.tp, table_type):
        raise ValueError(f"param must be of tp {table_type}, has {val.tp}")
    return val


def _parent_degate(val:Param|tuple[Param,...]):
    """Degates all base-like parents in val, used during degating of a :class:`Param`"""
    if isinstance(val, tuple):
        return tuple(v.degate() for v in val)
    return val.degate()


register_type(Param)
TV_Param = TV_ImData(subclass=Param, validator=_param_validator)


def _get_parentdef(parent_defs:tuple[ParentDef,...], name:str)->ParentDef:
    """Retrieve the ParentDef from tuple of ParentDef-s that has the name ``name``"""
    for pdef in parent_defs:
        if pdef.name == name:
            return pdef
    raise ValueError(f'{name} not in parent_defs')


def _proc_dimlimits(imdata:"ColumnDef", kwarg_append:dict)->dict:
    """
    Pre-process function for :class:`ColumnDef` dimlimits, 
    ensures dimensions match ndim of ColumnDef.
    """
    return dict(dims=arr_slc[imdata.ndim-1, 2])


def _check_dimlimits(val:np.ndarray, dims:tuple[...,int]=None, **kwargs)->np.ndarray:
    """Check function for :class:`ColumnDef` dimlimits"""
    val = np.asarray(val)
    if val.size == 1:
        val = val.reshape(1,1)
    if val.ndim == 1:
        val.reshape(-1,1)
    if val.shape[0] == 1:
        val = np.array([val for _ in range(dims[0])])
    if val.shape[1] == 1:
        val = np.repeat(val, 2, axis=1)
    return val


def _check_map_to(val:type)->type:
    """Check function for :class:`ColumnDef`, ensure is a :class:`BaseTable`"""
    if not issubclass(val, BaseTable):
        raise TypeError("map_to value must be a Table class")
    return val


_column_regex = re.compile(r'^[A-Za-z][\w_]*$') # : matches valid names of columns


def _check_TV_withnodename(val:tuple[type|TypeValidator,...])->tuple[TypeValidator,...]:
    """Check function for :class:`ColumnDef`, ensures all keytup keys are string-able"""
    val = tuple(TypeValidator.convert_type(tp) for tp in val)
    if any(tv.node_prefix is None for tv in val):
        raise ValueError("type has no nodename")
    return val


def _check_title_is_tex(title_is_tex:int)->int:
    """Check function for title_is_tex value of :class:`ColumnDef """
    title_is_tex = int(title_is_tex)
    if title_is_tex not in (0, 1, -1):
        raise ValueError("title is tex must be -1, 0, 1")
    return title_is_tex


_check_tex_rgx = re.compile(r'([^\{\}]*\{[^\{\}]*\})+[^\{\}]*')


def _check_tex(title_is_tex:int, title:str)->bool:
    """Test if title can be tex, return if should be treated as tex"""
    if title_is_tex == -1:
        return bool(_check_tex_rgx.fullmatch(title))
    return bool(title_is_tex)


_indexable_regex = re.compile(r'^[\w\d\s\(\)\^\-\+]*$')
_TV_indexable = TV_str(pattern=_indexable_regex)


class ColumnDef(_ImDataLike):
    """
    Class used as elements of the column_defs tuple in :class:`Table`.
    
    Parameters
    ----------
    name : str
        The name given to the column, this is the string used as the
        first element of the tuple specifying a column when indexing a :class:`Table`
    keytypes : tuple[type,...]
        A tuple of the type of each key used to specify the column
    offset : int, optional
        The number of rows by which the column differs from the size of the :class:`Table`.
        The default is 0.
    store : Literal['none', 'some', 'all']
        How column is stored, options are ``'none'`` (computed 
        each time), ``'some'`` (some keys are stored, _add_column must be
        manually called), ``'all'`` (every time a new column is computed, 
        it is automatically stored), and ``'user'`` (column only stored when user
        uses :meth:`DataSet.record_column` method), which means keys are . 
        Default is ``'user'``
    atomic : bool, optional
        Whether the rows of the column are independent of the gate
        Default is True.
    get_func : str, optional
        The name of the method called to generate the entire column
        as an array. If not specified, assume column is set during initialization
        of table. Default is empyt string.
    iter_func : str, optional
        The name of the method called to get an iterator that
        iterates through the rows of the column. If not specified, assume column 
        is set during initialization of table. Default is empyt string.
    get_derived : bool, optional
        If True, then ``get_func`` can be called from a derived :class:`Table`, 
        otherwise must always be called from origin :class:`Table`.
        Default is False.
    dtype : np.dtype, optional
        The datatype of the columns. Default is np.float64.
    typedef : np.dtype, optional
        Only when dtype is object, datatype of each row (which
        are arrays) of the column. Default (only when dtype is object) is float64.
    ndim : int, optional
        Dimensionality of column, Default is 1.
    dimlimits : np.ndarray[np.float64], optional
        ndim-1x2 array defining min/max size of each dimension (excluding the first) 
        of the column. Float array because must support inf dim limits. Otherwise
        all elements should have integral values. Default is [].
    fill : Any, optional
        The value used as the default fill value for the column.
        Only for non-atomic or columns with offset. Default is nan for floating
        columns, -1 for integral columns, and None for object columns.
    reg_func : str, optional
        name of method to call from Table (should be classmethod) that will 
        regularize and raise appropriate errors for out of range columns. 
        If not present, no regularization will take place. Default is empty string.
    check_func : str, optional
        name of method to call from Table (should be classmethod) that verifies 
        all values of column are valid (similar to validate_param). 
        Default is empty string.
    mapto : type, optional
        If present, indicates column is based on one :class:`Table`
        but outputs a column of size matching another :class:`Table` 
        (must be :class:`BaseTable`). 
        Default is None (field will not exist if not specified).
    remap : str, optional
        If present marks that the column is a convenience column,
        and the resultant column will be maped to another column. remap is a
        string specifying the method in the :class:`Table` that the current
        values will be given to in order to convert the specified column into
        its "real" column. Default is None (field will not exist if not specified).
    title : str, optional
        Default title for column. Default is None.
    title_func : str, optional
        name of method in :class:`Table` (should be classmethod),
        to call when calling :class:`Column.name` to get default title.
        Default is empty string.
    unit : str, optional
        Default unit for column. Default is empty string.
    index : str, optional
        Default index name for column (name used in csv export).
        Default is empty string.
    index_func : str, optional
        Name of method in :class:`Table` (should be classmethod)
        to call when calling :class:`Column.index_name` to get default csv index.
        Default is empty string.
    index_unit : str
        Default index unit name for column. Default is empty string.
    title_is_tex : bool
        If True :meth:`Column.name` will suround output with $ signs.
        Default determinded by whether title can be rendered as tex.

    """
    __slots__ = ('name', 'keytypes', 'offset', 'store', 'atomic', 'get_func', 
                 'iter_func', 'get_derived', 'dtype', 'typedef', 'ndim', 
                 'dimlimits', 'fill', 'reg_func', 'check_func', 'mapto', 'remap',
                 'title', 'title_func', 'unit', 'index_func', 'index','index_unit', 'title_is_tex')
    _setfuncs = ImDict(
        name=TV_str(pattern=_column_regex).check_val, 
        keytypes=_check_TV_withnodename, offset=TV_int.check_val, 
        store=TV_str(isin=('never', 'some', 'user', 'all')).check_val, 
        atomic=bool, get_func=TV_attrstr_allow_empty.check_val, 
        iter_func=TV_attrstr_allow_empty.check_val, get_derived=bool,
        dtype=TV_dtype.check_val, typedef=TV_dtype.check_val, ndim=TV_int(mn=1).check_val,
        dimlimits=TV_ndarray(mn=0, validator=_check_dimlimits, data_proc=_proc_dimlimits).check_val,
        mapto=TV_type(validator=_check_map_to).check_val, remap=TV_attrstr.check_val, 
        reg_func=TV_attrstr_allow_empty.check_val, check_func=TV_attrstr_allow_empty.check_val,
        title=str, title_func=TV_attrstr_allow_empty.check_val, unit=TV_str.check_val, 
        index=_TV_indexable.check_val, index_func=TV_attrstr.check_val,
        index_unit=_TV_indexable.check_val, title_is_tex=_check_title_is_tex)
    _defaults = ImDict(keytypes=lambda: tuple(), offset=0, reg_func='')
    _required = frozenset({'name', 'keytypes', 'offset'})

    name: str  #: name of column
    #: For Nested key type columns only, the types of the nested column, otherwise should be tuple
    keytypes: tuple[type]
    offset: int  #: Used in :attr:`MultiArrayValueDD.shape_offsets`, the number 
                 # by which the size of each spot differs from the base size
    store: str  #: whether or not to calculate new each time, can be 'never', 
                #: 'some', 'user', and 'all', if 'some', the user must manually save all 
                #: columns, if atomic or mapped, must be never
    atomic: bool  #: whether or not each element is independent from all others, 
                  # True for columns that only dependon the given data point
    get_func: str  #: Compute the given column, returns all spots as array, may 
                   #: not update other columns, returned value will be used to set column
    iter_func: str  #: whether the get_func returns the entire column, or creates an iterator
    get_derived: bool  #: whether get or iter func can be called directly from 
                       #: derived column, or if it must be called from origin column, 
                       #: default is False
    dtype: np.dtype  #: Compute each spot in the column one by one, returned, 
                     #: may not update other columns, returned value will be treated 
                     #: as spot in column
                     #: only for object type columns, the type of numpy array stored in each element
    typedef: np.dtype
    ndim: int  #: Number of dimensions in array, usually 1
    #: ndim-1 x 2 array defining minimum and maximum size of each dimension (excluding 1st)
    dimlimits: np.ndarray[np.int64]
    fill:Any #. Fill value for column
    reg_func: str #: function to call on Table to regularize or check column values
    check_func: str #: function to call on Table to check column values
    mapto: type  #: the type of table that the column maps to
    remap: str  #: if present, column is an alias, string is attr that will convert 
                #: column to "real" column
    title: str #: default title for column
    unit: str #: default unit for column
    index_title: str #: default index name for csv
    index_unit: str #: default unit name for csv
    title_is_tex: int #: whether to wrap output of name() in $$

    def __post_init__(self):
        if 'remap' in self:
            self.__post_init_remap__()
        else:
            self.__post_init_reg__()

    def __post_init_remap__(self):
        # Check that remapped column has no "standard" specifications
        if any(attr in self for attr in ('store', 'atomic', 'get_func', 'iter_func', 
                                         'get_derived', 'dtype', 'typedef', 'ndim', 
                                         'dimlimits', 'fill', 'check_func', 
                                         'title', 'title_func', 'unit', 
                                         'index', 'index_func', 'index_unit', 'title_is_tex')):
            raise ValueError('remaped columns cannot specify get_func, iter_func, dtype, '
                             'typedef, atomic, ndim, dimlimits, fill, check_func, title, '
                             'title_func, unit, index, index_func, undex_unit, title_is_tex')

    def __post_init_reg__(self):
        if 'atomic' not in self:
            super(_ImDataLike, self).__setattr__('atomic', True)
        if 'mapto' in self or not self.atomic:
            if 'store' not in self:
                super(_ImDataLike, self).__setattr__('store', 'never')
            elif self.store != 'never':
                raise ValueError("mapto and/or non-atomic columns must have store = 'never'")
        elif 'store' not in self:
            super(_ImDataLike, self).__setattr__('store', 'user')
        if 'get_func' not in self:
            super(_ImDataLike, self).__setattr__('get_func', '')
        if 'iter_func' not in self:
            super(_ImDataLike, self).__setattr__('iter_func', '')
        if 'check_func' not in self:
            super(_ImDataLike, self).__setattr__('check_func', '')
        if self.store in ('never', 'user') and (not self.get_func and not self.iter_func):
            raise ValueError("Must specify get_func or iter_func when store is not 'all'")
        if 'get_derived' not in self:
            getderived = not self.atomic and self.store not in ('all', 'some')
            super(_ImDataLike, self).__setattr__('get_derived', getderived)
        if not self.atomic and not self.get_derived:
            raise ValueError("atomic columns must always allow get_derived")
        if self.get_derived and (not self.get_func and not self.iter_func):
            raise ValueError("get derived columns must supply either get_func or iter_func")
        if self.store in ('all', 'some') and self.get_derived:
            raise ValueError("get_derived cannot be True for store all columns")
        if 'dtype' not in self:
            super(_ImDataLike, self).__setattr__('dtype', np.dtype('<f8'))
        if self.dtype == np.object_:
            if 'typedef' not in self:
                super(_ImDataLike, self).__setattr__('typedef', np.dtype('<f8'))
        elif 'typedef' in self:
            raise ValueError("Only np.object_ dtype columns may have typedef specified")
        if 'ndim' not in self:
            super(_ImDataLike, self).__setattr__('ndim', 1)
        if 'dimlimits' not in self:
            dimlimits = np.array([[0.0, np.inf] for i in range(self.ndim-1)]).reshape(-1,2)
            super(_ImDataLike, self).__setattr__('dimlimits', dimlimits)
        if self.offset != 0:
            if 'fill' not in self:
                if np.issubdtype(self.dtype, np.floating):
                    super(_ImDataLike, self).__setattr__('fill', self.dtype.type(np.nan))
                elif np.issubdtype(self.dtype, np.integer):
                    super(_ImDataLike, self).__setattr__('fill', self.dtype.type(-1))
                elif self.dtype == np.object_:
                    super(_ImDataLike, self).__setattr__('fill', np.empty((0,), dtype=self.typedef))
            elif not np.issubdtype(type(self.fill), self.dtype):
                raise TypeError("fill type ({type(self.fill).__name__}) incompatible with dtype: ({self.dtype})")
        elif 'fill' in self:
            raise ValueError("fill only specified for columns offset")
        if 'title_is_tex' not in self:
            super(_ImDataLike, self).__setattr__('title_is_tex', -1)

    @property
    def keylen(self)->int:
        """The number of keys expected for the column"""
        return len(self.keytypes) + ('mapto' in self)

    def get_name(self, col:"Column", include_unit:bool=False, origin:"DataSet"=None)->str:
        """
        Return the string-name (for axis titles etc) of col.

        Parameters
        ----------
        col : Column
            Column for which name is to be computed.
        include_unit : bool, optional
            If name should include unit. The default is False.
        origin : DataSet, optional
            DataSet to which col is linked. The default is None.

        Returns
        -------
        str
            Title of column.

        """
        kw = {'include_unit':include_unit}
        if origin is not None:
            kw['origin'] = origin
        if 'title' in col:
            title = col.title
        else:
            tp = col.source_param.tp
            if 'title_func' in self:
                return getattr(tp, self.title_func)(col, **kw)
            keytup = col.keytup[1:] if 'mapto' in self else col.keytup
            title = self.title if 'title' in self else col.col
            if len(keytup) == 1:
                title = f'{title}: {str(keytup[0])}'
            elif keytup:
                title += ':(' + ','.join(str(key) for key in keytup) + ')'
        tex_type = col.title_is_tex if 'title_is_tex' in col else self.title_is_tex
        unit = ''
        if include_unit:
            if 'unit' in col:
                unit = f' {col.unit}'
            elif 'unit' in self:
                unit = f' {self.unit}'
        if _check_tex(tex_type, title) or _check_tex(tex_type, unit):
            title = rf'${title}\:{unit}' if unit else rf'${title}$'
        else:
            title = f'{title}{unit}'
        return title

    def get_index(self, col:"Column", include_unit:bool=False, origin:"DataSet"=None)->str:
        """
        Return the string-indx (for csv/dataframe export) of col.

        Parameters
        ----------
        col : Column
            Column for which name is to be computed.
        include_unit : bool, optional
            If name should include unit. The default is False.
        origin : DataSet, optional
            DataSet to which col is linked. The default is None.

        Returns
        -------
        str
            index-name of column.

        """
        tp = col.source_param.tp
        kw = {'include_unit':include_unit}
        if origin is not None:
            kw['origin'] = origin
        if 'index_title' in col:
            index = col.index_title
        elif 'title' in col and _indexable_regex.match(col.title):
            index = col.title
        elif 'index_func' in self:
            return getattr(tp, self.index_func)(col, **kw)
        elif 'index' in self:
            index = self.index
        elif 'title' in self and _indexable_regex.match(self.title):
            index = self.title
        else:
            index = col.col
        if 'index_title' not in col or ('title' not in col or not _indexable_regex.match(col.title)):
            if len(col.keytup) == 1:
                index = f'{index}: {str(col.keytup[0])}'
            elif col.keytup:
                index += ':(' + ':'.join(str(key) for key in col.keytup) + ')'
            if 'offset' in col:
                index = f'{index} offset={col.offset}'
        if include_unit:
            if 'index_unit' in col:
                return f'{index} {col.index_unit}'
            if 'unit' in col and _indexable_regex.match(col.unit):
                return f'{index} {col.unit}'
            elif 'index_unit' in self:
                return f'{index} {self.index_unit}'
            elif 'unit' in self and _indexable_regex.match(self.unit):
                return f'{index} {self.unit}'
        return index


def _proc_col_gategroup(imdata:"Column", *args)->dict:
    """Pre-process for Column gategroup, identifies origin_param"""
    return dict(origin_param=imdata.param.origin_param)


def _check_gategroup(val:"GateGroup", origin_param:Param=None, **kwargs):
    """Check for Column gategroup, verifies origin params are compatible"""
    val = GateGroup.as_gategroup(val)
    if "columns" in val and val.origin_param != origin_param:
        raise ValueError("param and gate do not share the same base shape")
    return val


_TV_gategroup = TV_ImData(data_proc=_proc_col_gategroup,
                          validator=_check_gategroup)


def _proc_column_offset(imdata:"Column", *args)->dict:
    """Pre-process for Column offset, sets min/max value"""
    offset = imdata._get_coldef().offset
    return dict(mn=0 if offset > 0 else offset, mx=abs(offset))


def _proc_column_keytup(imdata:"Column", *args)->dict:
    """Pre-process for Column keytuple, sets column_def"""
    return dict(coldef=imdata._get_coldef())


def _make_column_keytup(val:tuple[Hashable,...], coldef:ColumnDef=None, **kwargs)->tuple[Hashable,...]:
    """Check function for Column keytup, verifies correct type/range for each value"""
    if isinstance(val, list):
        val = tuple(val)
    elif not isinstance(val, tuple):
        val = (val, )
    keytup = val
    if 'mapto' in coldef:
        keytup = val[1:]
        if not issubclass(val[0].tp, coldef.mapto):
            raise ValueError(
                f"{coldef.name} columns must map to {coldef.mapto.__name__} based parameters, got {val[0].tp.__name__}")
    if len(keytup) != len(coldef.keytypes):
        raise ValueError(f"incorrect number of keys to specify {coldef.name} column, "+
                         f"expected {len(coldef.keytypes)}, got {len(keytup)}")
    out = tuple(tv.check_val(kt) for kt, tv in zip(keytup, coldef.keytypes))
    if 'mapto' in coldef:
        out = (val[0],) + out
    return out


def _proc_column_string(imdata:"Column", *args)->dict:
    """Pre-process for Column name argument"""
    return dict(isin=tuple(coldef.name for coldef in imdata.source_param.tp.column_defs if 'remap' not in coldef))


class Column(_ImData):
    """
    Representation of a column within a table. At minimum to create a column,
    a ``source_param`` (a :class:`Param` ) and a ``col`` must be specified
    in order to define a Column. The ``col`` attribute is a string, the options
    of which are defined by the :class:`Table` type of the :class:`Param` .
    Some ``col`` specification also require additional keys, like photon selections
    etc. that must be defined in the ``keytup`` argument.
    
    If ``col`` specifies an offset, the ``offset`` field may be present, to specify
    how to align the rows of the offset column with non-offset.
    If the "natural" column has a negative offset (fewer rows than normal), then
    the ``fill`` attribute is included. ``fill`` is also present for non-atomic
    
    For ``col`` that are non-atomic, the ``gategroup`` specifies the rows to return,
    
    
    Parameters
    ----------
    source_param : Param
        The table from which the column derives
    col : str
        The name of the column
    keytup : tuple[Hashable,...]
        tuple of hashables specifying sub-column (depends on col)
    offset : int, optional
        Offset of column (only for specific columns with offset). Optional even
        for negative offset column types. Note however that when creating a gate,
        the input columns must have offset if column offset is non-zero.
    fill : Any, optional
        Value with with to fill non-specified values. Present only when such rows
        exist, namely column-types with negative offset
    gategroup : GateGroup, optional
        For atomic columns, this attribute is never present, rather it is applied
        to the param. For non-atomic columns, the param defines the values for
        which the column is evaluated, and the gategroup is applied afterwards.
    unit : str, optional
        **Not hashed** string for unit of column
    title : str, optional
        **Not hashed** string for title of column.
    index_title : str, optional
        **Not hashed** string for name of column in csv files, must not contain
        commas.
    """
    __slots__ = ('source_param', 'col', 'keytup', 'offset', 'fill', 'gategroup', 
                 'title', 'unit', 'index_title', 'index_unit', 'title_is_tex')
    _typeconversions = ImDict(source_param=TV_Param, col=TV_str(data_proc=_proc_column_string),
                              keytup=TV_tuple(
                                  validator=_make_column_keytup, data_proc=_proc_column_keytup),
                              offset=TV_int(data_proc=_proc_column_offset),
                              gategroup=_TV_gategroup, title=TV_str, unit=TV_str,
                              index_title=_TV_indexable, index_unit=_TV_indexable,
                              title_is_tex=TV_int(isin=(-1,0,1)))
    _required = frozenset({'source_param', 'col'})
    _hashskip = ('title', 'unit', 'index_title', 'index_unit', 'title_is_tex')

    source_param: Param #: Parameter defining table from which Column is defined, should be degated
    col: str  #: String idenifying the column from the table being defined
    keytup: tuple[Hashable, ...] #: tuple of keys (often empty) defining sub-column of table
    offset: int  #: only for columns with offset from shape, defines offset in indeces to allign for given purpose
    fill: Any  #: only when column with offset is negative, value to fill missing offsets with
    gategroup: "GateGroup"  #: the final gate of column, defines the

    def __new__(cls, source_param:Param, col:str, keytup:Sequence[Hashable]=None, 
              offset:int=None, fill:Any=None, gategroup:"GateGroup"=None, 
              title:str=None, unit:str=None, index_title:str=None, index_unit:str=None):
        coldef = source_param.tp._get_columndef(col)
        # regularize ketup and offset into kwargs for processing
        keytup = tuple() if keytup is None else keytup
        keytup = tuple(keytup) if isinstance(keytup, Sequence) else (keytup, )
        kwargs = dict(source_param=source_param, col=col, keytup=keytup)
        kwargs.update({key:val for key, val in zip(('title', 'unit', 'index_title', 'index_unit'), 
                                                   (title, unit, index_title, index_unit))
                       if val is not None})
        if offset is not None:
            if coldef.offset == 0 and offset != 0:
                raise ValueError(f"column {col} has not offset, must be 0 or None")
            if coldef.offset > 0 and fill is not None:
                raise ValueError(f"column {col} is a possitive offset column, cannot specify fill")
            kwargs['offset'] = offset
            if coldef.offset != 0 and fill is not None:
                kwargs['fill'] = fill
        if 'remap' in coldef:
            if coldef.reg_func:
                kwargs['keytup'] = getattr(kwargs['source_param'].tp, coldef.reg_func)(*kwargs['keytup'])
            args = tuple(kwargs[k] for k in ('col', 'keytup', 'offset', 'fill') if k in kwargs)
            remapped = getattr(kwargs['source_param'].tp, coldef.remap)(*args)
            ukwargs = kwarg_like(cls.__slots__[1:], remapped)
            kwargs.update(**{key:val for key, val in ukwargs.items() 
                             if key not in cls._hashskip or key not in kwargs}) # skip user set names
            coldef = kwargs['source_param'].tp._get_columndef(kwargs['col'])
        if coldef.reg_func:
            kwargs['keytup'] = getattr(kwargs['source_param'].tp, coldef.reg_func)(*kwargs['keytup'])
        kwargs.update(kwargs['source_param'].tp._regularize_column_kwargs(**{k:v for k, v 
                                                                            in kwargs.items() 
                                                                            if k not in cls._hashskip}))
        if gategroup is not None:
            kwargs['gategroup'] = gategroup
        return super().__new__(cls, **kwargs)

    def __post_init__(self):
        # check offset
        coldef = self._get_coldef()
        # ensure no gate in parameter for atomic params
        param: Param = self.keytup[0] if 'mapto' in coldef else self.source_param
        if coldef.atomic:
            if 'gategroup' in self:
                param = param.regate(self.gategroup & param.base_gate)
                if 'mapto' in coldef:
                    super(_ImData, self).__setattr__('keytup', (param, )+self.keytup[1:])
                else:
                    super(_ImData, self).__setattr__('source_param', param)
                super(_ImData, self).__delattr__('gategroup')
        else:
            if 'gategroup' not in self:
                super(_ImData, self).__setattr__('gategroup', param.base_gate)
            elif coldef.offset != 0 and self.gategroup != param.base_gate and 'offset' not in self:
                raise TypeError("must specify offset for gated non-atomic offset column")
        # check that offset is in range of [-column offset, column offset]
        if 'offset' in self and abs(coldef.offset) < self.offset:
            raise ValueError("offset of {self.offset} too large for collumn "+
                             "with offset of {coldef.offset}")
        # column offset of 0 means offset should not be specified
        if coldef.offset == 0:
            if 'offset' in self:
                super(_ImData, self).__delattr__('offset')
        # remove for non-negative column offsets, (no values to fill when column in larger than table size)
        if coldef.offset >= 0:
            if self.atomic:
                if 'fill' in self:
                    raise ValueError("fill can only be specified for negative offset values")
            else:
                if GateGroup.overlap(self.gategroup, self.param.base_gate) & 0b0010:
                    super(_ImData, self).__setattr__('fill', coldef.fill)
                elif 'fill' not in self:
                    if 'fill' in self:
                        super(_ImData, self).__delattr__('fill')
        # auto-set fill for negative column offsets when fill not already specified
        if coldef.offset < 0 and 'offset' in self and 'fill' not in self:
            if 'fill' in coldef:
                super(_ImData, self).__setattr__('fill', coldef.fill)
            elif np.issubdtype(coldef.dtype, np.floating):
                super(_ImData, self).__setattr__('fill', np.nan)
            elif np.issubdtype(coldef.dtype, np.integer):
                super(_ImData, self).__setattr__('fill', 0)
            elif np.issubdtype(coldef.dtype, np.str_):
                super(_ImData, self).__setattr__('fill', '')
            else:
                super(_ImData, self).__setattr__('fill', None)
        if 'fill' in self and not np.issubdtype(type(self.fill), coldef.dtype):
            try:
                fill = np.array(self.fill).astype(coldef.dtype).reshape(1)[0]
            except:
                raise ValueError("incompatible dtype of fill, expected " +
                                 f"{coldef.dtype}, got {type(self.fill)}")
            super(_ImData, self).__setattr__('fill', fill)
        if coldef.check_func:
            getattr(self.source_param.tp, coldef.check_func)(self)

    @property
    def param(self)->Param:
        """The :class:`Param` to which the Column is mapped"""
        if 'mapto' in self._get_coldef():
            return self.keytup[0]
        return self.source_param

    def _get_coldef(self)->ColumnDef:
        """Retrieve columndef of column"""
        return self.source_param.tp._get_columndef(self.col)

    @property
    def _get_func_args(self)->tuple[Hashable,...,int,Number]:
        """Get args for computing column form iter_func or get_func"""
        out = (self.col, ) + self.keytup
        if 'offset' in self:
            out += (self.offset,)
        if 'fill' in self:
            out += (self.fill, )
        return out

    def degate(self)->"Column":
        """Version of object which has no gate applied"""
        coldef = self._get_coldef()
        if coldef.atomic:
            if 'mapto' in coldef:
                return self._replace_fields(fields={'keytup':(self.keytup[0].degate(),)+self.keytup[1:]})
            return self._replace_fields(fields={'source_param':self.source_param.degate()})
        return self._replace_fields(fields={'gategroup':self.param.base_gate})

    def regate(self, gate:"GateGroup", *fill)->"Column":
        """
        Create a :class:`Column` otherwise identical to self, but with gate of ``gate``

        Parameters
        ----------
        gate : GateGroup
            New :class:`GateGroup` to apply to Column.

        Raises
        ------
        ValueError
            Incompatible gate (for a different base_param``.
        warnings
            gate is larger than usually used for particlar sort of param.

        Returns
        -------
        Column
            Regated :class:`Column`.

        """
        if not gate.nogate and gate.origin_param != self.param.origin_param:
            raise ValueError("gate and column do not share origin_param")
        elif gate.nogate and 'param' not in gate:
            gate = GateGroup(gate.truthtable, param=self.origin_param)
        if GateGroup.overlap(gate, self.param.max_gate) & 0b0010:
            raise warnings.warn("regating to gate larger than max_gate of param")
        if not self.atomic:
            repl =  {'gategroup': gate}
        elif 'mapto' in self._get_coldef():
            repl = {'keytup': (self.keytup[0].regate(gate),)+self.keytup[1:]}
        else:
            repl = {'source_param': self.source_param.regate(gate)}
        out = self._replace_fields(fields=repl)
        if len(fill) == 1:
            out = out.replace_fill(*fill)
        elif len(fill) > 2:
            raise ValueError("regate takes maximum of 2 arguments")
        return out

    def replace_fill(self, fill:Hashable)->"Column":
        """
        Return version of Column with new fill value

        Parameters
        ----------
        fill : Hashable
            Fill to change.

        Raises
        ------
        ValueError
            Column has no fill value to replace.

        Returns
        -------
        Column
            Column with updated fill.

        """
        coldef = self._get_coldef()
        if self.atomic and coldef.offset >= 0:
            raise ValueError("atomic, non-negative offset columns do not have fill attribute")
        return self._replace_fields(fields={'fill':fill})

    @property
    def atomic(self)->bool:
        """If ``True`` column values are independent of gate"""
        return self._get_coldef().atomic

    @property
    def base_param(self)->Param:
        """
        The :class:`Param` defining the rows of the column.
        Has ``tp`` which is subclass of :class:`BaseTable`
        """
        return self.param.base_param

    @property
    def base_gate(self)->"GateGroup":
        """The :class:`GateGroup` of the :attr:`Column.base_param`"""
        return self.gategroup if 'gategroup' in self else self.param.base_gate

    @property
    def parent_param(self)->Param:
        """
        :class:`Param` used for computing the gate of :attr:`Param.base_param`.

        .. note::

            This is usually the same as origin_param, and is only different if
            the gate is non-atomic
        """
        return self.parent_gate.base_param

    @property
    def parent_gate(self)->"GateGroup":
        """
        The parent_gate of the gate of this param.

        .. note::

            For atomic gates, this is always GG_all, it is not GG_all only if the
            gate is non-atomic.
        """
        if self.atomic:
            return self.param.parent_gate
        return self.param.base_gate

    @property
    def origin_param(self)->Param:
        """The Param defining all rows in the table, disregarding all gates"""
        return self.param.origin_param
    
    @property
    def size_offset(self)->int:
        """Column type-offset"""
        if 'offset' in self:
            return 0
        return self._get_coldef().offset
    
    def name(self, include_unit:bool=False, origin:"DataSet"=None)->str:
        """
        Get a string name of the column, starting with user set names, if present,
        and falling back on successive defaults.

        Parameters
        ----------
        include_unit : bool, optional
            Whether or not to incldue the unit in the name of the column. 
            The default is False.

        Returns
        -------
        str
            String name for self.

        """
        origin = origin.datas[0] if isinstance(origin, DataSetList) else origin
        kwargs = dict() if origin is None else {'origin':origin}
        return self._get_coldef().get_name(self, include_unit=include_unit, **kwargs)
        
    def index_name(self, include_unit:bool=False, origin:"DataSet"=None)->str:
        """
        Get a csv string anme of the column, starting with user set names, if present,
        and falling back on successive defaults.

        Parameters
        ----------
        include_unit : bool, optional
            Whether or not to incldue the unit in the name of the column. 
            The default is False.

        Returns
        -------
        str
            String name for self.

        """
        kwargs = dict() if origin is None else {'origin':origin}
        return self._get_coldef().get_index(self, include_unit=include_unit, **kwargs)
    
    @property
    def description(self)->str:
        """A YAML-like description of the Column"""
        return self.source_param.tp.get_column_description(self)
    
    @property
    def _sort_tuple(self)->tuple[Hashable]:
        out = [self.source_param._sort_tuple, self.col, _make_sortable(self.keytup)]
        if 'offset' in self:
            out.append(self.offset)
        if 'fill' in self:
            out.append(_make_sortable(self.fill))
        return tuple(out)


register_type(Column)
TV_Column = TV_ImData(subclass=Column)


def _column_sort(column:Column)->tuple:
    """
    Generate tuple from Column that can be used to sort columns. 
    Function used in as value of key in sorted
    """
    return column._sort_tuple


###############################################################################
# Name overlap codes as distinct variables
###############################################################################
# Overlap codes are bitmasks, each of the 4 bits represents whether a
# particular combination gateA/gateB True/False values is possible.
# If a combination is possible, the bit is 1, if impossible, the bit is 0.
# Bits are assigned like so 1<<0xBA
# The result is the following table:
###
# +----------+----------+----------+----------+
# | bit 3    | bit 2    | bit 1    | bit 0    |
# +----------+----------+----------+----------+
# | A=T, B=T | A=F, B=T | A=T, B=F | A=F, B=F |
# +----------+----------+----------+----------+
###############################################################################
GD_error: int = 0b0000 #: Indicates comparison between gates that cannot be compared
GD_nonenone: int = 0b0001 #: both gates are none gates
GD_noneall: int = 0b0010 #: gateB is none, gateA is all
GD_nonesome: int = 0b0011 #: gateB is none, gateA is neither none nore all
GD_allnone: int = 0b0100 #: gateB is all, gateA is none
GD_somenone: int = 0b0101 #: gateA is none, gateB is neither none nor all
# : Gate is inverse of other, ie where inside gate is True, outside gate is False, 
# and vise versa, NOTE: this is a subset of exclude
GD_complement: int = 0b0110
# : no overlap of True values, note this is a superset of inverse, if is both 
# inverse and exclude, should return inverse
GD_disjoint: int = 0b0111 #: Sets are disjoint- no overlap
GD_allall: int = 0b1000 #: Both gates are all gates
GD_equal: int = 0b1001 #: Gates are equal to each other
GD_someall: int = 0b1010 #: GateB is all, gate A is neither all or none
GD_superset: int = 0b1011  #: gateB is entirely inside gateA
GD_allsome: int = 0b1100 #: gateB is all gate, gateA is neither all or none
GD_subset: int = 0b1101  #: gateA is entirely inside gateB
#: Union of gates results in all. Both gates are not all
GD_unioncomplete: int = 0b1110
#: Gates are intersecting most common, sets are independent, all areas of ven diagram covered
GD_intersect: int = 0b1111
#: Valid overlap codes for a comparison function
GD_overlapcodes: tuple[int, ...] = frozenset({GD_error, GD_nonenone, GD_noneall,
                                              GD_nonesome, GD_allnone, GD_somenone,
                                              GD_complement, GD_disjoint, GD_allall,
                                              GD_equal, GD_someall, GD_subset,
                                              GD_allsome, GD_superset, GD_unioncomplete,
                                              GD_intersect})


def _GD_reverse(val):
    """convert code for a comp b to b comp a"""
    return (val & 0b1001) + ((val & 0b0100) >> 1) + ((val & 0b0010) << 1)


#: dictionary giving result if comparision order is reversed, ie b overlap a
GD_reverse_map = ImDict({g: _GD_reverse(g) for g in GD_overlapcodes})

GD_absolutes: frozenset[int] = frozenset({GD_nonenone, GD_noneall, GD_nonesome, GD_allnone,
                                         GD_somenone, GD_allall, GD_someall, GD_allsome})
GD_absolutes0: frozenset[int] = frozenset({GD_nonenone, GD_noneall, GD_allnone,
                                          GD_somenone, GD_allall, GD_someall})
GD_absolutes1: frozenset[int] = frozenset({GD_nonenone, GD_noneall, GD_nonesome,
                                          GD_allnone, GD_allall, GD_allsome})
GD_nones0: frozenset[int] = frozenset({GD_nonenone, GD_allnone, GD_somenone})
GD_nones1: frozenset[int] = frozenset({GD_nonenone, GD_noneall, GD_nonesome})
GD_alls0: frozenset[int] = frozenset({GD_noneall, GD_allall, GD_someall})
GD_alls1: frozenset[int] = frozenset({GD_allnone, GD_allall, GD_allsome})
GD_commons: frozenset[int] = frozenset({GD_complement, GD_disjoint, GD_equal, GD_subset,
                                       GD_superset, GD_unioncomplete, GD_intersect})


def _GD_invert(val: int)->int:
    """Convert code where from a comp b to ~a comp b"""
    return (0b1100 & val) >> 2 + (0b0011 & val) << 2


#: dictionary giving result of ~a overlap b
GD_invert_map: ImDict = ImDict({g: _GD_invert(g) for g in GD_overlapcodes})

#: dictionary given overlap code, what other overlap codes are subsets thereof
GD_subsets: ImDict = ImDict({i: frozenset(j for j in GD_overlapcodes
                                          if not (~i & j)) for i in GD_overlapcodes})

class GateDefinitionError(TypeError):
    """Error defining GateDef"""
    pass


class GateDefinition:
    """
    Abstract Base Class for GateDef and MappedGateDef, implements unified
    methods for adding gate comparison functions.
    """
    _gate_comps = FixedDict() #: gate comparison functions, should remain hidden

    def __post_init__(self):
        if self.func in self._registered_funcs:
            if self._registered_funcs[self.func] != self:
                if self.func.__module__ == '__main__':
                    warnings.warn(f"re-registering gate func {self.func}")
                else:
                    raise ValueError(f're-registering {self.func.__name__}')
        else:
            self._registered_funcs[self.func] = self

    @classmethod
    def set_gate_comparison(cls, gateA:"GateDefinition", gateB:"GateDefinition", 
                            func:Callable[["GateDefinition","GateDefinition"], int])->None:
        """
        Set a comparison function for two gate-types.

        Parameters
        ----------
        gateA : GateDefinition
            First GateDefinition (either GateDef or MappedGateDef).
        gateB : GateDefinition
            Second GateDefinition (either GateDef or MappedGateDef).
        func : Callable[[GateDefinition,GateDefinition], int]
            Function that will compute overlap code between gates of gatedef 
            gateA and gateB `func(gateA, gateB)->overlap code`.

        """
        if not isinstance(gateA, GateDefinition):
            raise TypeError("gateA must be a GateDefinition")
        if not isinstance(gateB, GateDefinition):
            raise TypeError("gateB must be a GateDefinition")
        if not callable(func):
            raise TypeError("func must be a callable")
        fparam = inspect.signature(func).parameters
        if any(p.kind == p.KEYWORD_ONLY and p.default == inspect._empty for p in fparam.values()):
            raise TypeError("func cannot have required kwargs")
        nreq = sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) 
                   and p.default == inspect._empty for p in fparam.values())
        if nreq > 2:
            raise TypeError("Too many required arguments in func, must accept maximum of 2 arguments")
        if nreq != 2 and all(p.kind != p.VAR_POSITIONAL for p in fparam.values()):
            raise TypeError("func accepts too few arguments, must be able to accept 2 arguments")
        cls._gate_comps[(gateA, gateB)] = func

    @classmethod
    def gate_compare(cls, gateA:"Gate_", gateB:"Gate_")->int:
        """
        Find overlap between gateA and gateB. Returns bitcode.
        
        +----------+----------+----------+----------+
        | bit 3    | bit 2    | bit 1    | bit 0    |
        +----------+----------+----------+----------+
        | A=T, B=T | A=F, B=T | A=T, B=F | A=F, B=F |
        +----------+----------+----------+----------+

        Parameters
        ----------
        gateA : Gate
            First gate.
        gateB : Gate
            Second gate.

        Raises
        ------
        GateDefError
            Back comparison function.

        Returns
        -------
        int
            Bit code for overlap

        """
        if (gateA.gatedef, gateB.gatedef) in cls._gate_comps:
            code = cls._gate_comps[gateA.gatedef, gateB.gatedef](gateA, gateB)
        elif (gateB.gatedef, gateA.gatedef) in cls._gate_comps:
            code = _GD_reverse(
                cls._gate_comps[gateB.gatedef, gateA.gatedef](gateB, gateA))
        elif (gateA.gatedef, type(gateB.gatedef)) in cls._gate_comps:
            code = cls._gate_comps[gateA.gatedef, type(gateB.gatedef)](
                gateA, gateB)
        elif (gateB.gatedef, type(gateA.gatedef)) in cls._gate_comps:
            code = _GD_reverse(
                cls._gate_comps[gateB.gatedef, type(gateA.gatedef)](gateB, gateA))
        else:
            code = GD_intersect
        if code not in GD_overlapcodes:
            raise GateDefinitionError("comparison function returned invalid code")
        if gateA.base_gate.truthtable.ndim:
            code = _parent_overlap(code, gateA, gateB)
        elif gateB.base_gate.truthtable.ndim:
            code = _GD_reverse(_parent_overlap(_GD_reverse(code), gateB, gateA))
        return code

    @property
    def name(self)->str:
        """Name of function of gate"""
        return f'{self.func.__module__}.{self.func.__name__}'


def _parent_overlap(code, gateA, gateB):
    """Evaluates overlap of two gates where gate a is non-atomic."""
    pcode = GateGroup.overlap(gateA.base_gate, GateGroup.as_gategroup(gateB))
    if gateA.expand:
        code = (0b0101&(code & pcode>>1))|(0b1010&((code&pcode)|(code<<1&pcode<<1)|(code&pcode<<1)))
    else:
        code = (code & pcode) | (0b0101 & ((pcode>>1&code)|(pcode&code>>1)))
    return code


def _gatedef_verify_nooffset(cols:tuple[Column,...], param:tupledict)->None:
    """
    Default Verify function for :attr:`GateDef.verify`, 
    ensures all columns will output same number of rows (no residual offset)
    """
    if any(col.size_offset != 0 for col in cols):
        raise ValueError("columns must have same size")


register_PyCode(_gatedef_verify_nooffset)


class GateDef(GateDefinition, _ImData):
    r"""
    Class used to specify a gate function. The first attribute, `func` is the
    function used to compute the gate from arrays of columns. This class can
    largely be thought of as a wrapper around such a gate function.
    
    Parameters
    ----------
    func : Callable[[np.ndarray,...],np.ndarray[np.bool\_]]
    params : tupledict
    nparents : np.ndarray[Number, Number], optional
        Min and Max (inclusive) number of columns in gate. The default is [1, inf].
    atomic : bool, optional
        If gate is atomic (each row is independent of other rows). The default is True.
    sortcol : bool, optional 
    regularize : Callable[[dict],dict]|Callable[[dict,tuple[int,...]],dict], optional
        Regularizaton function, if sortcol is True, has signature
        ``regularize(params:dict, sortcol:tuple[int,...])->dict``, 
        if sortcol is False, has signature
        ``regularize(params:dict)->dict``, returned dictionary should be
        regularized params. If not set the default is to pass.
    verify : Callable[[tuple[Column, ...], tupledict], None], optional
        Function to call in post_init to check Gate is valid.
        Has signature ``verify(column:tuple[Column,...], params:tupledict)->None``
        If not set the default is to pass.
    """
    __slots__ = ('func', 'params', 'nparents', 'atomic', 'sortcol', 'regularize', 'verify', 'defaults')
    _typeconversions = ImDict(func=TV_PyCode, params=TV_tupledict(typedefs=TV_type),
                       nparents=TV_ndarray(dims=arr_slc[2], mn=0, superdtype=np.number),
                       atomic=TV_bool, sortcol=TV_bool,
                       verify=TV_PyCode, regularize=TV_PyCode)
    _required = frozenset({'func', })
    _defaults = ImDict(params=lambda: dict(), nparents=lambda: np.array([1, np.inf]),
                       atomic=True, sortcol=False, 
                       regularize=lambda: _echo, verify=lambda: _gatedef_verify_nooffset)
    _hashskip = ('defaults', )
    _registered_funcs = FixedDict()

    func: Callable[[], np.ndarray[np.bool_]]
    params: tupledict[str, type]
    nparents: tuple[Real, Real]
    atomic: bool
    sortcol: bool
    regularize: Callable[[dict],dict]
    verify: Callable[[tuple[Column, ...], tupledict], None]
    defaults: dict[str, Any]


TV_GateDef = TV_ImData(subclass=GateDef)


def _pass_all(*args, **kwargs):
    """Passive function for verification checks- always passes"""
    pass


register_PyCode(_pass_all)

class MappedGateDef(GateDefinition, _ImData):
    r"""
    Class used to specify MappedGate functions.
    
    Parameters
    ----------
    func: Callable[[np.ndarray[np.bool\_],...],np.ndarray[np.bool\_]]
        Function defining the re-mapping function, has signature func(mask, \*\*params)
    params: tupledict[str, type]
        Definition of names:types of params to function
    verify: Callable[[dict],None]
        Validation function. Should raise error if params are invalid.
        Params are given as single tupledict in first argument
    defaults: dict[str, Any]
        Dictionary of default values for params.
    
    """
    __slots__ = ('func', 'params', 'verify', 'defaults')
    _typeconversions = ImDict(func=TV_PyCode, params=TV_tupledict(typedefs=TV_type), 
                              verify=TV_PyCode)
    _required = frozenset({'func', 'params'})
    _defaults = ImDict(verify=_pass_all)
    _hashskip = ('defaults', )
    _registered_funcs = FixedDict()

    func: Callable[[np.ndarray[np.bool_],tupledict],np.ndarray[np.bool_]]
    params: tupledict[str, type]
    verify: Callable[[dict],None]
    defaults: dict[str, Any]


TV_MappedGateDef = TV_ImData(subclass=MappedGateDef)

_TT_all = np.array(True, dtype=np.bool_)

_TT_none = np.array(False, dtype=np.bool_)

_TT_ft = np.array([False,   True], dtype=np.bool_)

_TT_tf = np.array([True,   False], dtype=np.bool_)

#: Truthtable for `&` gate
TT_and = np.array([[False, False],
                   [False,  True]], dtype=np.bool_)
#: Truthtable for `|` gate
TT_or = np.array([[False,  True],
                  [True,   True]], dtype=np.bool_)
#: Truthtable for equality (`@` gate)
TT_equal = np.array([[True,  False],
                     [False,  True]], dtype=np.bool_)
#: Truthtable for `^` gate
TT_nequal = ~TT_equal
#: Truthtable for forward implication (`>>` gate)
TT_implies = np.array([[True,   True],
                       [False,  True]], dtype=np.bool_)
#: Truthtable for subtraction (`-` gate)
TT_subtract = ~TT_implies

_TT_all.setflags(write=False)
_TT_none.setflags(write=False)
_TT_ft.setflags(write=False)
_TT_tf.setflags(write=False)
TT_and.setflags(write=False)
TT_or.setflags(write=False)
TT_equal.setflags(write=False)
TT_nequal.setflags(write=False)
TT_implies.setflags(write=False)
TT_subtract.setflags(write=False)

TT_name_map = ImDict({"and":TT_and, "or":TT_or, "equal":TT_equal,
                     "nequal":TT_nequal, "implies":TT_implies})
TT_int_map = ImDict(
    {0:TT_and, 1:TT_or, 2:TT_equal, -2:TT_nequal, 3:TT_implies})


def _check_TV_gateparam(val:dict, imdata:"Gate", colorder:tuple[int,...]=None, **kwargs):
    """Check function for :attr:`Gate.params` attribute, calls regularize for colorder"""
    if imdata['gatedef']['sortcol']:
        return imdata['gatedef']['regularize'](val, colorder)
    return imdata['gatedef']['regularize'](val)


def _proc_TV_gateparam(dct:"Gate", kwarg_append:dict)->dict:
    """Pre-process function for :attr:`Gate.params` attribute- sets order, typedefs and defaults"""
    order = tuple(dct['gatedef']['params'].keys())
    out = dict(order=order, required=order, typedefs=dct['gatedef']['params'],
                defaults=dct['defaults'] if 'defaults' in dct else {}, imdata=dct)
    if kwarg_append.get('colorder'):
        out['colorder'] = kwarg_append['colorder'][0]
    return out


def _check_columnsbase(val: tuple[Column,...], atomic_func:bool=False, sort=False, **kwargs)->tuple[Column,...]:
    """Check function for :attr:`Gate.columns` attribute- ensures correct order and common origin_param"""
    if isinstance(val, Column):
        val = (val, )
    if any(not isinstance(err := v, Column) for v in val):
        raise TypeError(f"Incorrect type in columns: {type(err)}, all values must be Column or ColumnMap")
    origin_params = tuple(c.origin_param for c in val)
    if any(origin_params[0] != b for b in origin_params[1:]):
        raise ValueError("columns must share origin_param")
    if atomic_func and any(col.size_offset != 0 for col in val):
        raise ValueError("columns of offset params must have offset specified")
    if atomic_func:
        val = tuple(col.degate() for col in val)
    else:
        gate = val[0].base_gate
        for col in val[1:]:
            gate = gate & col.base_gate
        val = tuple(col.regate(gate) for col in val)
    if sort:
        new_val = tuple(sorted(val, key=_column_sort))
        kwargs['colorder'].append(tuple(chain.from_iterable((i for i, old in 
                                                             enumerate(val) if v == old) 
                                                            for v in new_val)))
        val = new_val
    return val


def _proc_columnsdata(imdata:"Gate", kwarg_append:dict)->dict:
    mn, mx = imdata["gatedef"]["nparents"]
    sort = imdata['gatedef']['sortcol']
    kwarg_append['colorder'] = list()
    return dict(minsize=mn, maxsize=mx, atomic_func=imdata['gatedef'].atomic, sort=sort, colorder=kwarg_append['colorder'])


class Gate_:
    """
    Abstract Base Class for single gate objects.
    Defines logical operation name, func and description methods.
    """
    def __invert__(self) -> "GateGroup":
        return GateGroup(_TT_tf, self)

    def __and__(self, other):
        return GateGroup(TT_and, self, other)

    def __mul__(self, other):
        return self & other

    def __or__(self, other):
        return GateGroup(TT_or, self, other)

    def __add__(self, other):
        return self | other

    def __matmul__(self, other):
        return GateGroup(TT_equal, self, other)

    def __xor__(self, other):
        return GateGroup(TT_nequal, self, other)

    def __sub__(self, other):
        return GateGroup(TT_subtract, self, other)

    def __rshift__(self, other):
        return GateGroup(TT_implies, self, other)

    def __lshift__(self, other):
        return GateGroup(TT_implies, other, self)

    @property
    def _expand_false(self)->"Gate":
        """:class:`Gate` where :attr:`Gate.expand` is ``False``, if non-atomic"""
        if "expand" in self:
            return self._replace_fields(fields={"expand": False})
        return self

    @property
    def _expand_value(self)->bool:
        """False if atomic, otherwise value of expand"""
        return self.expand if 'expand' in self else False

    @property
    def name(self):
        """Name of GateDefinition, identifies the gating function"""
        return self.gatedef.name

    @property
    def func(self)->Callable:
        """Function that takes arrays of columns and params of gate to compute gate mask"""
        return self.gatedef.func

    def get_param_descr(self)->str:
        r"""Get string description of Gate\_"""
        return '\n'.join(self.base_param.tp._get_kv_str(key, val) for key, val in self.params.items())


class Gate(Gate_, _ImData):
    """
    Definition of gate (a.k.a. filter a.k.a. mask) over a particular :class:`BaseTable`
    
    Parameters
    ----------
    gatedef : GateDef
        Definition of gating function to apply to columns according to params.
        :class:GateDef` objects are wrappers around functions, which define
        what the parameters, and restrictions are for input values to function.
    columns : tuple[Column, ...]
        Tuple of columns used to create the gate. In cases of atomic GateDef and
        column, columns are automatically degated. If either is non-atomic,
        gate of column is retained.
    params : tupledict
        Parameters for gatefunction.
    expand : bool, optional
        **non-atomic gates only** value to fill rows of table outside of
        :attr:`Gate.base_gate` with when evaluating in :class:`GateGroup`
    title : str, optional, *not hashed*
        A string name for the Gate, note that this may not be carried over when
        :class:`GateGroup` s are created, as computing intersects sometimes
        creates new :class:`Gate`
    """
    __slots__ = ("gatedef", "columns", "params", "expand", "title")
    _typeconversions = ImDict(gatedef=TV_GateDef,
                              columns=TV_tuple(
                                  data_proc=_proc_columnsdata, validator=_check_columnsbase),
                              params=TV_tupledict(
                                  data_proc=_proc_TV_gateparam, validator=_check_TV_gateparam),
                                  expand=TV_bool, title=TV_str)
    _required = frozenset({'gate', 'columns', 'params'})
    _defaults = ImDict({'offset':0})
    _hashskip = frozenset({'title', })

    gatedef: GateDef  #: gate function, should take parameters as keyword arguments,
    columns: tuple[Column]  #: tuple of :class:`Column` given to :attr:`Gate.gatedef`
    params: tupledict[str, Hashable]  # : parameters defining gate, given as kwargs to :attr:`Gate.gatedef`
    expand: bool  # : the value for rows outside of parent_gate for non-atomic gates only
    title : str #: (mutable) Human readable name for Gate

    def __post_init__(self):
        if self.atomic:
            if 'expand' in self:
                raise ValueError("atomic columns do not have attribute 'expand'")
        elif 'expand' not in self:
            super(_ImData, self).__setattr__('expand', False)
        self.gatedef.verify(self.columns, self.params)

    @property
    def atomic(self)->bool:
        """Gate is dependant on selection of columns"""
        return self.gatedef.atomic and all(col.atomic for col in self.columns)

    @property
    def base_param(self)->Param:
        """
        :class:`Param` the defines the rows of the table which are handed to 
        :attr:`Gate.func` to evalute mask.
        """
        return self.base_gate.base_param

    @property
    def base_gate(self)->"GateGroup":
        """
        The :class:`GateGroup` the defines the rows of the table which are handed 
        to :attr:`Gate.func` to evalute mask.
        """
        base_gate = self.columns[0].base_gate
        for col in self.columns[1:]:
            base_gate = base_gate & col.base_gate
        return base_gate

    @property
    def parent_param(self)->Param:
        """:class:`Param` of :attr:`Gate.parent_gate`, the gate used to compute
        gate of :attr:`Gate.base_gate`.
        """
        return self.parent_gate.base_param

    @property
    def parent_gate(self)->"GateGroup":
        """
        :class:`GateGroup` used to compute columns of :attr:`base_gate`
        
        .. note::
            
            This is nearly always :attr:`Gate.origin_gate`, only in the case
            of multiply nested non-atomic gates is this untrue
        
        
        """
        parent_gate = self.columns[0].parent_gate
        for col in self.columns[1:]:
            parent_gate = parent_gate & col.parent_gate
        return parent_gate

    @property
    def origin_param(self)->Param:
        """
        :class:`Param` defining the table with no gates whatsoever.
        """
        return self.columns[0].origin_param

    def _columns(self)->tuple[Column,...]:
        """Columns regated to base_gate"""
        return tuple(col.regate(self.base_gate) for col in self.columns)

    def get_column_descrs(self, include_param:bool=False, include_source:bool=True)->str:
        """
        Get YAML-like description of all columns in gate

        Parameters
        ----------
        include_param : bool, optional
            Whether to include full :attr:`Column.param` in column description.
            The default is False.
        include_source : bool, optional
            For mapped columns only, whether to incldue source_param description. 
            The default is True.

        Returns
        -------
        str
            YAML-like definition of columns in Gate.

        """
        return '\n'.join(f'- {col.source_param.tp.get_column_descr(col, 0, include_param or not col.atomic, include_source)}'
                         for col in self.columns)

    def get_gate_description(self, indent:int=0, include_param:bool=False, include_source:bool=True)->str:
        """
        Get YAML-like description of Gate

        Parameters
        ----------
        indent : int, optional
            DESCRIPTION. The default is 0.
        include_param : bool, optional
            Whether to include full :attr:`Column.param` in descriptions of 
            columns. The default is False.
        include_source : bool, optional
            For mapped columns only, whether to incldue source_param in descriptions
            of columns. The default is True.

        Returns
        -------
        str
            YAML-like description of .

        """
        out = f'Gate: {self.gatedef.name}\nParams:\n{_indent(self.get_param_descr(),2)}\nColumns:\n'
        out += _indent(self.get_column_descrs(include_param, include_source), 2)
        return _indent(out, indent)
    
    @property
    def description(self)->str:
        """Descripton of gate"""
        return self.get_gate_description(include_param=True, include_source=True)
    
    @property
    def _sort_tuple(self)->tuple[Hashable,...]:
        """Tuple that can unambiguously be used to sort against other like objects"""
        out = [self.atomic, "Gate", self.gatedef.func.__name__, len(self.columns)]
        out += [tuple(col._sort_tuple for col in self.columns), _make_sortable(self.params)]
        return tuple(out)


def _gate_sort(gate:Gate_)->tuple:
    """Sorting function so gates can be unambiguously sorted into unchanging order"""
    return gate._sort_tuple


register_type(Gate)
TV_Gate = TV_ImData(subclass=Gate)


def _gate_locmask(gate:Gate, gategroup:Union[Gate,"GateGroup"])->bool:
    """Used to make tuple of if g should be true/false index in truthtable- 
    determines if gate is gategroup or a component thereof"""
    if isinstance(gategroup, GateGroup):
        return gate in gategroup.gates
    return gate == gategroup


def _mask_loc(gate_loc:tuple[int|None,...], loc:tuple[int])->tuple[int,...]:
    """mask loc based on gate_loc"""
    return tuple(l for g, l in zip(gate_loc, loc) if g)


def _proc_gategroup_columns(imdata:"GateGroup", kwarg_append:dict)->dict:
    """proc function for TypeValidator of GateGroup.gates, sets number of gates expected"""
    return dict(ncol=imdata.truthtable.ndim)


def _check_gategroup_columns(val:tuple[Column,...], ncol:int=None, **kwargs)->tuple[Column,...]:
    """Check function for GateGroup.gates, ensures correct number of gates and all Gates"""
    if isinstance(val, (Gate, GateGroup)):
        val = (val, )
    try:
        val = tuple(val)
    except:
        raise TypeError("Cannot interpret columns as tuple of columns")
    if len(val) != ncol:
        raise ValueError("expected {ncol} columns, but got {len(val)}")
    if any(not isinstance(err := v, (Gate_, GateGroup)) for v in val):
        raise TypeError(
            f"all elements of gates must be Gate or GateGroup objects, one is {type(err).__name__}")
    return val


def _check_gategroup_truthtable(val:np.ndarray[np.bool_], **kwargs):
    """Check func for truthtable of :class:`GateGroup`"""
    val = np.asarray(val, dtype=np.bool_)
    if any(s != 2 for s in val.shape):
        raise ValueError("truthtable must have all dimensions of size 2")
    return val


def _tt_index_single(n:int, i:int, j:int, vi:int, vj:int)->slice|int:
    if n == i:
        return vi
    if n == j:
        return vj
    return slice(None)

def _tt_tuple_index(n:int, i:int, j:int, vi:int, vj:int)->tuple[slice|int,...]:
    return tuple(_tt_index_single(nn, i, j, vi, vj) for nn in range(n))


class GateGroup(_ImData):
    r"""
    Class defines the rows of a gated :class:BaseTable`. It is fundamentally
    a logical operation on a set of :class:`Gate` arrays (a.k.a. fitler a.k.a. mask).
    
    Parameters
    ----------
    truthtable : np.ndarray[np.bool\_]
        n-d boolean array, all dimensions size 2. index 0 for ``False`` and 1
        for ``True`` defining which rows are 
        excluded (``truthtable[i,j,k...] == False``) and which are 
        included (``truthtable[i,j,k...] == True``) in the gated table
    gates : tuple[Gate\_,...]
        Tuple of gates (in order i, j, k ...) which serve as source of data
        to perform logical operation of truthtable
    param : Param, optional
        **Only for 0-d truthtable** When no :class:`Gate_` (s) (and therefore no 
        :class:`Columns` (s) to perform gating) what is the :attr:`GateGroup.origin_param`
        of GateGroup. This is optinal, if not supplied, GateGroup can be applied
        to any :class:`Param`
    title : str, optional, not hashed
        Optional string name to give to gate, not hashed and not included in
        logical operations.
    """
    __slots__ = ("truthtable", "gates", "param", "title")
    _typeconversions = ImDict(truthtable=TV_ndarray(dtype=np.bool_, validator=_check_gategroup_truthtable),
                              gates=TV_tuple(
                                  data_proc=_proc_gategroup_columns, validator=_check_gategroup_columns),
                              param=TV_Param, title=TV_str)
    _hashskip = ('title', )
    _required = frozenset({'truthtable', })

    # : truthtable for GateGroup, nd, where n is number of gates, all dimensions size 2
    truthtable: np.ndarray[np.bool_]
    #: n-lengths tuple of individual :class:`Gate` or :class:`MappedGate` objects
    #: that map to the indexes within :attr:`GateGroup.truthtable`
    gates: tuple[Gate_, ...] 
    param: Param #: The :class:`Param` defining which table the gate operates on

    def __new__(cls, *args, truthtable:str|np.ndarray[np.bool_]=None,
                gates:tuple[Union[Gate_,"GateGroup"],...]=None,
                param:Param|None=None, title:str=None):
        # GateGroup uses special new method that performs necessary broadcasts
        # This overwrites the standard setup, this __new__ method follows the
        # basic structure:
        #    1. Initial preprocessing
        #        a. parse arguments for specification in args or with kwargs
        #        b. Convert types
        #        c. Check valid dimensions etc.
        #    2. Perform expansion
        #        a. broadcast truthtable, as gates argument can include GateGroups
        #        b. Simplify/reduce truthtable and gates
        #        c. sort gates so gates have cannonical form
        #    3. Initiate table with super().__new__(...)
        # peel away args/kwargs specification (part 1b)
        if truthtable is None and args:
            truthtable, args = args[0], args[1:]
        if param is None and args and isinstance(args[-1], Param):
            args, param = args[:-1], args[-1]
        if gates is None:
            gates = args
            args = False
        if args:
            raise TypeError(
                "mixed args and kwargs style specification of values")
        # convert truthtable to numpy array, several possible maps (part 1b)
        if isinstance(truthtable, str):
            if truthtable not in TT_name_map:
                raise ValueError(
                    f"Unrecognized truthtable string {truthtable}")
            truthtable = TT_name_map[truthtable]
        elif isinstance(truthtable, int):
            if truthtable not in TT_int_map:
                raise ValueError(
                    f"Unrecognized truthtable int code, {truthtable}")
            truthtable = TT_int_map[truthtable]
        # check valid truthtable (1c)
        try:
            truthtable = np.asarray(truthtable, dtype=np.bool_)
        except Exception as e:
            raise TypeError(
                "truthtable must be value in TT_name_map, TT_int_map or boolean numpy array") from e
        if len(gates) != truthtable.ndim:
            raise ValueError(
                f"expected {truthtable.ndim} gates based on truthtable, got {len(gates)}")
        if any(d != 2 for d in truthtable.shape):
            raise ValueError("truthtable must have 2 elements per dimension")
        elif truthtable.ndim == 0:
            if param is None:
                return super().__new__(cls, truthtable, tuple())
            else:
                return super().__new__(cls, truthtable, tuple(), param)
        # check gates are Gate/GateGroups (1c)
        if any(not isinstance(err := g, (Gate_, GateGroup)) for g in gates):
            raise TypeError(
                f"all gates must be either Gate or GateGroup objects, not {type(err).__name__}")
        # check for shared base (1c)
        if param is None:
            for gate in gates:
                origin_param = gate.origin_param
                if origin_param is None:
                    continue
                if param is None:
                    param = origin_param
                elif param != origin_param:
                    raise ValueError("Gates have inconsistent origin_params")
        else:
            param = param.origin_param
        # Regularization requies gates to be list and truthtable needs to be modifiable
        gates = list(gates)
        truthtable = np.asarray(truthtable, dtype=np.int8)
        truthtable = truthtable if truthtable.flags['WRITEABLE'] else truthtable.copy()
        # broadcast truthtable, part 2a
        if any(isinstance(g, GateGroup) for g in gates):
            truthtable, gates = cls._broadcast_truthtable(truthtable, gates)
        #simplify truthtable, part 2c
        truthtable, gates = cls._simplify_truthtable(truthtable, gates)
        kws = dict() if title is None else {'title':title}
        # Check if table is all/none, can skip 2d, strait to initiation
        # set cannonical column order 2d
        if gates:
            gates, sort = zip(*sorted(([gate, i] for i, gate in enumerate(gates)), 
                                      key=lambda g: g[0]._sort_tuple))
            truthtable = np.asarray(truthtable.transpose(sort) > 0)
        return super().__new__(cls, truthtable, gates, param, **kws)

    @classmethod
    def _broadcast_truthtable(cls, truthtable:np.ndarray[np.bool_],
                              gategroups:tuple[Union[Gate_,"GateGroup"]]
                              )->tuple[np.ndarray[np.bool_],tuple[Union[Gate_,"GateGroup"]]]:
        """Expand a truthtable baesd on GateGroups into the form relying only on the Gates of each GateGroup"""
        # build set of all gates used
        # **NOTE** use of soted(set(..)) is necessary to remove duplicate gates
        all_gates = sorted(set(
            chain.from_iterable(g.gates if isinstance(g, GateGroup) else (g,) 
                                for g in gategroups)),key=_gate_sort)
        gate_tt = tuple(g.truthtable if isinstance(g, GateGroup) else _TT_ft 
                        for g in gategroups)
        gate_idxmap = tuple(tuple(_gate_locmask(g, gg) for g in all_gates)
                            for gg in gategroups)
        # allocate new table
        tt = np.empty([2 for _ in range(len(all_gates))], dtype=np.int8)
        # iterate over every position of output table and assign appropriate value
        for loc in product(*(range(2) for _ in range(len(all_gates)))):
            sloc = tuple(int(gg[_mask_loc(gl, loc)])
                         for gg, gl in zip(gate_tt, gate_idxmap))
            tt[loc] = truthtable[sloc]
        return tt, all_gates

    @classmethod
    def _simplify_truthtable(cls, truthtable:np.ndarray[np.int8],
                             gates:list[Gate_])->tuple[np.ndarray[np.bool_],list[Gate_]]:
        """Return truthtable, list[Gate] with all redundant gates removed"""
        for i in range(truthtable.ndim-1):
            for j in range(i+1, truthtable.ndim):
                overlap = GateDefinition.gate_compare(gates[i], gates[j])
                if overlap == 0b1111:
                    continue
                for vi in range(2):
                    for vj in range(2):
                        if not (overlap & 1<<(vi+2*vj)):
                            truthtable[_tt_tuple_index(truthtable.ndim, i, j, vi, vj)] = -1
        # check if truthtable is invariant relative to all gates, and reduce
        axi = 0
        while axi < truthtable.ndim:
            tt_f = np.asarray(truthtable[tuple(0 if i == axi else slice(None) for i in range(truthtable.ndim))])
            tt_t = np.asarray(truthtable[tuple(1 if i == axi else slice(None) for i in range(truthtable.ndim))])
            ig_f, ig_t = tt_f == -1, tt_t == -1
            if np.all(ig_f | ig_t | (tt_f == tt_t)):
                truthtable = tt_f
                truthtable[ig_f] = tt_t[ig_f]
                gates.pop(axi)
            else:
                axi += 1
        return truthtable, gates
    
    def __bool__(self):
        return True if len(self.gates) != 0 else bool(self.truthtable.reshape(1)[0])

    def __invert__(self):
        return self._replace_fields(fields={'truthtable':np.asarray(~self.truthtable)}, _strict=False)

    def __and__(self, other):
        return type(self)(TT_and, self, other)

    def __mul__(self, other):
        return self & other

    def __or__(self, other):
        return type(self)(TT_or, self, other)

    def __add__(self, other):
        return self | other

    def __matmul__(self, other):
        return type(self)(TT_equal, self, other)

    def __xor__(self, other):
        return type(self)(TT_nequal, self, other)

    def __sub__(self, other):
        return type(self)(TT_subtract, self, other)

    def __rshift__(self, other):
        return type(self)(TT_implies, self, other)

    def __lshift__(self, other):
        return type(self)(TT_implies, other, self)

    @property
    def nogate(self)->bool:
        """Boolean indicating whether gategroup is an all or none, or if it uses gates"""
        return not self.gates

    @property
    def single(self)->bool:
        """Boolean indicating if there is a single gate in gates, ie instance represents a single Gate"""
        return len(self.gates) == 1

    @property
    def atomic(self)->bool:
        return all(gate.atomic for gate in self.gates)

    @property
    def base_param(self)->Param:
        """Param that has the rows of self"""
        if 'param' not in self:
            return None
        return self.param.regate(self)

    @property
    def base_gate(self)->"GateGroup":
        """Same as self, present to keep base_gate property present in 
        :class:`Param`, :class:`Column` and :class:`GateGroup`"""
        return self

    @property
    def parent_param(self)->Param:
        """The :class:`Param` used to define the rows used to compute self.
        
        .. note::
            
            This is only different from :attr:`GateGroup.base_param` when self
            is non-atomic, and different from :attr:`GateGroup.origin_param` in
            cases of multiply nested non-atomic gategroups.
        """
        return self.parent_gate.base_param

    @property
    def parent_gate(self)->"GateGroup":
        """The :class:`GateGroup` used to define the rows used to compute self.
        
        .. note::
            
            This is only different from :attr:`GateGroup.base_gate` when self
            is non-atomic, and is not all in cases of multiply nested non-atomic 
            gategroups.
            
        """
        if self.nogate:
            return GateGroup(_TT_all, param=self.param)
        parent_gate = self.gates[0].parent_gate
        for gate in self.gates[1:]:
            parent_gate = parent_gate & gate.parent_gate
        return parent_gate

    @property
    def parent_gate_full(self)->"GateGroup":
        """
        The minimal :class:`GateGroup` the encompases the parent gate of all
        non-atomic columns. Ie the minimum number of columns covering all parent_gates
        """
        if self.nogate or self.atomic:
            return GateGroup(_TT_all, param=self.param)
        parent_gate = GateGroup(_TT_none, param=self.param)
        for gate in self.gates:
            if gate.atomic:
                continue
            parent_gate |= gate.parent_gate
        return parent_gate

    @property
    def origin_param(self)->Param:
        """:class:`Param` with no :class:`GateGroup` that defines all rows in table"""
        if 'param' not in self:
            return None
        return self.param

    @classmethod
    def overlap(cls, gateA:"GateGroup", gateB:"GateGroup")->int:
        """
        Compute the overlap of ``gateA`` and ``gateB`` as a bitcode

        +----------+----------+----------+----------+
        | bit 3    | bit 2    | bit 1    | bit 0    |
        +----------+----------+----------+----------+
        | A=T, B=T | A=F, B=T | A=T, B=F | A=F, B=F |
        +----------+----------+----------+----------+


        Parameters
        ----------
        gateA : GateGroup
            gategroup to compare.
        gateB : GateGroup
            gategroup to compare.

        Returns
        -------
        int
            Bitcode of gate overlap.

        """
        gateA, gateB = cls.as_gategroup(gateA), cls.as_gategroup(gateB)
        # all gates in common
        gates = sorted(set(chain(gateA.gates, gateB.gates)), key=_gate_sort)
        # location mask for gateA and gateB
        locA = tuple(gate in gateA.gates for gate in gates)
        locB = tuple(gate in gateB.gates for gate in gates)
        codes = {(i, j): GateDefinition.gate_compare(ga, gb)
                 for (i, ga), (j, gb) in permutations(enumerate(gates), 2)}
        ovlp = 0
        # iterate through all combinations of gates (positions in truthtable)
        for loc in product(*(range(2) for _ in range(len(gates)))):
            if any(not (1 << (loc[i]+2*loc[j])) & code for (i, j), code in codes.items()):
                continue
            # extract value of gate at given position in truthtable
            tA = gateA.truthtable[tuple(
                l for i, l in enumerate(loc) if locA[i])]
            tB = gateB.truthtable[tuple(
                l for i, l in enumerate(loc) if locB[i])]
            ovlp |= 1 << (tA+tB*2)
        return ovlp

    def __contains__(self, key):
        if isinstance(key, str):
            return super().__contains__(key)
        elif isinstance(key, Gate):
            key = GateGroup.as_gategroup(key)
        return not (GateGroup.overlap(key, self) & 0b0010)

    @classmethod
    def as_gategroup(cls, gate:Union[Gate_,"GateGroup"])->"GateGroup":
        """
        Function to ensure :class:`Gate` or :class:`GateGroup` is a :class:`GateGroup`

        Parameters
        ----------
        gate : Gate|GateGroup
            Input to ensure is a :class:`GateGroup`.

        Raises
        ------
        TypeError
            Input cannot be converted into :class:`GateGroup`.

        Returns
        -------
        GateGroup
            Input rendered as a :class:`Gategroup`.

        """
        if isinstance(gate, Gate_):
            return cls(_TT_ft, gate)
        if not isinstance(gate, cls):
            raise TypeError(
                f"cannot convert {type(gate).__name__} to GateGroup")
        return gate

    def get_gategroup_descr(self, indent:int=0, include_param:bool=False, include_source:bool=True)->str:
        """
        Produce YAML-like description of GateGroup, does not include GateGroup label.

        Parameters
        ----------
        indent : int, optional
            Indent level. Number of spaces to append after each new line. The default is 0.
        include_param : bool, optional
            Whether to include description of the :attr:`Column.param` of each
            Column in the Gates. The default is False.
        include_source : bool, optional
            Whether to include description of the :attr:`Column.source_param` 
            of mapped Column. The default is True.

        Returns
        -------
        str
            YAML-like definition of GateGroup.

        
        """
        out = 'truthtable: ' + _indent(str(self.truthtable), 12).lstrip()
        out += '\nGates:\n'
        out += _indent('- '+ '\n- '.join(gate.get_gate_description(2, include_param, include_source).lstrip()
                                         for gate in self.gates), 2)
        return _indent(out, indent)

    def get_description(self, indent:int=0, include_param:bool=False, include_source:bool=True)->str:
        """
        Produce YAML-like description of GateGroup, includes GateGroup label.

        Parameters
        ----------
        indent : int, optional
            Indent level. Number of spaces to append after each new line. The default is 0.
        include_param : bool, optional
            Whether to include description of the :attr:`Column.param` of each
            Column in the Gates. The default is False.
        include_source : bool, optional
            Whether to include description of the :attr:`Column.source_param` 
            of mapped Column. The default is True.

        Returns
        -------
        str
            YAML-like definition of GateGroup.

        """
        out = f'GateGroup:\n{_indent(self.get_gategroup_descr(0, include_param, include_source),2)}'
        return _indent(out, indent)
    
    @property
    def description(self)->str:
        """YAML-like definition of GateGroup."""
        return  self.get_description() + f'\n{_indent(self.param.description, 2)}'
    
    @property
    def _sort_tuple(self)->tuple[int,tuple[int,...],tuple[tuple[Hashable,...]]]:
        """Tuple that can unambiguously be used to sort against other like objects"""
        out = [self.truthtable.ndim, _make_sortable(self.truthtable)]
        out += [g._sort_tuple for g in self.gates]
        return tuple(out)


register_type(GateGroup)
GG_all = GateGroup(_TT_all) #: General gate that includes
GG_none = GateGroup(_TT_none) #: General gate that excludes all rows
TV_GateGroup = TV_ImData(subclass=GateGroup)


def _proc_TV_mappedgateparam(dct:"Gate", kwarg_append:dict)->dict:
    """Pre-process function for mappedgate params attribute, sets key and type definitions"""
    order = tuple(dct['gatedef']['params'].keys())
    out = dict(order=order, required=order, typedefs=dct['gatedef']['params'],
                defaults=dct['defaults'] if 'defaults' in dct else {}, imdata=dct)
    return out


def _proc_TV_mappedgate_mask_gate(imdata:"MappedGate", kwarg_append:dict)->dict:
    """Pre-process function for mappedgate mask_gate function. Give check func source_gate"""
    return dict(source_gate=imdata['source_gate'])


def _check_mappedgate_mask_gate(val:"GateGroup", source_gate:"GateGroup"=None, **kwargs)->"GateGroup":
    """Check func for mappedgate mask_gate function, checks origin_params match between source_gate and mask_gate"""
    val = source_gate if val is None else GateGroup.as_gategroup(val)
    if val.origin_param != source_gate.origin_param:
        raise ValueError("Origin params of source_gate and mask_gate to not match")
    return val & source_gate


class MappedGate(Gate_, _ImData):
    """
    Special subclass of :class:`Gate_` that re-arranges valuse mask_gate.
    MappedGates are always non-atomic.
    Used when rows are inter-related to one another based on their order.
    
    Internally, when creating gate mask the following call is made 
    (``mg`` is a) MappedGate, ``data`` is a :class:`DataSet`::
        
        mg.gatedef.func(data.get_column(mg.mask_gate, relative=mg.source_gate), *mg.params)
    
    Parameters
    ----------
    gatedef : MappedGateDef
        Defines used to remap mask_gate.
    params : tupledict
        Params handed as kwargs to ``gatedef.func`` to remap mask_gate.
    source_gate : GateGroup
        :class:`GateGroup` defining the row-mask to re-arrange. This is the
        ``base_gate`` of the Mapped Gate.
    mask_gate : GateGroup
        Gate defining row-value mask to re-arrange. Values of this gate are
        masked by source_gate, and the gatedef function re-arranges that boolean array.
    expand : bool
        expand value of mask, the value columns not within source_gate automatically
        receive.
    
    """
    __slots__ = ('gatedef', 'params', 'source_gate', 'mask_gate', 'expand', 'title')
    _typeconversions = ImDict(gatedef=TV_MappedGateDef, 
                              params=TV_tupledict(data_proc=_proc_TV_mappedgateparam), 
                              source_gate=TV_GateGroup, 
                              mask_gate=TV_GateGroup(data_proc=_proc_TV_mappedgate_mask_gate,
                                                      validator=_check_mappedgate_mask_gate), 
                              expand=TV_bool, title=TV_str)
    _required = frozenset({'gatedef', 'params', 'source_gate', 'mask_gate'})
    _defaults = ImDict(expand=False)
    _hashskip = frozenset({'title', })
    
    #: Mapping function definition
    gatedef: MappedGateDef
    #: keyword arguments hannded to :attr:`MappedGate.gatedef.func` when creating gate
    params: tupledict
    #: Gate masking origin_param, :attr:`MappedGate.gatedef.func` will be handed
    #: a bolean mask the size of this gate.
    source_gate: "GateGroup"
    #: Gate used to generate boolean mask
    mask_gate: "GateGroup"
    #: Value taken by mask for rows outside of :attr:`MappedGate.source_gate`
    expand : bool
    #: (mutable) Human readable title for MappedGate
    title : str
    
    def __post_init__(self):
        self.gatedef.verify(self.params)

    @property
    def atomic(self)->bool:
        """Gate is dependant on selection of columns"""
        return False

    @property
    def base_param(self)->Param:
        """
        :class:`Param` the defines the rows of the table which are handed to 
        :attr:`MappedGate.func` to evalute mask.
        """
        return self.source_gate.base_param

    @property
    def base_gate(self)->"GateGroup":
        """
        The :class:`GateGroup` the defines the rows of the table which are handed 
        to :attr:`Gate.func` to evalute mask.
        """
        return self.source_gate

    @property
    def parent_param(self)->Param:
        """:class:`Param` of :attr:`Gate.parent_gate`, the gate used to compute
        gate of :attr:`Gate.base_gate`.
        """
        return self.source_gate.parent_param

    @property
    def parent_gate(self)->"GateGroup":
        """
        :class:`GateGroup` used to compute columns of :attr:`base_gate`
        
        .. note::
            
            This is nearly always :attr:`Gate.origin_gate`, only in the case
            of multiply nested non-atomic gates is this untrue
        
        
        """
        return self.source_gate.parent_gate

    @property
    def origin_param(self)->Param:
        """
        :class:`Param` defining the table with no gates whatsoever.
        """
        return self.source_gate.origin_param

    def get_gate_description(self, indent:int=0, include_param:bool=False, include_source:bool=True)->str:
        out = (f'MappedGate: {self.gatedef.name}\nParams:\n{_indent(self.get_param_descr(),2)}\n' +
               f'Source Gate:\n{self.source_gate.get_description(2, include_param, include_source)}\n'+
               f'Mask Gate:\n{self.mask_gate.get_description(2, include_param, include_source)}')
        return _indent(out, indent)
    
    @property
    def description(self)->str:
        """Description of mapped gate"""
        return self.get_gate_description(include_param=True, include_source=True)
    
    @property
    def _sort_tuple(self):
        """Tuple that can unambiguously be used to sort against other like objects"""
        out = [False, "MappedGate", self.gatedef.name, _make_sortable(self.params)]
        out += [self.source_gate._sort_tuple, self.mask_gate._sort_tuple]
        return tuple(out)


###############################################################################
# Define dynamic data structures
###############################################################################
def _gate_expand(sub:np.ndarray, mask:np.ndarray[np.bool_], fill:np.number)->np.ndarray[np.uint8]:
    """
    Used for creating arrays for columns with base_gates larger than parent gate
    Expand array sub into mask, filling mask == False values with fill
    """
    if isinstance(mask, slice):
        return sub[mask]
    if sub.dtype != np.object_:
        out = np.ones(mask.shape, dtype=sub.dtype)*fill
    else:
        out = np.empty(mask.shape, dtype=sub.dtype)
        for i in range(mask.size):
            out[i] = fill
    out[mask] = sub
    return out


@dataclass(frozen=True)
class CacheID:
    """
    Dataclass for storing a category of array in :attr:`DataSet._array_cache`
    """
    tp: type
    params:tupledict
    name: str


class DataSet:
    """
    Semi-abstract base class designed to hold the underlying data from which 
    :class:`Table` classes compute their columns and values.
    
    Parameters
    ----------
    group : None | tb.Group | Callable[[],tb.Group]
        HDF5 group in whic to save data, if None, then data is not save on disk.
    autosave : bool, optional
        When group is created, whether to save data as soon as computed.
        The default is False.
    meta : dict, optional
        Additional metadata on object, generally subtype specific. 
        The defualt is None.
    track : bool, optional
        Whether to track file, if tracked then close file when DataSet object
        is deleted.
    file :tb.File, optional
        File in which to save data. The default is None.
    group_no : int | bool, optional
        If True, then group is created within group with name "groupname", 
        if number, then data saved in subgroup with name "groupname[group_no]"
        if False, data saved directly in group. The default is 1.
    **kwargs : Any
        Additinal arguments passed to __init_data__ (subclass dependant)
    """
    _group_name:ClassVar[str] = 'dataset'

    _tables: dict[Param,"Table"]
    # : dictionary of codes for (relatitive GateGroup, Gate) : map
    _gates: dict[Gate,tuple[GateGroup, np.ndarray[np.bool_]]]
    # : nested dictionary of masks {requested:{relative:mask}} for gategroups
    _gategroups: dict[GateGroup,dict[GateGroup,np.ndarray[np.bool_]]]
    _group: _GroupFuture  # location to store HDF5 data, None otherwise
    _meta: DiskDict # disk dict for storing additional metadata on the object.
    _array_cache : weakref.WeakValueDictionary #: cache for storing arrays from tables that might be reused
    _autosave: bool #: interanal private variable storing autosave state
    _finalizers: WVD #: WeakValueDictionary of all files tracking files

    def __init__(self, group:GroupFuture=None, autosave:bool=False, meta:dict=None, 
                 track:bool=True, file:tb.File=None, group_no:int|bool=1, **kwargs):
        self._autosave = bool(autosave)
        self._track = bool(track)
        if file is not None and not isinstance(file, tb.File):
            raise TypeError(f"file must be None or tables.File, got {type(file).__name__}")
        self._file = file
        group = group if isinstance(group, _GroupFuture) else _GroupFuture(group, self, self._track_callback)
        if group_no is not False:
            groupname = type(self)._group_name
            if group_no is not True:
                groupname += str(group_no)
            self._group = group._create_groupfuture(groupname)
            if 'dataID' in self._group:
                self._dataID = self._group._group.dataID.read().decode()
        self._finalizers = WVD()
        self._tables = dict()
        self._gates = dict()
        self._gategroups = dict()
        self._meta = DiskDict(meta, self._get_meta_group, self.autosave)
        self._array_cache = weakref.WeakValueDictionary()
        self.__init_data__(**kwargs)

    # like abstract method, does nothing by default
    def __init_data__(**kwargs)->None:
        r"""
        Method for subclasses of DataSet to re-implement for initiating any
        data at end of object instantiation. init passes data argument, and
        all \*'*kwargs to _init_data_
        """
        pass

    def _calc_dataID(self)->bytes:
        """Compute idenifier for data- so loading from HDF5 can discriminate different data sources"""
        return b'\x00'

    def _get_dataID(self)->bytes:
        """Return already computed dataID if exists, or compute and save"""
        if not hasattr(self, '_dataID'):
            self._dataID = self._calc_dataID()
        return self._dataID

    @property
    def dataID(self)->str:
        """Hash-like property, should be implemented for subclasses"""
        return self._get_dataID().hex()

    def _add_to_cache(self, tp:type, param:tupledict, name:str, arr:np.ndarray)->None:
        """
        Add an array to the cache. Must identify the type of table createing,
        the relevant params (omit any param keys that do not change values of array),
        and a name for the array.

        Parameters
        ----------
        tp : type
            Table-type that cache is linked to.
        param : tupledict
            Relevant params to identify Param of table from which arrises
            May omit values from the normal Param tupledict if they do not
            influence .
        name : str
            String description of array.
        arr : np.ndarray
            Array to store.

        """
        self._array_cache[CacheID(tp, param, name)] = arr

    def _get_from_cache(self, tp:type, param:tupledict, name:str, getter:Callable[[],np.ndarray]=None)->np.ndarray:
        """
        Retrieve (potentially) cached array.
        Provide the type, params and name to get a key from the cahce. getter
        may also be supplied, if key is not in cache, then getter is called to
        create array, save in cache and return

        Parameters
        ----------
        tp : type
            Table-type that cache is linked to.
        param : tupledict
            Relevant params to identify Param of table from which arrises
            May omit values from the normal Param tupledict if they do not
            influence .
        name : str
            String description of array.
        getter : Callable[[],np.ndarray], optional
            Callable to generate array, takes no arguments. The default is None.

        Returns
        -------
        np.ndarray
            Array either retrieved from cache or computed with getter.

        """
        cacheid = CacheID(tp, param, name)
        if cacheid in self._array_cache:
            return self._array_cache[cacheid]
        if getter is not None:
            out = getter()
            self._array_cache[cacheid] = out
            return out
        return None

    def _remove_from_cache(self, tp:type, param:tupledict, name:str)->np.ndarray:
        """
        Retrieve (potentially) cached array, and remove.
        Provide the type, params and name to get a key from the cahce. getter
        may also be supplied, if key is not in cache, then getter is called to
        create array, save in cache and return


        Parameters
        ----------
        tp : type
            Table-type that cache is linked to.
        param : tupledict
            Relevant params to identify Param of table from which arrises
            May omit values from the normal Param tupledict if they do not
            influence .
        name : str
            String description of array.
        
        Returns
        -------
        None|np.ndarray
            None if array is not in cache, otherwise array from cache.

        """
        return self._array_cache.pop(CacheID(tp, param, name), None)

    def _get_cache_type(self, tp:type, param:tupledict=None)->dict[str:np.ndarray]:
        """
        Retrieve all array as {name:array} dictionary that match both tp and param.

        Parameters
        ----------
        tp : type
            Table-type that cache is linked to.
        param : tupledict
            params of interest, must match how array was stored.
        
        Returns
        -------
        dict[str:np.ndarray]
            Dictionary of matching arrays, keys are the names of each array.

        """
        return WVD({k:v for k, v in self._array_cache.items() 
                    if issubclass(k.tp, tp) and param is None 
                    or all(k in k.params and k.params[k] == v for k, v in param.items())})

    @property
    def file(self)->None|tb.File:
        """File where data is saved"""
        return self._file if self._group._created else None

    @property
    def autosave(self):
        """If tables are automatically save to HDF5 file"""
        return self._autosave

    @autosave.setter
    def autosave(self, val:bool):
        self._autosave = bool(val)

    def _get_autosave(self)->bool:
        """Callable version for use with DiskDict autosave"""
        return self._autosave

    def _track_callback(self, group:_GroupFuture)->None:
        """Callback function for Table's groups"""
        file = group._file
        if self._file is not None and file != self._file:
            warnings.warn("group where data saved is from different file than expected")
            if self._file in self._finalizers:
                self._finalizers[self._file].finalize_owner(weakref.ref(self))
        self._file = file
        if self._track:
            self._finalizers[file] = _FileFinalizer(file, self)

    def _get_meta_group(self)->tb.Group:
        """
        Create the group for metadata of self.

        Returns
        -------
        tb.Group
            Group for storing metadata.

        """
        return self._group._create_groupfuture('meta')

    def _param_table_name(self, param:Param)->str:
        """
        Generate name for node in HDF5 file for storing a :class:`Table` from :class:`Param`
        """
        module = param.tp.__module__.replace('.', '_')
        return f"TABLE__{module}_{param.tp.__name__}"

    def _get_param_name(self, param:Param, group:GroupFuture=None)->str:
        """Get name of group for param, if _group not created, returns the first expected name"""
        group = self._group if group is None else group
        group = group if isinstance(group, _GroupFuture) else _GroupFuture(group)
        name = self._param_table_name(param)
        i = 0
        if not group._created:
            return f'{name}_{i}'
        while (groupname := f'{name}_{i}') in group:
            if TypeValidator.read_any(group[f'{groupname}/param']) == param:
                return groupname
            i += 1
        return groupname

    def check_param_saved(self, param:Param)->bool:
        """
        Check if param is saved in HDF5 file.
        
        Parameters
        ----------
        param : Param
            The parameter to check if saved.
        
        Returns
        -------
        bool
            :code:`True` if param saved, :code:`False` if not.
        
        """
        return self._get_param_name(param) in self._group

    def _get_group_from_param(self, param:Param, group:GroupFuture=None)->_GroupFuture:
        """Generate a tb.Group in which to save the given param"""
        group = self._group if group is None else group
        fgroup = group if isinstance(group, _GroupFuture) else _GroupFuture(group)
        if not fgroup._creatable:
            return _GroupFuture(None)
        if fgroup._created:
            if (groupname := self._get_param_name(param, fgroup)) in fgroup:
                return _GroupFuture.create_dependant(group[groupname], fgroup)
        def create_group()->tb.Group:
            groupname = self._get_param_name(param, fgroup)
            if fgroup in group:
                return fgroup[groupname]
            out = fgroup._create_group(groupname)
            param.write_group(out, "param")
            return out
        out = _GroupFuture.create_dependant(create_group, fgroup)
        def group_callback(grp:_GroupFuture)->None:
            if self.check_param_saved(param):
                out._create()
        fgroup._add_callback(group_callback)
        return out

    def load_table(self, group:tb.Group, overwrite:bool=False)->"Table":
        """
        Load a group storing the saved values for a :class:`Table`.

        Parameters
        ----------
        group : tb.Group
            Group from which to load table values.
        overwrite : bool, optional
            Whether to overwrite table instance if already computed not from HDF5 file.
            The default is False.

        Raises
        ------
        ValueError
            Table already calculated, no need to load.

        Returns
        -------
        Table
            :class:`Table` object represented by ``group``.

        """
        table = Table.load_group(group, self)
        if table.param in self._tables and not overwrite:
            raise ValueError("table already exists, cannot reload")
        self._tables[table.param] = table
        return table

    def get_table(self, param:Param, gate:GateGroup=None)->"Table":
        """
        Return the table specified by the param, will create the table if not
        already created.
        
        Parameters
        ----------
        param : Param
            :class:`Param` definining desired :class:`Table`
        gate : GateGroup, optional
            :class:`GateGroup` to set rows of table to return. 
            Uses :meth:`Param.regate`. If `None` use gate of `param`.
            The default is None.
        
        Returns
        -------
        Table
            The requested table based on ``param``.
        """
        if gate is not None:
            param = param.regate(gate)
        if param in self._tables:
            return self._tables[param]
        self._tables[param] = param.tp(param, self)
        return self._tables[param]
    
    def get_column(self, column:Column, gate:GateGroup=None)->np.ndarray:
        """
        Return the specified column from the table

        Parameters
        ----------
        column : Column
            :class:`Column` to retrive from self.
        gate : GateGroup, optional
            :class:`GateGroup` to set rows of table to return. 
            Uses :meth:`Column.regate`. If `None` use gate of `column`.
            The default is None.
        
        Returns
        -------
        np.ndarray
            Array of column values (rows).

        """
        if gate is not None:
            column = column.regate(gate)
        table = self.get_table(column.source_param)
        out = table[column._get_func_args]
        if 'gategroup' in column and column.gategroup != column.parent_gate:
            mask = self._get_gategroup_mask(column.gategroup, column.parent_gate)
            if column.gategroup - column.parent_gate:
                emask = self._get_gategroup_mask(column.parent_gate, column.gategroup)
                eout = np.ones(emask.shape, dtype=out.dtype)*column.fill
                eout[emask] = out[mask]
                out = eout
            else:
                out = out[mask]
        return out
    
    def iter_column(self, column:Column, gate:GateGroup=None)->Iterator:
        """
        Iterate over values in specified Column

        Parameters
        ----------
        column : Column
            :class:`Column` to retrive from self.

        Yields
        ------
        Any
            Column value, iterator yields one row at a time.

        """
        if gate is not None:
            column = column.regate(gate)
        yield from self.get_table(column.source_param).iter_column(column)
    
    def record_column(self, column:Column)->np.ndarray:
        """
        Save specified column in memory.

        Parameters
        ----------
        column : Column
            Column to be saved.

        Returns
        -------
        np.ndarray
            Values of column.

        """
        return self.get_table(column.param).record_column(column)

    def _get_table_rep(self, param:Param)->Union["Table",Callable[[],"Table"]]:
        """
        Returns either the table specified by the param, **IF** it has already
        been computed, otherwise return a lambda function that when called,
        will create the table
        """
        if param in self._tables:
            return self._tables[param]
        return lambda: self.get_table(param)

    def has_table(self, param:Param)->bool:
        """
        Determine whether or not the table has been computed.
        
        Parameters
        ----------
        param : Param
            :class:`Param` defining a :class:`Table` to see if it has already
            been created in self.
        
        Returns
        -------
        bool
            Whether or not :class:`Table` for ``param`` has been calculated.
        """
        return param in self._tables
    
    def has_table_saved(self, param:Param)->bool:
        return self.has_table(param) or self.check_param_saved(param)

    def _calc_gate(self, gate:Gate, relative:GateGroup)->np.ndarray[np.uint8]:
        """
        Calculate a gate relative to ``relative`` 
        (``relative`` is usually GG_all, but still must specify)

        .. note::

            This is "dumb" function, it does not check that supergate is valid.


        .. note:: 

            this method returns a uint8 so can be used for indexing into truthtable
        """
        return gate.func(*(self.get_column(col.regate(relative)) 
                           for col in gate.columns), **gate.params).astype(np.uint8)

    def _calc_mappedgate(self, mapgate:MappedGate, relative:GateGroup)->np.ndarray[np.uint8]:
        """
        Calculate a mappedgate relative to ``relative`` 
        (``relative`` is usually GG_all, but still must specify)

        .. note::

            This is "dumb" function, it does not check that supergate is valid.


        .. note:: 

            this method returns a uint8 so can be used for indexing into truthtable
        """
        mask = mapgate.func(self._get_gategroup_mask(mapgate.mask_gate, mapgate.source_gate)).astype(np.uint8)
        if relative == mapgate.source_gate:
            return mask
        mask = mask[self._get_gategroup_mask(relative, mapgate.source_gate)]
        if not GateGroup.overlap(relative, mapgate.source_gate) & 0b0010:
            return mask
        return _gate_expand(mask, self._get_gategroup_mask(mapgate.source_gate, relative), mapgate.expand)

    def _get_gate(self, gate:Gate_, relative:GateGroup)->np.ndarray[np.uint8]:
        """
        Retrieve gate relative to supergate, if gate exists relative to larger
        gate in _gates, then retrieve and mask, otherwise call _calc_gate
        """
        # Structure of _gate_gate:
        # 1. check if gate has already been calculated
        #     a. if relative is within the scope calucated for _gates, return properly masked
        #     b. special case for non-atomic: if computed for parent_gate, then expand and return
        #     c. if neither a nor b are satisfied, proceed to calculate entire gate in part 2
        # 2. Compute gate and store in _gates
        #     a. if atomic, calculate according to relative
        #     b. if non-atomic, calcualte by which is smaller: relative or parent_gate
        #     c. return, if non-atomic expanding appropriately
        if isinstance(gate, MappedGate):
            return self._calc_mappedgate(gate, relative)
        parent_gate = gate.parent_gate
        gate_ef = gate._expand_false
        if gate_ef in self._gates:
            rel, out = self._gates[gate_ef]
            # case of relative is within rel, so just mask and return
            if not GateGroup.overlap(relative, rel) & 0b0010:
                return out[self._get_gategroup_mask(relative, rel)]
            # case of non-atomic where gate is already computed for parent_gate, so must expand
            if not gate.atomic and rel == parent_gate:
                return _gate_expand(out[self._get_gategroup_mask(relative, rel)], 
                                    self._get_gategroup_mask(rel, relative), 
                                    np.uint8(gate.expand))
        # case of atomic gate
        if gate.atomic:
            self._gates[gate] = (relative, self._calc_gate(gate, relative))
            return self._gates[gate][1]
        # case of non-atomic where relative is inside parent_gate (do not need to expand)
        if not GateGroup.overlap(relative, parent_gate) & 0b0010:
            self._gates[gate_ef] = (relative, self._calc_gate(gate._expand_false, relative))
            return self._gates[gate_ef][1]
        # final case of parent gate is outside of parent_gate, so must expand
        self._gates[gate_ef] = (parent_gate, self._calc_gate(gate._expand_false, parent_gate))
        return _gate_expand(self._gates[gate_ef][1], self._get_gategroup_mask(parent_gate, relative),
                            np.uint8(gate.expand))

    def _clear_gates(self, origin_param:Param)->None:
        """Remove all gates from _gates cache that have origin_param matching origin_param"""
        check = tuple(gt for gt in self._gates.keys() if gt.origin_param == origin_param)
        for gt in check:
            self._gates.pop(gt)

    def _retrieve_gategroup_mask(self, gategroup:GateGroup, relative:GateGroup)->np.ndarray[np.bool_]:
        """
        If possible, return mask for gategroup relative to relative. If not
        possible without calculating a gate, return the gategroup and relative
        GateGroups that must be calculated to enable said mask to be calculated
        """
        comp = (gategroup, relative)
        if gategroup in self._gategroups:
            if relative in self._gategroups[gategroup]:
                return self._gategroups[gategroup][relative]
            for g in self._gategroups[gategroup].keys():
                if GateGroup.overlap(relative, g) & 0b0010:
                    continue
                relmask = self._retrieve_gategroup_mask(relative, g)
                if isinstance(relmask, tuple):
                    comp = (relative, g)
                    continue
                out = self._gategroups[gategroup][g][relmask]
                return out
        return comp
    
    def _compute_gategroup_mask(self, gategroup, relative)->np.ndarray[np.bool_]:
        """Compute a gategroup's mask relative to relative from underlying gates masks"""
        return gategroup.truthtable[tuple(self._get_gate(gate, relative) 
                                          for gate in gategroup.gates)]

    def _get_gategroup_mask(self, gategroup:GateGroup, relative:GateGroup,
                            save:bool=True)->slice|np.ndarray[np.bool_]:
        r"""
        Get mask for gategroup of a column already masked by relative

        Parameters
        ----------
        gategroup : GateGroup
            The GateGroup for which to evaluate the values of the returned mask.
        relative : str|GateGroup, optional
            The GateGroup that serves as the "origin" of the mask, ie the GateGroup
            of the array being masked. The default is 'parent'.

        Returns
        -------
        slice|np.ndarray[np.bool\_]
            Mask or slice which will return gategroup when indexing a column gated
            by relative

        """
        # catch cases where array need not be changed, so return slice
        if gategroup.nogate:
            return slice(None) if gategroup.truthtable else slice(0, 0)
        if gategroup == relative:
            return slice(None)
        # check if gategroup is specified
        out = self._retrieve_gategroup_mask(gategroup, relative)
        if isinstance(out, np.ndarray):
            if save:
                _nested_set(self._gategroups, (gategroup, relative), out)
            return out
        _nested_set(self._gategroups, out, self._compute_gategroup_mask(*out))
        mask = self._retrieve_gategroup_mask(gategroup, relative)
        if save:
            _nested_set(self._gategroups, (gategroup, relative), mask)
        else:
            _nested_pop(self._gategroups, out)
        return mask

    def _trim_gategroups(self)->None:
        """
        Remove all gategroup masks from cache not currently associated with
        a table
        """
        tbmask = tuple(id(table._mask) for table in self._tables.values() if table._derived)
        for reldict in self._gategroups.values():
            remove = tuple(rel for rel, m in reldict.items() if id(m) not in tbmask)
            for rel in remove:
                reldict.pop(rel)
        gategroups = tuple(gg for gg in self._gategroups.keys())
        for gg in gategroups:
            if len(self._gategroups[gg]) == 0:
                self._gategroups.pop(gg)
    
    def _rebase_tables(self, gate:GateGroup, base_param:Param, origin_param:Param)->None:
        """
        Rebase all tables with matching origin_param to gate, 
        base_param is base_param of gate
        """
        table = self.get_table(base_param)
        table.convert_to_base_table()
        # rebase tables, start by making tuple so keys can be poped from dictionary
        tables = tuple(param for param in self._tables.keys())
        for param in tables:
            # keep tables belonging to params with different origin_param
            if param.origin_param != origin_param or param == base_param:
                continue
            table = self._tables[param]
            if not param.base_gate.atomic and GateGroup.overlap(param.base_gate.parent_gate_full, gate) & 0b0010:
                if self._tables[param]._cache._group.created:
                    self._tables[param]._cache.clear_memory()
                else:
                    self._tables.pop(param)
                continue
            intersect = param.base_gate & gate
            if intersect == param.base_gate:
                table._rebase(gate)
            elif param.base_gate != gate:
                new_param = param.regate(intersect)
                table = self.get_table(new_param)
                table._rebase(gate)
                self._tables.pop(param)

    def _trim_relgate(self, relgate:GateGroup)->None:
        """
        Remove all masks in self._gategroups that are relative to relgate.
        Used to remove masks after rebasing to another gate.
        """
        for gg in self._gategroups.keys():
            _nested_pop(self._gategroups, (gg, relgate))
    
    def _rebase_gates(self, gate:GateGroup, origin_param:Param)->None:
        """
        Internal function for rebasing all masks in self._gategroup with matching 
        origin_param to gate
        """
        # create new versions of gates to rebase
        new_gates, pop_gates = dict(), list()
        for gt, (rel, _) in self._gates.items():
            if rel.origin_param != origin_param:
                continue
            ovlp =  GateGroup.overlap(rel, gate) & 0b0010
            if ovlp == 0b0110:
                pop_gates.append(rel)
                continue
            elif not ovlp & 0b0010:
                continue
            new_rel = rel&gate
            new_gates[gt] = (new_rel, self._get_gate(gt, new_rel))
        # get counts of relative gates and gategroups that are based on origin_param
        relcntr, gategroups = Counter(), list()
        for gategroup, reldict in self._gategroups.items():
            if gategroup.origin_param != origin_param:
                continue
            gategroups.append(gategroup)
            relcntr.update(reldict.keys())
        # Trim/rebase gategroups
        for i, gategroup in enumerate(gategroups):
            reldct = self._gategroups[gategroup]
            rels = list(reldct.keys())
            # check each relgate and set new, trimmed version, if possible
            for rel in rels:
                if not GateGroup.overlap(rel, gate) & 0b0010:
                    continue
                new_rel = rel&gate
                mask = self._retrieve_gategroup_mask(gategroup, new_rel)
                if isinstance(mask, np.ndarray):
                    _nested_set(self._gategroups, (gategroup, new_rel), mask)
                relcntr.subtract((rel, )) # reduce counter of references to rel in _gategroups
                # no more uses of relative gate, remove from all dictionaries
                if relcntr[rel] == 0:
                    self._trim_relgate(rel)
        # trim _gates now that all gategroups have been assesed and trimmed
        for gt in pop_gates:
            self._gates.pop(gt)
        self._gates.update(new_gates)
        
    def rebase(self, gate:GateGroup, clear_gates:bool=False)->None:
        """
        Rebase tables to gate. This is used to reduce memory usage. Any stored
        :class:`Table` with a gate larger than ``gate`` is removed.

        Parameters
        ----------
        gate : GateGroup
            The new "base" gate of the data.
        
        """
        base_param, origin_param = gate.base_param, gate.origin_param
        self._rebase_tables(gate, base_param, origin_param)
        if clear_gates:
            self._trim_gategroups()
            self._clear_gates(origin_param)
        else:
            self._rebase_gates(gate, origin_param)

    def _get_gate_size(self, gate:GateGroup)->int:
        """
        Determine size of gate, **if possible from already determined gates**
        THIS METHOD IS NOT COMPLETE, USE LATER.
        """
        # this is beta method
        if gate in self._gategroups:
            for rel, mask in self._gategroups[gate].items():
                if gate in rel:
                    return mask.sum()
        return None
    
    def clear_memory(self)->None:
        """
        Clear all cache dictionaries. 
        This frees memeory, but all computations must be performed again.
        :meth:`PhotonData.rebase` is usually more useful, as choice of rebase
        gate allows selecting which arrays to keep, preventing recomputation of
        columns with gates within the specified gate.
        """
        self._tables = dict()
        self._gates = dict()
        self._gategroups = dict()

    def get_gategroup(self, gategroup:GateGroup, relative:str|GateGroup='parent')->np.ndarray[np.bool_]:
        r"""
        Retrieve the boolean array defining the columns of ``gategroup`` relative
        to ``relative``. Relative can is either a :class:`GateGroup`, ``"all"``
        or ``"parent"``. If ``"all"`` treat relative like an all gate, if 
        ``"parent"`` relative is ``gategroup.parent_gate``.

        Parameters
        ----------
        gategroup : GateGroup
            :class:`GateGroup` to retrieve boolean mask.
        relative : str|GateGroup, optional
            Gate defining columns in mask. The default is 'parent'.

        Raises
        ------
        ValueError
            Bad definition of relative.
        TypeError
            Wrong type of either gategroup or relative.

        Returns
        -------
        np.ndarray[np.bool\_]
            Mask of columns in gategroup relative to columns in parent.

        """
        # preproces gategroup and relative so both are GateGroups
        gategroup = GateGroup.as_gategroup(gategroup)
        if isinstance(relative, (GateGroup, Gate_)):
            relative = GateGroup.as_gategroup(relative)
        elif isinstance(relative, (str, bytes)):
            relative = str(relative)
            if relative == 'parent':
                relative = gategroup.parent_gate
            elif relative == 'all':
                relative = GateGroup(truthtable=_TT_all,
                                     param=gategroup.origin_param)
            else:
                raise ValueError(
                    f"invalid relative option: '{relative}', must be 'parent', or 'all'")
        else:
            raise TypeError(
                f"relative must be GateGroup, 'parent', or 'all', got object of type {type(relative)}")
        # _get_gategroup_mask does actual processing
        out = self._get_gategroup_mask(gategroup, relative)
        return out

    def get(self, comp:Param|Column|Gate_|GateGroup, gate:str|Gate|GateGroup=None)->Union["Table",np.ndarray]:
        r"""
        Retrieve the :class:`Table`,array or mask corresponding to the given
        :class:`Param`, :class:`Column`, or :class:`GateGroup` respectively given
        in ``comp``.

        Parameters
        ----------
        comp : Param | Column | Gate\_ | GateGroup
            Specification of type of data to retrieve from dataset.
        gate : str | Gate\_ | GateGroup, optional
            :class:`GateGroup` to be applied to ``comp``, unless ``comp`` is a
            :class:`GateGroup` or :class:`Gate_`, in which case it serves as the
            ``relative`` argument to :meth:`DataSet.get_gategroup`, defining which
            gate to use as the base of the returned mask array. The default is None.

        Raises
        ------
        TypeError
            wrong type of input.

        Returns
        -------
        Table | np.ndarray
            :class:`Table` of :class:`Param`, or array of :class:`Column` or 
            :class:`GateGroup`.

        """
        if isinstance(comp, Gate_):
            comp = GateGroup.as_gategroup(comp)
        if isinstance(comp, (GateGroup)):
            return self.get_gategroup(comp, relative='parent' if gate is None else gate)
        if gate is not None:
            comp = comp.regate(gate)
        if isinstance(comp, Param):
            return self.get_table(comp)
        elif isinstance(comp, Column):
            return self.get_column(comp)
        raise TypeError('can only get Param, Column, Gate, or GateGroup objects')

    def get_frame_dict(self, *args:Column, names:Sequence[str]=None, gate:GateGroup=None,
                       include_unit:bool=False, index_name:bool=False)->dict[str,np.ndarray]:
        r"""
        Get a dictionary of the columns given in \*args that can be converted
        into a pandas dataframe.

        Parameters
        ----------
        *args : Column
            :class:`Column` s to include in table.
        names : Sequence[str], optional
            Column names (must be same length as args) to give to each column. 
            If None, use default names of columns supplied by :func:`Column.name` .
            The default is None.
        gate : GateGroup, optional
            Gate to apply to all columns, over-rides any gates in columns.
            If not specified, take intersect of gates of each column.
            The default is None.
        include_unit : bool, optional
            Whether to include unit in column names. The default is False.
        index_name bool, optional
            If True use names compatible with saving to csv file. 
            The default is False

        Raises
        ------
        ValueError
            Incompatible origin_param between columns- ie incompatible rows.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary of arrays, usuable to create pandas dataframe.

        """
        if gate is None:
            gate = args[0].base_gate
            for arg in args:
                try:
                    gate &= arg.base_gate
                except Exception as e:
                    raise ValueError("columns have different base_param, cannot generate data frame") from e
        names = tuple(None for _ in range(len(args))) if names is None else names
        cdict = dict()
        for i, (arg, name) in enumerate(zip(args, names)):
            if arg in args[:i]:
                warnings.warn(f"column {arg.col} {arg.keytup} specified twice, skipping second instance")
                continue
            if name is None:
                name = arg.index_name(include_unit, self) if index_name else arg.name(include_unit, self)
            if name in cdict:
                warnings.warn(f"renaming duplicate key {name}")
                n = 1
                while f'{name} {n}' in cdict:
                    n += 1
                name = f'{name} {n}'
            cdict[name] = self.get_column(arg.regate(gate))
        return cdict

    def get_frame(self, *args:Column, names:Sequence[str]=None, gate:GateGroup=None,
                  multi_index:bool=False, include_unit:bool=False, index_name:bool=False)->pd.DataFrame:
        r"""
        Create a pandas dataframe from the columns specified in \*args.

        Parameters
        ----------
        *args : Column
            :class:`Column` s to include in table.
        names : Sequence[str], optional
            Column names (must be same length as args) to give to each column. 
            If None, use default names of columns supplied by :func:`Column.name` .
            The default is None.
        gate : GateGroup, optional
            Gate to apply to all columns, over-rides any gates in columns.
            If not specified, take intersect of gates of each column.
            The default is None.
        multi_index : bool, optional
            When column rows are arrays, create mulit-index so each sub-value
            is given a separte row. The default is False.
        include_unit : bool, optional
            Whether to include unit in column names. The default is False.
        index_name bool, optional
            If True use names compatible with saving to csv file. 
            The default is False

        Raises
        ------
        ValueError
            One or more columns incmopatible with multi-index.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame of the selected columns.

        """
        cdict = self.get_frame_dict(*args, names=names, gate=gate, 
                                    include_unit=include_unit, 
                                    index_name=index_name)
        if multi_index:
            cnew = dict()
            for key, col in cdict.items():
                try:
                    cnew[key] = np.concatenate(col)
                except Exception as e:
                    raise ValueError(f"column {key} is not a column of arrays, cannot multi-index") from e
                if len(cnew) == 1:
                    idx = tuple(chain.from_iterable(((i,j) for j in range(arr.size)) for i, arr in enumerate(col)))
                    mi = pd.MultiIndex.from_tuples(idx)
        return pd.DataFrame(cdict, index=mi) if multi_index else pd.DataFrame(cdict)

    def to_csv(self, *args:Column, path_or_buf:None|str|os.PathLike=None, names:Sequence[str]=None, 
               gate:GateGroup=None, include_unit:bool=False, multi_index:bool=False, **kwargs)->str|None:
        r"""
        Create a CSV of :class:`Column` specified in \*args. If the first
        argument is not a column, treat as argument to ``path_or_buf``
        
        >>> data.to_csv(col1, col2, path_or_buf='save.csv')
        
        is equivalent to
        
        >>> data.to_csv('save.csv', col1, col2)
        

        Parameters
        ----------
        *args : Column
            Columns to save in csv file.
        path_or_buf : None|str|os.PathLike, optional
            File or buffer to write to. If None, and not specified as first
            arg, return str of csv. Internally, this uses
            `pd.DataFrame.csv <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html>`_ . 
            The default is None.
        names : Sequence[str], optional
            Sequence of names to give to each column, if specified must be same
            length as number of columns specified. If None/not specified, will
            use index_name of each column. The default is None.
        gate : GateGroup, optional
            GateGroup to apply to all input columns. The default is None.
        include_unit : bool, optional
            Include unit string in column heading. The default is False.
        multi_index : bool, optional
            Used for columns whose rows are arrays. Will cause arrays to be "flattened"
            with index specified as multi-index. This is primarily for 'ph\_...' 
            columns. If True, the size of each row must be consistent between columns. 
            The default is False.
        **kwargs : Any
            Additional keyword arguments passed to 
            `pd.DataFrame.csv <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html>`_ .

        Raises
        ------
        TypeError
            Incompatible input.

        Returns
        -------
        str|None
            If path_or_buf is None, returns string representation of csv, otherwise
            None, and csv is written to path_or_buf.

        """
        if not args:
            raise TypeError("must specify at least one column")
        if not isinstance(args[0], Column):
            if path_or_buf is not None:
                raise TypeError("path_or_buf specified as arg and kwarg")
            path_or_buf, args = args[0], args[1:]
        if not args:
            raise TypeError("must specify at least one column")
        df = self.get_frame(*args, names=names, gate=gate, include_unit=include_unit,
                            multi_index=multi_index, index_name=True)
        return df.to_csv(path_or_buf=path_or_buf, **kwargs)

    def set_group(self, group:tb.Group, strict:bool=True)->None:
        """
        Set the HDF5 group in which tables are saved.

        Parameters
        ----------
        group : tb.Group
            Group in which to write tables.
        strict : bool, optional
            If :code:`True` raise an error if the group is alread set.
            This prevents splitting data between groups. 
            The default is True.

        Raises
        ------
        TypeError
            Group already set.

        """
        if self._group._created:
            if strict:
                raise TypeError("Cannot set new group if strict=True")
            warnings.warn(f"{self.__class__.__name__} group changed")
        self._group = _GroupFuture(group, self, self._track_callback)
        for table in self._tables.values():
            if table._derived:
                continue
            table._cache.reset_group(self._get_group_from_param(table.param))

    def save(self, *args:Param, group:tb.Group=None)->tb.Group:
        """
        Save computed tables into HDF5 file. 
        With no arguments, saves all currently stored tables into the 
        default location of the photon-HDF5 file ``/usr/FRETBursts/``.
        
        May specify specific tables to save through args, which must be :class:`Param`.
        If kwargs group is specified, tables stored within that HDF5 group.

        Parameters
        ----------
        *args : Param
            Tables to save, if none specified, then all computed tables saved.
        group : tb.Group, optional
            HDF5 group into which to save. The default is None.

        Returns
        -------
        group : tb.Group
            HDF5 group in which tables were saved.

        """
        if not args:
            args = (table.param for table in self._tables.values() if not table._derived) 
        for param in args:
            table = self.get_table(param)
            if group is None:
                table.save()
            else:
                table.save(self._get_group_from_param(param, group))
        group = self._group._group if group is None else group
        return group

    def close(self, strict:bool=False)->None:
        """
        Close the file(s) to which the :class:`DataSet` is attached.

        Parameters
        ----------
        strict : bool, optional
            Force closing of file if :code:`True`.
            
            If :code:`False` then detach :class:`DataSet` from tracking file,
            and close only if no other :class:`DataSet` are tracking the file.
            
            The default is False.

        """
        for finalizer in self._finalizers.values():
            finalizer.finalize_owner(weakref.ref(self), strict)


class DataSetList:
    """
    Convenience class for organizing multiple :class:`DataSet` objects together.
    Provides the basic get methods to retrieve equivalent :class:`Tables` and
    their columns from all datasets at the same time.
    
    Parameters
    ----------
    datas: Sequence[DataSet]
        The underlying :class:`DataSet` objects. Converted to tuple internally.
    
    """
    #: name to give to HDF5 groups based on this class attribute.
    #: Should be renamed for each subclass
    _group_name:ClassVar[str] = 'dataset' 
    
    def __init__(self, datas:Sequence[DataSet]):
        if not isinstance(datas[0], DataSet):
            raise TypeError(f"DataSetList can only organize DataSet objects, not {type(datas[0])}")
        self.settype = type(datas[0])
        if any(not isinstance((err:=d), self.settype) for d in datas):
            raise TypeError(f"DataSetList can only organize DataSet objects, not {type(err)}")
        self._datas = tuple(datas)
    
    @property 
    def datas(self):
        """Tuple of the underlying :class:`DataSet` objects that compose self"""
        return self._datas
        
    def get_table(self, param:Param, gate:GateGroup=None)->tuple["Table",...]:
        """
        Retrieve tuple of :class:`Table` objects of ``param``.

        Parameters
        ----------
        param : Param
            :class:`Param` defining table to retrieve.
        gate : GateGroup, optional
            Gate to apply to param. The default is None.

        Returns
        -------
        tuple[Table,...]
            tuple of :class:`fretbursts.datamodel.tables.Table` objects for each 
            dataset arrising from param.

        """
        if gate is not None:
            param = param.regate(gate)
        return tuple(data.get_table(param) for data in self._datas)
    
    def iter_table(self, param:Param, gate:GateGroup=None)->tuple["Table",...]:
        """
        Iterate over :class:`Table` objects of ``param`` in each dataset.

        Parameters
        ----------
        param : Param
            :class:`Param` defining table to retrieve.
        gate : GateGroup, optional
            Gate to apply to param. The default is None.

        Yields
        -------
        Table
            :class:`fretbursts.datamodel.tables.Table` object for each dataset 
            arrising from param.

        """
        if gate is not None:
            param = param.regate(gate)
        return tuple(data.get_table(param) for data in self._datas)
    
    def get_column(self, column:Column, gate:GateGroup=None)->tuple[np.ndarray,...]:
        """
        Get tuple of the arrays corresponding to column.

        Parameters
        ----------
        column : Column
            :class:`Column` to retrieve from each dataset.
        gate : GateGroup, optional
            Gate to apply to arrays. The default is None.

        Returns
        -------
        tuple[np.ndarray,...]
            tuple of the output arrays.

        """
        if gate is not None:
            column = column.regate(gate)
        return tuple(data.get_column(column) for data in self._datas)
    
    def iter_column(self, column:Column, gate:GateGroup=None, flatten:bool=False)->Iterator[Iterator|Any]:
        """
        Create an iterator over column values. If ``flatten=False`` then behaves
        like iterator over :meth:`DataSetList.get_column` , ie iterates over each
        dataset, which itself is an iterator over each row. If ``flatten=True``
        then behaves like iterator over :meth:`DataSetList.concatenate_column`.

        Parameters
        ----------
        column : Column
            :class:`Column` to iterate over in each dataset.
        gate : GateGroup, optional
            Gate to apply to arrays. The default is None.
        flatten : bool, optional
            Whether to iterate directly over The default is False.

        Yields
        ------
        Iterator|Any
            If ``flatten=False``, iterator over iterator of table rows, ie:
            >>> for table in datasetlist.itercolumn(col):
            >>>     for row in table:
            >>>         ...
            
            
            If ``flatten=True``, iterator over iterator of table rows, ie:
            >>> for row in datasetlist.itercolumn(col, flatten=True):
            >>>     ...
            
            
        """
        if gate is not None:
            column = column.regate(gate)
        gen = (data.iter_column(column) for data in self._datas)
        if flatten:
            yield from chain.from_iterable(gen)
        else:
            yield from gen

    def concatenate_column(self, column:Column, gate:GateGroup=None)->np.ndarray:
        """
        Concatenate the arrays of array for ``column`` in each dataset into single
        numpy array

        Parameters
        ----------
        column : Column
            :class:`Column` to concatenate.
        gate : GateGroup, optional
            Gate to apply to ``column``. The default is None.

        Returns
        -------
        np.ndarray
            Array of each dataset concatenated into single array.

        """
        return np.concatenate(self.get_column(column, gate=gate))

    def record_column(self, column:Column)->tuple[np.ndarray,...]:
        """
        Save specified column in memory.
        
        .. note::
            
            This recoreds in RAM, use :meth:`DataSetList.save` to record to HD5
            file.

        Parameters
        ----------
        column : Column
            Column to be saved.

        Returns
        -------
        tuple[np.ndarray,...]
            Values of column.

        """
        return tuple(data.record_column(column) for data in self._datas)

    def get(self, comp:Param|Column, gate:GateGroup=None)->tuple[Union["Table",np.ndarray],...]:
        """
        Retrieve either a tuple of :class:`Table` or ``np.ndarray`` objects
        represented by the input :class:`Param:` or :class:`Table`.

        Parameters
        ----------
        comp : Param |  Column
            Value to retrieve.
        gate : GateGroup, optional
            Gate applie to comp, if None, use gate of instruction object. 
            The default is None.

        Returns
        -------
        tuple[Table|np.ndarray, ...]
            Tuple of retrieved values.

        """
        if isinstance(comp, Param):
            return self.get_table(comp, gate=gate)
        return self.get_column(comp, gate=gate)
    
    def get_frame(self, *args:Column, names:Sequence[str]=None, gate:GateGroup=None,
                  multi_index:bool=False, include_unit:bool=False, index_name:bool=False)->pd.DataFrame:
        r"""
        Create a pandas dataframe from the columns specified in \*args. Concatentes
        each DataSet object in datas.

        Parameters
        ----------
        *args : Column
            :class:`Column` s to include in table.
        names : Sequence[str], optional
            Column names (must be same length as args) to give to each column. 
            If None, use default names of columns supplied by :func:`Column.name` .
            The default is None.
        gate : GateGroup, optional
            Gate to apply to all columns, over-rides any gates in columns.
            If not specified, take intersect of gates of each column.
            The default is None.
        multi_index : bool, optional
            When column rows are arrays, create mulit-index so each sub-value
            is given a separte row. The default is False.
        include_unit : bool, optional
            Whether to include unit in column names. The default is False.
        index_name bool, optional
            If True use names compatible with saving to csv file. 
            The default is False

        Raises
        ------
        ValueError
            One or more columns incmopatible with multi-index.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame of the selected columns.

        """
        return pd.concat((d.get_frame(*args, names=names, gate=gate, multi_index=multi_index, 
                                      include_unit=include_unit, index_name=index_name) 
                          for d in self.datas), ignore_index=True)
    
    def to_csv(self, *args:Column, path_or_buf:None|str|os.PathLike=None, names:Sequence[str]=None, 
               gate:GateGroup=None, include_unit:bool=False, multi_index:bool=False, **kwargs)->str|None:
        r"""
        Create a CSV of :class:`Column` specified in \*args. If the first
        argument is not a column, treat as argument to ``path_or_buf``.
        Concatentates each DataSet object in datas.
        
        >>> data.to_csv(col1, col2, path_or_buf='save.csv')
        
        is equivalent to
        
        >>> data.to_csv('save.csv', col1, col2)
        

        Parameters
        ----------
        *args : Column
            Columns to save in csv file.
        path_or_buf : None|str|os.PathLike, optional
            File or buffer to write to. If None, and not specified as first
            arg, return str of csv. Internally, this uses
            `pd.DataFrame.csv <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html>`_ . 
            The default is None.
        names : Sequence[str], optional
            Sequence of names to give to each column, if specified must be same
            length as number of columns specified. If None/not specified, will
            use index_name of each column. The default is None.
        gate : GateGroup, optional
            GateGroup to apply to all input columns. The default is None.
        include_unit : bool, optional
            Include unit string in column heading. The default is False.
        multi_index : bool, optional
            Used for columns whose rows are arrays. Will cause arrays to be "flattened"
            with index specified as multi-index. This is primarily for 'ph\_...' 
            columns. If True, the size of each row must be consistent between columns. 
            The default is False.
        **kwargs : Any
            Additional keyword arguments passed to 
            `pd.DataFrame.csv <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html>`_ .

        Raises
        ------
        TypeError
            Incompatible input.

        Returns
        -------
        str|None
            If path_or_buf is None, returns string representation of csv, otherwise
            None, and csv is written to path_or_buf.

        """
        if not args:
            raise TypeError("must specify at least one column")
        if not isinstance(args[0], Column):
            if path_or_buf is not None:
                raise TypeError("path_or_buf specified as arg and kwarg")
            path_or_buf, args = args[0], args[1:]
        if not args:
            raise TypeError("must specify at least one column")
        df = self.get_frame(*args, names=names, gate=gate, include_unit=include_unit,
                            multi_index=multi_index, index_name=True)
        return df.to_csv(path_or_buf=path_or_buf, **kwargs)
    
    def save(self, *args:Param, group:tb.Group=None, name:Callable[[int],str]=None, **kwargs)->list[tb.Group]:
        """
        Save computed tables into HDF5 file. 
        With no arguments, saves all currently stored tables into the 
        default location of the photon-HDF5 file ``/usr/FRETBursts/``.
        
        May specify specific tables to save through args, which must be :class:`Param`.
        If kwargs group is specified, tables stored within that HDF5 group.

        Parameters
        ----------
        *args : Param
            Tables to save, if none specified, then all computed tables saved.
        group : tb.Group, optional
            HDF5 group into which to save. The default is None.
        name : Callable[[int],str]
            If specified, each :class:`DataSet` identified by name[n] in subgroup of group.
            Otherwise, name each subgroup [dataset._group_name][n] 
            The default is None.

        Returns
        -------
        group : tb.Group
            HDF5 group in which tables were saved.

        """
        
        if group is None:
            return list(data.save(*args, **kwargs) for data in self._datas)
        if name is None:
            file = lambda i: group._v_file.create_group(group, f'{type(self)._group_name}{i}')
        else:
            file = lambda i: group._v_file.create_group(group, name(i))
        return list(data.save(*args, group=file(i)) for i, data in enumerate(self._datas))
    
    def close(self, strict:bool=False)->None:
        """
        Close the file(s) to which the :class:`DataSetList` is attached.

        Parameters
        ----------
        strict : bool, optional
            Force closing of file if :code:`True`.
            
            If :code:`False` then detach :class:`DataSetList` from tracking file,
            and close only if no other :class:`DataSet` are tracking the file.
            
            The default is False.

        """
        for data in self._datas:
            data.close(strict)


DataS = DataSet|DataSetList

###############################################################################
#### Classes for getting partial parameters/delayed parameters from Table  ####
###############################################################################
class _ParentsTuple:
    """
    Special type for accessing the :class:`Table` that are parents of a
    :class:`Table` listed in the parents :class:`tupledict` as tuples.
    """
    __slots__ = ('_origin', '_parents')
    _origin: DataSet #: data on which _ParentsTuple is based
    _parents: dict[str,Param] #: dictionary of :attr:`Param.parents`
    
    def __init__(self, origin:DataSet, parents:dict[str,Param]):
        super().__setattr__('_origin', origin)
        super().__setattr__('_parents', parents)
        
    def __getattribute__(self, attr):
        raise AttributeError("Parents have no attributes")

    def __setitem__(self, key, val):
        raise AttributeError("Cannot assign to parents dict")
    
    def __getitem__(self, key):
        return super().__getattribute__('_origin').get_table(super().__getattribute__('_parents')[key])
    
    def __len__(self):
        return len(super().__getattribute__('_parents'))
    
    def __length_hint__(self):
        return len(super().__getattribute__('_parents'))
    
    def __iter__(self):
        for parent in super().__getattribute__('_parents'):
            yield super().__getattribute__('_origin').get_table(parent)

    
class _ParentsDict:
    """
    Special type for :class:`Table` objects, when __getitem__ is used, returns
    the parent table requested. This is useful because it allows for computation
    of a given parent to be delayed until first requested.
    Mimics a dictionary.
    
    Parameters
    ----------
    oritin : DataSet
        The :class:`DataSet` on which the parent table is based
    parents : dict[str:Param]
        The parents attr of the Param which the table is based
    """
    __slots__ = ('_origin', '_parents')
    _origin : DataSet #: the data to which parents is linked
    _parents : dict[str,Param] #: dictionary of :attr:`Param.parents`
    
    def __init__(self, origin:DataSet, parents:dict[str,Param]):
        super().__setattr__('_origin', origin)
        parents = {key:_ParentsTuple(origin, val) if isinstance(val, tuple) else val 
                   for key, val in parents.items()}
        super().__setattr__('_parents', parents)
        
    def __getattribute__(self, attr):
        if attr not in ('keys', 'values', 'items', 'get'):
            raise AttributeError("Parents have no attributes")
        return super().__getattribute__(attr)

    def __getattr__(self, attr):
        return self[attr]
    
    def __setitem__(self, key, val):
        raise AttributeError("Cannot assign to parents dict")
    
    def __getitem__(self, key):
        out = super().__getattribute__('_parents')[key]
        if isinstance(out, _ParentsTuple):
            return out
        return super().__getattribute__('_origin').get_table(out)
    
    def __len__(self):
        return len(super().__getattribute__('_parents'))
    
    def __length_hint__(self):
        return len(super().__getattribute__('_parents'))

    def keys(self):
        """Mimic keys() of dictionary, iterate over all keys in _ParentsDict"""
        return super().__getattribute__('_parents').keys()

    def values(self):
        """Mimic values() of dictionary, iterate over all values in _ParentsDict"""
        for key in super().__getattribute__('_parents').keys():
            return self[key]

    def items(self):
        """Mimic items() of dictionary, iterate over all key:value pairs in _ParentsDict"""
        for key in super().__getattribute__('_parents').keys():
            return key, self[key]

    def get(self, key, default=None):
        """Mimic get() of dictionary, retrieve key, return default if does not exist in dictionary"""
        try:
            out = self[key]
        except:
            out = default
        return out
    
        
class TableConstructionError(ValueError):
    """Incorrect definition of subclass of Table"""
    pass


###############################################################################
##### String handling functions to help with creating table descriptions  #####
###############################################################################


###############################################################################
class TableLike(type):
    """Metaclass to ideintify a table that can be created (not partial)"""
    def __subclasscheck__(self, subclass):
        if any(not hasattr(subclass, attr) for attr in ('param_defs', 'parent_defs', 'column_defs')):
            return False
        if any(all(pdef.name != rc for pdef in subclass.param_defs) 
               for rc in getattr(self, 'required_params', [])):
            return False
        if any(all(pdef.name != rc for pdef in subclass.param_defs) 
               for rc in getattr(self, 'required_parents', [])):
            return False
        if any(all(cdef.name != rc for cdef in subclass.column_defs) 
               for rc in getattr(self, 'required_columns', [])):
            return False
        return True


class FullTable(metaclass=TableLike):
    """Metaclass to allow issubclass on any table-like that can actually be used to create table"""
    pass


class Table:
    """
    The ``Table`` type is responsible for data processing turning instructions
    found in :class:`Param` objects into actual processed values from raw data
    supplied by :class:`DataSet` objects.
    
    New subclasses of ``Table`` should never be subclassed directly from ``Table``,
    rather they should be subclassed from either :class:`BaseTable` or :class:`ChildTable`.
    These two subclasses serve as the "true" abstract base classes for all actual
    classes of ``Table``.
    
    Subclasses of ``Table`` should define the following:
        
        #. ``param_defs`` (a class variable) A tuple of :class:`ParamDef` objects
           defining the parameters, and their types that define the table.
        #. ``parent_defs`` (a class variable) A tuple of :class:`ParentDef` objects
           defining the parents and their types that define the table.
        #. ``column_defs`` (a class variable) A tuple of :class:`ColumnDef` objects
           defining the columns that the table will have, including the method calls
           used to create
           each column.
        #. ``__init_columns__`` (as a standard method) is a method that should be
           called whenever a new instance of a non-derived (ie no table above from
           which current is a masked version of). This should create and add any
           columns that are not computed each time the column is accessed.
    
    Additional methods can be defined, depending on the needs (``Table`` implements
    these methods in simple, always pass ways):
        
        #. ``validate_param``, should be a classmethod with siganture
           ``(cls, param:Param)->None``, this defines a validator function that
           takes Param with ``tp`` attribute of the given ``Table``, and ensures
           all values in parameters are valid, useful when simple fixed min/max
           etc. values are insufficient. Should raise appropriate error if not
           valid.
        #. ``param_preprocess`` (should be a classmethod) preprocess parameters
           dictionary. This can be used to create "alternate" parameter contruction
           keys etc. Should have the
           signature: ``(cls, params:Union[Sequence[tuple[str, Any]],tupledict])->dict``

    Finally, appropriately named methods for computing each column should be named.
    If they produce column as an iterator, it is recomened the method name start with
    ``_iter``, or if they return a column (more common), the method name should
    start with ``_get``. 
    
    Parameters
    ----------
    param : Param
        Defines what values to use to create/process raw data into a table.
        The ``tp`` attribute of param must be the same as the subclass of the
        ``Table`` to which it is assigned
    origin : DataSet
        Raw data on which the Table is based
    """
    _param: Param  #: full parameter definition for column
    _origin: DataSet  #: DataSet for which data is based
    _parents: dict[str, "Table"]  #: dictionary of parent tables
    _derived: bool  #: whether or not derived
    # : DiskDict storing all atomic non-dynamic columns, used only for primary table
    _cache: DiskDict | Callable[[],DiskDict]
    # the following are specific to derived tables
    #: Reference to primary table of table, specified if masked.
    _base: Union["Table", None]
    #: Value depends on wether _base is None, if so, then should be slice(None) if there is no gate,
    _mask: np.ndarray[np.bool_]
    #: The names and types that define the parameters of an instance of the Table.
    #: **Should be specified per subclass.**
    param_defs:ClassVar[tuple[...,ParamDef]] 
    #: tuple of ParentDefs, define the parents of an instance of the Table.
    #: **Should be specified per subclass.**
    parent_defs: ClassVar[tuple[...,ParentDef]] 
    #: tuple of ColumnDefs, defines all columns in the table.
    #: If columns are marked as non-atomic, they will not be saved in HDF5 format.
    #: **Should be specified per subclass.**
    column_defs:ClassVar[tuple[...,ColumnDef]] 
    #: **Specific to subclass** Methods to use as classmethod on param
    _parammethods:ClassVar[ImDict] = frozenset()
    

    @classmethod
    def _get_val_str(cls, value:Any, keylen:int)->str:
        """
        Function for turning values into str for YAML-like representations.
        Subclasses can implement their own to give special representations to
        particular data types in val.
        """
        vstr = [f"{value.__module__}.{value.__name__}",] if callable(value) else str(value).split('\n')
        return '\n'.join(' '*keylen+f'  {ln}' if i else ln for i, ln in enumerate(vstr))

    @classmethod
    def _get_kv_str(cls, key:str, value:Any, indent_level:int=0)->str:
        """
        Function for turning :attr:`Param.params` key:value pairs into YAML-
        like representation. 
        Subclasses can implement custom representations
        """
        if isinstance(value, tuple):
            vals = [cls._get_val_str(val, 0) for val in value]
            if any('\n' in v for v in vals) or sum(len(v) for v in vals) > (50 - indent_level):
                return f'{key}: (\n  -' + _indent('\n-'.join(cls._get_val_str(v, 0)
                                                         for v in value), 2).lstrip() + '\n  )'
            return f'{key}: (' + ', '.join(vals) + ')'
        return f'{key}: ' + cls._get_val_str(value, len(key))

    def __init_subclass__(cls):
        if not hasattr(cls, 'param_defs'):
            return
        if not hasattr(cls, 'parent_defs'):
            return
        if not hasattr (cls, 'column_defs'):
            return 
        if any(not isinstance(p, ParamDef) for p in cls.param_defs):
            raise TableConstructionError('param_defs must be  ParamDef')
        if any(not isinstance(p, ParentDef) for p in cls.parent_defs):
            raise TableConstructionError('parent_defs must be  ParentDef')
        if any(not isinstance(c, ColumnDef) for c in cls.column_defs):
            raise TableConstructionError('column_defs must be  ColumnDef')
        if sum(isinstance(pdef, ParentDef) and pdef.is_base for pdef in cls.parent_defs) > 1:
            raise TableConstructionError("Only single parent may be defined as base")
        register_type(cls)

    def __init__(self, param:Param, origin:DataSet):
        self._init_universal_(param, origin)
        if origin.check_param_saved(param):
            self._init_new_()
        else:
            for parent_param, table in origin._tables.items():
                if not (Param.param_comp(param, parent_param) & 0b0010):
                    self._init_derived_(table)
                    break
            if not hasattr(self, '_derived'):
                self._init_new_()
        self.__post_init__()
    
    def __post_init__(self):
        """Function called after all other inits for table, 
        used per subclasses for final processing"""
        pass

    def _init_universal_(self, param:Param, origin:DataSet):
        self._param = param
        self._origin = origin
        self._parents = _ParentsDict(self.origin, self.param.parents)
    
    # @abstractmethod
    def _init_new_(self)->None:
        """
        This method implemetns the calls for setting up a new table. 
        :class:`BaseTable` and :class:`ChildTable` implement this method, and
        further subclasses generally should not need to re-implemnt
        """
        raise NotImplementedError(
            "Direct subclasses of Table should not be implemented, but be subclass of BaseTable or ChildTable")

    # @abstractmethod
    def _init_derived_(self, parent:"Table")->None:
        """
        This method implemetns the calls for setting up a table that is a subset
        of another table. 
        :class:`BaseTable` and :class:`ChildTable` implement this method, and
        further subclasses generally should not need to re-implemnt
        """
        raise NotImplementedError("subclasses must implement this method (should be in BaseTable or ChildTable)")

    # @abstractmethod
    def __init_columns__(self)->None:
        """
        Method called at end of init, should build and assign initial "default" columns
        """
        pass

    def _validate_load_(self)->None:
        """Validate values loaded match sizes etc."""
        raise NotImplementedError("subclasses must implement this method (should be in BaseTable or ChildTable)")

    def convert_to_base_table(self)->"Table":
        """
        Convert table (inplace) into non-derived table. This ensures all columns
        are stored in separate arrays and do not need to be masked when fetching,
        and that this table (with gate) can be saved in unique HDF5 group.
        This method is emplementd
        """
        if not self._derived:
            return self
        self._cache = DiskDict(group=self.origin._get_group_from_param(self._param), 
                               autosave=self.origin._get_autosave)
        for key, val in self._bcache.items():
            self._cache[key] = val
        delattr(self, '_base')
        delattr(self, '_mask')
        self._derived = False
        return self
    
    def _rebase(self, gate:GateGroup)->"Table":
        """Change :attr:`Table._base` (if derived) to table with gategroup gate"""
        if gate == self.param.base_gate:
            return self.convert_to_base_table()
        if not issubclass(gate.origin_param.tp, BaseTable):
            raise ValueError("can only rebase to BaseTable type Param")
        if gate.origin_param != self.param.origin_param:
            raise ValueError("Mismatched origin_params, cannot rebase to given param")
        if GateGroup.overlap(self.param.base_gate, gate) & 0b0010:
            raise ValueError("can only rebase to param with gate larger than current gate")
        self._mask = self.origin._get_gategroup_mask(self.param.base_gate, gate)
        if not self._derived:
            delattr(self, '_cache')
            self._derived = True
        return self
    
    @classmethod
    def load_group(cls, group:tb.Group, origin:None|DataSet=None)->"Table":
        """
        Load a specifc HDF5 group as a table.

        .. note::

            This method should rarely be called directly, as groups should be
            saved and loaded from a larger FRETBursts HDF5 data structure.

        Parameters
        ----------
        group : tb.Group
            pytables object representing the HDF5 group from which the table
            is to be loaded.
        origin : None|DataSet, optional
            The DataSet object connected to the data, if available. The default is None.

        Raises
        ------
        TypeError
            Wrong type for argument group.

        Returns
        -------
        Table
            Table containing all saved columns of specified param.

        """
        if not isinstance(group, tb.Group):
            raise TypeError(
                f"group must be a pytables Group, got {type(group)}")
        if origin is not None and not isinstance(origin, DataSet):
            raise TypeError("origin must be a DataSet or None")
        obj = cls.__new__(cls)
        obj._param = Param.load_group(group['param'])
        obj._origin = origin
        obj._cache = DiskDict(group=group, 
                              autosave=True if origin is None else origin._get_autosave)
        if origin is not None:
            obj._parents = _ParentsDict(obj.origin, obj.param.parents)
            obj._validate_load_()
        return obj

    @classmethod
    def _param_preprocess(cls, params:Sequence[tuple[str, Any]]|tupledict, parents:dict[str,Param])->tuple[dict, dict]:
        """Method actually called to pre-process param. Used so subclasses can
        add additional validation without final Table needing to call super()"""
        return cls.param_preprocess(params, parents)

    @classmethod
    def param_preprocess(cls, params:Sequence[tuple[str, Any]]|tupledict, parents:dict[str,Param])->tuple[dict, dict]:
        """
        Pre-processor for Param params dict/tupledict. This method is called when
        processing the input of the params field when instantiating a new Param.
        This allows for pre-processing input to Param.params in order to regularize
        inputs.


        Parameters
        ----------
        params : Sequence[tuple[str,Any]]|tupledict
            Param.params to be pre-processed

        Returns
        -------
        tupledict
            Processed/regularized params input.

        """
        return params, parents

    @classmethod
    def _validate_param(cls, param:Param)->None:
        """Method actually called to validate param. Used so subclasses can
        add additional validation without final Table needing to call super()"""
        cls.validate_param(param)

    @classmethod
    def validate_param(cls, param:Param)->None:
        """
        Function that checks parameter is valid. Will run at end of 
        __post_init__ when creating new Param
        Used if parameter values have non-trivial dependencies
        """
        pass

    @classmethod
    def _regularize_column_kwargs(cls, **kwargs)->dict[str,Any]:
        """
        Called before column is instantiated, primarily used to convert types
        in keytupto cannonical form and preventing disallowed keys.
        
        Takes source_param, col, ketup, offset, and fill as kwargs, returns
        dictionary with components adjusted accordingly.
        """
        return kwargs

    @property
    def group(self)->tb.Group|None:
        """
        The group where Table data will be saved, if callable, then group has not
        been created, and calling hdf5_group(self.param) will create and return
        the group.
        """
        return self._group._groupcurrent

    @property
    def param(self)->Param:
        """The Param object defining the Table"""
        return self._param

    @property
    def parents(self)->tupledict[str,'Table']:
        """References to each parent Table, or callable to generate said parent Table"""
        return self._parents

    @property
    def origin(self)->DataSet:
        """The DataSet on which the Table is based"""
        return self._origin
    
    @property
    def _bcache(self)->DiskDict:
        """Cache that can look at cache of Table._base, and return approriatly mased arrays"""
        if self._derived:
            return MaskedDD(self._base._bcache, self._mask)
        return self._cache

    @classmethod
    # @abstractmethod
    def max_gate_from_param(cls, param:Param)->GateGroup:
        """Returns largest reasonable gate of param"""
        return GateGroup(truthtable=_TT_all, param=param.origin_param)

    @classmethod
    def _get_columndef(cls, columnname:str)->ColumnDef:
        """
        Retrieves the ColumnDef of the columnname (str) specified
        """
        for coldef in cls.column_defs:
            if columnname == coldef.name:
                return coldef
        raise AttributeError(f'{cls.__name__} has no column {columnname}')

    def _add_column(self, col:str, keys:tuple[Hashable,...], array:np.ndarray)->None:
        """
        **Private method used by :class:`Tables` subclasses**
        **Should only be called by other methods of self, never outwardly by user**
        
        Add a column to the table's cache.

        Parameters
        ----------
        col : str
            name oc column to add.
        keys : tuple[Hashable,...]
            Additional keys required for defining column.
        array : np.ndarray
            array of values to add to cache.

        Raises
        ------
        ValueError
            Derived table, inspect the code of your table.
        TypeError
            keys of wrong type or length.
        
        :meta public:
        """
        if self._derived:
            raise ValueError("cannot add colunmn to derived table")
        coldef = self._get_columndef(col)
        if coldef.store == 'never':
            raise TableConstructionError("trying to add non-stored column")
        if len(keys) != len(coldef.keytypes):
            raise TypeError("length of keys does not match column, expecting "
                            f"{len(coldef.keytypes)}, got {len(keys)}")
        try:
            keys = tuple(tv.check_val(key) for key, tv in zip(keys, coldef.keytypes))
        except Exception as e:
            raise TypeError("one or more keys is of wrong type", *e.args) from e
        array = np.asarray(array, dtype=coldef.dtype)
        self._check_new_column(coldef, keys, array)
        self._cache[(col, ) + keys] = array

    def _get_keys(self, keys:tuple[str,Hashable,...])->tuple[ColumnDef,tuple[Hashable,...],int,Any]:
        """Interpret keys as (coldef, (keys,...), offset, fill)"""
        if isinstance(keys, str):
            keys = (keys, )
        coldef, keys = self._get_columndef(keys[0]), keys[1:]
        if coldef.reg_func:
            keys = getattr(self, coldef.reg_func)(*keys)
        offset_fill = len(keys) - coldef.keylen
        keytup = keys,
        # case of column with same size as table
        if coldef.offset == 0:
            if offset_fill > 0:
                raise KeyError(
                    f"incorrect number of keys, expected {len(coldef.keytypes)+1}, got {len(keys)}")
        # case of column larger than size as table, so cannot have fill value
        elif coldef.offset > 0:
            if offset_fill == 1:
                keytup = keys[:-1], keys[-1]
            elif offset_fill != 0:
                raise KeyError(f"expected {len(coldef.keytypes)+1} keys, and " + 
                               f"possibly one offset, got {len(keys)}")
        # case of column smaller than size of table, so need fill value
        elif coldef.offset < 0:
            if offset_fill == 2:
                keytup = keys[:-2], keys[-2], keys[-1]
            elif offset_fill == 1:
                keytup = keys[:-1], keys[-1], coldef.fill
            elif offset_fill != 0:
                raise KeyError(
                    f"expected {len(coldef.ketypes)+1} keys, and possibly " +
                    "offset and fill value, got {len(keys)}")
        # verify offset has valid value
        if offset_fill > 0:
            if not hasattr(keytup[1], "__index__"):
                raise TypeError(
                    f"offset value must be positive integer, got {type(keytup[1])}")
            if keytup[1] < 0 or keytup[1] > abs(coldef.offset):
                raise ValueError("offset out of range, must be between 0 and " +
                                 f"{coldef.offset}, got {keytup[1]}")
        if 'remap' in coldef:
            new_keys = getattr(self.param.tp, coldef.remap)(coldef.name, *keytup)
            return self._get_keys((new_keys[0],)+new_keys[1]+new_keys[2:])
        if coldef.reg_func:
            keytup = (getattr(self, coldef.reg_func)(*keytup[0]),) + keytup[1:]
        keytup = (coldef, ) + keytup + (None,)*(3-len(keytup))
        return keytup

    def _check_new_column(self, coldef:ColumnDef, keys:tuple[Hashable,...], array:np.ndarray)->None:
        """Confirm size and shape of array matches expected"""
        if 'mapto' in coldef:
            self.origin.get_table(keys[0])._check_array_size(coldef.offset, array)
        else:
            self._check_array_size(coldef.offset, array)
        if array.ndim != coldef.ndim or any(s < low or s > high for (low, high), s in zip(coldef.dimlimits, array.shape[1:])):
            raise TableConstructionError(f'Size of column incompatible with size of table, expected {coldef.dimlimits}, got {array.shape}')
        if not np.issubdtype(array.dtype, coldef.dtype):
            raise TableConstructionError(f'Wrong column dtype, expected {coldef.dtype}, got {array.type}')
            if coldef.dtype == np.object_ and any(not np.issubdtype(arr.dtype, coldef.typedef) 
                                                for arr in array.reshape(-1)):
                raise TableConstructionError('Wrong internal type of columns')

    def _check_array_size(self, offset:int, array:np.ndarray)->None:
        """Check that array matches size of table. Used in verifying new column is valid"""
        if array.shape[0] - offset != self.size:
            raise TableConstructionError(f"Array has the wrong number of rows expected {self.size+offset}, got {array.shape[0]}")

    def _compute_array_column(self, coldef:ColumnDef, keys:tuple[Hashable])->np.ndarray:
        """
        **Should only be called on base columns, or if coldef.get_derived**
        Perform computation of column, output as array.
        """
        if coldef.get_func:
            out = getattr(self, coldef.get_func)(*keys)
        else:
            if coldef.ndim == 1:
                out = np.fromiter(getattr(self, coldef.iter_func)(*keys), coldef.dtype)
            else:
                out = np.array(list(getattr(self, coldef.iter_func)(*keys)), coldef.dtype)
        self._check_new_column(coldef, keys, out)
        if coldef.store == 'all':
            self._cache[((coldef.name, )+ keys)] = out
        return out

    def _compute_iter_column(self, coldef:ColumnDef, keys:tuple[Hashable,...])->Iterator:
        """
        **Should only be called on base columns, or if coldef.get_derived**
        Perform computation of column output as iterator.
        """
        if self._derived and not coldef.get_derived:
            raise TableConstructionError("only non-derived columns or columns marked get_derived may call _compute_iter_column")
        if coldef.iter_func:
            column_iter = getattr(self, coldef.iter_func)(*keys)
            if coldef.store == 'all':
                size, arr, cont = 0, list(), True
                expected_size = self.size + coldef.offset
                while cont:
                    try:
                        out = next(column_iter)
                    except StopIteration:
                        cont = False
                        raise TableConstructionError('column is too small')
                    size += 1
                    if size < expected_size:
                        arr.append(out)
                        yield out
                    else:
                        try:
                            next(column_iter)
                        except StopIteration:
                            array = np.array(arr, dtype=coldef.dtype)
                            self._check_new_column(coldef, keys, array)
                            self._cache[(coldef.name,)+keys] = array
                            yield out
                            cont = False
                        else:
                            cont = False
                            raise TableConstructionError('column is too long')
            else:
                yield from column_iter
        else:
            array = self._compute_array_column(coldef, keys)
            if coldef.store == 'all':
                self._cache[(coldef.name,)+keys] = array
            yield from array

    def __getitem__(self, keys:tuple[str,Hashable,...]):
        coldef, keys, offset, fill = self._get_keys(keys)
        ckeys = (coldef.name,) + keys
        if coldef.store != 'never' and  ckeys in self._bcache:
            out, mask = self._bcache[ckeys], False
        elif not self._derived or coldef.get_derived:
            out, mask = self._compute_array_column(coldef, keys), False
        else:
            out, mask = self._base._compute_array_column(coldef, keys), True
        if offset is not None:
            if coldef.offset > 0:
                out = out[offset:self.size+offset]
            else:
                nout = np.empty(self.size, dtype=coldef.dtype)
                nout[offset:self.size+coldef.offset+offset] = out
                nout[:offset] = fill
                nout[self.size+coldef.offset+offset:] = fill
                out = nout
        if mask:
            out = out[self._mask]
        return out

    def iter_column(self, *args)->Iterator:
        r"""Iterator over given columns specified by \*args"""
        if len(args) == 1 and isinstance(args[0], Column):
            args = args[0]._get_func_args
        coldef, keys, offset, fill = self._get_keys(args)
        ckeys = (coldef.name,) + keys
        if coldef.offset < 0 and offset is not None:
            for _ in range(offset):
                yield fill
        if coldef.store != 'never' and ckeys in self._bcache:
            column_iter = self._bcache.iter_key(ckeys)
        elif not self._derived or coldef.get_derived:
            column_iter = self._compute_iter_column(coldef, keys)
        else:
            column_iter = _masked_iter(self._base._compute_iter_column(coldef, keys), self._mask)
        if coldef.offset > 0:
            yield from _delayed_iter(column_iter, coldef.offset, offset)
        else:
            yield from column_iter
        if coldef.offset < 0 and offset is not None:
            for _ in range(-coldef.offset-offset):
                yield fill

    def record_column(self, *args)->np.ndarray:
        """
        Record/save column in RAM, only for columns marked as store='user'
        May specify column either as str, [keys, ...] or as Column.
        """
        if len(args) == 1 and isinstance(args[0], Column):
            args = args[0]._get_func_args
        if self._derived:
            return self._base.record_column(*args)[self._mask]
        coldef, keys, offset, fill = self._get_keys(args)
        ckeys = (coldef.name,)+keys
        if ckeys in self._cache:
            return self._cache[ckeys]
        array = self[args]
        if coldef.store == 'user':
            self._add_column(coldef.name, keys, array)
        return array

    def save(self, group:tb.Group=None, include_dataID:bool=False)->tb.Group:
        """
        Save all recorded column into HDF5 group.

        Parameters
        ----------
        group : tb.Group, optional
            Group in which to save table, if None, uses default set at beginning.
            The default is None.
        
        Returns
        -------
        tb.Group
            tables Group where table was saved.
        
        """
        if group is None:
            group = self._cache.save()
        if isinstance(group, _GroupFuture):
            group = group._create()
        if 'param' not in group:
            self.param.write_group(group, 'param')
        if include_dataID and 'dataID_' not in group:
            group._v_file.create_array(group, 'dataID_', self.origin._get_dataID())
        return self._cache.save(group)

    @classmethod
    def get_param_paramsdescr(cls, param:Param, indent:int=0)->str:
        """
        Get description of param's :attr:`Param.params`.

        Parameters
        ----------
        param : Param
            Param to describe the params attr.
        indent : int, optional
            Indentation level. The default is 0.

        Returns
        -------
        str
            YAML-like description of .

        """
        return _indent('\n'.join(cls._get_kv_str(key, value, indent) 
                                 for key, value in param.params.items()), indent)

    @classmethod
    def get_param_parentsdescr(cls, param:Param, indent:int=0, include_gate:bool=False)->str:
        """
        Get YAML-like description of param's :attr:`Param.parents`.

        Parameters
        ----------
        param : Param
            Param from whom to describe parents.
        indent : int, optional
            Indentation level. The default is 0.
        include_gate : bool, optional
            Whether to include gate of parents in their descriptions. 
            The default is False.

        Returns
        -------
        str
            YAML-like description of parents.

        """
        out = ''
        inc_gate = include_gate or issubclass(cls, BaseTable)
        for key, value in param.parents.items():
            out += f'\n{key}:\n'
            if not isinstance(value, Param):
                for i, val in enumerate(value):
                    txt = val.tp.get_param_description(val, 2, inc_gate).lstrip()
                    out += _indent(f'- {txt}', 2)
            else:
                out += value.tp.get_param_description(value, 2, inc_gate)
        return _indent(out.strip(), indent)

    @classmethod
    def get_param_descr(cls, param:Param, indent:int=0, include_gate:bool=True)->str:
        """
        Get YAML-like description of param.

        Parameters
        ----------
        param : Param
            Param to describe.
        indent : int, optional
            Indentation level. The default is 0.
        include_gate : bool, optional
            Whether to include description of gate as well. The default is True.

        Returns
        -------
        str
            YAML-like description of params of Param.

        """
        out = str()
        if param.params:
            out += 'Params:\n' + cls.get_param_paramsdescr(param, 2)
        if param.parents:
            out += '\nParents:\n' + cls.get_param_parentsdescr(param, 2)
        if include_gate:
            if not param.base_gate:
                out += '\nGateGroup: Empty Gate'
            if param.base_gate.truthtable.ndim:
                out += '\n' + param.base_gate.get_description()
        return _indent(out, indent)

    @classmethod
    def get_param_description(cls, param:Param, indent:int=0, include_gate:bool=True)->str:
        """
        Class method to generate human-readable description/definition of a
        :class:`Param` with :attr:`Param.tp` of the same class as :class:`Table`
        calling it.

        Parameters
        ----------
        param : Param
            Object to create description of.
        indent : int, optional
            How many left spaces to pad each line. The default is 0
        include_gate : bool, optional
            Whether to include GateGroup (if present) in description.
            The default is True

        Returns
        -------
        str
            Human readable description of Param of given class.

        """
        txt = cls.get_param_descr(param, 0, include_gate)
        return _indent(f'Table: {param.tp.__name__}\n{txt}', indent)

    @classmethod
    def get_column_keydescr(cls, column:Column, indent:int=0)->str:
        """
        Get description of column keys (col + keytup) in YAML-like format.

        Parameters
        ----------
        column : Column
            Column of which to describe keys.
        indent : int, optional
            Indentation of output. The default is 0.

        Returns
        -------
        str
            Description of column keys.

        """
        out = column.col
        if column.keytup:
            keytup = column.keytup[1:] if 'mapto' in column._get_coldef() else column.keytup
            out += ', ' + ', '.join(str(key) for key in keytup)
        if 'offset' in column:
            out += f', offset={column.offset}'
        if 'fill' in column:
            out += f', fill={column.fill}'
        return _indent(out, indent)

    @classmethod
    def get_column_maptodescr(cls, column:Column, indent:int=0)->str:
        """
        Get YAML-like description of source_param of column ()

        Parameters
        ----------
        column : Column
            Column to describe.
        indent : int, optional
            Indentation of output. The default is 0.

        Returns
        -------
        str
            Description of source_param of column.

        """
        return _indent(column.source_param.description, indent)

    @classmethod
    def get_column_descr(cls, column:Column, indent:int=0, 
                         include_param:bool=False, include_source:bool=True)->str:
        """
        Get YAML-like description of a column based on subtype of Table.
        
        Parameters
        ----------
        column : Column
            Column to get descriptio nof.
        indent : int, optional
            Indentation to add to output. The default is 0.
        include_param : bool, optional
            Whether to include description of param on which column is based. 
            The default is True.
        include_source : bool, optional
            If column is mapped, whether to include description of source_param
            of column. The default is True.

        Returns
        -------
        str
            YAML-like description of column.

        """
        param = column.param
        out = f'{param.tp.__name__}, {cls.get_column_keydescr(column)}'
        if include_param:
            out += '\n' + param.tp.get_param_descr(param, 2)
        if include_source and 'mapto' in column._get_coldef():
            out += '\nSource:\n' + _indent(cls.get_column_maptodescr(column),2)
        if 'gategroup' in column:
            out += '\nFilter Gate:\n' + _indent(column.gategroup.get_description(), 2)
        return _indent(out, indent)

    @classmethod
    def get_column_description(cls, column:Column, indent:int=0, 
                               include_param:bool=True, include_source:bool=True)->str:
        """
        Get YAML-like description of a column based on subtype of Table.
        Prepends Column label.

        Parameters
        ----------
        column : Column
            Column to get descriptio nof.
        indent : int, optional
            Indentation to add to output. The default is 0.
        include_param : bool, optional
            Whether to include description of param on which column is based. 
            The default is True.
        include_source : bool, optional
            If column is mapped, whether to include description of source_param
            of column. The default is True.

        Returns
        -------
        str
            YAML-like description of column.

        """
        out = f'Column: {cls.get_column_descr(column, 0, include_param, include_source)}'
        return _indent(out, indent)


class BaseTable(Table):
    """
    Table which defines the rows of a table.
    
    Tables should be subclassed from either this class or :class:`ChildTable`
    depending on if they define rows or match the rows of an existing table
    """
    _init_with_gate: ClassVar[bool] = False #: whether non-derived table can be initialized with a gate

    _param: Param  # : full parameter definition for column
    _origin: DataSet  # : DataSet for which data is based
    _parents: dict[str, Table]  # : dictionary of parent tables
    _derived: bool  # : whether or not derived
    # : DiskDict storing all atomic non-dynamic columns, used only for primary table
    _cache: DiskDict|Callable[[],DiskDict]
    # the following are specific to derived tables
    # : Reference to primary table of table, specified if masked.
    _base: Union["Table", None]
    # : Value depends on whether _base is None, if so, then should be slice(None) if there is no gate,
    _mask: np.ndarray[np.bool_]
    _size: int  # : number of rows in table

    def _init_new_(self):
        if not self.origin.check_param_saved(self.param) and 'gategroup' in self.param and not self._init_with_gate:
            table = self.origin.get_table(self.param.degate())
            self._init_derived_(table)
            return
        self._cache = DiskDict(group=self.origin._get_group_from_param(self._param), 
                               autosave=self.origin._get_autosave)
        self._derived = False
        if len(self._cache):
            self._validate_load_()
        else:
            self._size = None
            self.__init_columns__()

    def _init_derived_(self, parent:"BaseTable"):
        self._derived = True
        self._base = parent
        self._mask = self.origin._get_gategroup_mask(self.param.base_gate, parent.param.base_gate)
        self._size = 0 if isinstance(self._mask, slice) else self._mask.sum()

    def _validate_load_(self)->None:
        self._size = None
        for colname, colarray in self._cache.items():
            if not isinstance(colname, tuple):
                continue
            size = colarray.size - self._get_columndef(colname[0]).offset
            self._size = size
            break
        self._derived = False

    def _rebase(self, gate:GateGroup)->"BaseTable":
        """If table is derived, set so that _base has gategroup of gate"""
        if super(BaseTable, self)._rebase(gate)._derived:    
            self._base = self.origin.get_table(gate.base_param)
        return self

    @property
    def size(self)->int:
        """
        Number of rows in table, if None, table has not yet computed
        the first column, and thus number of rows in unknown. Must get any
        columns to set
        """
        return self._size

    @property
    def base_table(self)->"BaseTable":
        """Table which defines rows. Returns self, as by definition a BaseTable defines the base"""
        return self

    def _check_array_size(self, offset:int, array:np.ndarray)->None:
        """
        Verify that array is appropriate size given offset for table.
        re-written from :class:`Table` so that if _size is None, can
        set size of table based on first time column is computed
        """
        if self._size is None:
            self._size = array.shape[0] - offset
        super(BaseTable, self)._check_array_size(offset, array)


class ChildTable(Table):
    """
    Table which has same rows defined by some table in the tree of its parents.
    """
    _param: Param  # : full parameter definition for column
    _origin: DataSet  # : DataSet for which data is based
    _parents: dict[str, Table]  # : dictionary of parent tables
    _derived: bool  # : whether or not derived
    # : DiskDict storing all stored columns, used only for primary table (**note** these are never mapped or atomic)
    _cache: DiskDict|Callable[[], DiskDict]
    # attributes bellow are specific to derived tables
    # : Reference to primary table of table, specified if masked.
    _base: Union["Table", None]
    _base_table: BaseTable  # : reference to the non-gated version of the current table

    def _init_new_(self):
        self._cache = DiskDict(group=self.origin._get_group_from_param(self.param), 
                               autosave=self.origin._get_autosave)
        if len(self._cache):
            self._validate_load_()
        else:
            self._derived = False
            self._base_table = self.origin.get_table(self.param.base_param)
            self.__init_columns__()

    def _init_derived_(self, parent:"ChildTable"):
        self._derived = True
        self._base = parent
        self._base_table = self.origin.get_table(self.param.base_param)
        self._mask = self.origin._get_gategroup_mask(self.param.base_gate, parent.param.base_gate)

    def _validate_load_(self)->None:
        self._derived = False
        self._base_table = None
        if self.origin.has_table(self.param.base_param):
            self._base_table = self.origin.get_table(self.param.base_param)
            for colname, colarray in self._cache.items():
                if not isinstance(colname, tuple):
                    continue
                colsize = colarray.size - self._get_columndef(colname[0]).offset
                if colsize != self.size:
                    raise TypeError("group corresponds to table from a different data set, "
                                    f"column sizes different {colsize} vs {self.size}")

    def _rebase(self, gate:GateGroup)->"ChildTable":
        """Rebase table so that _base_table and _base point to tables with gategroup gate"""
        if super(ChildTable, self)._rebase(gate)._derived:
            self._base_table = self.origin.get_table(gate.base_param)
            self._base = self.origin.get_table(self.param.regate(gate))
        return self

    @property
    def base_table(self)->BaseTable:
        """
        The :class:`BaseTable` object on which the current table is based (matching rows).
        This is the base param of the :class:`Param` on which the table is based.
        """
        if self._base_table is None:
            self._base_table = self.origin.get_table(self.param.base_param)
            for colname, colarray in self._cache.items():
                colsize = colarray.size - self._get_columndef(colname[0]).offset
                if colsize != self._base_table.size:
                    raise ValueError("Inconsistent size between retrieved base_table and self")
                break
        return self._base_table

    @property
    def size(self)->int:
        """Number of rows in table"""
        return self.base_table.size

    @classmethod
    def _get_base_param(cls, param:Param)->Param:
        """Return the base param of the table, iterates through parents"""
        return param.parents[_get_baseparent(cls.parent_defs).name]
