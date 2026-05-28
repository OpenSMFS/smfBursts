#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author Paul David Harris
# email: harripd@gmail.com
# created: 09/04/2025
# purpose: base classes for immutable data
"""
This module creates a system for building customizable, weak-to-strongly typed
immutable data objects. The two main components of this are 
the :class:`TypeValidator` and :class:`_ImData` classes.

The :class:`TypeValidator` objects have several methods used to check, read, and
write other objects for their correct type (or conversion to correct type),
reading and writing from HDF5 files.

The :class:`_ImData` class is used as a abstract base class for creating classes
that have specific fields which are immutable, and can be accessed in a dictionary
like way, and can be wirtten and read from HDF5 files.
"""
from typing import ClassVar, Any, Union
from collections import Counter
from collections.abc import Callable, Sequence, Hashable
from functools import partial
from itertools import product, repeat
import types
import warnings
import re
from numbers import Number

import numpy as np
import tables as tb

from .utils import (_DataLike, _ImDataLike, ImDict, tupledict, FixedDict, 
    _dimscompare, _echo, iter_funcinput, _tuple_array, _eq, 
    _iter_tbgroup_numeric, _const_hash)


def _tuple_arr_tdct(arr:Union[dict,tuple[str,Any]]):
    """Converts arr to a tupledict"""
    if isinstance(arr, tupledict):
        return tupledict(*((k, _tuple_array(v)) for k, v in arr.items()))
    if isinstance(arr, dict):
        return tupledict.from_order(sorted(arr.keys()), **{k:_tuple_arr_tdct(v) 
                                                         for k, v in arr.items()})
    return _tuple_array(arr)


def _emptydict(*args, **kwargs):
    """returns empty dict, simple creator function"""
    return dict()


def _make_emptydict(*args, **kwargs):
    """
    Callable that returns another callable that always returns empty dict, used
    for default when default calls any callable default (facotry-like behavior)
    """
    return _emptydict


def _return_first(val, *args, **kwargs):
    """dummy function, returns first argument"""
    return val


class TypeValidator(_ImDataLike):
    r"""
    Core class for creating type-enforcing behavior.
    
    Key attributes
    --------------
    type\_ : type
        The type of data the TypeValidator object expects
    check : callable
        A function that checks if the input is valid, and returns appropriately
        converted (immutable) object. The signature must accept the first argument
        as the value to be checked, all additional arguments should be optional
        keyword arguments used for specifying limits or additional conditions
        on typevalidator.
    write : callable
        A function that writes the specified type to an HDF5 file. Must have the
        signature ``write(group:tables.Group, name:str, val:Any)`` where
        group is the group in which to write the value, name is the subgroup name
        (should create a node) and val is the val to be recorded in group/name.
    node_prefix : str, optional
        *Only for types that can be written as nodenames.* Name to prepend to 
        the string prepresentation of a value as a node name.
    node_repr : callable, optional
        *Only for types that can be written as nodenames.* A callable that takes 
        as a single argument a type, and converts it into a string that uniquiely 
        represents the object, used for creating node names.
    node_read: callable, optional
        *Only for types that can be written as nodenames.* A callable that takes
        the string of a HDF5 node name and return the value it represents.
    data_proc : callable, optional
        Function used by :meth:`TypeValidator.check_val` (if specified) to produce
        a set of kwargs passed to :attr:`TypeValidator.check`.
        Should have the signature ``(predata*, **validator_dict)->dict``.
    ckwargs : dict, optional
        dictionary of keyword arguments passed by :class:`TypeValidator` to
        :attr:`TypeValidator.check`
    validator : callable, optional
        A pre-processor for val, a callable that accepts ``(val, **validator_dict)``
        and returns val converted appropriately.
    validator_dict : dict, optional
        keyword arguments passed to validator function
    """
    __slots__ = ('type_', 'check', 'write', 
                 'node_prefix', 'node_repr', 'node_read', 'node_check',
                 'data_proc', 'ckwargs', 'validator', 'validator_dict')
    _defaults = ImDict(node_prefix=None, node_repr=None, node_read=None, node_check=None,
                       data_proc=_make_emptydict, ckwargs=_emptydict, 
                       validator=lambda:_return_first, validator_dict=_emptydict)
    _required = ('type_', 'check', 'write')
    _type_map:ClassVar[FixedDict[type,"TypeValidator"]] = FixedDict()
    _node_prefixes:ClassVar[FixedDict[str,"TypeValidator"]] = FixedDict()
    _node_types:ClassVar[FixedDict[type,"TypeValidator"]] = FixedDict()
    _grouptypes:ClassVar[FixedDict[str,Callable]] = FixedDict()
    _repstr_regex:ClassVar[re.Pattern] = re.compile(r'__(?P<type>.+?)__(?P<value>.+)')

    type_:type
    check:Callable
    write:Callable
    node_prefix:str
    node_repr:Callable[[Hashable],str]
    node_read:Callable[[str],Hashable]
    node_check:Callable[[Hashable],bool]
    data_proc:Callable
    ckwargs:dict
    validator:Callable
    validator_dict:dict

    def __post_init__(self):
        if self.type_ not in self._type_map:
            self._type_map[self.type_] = self
            if self.node_prefix is not None:
                if not callable(self.node_repr):
                    raise ValueError("must specify node_repr with node_prefix")
                if not callable(self.node_read):
                    raise ValueError("must specify node_read with node_prefix")
                self._node_prefixes[self.node_prefix] = self
                self._node_types[self.type_] = self

    def __hash__(self):
        return _const_hash(tuple((k, _const_hash(_tuple_arr_tdct(getattr(self, k)))) 
                                 for k in self.__slots__))

    def __eq__(self, other):
        return all(_tuple_arr_tdct(getattr(self, k)) == _tuple_arr_tdct(getattr(other, k)) for k in self.__slots__)

    def __call__(self, data_proc:Callable=None, validator:Callable=None, validator_dict=None, **kwargs):
        """
        Generate new sub-type of TypeValidator, which allows setting additional
        limits on values used.

        Parameters
        ----------
        data_proc : Callable, optional
            Function taking current build (all values specified earlier in slots
            order) of data, and returning dictionary of additional keyword arguments
            to be passed to validator and check function. The default is None.
        validator : Callable, optional
            Function taking value and copy of limits specified in keyword arguments
            used and return value or raises error, including those produced by
            data_proc. The default is None.
        **kwargs : TYPE
            Additional limitations passed to the check function of validator.

        Returns
        -------
        TypeValidator
            New TypeValidator which implements additional checks specified.

        """
        ckwargs = self.ckwargs.copy()
        ckwargs.update(kwargs)
        data_proc = self.data_proc if data_proc is None else data_proc
        validator = self.validator if validator is None else validator
        validator_dict = self.validator_dict if validator_dict is None else validator_dict
        return TypeValidator(self.type_, self.check, self.write,
                             self.node_prefix, self.node_repr, self.node_read, self.node_check,
                             data_proc, ckwargs, validator, validator_dict)

    def check_val(self, val:Any, *predata, **kwargs):
        r"""
        Verify that a value is valid for the given attribute. Can use second
        argument for further validation if data_proc is specified.

        Parameters
        ----------
        val : Any
            Value should be subtype of TypeValidator type\_ attribute.
        *predata : 
            additional arguments fed to data_proc.

        Raises
        ------
        ValidatorConstructionError
            Too many arguments passed to function
        TypeError
            Invalid value.

        Returns
        -------
        Any
            Validated value, should be immutable version of val.

        """
        ckwargs = self.ckwargs.copy()
        ckwargs.update(kwargs)
        ckwargs.update(self.data_proc(*predata, **self.validator_dict))
        val = self.validator(val, **ckwargs)
        return self.check(val, **ckwargs)

    @classmethod
    def convert_type(cls, type_:Union[type,"TypeValidator"])->"TypeValidator":
        r"""
        Get the approriate type validator for a given type.

        Parameters
        ----------
        type_ : type
            Type for finding the given validator.

        Raises
        ------
        ValueError
            type\_ does not have a corresponding TypeValidator.

        Returns
        -------
        TypeValidator
            TypeValidator for the given type.

        """
        if isinstance(type_, TypeValidator):
            return type_
        if type_ in cls._type_map:
            return cls._type_map[type_]
        ranks = list()
        for cls_, tv in cls._type_map.items():
            if issubclass(type_.__origin__ if isinstance(type_, types.GenericAlias) else type_, cls_):
                try:
                    idx = type_.__mro__.index(cls_)
                except ValueError:
                    idx = -1
                ranks.insert(idx, tv)
        if ranks:
            return ranks[0]
        raise ValueError(f"unregistered type {type_}")

    @classmethod
    def check_any(cls, val:Any, predata=None, type_=None, **kwargs)->Hashable:
        """
        Check if a value can be written by any of the existing :class:`TypeValidator`
        objects.

        Parameters
        ----------
        val : Any
            Value to check/convert into hashable writable type.
        predata : TYPE, optional
            Additional arguments to pass to :meth:`TypeValidator.check_val`. The default is None.
        type_ : TYPE, optional
            DESCRIPTION. The default is None.
            
        **kwargs : Any
            Values passed to check_val function of selected type

        Returns
        -------
        Hashable
            val converted to a type available in registered TypeValidators.

        """
        tv = cls.convert_type(type(val)) if type_ is None else cls.convert_type(type_)
        pdtup = tuple() if predata is None else predata
        pdtup = pdtup if isinstance(pdtup, tuple) else (pdtup, )
        return tv.check_val(val, *pdtup, **kwargs)

    @classmethod
    def write_any(cls, group:tb.Group, name:str, val:Any)->tb.Node:
        """
        Create a node in group with the name of name, storing the value of val.
        Assumes that val has already been passed through a TypeValidator.check_any.

        Parameters
        ----------
        group : tb.Group
            Group in which to create node in HDF5 file.
        name : str
            Name to give new node.
        val : Any
            Value to save in Node.

        Returns
        -------
        tb.Node
            Node created storing val.

        """
        return cls.convert_type(type(val)).write(group, name, val)

    @classmethod
    def val_to_nodename(cls, val:Any)->str:
        """
        Get node name string representation of val.

        Parameters
        ----------
        val : Any
            value to be saved in a nodename.

        Raises
        ------
        TypeError
            val cannot be representated as a nodename string.

        Returns
        -------
        str
            string of nodename representation of val.

        """
        tv = cls.convert_type(type(val))
        if tv.node_prefix is None:
            raise TypeError(f"{type(val)} cannot be used in a nodename")
        return f'{tv.node_prefix}_{tv.node_repr(val)}'

    @classmethod
    def read_nodename(cls, name:str)->Hashable:
        """
        Read a node name as a python object.

        Parameters
        ----------
        name : str
            name of a HDF5 node.

        Raises
        ------
        KeyError
            Nodename not readable as python object.

        Returns
        -------
        Hashable
            Nodename converted to python-value.

        """
        for prefix, ht in cls._node_prefixes.items():
            if name.startswith(f'{prefix}_'):
                return ht.node_read(name.split(f'{prefix}_', 1)[1])
        raise KeyError(f"{name} does not start with recognized type identifier prefix")

    @classmethod
    def _get_subgroup(cls, dct:dict, args:tuple[str,...,Any], prev:tuple[str]=None)->dict|Any:
        """Recursion algorithm to get gruoptype"""
        prev = tuple() if prev is None else prev
        arg, args = args[0], args[1:]
        if arg not in dct:
            raise KeyError("{arg} not present in {prev}")
        return cls._get_subgroup(dct[arg], args, prev=prev+(arg, )) if args else dct[arg]

    @classmethod
    def get_subgroup(cls, *args:str)->dict[str:Any]:
        """
        Get the dictionay/TypeValidator/type of particular group-type.

        Parameters
        ----------
        *args : str
            Nested Group identifies.

        Returns
        -------
        dict[str:Any]
            Subgroup dictionary of identified group-type.

        """
        return cls._get_subgroup(cls._grouptypes, args)

    @classmethod
    def is_grouptype(cls, *args:str)->bool:
        """
        Determine if nested keys in args are a registered group

        Parameters
        ----------
        *args : str
            Nested group-type names.

        Returns
        -------
        bool
            If the input keys represent a grouptype

        """
        try:
            subgroup = cls.get_subgroup(args)
        except KeyError:
            return False
        return not isinstance(subgroup, FixedDict)

    @classmethod
    def _recurse_subgroups(cls, dct:dict, final:Callable, args:tuple[str,...,Any], curgroup:tuple[str,...]=None)->Any:
        """
        Recursion function down subgroups of dictionary. dct is current dictionary,
        final is a finalizer function, not used until at last step of recursion,
        args is the remaining nested keys to add, and curgroup is previous keys.
        """
        curgroup = tuple() if curgroup is None else curgroup
        arg, args = args[0], args[1:]
        if not isinstance(arg, str):
            raise ValueError("all but final argument must be str")
        if arg in dct:
            return cls._recurse_subgroups(dct[arg], final, args, curgroup=curgroup+(arg, ))
        return final(dct, arg, args, curgroup)

    @classmethod
    def _init_subgroup(cls, dct:dict, key:str, args:tuple[str,...,Any], curgroup)->dict[str,Any]:
        """Terminating function for registering grouptype- adds FixedDict with __call__ method assiged as last argument"""
        if len(args) != 1:
            raise ValueError(f'args too deep, must first assign callable to {curgroup}')
        arg = args[0]
        if not callable(arg):
            raise TypeError('terminal value in args must be callable')
        dct[key] = FixedDict()
        dct[key]['__call__'] = arg
        return dct

    @classmethod
    def _init_subclass(cls, dct:dict[str:Any], key:str, args:tuple[str], curgroup:tuple[str])->dict:
        if len(args) != 1:
            raise ValueError(f'args too deep, must first assign callable to {curgroup[:-1]}')
        arg = args[0]
        dct[key] = arg
        return dct

    @classmethod
    def register_grouptype(cls, *args:str|Any)->dict[str:Any]:
        r"""
        Register a *category* of typevalidators. The first n-1 args are nested
        categories, the last must be new, and the final argument is a read
        function for the category of grouptype.

        Parameters
        ----------
        *args : str, ..., Callable[[tb.Group,dict[str:Any],Any,...]Any]
            Nested categories of groups, final must be Callable with signature
            ``func(group:tb.Group, tv:dict[str,Any], *args:str)``
            where group is a tb.Group to be read from HDF5 file, tv is the
            dictionary of current TypeValidator sub-group, and \*args is the 
            '-' split strings of the group title.

        Returns
        -------
        dict[str:Any]
            Dictionary of grouptype registered.

        """
        if not args:
            return
        return cls._recurse_subgroups(cls._grouptypes, cls._init_subgroup, args)

    @classmethod
    def register_groupclass(cls, *args:str|Callable)->dict[str:Any]:
        """
        Register a class under group type. All args but last are nested group
        categories, final arg is Callable linked to second to last arg, which
        is the key in the dictionary of the third to last arg, and this
        third-to-last arg has a function that can inspect the sub-args to get
        the relevant callable.
        
        The concept is as follows: 
        
        >>> TypeValidator.register_groupclass('GroupType', 'subtype', func)
        
        Where grouptype is a general category of types that share common processing
        features, subtype identifies the specific type, and func is a final
        function the GroupType call func can access to get subtype-specific features.

        Parameters
        ----------
        *args : str, ... Callable
            Nested categories of group, and final "finalizer".

        Returns
        -------
        Callable
            Callable.

        """
        if not args:
            return
        return cls._recurse_subgroups(cls._grouptypes, cls._init_subclass, args)

    @classmethod
    def read_any(cls, group:tb.Node)->Any:
        """
        Read HDF5 group as python object. 
        Inspects the title, uses __call__ key
        from the first str of title after splitting by '-'.

        Parameters
        ----------
        group : tb.Node
            Node to load.

        Returns
        -------
        Any
            Python object represented by group.

        """
        tvargs = (group._v_title if isinstance(group._v_title, str) else group._v_title.decode()).split('-')
        tv = cls._grouptypes[tvargs[0]]
        return tv['__call__'](group, tv, *tvargs[1:])

    @property
    def has_node_repr(self)->bool:
        """If the TypeValidator object has a node name string representation"""
        return self.node_repr is not None

    @classmethod
    def val_has_node_repr(cls, val:Any)->bool:
        """
        If val has a node name string representation

        Parameters
        ----------
        val : Any
            Python object to check.

        Returns
        -------
        bool
            ``True`` if val has node name representation, ``False`` otherwise.

        """
        tv = cls.convert_type(type(val))
        if tv.has_node_repr:
            if tv.node_check is None:
                return True
            return tv.node_check(val)
        return False


class ValidatorConstructionError(ValueError):
    """Error indicating that validator has some error"""
    pass


###############################################################################
### Fucntions for defining TypeDefs
###############################################################################
def _type_name(val:type)->str:
    """Convert a type into a string that can be saved"""
    return f'{val.__module__}.{val.__name__}'


def _val_type_name(val:Any)->str:
    tp = TypeValidator.convert_type((type(val))).type_
    return _type_name(tp)


def read_byteslike(group:tb.Group, dct:dict, *args)->Any:
    convert = dct.get(args[0], _return_first)
    if isinstance(convert, dict):
        dct = convert
        convert = convert['__call__']
    return convert(group.read(), dct, *args[1:])


def write_byteslike(group:tb.Group, name:str, val:Any, 
                    title_func:Callable[[Any],str]=_val_type_name, 
                    convert:Callable[[Any],Any]=_echo)->tb.Node:
    obj = convert(val)
    title = f'byteslike-{title_func((val))}'
    if isinstance(obj, np.ndarray):
        return group._v_file.create_carray(group, name, obj=obj, title=title)
    return group._v_file.create_array(group, name, obj, title=title)


TypeValidator.register_grouptype('byteslike', read_byteslike)


def register_byteslike(type_:type, check:Callable, 
                       read:Callable[[Any,dict,...],Any]=_return_first,
                       write:Callable[[Any],Union[bytes,np.ndarray]]=None,
                       node_prefix:str=None, node_repr:Callable[[Any],str]=None, 
                       node_read:Callable[[str],Any]=None, node_check:Callable[[Hashable],bool]=None,
                       name:str=None, title_func:Callable[[Any],str]=None)->TypeValidator:
    if name is None:
        TypeValidator.register_groupclass('byteslike', _type_name(type_), read)
        title_func = _val_type_name if title_func is None else title_func
    else:
        TypeValidator.register_grouptype('byteslike', name, read)
        if title_func is None:
            title_func = lambda val: name
        else:
            title_func = lambda val: f'{name}-{title_func(val)}'
    if write is None:
        if name is None:
            write = write_byteslike
        else:
            write = partial(write_byteslike, title_func=title_func)
    else:
        write = partial(write_byteslike, convert=write, title_func=title_func)
    if node_prefix is None:
        kwargs = dict()
    else:
        kwargs = dict(node_prefix=node_prefix, node_repr=node_repr, 
                      node_read=node_read)
        if node_check is not None:
            kwargs['node_check'] = node_check
    return TypeValidator(type_, check, write, **kwargs)


def _check_num(supertype:type, outtype:type, val:Number, mn:Number=-np.inf, mx:Number=np.inf, 
               isin:tuple[Number,...]|list[Number]|set[Number]=None, 
               notin:tuple[Number,...]|list[Number]|set[Number]=None,
               **kwargs)->int:
    """
    Function for number type TypeValidators, checks that val can be converted,
    and if so, convert to specified type
    
    Parameters
    ----------
    supertype : type
        valid types of val
    outtype : type
        type that the number will be
    val : number
        Number to check
    mn : number, optional
        Minimum value, Default is -inf
    mx : number, optional
        Maximum value, Default is inf
    isin : Sequence[number] | set[number], optional
        Set-like of values, if not None, val must be equal to one element.
        Default is None
    notin : Sequence[number] | set[number], optional
        Set-like of values, if not None, val must not be equal to any element.
        Default is None
    
    """
    if isinstance(val, np.ndarray) and val.size == 1:
        val = val.reshape(1)[0]
    if not np.issubdtype(type(val), supertype):
        raise TypeError(f"{val} is not an {supertype}")
    if val < mn or val > mx:
        raise ValueError(f"{val} out of valid range")
    if isin is not None and val not in isin:
        raise ValueError(f"invalid value {val}, must be one of the following:{isin}")
    return outtype(val)


def make_check_numeric(type_:type, supertype:Union[tuple[type,...], type])->Callable:
    """
    Convenience function for preparing a new number check_function.
    
    **For use with :class:`TypeValidator`**
    
    Parameters
    ----------
    type_ : type
        The type, or casting function that the value should be
    supertype : type
        The supertype (type wich val must be instance of)    
    """
    return partial(_check_num, supertype, type_)


def nhex(val:int)->str:
    """
    Hex string where negative sign replaced with n

    Parameters
    ----------
    val : int
        Value to turn into hex string.

    Returns
    -------
    str
        hex string of val, negative sign replaed with n.

    """
    return hex(val).replace('-','n')


def rnhex(val:str)->int:
    """
    Inverse of :func:`nhex`, takes string produced by :func:`nhex` and returns
    cooresponding int

    Parameters
    ----------
    val : str
        Hex value to convert to int.

    Returns
    -------
    int
        Integer value of hex string.

    """
    return int(val.replace('n','-'), 0)


def node_repr_int(val:int)->str:
    """
    Get the string representation of an integer for node name string.
    
    **For use with** :class:`TypeValidator`

    Parameters
    ----------
    val : int
        value to get string representation of.

    Returns
    -------
    str
        string representation of an integer for node name string.

    """
    return nhex(val)


def node_read_int(val:str)->int:
    """
    Read node name as int
    
    **For use with** :class:`TypeValidator`

    Parameters
    ----------
    val : str
        node name string of int.

    Returns
    -------
    int
        int python object of val.

    """
    return rnhex(val)


def node_repr_float(val:float)->str:
    """
    Get the string representation of an float for node name string.
    
    **For use with :class:`TypeValidator`**

    Parameters
    ----------
    val : int
        value to get string representation of.

    Returns
    -------
    str
        string representation of a float for node name string.

    """
    num, dem = val.as_integer_ratio()
    return f'{nhex(num)}_{nhex(dem)}'


def node_read_float(val:str)->float:
    """
    Read node name as float
    
    **For use with** :class:`TypeValidator`

    Parameters
    ----------
    val : str
        node name string of int.

    Returns
    -------
    float
        float python object of val.

    """
    num, dem = val.split('_')
    return rnhex(num) / rnhex(dem)


_bool_str_map = {True:'True', False:'False'}
_str_bool_map = {v:k for k, v in _bool_str_map.items()}


def node_repr_bool(val:bool)->str:
    """
    Get the string representation of bool for node name string.
    
    **For use with :class:`TypeValidator`**

    Parameters
    ----------
    val : bool
        value to get string representation of.

    Returns
    -------
    str
        string representation of bool for node name string.

    """
    return _bool_str_map[val]


def node_read_bool(val:str)->bool:
    """
    Read node name as bool
    
    **For use with** :class:`TypeValidator`

    Parameters
    ----------
    val : str
        node name string of bool.

    Returns
    -------
    bool
        bool python object of val.

    """
    return _str_bool_map[val]


check_int = make_check_numeric(int, np.integer)
check_float = make_check_numeric(float, np.number)
check_bool = make_check_numeric(bool, np.bool_)

TV_int = register_byteslike(int, check_int, node_prefix='int', node_repr=node_repr_int, node_read=node_read_int)
TV_float = register_byteslike(float, check_float, node_prefix='float', node_repr=node_repr_float, node_read=node_read_float)
TV_bool = register_byteslike(bool, check_bool, node_prefix='bool', node_repr=node_repr_bool, node_read=node_read_bool)


def check_str(val:str, isin:Sequence[str]=None, startswith:str=None, endswith:str=None, 
              pattern:re.Pattern=None, allow_empty:bool=False, **kwargs)->str:
    """
    **For use with :class:`TypeValidator`**
    
    Check if val is string, validated by kwargs.

    Parameters
    ----------
    val : str
        value to check/convert.
    isin : Sequence[str], optional
        Sequence of strings, if not None, then val must be one of the strings
        in ``isin``. The default is None.
    startswith : str, optional
        val must start with startswith if not None. The default is None.
    endswith : str, optional
        val must end with endswith if not None. The default is None.
    pattern : re.Pattern, optional
        If not None, ``pattern.match(val)`` must a value with a ``True``
        "truthiness". The default is None.
    allow_empty : bool, optional
        If true, then empty strings allowed regardless of other kwargs. 
        The default is False.
    **kwargs : TYPE
        DESCRIPTION.

    Raises
    ------
    TypeError
        val is not str.
    ValueError
        Bad value.

    Returns
    -------
    str
        val, but ensured to be str.

    """
    if not np.issubdtype(type(val), np.str_):
        raise TypeError(f'{val} is not a str')
    val = str(val)
    if allow_empty and not val:
        return val
    if isin is not None and val not in isin:
        raise ValueError(f"{val} invalid option, must be one of {isin}")
    if startswith is not None and not val.startswith(startswith):
        raise ValueError(f"{val} must start with {startswith}")
    if endswith is not None and not val.endswith(endswith):
        raise ValueError(f"{val} must end with {endswith}")
    if pattern is not None and not pattern.match(val):
        raise ValueError(f'{val} does not match predefined pattern')
    return val


def dread_str(arr:bytes, dct:dict)->str:
    """
    Direct read of string

    Parameters
    ----------
    arr : bytes
        Bytes to convert to sting.
    dct : dict
        Needed for type-validator mapping, unused.

    Returns
    -------
    str
        String interpretation of bytes, read from HDF5 file.

    """
    return arr.decode()


def dwrite_str(val:str)->bytes:
    """
    Direct write, turns str into bytes.

    Parameters
    ----------
    val : str
        string to write to HDF5 bytes array.

    Returns
    -------
    bytes
        bytes to be written to HDF5 array.

    """
    return val.encode()


attr_regex = re.compile(r'^[A-Za-z_][\w_]*$')


def node_check_str(val:str)->bool:
    """TypeValidator node-check function for TV_str"""
    return bool(attr_regex.match(val))


TV_str = register_byteslike(str, check_str, dread_str, dwrite_str, 'str', _echo, _echo, node_check_str)
TV_attrstr = TV_str(pattern=attr_regex)
TV_attrstr_allow_empty = TV_str(pattern=attr_regex, allow_empty=True)


def check_bytes(val:bytes, **kwargs)->bytes:
    """
    **For use with :class:`TypeValidator`**
    
    Check if val is bytes.


    Parameters
    ----------
    val : bytes
        val to be checked/converted.
    **kwargs : TYPE
        all kwargs ignored.

    Raises
    ------
    TypeError
        val cannot be read as bytes.

    Returns
    -------
    bytes
        val, ensured to be bytes.

    """
    try:
        val = bytes(val)
    except Exception as e:
        raise TypeError("cannot convert {type(val)} to bytes") from e
    return val
       
 
TV_bytes = register_byteslike(bytes, check_bytes)


def init_write_group(group:tb.Group, name:str, typename:str)->tuple[tb.File, tb.Group]:
    """
    Convenience function for ``write_`` functions, creates a group names ``name``
    inside of ``group`` and adds a ``Type_`` array set to the specified ``typename``
    and returns the group. Called at beginnig of non-direct ``write_`` functions.

    Parameters
    ----------
    group : tb.Group
        Group inside which to create group to contain HDF5 representation of 
        a given type.
    name : str
        name of new group.
    typename : str
        string name of new group.

    Returns
    -------
    file : tb.File
        tables.File in which group was created.
    group : tb.Group
        newly created group, where additional information should be written.

    """
    file = group._v_file
    group = file.create_group(group, name, title=typename)
    return file, group


def check_none(val:None, *args:Any, **kwargs:Any)->None:
    """TypeValidator check function for TV_none"""
    if val is not None:
        raise TypeError(f"None must be None, not {val}")
    return None


def dread_none(arr:bytes, dct:dict)->None:
    """Direct read for reading None type array"""
    if arr != 'None'.encode():
        raise ValueError("malformed None group")
    return None


def dwrite_none(val:None)->bytes:
    """Direct write function for None value"""
    if val is not None:
        raise TypeError("trying to record non-None object as None")
    return 'None'.encode()


TV_none = register_byteslike(type(None), check_none, dread_none, dwrite_none)


def check_arbsequence(type_:type, typecall:Callable[[Sequence],Sequence], vals:Sequence, 
                       typedefs:Union[type,TypeValidator,Sequence[Union[type,TypeValidator]]]=None,
                       minsize:int=None, maxsize:int=None, **kwargs)->Sequence:
    r"""
    Internal funciton for checking Sequence types, used in TypeValidator objects.
    The true type is provided by the first argument, a Callable that takes the
    type to convert to correct type is provides as second argument. Expected to
    be used in a partial function which specifies the first two already named
    arguments.

    Parameters
    ----------
    type_ : type
        Type object for isinstance calls that vals should match/be converted to.
    typecall : Callable[[Sequence],Sequence]
        Callable that takes val and converts to type, often same as type\_.
    vals : Sequence
        Value to be check/turned into given sequence type.
    typedefs : Union[type,TypeValidator,Sequence[Union[type,TypeValidator]]], optional
        DESCRIPTION. The default is None.
    minsize : int, optional
        minimum length of sequence. The default is None.
    maxsize : int, optional
        Maximum lenght of object. The default is None.
    **kwargs : Any
        Unused, but needed so that sub-typedefs can take additional kwargs.

    Raises
    ------
    ValueError
        Sequence is either too short or too long.
    TypeError
        type\_ is not hashable, and therefore invalid for storing with TypeValidator.

    Returns
    -------
    Sequence
        val converted to type\_ by typecall.

    """
    type_ = type(vals) if type_ is None else type_
    if minsize is not None and minsize > len(vals):
        raise ValueError(f"sequnce is not long enough, must have at least {minsize} values")
    if maxsize is not None and maxsize < len(vals):
        raise ValueError(f"sequnce is too long, must have maximum of {maxsize} values")
    if not issubclass(type_, Hashable):
        raise TypeError("type_ must be hashable")
    if typedefs is None or isinstance(typedefs, (type, TypeValidator)):
        typedefs = repeat(typedefs)
    elif minsize is not None and minsize > len(typedefs):
        raise ValidatorConstructionError("bad TypeValidator, specified minsize and sequence-like typedefs")
    elif maxsize is not None and minsize < len(typedefs):
        raise ValidatorConstructionError("bad TypeValidator, specified maxsize and sequence-like typedefs")
    elif len(typedefs) != len(vals):
        raise ValueError("Sequence is of incorrect size")
    out, diff = list(), False
    for i, (typedef, val) in enumerate(zip(typedefs, vals)):
        temp = TypeValidator.check_any(val, type_=typedef)
        out.append(temp)
        diff = True if temp is not val else diff
    out = typecall(out) if diff or not isinstance(vals, type_) else vals
    return out
    

def write_arbsequence(group:tb.Group, name:str, val:Sequence):
    """
    **To be used with :class:`TypeValidator`**
    
    Write any sequence type to a group

    Parameters
    ----------
    group : tb.Group
        Outer (existing) group in which to create new group.
    name : str
        Name for new group, which will contain HDF5 representation of sequence.
    val : Sequence
        Sequence to write to HDF5 group.

    Returns
    -------
    tb.group
        group containing HDF5 representation.

    """
    file, group = init_write_group(group, name, f'Sequence-{_type_name(type(val))}')
    for i, v in enumerate(val):
        TypeValidator.write_any(group, f'data_{i}', v)
    return group


def read_arbsequence(group:tb.Group, dct:dict, tp:str)->Sequence:
    r"""
    **For use with :class:`TypeValidator`**
    
    Read group node representing a Sequence as a python object.

    Parameters
    ----------
    group : tb.Group
        group containing sequence representation.
    dct : dict
        dictionary containing mapping of values in SequenceType\_ to function
        creating correct type from list. This comes 
        from ``TypeValidator._grouptypes['Sequence']``

    Returns
    -------
    Sequence
        python object HDF5 group represents.

    """
    typecall = dct[tp]
    out = [TypeValidator.read_any(g) for g in _iter_tbgroup_numeric(group, 'data_')]
    out = typecall(out)
    return out


TypeValidator.register_grouptype('Sequence', read_arbsequence)


def make_arbsequence(type_:type, typecall:Callable[[list],Sequence]=None)->Callable:
    """
    Create a ``check_`` function for use with :class:`TypeValidator` of a specific
    Sequence type

    Parameters
    ----------
    type_ : type
        python type object of Sequence type (must work as second argument to isinstance).
    typecall : Callable[[list],Sequence], optional
        Function for creating sequence of type ``type_`` if None, use ``type_``. 
        The default is None.

    Returns
    -------
    Callable
        ``check_`` type function **for use with** :class:`TypeValidator`.

    """
    typecall = type_ if typecall is None else typecall
    TypeValidator.register_groupclass('Sequence', _type_name(type_), typecall)
    return partial(check_arbsequence, type_, typecall)


def node_repr_arbsequence(val:tuple)->str:
    """
    Create node name string representation of sequence of objects that all must
    also have node name string representations

    Parameters
    ----------
    val : tuple
        tuple of objects with node name string representations.

    Returns
    -------
    str
        node name string representation of val.

    """
    return '__'.join(TypeValidator.val_to_nodename(v) for v in val)


def node_read_arbsequence(typeconverter:Callable, val:str)->Sequence[Hashable]:
    """
    Read HDF5 nodename as sequence, designed to be used in partial function
    specifying sequence type.
    
    To be used inside of node read function with :class:`TypeValidator`

    Parameters
    ----------
    typeconverter : Callable
        Convert generator into sequence type
    val : str
        inpute node name string.

    Returns
    -------
    Sequence[Hashable]
        Sequence of python objects.

    """
    return typeconverter(TypeValidator.read_nodename(v) for v in val.split('__'))


def node_check_arbsequence(val:Sequence[Hashable])->bool:
    """Generallized Sequence-like TypeValidator node-name check function"""
    return all(TypeValidator.val_has_node_repr(v) for v in val)


check_tuple = make_arbsequence(tuple)
node_read_tuple = partial(node_read_arbsequence, tuple)
check_frozenset = make_arbsequence(frozenset)
node_read_frozenset = partial(node_read_arbsequence, frozenset)

TV_tuple = TypeValidator(tuple, check_tuple, write_arbsequence, 'tuple', node_repr_arbsequence, node_read_tuple, node_check_arbsequence)
TV_frozenset = TypeValidator(frozenset, check_frozenset, write_arbsequence, 'frozenset', node_repr_arbsequence, node_read_frozenset, node_check_arbsequence)


def _check_array_dims(val:np.ndarray, dims:tuple[Union[int,slice],...], 
                      mindim:Union[None,int], maxdim:Union[None,int], square:bool)->None:
    """Check dimensions of array are valid"""
    if dims is not None and not _dimscompare(val.shape, dims):
        raise ValueError(f"array or sequecne has incorrect dimensions: got {val.shape}, expected {dims}")
    if mindim is not None and mindim > val.ndim:
        raise ValueError(f'matrix must be at least {mindim}D')
    if maxdim is not None and maxdim < val.ndim:
        raise ValueError(f'matrix must be not more than {maxdim}D')
    if square and any(val.shape[0] != s for s in val.shape[1:]):
        raise ValueError("array is not square")
    

def _check_array_vals(val:np.ndarray, mn:np.number=None, mx:np.number=None)->None:
    """Check array values are in range"""
    if mn is not None and np.any(mn > val):
        raise ValueError(f'one or more values less than minimum value of {mn}')
    if mx is not None and np.any(mx < val):
        raise ValueError(f'one or more values greater than maximum value of {mx}')


def _check_superdtype(val:np.ndarray, superdtype:type)->None:
    """Check array is of appropritate type to be converted"""
    if superdtype is not None and not np.issubdtype(val.dtype, superdtype):
        raise TypeError(f'must be array of type {superdtype}')
    

def _check_objectdtype(val:np.ndarray[np.object_], typedefs:Union[TypeValidator,np.ndarray[TypeValidator]])->None:
    """Check the arrays within an object array match typdefs"""
    if typedefs is None or isinstance(typedefs, (type, TypeValidator)):
        typedefs = np.array([typedefs], dtype=np.object_)
    typedefs = np.broadcast_to(typedefs, val.shape)
    out = np.empty(val.shape, dtype=np.object_)
    diff = False
    for ij in product(*(range(i) for i in val.shape)):
        out[ij] = TypeValidator.check_any(val[ij], type_=typedefs[ij])
        diff = True if out[ij] is not val[ij] else diff
    if any(not isinstance(v, np.ndarray) for v in out.flat):
        warnings.warn("Object array of non-array objects, will not be able to save to HDF5 file")
    else:
        dtype = out.reshape(-1)[0].dtype
        if any(v.dtype != dtype for v in out.flat):
            warnings.warn("Object array of differently typed numpy arrays, will not be able to save to HDF5 file")
    return out if diff else val


def check_array(val:np.ndarray, superdtype:Union[np.dtype,type]=None, dtype:np.dtype=None, 
                mn:np.number=None, mx:np.number=None, square:bool=False,
                mindim:int=None, maxdim:int=None, dims:tuple[Union[int,slice], ...]=None, 
                typedefs:Union[type,TypeValidator,Sequence[Union[type,TypeValidator]]]=None, 
                **kwargs)->np.ndarray:
    """
    Check function for TV_ndarray, check and if possible converts array to
    specifed dtype etc.

    Parameters
    ----------
    val : np.ndarray
        Array to check.
    superdtype : Union[np.dtype,type], optional
        Passed as second argument to np.issubdtype, allowable input dtypes. 
        The default is None.
    dtype : np.dtype, optional
        Final dtype of array, called with astype. The default is None.
    mn : np.number, optional
        Minimum allowable value any element in array. The default is None.
    mx : np.number, optional
        Maximum allowable value of any element in array. The default is None.
    square : bool, optional
        If ``True`` all dimensions of array must be the same size. The default is False.
    mindim : int, optional
        Minimum number of dimensions. The default is None.
    maxdim : int, optional
        Maximum number of dimensions. The default is None.
    dims : tuple[int|slice, ...], optional
        Definition of allowale size of each dimension. The default is None.
    typedefs : Union[type,TypeValidator,Sequence[Union[type,TypeValidator]]], optional
        **Only used if dtype is** ``np.object_``. Type definitions for each
        element of array.
        The default is None.
    **kwargs : TYPE
        DESCRIPTION.

    Raises
    ------
    TypeError
        Incompatible type to convert to array.
    ValueError
        One of values, (min/max) or dimensions does not match specifications.

    Returns
    -------
    np.ndarray
        val, with appropriate dtype.

    """
    try:
        v = np.asarray(val)
    except Exception as e:
        raise TypeError(f"cannot convert {type(val)} to a numpy array array") from e
    _check_superdtype(v, superdtype)
    _check_array_dims(v, dims, mindim, maxdim, square)
    if typedefs is not None:
        if dtype is not None and dtype != np.object_:
            raise ValueError("cannot use typdefs for non-object arrays")
        v = _check_objectdtype(v, typedefs)
    _check_array_vals(v, mn, mx)
    if dtype is not None:
        v = np.asarray(v, dtype=dtype)
    if v is val:
        v = v.copy()
    v.setflags(write=False)
    return v


def write_array(group:tb.Group, name:str, val:np.ndarray, **kwargs)->tb.Group:
    """
    Write function for TV_ndarray, Creates array ``val`` of name ``name`` in group.

    Parameters
    ----------
    group : tb.Group
        Outer (existing) group in which to create new group.
    name : str
        Name for new group, which will contain HDF5 representation of sequence.
    val : np.ndarray
        Array to write to group.
    **kwargs
        Unused

    Returns
    -------
    tb.Group
        Group where array is written.

    """
    if val.dtype != np.object_:
        return group._v_file.create_carray(group, name, obj=val, title='ndarray')
    group = group._v_file.create_vlarray(group, name, tb.Atom.from_dtype(val.reshape(-1)[0].dtype), 
                                         expectedrows=val.size,
                                         title='ndarray-'+'-'.join(str(i) for i in val.shape))
    for v in val.flat:
        group.append(v)
    return group


def read_array(group:tb.Group, dct:dict, *args)->np.ndarray:
    """
    Read function for TV_ndarray.
    
    .. note::
        
        Only used for reading object arrays, as other arrays can be directly
        written as arrays.

    Parameters
    ----------
    group : tb.Group
        Group in which array is located.
    dct : dict
        TypeValidator grouptype dictionary, unused.

    Returns
    -------
    out : np.ndarray
        numpy array of hdf5 group.

    """
    arr = group.read()
    if args:
        shape = tuple(int(s) for s in args)
        arr = np.array(arr, dtype=np.object_).reshape(shape)
    return arr


TypeValidator.register_grouptype('ndarray', read_array)
TV_ndarray = TypeValidator(np.ndarray, check_array, write_array)


def _tupledict_fromorder(val:Union[dict,tupledict,Sequence[tuple[str,Any]]], 
                         order:Sequence[str])->tupledict:
    """
    Convert val to tupledict with order of keys defined by order, keys in order may
    be skipped in val and will not appear, but keys unique to val will raise error
    """
    if not isinstance(val, (dict, tupledict)):
        if all((isinstance(v, (list, tuple) and len(v) == 2 and v[0] in order) for v in val)):
            val = tupledict.from_order(order, **{k:v for k, v in val})
        else:
            try:
                val = tupledict.from_order(order, *val)
            except Exception as e:
                raise TypeError(f"cannot read {val} as a tuple dict with keys {order}") from e
    else:
        try:
            val = tupledict.from_order(order, **{k:v for k, v in val.items()})
        except Exception as e:
            raise TypeError(f"cannot read {val} as a tuple dict with keys {order}") from e
    return val


def _get_tupledictargs(val:Any)->tupledict|Sequence[tuple[str,Any]]:
    """Convert val into an iteratable that will return tuples of key, value pairs"""
    if isinstance(val, tupledict):
        return val
    if isinstance(val, dict):
        if all(isinstance(k, str) for k in val.keys()):
            return tuple(val.items())
        return None
    if isinstance(val, (Sequence, set, frozenset)):
        if all(isinstance(v, Sequence) and len(v) == 2 and isinstance(v, str) for v in val):
            if all(c==1 for c in Counter((v[0] for v in val)).values()):
                return val
    return None
    


def _tupledict_fromany(val:tuple[str,Any]|dict|tupledict)->tupledict:
    """Convert value to tupledict"""
    if isinstance(val, tupledict):
        return val
    if isinstance(val, dict):
        return tupledict(**val)
    return tupledict(*val)


def check_tupledict(val:Union[tupledict,dict,Sequence[tuple[str,Any],...]], 
                    order:tuple[str,...]=None, required:set[str]=None, 
                    typedefs:Union[dict[str,TypeValidator],TypeValidator]=None, 
                    defaults:dict[str,Any]=None, **kwargs)->tupledict:
    """
    Check function for TV_tupledict, check/convert val to tupledict

    Parameters
    ----------
    val : tupledict | dict | Sequence[tuple[str,Any],...]
        Input value, to be converted to tupledict.
    order : tuple[str,...], optional
        Order of keys in tupledict. Default is None
    required : set[str], optional
        Keys that are required to be contained in tupledict
    typedefs : dict[str:TypeValidator] | TypeValidator | None, optional
        TypeValidator for each object in tupledict, if specified as TypeValidator
        instead of Sequence thereof, apply same TypeValidator to all. Can also
        specify as type, and internally converted to TypeValidator. Default is None.
    defaults : dict[str:Any]
        Default values, if key not present in val, fill with value in defaults.
        Default is None
    
    Raises
    ------
    ValidatorConstructionError
        typedefs incorrectly constructed
    ValueError
        missing required key
    
    Returns
    -------
    tupledict
        val, as tupledict

    """
    tdefs = _get_tupledictargs(typedefs)
    if order:
        if tdefs is not None:
            typedefs = {k:v for k, v in tdefs}
        elif isinstance(typedefs, Sequence):
            if len(typedefs) > len(order):
                raise ValidatorConstructionError("Too many typedefs specified given order")
            typedefs = {k:v for k, v in zip(order, typedefs)}
        elif isinstance(typedefs, (type, TypeValidator)):
            tvdefs = TypeValidator.convert_type(typedefs)
            typedefs = {k:tvdefs for k in order}
        elif typedefs is None:
            typedefs = dict()
        else:
            raise ValidatorConstructionError(f"typedefs is of invalid format, {typedefs}")
    else:
        if tdefs is not None:
            order = tuple(k for k, _ in tdefs)
            typedefs = {k:TypeValidator.convert_type(t) for k, t in tdefs}
        else:
            val = _get_tupledictargs(val)
            order = tuple(k for k, _ in val)
            if val is None:
                raise ValueError("Must specify order or typedefs if not specifying only values of tupledict")
            if isinstance(typedefs, (type, TypeValidator)):
                typedefs = {k:typedefs for k, _ in val}
            elif isinstance(typedefs, Sequence):
                if len(typedefs) != order:
                    raise ValueError("mismatched number of typedefs and expected values")
                typedefs = {k:td for k, td in zip(order, typedefs)}
            else:
                typedefs = dict()
    if isinstance(val, (dict, tupledict)):
        out = {k:TypeValidator.check_any(v, predata=val, type_=typedefs.get(k, None)) for k, v in val.items()}
    elif all(isinstance(v, Sequence) and len(v) == 2 for v in val):
        out = {k:TypeValidator.check_any(v, predata=val, type_=typedefs.get(k, None)) for k, v in val}
    elif order and len(val) <= len(order):
        out = {k:TypeValidator.check_any(v, predata=val, type_=typedefs.get(k, None)) for k, v in zip(order, val)}
    else:
        if order:
            raise ValueError("too many values to convert to expected tupledict specification")
        else:
            raise ValueError("no default order specified, cannot infer keys")
    out = tupledict.from_order(order, defaults_=defaults, **out)
    if required is not None and (err:= set(required) - set(out.keys())):
        raise ValueError(f'Missing required keys: {err}')
    return out


def write_tupledict(group:tb.Group, name:str, val:tupledict)->tb.Group:
    """
    Write for TV_tupledict writes a tupledict to HDF5 group.

    Parameters
    ----------
    group : tb.Group
        Group in which to create HDF5 representation.
    name : str
        Name of group that will represent val.
    val : type
        tupledict to store in HDF5 file

    Returns
    -------
    tb.Group
        Group in which HDF5 representation exists.

    """
    file, group = init_write_group(group, name, 'tupledict')
    file.create_array(group, 'keys', np.array(list(val.keys())))
    for k, v in val.items():
        TypeValidator.write_any(group, k, v)
    return group


def read_tupledict(group:tb.Group, dct:dict)->tupledict:
    """
    Read function for TV_tupledict, reads a tupledict object from HDF5 group

    Parameters
    ----------
    group : tb.Group
        Group to read type from.
    dct : dict
        TypeValidator grouptypes, dictionary of type name strings and
        their corresponding type, unused.

    Returns
    -------
    tupledict
        Python tupledict loaded from HDF5 file.

    """
    keys = (k.decode() for k in group.keys.read())
    return tupledict(*((k, TypeValidator.read_any(group[k])) for k in keys))


def node_repr_tupledict(val:tupledict)->str:
    """
    Create string representation of tupledict

    Parameters
    ----------
    val : tupledict
        tupledict to represent as string.

    Returns
    -------
    str
        string representation of tupledict.

    """
    return '__'.join(f'{k}_{TypeValidator.val_to_nodename(v)}' for k, v in val.items())


def node_read_tupledict(val:str)->tupledict:
    """
    Nodename read function for tupledict nodename

    Parameters
    ----------
    val : str
        Name of node being read.

    Returns
    -------
    tupledict
        tupledict of node-name.

    """
    return tupledict(*((sub.split('_',1)[0], TypeValidator.read_nodename(sub.split('_',1)[1])) 
                       for sub in val.split('__')))


TypeValidator.register_grouptype('tupledict', read_tupledict)
TV_tupledict = TypeValidator(tupledict, check_tupledict, write_tupledict, 'tupledict', node_repr_tupledict, node_read_tupledict)

_pycode_subtypes = dict()

PyCode = Union[Callable,type]


def check_PyCode(val:PyCode, subtype:Hashable=None, **kwargs)->PyCode:
    """
    Check function for PyCode, validate that val is PyCode

    Parameters
    ----------
    val : PyCode
        Registered PyCode object.
    subtype : Hashable, optional
        To which pycode subtype the PyCode must belong. The default is None.
    **kwargs 
        Ignored but necessary so TV_PyCode can have additional validators etc.

    Raises
    ------
    AttributeError
        Object cannot be stored as PyCode.
    ValueError
        Not a registerd PyCode function.

    Returns
    -------
    PyCode
        val.

    """
    if not hasattr(val, '__name__'):
        raise AttributeError("functions must have attribute __name__")
    mname = _type_name(val)
    pycodedct = TypeValidator.get_subgroup('byteslike', 'pycode')
    if mname not in pycodedct or val != pycodedct[mname]:
        raise ValueError(f"{val} not registered, perhaps need to import another module")
    if subtype is not None and val not in _pycode_subtypes[subtype]:
        raise ValueError(f"{val} is not valid PyCode subtype of {subtype}")
    return val


def dread_PyCode(arr:bytes, dct:dict, *args:str)->PyCode:
    """
    Direct-read function converts byes to PyCode object

    Parameters
    ----------
    arr : bytes
        Bytes of array in HDF5 file.
    dct : dict
        Dictionary of bytes to PyCode.
    *args : str
        ignored.

    Returns
    -------
    PyCode
        Python code object (usually function or class).

    """
    return dct[arr.decode()]


def dwrite_PyCode(val:PyCode)->bytes:
    """
    Direct-write function for PyCode, returns bytes-representation of PyCode.
    (based on name of PyCode)

    Parameters
    ----------
    val : PyCode
        Python code object.

    Returns
    -------
    bytes
        Bytes array to write to disk.

    """
    return _type_name(val).encode()


def node_repr_pycode(val:PyCode)->str:
    """
    Returns nodename string representation of PyCode

    Parameters
    ----------
    val : PyCode
        PyCode object to convert to node name string.

    Returns
    -------
    str
        Node name string of val.

    """
    module = val.__module__.replace('.', 'D_O_T')
    return f'MODULE_{module}_NAME_{val.__name__}'


def node_read_pycode(val:str)->PyCode:
    """
    Get PyCode that node name string represents

    Parameters
    ----------
    val : str
        node name string of PyCode.

    Returns
    -------
    PyCode
        Python object that string represents.

    """
    module, name = val.split('_NAME_')
    module = module.split('MODULE_', 1)[1].replace('D_O_T', '.')
    return TypeValidator.get_subgroup('byteslike', 'pycode')[f'{module}.{name}']


def node_check_pycode(val:PyCode)->bool:
    """
    Node-check function for PyCode object.

    Parameters
    ----------
    val : PyCode
        PyCode object to check.

    Returns
    -------
    bool
        Valid PyCode for writing in node-name.

    """
    if 'D_O_T' in val.__module__ or 'D_O_T' in val.__name__:
        return False
    return True


TV_PyCode = register_byteslike(Callable, check_PyCode, dread_PyCode, dwrite_PyCode, 'pycode', 
                               node_repr_pycode, node_read_pycode, node_check_pycode, name='pycode')


def register_PyCode(pycode:PyCode, subtype:Hashable=None, subval:Any=None):
    """
    Register (make new entry in dictionary of PyCode objects) a new python function
    or class as "PyCode".

    Parameters
    ----------
    pycode : PyCode
        Python class or function to be added to PyCode objects.
    subtype : Hashable, optional
        Usually a string, a category to which multiple PyCode objects can belong. 
        Use if there are several objects that belong to a group, giving them the
        same subtype.
        The default is None.
    subval : Any, optional
        Value to store in PyCode suptypes. The default is None.

    Raises
    ------
    ValueError
        pycode is invalid for use as PyCode object.

    """
    TypeValidator.register_groupclass('byteslike', 'pycode', _type_name(pycode), pycode)
    if subtype is not None:
        if subtype not in _pycode_subtypes:
            _pycode_subtypes[subtype] = dict()
        if pycode.__module__ != '__main__' and pycode in _pycode_subtypes[subtype]:
            raise ValueError(f"cannot registerer {pycode}")
        _pycode_subtypes[subtype][pycode] = subval


def inpycodesubtype(subtype:Hashable, pycode:PyCode)->bool:
    """Check if pycode belongs to subtype"""
    return pycode in _pycode_subtypes[subtype]


def get_pycode_subval(subtype:Hashable, pycode:PyCode, default=None)->Any:
    """Get subval of registered pycode object"""
    if inpycodesubtype(subtype, pycode):
        return _pycode_subtypes[subtype][pycode]
    return default


register_PyCode(_echo)


def check_type(val:type, subclass:type=None, warn:bool=False)->type:
    """
    Check function for TV_type, ensure val is a type.

    Parameters
    ----------
    val : type
        object of type type.
    subclass : type, optional
        type of which val must be a subclass of. The default is None.
        
    warn : bool, optional
        Warn if subclass not registered.

    Raises
    ------
    TypeError
        val is not a type or appropriate subclass of subclass.

    Returns
    -------
    type
        same as val.

    """
    if not isinstance(val, type):
        raise TypeError(f'{val} is not a type object')
    if subclass is not None and not issubclass(val, subclass):
        raise TypeError(f'{val} is not subclass of {subclass}')
    if warn and all(val is not v for v in TypeValidator.get_subgroup('byteslike', 'type').values()):
        warnings.warn(f"{val} not registered type, will not be able to save to HDF5 file")
    return val


def dread_type(arr:bytes, dct:dict)->type:
    """
    Direct read function for type. Interprets type HDF5 bytes array as python type.

    Parameters
    ----------
    arr : bytes
        HDF5 bytes array.
    dct : dict
        Mapping of bytes to python type.

    Returns
    -------
    type
        Type represented by bytes in HDF5 file.

    """
    return dct[arr.decode()]


def dwrite_type(val:type)->bytes:
    """
    Write for TV_type, writes a type to HDF5 group.

    Parameters
    ----------
    group : tb.Group
        Group in which to create HDF5 representation.
    name : str
        Name of group that will represent val.
    val : type
        Store HDF5 object, places 'type' array in group, type array is bytes,
        with value of ``f'{val.__module__}.{val.__name__}'.encode()``.

    Returns
    -------
    tb.Group
        Group in which HDF5 representation exists.

    """
    return _type_name(val).encode()


def register_type(type_:type)->None:
    """
    Allow type to by recorded in HDF5 gruop.
    
    .. note::
        
        This is for storing ``type`` objects, not instances of types.

    Parameters
    ----------
    type_ : type
        Type to add to those available to store in HDF5 file.

    """
    TypeValidator.register_groupclass('byteslike', 'type', _type_name(type_), type_)


TV_type = register_byteslike(type, check_type, dread_type, dwrite_type, name='type')

register_type(int)
register_type(float)
register_type(bool)
register_type(str)
register_type(tuple)
register_type(np.ndarray)


def _typewithnodename(val:type, **kwargs)->type:
    """Validator for TV_typewithnodename, which derives from TV_type.
    ensures instances of type can be recorded as node names"""
    if TypeValidator.convert_type(val).node_prefix is None:
        raise TypeError(f"{val} does not have a node name representation")
    return val


TV_typewithnodename = TV_type(validator=_typewithnodename)


def check_dtype(val:np.dtype, size:int=None, kinds:tuple[str,...]=None, **kwargs)->np.dtype:
    """
    Check function for TV_dtype, ensures value is converted to dtype (``calls np.dtype(val)``)

    Parameters
    ----------
    val : np.dtype
        DESCRIPTION.
    size : int, optional
        Number of bytes in dtype. The default is None.
    kinds : tuple[str,...], optional
        Valid kinds (dtype.kind values). The default is None.
    
    Raises
    ------
    ValueError
        dtype does not match size or kinds.

    Returns
    -------
    np.dtype
        val as a dtype.

    """
    dtype = np.dtype(val)
    if size is not None and dtype.size != size:
        raise ValueError(f'dtype has wrong byte-width, expected {size}, got {dtype.itemsize}')
    if kinds is not None and dtype.kind in _tuple_array(kinds):
        raise ValueError(f"datatype of incorrect kind, expected one of {kinds}, got {dtype.kind}")
    return dtype


def dread_dtype(arr:bytes, dct:dict)->np.dtype:
    """
    Direct read of dtype bytes array.

    Parameters
    ----------
    arr : bytes
        Bytes array from HDF5 file to read as dtype.
    dct : dict
        Ignored.

    Returns
    -------
    np.dtype
        numpy data type.

    """
    return np.dtype(arr.decode())


def dwrite_dtype(val:np.dtype)->bytes:
    """
    Direct write function for numpy dtype. Returns bytes representation of
    numpy dtype for writing to HDF5 file.

    Parameters
    ----------
    val : np.dtype
        numpy dtype to write to HDF5 file.

    Returns
    -------
    bytes
        Bytes representation of numpy data type.

    """
    return val.str.encode()


TV_dtype = register_byteslike(np.dtype, check_dtype, dread_dtype, dwrite_dtype)


class _ImData(_DataLike):
    """
    Abstract class inheriting from _DataLike to function as an immutable DataClass
    with a hash function.
    
    There are a number of key class variables used to control behavior:
        
    - ``__slots__`` is required, specifying all class attributes
    - ``_typeconversions`` a :class:`smfbursts.utils.misc.ImDict` with keys
      that are values also in ``__slots__``, and values are :class:`TypeValidator`
      objects that control.
    - ``_required`` frozenset of all attributes that are required to be
      present in instance of subclass
    - ``_hashskip`` tuple or frozenset of all
    - ``_defaults`` ImDict of default values, if value is callable, then will treat as factory function
    
    :meta public:
    """
    _hashskip = tuple() #: values that need not be considered in equality or hashing
    _typeconversions:ClassVar[ImDict[str,Union[TypeValidator,type]]] = ImDict()
    #: stores all subclasses of _ImData, subclasses MUST NOT overwrite previous names
    _registered:ClassVar[FixedDict[str,"_ImData"]] = FixedDict() 

    def __init_subclass__(cls):
        cls._registered[_type_name(cls)] = cls
        
    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        proc_kwargs = dict()
        for key, value in iter_funcinput(cls.__slots__, cls._defaults, cls._required, *args, **kwargs):
            if key in cls._typeconversions:
                value = cls._typeconversions[key].check_val(value, obj, proc_kwargs)
            elif key not in cls._hashskip:
                value = TypeValidator.check_any(value, obj)
            super(_ImData, obj).__setattr__(key, value)
        obj.__post_init__()
        return obj

    def _replace_fields(self, fields:dict[str:Any]=None,  pop:tuple[str,...]=None, _strict:bool=True)->"_ImData":
        """
        Return modified copy of _ImData subclass.

        Parameters
        ----------
        fields : dict[str,Any], optional
            Key:value pairs of fields to replace in new copy. The default is None.
        pop : tuple[str, ...], optional
            Fields to remove from new _ImData. The default is None.
        _strict : TYPE, optional
            If :code:`True` peform type validation of new object, set to :code:`False`
            only if confident that new fields are valid. The default is True.

        Raises
        ------
        ValueError
            One or more fields cannot be changed or removed.

        Returns
        -------
        _ImData
            Copy of object with specified fields replaced or pop.

        """
        fields = dict() if fields is None else fields
        pop = tuple() if pop is None else pop
        if any((err:=key) in pop for key in fields.keys()):
            raise ValueError(f"{err} appears in both fields and pop, cannot have overlapping fields")
        cls = type(self)
        if any((err:=key) in cls._required for key in pop):
            raise ValueError(f"cannot pop {err} from {cls.__name__}: it is a required field")
        obj = object.__new__(cls)
        recheck = _strict
        proc_kwargs = dict()
        for key in cls.__slots__:
            if key in pop:
                continue
            check = key in fields or recheck
            if key in fields:
                value = fields[key] 
            elif key in self:
                value = getattr(self, key)
            else:
                continue
            if check:
                if key in cls._typeconversions:
                    value = cls._typeconversions[key].check_val(value, obj, proc_kwargs)
                elif key not in cls._hashskip:
                    value = TypeValidator.check_any(value, obj)
                recheck = _strict
            super(_ImData, obj).__setattr__(key, value)
        if _strict:
            obj.__post_init__()
        return obj

    def replace_field(self, attr:str, value:Any)->"_ImData":
        """
        Create a new object, identical to self, except attr replaced with value.

        Parameters
        ----------
        attr : str
            Attribute to replace.
        value : Any
            Value to replace attribute `attr` with.

        Returns
        -------
        "_ImData"
            New object with attribute attr replaced.

        """
        return self._replace_fields({attr:value})

    def __copy__(self):
        obj = object.__new__(type(self))
        for k, v in self.items():
            super(_ImData, obj).__setattr__(k, v)
        return obj

    def copy(self):
        """Make duplicate object"""
        return self.__copy__()

    def __post_init__(self):
        """Sublcasses may implement for finalization/checking. Called at end of init"""
        pass

    def __setattr__(self, key, value):
        if key not in self._hashskip:
            raise AttributeError(f"{self.__class__} does not support assignment")
        if key in self._typeconversions:
            value = self._typeconversions[key].check_val(value, self)
        super(_ImData, self).__setattr__(key, value)
    
    def __delattr__(self, attr):
        if attr not in self.__slots__:
            raise AttributeError(f"{type(self).__name__} has no attribute {attr}")
        if attr not in self._hashskip:
            raise AttributeError(f"Cannot delete hashed attribute {attr} from immutable type {type(self).__name__}")
        super(_ImData, self).__delattr__(attr)

    def __hash__(self):
        return _const_hash(tuple((key, _const_hash(_tuple_arr_tdct(val))) 
                                 for key, val in self.items(skip=self._hashskip)))

    def __eq__(self, other):
        if type(self) == type(other):
            if _const_hash(self) == _const_hash(other):
                return all(_eq(v, other[k]) for k, v in self.items() if k not in self._hashskip)
        return False

    def write_group(self, group:tb.Group, name:Union[str,None]=None)->tb.Group:
        """
        Write object into ``group/name`` 

        Parameters
        ----------
        group : tb.Group
            Group in which to create group `name` is created that represents self.
        name : Union[str,None], optional
            Name to give group representing self. The default is None.

        Returns
        -------
        tb.Group
            group where self was written.

        """
        if name is None:
            name = self.__class__.__name__
        file, group = init_write_group(group, name, 'ImData')
        file.create_array(group, 'ImDataType_', obj=_type_name(type(self)).encode())
        slots = list()
        for key, value in self.items(skip=self._hashskip):
            TypeValidator.write_any(group, key, value)
            slots.append(key)
        file.create_carray(group, 'slots_', obj=np.array(slots))
        return group

    def _get_typeconversions(self, k:str, val:Any)->TypeValidator:
        """
        Get :class:`TypeValidator` for given key ``k`` in _ImData subclass.

        Parameters
        ----------
        k : str
            Name of key in _ImData sublass to get TypeValidator for.
        val : TYPE
            Alternate value to get typeconverter for.

        Returns
        -------
        TypeValidator
            TypeValidator for key.

        """
        if k in self._typeconversions:
            return self._typeconversions[k]
        return TypeValidator.convert_type(type(val))

    @classmethod
    def load_group(cls, group:tb.Group):
        """
        Load _ImData object recorded in ``group`` into memory as python object.

        Parameters
        ----------
        group : tb.Group
            Group to load.

        Raises
        ------
        ValueError
            Unrecognized _ImData subclass, another module needs to be loaded that
            defines the subclass of _ImData.
        TypeError
            group does not represent _ImData subclass.

        Returns
        -------
        _ImData
            Python _ImData object (in memory).

        """
        cls_name = group['ImDataType_'].read().decode()
        if cls == _ImData:
            if cls_name not in _ImData._registered:
                raise ValueError("{cls_name} not a registered _ImData type, must import additional module")
            cls = _ImData._registered[cls_name]
            if not issubclass(cls, _ImData):
                raise TypeError("malformed node, unrecognized _ImData type")
        elif cls.__name__ != cls_name:
            warnings.warn("mismatch between expected class and class name recorded in hdf5 group")
        kwargs = dict()
        for key in (k.decode() for k in group.slots_.read()):
            kwargs[key] = TypeValidator.read_any(group[key])
        return cls(**kwargs)


def check_ImData(val:_ImData, subclass:type=None, **kwargs):
    """
    Check function for TV_ImData, ensures subclass of :class:`_ImData`

    Parameters
    ----------
    val : _ImData
        Value to check.
    subclass : type, optional
        val must be subclass of this, if not None. The default is None.
    **kwargs : TYPE
        DESCRIPTION.

    Raises
    ------
    TypeError
        Val is not of correct type (:class:`_ImData` or subclass).

    Returns
    -------
    _ImData
        val, echoed.

    """
    if not isinstance(val, _ImData):
        raise TypeError(f'must be subclass of _ImData, got {type(val)}')
    if subclass is not None and not isinstance(val, subclass):
        raise TypeError(f'must be subclass of {subclass}')
    return val


def write_ImData(group:tb.Group, name:str, val:_ImData)->tb.Group:
    """
    Write function for TV_ImData.

    Parameters
    ----------
    group : tb.Group
        Group in which to create HDF5 representation.
    name : str
        Name of group that will represent val.
    val : _ImData
        Object to record.

    Returns
    -------
    tb.Group
        Group representing val in HDF5 file.

    """
    return val.write_group(group, name=name)


def read_ImData(group:tb.Group, dct:dict)->_ImData:
    """
    Read function ofr TV_ImData

    Parameters
    ----------
    group : tb.Group
        Group to read into python object.
    dict
        TypeValidator._grouptypes[grouptype].

    Returns
    -------
    _ImData
        Loaded python object.

    """
    return _ImData.load_group(group)


TypeValidator.register_grouptype('ImData', read_ImData)
TV_ImData = TypeValidator(_ImData, check_ImData, write_ImData)
