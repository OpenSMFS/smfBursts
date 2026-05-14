#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created: 16/03/2025
# author Paul David Harris
"""
This module provides a means of accessing RAM and HDF5 files in dictionary like
objects function essentially the same.

The main class of this module is the :class:`DiskDict` which uses the different
methods supplied by :class:`smfbursts.immutabledata.TypeValidator` objects to create a
representation of a dictionary inside of a HDF5-file. :class:`DiskDict` and
its subclasses can either store data in memory or in a HDF5 file.

This is useful for optimizing memory usage, as well as saving data and reopening.
"""
from typing import ClassVar, Union, Any
from collections.abc import Callable, Hashable, Iterator
from abc import ABC, abstractmethod

import numpy as np
import tables as tb

from .utils import FixedDict, ImDict, _GroupFuture, GroupFuture, GroupArg, _masked_iter
from .immutabledata import TypeValidator


def _iterrows(node:tb.Array)->np.ndarray:
    """Iterate over rows in a tables array"""
    for i in range(node.shape[0]):
        yield node[i]

class DiskDict:
    """
    Base class for group of objects that mimic behavior of dictionaries,
    though with fixed types of keys and controled types of values, that can be stored
    in HDF5 files.
    
    Parameters
    ----------
    dct : dict, optional
        Initial key-value pairs of DiskDict. The defualt is None
    group: GroupFuture
        The HDF5 group where data is to be stored. The default is None.
    autosave : bool, optional
        If :code:`True`, then as soon as key is added to dictionary, it is written
        to HDF5 file. May also be callable that when called with no arguments
        returns a boolean indicating if autosave is active or not.
        The default is False.
    """
    _dict_types:ClassVar[FixedDict[str,type]] = FixedDict()
    _freeze:ClassVar[bool] = False
    _exclude_groups:ClassVar[tuple[str]] = ('param', 'dataID_', 'dictdef_')
    _group: _GroupFuture
    _cache:dict
    _frozen:bool
    _autosave:bool|Callable[[],bool]
    
    def __init_subclass__(cls):
        TypeValidator.register_groupclass('DiskDict', cls.__name__, cls)
        
    def __init__(self, dct:Union[dict,None]=None, group:GroupFuture=None, 
                 autosave:bool|Callable[[],bool]=False):
        if group is not None and not isinstance(group, (tb.Group, Callable, _GroupFuture)):
            raise TypeError("group must be GroupFuture subtype (None, tables.Group, Callable[[],tables.Group], _GroupFuture)")
        if dct is not None and not isinstance(dct, dict):
            raise TypeError("dct must be None or dictionary")
        dct = {} if dct is None else dct
        # dct = {self.convert_key(k):self.convert_value(v) for k, v in dct.items()} # ensures correct key types
        self._group = group if isinstance(group, _GroupFuture) else _GroupFuture(group, self)
        self._cache = dict()
        self._autosave = autosave
        self._frozen = False
        for key, val in dct.items():
            self[key] = val
        self._frozen = self._freeze
    
    def _key_to_node(self, key:Hashable)->str:
        """Convert a python-object key to the str name of the group/array to which 
        the value will be saved in the HDF5 group"""
        self.convert_key(key)
        return self.key_to_node(key)
    
    @classmethod
    def key_to_node(cls, key:Hashable)->str:
        """
        Convert a dictionary key into the name of the node in an HDF5 group.

        Parameters
        ----------
        key : Hashable
            Key, should be valid key of DiskDict.

        Returns
        -------
        str
            Name of corresponding node in HDF5 file.

        """
        return TypeValidator.val_to_nodename(key)
    
    def _node_to_key(self, node:tb.Node)->Hashable:
        """Convert a node/group of the HDF5 file to the python object key that
        the value will be represented in the DiskDict"""
        return TypeValidator.read_nodename(node._v_name)

    def convert_key(self, key:Hashable)->Hashable:
        """Function that "regularizes" a new key, primarily for checking the type of the key"""
        try:
            TypeValidator.val_to_nodename(key)
        except Exception as e:
            raise TypeError("cannot convert {type(key)} to nodename") from e
        return key

    def convert_value(self, key:Hashable, value:Any)->Any:
        """Function that "regularizes" a new value, primarily for checkin gthe type/shape of a value
        ... note::
            
            always called after :meth:`DiskDict.convert_key` and so key should 
            already be "regularized"
        """
        return TypeValidator.check_any(value)

    def _read_group(self, group:tb.Group, nodename:str)->Any:
        """
        Reads the given node in the group, ie the group is the higher level group,
        and nodename is the value returned from :meth:`DiskDict._key_to_node` and
        it is this node/key that should be read and value returned
        """
        return TypeValidator.read_any(group[nodename])

    def _read_item(self, key:Hashable)->Any:
        """
        Internal method that reads the key from the HDF5 group. This method
        call relevant conversion functions before calling :meth:`DiskDict._read_group`
        """
        nodename = self._key_to_node(key)
        if nodename not in self._group:
            raise KeyError(f"{key} not in dictionary")
        return self._read_group(self.group, nodename)

    def _write_group(self, group:tb.Group, nodename:str, value:Any)->tb.Group:
        """Internal method to write value to nodename (already converted) to group"""
        return TypeValidator.write_any(group, nodename, value)

    def _write_item(self, key:Hashable, value:Any, group:Union[tb.Group,None]=None)->None:
        """Internal command to write a particular item, based on (non-converted) key"""
        group = self._group._group if group is None else group
        if group is None:
            raise ValueError("group not set, cannot write item to HDF5 file")
        nodename = self._key_to_node(key)
        if nodename not in group:
            self._write_group(group, nodename, value)
    
    def __getitem__(self, key):
        key = self.convert_key(key)
        if key in self._cache:
            return self._cache[key]
        if self._key_to_node(key) not in self._group:
            raise KeyError(f"{key} not in DiskDict")
        self._cache[key] = self._read_item(key)
        return self._cache[key]
    
    def __setitem__(self, key, value):
        key = self.convert_key(key)
        if key in self:
            raise TypeError(f"{key} already specified, cannot re-assign")
        if self._frozen:
            raise TypeError("{self.__class__.__name__} does not support assignment")
        value = self.convert_value(key, value)
        self._cache[key] = value
        if self.autosave:
            if self.group is None:
                raise ValueError('No hdf5 file specified, cannot write to file')
            self._write_item(key, value)
            
    def __contains__(self, key):
        try:
            key = self.convert_key(key)
        except:
            return False
        if key in self._cache:
            return True
        return self._key_to_node(key) in self._group
    
    def __len__(self):
        return len(list(self.keys()))
                
    def keys(self)->Iterator[Hashable]:
        """Iterate over keys of diskdict"""
        yield from self._cache.keys()
        if self._group._created:
            for g in self.file.iter_nodes(self._group._group):
                if g._v_name in self._exclude_groups:
                    continue
                key = self._node_to_key(g)
                if key in self._cache:
                    continue
                yield key

    def values(self)->Iterator[Any]:
        """Iterate over values in diskdict"""
        yield from (self[key] for key in self.keys())

    def items(self)->Iterator[tuple[Hashable,Any]]:
        """Iterate over key-value pairs in diskdict"""
        yield from ((key, self[key]) for key in self.keys())

    def pop(self, key:Hashable, *default:Any)->Any:
        """
        Remove gey from dictionary, and return it's value

        Parameters
        ----------
        key : Hashable
            Key to remove from dictionary.
        default : Any
            Value to return if key not in dictionary (If supplied).

        Raises
        ------
        TypeError
            Too many arguments.
        KeyError
            Key not in dictionary.
        AttributeError
            Key cannot be popped.

        Returns
        -------
        Any
            Value of key in dictionary or default.

        """
        if len(default) > 1:
            raise TypeError(f"pop expected maximum of 2 arguments, got {len(default)+1}")
        if key not in self:
            if default:
                return default[0]
            raise KeyError(f"{key} not in dictionary")
        if self._key_to_node(key) in self._group:
            raise AttributeError(f"{key} already written to disk, cannot remove from dictionary")
        return self.cache.pop(key)

    def _get_prop(self, key:Hashable, prop:str)->Any:
        """Get property ``prop`` from self._cache (HDF5 node) corresponding to key"""
        if key not in self:
            raise KeyError(f"{key} not in dictionary")
        if key in self._cache:
            return getattr(self._cache[key], prop)
        elif self._key_to_node(key) in self._group:
            return getattr(self.group[key], prop)

    @property
    def file(self)->Union[tb.File,None]:
        """
        tables File object representing the file the DiskDict is attached to,
        None if no file is set
        """
        return self._group._filefuture

    @property
    def group(self)->Union[tb.Group,None]:
        """
        table Group in which all values of DiskDict are saved, None if no group
        is set
        """
        return self._group._group

    @group.setter
    def group(self, group:tb.Group):
        if self._group._created:
            raise TypeError("group already set")
        self._group = group if isinstance(group, _GroupFuture) else _GroupFuture(group, self)

    @classmethod
    def load_group(cls, group:tb.Group, autosave:bool=False, load_all:bool=False)->"DiskDict":
        """
        Load HDF5 group into DiskDict

        Parameters
        ----------
        group : tb.Group
            Group from which to load saved data.
        autosave : bool, optional
            Whether to automatically save values to disk. The default is False.
        load_all : bool, optional
            Whether to load all values into memory and detach from current HDF5 file.
            The default is False.

        Returns
        -------
        out : DiskDict
            File loaded.

        """
        ocls = TypeValidator._grouptypes['DiskDict'][group.dictdef_.read().decode()]
        out = ocls(group=group, autosave=autosave)
        if load_all:
            out.load_to_memory()
        return out

    def save(self, group:GroupArg=None)->tb.Group:
        """
        Flush all data into HDF5 file

        Parameters
        ----------
        group : tb.Group|None, optional
            If set, all data will be saved into the given hdf5 group, otherwise
            use the current group the DiskDict is using. The default is None.

        Returns
        -------
        tb.Group.
            tables Group where data was saved
        """
        group = self._group._create() if group is None else group
        if group is None:
            raise ValueError("No group specified in which to save file")
        if 'dictdef_' not in group:
            group._v_file.create_array(group, 'dictdef_', type(self).__name__.encode())
        for key, value in self._cache.items():
            self._write_item(key, value, group=group)
        return group

    def clear_memory(self, group:GroupFuture=None)->None:
        """
        Flush all data to group and clear cached values, freeing up memory.

        Parameters
        ----------
        group : tb.Group|None, optional
            If set data will be saved to given group instead of current group
            of DiskDict. The default is None.

        Returns
        -------
        None
        
        """
        if group is not None:
            self.group = group if isinstance(group, _GroupFuture) else _GroupFuture(group, self)
        self.save()
        self._cache = dict()

    def load_to_memory(self)->None:
        """Load all values to memory"""
        _ = list(self.items())
    
    def reset_group(self, group:GroupFuture=None)->None:
        """
        Load all values to memory and switch group to given group.

        Parameters
        ----------
        group : GroupFuture, optional
            New group to assign to dictionary, if None, will cause dictionary to
            be entirely on disk. The default is None.

        """
        self.load_to_memory()
        self._group = group if isinstance(group, _GroupFuture) else _GroupFuture(group, self)
        if self.autosave:
            self.save()

    @property
    def autosave(self)->bool:
        """If new keys are automatically recorded in HDF5 file"""
        if callable(self._autosave):
            return self._autosave()
        return self._autosave

    @property
    def inmemory(self, key:Hashable)->bool:
        """If a given key has been loaded into memory"""
        return self.convert_key(key) in self._cache

    def iter_key(self, key:Hashable)->Iterator:
        """
        Iterate over values in key (value must be iterable, ie array)

        Parameters
        ----------
        key : Hashable
            Key to iterate over.

        Raises
        ------
        ValueError
            Cannot iterate over non-array value.
        KeyError
            Key not in dictionary.

        Yields
        ------
        Iterator
            Iterator over each row in key.

        """
        key = self.convert_key(key)
        if key in self._cache:
            yield from self._cache[key]
        elif (nodename:=self._key_to_node(key)) in self._group:
            if not self._group[nodename]._v_title.startswith('ndarray'):
                raise ValueError("{key} is not a numpy array")
            shape = tuple(int(i) for i in self._group[nodename]._v_title.split('-')[1:])
            if subshape := shape[1:]:
                rowcount = np.prod(subshape)
                i = 0
                for row in _iterrows(self._group[nodename]):
                    if i == 0:
                        out = np.empty(rowcount, dtype=np.object_)
                    out[i] = row
                    i += 1
                    if i == rowcount:
                        i = 0
                        yield out.reshape(subshape)
            else:
                yield from _iterrows(self._group[nodename])
        else:
            raise KeyError("{key} does not exist in this dictionary")

    def get_from_index(self, key:Hashable, index:Hashable)->Any:
        """
        Retrieve specific index within a given key

        Parameters
        ----------
        key : Hashable
            Key in dictionary.
        index : Hashable
            index to access from array.

        Raises
        ------
        TypeError
            Key contains a non-array value.

        Returns
        -------
        Any
            value at index in array specified by key.

        """
        key = self.convert_key(key)
        if key in self._cache:
            return self._cache[key][index]
        elif (nodename:=self._key_to_node(key)) in self._group:
            if not self._group[nodename]._v_title.startswith('ndarray'):
                raise TypeError("{key} is not a numpy array")
            if len(self._group[nodename]._v_title.split('-')) > 2:
                return self[key][index]
            else:
                return self._group[nodename][index]


TypeValidator.register_grouptype('DiskDict', DiskDict.load_group)
TypeValidator.register_groupclass('DiskDict', DiskDict.__name__, DiskDict)


class VattrDD(DiskDict):
    """
    Subclass of :class:`DiskDict` where the keys are strings specified
    *at the time of object creation*.
    Specify keys with attrs kwarg upon object reation.
    """
    def __init__(self, attrs:frozenset=None, **kwargs):
        if attrs is None:
            raise ValueError("must set attrs")
        attrs = frozenset(attrs)
        if any(not isinstance(attr, str) for attr in attrs):
            raise ValueError("all attrs must be of type str")
        self._attrs = attrs
        super().__init__(**kwargs)

    def convert_key(self, key:str)->str:
        """Function that "regularizes" a new key, primarily for checking the type of the key"""
        if key not in self._attrs:
            raise ValueError(f"{key} not valid key name, must be in {self._attrs}")
        return super().convert_key(key)


class AttrDD(DiskDict, ABC):
    """
    Abstract subclass of :class:`DiskDict` where valid keys are specified by
    subclass.
    """
    @abstractmethod
    def _attrs(cls)->frozenset:
        """Set of valid keys of subclass of AttrDD"""
        raise NotImplementedError("Incomplete subtype of AttrDD")
    
    def convert_key(self, key:str)->str:
        """Function that "regularizes" a new key, primarily for checking the type of the key"""
        if key not in self._attrs:
            raise ValueError(f"'{key}' not valid key name, must be in {self._attrs}")
        return super().convert_key(key)


class MappedAttrDD(DiskDict, ABC):
    """
    Abstract sublcass of DiskDict where keys are defined by a name_map, which
    maps key values to node-names.
    """
    @property
    @abstractmethod
    def _name_map(self)->ImDict:
        """Map of key to node-name values"""
        raise NotImplementedError("subclasses must specify as tupledict")
    
    @classmethod
    def key_to_node(cls, key:str)->str:
        """Get name of node from key"""
        return cls._name_map[key]
    
    def _node_to_key(self, node:tb.Node)->str:
        """Get key from HDF5 node"""
        return {v:k for k, v in self._name_map.items()}[node._v_name]
    
    def convert_key(self, key:str)->str:
        """Function that "regularizes" a new key, primarily for checking the type of the key"""
        if key not in self._name_map:
            raise ValueError(f'{key} not valid key, must be in {self._name_map.keys()}')
        return key


class TypedValueDD(DiskDict, ABC):
    """
    DiskDict where values are typed through _valtype classmethod that maps
    keys to a TypeValidator.
    """
    @abstractmethod
    def _valtype(cls, key:Hashable)->TypeValidator:
        """Get TypeValidator for value from key"""
        raise NotImplementedError("TypedValueDD incomplete")

    def convert_value(self, key:Hashable, value:Any)->Any:
        """
        Convert value to correct type, given key

        Parameters
        ----------
        key : Hashable
            Key used to define type that ``value`` should be.
        value : Any
            value to be converted.

        Returns
        -------
        Any
            Converted value to store in TypeValueDD.

        """
        return self._valtype(key).check_any(value, self)


class NestedDD(DiskDict):
    """
    DiskDict where keys are ensured to be a tuple
    """
    def convert_key(self, key:tuple)->tuple:
        """Function that "regularizes" a new key, primarily for checking the type of the key"""
        if isinstance(key, str):
            key = (key, )
        if any(not TypeValidator.val_has_node_repr(v) for v in key):
            raise TypeError("cannot read key as tuple of values with nodename representations")
        return key

    def _node_to_key(self, node:tb.Node)->str:
        """Convert HDF5 node to key"""
        return tuple(TypeValidator.node_read(n) for n in node._v_name.split('__'))

    @classmethod
    def key_to_node(cls, key):
        """Get name of node from key"""
        return '__'.join(TypeValidator.val_to_nodename(k) for k in key)


def _dd_echo(key, value):
    """Echo key value input, for use as checkfunc in SubDiskDict"""
    return key, value


class SubDiskDict:
    """
    Alias a specific subset of a NestedDiskDict.
    """
    __slots__ = ('_diskdict', '_prekey', '_check_func')
    def __init__(self, diskdict:DiskDict, prekey:Hashable, check_func:Callable[[Hashable,Any],tuple[Hashable,Any]]=_dd_echo):
        super().__setattr__('_diskdict', diskdict)
        if not isinstance(prekey, tuple):
            prekey = (prekey, )
        super().__setattr__('_prekey', prekey)
        super().__setattr__('_check_func', check_func)

    def __getattribute__(self, attr):
        if attr not in ('keys', 'items', 'values', 'get'):
            raise AttributeError("SubDiskDict has not attributes")
        return super().__getattribute__(attr)

    def __setitem__(self, key, value):
        key, value = super().__getattribute__('_check_func')(key, value)
        if not isinstance(key, tuple):
            key = (key, )
        key = super().__getattribute__('_prekey') + key
        super().__getattribute__('_diskdict')[key] = value
    
    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key, )
        key = super().__getattribute__('_prekey') + key
        return super().__getattribute__('_diskdict')[key]
        

    def __contains__(self, key):
        if not isinstance(key, tuple):
            key = key, 
        key = super().__getattribute__('_prekey') + key
        return key in super().__getattribute__('_diskdict')

    def keys(self):
        """Iterator over keys in SubDiskDict"""
        prekey = super().__getattribute__('_prekey')
        for key in super().__getattribute__('_diskdict').keys():
            if isinstance(key, tuple) and prekey == key[:len(prekey)]:
                yield key[1:]

    def items(self):
        """Iterator over key:value pairs in SubDiskDict"""
        prekey = super().__getattribute__('_prekey')
        for key, val in super().__getattribute__('_diskdict').items():
            if isinstance(key, tuple) and prekey == key[:len(prekey)]:
                yield key[1:], val

    def values(self):
        """Iterator over values in SubDiskDict"""
        prekey = super().__getattribute__('_prekey')
        for key, val in super().__getattribute__('_diskdict').items():
            if isinstance(key, tuple) and prekey == key[:len(prekey)]:
                yield val

    def get(self, key:Hashable, default:Any=None):
        """Get key from SubDiskDict, return default if key not in SubDiskDict"""
        if not isinstance(key, tuple):
            key = key, 
        key = super().__getattribute__('_prekey') + key
        if key in (diskdict:= super().__getattribute__('_diskdict')):
            return diskdict[key]
        return default


class MaskedDD:
    """
    Alias of DiskDict, all values must be arrays, always returns arrays masked
    by mask- used with tables that are derived from other tables.
    """
    __slots__ = ('_parent', '_mask')
    _parent:DiskDict
    _mask:np.ndarray[np.bool_]
    def __init__(self, parent, mask):
        self._parent = parent
        self._mask = mask
    
    def __contains__(self, key):
        return key in self._parent
    
    def __getitem__(self, key):
        return self._parent[key][self._mask]
    
    def iter_key(self, key:Hashable)->Iterator:
        """
        Iterate over masked rows in key.

        Parameters
        ----------
        key : Hashable
            Key to iterate over.

        Yields
        ------
        Iterator
            Masked iterator over each row in array.

        """
        yield from _masked_iter(self._parent.iter_key(key), self._mask)
    
    def keys(self):
        """Iterator over keys in MaskedDD"""
        yield from self._parent.keys()
    
    def values(self):
        """Iterator over values in MaskedDD"""
        yield from (self[k] for k in self._parent.keys())
    
    def items(self):
        """Iterator over key:value pairs in MaskedDD"""
        yield from ((k, self[k]) for k in self._parent.keys())
    
    def get(self, key:Hashable, default:Any=None):
        """Get key from MappedDD, if not present, return default"""
        if key in self:
            return self[key]
        return default
    
    @property
    def _group(self)->_GroupFuture:
        """Group where unmasked arrays are stored"""
        return self._parent._group