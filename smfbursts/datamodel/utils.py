#
# smfBursts - A single-molecule fluorescence burst analysis toolkit.
#
# Copyright (C) 2025 Paul David Harris <harripd@gmail.com>
#
"""
Utility functions.
"""
import weakref
from itertools import chain, product, repeat
from collections.abc import Callable, Sequence, Iterator, Iterable, Hashable
from textwrap import wrap
from typing import ClassVar, Any, Literal
from numbers import Integral, Number, Real
import math
import warnings

import numpy as np
import tables as tb


IndexType = tuple[int|slice|None,...]

_nan = np.nan


def _const_hash(val:Hashable)->int:
    """Hash function that is consistent between python kernels"""
    if isinstance(val, Callable):
        val = f'{val.__module__}.{val.__name__}'
    if isinstance(val, str):
        return hash(tuple(ord(b) for b in val))
    if isinstance(val, Number):
        if math.isnan(val):
            return 0xf87f<<42
        return hash(val)
    if isinstance(val, Sequence):
        return hash(tuple(_const_hash(v) for v in val))
    return hash(val)


def _tuple_array(array:np.ndarray)->tuple[Number]:
    """Convert any sequence into a tuple, works recursively on internal sequences,
    used primarily to convert numpy arrays to nested tuples so they can be hashed"""
    if isinstance(array, np.ndarray):
        if array.ndim == 0:
            return array.reshape(1)[0]
        return tuple(_tuple_array(a) for a in array)
    if isinstance(array, tuple):
        return tuple(_tuple_array(v) for v in array)
    if isinstance(array, Callable):
        try:
            return 'callable', f'{array.__module__}.{array.__name__}'
        except AttributeError:
            return array
    if np.issubdtype(type(array), np.number) and np.isnan(array):
        return _nan
    return array


def _as_sortable(val:np.ndarray|Hashable)->Hashable:
    """sorted key function for general sequences"""
    if np.issubdtype(type(val), np.number) or isinstance(val, str):
        return val
    if isinstance(val, np.ndarray):
        return _tuple_array(val)
    return hash(val)


def _make_sortable(val:np.ndarray|Hashable)->tuple[Hashable,...]|int|str:
    """Convert val to a sortable sequence"""
    if isinstance(val, np.ndarray):
        val = tuple(val)
    if isinstance(val, tuple):
        val = tuple(_make_sortable(v) if isinstance(v, tuple) else _as_sortable(v) for v in val)
    if isinstance(val, (Real, str, bytes)):
        return val
    if hasattr(val, '_sort_tuple'):
        tp = type(val)
        return (f'{tp.__module__}.{tp.__name__}', val._sort_tuple)
    return _const_hash(val)


class _Sortable:
    __slots__ = ('key', )
    def __init__(self, key):
        self.key = key

def hash_array(array:np.ndarray)->int:
    """Hash a numpy array"""
    return hash(_tuple_array(array))


def _is_list_of_arrays(obj):
    """Test if obj is a list (exclusively list) of numpy arrays"""
    return isinstance(obj, list) and np.all([isinstance(v, np.ndarray) for v in obj])


def _eq(a, b):
    """
    equals function that ensures objects are
    1. Of same type (int 1 != float 1.0)
    2. Compares sequences an an all fashion, requiring both sequences be of same length
    3. Never fails
    4. Two nan values are considered equal
    """
    if type(a) != type(b): return False
    if isinstance(a, np.ndarray):
        if a.shape != b.shape: return False
        if a.dtype == np.object_:
            if b.dtype != np.object_: return False
            return all(_eq(aa, bb) for aa, bb in zip(a.ravel(), b.ravel()))
        if np.all(a == b):
            return True
        return np.all((a == b) | (np.isnan(a)&np.isnan(b)))
    if isinstance(a, Sequence) and not isinstance(a, str):
        if len(a) != len(b): return False
        return all(_eq(aa, bb) for aa, bb in zip(a, b))
    return a == b or (isinstance(a, Number) and math.isnan(a) and math.isnan(b))


def _echo(arr:Any)->Any:
    """Simple function that returns the argument given, for use as dummy function."""
    return arr


def make_objectarray(arrs:np.ndarray)->np.ndarray[np.object_]:
    r"""
    Convert any array into 1D object array.

    Parameters
    ----------
    arrs : np.ndarray
        numpy array to convert to object array.

    Returns
    -------
    np.ndarray[np.object\_]
        1D object type numpy array.

    """
    out = np.empty(len(arrs), dtype=np.object_)
    for i, arr in enumerate(arrs):
        out[i] = arr
    return out
        

def make_immutable(array:np.ndarray|tuple|list|Hashable, copy:bool=False)->np.ndarray|tuple|Hashable:
    """
    Convert array into an immutable form. For lists, convert to tuples,
    for numpy arrays set 'writeable' to ``False``. Recurses through elements,
    and if they are also sequences, ensure they are also

    Parameters
    ----------
    array : np.ndarray|tuple|list|Hashable
        Object to be converted into immutable form.
    copy : bool, optional
        If True, return value is always a copy of first argument, if False,
        return value may be same as first argument, for numpy arrays this may
        mean that the input is now also immutable. The default is False.

    Returns
    -------
    np.ndarray | tuple | Hashable
        Input in an immutable form.

    """
    if isinstance(array, (list, tuple)) and all(not np.issubdtype(type(arr, np.number) for arr in array)):
        out = make_objectarray(array)
    elif isinstance(array, np.ndarray) and array.dtype == np.object_:
        out = np.empty(array.shape, dtype=object) if copy else array
        for i in product(*(range(n) for n in array.shape)):
            out[i] = make_immutable(array[i])
    else:
        out = np.asarray(array)
    if out.flags['WRITEABLE']:
        out = out.copy() if copy else out
        out.setflags(write=False)
    return out


def _iter_tbgroup_numeric(group:tb.Group, prefix:str)->Iterable[tb.Node]:
    """
    Iterate over groups labeled {prefix}{i} in HDF5 group, 
    must start at 0, ends iteration at first number not present
    (ie if number is skipped, iteration will end).
    """
    i = 0
    while (g:=f'{prefix}{i}') in group:
        yield group[g]
        i += 1

# dictionary of floating point value to prefix of SI prefixes, requires TEX
_unit_prefixes_tex = {1e-30:'q', 1e-27:'r', 1e-24:'y', 1e-21:'z', 1e-18:'a', 1e-15:'f',
                  1e-12:'p', 1e-9:'n', 1e-6:r'\mu ', 1e-3:'m', 1e-2:'c', 1e-1:'d',
                  1.0:'', 1e1:'da', 1e2:'h', 1e3:'k', 1e6:'M', 1e9:'G', 1e12:'T',
                  1e15:'P', 1e18:'E', 1e21:'Z', 1e24:'Y', 1e27:'R', 1e30:'Q', 
                  True:'', False:''}

# dictionary of floating point value to prefix of SI prefixes, no r"\" characters
_unit_prefixes_notex = _unit_prefixes_tex.copy()
_unit_prefixes_notex[1e-6] = 'u'


def get_unit_prefix(factor:float, tex:bool=True)->str:
    """Get the expected SI unit-prefix based on rescale-factor"""
    if factor is True or factor is False:
        return ''
    factor = 10.0**factor if isinstance(factor, Integral) else factor
    _unit_prefixes = _unit_prefixes_tex if tex else _unit_prefixes_notex
    return _unit_prefixes.get(float(factor), f'{factor}*')


class ImDict(dict):
    """
    Dictionary that cannot be changed after instantiation, ie immutable dict.
    
    All methods that mutate dictionary disabled, like ``update`` and ``pop``
    """
    def __setitem__(self, k, v):
        raise AttributeError("ImDict does not support assignment")
    
    def update(self, *args, **kwargs):
        raise TypeError("update is disabled for fixed dict")
        
    def pop(self, *args, **kwargs):
        raise TypeError("pop is disabled for fixed dict")


class FixedDict(dict):
    """
    Dictionary where keys can be created, but once set cannot be changes.
    
    Different from :class:`ImDict` in that new values can be added.
    """
    def __setitem__(self, key, value):
        if key in self:
            raise ValueError(f"{key} already set")
        if isinstance(value, np.ndarray):
            value = make_immutable(value)
        super().__setitem__(key, value)
    
    def update(self, *args, **kwargs):
        """
        Add new values to dictionary, works like update from built-in ``dict``.
        
        """
        update = dict(*args, **kwargs)
        if any((err:=k) in self for k in update.keys()):
            raise KeyError(f'{err} already set')
        super().update(update)
    
    def pop(self, *args, **kwargs):
        raise TypeError("pop is disabled for fixed dict")


class tupledict(tuple):
    """
    An ordered, immutable dictionary where all keys must be strings.
    
    Internally stored as a tuple of key-value pairs.
    
    Allows indexing and slicing like a tuple as well.
    
    >>> td = tupledict(('a', 1), ('b',2))
    >>> td['a']
    1
    >>>td[0]
    1
    >>>td[:]
    (('a', 1), ('b', 2))
    
    
    """
    def __new__(cls, *args, **kwargs):
        if bool(kwargs) and bool(args):
            raise TypeError("must be either iterable of key, value pairs, or kwargs, not both")
        if kwargs:
            iterable = tuple((key, value) for key, value in kwargs.items())
        else:
            if len(set(key for key, _ in args)) != len(args):
                raise ValueError("one or more keys repeated")
            iterable = args if not isinstance(args, dict) else args.items()
        return super().__new__(tupledict, (cls._verify_input(v) for v in iterable))
    
    @classmethod
    def _verify_input(cls, value):
        """check each key:value pair, and convert if necessary"""
        if len(value) != 2:
            raise ValueError("must be sequence of pairs of keys and values")
        k, v = value
        try:
            kn = str(k)
        except Exception as e:
            raise TypeError(f"cannot interpret {k} as str") from e
        if kn != k:
            raise TypeError(f"cannot interpret {k} as str")
        try:
            hash(_tuple_array(v))
        except Exception as e:
            raise TypeError(f"cannot hash {v}") from e
        return kn, v
    
    @classmethod
    def from_order(cls, order:Sequence[str], *args:Any, defaults_:dict[str,Any]=None, **kwargs:Any)->"tupledict":
        r"""
        Create a tupledict with keys in the order specified in the first argument.
        Can specify values either as positional arguments, which are assigned keys
        starting from the first key in the first argument. Kwargs can be used to
        specify values of specific keys, allowing out-of-order specification
        of values. defaults\_ kwarg can be used to specify default values of keys
        not specified in either args or kwargs.

        Parameters
        ----------
        order : Sequence[str]
            Sequence of keys, specifies order in which they will appear in output.
        *args : Any
            Values for keys, filled out in order in which appear in ``order`` argument
        defaults_ : dict[str,Any], optional
            Dictionary or mapping of key to default value. The default is None.
        **kwargs : Any
            Value for additional key sin tupledict.

        Raises
        ------
        TypeError
            Arg speciifies same key as kwargs.
        ValueError
            key in kwargs not found in order.

        Returns
        -------
        tupledict
            tupledict of specification.

        """
        defaults_ = dict() if defaults_ is None else defaults_
        cargs = list(zip(order, args))
        if any((err:=arg[0]) in kwargs for arg in cargs):
            raise TypeError(f"multiple arguments for {err}")
        for key in order:
            if key in kwargs:
                cargs.append((key, kwargs.pop(key)))
            elif key in defaults_:
                cargs.append((key, defaults_[key]))
        if kwargs:
            raise ValueError(f"names {list(kwargs.keys())} not in order")
        return cls(*cargs)

    def __getitem__(self, k):
        if hasattr(k, '__index__'):
            return super().__getitem__(k)[1]
        elif isinstance(k, slice):
            return self.__class__(*(super(self.__class__, self).__getitem__(i) 
                                    for i in range(*k.indices(len(self)))))
        for key, value in super().__iter__():
            if key == k:
                return value
        raise KeyError(f'{k} not in tupledict')

    def __getattr__(self, attr):
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError(f"{attr} not in tupledict")

    def __hash__(self):
        return hash(tuple((k, _tuple_array(v) ) for k, v in self.items()))

    def __eq__(self, other):
        if not isinstance(other, tupledict): return False
        if len(other) != len(self): return False
        return all(sk==ok and _eq(sv, ov) for (sk, sv), (ok, ov) in zip(self.items(), other.items()))

    def keys(self)->Iterator[str]:
        """
        Iterator over all keys in tupledict, yielded in order

        Yields
        ------
        Iterator[str]
            Iterator over all keys in tupledict, yielded in order.

        """
        yield from (k for k, v in self)

    def values(self)->Iterator[Any]:
        """
        Iterator over all values in tupledict, yielded in order.

        Yields
        ------
        Iterator[Any]
            Iterator over all values in tupledict, yielded in order.

        """
        yield from (v for k, v in self)

    def items(self)->Iterator[tuple[str, Any]]:
        """
        Iterator over each key-value pair, yielded in order.

        Returns
        -------
        Iterator[tuple[str, Any]]
            Iterator over each key-value pair, yielded in order..

        """
        return iter(self)

    def get(self, key:str, default:Any=None)->Any:
        """
        Retrieve key from dictionary, if key not present, return default. 

        Parameters
        ----------
        key : str
            key to retrieve from tupledict.
        default : any, optional
            Value to return if key not in tupledict. The default is None.

        Returns
        -------
        Any
            value of key, else default.

        """
        for k, v in self:
            if k == key:
                return v
        return default

    def has_val(self, val:Any)->bool:
        """
        Determine of val is a value in the tupledict.

        Parameters
        ----------
        val : Any
            value to test if it is present as a value in the key-value paris
            of the tupledict.

        Returns
        -------
        bool
            Whether or not val is present as a value in the tupledict.

        """
        return any(val == v for v in self.values())

    def __contains__(self, k):
        if hasattr(k, '__index__'):
            return k < len(self) if k >= 0 else (abs(k) - 1) < len(self)
        return any(k == key for key in self.keys())

    @property
    def asdict(self)->dict[str,Any]:
        """The tupledict as a simple dictionary, note each access creates a new dictionary"""
        return {k:v for k, v in self.items()}


class MutDict(dict):
    """
    Dicionary that adds the `mut` property, which is ``False`` if the dictionary
    has not changed since creation, and ``True`` if any operation that changes
    the key-value pairs has been performed.
    """
    _orig:ImDict() #: Dictionary of original keys, used to compare when values are changed
    
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self._orig = ImDict({k:v for k, v in self.items()})
        for key, value in self.items():
            if type(value) == dict:
                super().__setitem__(key, MutDict(value))

    @property
    def added(self)->bool:
        """Whether any values have been added to the dictionary after creation"""
        if len(self) > len(self._orig):
            return True
        return any(k not in self._orig for k in self.keys())

    @property
    def removed(self)->bool:
        """Whether any values have been removed from dictionary after creation"""
        if len(self) < len(self._orig):
            return True
        return all(k not in self for k in self._orig.keys())

    def _check_mut(self)->bool:
        """Check for any differences with original dictionary"""
        return any(self[k].mut if isinstance(self[k], MutDict) else not _eq(self[k], v) 
                   for k, v in self._orig.items())

    @property
    def mut(self)->bool:
        """If dictionary has changes after creation"""
        return self.added or self.removed or self._check_mut()


def iter_funcinput(_slots:Sequence[str], _defaults:ImDict[str:Any|Callable[[],Any]], 
                   _required:frozenset[str], *args, **kwargs)->Iterator[tuple[str,Any]]:
    r"""
    Iterator for parseing \*args, \*\*kwargs by the specification in _slots and _defaults.
    _slots defines sequence of names of arguments, _defaults is dictonary of
    default values, note that if a value in _defaults is callable, it is assumed
    to be a "factory" function, ie it will be called with no argumnets, and the
    return value is the value assigned to that argument.
    Iterators over arguments, so further processing may take place.
    
    Parameters
    ----------
    _slots : Sequence[str]
        Names of each function argument.
    _defaults : ImDict[str:Any|Callable[[],Any]]
        Dictionary of default values/factory functions.
    _required : frozenset[str]
        Arguments required to be set.
    *args : Any
        Args of function being interpreted.
    **kwargs : Any
        Kwargs of function being interpreted.

    Raises
    ------
    TypeError
        args/kwargs do not match.

    Yields
    ------
    Iterator[tuple[str,Any]]
        Tuple of (arg_name, arg_value)

    """
    if len(args) > len(_slots):
        raise TypeError(f'too many arguments, maximum of {len(_slots)} allowed, got {len(args)}')
    if any((err:=key) for key in kwargs.keys() if key in _slots[:len(args)]):
        raise TypeError(f'got multiple arguments for {err}')
    if any((err:=k) not in _slots for k in kwargs.keys()):
        raise TypeError(f"unexpected keyword argument {err}")
    for slot in _slots:
        if args:
            arg, args = args[0], args[1:]
            yield slot, arg
        elif slot in kwargs:
            yield slot, kwargs[slot]
        elif slot in _defaults:
            yield slot, _defaults[slot]() if callable(_defaults[slot]) else _defaults[slot]
        elif slot in _required:
            raise TypeError(f"missing required arguments {slot}")


def kwarg_like(slots:Sequence[str], seq:Sequence[Any])->dict[str, Any]:
    """
    Convert sequence of inputs, unpacking tupledicts, into dictionary, with keys
    based on slots.

    Parameters
    ----------
    slots : Sequence[str]
        Strings used, and defining order of keys in output.
    seq : Sequence[Any]
        Sequence of values, if first is tupledict, unpack.

    Raises
    ------
    ValueError
        Bad input.
    TypeError
        Bad input.

    Returns
    -------
    dict[str, Any]
        Dict of seq converted to be as though it were kwargs.

    """
    if isinstance(seq, (dict, tupledict)):
        if any((err:=key) not in slots for key in seq.keys()):
            raise ValueError(f"unrecognized keyword argument {err}")
        return seq if isinstance(seq, dict) else seq.asdict
    kwargs = dict()
    if isinstance(seq[-1], (dict, tupledict)):
        kwargs.update(seq[-1].asdict if isinstance(seq, tupledict) else seq[-1])
        seq = seq[:-1]
    for slot, val in zip(slots, seq):
        if slot in kwargs:
            raise TypeError(f"mutiple arguments for {slot} found")
        kwargs[slot] = val
    if any((err:=key) not in slots for key in kwargs.keys()):
        raise ValueError(f"unrecognized keyword argument {err}")
    return kwargs


def _tuple_kwarg(val:Any)->tuple:
    """Ensure that val is a tuple"""
    val = tuple() if val is None else val
    val = val if isinstance(val, tuple) else (val, )
    return val


def _hexstr(val:int)->str:
    """Get a hex string, always 0x+16 characters, never negative"""
    hx = format(np.uint64(val), 'x')
    return '0x' + '0'*(16-len(hx)) + hx


class _DataLike:
    """
    Abstract class that functions like a dataclass, but guarunteeing the method uses
    slots, and allowing for dictionary-like key referencing and iteration.
    """
    _defaults:ClassVar[ImDict[str:Callable]] = ImDict()
    _required:ClassVar[frozenset[str]] = frozenset()
    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        for k, v in iter_funcinput(cls.__slots__, cls._defaults, cls._required, *args, **kwargs):
            setattr(obj, k, v)
        obj.__post_init__()
        return obj

    def __post_init__(self):
        pass

    @classmethod
    def class_fields(cls)->tuple[str,...]:
        """The allowed attributes of the object the object"""
        return tuple(slot for slot in cls.__slots__)
    
    def __setattr__(self, attr, value):
        if attr not in self.__slots__:
            if hasattr(type(self), attr) and isinstance(getattr(type(self), attr), property) and getattr(type(self), attr).fset is not None:
                getattr(type(self), attr).fset(self, value)
            else:
                raise AttributeError(f'{type(self).__name__} object has no attribute {attr}')
        else:
            super().__setattr__(attr, value)
        
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        self.__setattr__(key, value)
    
    def __contains__(self, key):
        return key in self.__slots__ and hasattr(self, key)
    
    def keys(self, skip:None|str|tuple[str,...]=None)->Iterator[str]:
        """
        Iterate over all attributes except those in skip.

        Parameters
        ----------
        skip : None | str | tuple[str,...], optional
            Set/Sequence of names of attributes to skip. The default is None.

        Yields
        ------
        Iterator[str]
            Names of each key.

        """
        skip = _tuple_kwarg(skip)
        yield from (key for key in self.__slots__ if hasattr(self, key) if key not in skip)
    
    def values(self, skip:None|str|tuple[str,...]=None)->Iterator[Any]:
        """
        Iterate over values pairs in data.

        Parameters
        ----------
        skip : None | str | tuple[str,...], optional
            Keys to skip in iteration. The default is None.

        Yields
        ------
        Iterator[Any]
            Values in data.

        """
        yield from (getattr(self, key) for key in self.keys(skip=skip))
    
    def items(self, skip:None|str|tuple[str,...]=None)->Iterator[tuple[str,Any]]:
        """
        Iterate over key:value pairs in data.

        Parameters
        ----------
        skip : None | str | tuple[str,...], optional
            Keys to skip in iteration. The default is None.

        Yields
        ------
        Iterator[tuple[str,Any]]
            (key value) pairs in data, first.

        """
        yield from ((key, getattr(self, key)) for key in self.keys(skip=skip))
    
    def __repr__(self):
        rep = (f'{key} = {getattr(self, key)}' for key in self.keys())
        return f'{self.__class__}\n' +'\n'.join(chain.from_iterable(wrap(string, subsequent_indent='    ') 
                                                                    for string in rep))

    def get(self, key:str, default:Any=None)->Any:
        """
        Retrieve the value of a key from data. If key not present, return default

        Parameters
        ----------
        key : str
            Key to retrieve.
        default : Any, optional
            Value to return if key not in data. The default is None.

        Returns
        -------
        Any
            Value of key, or default value if key not in data.

        """
        if key in self:
            return self[key]
        return default


class _ImDataLike(_DataLike):
    """
    Abstract class that functions like a frozen dataclass, but guarunteeing the 
    method uses slots, and allowing for dictionary-like key referencing and iteration.
    """
    _defaults:ClassVar[ImDict[str,Callable]] = ImDict()
    _required:ClassVar[frozenset[str]] = frozenset()
    _setfuncs:ClassVar[ImDict[str,Callable]] = ImDict()

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        for k, v in iter_funcinput(cls.__slots__, cls._defaults, cls._required, *args, **kwargs):
            super(_ImDataLike, obj).__setattr__(k, cls._setfuncs.get(k, _echo)(v))
        obj.__post_init__()
        return obj

    def __setattr__(self, attr, val):
        raise AttributeError("ImDataLike does not support assignment")
    
    def __delattr__(self, attr, val):
        raise AttributeError("ImData does not support deletion of attributes")


def _dict_update(dct:dict, update:dict)->dict:
    """Return updated dictionary copy, used to combine dictionaries without changing originals"""
    return {k:v for k, v in chain(dct.items(), update.items())}


def _return_update(dct:dict, update:dict)->dict:
    """Update dct with update, and return the updated dict (not a copy)"""
    dct.update(update)
    return update


def _recurse_copy(val:Any):
    """Recursively traverse object ensuring all subvalues are also copied"""
    if isinstance(val, (tuple, list)):
        return [_recurse_copy(v) for v in val]
    if isinstance(val, dict):
        return {k:_recurse_copy(v) for k, v in val.items()}
    if hasattr(val, 'copy'):
        return val.copy()
    return val


class SequenceDefaults:
    """
    Special defaults object type, each input kwarg is assumed to be a sequence
    of values.
    Input kwargs become attributes, but when accessed, the value is always a copy
    of the original, this way modifications can take place "inplace" when
    using an attribute as a function input etc.
    
    Stored values may also be accessed through indexing.
    If the index is a tuple/list, the output will be merged and broadcast.
    This means that if any key is a list/tuple (of dictionaries), the output will
    also be a sequence of dictionaries. If other indexes are only dictionaries,
    the output will be the first key with each internal dictionary updated with
    the dictionary of the second key. Values that are neither sequences or dictionaries
    are treated as single values. 
    
    Keys may also be wrapped in a 2-tuple of (str, key), in which case the value
    of the key (unwrapped if value is a sequence of non-dicts) becomes a dictionary
    of {str:key}.
    """
    def __init__(self, **kwargs):
        self._kwargs = kwargs
    
    def __getattr__(self, name):
        try:
            out = self[name]
        except KeyError:
            raise AttributeError(f"no attribute {name}")
        return out
        
    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self._kwargs:
                return _recurse_copy(self._kwargs[key])
        # This makes non-strings "pass through"
        if not isinstance(key, (list, tuple)):
            return key
        out = None
        for k in key:
            if isinstance(k, (list, tuple)):
                name, val = k[0], self[k[1]]
            else:
                name, val = None, self[k]
            if out is None:
                if isinstance(val, (list, tuple)):
                    out = val if name is None else [{name:v} for v in val]
                else:
                    out = val if name is None else {name:val}
            elif isinstance(out, (list, tuple)):
                val = val if isinstance(val, (list, tuple)) else repeat(val)
                for o, v in zip(out, val):
                    if name is None:
                        o.update(_recurse_copy(v))
                    else:
                        o.update({name:_recurse_copy(v)})
            elif isinstance(val, (list, tuple)):
                if name is None:
                    out = [_return_update(v, out) for v in val]
                else:
                    out = [_return_update({name:v}, out) for v in val]
            else:
                if name is None:
                    out.update(val)
                else:
                    out.update({name:val})
        return out
    
    def add(self, key:str, val:Any)->None:
        """
        Add a Sequence or dictionary to the SequenceDefaults
        
        Arguments
        ---------
        key : str
            Name new attribute will take
        val: Any
            The object, dictionary or sequence of dictionaries the key will take
        """
        self._kwargs[key] = val


def _ndarray_to_str(string:np.ndarray|bytes)->str:
    """Convert byte-like array to string"""
    if isinstance(string, np.ndarray):
        string = np.atleast_1d(string)
        if string.size == 1:
            string = string[tuple(0 for _ in range(string.ndim))].decode()
        else:
            string = string.astype(np.str_)
    else:
        string = string.decode()
    return string

##################################################
### Functions for working with nested dictionaries
##################################################
def _nested_in(dct:dict, keys:tuple[Hashable,...])->bool:
    """Check if keys is in dct, where dct is assumed to be a *nested* dictionary"""
    if keys[0] in dct:
        return True if len(keys) == 1 else _nested_in(dct[keys[0]], keys[1:])
    return False


def _nested_get(dct:dict, keys:tuple, default:Any=None)->Any:
    """Key keys from nested dictionary dct"""
    for k in keys:
        if k in dct:
            dct = dct[k]
        else:
            return default
    return dct


def _nested_set(dct:dict, keys:tuple[Hashable,...], val:Any)->Any:
    """Set keys to val in *nested* dictionary dct- will create super-nested dictionaries"""
    for key in keys[:-1]:
        if key not in dct:
            dct[key] = dict()
        dct = dct[key]
    dct[keys[-1]] = val
    return val


def _nested_pop(dct:dict, keys:tuple[Hashable,...], default:Any=None)->Any:
    """Pop key from nested dictionary dct"""
    final = len(keys) - 1
    for i, k in enumerate(keys):
        if k in dct:
            if i != final:
                dct = dct[k]
            else:
                val = dct.pop(k)
        else:
            return default
    return val


def _inner_nested_items(dct:dict, outer:tuple[Hashable,...])->Iterator[tuple[Hashable,Any], bool, None]:
    """Iterate over items (key, value pairs) of outer insided nested dictionary"""
    diter = iter(dct.items())
    for key, val in diter:
        if isinstance(val, dict):
            yield from _inner_nested_items(val, outer+(key,))
        else:
            adv = yield outer + (key, ), val
            if adv:
                break


def _nested_items(dct:dict)->Iterator[tuple[Hashable,Any], bool, None]:
    """Iterator over key value pairs in nested dictoinary"""
    yield from _inner_nested_items(dct, tuple())


def _indent(val:str, indent:int)->str:
    """Indent a string"""
    return ' '*indent+('\n'+' '*indent).join(val.split('\n'))

########################
### Additional functions
########################
def union_multi(*args:Number|np.ndarray)->np.ndarray:
    """
    Repeted application of np.uion1d on each argument.

    Parameters
    ----------
    *args : number | np.ndarray
        Number or array to create union of.

    Returns
    -------
    np.ndarray
        Union of all arguments.

    """
    if len(args) == 0:
        return np.array([], dtype=np.int64)
    if len(args) == 1:
        return args[0]
    union = np.union1d(args[0], args[1])
    for arr in args[2:]:
        union = np.union1d(union, arr)
    return union


def intersect_multi(*args:Number|np.ndarray)->np.ndarray:
    """
    Repeted application of np.intersect1d on each argument.

    Parameters
    ----------
    *args : number | np.ndarray
        Number or array to create union of.

    Returns
    -------
    np.ndarray
        Union of all arguments.

    """
    if len(args) == 0:
        return np.array([], dtype=np.int64)
    intersect = args[0]
    for arg in args[1:]:
        intersect = np.intersect1d(intersect, arg)
    return intersect


def _arrays_identical(*args:np.ndarray)->bool:
    """Check if all arrays in arguments are exactly equal to one another,
    returns False automatically if any arrays have a different shape than the
    others 
    """
    ref = args[0]
    for arg in args[1:]:
        if arg.shape != ref.shape or np.any(arg != ref):
            return False
    return True


def _arrays_unique(*args:np.ndarray)->bool:
    """Test if all elements in all arrays in args are unique"""
    all_det = np.concatenate(args)
    return all_det.size == np.unique(all_det).size


def _arrays_geq(arr0:np.ndarray, arr1:np.ndarray)->bool:
    """
    Test if all elements of arr1 are in arr0, ie if arr0 is >= arr1 in size
    and elements
    """
    return all(i in arr0 for i in arr1)


def _large_equal(val0:list|tuple|np.ndarray, val1:list|tuple|np.ndarray)->bool:
    """
    Determine if all values in val0 are the same as in val1, works recursively
    on lists, tuples, and numpy arrays

    Parameters
    ----------
    val0 : list, tuple, numpy array
    val1 : list, tuple, numpy array

    Returns
    -------
    bool
        
    """
    if type(val0) != type(val1):
        return False
    elif isinstance(val0, np.ndarray):
        if val0.shape == val1.shape:
            return np.all(val0 == val1)
        return False
    elif isinstance(val0, (list, tuple)):
        if len(val0) == len(val1):
            return all(_large_equal(v0, v1) for v0, v1 in zip(val0, val1))
        else:
            return False
    else:
        return val0 == val1


def dict_equal(*dicts:dict)->bool:
    """
    Determine if all key:value pairs in set of dictionaries are the same.

    Parameters
    ----------
    *dicts : dict
        Arbitrarily long set of dictionaries.

    Returns
    -------
    bool
        if all input dictionaries contain the same elements.

    """
    key0 = dicts[0].keys()
    comp = np.all([key0 == dct.keys() for dct in dicts])
    if comp:
        for key, val0 in dicts[0].items():
            for dct in dicts[1:]:
                if not _large_equal(val0, dct[key]):
                    return False
    return comp


def s_equal(*lsts:Sequence)->bool:
    """
    Test if all sequences contain the same set of elements

    Parameters
    ----------
    *lsts : Sequence
        Sequences to test if they are equal.

    Returns
    -------
    bool
        If all sequences contain exactly the same elements as each other.

    """
    return all(f in l0 for f, l0 in product(chain.from_iterable(l for l in lsts), lsts))


def _nested_dict_merge(dctin:dict, dctnew:dict, strict:bool, prekey:Sequence[Hashable]=None)->dict:
    prekey = (prekey, ) if prekey is None else tuple(prekey)
    for key, val in dctnew.items():
        if key in dctin:
            if isinstance(val, dict) and isinstance(dctin[key], dict):
                _nested_dict_merge(dctin[key], val, strict, prekey=prekey+(key,))
            elif strict:
                if not _large_equal(dctin[key], val):
                    raise ValueError(f"Cannot change value of {prekey+(key,)}")
            else:
                dctin[key] = val
        else:
            dctin[key] = val
    return dctin


def _nested_dict_mergekey(update:dict, keys:Sequence[str], val:dict, strict:bool)->dict:
    """
    Merge nested dictionaries, 
    strict means existing shared keys must be equal otherwise raise an error, 
    prekey indicates recursive call from larger dictionary (rarely used except by during recursion)
    """
    for key in keys[:-1]:
        if key not in update:
            update[key] = dict()
        update = update[key]
    key = keys[-1]
    if key in update:
        if isinstance(update[key], dict) and isinstance(val, dict):
            _nested_dict_merge(update[key], val, strict)
        elif strict:
            if not _large_equal(update[key], val):
                raise ValueError(f"{keys} already exists and values are not compatible")
        else:
            update[key] = val
    else:
        update[keys[-1]] = val
    return update


def _expand_by_index(indices, values, start):
    """
    Return a list mapping the values in *values* to their cooresponding indeces in 
    *indeces*, useful for putting all fields in an HDF5 file in numerical ascending
    order. *start* indicates the starting index (should be either 0 or 1), because
    photon HDF5 has make inconsistent choices on which fields should be numberedf
    from 0 like photon_data, and which should be number from 1 like alex_period

    """
    if len(indices) != len(values):
        raise ValueError("indices and values must be the same length")
    length = max(indices) + 1 - start
    out = [None for _ in range(length)]
    for idx, value in zip(indices, values):
        out[idx-start] = value
    return out


def _insert_list(dct:dict[Hashable,list], key:Hashable, value:any):
    """Assumes dct is a dictionary of lists, inserts value at end of list in key"""
    if key in dct:
        dct[key].append(value)
    else:
        dct[key] = [value, ]


def _numdict_to_tuple(spec:dict, start:int=1)->tuple:
    """Convert dictionary with integer keys into tuple, ordere by the numbers of the keys"""
    order = sorted(spec.keys())
    return tuple(spec.get(i, None) for i in order)


def _ascending_dict_to_tuple(spec:dict, start:int|None=1)->tuple|None:
    """Return tuple of values in a dictionary with integer keys, of range(start,...)
    and fills missing numbers with None"""
    if start is None:
        start = min(spec.keys())
    stop = max(spec.keys()) + 1
    return tuple(spec.get(i, None) for i in range(start, stop))

    
def _ascending_dict_to_tuple_strict(spec:dict, start:int=1)->tuple|None:
    """
    Build a tuple from a dictionary of ascending keys, 
    returns None if anykey is skipped
    """
    if any(i not in spec for i in range(start, len(spec)+start)):
        return None
    return tuple(spec[i] for i in range(start, len(spec)+start))


def enumerate_intersects(*args:Sequence[np.ndarray])->Iterator[tuple[int,np.ndarray]]:
    """
    Enumerate the intersection of individual elements of all input arrays.
    Returns (index, intersect), where index is an int, and intersect is the intersection
    of the particular intersection of n_i, m_j... arrays.

    Parameters
    ----------
    *args : Sequence[np.ndarray]
        Slice of a given cross section of indexes.

    Raises
    ------
    ValueError
        Index occurs multiple times in a single arg, or union of each arg is different.

    Yields
    ------
    tuple[int, np.ndarray]
        enumeration of intersect, ie (index, intersect).

    """
    unions = [union_multi(*arg) for arg in args if len(arg)]
    union = union_multi(*unions)
    if any(np.setdiff1d(union, u).size for u in unions):
        raise ValueError("Inconsistent detector ids in inputs")
    strides = np.cumprod([1,] + [len(arg) for arg in args[:0:-1]], dtype=np.uint8)[::-1]
    args = tuple(arg if len(arg) else (union, ) for arg in args)
    for cmbs in product(*(enumerate(arg) for arg in args)):
        ijk, sets = zip(*cmbs)
        yield np.sum(np.array(ijk, dtype=np.uint8)*strides), intersect_multi(*sets)


def _delayed_iter(iterator:Iterator[Any], delay:int, offset:int)->Iterator[Any]:
    """
    Iterator that stores ``delay`` items in buffer before returning, and skips
    offset first values

    Parameters
    ----------
    iterator : Iterator[Any]
        Iterator to delay.
    delay : int
        number of iteration to delay returning value.
    offset : int
        number of iterations to skip at the beginning.

    Yields
    ------
    Iterator[Any]
        Iterator that has already evaluated delay ahead.

    """
    iterator = iter(iterator)
    for _, _ in zip(range(offset), iterator):
        pass
    accum = [v for _, v in zip(range(delay-offset), iterator)]
    for v in iterator:
        accum.append(v)
        yield accum.pop(0)


def _masked_iter(gen:Iterator[Any], mask:np.ndarray[np.bool_])->Iterator[Any]:
    """
    Speical iterator over an iterator, not yielding false values in mask.
    Ensures iterator called to the end, even if final value of mask if False.

    Parameters
    ----------
    gen : Iterator[Any]
        Iterator to iterate over, removing mask.
    mask : np.ndarray[np.bool_]
        mask array for which values in iterator to return.

    Raises
    ------
    ValueError
        gen is not same size as mask.

    Yields
    ------
    Iterator[Any]
        masked iterator that calls gen to completion on final expexted call
        to next.

    """
    gen = iter(gen)
    if isinstance(mask, slice):
        for _ in gen:
            pass
        return
    i, size = 0, mask.shape[0]
    while i < size and not mask[i]:
        next(gen)
        i += 1
    while i < size:
        out = next(gen)
        i += 1
        while i < size and not mask[i]:
            next(gen)
            i += 1
        yield out
    try:
        next(gen)
    except StopIteration:
        pass
    else:
        raise ValueError("too many values in iterator")


class _Arr_slc:
    """
    Instances have items that are the key, ie stores no data, and echos the
    input of ``__getitem__``
    """
    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = index, 
        return index

arr_slc = _Arr_slc()


def _dim_comp(s:int, d:int|slice)->bool:
    if isinstance(d, slice):
        mn, mx, c = d.indices(s+1)
        if c != 1:
            raise ValueError("specifying valid range of values does not permit non-1 strides")
        return s >= mn and s <= mx
    return s == d


def _dimscompare(shape:tuple[int], dims:tuple[int|slice,...])->bool:
    """Determine if specification of dims (indexes) compatible with array of shape shape"""
    nellipsis = sum(d is Ellipsis for d in dims)
    if nellipsis > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    elif nellipsis == 0 and len(shape) != len(dims):
        return False
    elif nellipsis == 1 and len(shape) + 1 < len(dims):
        return False
    for s, d in zip(shape, dims):
        if d is Ellipsis:
            for s, d in zip(reversed(shape), reversed(dims)):
                if d is Ellipsis:
                    break
                if _dim_comp(s, d):
                    continue
                return False
            break
        if _dim_comp(s, d):
            continue
        return False
    return True


def _dim_mask_iter(index:np.ndarray[np.bool_], shape:tuple[int])->tuple[int,tuple[int,...],Iterator]:
    """
    _dim_iter inner function for when index is boolean mask
    returns size, shape, [iterator over mask]
    """
    if index.size!= np.prod(shape):
        raise ValueError("incompatible shape")
    if index.ndim != 1 and index.shape != shape:
        raise ValueError("incompatible shape")
    size = index.sum()
    return size, (size,), (idx for idx in product(*(range(s) for s in shape)) if index[idx])


def _n_to_index(n:int, shape:tuple[int,...])->tuple[int,...]:
    """convert int (1D position in array), to tuple[int,...] index of array with shape shape """
    strides = np.cumprod((shape+(1,))[::-1])[::-1]
    return tuple((n % strides[:-1]) // strides[1:])


def _dim_iter_tuple(index:IndexType, shape:tuple[int,...])->tuple[int,tuple[int,...],Iterator]:
    """_dim_iter for index when index is tuple, returns size, shape, [iterator indexe(s)]"""
    N = np.arange(np.prod(shape)).reshape(shape)
    Ni = N[index]
    return Ni.size, Ni.shape, (_n_to_index(i) for i in Ni.reshape(-1))


def _dim_iter(index:IndexType, shape:tuple[int,...])->tuple[int,tuple[int,...],Iterator]:
    """
    Create size, shape, [iterator over indexes] 
    of an index (multiple forms), [shape of array]
    """
    if isinstance(index,np.ndarray) and index.dtype == np.bool_:
        return _dim_mask_iter(index, shape)
    if not isinstance(index, tuple):
        index = (index, )
    return _dim_iter_tuple(index, shape)


def _mask_loc(sub_locs:tuple[bool,...], loc:tuple[int])->tuple[int,...]:
    """mask loc based on gate_loc"""
    return tuple(l for g, l in zip(sub_locs, loc) if g)


def broadcast_truthtable(truthtable:np.ndarray[np.bool_], seqs:Sequence[Hashable|Any],
                         expand_func:Callable[[Any],Sequence[Hashable]],
                         sort_key:Callable[[Hashable],int]=None, dtype:np.dtype=np.bool_,
                         )->tuple[np.ndarray[np.bool_],list[Hashable]]:
    r"""
    Broadcast/expand a truthtable based on truthables infered from seqs using
    expand_func. seqs must be a sequence of objects that can be converted into
    "sub" truthtables, and seqeuences of sortables.

    Parameters
    ----------
    truthtable : np.ndarray[np.bool\_]
        Truthtable to be expanded.
    seqs : Sequence[Hashable|Any]
        Sequence of objects that may or may not need "expanding".
    expand_func : Callable[[Any],Sequence[Hashable]]
        Function that takes elements of seqs, and returns 2 tuple of 
        sequence of sub-elements, and truthtable (ndim truthtable = len(sequence).
    sort_key : Callable[[Hashable],int], optional
        key callable for key of sorted. The default is None.
    dtype : np.dtype, optional
        datatype of returned truthtable. The default is np.bool\_.

    Returns
    -------
    tt : np.ndarray[np.bool\_]
        Expanded truthtable.
    all_subs : list[Hashable]
        Sorted sequence defining .

    """
    # build set of all gates used
    expand, truthtables = zip(*(expand_func(element) for element in seqs))
    # soted(set(..)) removes duplicate gates
    all_subs = sorted(set(chain.from_iterable(expand)),key=sort_key)
    idxmap = tuple(tuple(sub in subs for sub in all_subs) for subs in expand)
    # allocate new table
    tt = np.empty([2 for _ in range(len(all_subs))], dtype=dtype)
    # iterate over every position of output table and assign appropriate value
    for loc in product(*(range(2) for _ in range(len(all_subs)))):
        sloc = tuple(int(tt[_mask_loc(idx, loc)]) for tt, idx 
                     in zip(truthtables, idxmap))
        tt[loc] = truthtable[sloc]
    return tt, all_subs


class _FileFinalizer:
    """
    Internal class for managing open files. When owner is deleted, will close file.
    """
    _files:ClassVar[dict[Hashable:"_FileFinalizer"]] = dict()
    def __new__(cls, file:tb.File, owner:int):
        if not file.isopen:
            return
        ownerwr = weakref.ref(owner)
        if file in cls._files:
            obj = cls._files[file]
        else:
            obj = object.__new__(cls)
            obj._file = file
            obj._finalizers = weakref.WeakKeyDictionary()
            cls._files[file] = obj
        obj._finalizers[owner] = weakref.finalize(owner, obj.finalize_owner, ownerwr)
        return obj

    @property
    def file(self)->tb.File:
        """tb.File being managed"""
        return self._file

    def finalize_owner(self, weakref:weakref.ReferenceType=None, strict:bool=False)->None:
        """
        Close owner if no other references to owner exist.

        Parameters
        ----------
        weakref : weakref.ReferenceType, optional
            DESCRIPTION. The default is None.
        strict : bool, optional
            If True, close file regardless of other open references.
            The default is False.

        """
        if weakref is not None and weakref() is not None:
            self._finalizers.pop(weakref(), None)
        if not self._finalizers or strict:
            if self._file.isopen:
                self._file.close()
            _FileFinalizer._files.pop(self._file, None)
        elif not self._file.isopen:
            _FileFinalizer._files.pop(self._file, None)


class _GroupFuture:
    """
    Internal class for creating a "promise" of a tables tb.Group
    """
    _groupfuture:None|Callable[[],tb.Group]|tb.Group
    _parent:weakref.ref
    _filefuture:None|tb.File
    _callback:list[Callable[["_GroupFuture"],None]]

    def __init__(self, group:None|Callable[[],tb.Group]|tb.Group, 
                 parent:Any=None, callback:Callable[["_GroupFuture"],None]=None,
                 file:tb.File=None):
        self._parent = None if parent is None else weakref.ref(parent)
        self._groupfuture = self._check_groupfuture(group)
        if isinstance(group, tb.Group):
            if file is not None and file != group._v_file:
                warnings.warn("File of group different from expected")
            file = group._v_file
        self._filefuture = self._check_filefuture(file)
        if callable(group):
            self._callback = list() if callback is None else [callback,]
        elif isinstance(group, tb.Group) and callback is not None:
            callback(self, group)
            self._callback = list()
        else:
            self._callback = list()

    @classmethod
    def create_dependant(cls, group:None|Callable[[],tb.Group]|tb.Group, parent:"_GroupFuture", 
                         callback:None|Callable[["_GroupFuture"],None]=None)->"_GroupFuture":
        """
        Create a new _GroupFuture that is a subgroup of parent.

        Parameters
        ----------
        group : None | Callable[[],tb.Group] | tb.Group
            Group or Callable creating group.
        parent : _GroupFuture
            GroupFuture on which new future will be dependant.
        callback : Callable[["_GroupFuture"],None], optional
            Callback when group is created,, will take parent as only argument 
            when called. The default is None.

        Returns
        -------
        _GroupFuture
            Newly created GroupFuture, whic.

        """
        obj = object.__new__(cls)
        obj._parent = parent._parent
        obj._filefuture = parent._filefuture
        if obj._parent is not None and obj._parent() is None:
            obj._groupfuture = None
        else:
            obj._groupfuture = cls._check_groupfuture(group)
        if callable(obj._groupfuture):
            obj._callback = list() if callback is None else [callback, ]
        elif isinstance(obj._groupfuture, tb.Group) and callback is not None:
            callback(obj)
            obj._callback = list()
        else:
            obj._callback = list()
        return obj

    @classmethod
    def _check_groupfuture(cls, groupfuture:None|Callable[[],tb.Group]|tb.Group)->None|Callable[[],tb.Group]|tb.Group:
        """
        Internal funciton, used to verify input to a create method is a 
        valid type from which to crate a GroupFuture.
        """
        if groupfuture is None or callable(groupfuture) or isinstance(groupfuture, tb.Group):
            return groupfuture
        raise TypeError(f"invalid type ({type(groupfuture).__name__} for group future)")

    @classmethod
    def _check_filefuture(cls, filefuture:None|tb.File)->None|tb.File:
        """Internal function to ensure filefuture argument is valid filefuture"""
        if filefuture is None or isinstance(filefuture, tb.File):
            return filefuture
        raise TypeError("filefuture must be None or table.File")

    def _verify_exists(self):
        """
        Internal function, checks all necessary compenents exists, 
        if not, reset _groupfuture attr to None
        """
        if self._parent is not None and self._parent() is None:
            self._groupfuture = None
        if self._filefuture is not None and not self._filefuture.isopen:
            self._groupfuture = None

    def _create(self)->tb.Group:
        """
        If hdf5 group has not already been created, create it, then return HDF5 Group.

        Returns
        -------
        tb.Group
            HDF5 group for GroupFuture.

        """
        self._verify_exists()
        if self._groupfuture is None:
            raise AttributeError("no group exists for interaction")
        if callable(self._groupfuture):
            group = self._groupfuture()
            if not isinstance(group, tb.Group):
                self._groupfuture = None
                self._parent = None
                raise TypeError(f"callable returned wrong type, must create a table Group, got {type(group).__name__}")
            if self._filefuture is not None and self._filefuture != group._v_file:
                warnings.warn("File created different from expected")
            self._filefuture = group._v_file
            self._groupfuture = group
            for callback in self._callback:
                callback(self)
            delattr(self, '_callback')
        return self._groupfuture

    @property
    def _group(self)->tb.Group:
        """HDF5 Group, accessing this attribute will create the group if it is not yet created"""
        return self._create()

    @property
    def _creatable(self)->bool:
        """
        Whether the group has been **or can be** be created- 
        if not linked to HDF5 file, returns False
        """
        self._verify_exists()
        return self._groupfuture is not None

    @property
    def _created(self)->bool:
        """Whether the group has already been created"""
        self._verify_exists()
        return isinstance(self._groupfuture, tb.Group)

    @property
    def _groupcurrent(self)->None|tb.Group:
        """'Current' state of group: None if group is not created, else the group"""
        if self._created:
            return self._groupfuture
        return None

    @property
    def _file(self)->None|tb.File:
        """tb.File of groupfuture, will create if it does nto already exist"""
        self._create()    
        return self._filefuture

    def __contains__(self, key):
        self._verify_exists()
        if isinstance(self._groupfuture, tb.Group):
            return key in self._groupfuture
        return False

    def __getattr__(self, attr):
        self._verify_exists()
        if isinstance(self._groupfuture, tb.Group):
            return getattr(self._groupfuture, attr)
        raise AttributeError(f"non-created group has no attribute {attr}")

    def __getitem__(self, key):
        self._verify_exists()
        if isinstance(self._groupfuture, tb.Group):
            return self._groupfuture[key]
        raise KeyError(key)

    def _create_group(self, name:str, **kwargs)->tb.Group:
        """
        Create a group inside GroupFuture._group with name name. Kwargs passed
        to tb.File.create_group

        Parameters
        ----------
        name : str
            Name of group to create.
        **kwargs : Any
            Kwargs passed to tb.File.create_group.

        Returns
        -------
        tb.Group
            Newly created HDF5 group.

        """
        return self._group._v_file.create_group(self._group, name, **kwargs)

    def _create_array(self, name:str, val:Any, arrtp:Literal['a','c','e']='a', **kwargs)->tb.Array:
        """
        Create array within GroupFuture._group with name name and set to val.
        arrtp sets type of array created, 
        kwargs passed to relevant create_arrray method


        Parameters
        ----------
        name : str
            Name of new HDF5 array.
        val : Any
            Array to write to HDF5 file.
        arrtp : Literal['a','c','e'], optional
            Type of array created. 
            
            - 'a': standard array, '. 
            - 'c': carray
            - 'e': earray
            
            The default is 'a'.
        **kwargs : Any
            kwargs passed to create_[c/e]array.

        Returns
        -------
        tb.Array
            HDF5 array object just created.

        """
        if arrtp == 'a':
            return self._group._v_file.create_array(self._group, name, val, **kwargs)
        if arrtp == 'c':
            return self._group._v_file.create_carray(self._group, name, obj=val, **kwargs)
        if arrtp == 'e':
            return self._group._v_file.create_earray(self._group, name, obj=val, **kwargs)

    def _create_groupfuture(self, name:str, postinit:Callable[[tb.Group],None]=None, **kwargs)->"_GroupFuture":
        """
        Create a _GroupFuture under current group.

        Parameters
        ----------
        name : str
            Name of sub-group.
        postinit : Callable[[tb.Group],None], optional
            Function called after group creation, takes 1 argument: the HDF5 group. 
            The default is None.
        **kwargs : Any
            Passed to tb.File.create_group upon creation.

        Returns
        -------
        _GroupFuture
            Group future for new group.

        """
        if not self._creatable:
            return type(self).create_dependant(None, self)
        if name in self:
            return _GroupFuture.create_dependant(self[name], self)
        # function for creating group
        def create_group()->tb.Group:
            group = self[name] if name in self else self._create_group(name, **kwargs)
            if postinit is not None:
                postinit(group)
            return group
        # if self is created, go all the way, otherwise make callable and add callback
        if self._created:
            return type(self).create_dependant(create_group(), self)
        out = _GroupFuture.create_dependant(create_group, self)
        # this way if when self is created, dependants can also be created if their group already exists
        self._callback.append(lambda g: out.create() if name in self else None)
        return out

    def _create_arrayfuture(self, name:str, val:Any, **kwargs)->None|tb.Array|Callable[[],tb.Array]:
        """
        Create array in _GroupFuture, if group not created, return Callable to create.

        Parameters
        ----------
        name : str
            Name of array to be created.
        val : Any
            Array to write to disk, when array is created.
        **kwargs : Any
            Kwargs passed to _GroupFuture._create_array.

        Returns
        -------
        None|tb.Array|Callable[[],tb.Array]
            None if not creatable, if created, return HDF5 array, if creatable
            return Callable that when called creates array and return HDF5 array.

        """
        if not self._creatable:
            return None
        if name in self:
            return self[name]
        if self._created:
            return self._create_array(name, val, **kwargs)
        return lambda : self._create_array(name, val, **kwargs)

    def _assign_parent(self, parent:Any)->None:
        """
        Set parent of _GroupFuture.

        Parameters
        ----------
        parent : Any
            Object that when deleted also renders _GroupFuture uncreatedable (parent).

        """
        self._parent = weakref.ref(parent)

    def _add_callback(self, callback:Callable[["_GroupFuture"],None])->None:
        """
        Add function that is called when group is created. 
        Used primarily to set finalizer.

        Parameters
        ----------
        callback : Callable[["_GroupFuture"],None]
            Function, called after group is created, takes the GroupFuture as only argument.

        Raises
        ------
        TypeError
            Bad calback.

        """
        self._verify_exists()
        if not callable(callback):
            raise TypeError("callack must be callable")
        if callable(self._groupfuture):
            self._callback.append(callback)


GroupArg = None|tb.Group
GroupFuture = None|Callable[[],tb.Group]|tb.Group|_GroupFuture

def weakref_alive_test(ref:None|weakref.ReferenceType)->bool:
    """
    ref must be None or weakreference object, test if reference is alive, None->False

    Parameters
    ----------
    ref : None | weakref.ReferenceType
        Weak-reference to test if allive.

    Returns
    -------
    bool
        Whether references is alive.

    """
    if ref is None:
        return False
    return ref() is not None


###############################################################################
###################### Dynamic has-numba based decorators #####################
###############################################################################
try:
    import numba
except:
    has_numba = False
else:
    has_numba = True

def fjit(*args, **kwargs):
    """
    Alias for numba.jit, unless ``has_numba = False``, then just pass original
    function.
    """
    if has_numba:
        return numba.jit(*args, **kwargs)
    else:
        return lambda x: x
    
    
class _fdummy:
    """Echo type for when getting item or attr should return None"""
    def __getattr__(self, attr):
        pass
    def __getitem__(self, item):
        pass


_dummy = _fdummy()


class _fnumba:
    """
    Alias of numba type, unless ``has_numba = False`` in which case echos input
    """
    def __getattr__(self, attr):
        if has_numba:
            return getattr(numba, attr)
        return _dummy
    
    @property
    def has_numba(self)->bool:
        return has_numba


fnumba = _fnumba()
