#
# FRETBursts - A single-molecule FRET burst analysis toolkit.
#
# Copyright (C) 2014 Antonino Ingargiola <tritemio@gmail.com>
#
"""
Misc utility functions
"""

import os
import sys
from warnings import warn
from itertools import chain, product
from collections.abc import Iterable, Callable, Sequence, Iterator
from textwrap import wrap
from typing import ClassVar, Hashable, Any, Union
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import tables as tb


def _tuple_array(array):
    if isinstance(array, np.ndarray):
        if array.ndim == 0:
            return array.reshape(1)[0]
        return tuple(_tuple_array(a) for a in array)
    if isinstance(array, tuple):
        return tuple(_tuple_array(v) for v in array)
    if isinstance(array, Callable):
        try:
            return 'callable', array.__name__
        except AttributeError:
            return array
    return array


def _as_sortable(val):
    if np.issubdtype(type(val), np.number) or isinstance(val, str):
        return val
    return hash(val)


def _make_sortable(val):
    if isinstance(val, np.ndarray):
        val = tuple(val)
    if isinstance(val, tuple):
        val = tuple(_make_sortable(v) if isinstance(v, tuple) else _as_sortable(v) for v in val)
    return val


def hash_array(array):
    return hash(_tuple_array(array))


def _is_list_of_arrays(obj):
    return isinstance(obj, list) and np.all([isinstance(v, np.ndarray) for v in obj])


def _eq(a, b):
    """
    equals function that ensures objects are
    1. Of same type (int 1 != float 1.0)
    2. Compares sequences an an all fassion, requiring both sequences be of same length
    3. Never fails
    """
    if type(a) != type(b): return False
    if isinstance(a, np.ndarray):
        if a.shape != b.shape: return False
        if a.dtype == np.object_:
            if b.dtype != np.object_: return False
            return all(_eq(aa, bb) for aa, bb in zip(a.ravel(), b.ravel()))
        return np.all(a == b)
    if isinstance(a, Sequence) and not isinstance(a, str):
        if len(a) != len(b): return False
        return all(_eq(aa, bb) for aa, bb in zip(a, b))
    return a == b
        

def _echo(arr:Any)->Any:
    return arr


def make_objectarray(arrs):
    out = np.empty(len(arrs), dtype=np.object_)
    for i, arr in enumerate(arrs):
        out[i] = arr
    return out
        

def make_immutable(array, copy=False):
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


def iterator_to_object_array(itr:iter, final:tuple[type], read:Callable):
    if isinstance(itr, final):
        return read(itr)
    arr_list = list(iterator_to_object_array(sub, final, read) for sub in itr)
    out = np.empty(len(arr_list), dtype=np.object_)
    for i, arr in enumerate(arr_list):
        out[i] = arr
    return out


def _iter_tbgroup_numeric(group:tb.Group, prefix:str):
    i = 0
    while (g:=f'{prefix}{i}') in group:
        yield group[g]
        i += 1


class ImDict(dict):
    def __setitem__(self, k, v):
        raise AttributeError("ImDict does not support assignment")
    
    def update(self, *args, **kwargs):
        raise TypeError("update is disabled for fixed dict")
        
    def pop(self, *args, **kwargs):
        raise TypeError("pop is disabled for fixed dict")


class FixedDict(dict):
    def __setitem__(self, key, value):
        if key in self:
            raise ValueError(f"{key} already set")
        if isinstance(value, np.ndarray):
            value = make_immutable(value)
        super().__setitem__(key, value)
    
    def update(self, *args, **kwargs):
        update = dict(*args, **kwargs)
        if any((err:=k) in self for k in update.keys()):
            raise KeyError(f'{err} already set')
        super().update(update)
    
    def pop(self, *args, **kwargs):
        raise TypeError("pop is disabled for fixed dict")


class tupledict(tuple):
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
    def from_order(cls, order, *args, defaults_:dict=None, **kwargs):
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
            raise ValueError("names {kwargs.keys()} not in order")
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
        return self.__getitem__(attr)
    
    def __hash__(self):
        return hash(tuple((k, _tuple_array(v) ) for k, v in self.items()))
    
    def __eq__(self, other):
        if not isinstance(other, tupledict): return False
        if len(other) != len(self): return False
        return all(sk==ok and _eq(sv, ov) for (sk, sv), (ok, ov) in zip(self.items(), other.items()))
    
    def keys(self):
        yield from (k for k, v in self)

    def values(self):
        yield from (v for k, v in self)

    def items(self):
        return iter(self)
    
    def get(self, key:str, default:any=None)->Any:
        for k, v in self:
            if k == key:
                return v
        return default

    def has_val(self, o):
        return any(o == v for v in self.values())
    
    def __contains__(self, k):
        if hasattr(k, '__index__'):
            return k < len(self) if k >= 0 else (abs(k) - 1) < len(self)
        return any(k == key for key in self.keys())
    
    @property
    def asdict(self):
        return {k:v for k, v in self.items()}


class MutDict(dict):
    _mut = False
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        for key, value in self.items():
            if type(value) == dict:
                super().__setitem__(key, MutDict(value))

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._mut = True

    @property
    def mut(self):
        return self._mut or any(v.mut for v in self.values() if type(v) == MutDict)

def iter_funcinput(slots:tuple[str], defaults:ImDict[str:Callable], required:frozenset[str], *args, **kwargs):
    if len(args) > len(slots):
        raise TypeError(f'too many arguments, maximum of {len(slots)} allowed, got {len(args)}')
    if any((err:=key) for key in kwargs.keys() if key in slots[:len(args)]):
        raise TypeError(f'got multiple arguments for {err}')
    if any((err:=k) not in slots for k in kwargs.keys()):
        raise TypeError(f"unexpected keyword argument {err}")
    for slot in slots:
        if args:
            arg, args = args[0], args[1:]
            yield slot, arg
        elif slot in kwargs:
            yield slot, kwargs[slot]
        elif slot in defaults:
            yield slot, defaults[slot]() if callable(defaults[slot]) else defaults[slot]
        elif slot in required:
            raise TypeError(f"missing required arguments {slot}")


def _tuple_kwarg(val:Any)->tuple:
    val = tuple() if val is None else val
    val = val if isinstance(val, tuple) else (val, )
    return val


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
    def class_fields(cls):
        return tuple(slot for slot in cls.__slots__)

    def __setattr__(self, attr, value):
        if attr not in self.__slots__:
            raise AttributeError(f'{type(self).__name__} object has no attribute {attr}')
        super().__setattr__(attr, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __contains__(self, key):
        return key in self.__slots__ and hasattr(self, key)

    def keys(self, skip=None):
        if skip is None:
            skip = tuple()
        elif isinstance(skip, str):
            skip = (skip, )
        yield from (key for key in self.__slots__ if hasattr(self, key) if key not in skip)

    def values(self, skip=None):
        yield from (getattr(self, key) for key in self.keys(skip=skip))

    def items(self, skip=None):
        yield from ((key, getattr(self, key)) for key in self.keys(skip=skip))

    def __repr__(self):
        rep = (f'{key} = {getattr(self, key)}' for key in self.keys())
        return f'{self.__class__}\n' +'\n'.join(chain.from_iterable(wrap(string, subsequent_indent='    ') 
                                                                    for string in rep))


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
    

class HistData:
    """Stores histogram counts and bins and provides derived fields.

    Attributes:
        counts (array, ints): array of counts in each bin
        bins (array): array of bin edges. Size is size(counts) + 1.
        bincenters (array): array of bin  centers. Size is size(counts).
        pdf (array, floats): array of normalized counts (aka PDF)
    """
    def __init__(self, counts, bins):
        self.counts = counts
        self.bins = bins
        self.binwidth = bins[1] - bins[0]

    @property
    def bincenters(self):
        if not hasattr(self, '_bincenters'):
            self._bincenters = self.bins[:-1] + 0.5*self.binwidth
        return self._bincenters

    @property
    def pdf(self):
        if not hasattr(self, '_pdf'):
            self._pdf = np.array(self.counts, dtype=np.float64)
            self._pdf /= (self.counts.sum() * self.binwidth)
        return self._pdf


def _ndarray_to_str(string:np.ndarray|bytes):
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
    if keys[0] in dct:
        return True if len(keys) == 1 else _nested_in(dct[keys[0]], keys[1:])
    return False

def _nested_get(dct:dict, keys:tuple, default=None)->Any:
    for k in keys:
        if k in dct:
            dct = dct[k]
        else:
            return default
    return dct

def _nested_set(dct:dict, keys:tuple[Hashable,...], val:Any)->Any:
    for key in keys[:-1]:
        if key not in dct:
            dct[key] = dict()
        dct = dct[key]
    dct[keys[-1]] = val
    return val

def _inner_nested_items(dct:dict, outer:tuple[Hashable,...])->Iterator[tuple[Hashable,Any], bool, None]:
    diter = iter(dct.items())
    for key, val in diter:
        if isinstance(val, dict):
            yield from _inner_nested_items(val, outer+(key,))
        else:
            adv = yield outer + (key, ), val
            if adv:
                break

def _nested_items(dct:dict)->Iterator[tuple[Hashable,Any], bool, None]:
    yield from _inner_nested_items(dct, tuple())


########################
### Additional functions
########################
def clk_to_s(t_ck, clk_p=12.5*1e-9):
    """Convert clock cycles to seconds."""
    return t_ck*clk_p


def pprint(s, mute=False):
    """Print immediately, even if inside a busy loop."""
    if mute: return
    sys.stdout.write(s)
    sys.stdout.flush()


def deprecate(function, old_name, new_name):
    def deprecated_function(*args, **kwargs):
        pprint("Function name %s is deprecated, use %s instead.\n" %\
                (old_name, new_name))
        res = function(*args, **kwargs)
        return res
    return deprecated_function


def shorten_fname(f):
    """Return a path with only the last subfolder (i.e. measurement date)."""
    return '/'.join(f.split('/')[-2:])


def binning(times, bin_width_ms=1, max_num_bins=1e5, clk_p=12.5e-9):
    """Return the binned histogram of array times."""
    bin_width_clk = (bin_width_ms*1e-3)/clk_p
    num_bins = min(times.max()/bin_width_clk, max_num_bins)
    h = np.histogram(times[times<(num_bins*bin_width_clk)], bins=num_bins)
    return h


def mkdir_p(path):
    """Create the path if not existent, otherwise do nothing.
    If `path` exists, and is not a dir, raise an exception.
    """
    import errno
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise exc


def download_file(url, save_dir='./'):
    """Download a file from `url` saving it to disk.

    The file name is taken from `url` and left unchanged.
    The destination dir can be set using `save_dir`
    (Default: the current dir).
    """
    # Check if local path already exist
    fname = url.split('/')[-1]
    print('URL:  %s' % url)
    print('File: %s\n ' % fname)

    path = '/'.join([os.path.abspath(save_dir), fname])
    if os.path.exists(path):
        print('File already on disk: %s \nDelete it to re-download.' % path)
        return

    from urllib.request import urlopen, urlretrieve
    from urllib.error import HTTPError, URLError

    # Check if the URL is valid
    try:
        urlopen(url)
    except URLError as e:
        print('Wrong URL or no connection.\n\nError:\n%s\n' % e)
    except HTTPError:
        print('URL not found: ' + url)
        return

    # Download the file
    def _report(blocknr, blocksize, size):
        current = blocknr*blocksize/2**20
        sys.stdout.write(
            "\rDownloaded {0:4.1f} / {1:4.1f} MB".format(current, size/2**20))
    mkdir_p(save_dir)
    urlretrieve(url, path, _report)


def union_multi(*args)->np.ndarray:
    """Repeted application of np.uion1d on each argument"""
    if len(args) == 0:
        return np.array([], dtype=np.int64)
    if len(args) == 1:
        return args[0]
    union = np.union1d(args[0], args[1])
    for arr in args[2:]:
        union = np.union1d(union, arr)
    return union


def intersect_multi(*args)->np.ndarray:
    """Repeted application of np.intersect1d on each argument"""
    if len(args) == 0:
        return np.array([], dtype=np.int64)
    intersect = args[0]
    for arg in args[1:]:
        intersect = np.intersect1d(intersect, arg)
    return intersect


def _arrays_identical(*args)->bool:
    """Check if all arrays in arguments are exactly equal to one another,
    returns False automatically if any arrays have a different shape than the
    others 
    """
    ref = args[0]
    for arg in args[1:]:
        if arg.shape != ref.shape or np.any(arg != ref):
            return False
    return True


def _arrays_unique(*args)->bool:
    """Teset if all elements in all arrays in args are unique"""
    all_det = np.concatenate(args)
    return all_det.size == np.unique(all_det).size


def _arrays_geq(arr0, arr1)->bool:
    for i in arr1:
        if i not in arr0:
            return False
    return True


def _large_equal(val0, val1)->bool:
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
        if np.any(val0.shape != val1.shape):
            return False
        else:
            return np.all(val0 == val1)
    elif isinstance(val0, (list, tuple)):
        if len(val0) == len(val1):
            return np.all([_large_equal(v0, v1) for v0, v1 in zip(val0, val1)])
        else:
            return False
    else:
        return val0 == val1


def dict_equal(*dicts)->bool:
    """
    Determine if all key:value pairs in set of dictionaries are the same.

    Parameters
    ----------
    *dicts : dict
        Arbitrarily long set of dictionaries.

    Returns
    -------
    bool

    """
    key0 = dicts[0].keys()
    comp = np.all([key0 == dct.keys() for dct in dicts])
    if comp:
        for key, val0 in dicts[0].items():
            for dct in dicts[1:]:
                if not _large_equal(val0, dct[key]):
                    return False
    return comp


def s_equal(*lsts):
    return np.all([f in l0 for f, l0 in product(chain.from_iterable(l for l in lsts), lsts)])


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


def _insert_list(dct, key, value):
    if key in dct:
        dct[key].append(value)
    else:
        dct[key] = [value, ]


def _numdict_to_tuple(spec:dict, start:int=1)->tuple:
    """Convert dictionary with integer keys into tuple, ordere by the numbers of the keys"""
    order = sorted(spec.keys())
    return tuple(spec.get(i, None) for i in order)


def _ascending_dict_to_tuple(spec:dict, start:int|None=1)->tuple|None:
    if start is None:
        start = min(spec.keys())
    stop = max(spec.keys()) + 1
    return tuple(spec.get(i, None) for i in range(start, stop))

    
def _ascending_dict_to_tuple_strict(spec:dict, start:int=1)->tuple|None:
    """Build a tuple from a dictionary of ascending keys, returns None if any
    key is skipped"""
    if any(i not in spec for i in range(start, len(spec)+start)):
        return None
    return tuple(spec[i] for i in range(start, len(spec)+start))


def enumerate_intersects(*args):
    unions = [union_multi(*arg) for arg in args if len(arg)]
    union = union_multi(*unions)
    if any(np.setdiff1d(union, u).size for u in unions):
        raise ValueError("Inconsistent detector ids in inputs")
    strides = np.cumprod(([len(arg) for arg in args]+[1,])[:0:-1], dtype=np.uint8)
    args = tuple(arg if len(arg) else (union, ) for arg in args)
    for cmbs in product(*(enumerate(arg) for arg in args)):
        ijk, sets = zip(*cmbs)
        yield np.sum(np.array(ijk, dtype=np.uint8)*strides), intersect_multi(*sets)

class _Arr_slc:
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

def _dimscompare(shape:tuple[int], dims:tuple[Union[int,slice]])->bool:
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

IndexType = tuple[Union[int,slice,None],...]


def _dim_mask_iter(index:np.ndarray[np.bool_], shape:tuple[int])->tuple[int,tuple[int,...],Iterator]:
    if index.size!= np.prod(shape):
        raise ValueError("incompatible shape")
    if index.ndim != 1 and index.shape != shape:
        raise ValueError("incompatible shape")
    index.reshape(shape)
    size = index.sum()
    return size, (size,), (idx for idx in product(*(range(s) for s in shape)) if index[idx])


def _n_to_index(n:int, shape:tuple[int,...])->tuple[int,...]:
    strides = np.cumprod((shape+(1,))[::-1])[::-1]
    return tuple((n % strides[:-1]) // strides[1:])


def _dim_iter_tuple(index:IndexType, shape:tuple[int,...])->tuple[int,tuple[int,...],Iterator]:
    N = np.arange(np.prod(shape)).reshape(shape)
    Ni = N[index]
    return Ni.size, Ni.shape, (_n_to_index(i) for i in Ni.reshape(-1))


def _dim_iter(index:IndexType, shape:tuple[int,...])->tuple[int,tuple[int,...],Iterator]:
    if isinstance(index,np.ndarray) and index.dtype == np.bool_:
        return _dim_mask_iter(index, shape)
    if not isinstance(index, tuple):
        index = (index, )
    return _dim_iter_tuple(index, shape)


def _close_file(file:tb.File):
    if file is not None and file.isopen:
        file.close()


try:
    import numba
except:
    has_numba = False
else:
    has_numba = True

def fjit(*args, **kwargs):
    if has_numba:
        return numba.jit(*args, **kwargs)
    else:
        return lambda x: x
    
    
class _fdummy:
    def __getattr__(self, attr):
        pass
    def __getitem__(self, item):
        pass
_dummy = _fdummy()

class _fnumba:
    def __getattr__(self, attr):
        if has_numba:
            return getattr(numba, attr)
        return _dummy
fnumba = _fnumba()


######################
# Deprecated functions
######################
def selection_mask(arr, values):
    """
    DEPRECATED: replace with numpy isin function
    Return a boolean mask, True when `arr` is one element of `values`.

    This function generalizes the comparison `arr == values`
    where `values` can be a scalar or a sequence.
    If a sequence, the returned mask is True each time `arr` has an element
    in `values`.

    Arguments:
        arr (array): input array for which a mask is computed.
        values (scalar or sequence): one or more values to be selected in
            `arr`.
    
    Returns:
        Boolean mask same size as `arr`. True where `arr` has an element 
        in `values`.
    """
    warn(DeprecationWarning("use numpy isin instead"))
    return np.isin(arr, values)
