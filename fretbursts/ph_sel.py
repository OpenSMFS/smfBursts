#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: Paul David Harris
# created: Aug 1 2022
"""
This module is dedicated to handling and specifying *photon streams* in an abstract way.
*Photon streams* are defined by the nature of the detector, and during which excitation 
period the photon arrived, *photon streams*.
While detectors are stored as simple indices, during conversion *to PhotonHDF5*
the setup of the detectors was defined.

This assigns photons some of the following categories:

=============    =============================================
Detector type    Descrition
=============    =============================================
ex               excitation period (usually specrally defined)
em               emmission spectral channel of arriving photon
pol              Polarization of emission
split            detector part of split channel
=============    =============================================

These streams are given integer indices. These are stored as uint8, so up to 256
streams in any channel may be defined


To make code more readabe, *ex/em* and *pol* indices have default aliasses.

=====   =====   ========
Type    Index   Aliases
=====   =====   ========
ex/em   0       D
ex/em   1       A
pol     0       P or par
pol     1       S or per
=====   =====   ========

Photon selectiosn are achieved through 2 classes, a base class the user usually doesn't
interact with, and a wrapper class that is the general way of specifying and selecting
photon streams.

#. :class:`PhStream` the foundational photon selection class, which defines a set of detectors with certain traits in common
#. :class:`PhSel` the higher level class, which allows any arbitrary selection of streams, by concatenating multiple :class:`Ph_stream` objects

With :class:`PhSel` , any stream can be specified, and any combination can be specified.

The synax is `[streamcode1]typecode1[streamcode2]typecode2...` ::
    
    phDA = PhSel('DexAem')


Selections are defined by strings, and if you want the union of several selections
you can specify multiple sub-streams by separating them with a single underscore::
    
    PhSel('DexDem_AexAem')
    
If any of the *ex*, *em*, *pol*, or *split* are not specified, then the selection
will not disitinguish photons based on that stream.

For instance::
    
    PhSel('Dex')
    
will take all photons during Donor excitation, regarless of whether they came in 
the Donor or Acceptor emission channels, or polarization, or split.

Finally, :class:`Ph_stream` and :class:`PhSel` are immutable and hashable, and 
thus can be used as dictionary keys.
"""

import functools
from itertools import chain, product, combinations
from typing import Union, ClassVar
from collections.abc import Iterable, Sequence, Hashable, Iterator
import re

import numpy as np
import tables as tb

from .datamodel.utils import ImDict, union_multi, _tuple_array
from .datamodel.immutabledata import _ImData, TypeValidator, init_write_group, register_byteslike

class ChannelSet:
    """
    Logical representation of set of integers (uses uint8).
    Allows definition of positive (inclusive) or negative (exclusive) sets.
    
    Parameters
    ----------
    kind : bool
        True for positive sets (all numbers in elements are incldued)
        False for negative sets (all numbers in elements are excluded)
    elements : frozenset[np.uint8]
        uint8 indexes to be incldued or excluded.
    """
    __slots__ = ('kind','elements')
    kind:bool
    elements:frozenset[np.uint8]
    def __init__(self, kind:bool, elements:frozenset[np.uint8]):
        super().__setattr__('kind', bool(kind))
        super().__setattr__('elements', frozenset(elements))
        
    def __setattr__(self, name, value):
        raise AttributeError("ChannelSet does not support assignment")
    
    def __bool__(self)->bool:
        return False if self.kind is True and len(self.elements) == 0 else True    
        
    def __invert__(self)->"ChannelSet":
        return type(self)(not self.kind, self.elements)
    
    def __and__(self, other:"ChannelSet")->"ChannelSet":
        """Equivalent to intersect"""
        if self.kind:
            if other.kind:
                return type(self)(True, self.elements & other.elements)
            else:
                return type(self)(True, self.elements - other.elements)
        else:
            if other.kind:
                return type(self)(True, other.elements - self.elements)
            else:
                return type(self)(False, self.elements | other.elements)
    
    def __or__(self, other:"ChannelSet")->"ChannelSet":
        """Equivalent to union"""
        if self.kind:
            if other.kind:
                return type(self)(True, self.elements | other.elements)
            else:
                return type(self)(False, other.elements - self.elements)
        else:
            if other.kind:
                return type(self)(False, self.elements - other.elements)
            else:
                return type(self)(False, self.elements & other.elements)
            
    def __matmul__(self, other):
        return type(self)(self.kind != other.kind, self.elements ^ other.elements)
    
    def __xor__(self, other:"ChannelSet")->"ChannelSet":
        """Elements in one and only one of the two sets"""
        return ~self.__matmul__(other)
    
    def __sub__(self, other:"ChannelSet")->"ChannelSet":
        return (self^other) & self
    
    def __iter__(self)->Iterable[np.uint8]:
        yield from (i for i in self.elements)
    
    def __hash__(self)->int:
        return hash((self.kind, self.elements))
    
    def __eq__(self, other)->bool:
        return self.kind == other.kind and self.elements == other.elements
    
    def __contains__(self, other:"ChannelSet")->bool:
        if isinstance(other, ChannelSet):
            return self & other == other
        else:
            if self.kind:
                return other in self.elements
            else:
                return other not in self.elements
    
    def __le__(self, other:"ChannelSet")->bool:
        return self in other
    
    def __lt__(self, other:"ChannelSet")->bool:
        return self in other and other not in self
    
    def __ge__(self, other:"ChannelSet")->bool:
        return other in self
    
    def __gt__(self, other:"ChannelSet")->bool:
        return other in self and self not in other
    
    @property
    def _sel_repr(self)->str:
        """repr of selection for use with :class:`PhStream`"""
        if not self:
            return 'empty'
        text = str() if self.kind else '~'
        if len(self.elements) == 1:
            text += str(list(self.elements)[0])
        else:
            text += '[' + ','.join(str(s) for s in self.elements) + ']'
        return text
                
    def __str__(self)->str:
        if not self:
            return "emtpy"
        elif not self.kind and len(self.elements) == 0:
            return "all"
        return f'({self.kind}):(' +  ', '.join(str(s) for s in self.elements) + ')'
    
    def __repr__(self)->str:
        return "Channel set: " + str(self)
    
    def render_positive(self, n_streams:int, convert_all:bool=False)->"ChannelSet":
        """
        Convert to positive definition of Channel set, based on number of channels
        (``n_streams``).

        Parameters
        ----------
        n_streams : int
            Number of distinct types in channel.
        convert_all : bool, optional
            If positive definition of self encompases all streams, return empty
            negative (all) ChannelSet. The default is False.

        Raises
        ------
        ValueError
            n_streams too small, self positively defines streams larger than n_streams.

        Returns
        -------
        ChannelSet
            Possitively defined :class:`ChannelSet`.

        """
        if self.kind:
            if any(element >= n_streams for element in self.elements):
                raise ValueError("Set exceeds number of streams in this category")
            if convert_all:
                if self.elements == frozenset(np.uint8(i) for i in range(n_streams)):
                    return type(self)(False, {})
            return self
        else:
            ret = set(np.uint8(i) for i in range(n_streams) if i not in self.elements)
            if convert_all and len(ret) == n_streams:
                return type(self)(False, {})
            return type(self)(True, ret)
    
    @property
    def positive_all(self)->bool:
        """If defined positibely or empty negative (all)"""
        return self.kind if self.elements else not self.kind
    
    def tex_str(self, stream_names:dict[str:dict[int|frozenset[int]:str]]=None)->str:
        """
        Convert to math-tex formated string

        Parameters
        ----------
        stream_names : dict[str:dict[int|frozenset[int]:str]], optional
            Names for each channel, given as nested dict, outer dicts have 
            keys of channel name, inner keys are int, specifing index, values
            are stream names. The default is None.

        Returns
        -------
        str
            math-tex formated string, assumed to be part of larger string so not
            wrapped with $$.

        """
        stream_names = dict() if stream_names is None else stream_names
        if not self:
            return stream_names.get((True, frozenset()), '')
        if not self.kind and (False, self.elements) in stream_names:
            return stream_names[(False, self.elements)]
        if self.elements in stream_names:
            return stream_names[self.elements] if self.kind else rf'\neg {stream_names[self.elements]}'
        ret = ','.join(stream_names.get(s, str(s)) for s in self.elements)
        if len(self.elements) > 1:
            ret = f'[{ret}]'
        return ret if self.kind else rf'\neg {ret}'


_csall = ChannelSet(False, frozenset())
_csempty = ChannelSet(True, frozenset())


def check_channelset(val:Union[ChannelSet,Sequence[Union[bool,int]]], **kwargs)->ChannelSet:
    """
    Check function for use with 
    :class:`~fretbursts.datamodel.immutabledata.TypeValidator` to check that
    a object is a :class:`ChannelSet`

    Parameters
    ----------
    val : ChannelSet
        Value to check.
    **kwargs 
        Ignored, used to be compatible with 
        :class:`~fretbursts.datamodel.immutabledata.TypeValidator`

    Raises
    ------
    ValueError
        Wrong type/cannot convert to :class:`ChannelSet`.

    Returns
    -------
    ChannelSet
        Verified :class:`ChannelSet` value.

    """
    if  isinstance(val, ChannelSet):
        return val
    has_t = any(True is elem for elem in val)
    has_f = any(False is elem for elem in val)
    if has_t and has_f:
        raise ValueError("Cannot have positive and negative definition in one category")
    det_set = set(elem for elem in val if elem is not True and elem is not False)
    for element in det_set:
        if not np.issubdtype(type(element), np.integer) and element >= 0 and element <256:
            raise ValueError("Only positive integer values allowed in streams, cannot convert to ChannelSet")
    return ChannelSet(not has_f, (np.uint8(det) for det in det_set))


def write_channelset(group:tb.Group, name:str, val:ChannelSet)->tb.Group:
    """
    Write fucntion 

    Parameters
    ----------
    group : tb.Group
        HDF5 Group in which to create representation of val.
    name : str
        Name of group to create.
    val : ChannelSet
        :class:`ChannelSet` to write to HDF5 group group.name.

    Returns
    -------
    group : tb.Group
        HDF5 group written to file.

    """
    file, group = init_write_group(group, name, 'ChannelSet')
    rep = np.array([int(val.kind), ] + list(val.elements), dtype=np.uint8)
    file.create_array(group, 'chset', rep)
    return group


def read_channelset(group:tb.Group, dct:dict)->ChannelSet:
    r"""
    Read function for use with 
    :class:`fretbursts.datamodel.immutabledata.TypeValidator` to write 
    :class:`ChannelSet` to 

    Parameters
    ----------
    group : tb.Group
        Group to read as channel set.
    dct : dict
        Unused, necessary to be compatible with :class:`TypeValidator`, dictionary
        of \_grouptypes.

    Returns
    -------
    ChannelSet
        ChannelSet represented  by HDF5 group.

    """
    rep = group.chset.read()
    return ChannelSet(rep[0], rep[1:])


TypeValidator.register_grouptype("ChannelSet", read_channelset)
TV_channelset = TypeValidator(ChannelSet, check_channelset, write_channelset)

_StreamTypes = ('ex', 'em', 'pol', 'split')


class PhStream(_ImData):
    """
    Representation of square set of Channel categories.
    
    Takes 4 :class:`ChannelSet` objects, 1 for each categorry of channel: 'ex',
    'em', 'split', and 'pol'
    
    Parameters
    ----------
    ex : ChannelSet, optional
        excitation channels, default is ChannelSet(False, {})
    em : ChannelSet, optional
        emission channels, default is ChannelSet(False, {})
    pol : ChannelSet, optional
        polarization channels, default is ChannelSet(False, {})
    split : ChannelSet, optional
        split channels, default is ChannelSet(False, {})
        
    """
    __slots__ = _StreamTypes
    _typeconversions = ImDict({key:TV_channelset for key in _StreamTypes})
    _defaults = ImDict({key:ChannelSet(False, frozenset()) for key in _StreamTypes})
    
    # catch case of single empty stream, convert to all empty
    def __post_init__(self):
        if not self:
            for attr in self._all_keys():
                super(_ImData, self).__setattr__(attr, _csempty)
    
    def __eq__(self, other):
        if isinstance(other, PhStream):
            return super().__eq__(other)
        if isinstance(other, PhSel):
            return other.__eq__(self)
        return False
    
    __hash__ = _ImData.__hash__ # python automatically resets hash if a new eq method is defined
    
    def __bool__(self):
        return all(bool(det) for det in self._all_values())
    
    def _all_keys(self):
        """keys() like iterator, ensures all channel categories included"""
        yield from (cat for cat in self.__slots__)
    
    def _all_values(self)->Iterator[ChannelSet]:
        """values() like iterator, ensures all channel sets included"""
        yield from (self[cat] for cat in self.__slots__)
    
    def _all_items(self)->Iterator[tuple[str, ChannelSet]]:
        """items() like iterator, enuring iteration over all channel categories"""
        yield from ((cat, self[cat]) for cat in self.__slots__)
    
    def keys(self, skip:Union[Sequence[Hashable],set[Hashable],frozenset[Hashable]]=None)->Iterator[str]:
        """
        Iterator over each channel category name.

        Parameters
        ----------
        skip : Union[Sequence[Hashable],set[Hashable],frozenset[Hashable]], optional
            DESCRIPTION. The default is None.

        Yields
        ------
        str
            String name of channel category.

        """
        skip = skip if skip is not None else tuple()
        yield from (cat for cat in self.__slots__ if self[cat] != ChannelSet(False, frozenset()) and cat not in skip)
    
    
    def __contains__(self, other):
        if isinstance(other, PhSel):
            return other in PhSel(self)
        elif isinstance(other, PhStream):
            intersect = _stream_intersect(self, other)
            return intersect == other
        raise TypeError(f"cannot assess {type(other)} contained in PhSel object")
    
    def __invert__(self):
        if not self:
            return type(self)()
        inverted = tuple(PhStream(**{ch:~cset if chan == ch else _csall
                                     for ch in self._all_keys()}) 
                         for chan, cset in self._all_items())
        if len(inverted) == 1:
            return inverted[0]
        out = inverted[0]
        for inv in inverted[1:]:
            out |= inv
        return out
    
    def __le__(self, other):
        return self in other
    
    def __ge__(self, other):
        return other in self
    
    def __lt__(self, other):
        return self in other and other not in self
    
    def __gt__(self, other):
        return other in self and self not in other
    
    def __and__(self, other):
        if isinstance(other, PhStream):
            return _stream_intersect(self, other)
        elif isinstance(other, PhSel):
            return other & self
        raise TypeError(f"unsupported operand types for &: PhStream and {other.__name__}")
    
    def __or__(self, other):
        if isinstance(other, PhSel):
            return other | self
        elif not isinstance(other, PhStream):
            raise TypeError(f"unsupported operand types for |: PhStream and {other.__name__}")
        # get set decomposition of self and other
        streams = set(chain(*_stream_shadows(self, other)))
        # eliminate repeat streams
        comb = tuple(_stream_combine(*streams))
        return comb[0] if len(comb) == 1 else PhSel(*comb)
    
    def __xor__(self, other):
        if isinstance(other, PhSel):
            return other ^ self
        elif not isinstance(other, PhStream):
            raise TypeError(f"unsupported operand types for ^: PhStream and {other.__name__}")
        # get set decomposition of self and other
        streamS, streamO = _stream_shadows(self, other)
        # eliminate repeat streams
        streams = set(streamS) ^ set(streamO)
        streams = _stream_combine(*streams)
        comb = _stream_combine(*streamS)
        return comb[0] if len(comb) == 1 else PhSel(*comb)
    
    def __matmul__(self, other):
        if not isinstance(other, (PhStream, PhSel)):
            raise TypeError(f"unsupported operand types for @: PhStream and {other.__name__}")
        return ~(self ^ other)
    
    def __add__(self, other):
        if not isinstance(other, (PhStream, PhSel)):
            raise TypeError(f"unsupported operand types for +: PhStream and {other.__name__}")
        return self | other
        
    def __sub__(self, other):
        if isinstance(other, PhSel):
            out = PhSel(self) - other
            if len(out.streams) == 1:
                out = list(out.streams)[0]
            return out
        elif isinstance(other, PhStream):
            streamS, streamO = _stream_shadows(self, other)
            streamS = set(streamS) - set(streamO)
            comb = _stream_combine(*streamS)
            return comb[0] if len(comb) == 1 else PhSel(*comb)
        raise TypeError(f"unsupported operand types for -: PhStream and {other.__name__}")
    
    def __str__(self):
        return ''.join(cset._sel_repr + name for name, cset in self._all_items() if ~cset)
    
    def __repr__(self):
        text = str(self.__class__) + '\n'
        text += '\n'.join(f'{name} = {str(cset)}' for name, cset in self._all_items())
        return text
    
    def render_positive(self, detdef:"DetDef", convert_all:bool=False)->"PhStream":
        """
        Positive representation of self based on :class:`DetDef` definition of
        number of streams in each channel.

        Parameters
        ----------
        detdef : DetDef
            :class:`DetDef` defining number of channels in each stream.
        convert_all : bool, optional
            If :code:`True`, remove stream definition for streams that span detdef. 
            The default is False.

        Returns
        -------
        PhStream
            Positive version of the PhStream.

        """
        return type(self)(**{cat:self[cat].render_positive(n, convert_all=convert_all) 
                             for cat, n in detdef.items()})
    
    @property
    def positive(self)->bool:
        """If all streams defined in a positive manner"""
        return all(s.kind for s in self._all_values())
    
    @property
    def positive_all(self)->bool:
        """All streams defined as all or in possitive manner"""
        return all(s.positive_all for s in self._all_values())
    
    def tex_str(self, detdef:"DetDef"=None, name:str='f', 
                stream_names:dict[str:dict[int|frozenset[int]:str]]=None)->str:
        """
        Generate tex string representing PhSel using information provided.

        Parameters
        ----------
        detdef : DetDef, optional
            DetDef of originating setup. The default is None.
        name : str, optional
            Name given to sort of parameter being extracted (ie n, f, I etc). 
            The default is 'f'.
        stream_names : dict[str:dict[int|frozenset[int]:str]], optional
            Names for each channel, given as nested dict, outer dicts have 
            keys of channel name, inner keys are int, specifing index, values
            are stream names. The default is None.
        
        Returns
        -------
        str
            Math tex formated expression for string 
            (assumed to be portion of larger string, so not wrapped in $).

        """
        stream_names = dict() if stream_names is None else stream_names
        st = self if detdef is None else self.render_positive(detdef, convert_all=True)
        sup, sub = '', ''
        for chan, cset in st.items():
            cstring = f'{cset.tex_str(stream_names.get(chan))}{chan}'
            if chan == 'ex':
                sub = cstring
            else:
                sup += cstring
        out = name
        out += '_{%s}'%sub if sub else ''
        out += '^{%s}'%sup if sup else ''
        return out


_psall = PhStream()
_psempty = PhStream(ex=_csempty, em=_csempty, pol=_csempty, split=_csempty)


def _stream_zip(*streams:PhStream)->tuple[str,ChannelSet,...]:
    """
    Iterator for PhStream, returning (category, definition0, definition1...) for
    each iteration.
    """
    for cat in streams[0]._all_keys():
        yield (cat, ) + tuple(stream[cat] for stream in streams)


def _field_comp(stream0:PhStream, stream1:PhStream)->PhStream:
    """Return dictionary saying if a given field is the same in two streams"""
    return {cat:det0==det1 for cat, det0, det1 in _stream_zip(stream0, stream1)}


def _stream_join(stream0:PhStream, stream1:PhStream)->PhStream:
    """NOTE: use only when check par parrallel streams has been perfomred
    Merges two photon streams together into single compbined photon stream"""
    return PhStream(**{cat:(det0 | det1) for cat, det0, det1 in _stream_zip(stream0, stream1)})


def _stream_num_diff_cat(stream0:PhStream, stream1:PhStream)->int:
    """Number of channels different between stream0 and stream1"""
    return sum(not det for det in _field_comp(stream0, stream1).values())


def _stream_par(stream0:PhStream, stream1:PhStream)->bool:
    """Returns True if stream0 and stream1 differ only in a single category"""
    return _stream_num_diff_cat(stream0, stream1) <= 1


def _stream_intersect(stream0:PhStream, stream1:PhStream)->PhStream:
    """Returns the overlap of stream0 and stream1"""
    intersect = {cat:(det0 & det1) for cat, det0, det1 in _stream_zip(stream0, stream1)}
    return PhStream(**intersect)


def _stream_combs(sdict:dict[str,ChannelSet])->tuple[PhStream,...]:
    """Given a dictionary of categories and sub streams, return tuple of all possible combinations as PhStream objects"""
    # to maintain order (could alternatively use ordereddict)
    slist = [(cat, dets) for cat, dets in sdict.items()]
    scats, sdets = [cat for cat, det in slist], [det for cat, det in slist]
    # generate product of all combinations of streams
    combs = (PhStream(**{cat:det for cat, det in zip(scats, dets)}) for dets in product(*sdets))
    return tuple(comb for comb in combs if comb) # filter out None streams


def _stream_shadows(*streams:PhStream)->tuple[tuple[PhStream,...],...]:
    """Get minimum divisions of each stream compared to other streams"""
    shadows = [{cat:list() for cat in stream._all_keys()} for stream in streams]
    for cat, *dets in _stream_zip(*streams):
        # create shadow of each combination
        shadow = {dets[0],}
        for det in dets[1:]:
            shadow = set(s for s in chain.from_iterable((det&s, det-s, s-det) for s in shadow) if s)
        for i, sdict in enumerate(shadows):
            sdict[cat] = [s for s in shadow if s in dets[i] and s]
    return tuple(_stream_combs(sdict) for sdict in shadows)


def _stream_combine(*streams:PhStream)->frozenset[PhStream]:
    """Unite all parallel streams so that no parrallel sets remain"""
    stream_combs = [set(streams), ]
    new_streams = True
    while new_streams:
        new_streams = list()
        for a, b in combinations(stream_combs[-1], 2):
            if _stream_par(a,b): new_streams.append(_stream_join(a,b))
        for streamsA, streamsB in combinations(stream_combs, 2):
            for a, b in product(streamsA, streamsB):
                if _stream_par(a,b): new_streams.append(_stream_join(a,b))
        new_streams = set(stream for stream in new_streams 
                       if not any(stream == comp for comp in chain(*stream_combs)))
        if new_streams: stream_combs.append(new_streams)
    streams_new = list(chain(*stream_combs))
    streams_new = frozenset(stream for stream in streams_new if not any(stream < streamB for streamB in streams_new))
    return streams_new


class PhSel:
    """
    Representation of any combination of photon streams. Streams may be defined
    positively, where specific streams are included, or negatively, where specific
    streams are excluded.
    
    Logical operations can be performed, where it behaves according to the same
    rules as sets, addition behaves like or. PhSel objects are immutable and
    hashable, and so can be compared with == as well as used as dictionary keys.
    
    The standard way of instantiating is through a string. For instance
    
    >>> deaem = frb.PhSel('0ex1em')
    
    creates an object indicating the 0th (in FRET traditionally the donor), and
    1st (in FRET traditionally the accetor) excitation and emission streams
    (ie donor-excitation, acceptor emission).
    
    Channels come in 
    4 categories: ``ex``, ``em``, ``pol``, and ``split``, cooresponding to 
    excitation, emission, polarization and split channels. If a category is ommitted,
    it is assumed that all channels in that category are used.
    
    Compounding is allowed
        
    >>> aexdaem = frb.PhSel('1ex[0,1]em1pol')
    
    represents 1st excitation 1st polarization (generally perpendicular), and 
    both 0th and 1st emission channels (in 2-color experiment, this would be all,
    but if there are 3 or more colors, this will exclude 2nd ... channels)
    
    Futher, multiple selections can be combined by separating stream definitions
    with underscores ``_``
    
    >>> dex_aexaem = frb.PhSel('0ex_1ex1em')
    
    The above represents all streams from the 0th excitation, along with the
    1st excitation 1st emission channel. In a 2 color FRET experiment with ALEX,
    this would represent all "relevant" photon streams, as it excludes the
    acceptor excitation donor emission stream.
    
    """
    __slots__ = ('streams', )
    # part map for mapping old PhSel names to index(es)
    _part_map = ImDict({'em':{'D':{np.uint8(0)}, 'A':{np.uint8(1)}, 'DA':{np.uint8(0), np.uint8(1)}},
                        'ex':{'D':{np.uint8(0)}, 'A':{np.uint8(1)}, 'DA':{np.uint8(0), np.uint8(1)}},
                        'pol':{'P':{np.uint8(0)}, 'S':{np.uint8(1)}, 'par':{np.uint8(0)}, 'per':{np.uint8(1)}},
                        'split':None})
    # map for recording phsel string as node-name
    _attr_repl_dict = ImDict({'[':'b', ']':'B', '~':'n', ',':'c'})
    
    streams: frozenset[PhStream]
    
    def __init__(self, *args, **kwargs):
        if args and kwargs:
            raise ValueError("Cannot mix args and kwargs in creating new PhSel object")
        # will treat *args as set of PhStreams, later update to have full string etc. parser
        pargs = list()
        if args:
            for arg in args:
                if isinstance(arg, str):
                    pargs += _parse_streams(arg, self._part_map)
                elif isinstance(arg, PhSel):
                    pargs += arg.streams
                elif isinstance(arg, PhStream):
                    pargs.append(arg)
                else:
                    raise TypeError(f"Arguments must be str, PhStream or PhSel arguments, got {type(arg)}")
        if kwargs:
            if 'Dex' in kwargs or 'Aex' in kwargs:
                pargs += _old_phsel(kwargs)
                if kwargs:
                    raise ValueError("Cannot mix Dex=/Aex= style and ex/em/pol/split style in creating new PhSel object")
            else:
                pargs += [PhStream(**kwargs), ]
        sub_streams = _stream_shadows(*pargs) if pargs else frozenset()
        super().__setattr__('streams', _stream_combine(*chain(*sub_streams)))
        
    def __setattr__(self, attr):
        raise AttributeError("PhSel does not suport assignment")
    
    def __hash__(self)->int:
        return hash(self.streams)
    
    def __contains__(self, other)->bool:
        if isinstance(other, PhSel):
            return all(stream in self for stream in other.streams)
        elif isinstance(other, PhStream):
            return any(other in stream for stream in self.streams)
        else:
            raise TypeError("Can only asses PhSel and PhStream can be contained by PhSel,"
                            " got type {type(other)}")
    
    def __iter__(self)->PhStream:
        for stream in self.streams:
            yield stream
    
    def __bool__(self):
        return bool(self.streams)
    
    def __invert__(self):
        invs = set(chain.from_iterable(_chain_streams(~stream) for stream in self.streams))
        invs = chain.from_iterable(_stream_shadows(*tuple(invs)+tuple(self.streams))[:len(invs)])
        invs = set(inv for inv in invs if inv not in self)
        return type(self)(*invs)
    
    def __eq__(self, other)->bool:
        if isinstance(other, PhStream):
            return len(self.steams) == 1 and list(self.streams)[0] == other
        elif not isinstance(other, PhSel):
            return False
        return self.streams == other.streams
    
    def __leq__(self, other):
        return other in self
    
    def __lt__(self, other):
        return (other in self) and (self not in other)
    
    def __geq__(self, other):
        return self in other
    
    def __gt__(self, other):
        return (self in other) and (other not in self)
    
    def __and__(self, other):
        if not isinstance(other, (PhStream, PhSel)):
            raise TypeError(f"unsupported operand types for &: PhStream and {other.__name__}")
        if not self:
            return self
        if isinstance(other, PhStream):
            streams = (other & stream for stream in self.streams)
        elif isinstance(other, PhSel):
            streams = chain.from_iterable(_chain_streams(self & stream) for stream in other.streams)
        return PhSel(*streams)
    
    def __or__(self, other)->"PhSel":
        if not isinstance(other, (PhStream, PhSel)):
            raise TypeError(f"unsupported operand types for |: PhStream and {other.__name__}")
        if isinstance(other, PhStream):
            streams = chain.from_iterable(_chain_streams(other | stream) for stream in self.streams) if self else (other, )
        elif isinstance(other, PhSel):
            streams = chain.from_iterable(_chain_streams(self | stream) for stream in other.streams)
        return PhSel(*streams)
    
    def __xor__(self, other)->"PhSel":
        if not isinstance(other, (PhStream, PhSel)):
            raise TypeError(f"unsupported operand types for ^: PhStream and {other.__name__}")
        shaddows = chain.from_iterable(_stream_shadows(*tuple(self.streams)+tuple(_chain_streams(other))))
        xor = (stream for stream in shaddows if not ((stream in self) and (stream in other)))
        return PhSel(*xor)
    
    def __matmul__(self, other)->"PhSel":
        if not isinstance(other, (PhStream, PhSel)):
            raise TypeError(f"unsupported operand types for @: PhStream and {other.__name__}")
        return ~(self ^ other)
    
    def __add__(self, other)->"PhSel":
        if not isinstance(other, (PhStream, PhSel)):
            raise TypeError(f"unsupported operand types for +: PhStream and {other.__name__}")
        return self | other
    
    def __sub__(self, other)->"PhSel":
        if not isinstance(other, (PhStream, PhSel)):
            raise TypeError(f"unsupported operand types for -: PhStream and {other.__name__}")
        if not self:
            return self
        shaddows = chain(*_stream_shadows(*tuple(_chain_streams(self))+tuple(_chain_streams(other))))
        xor = (stream for stream in shaddows if (stream in self) and (stream not in other))
        return PhSel(*xor)
    
    def __str__(self):
        if not self.streams:
            return 'none'
        elif self == phsel_all:
            return 'all'
        return '_'.join(str(stream) for stream in self.streams)
        
    
    def __repr__(self):
        text = str(self.__class__) + '\n'
        text += '\n'.join(stream.__repr__() for stream in self.streams)
        return text
    
    def render_positive(self, detdef:"DetDef", convert_all:bool=False)->"PhSel":
        """
        Returns a :class:`PhSel` object with all positive stream definitions, meaning
        that there are no "not X" streams.

        Parameters
        ----------
        detdef : DetDef
            :class:`DetDef` object indicating number of channels in each category.
        convert_all : bool, optional
            If True, and the positive object encompasses all streams in ``detdef`
            then return a ``PhSel('all')`` object. The default is False.

        Returns
        -------
        PhSel
            Positive :class:`PhSel`.

        """
        return type(self)(*(phs.render_positive(detdef, convert_all=convert_all) 
                            for phs in self.streams))
    
    def write_group(self, group:tb.Group, name:Union[str,None]=None)->tb.Array:
        """
        Record a PhSel object in an HDF5 file.

        Parameters
        ----------
        group : tb.Group
            HDF5 group in which to create new array with name ``name`` representing
            the PhSel.
        name : str | None, optional
            Name to give array created to store :class:`PhSel`, if None, name defaults
            to 'phsel'. The default is None.

        Returns
        -------
        tb.Array
            Array object where class:`PhSel` was recorded, stores string represenation
            of :class:`PhSel` as bytes array.

        """
        if name is None:
            name = 'phsel'
        return TypeValidator.write_any(group, name, self)

    @classmethod
    def load_group(cls, group:tb.Array)->"PhSel":
        """
        Create (load) :class:`PhSel` object from HDF5 array.

        Parameters
        ----------
        group : tb.Array
            Array to load as PhSel (PhSel are stored as byte array of string
            representation of PhSel).

        Returns
        -------
        PhSel
            Python object of stored array.

        """
        return cls(group.phsel.decode())
    
    @property
    def attr_str(self)->str:
        r"""Alpha-numeric string representation suitable for name of HDF5 nodes 
        (replace reserved characters such as \[ with alphabetic replacement)"""
        string = str(self)
        for key, val in self._attr_repl_dict.items():
            string = string.replace(key, val)
        return string
    
    @classmethod
    def from_attr_str(cls, attr:str)->"PhSel":
        """
        Convert alpha-numeric string representation of :class:`PhSel` from 
        :attr:`PhSel.attr_str` back to :class:`PhSel` object.

        Parameters
        ----------
        attr : str
            String definition of PhSel with only alpha-numeric .

        Returns
        -------
        PhSel
            PhSel representation of attr.

        """
        for key, val in cls._attr_repl_dict.items():
            attr = attr.replace(val, key)
        return cls(attr)
    
    @property
    def positive(self)->bool:
        """If curret object only defines channels with possitive definitions"""
        return all(s.positive for s in self.streams)
    
    @property
    def positive_all(self)->bool:
        """True if is positive, or all channels empty negative (all)"""
        return all(s.positive_all for s in self.streams)
    
    def _get_union_set(self, stream:str)->ChannelSet:
        """Get :class:`ChannelSet of maximal cross section along given stream"""
        st_iter = iter(self)
        cset = next(st_iter)[stream]
        for st in st_iter:
            cset |= st[stream]
        return cset
    
    @property
    def ex(self)->ChannelSet:
        """:class:`ChannelSet` of maximal cross section of excitation"""
        return self._get_union_set('ex')
    
    @property
    def em(self)->ChannelSet:
        """:class:`ChannelSet` set of maximal cross section of emission"""
        return self._get_union_set('em')
    
    @property
    def pol(self)->ChannelSet:
        """:class:`ChannelSet` set of maximal cross section of polarization"""
        return self._get_union_set('pol')
    
    @property
    def split(self)->ChannelSet:
        """:class:`ChannelSet` set of maximal cross section of split"""
        return self._get_union_set('split')
    
    def tex_str(self, detdef:"DetDef"=None, name:str='f', 
                stream_names:dict[str:dict[int|frozenset[int]:str]]=None, reduce:bool=True)->str:
        """
        Generate tex string representing PhSel using information provided.

        Parameters
        ----------
        detdef : DetDef, optional
            DetDef of originating setup. The default is None.
        name : str, optional
            Name given to sort of parameter being extracted (ie n, f, I etc). 
            The default is 'f'.
        stream_names : dict[str:dict[int|frozenset[int]:str]], optional
            Names for each channel, given as nested dict, outer dicts have 
            keys of channel name, inner keys are int, specifing index, values
            are stream names. The default is None.
        reduce : bool, optional
            Whether to remove channels that cover all space in detdef. The default is True.

        Returns
        -------
        str
            Math tex formated expression for string 
            (assumed to be portion of larger string, so not wrapped in $).

        """
        streams = list(self.streams)
        if reduce:
            cur = PhSel('none')
            for i in range(len(streams)):
                streams[i] = streams[i] - cur
                cur += streams[i]
        return r'\:+\:'.join(stream.tex_str(detdef, name, stream_names) 
                             for stream in streams)
    

def str_long_less(a:str, b:str)->int:
    """
    Used to produce sorting of strings. Returns the difference between
    the first letter to be different between the strings, if one string is longer
    return difference between their lengths.

    Parameters
    ----------
    a : str
        first string.
    b : str
        second string.

    Returns
    -------
    int
        Comparison size.

    """
    for aa, bb in zip(a,b):
        an, bn = ord(aa), ord(bb)
        if an == bn:
            continue
        return an - bn  
    return len(b) - len(a)


def _chain_streams(val:Union[PhStream,PhSel])->Iterator[PhStream]:
    """
    Simple wrapper, always returns iterator of PhStream objects
    if given PhStream, length 1, returning just PhStream on first yield, otherwise
    iterate over each PhStream in val.streams
    """
    if isinstance(val, PhSel):
        yield from val.streams
    else:
        yield from (val, )


sep = re.compile(r'^((\s*[,;]?\s*)|_|())$')


def _stream_iter(streams:str, id_rgx:str, cat_rgx:str)->re.Match:
    """Iterate over stream definition, 1 channel at a time, returns regex of stream match"""
    stream_frag_regex = re.compile(fr'((\~|\!|\^)?(({id_rgx})|(\d+)|(\[([^\[\]]+)\]))({cat_rgx}))+')
    pos = 0 # where in the stream the last match came from
    for mtch in stream_frag_regex.finditer(streams):
        beg, pos_t = mtch.span()
        if not sep.match(streams[pos:beg]):
            raise ValueError(f"Invalid termination sequence: '{streams[pos:beg]}'")
        pos = pos_t
        yield mtch
    if not sep.match(streams[pos:]):
        raise ValueError(f"Invalid stream separator: '{streams[pos:]}'")


def check_keys(part_map:dict[str:set[np.uint8]])->list[str]:
    """Check part_map of phsel is valid and retunr list of keys"""
    stream_str = list(part_map.keys())
    if any(not isinstance(strm, str) for strm in stream_str):
        raise KeyError("part_map keys must be strings")
    return stream_str


def gen_re_fs(cat_strs:Sequence[str])->str:
    """Generate string for regex of possible stream types"""
    cat_strs = list(set(cat_strs))
    cat_strs.sort(key=functools.cmp_to_key(str_long_less))
    return'|'.join(cat_strs)    


_asep = re.compile(r'^,?\s*$')
is_asep = lambda s: bool(_asep.match(s))

#: Photon selection representing all possible streams
phsel_all = PhSel(_psall)
#: Photon selection representing no streams (useful for testing when logical operation returns none)
phsel_none = PhSel() 


def _parse_streams(streams:str, part_map:dict[str:set[np.uint8]])->PhSel:
    """Parse str as a stream input"""
    if streams.lower() == 'all':
        return phsel_all
    if streams.lower() in ('empty', 'none'):
        return phsel_none
    cat_rgx = gen_re_fs(check_keys(part_map))
    if any(c not in PhStream.__slots__ for c in cat_rgx.split('|')):
        raise ValueError("part_map incorrect")
    id_rgx = gen_re_fs(chain.from_iterable(check_keys(val) for val in part_map.values() if val is not None))
    stream_cat_regex = re.compile(fr'(\~|\!|\^)?(({id_rgx})|(\d+)|(\[([^\[\]]+)\]))({cat_rgx})')
    arr_regex = re.compile(fr'\s*(({id_rgx})|(\d+))\s*')
    ph_streams = list()
    for match_stream in _stream_iter(streams, id_rgx, cat_rgx):
        ph_stream = dict()
        for cat_prt in stream_cat_regex.finditer(match_stream.group()):
            kind = not cat_prt.group(1)
            categ = cat_prt.group(7)
            if cat_prt.group(3):
                if (cat_prt.group(3) in part_map[categ]):
                    dets = ChannelSet(kind, part_map[categ][cat_prt.group(3)])
                else:
                    if cat_prt.group(3).isnumeric():
                        dets = ChannelSet(kind, {np.uint8(cat_prt.group(3))})
                    else:
                        raise KeyError(f"{cat_prt.group(3)} not defined for {categ} type")
            elif cat_prt.group(4):
                dets = ChannelSet(kind, {np.uint8(cat_prt.group(4))})
            elif cat_prt.group(6):
                dets = set()
                det_str = cat_prt.group(6)
                pos = 0
                for det in arr_regex.finditer(det_str):
                    span = det.span()
                    if not is_asep(det_str[pos:span[0]]):
                        raise KeyError(f"Invalid separator: {det_str[pos:span[0]]} between stream indices")
                    if det.group(2):
                        if det.group(2) in part_map[categ]:
                            dets |= part_map[categ][det.group(2)]
                        else:
                            if det.group(2).isnumeric():
                                dets |= {np.uint8(det.group(2))}
                            else:
                                raise KeyError(f"{det.group(2)} not defined for {categ} type")
                    elif det.group(3):
                        dets |= {np.uint8(det.group(3))}
                    pos = span[1]
                if not is_asep(det_str[pos:]):
                    raise KeyError(f"Invalid termination of array: {det_str[pos:]}")
            ph_stream[categ] = ChannelSet(kind, dets)
        ph_streams.append(PhStream(**ph_stream))
    return PhSel(*ph_streams)


def _old_phsel(kwargs):
    """Function for translating old Ph_sel syntax to new PhSel"""
    pargs = list()
    Dex = kwargs.pop('Dex', None)
    if Dex:
        pargs += PhSel(f'Dex{Dex}').streams
    Aex = kwargs.pop('Aex', None)
    if Aex:
        pargs += PhSel(f'Aex{Aex}').streams
    return pargs


def phsel_union(*args:PhSel)->PhSel:
    """
    Get the :class:`PhSel` object that is the union of all input :class:`PhSel` s.
    Useful for taking a set of :class:`PhSel` and retrieving the combination of
    all of them.

    Parameters
    ----------
    *args : PhSel
        :class:`PhSel` to combine.

    Returns
    -------
    PhSel
        Union of input :class:`PhSel` s.

    """
    out = args[0]
    for arg in args[1:]:
        out |= arg
    return out


class DetDef:
    """
    Definition of Number of streams in each category, used to specify how to
    interpret detectors array, and in converting between :class:`PhSel` and dets ids.
    
    Parameters
    ----------
    ex : int, optional
        Number of excitation channels. Default is 1.
    em : int, optional
        Number of emission channels. Default is 1.
    pol : int, optional
        Number of polarization channels. Default is 1.
    split : int, optional
        Number of split channels. Default is 1.
    """
    __slots__ = ('shape', 'strides')
    _params:ClassVar[tuple[str]] = _StreamTypes
    
    shape:np.ndarray[np.uint8]
    strides:np.ndarray[np.uint8]
    
    def __init__(self, *args, **kwargs):
        if len(args) > len(self._params):
            raise TypeError("too many arguments for DetDef")
        repeats = [k for k in kwargs.keys() if k in self._params[:len(args)]]
        if repeats:
            raise TypeError("got multiple instances for the following arguments: {repeats}")
        shape = np.ones(len(self._params), dtype=np.uint8)
        for i, value in enumerate(args):
            shape[i] = value
        for key, value in kwargs.items():
            if key not in self._params:
                raise TypeError(f"DetDef got unexpected keyword argument '{key}'")
            for i, name in enumerate(self._params):
                if name == key:
                    shape[i] = value
                    break
        strides = np.empty(shape.shape, dtype=np.uint8)
        strides[:-1] = shape[1:]
        strides[-1] = 1
        strides = np.cumprod(strides[::-1])[::-1]
        shape.setflags(write=False)
        strides.setflags(write=False)
        super().__setattr__('shape', shape)
        super().__setattr__('strides', strides)

    def __setattr__(self, attr, value):
        raise AttributeError("DetDef does not support assignment")

    def __getattr__(self, attr):
        if attr.endswith('_stride'):
            array, attr_ = self.strides, attr.split('_stride')[0]
        else:
            array, attr_ = self.shape, attr
        for i, name in enumerate(self._params):
            if name == attr_:
                return array[i]
        raise AttributeError(f"DetDef has no attribute {attr}")
    
    def __getitem__(self, key):
        return self.__getattr__(key)
    
    def __hash__(self):
        return hash(_tuple_array(self.shape))
    
    def __eq__(self, o):
        return np.all(self.shape == o.shape)

    def items(self)->Iterator[tuple[str, int]]:
        """Iterator over each channel, yielding (channel name, size) as tuples of (str, int)."""
        yield from zip(self._params, self.shape)
    
    def write_group(self, group:tb.Group, name:str|None=None)->tb.Array:
        """
        Write detdef to group of name ``name`` inside hdf5 group ``group``

        Parameters
        ----------
        group : tb.Group
            HDF5 group in which to write the detdef.
        name : str|None, optional
            Name of array for detdef to be written. If ``None``, defaults to `DetDef`
            The default is None.

        Returns
        -------
        tb.Array
            Array where detdef is written (has name ``name``)

        """
        if name is not None:
            name = 'DetDef'
        return TypeValidator.write_any(group, name, self)

    @classmethod
    def load_group(cls, group:tb.Array)->"DetDef":
        """
        Load a HDF5 array as :class:`DetDef`

        Parameters
        ----------
        group : tb.Array
            size 4 uint8 array to interpret as :class:`DetDef`.

        Returns
        -------
        DetDef
            DetDef object corresponding to saved definition.

        """
        shape = group.read()
        return cls(*shape)
    
    def get_det_id(self, idxs:np.ndarray[np.integer])->int:
        """
        From size 4 array of channel index, get the detector id.

        Parameters
        ----------
        idxs : np.ndarray[np.integer]
            Size 4 array of channel indexes.

        Returns
        -------
        int
            detector id.

        """
        return np.sum(idxs*self.strides)
    
    @property
    def size(self)->int:
        """Total number of single streams in DetDef"""
        return np.prod(self.shape)
    
    def _get_stream_id(self, phstream:PhStream)->np.ndarray[np.uint8]:
        """Get stream_ids (indices in dets array) from phstream"""
        stream = phstream.render_positive(self)
        pstream = product(*(tuple((name, i) for i in ids) for name, ids in stream._all_items()))
        idxs = [sum(self[f'{p}_stride']*i for p, i in ds) for ds in pstream]
        return np.unique(idxs).astype(np.uint8)
        
    def get_stream_ids(self, phsel:PhSel)->np.ndarray[np.uint8]:
        """
        Retrieve the stream ids (det ids) of a :class:`PhSel` object based
        on self.

        Parameters
        ----------
        phsel : PhSel
            Object to convert to stream ids (det ids).

        Returns
        -------
        np.ndarray[np.uint8]
            Array of all stream ids (det ids) in :class:`PhSel`.

        """
        if isinstance(phsel, PhStream):
            phsel = (phsel, )
        return union_multi(*(self._get_stream_id(phs) for phs in phsel))
    
    def _stream_id_to_PhStream(self, stream_id:int)->PhStream:
        """Convert single stream_id (must be int) to :class:`PhStream`"""
        kwargs = dict()
        if stream_id >= np.prod(self.shape):
            raise ValueError("stream_id {stream_id} out of detdef range ({np.cumprod(self.shape)})")
        for param, dim, stride in zip(self._params, self.shape, self.strides):
            if dim == 1:
                continue # if shape at given point is 1, then given param is irrelevant, equivalent to all
            kwargs[param], stream_id = divmod(stream_id, stride)
        kwargs = {k:[v] for k, v in kwargs.items()} # because PhStream assumes inputs are sequences
        return PhStream(**kwargs)
    
    def stream_ids_to_PhSel(self, stream_ids:np.ndarray[np.uint8], convert_all:bool=True)->PhSel:
        """
        Convert stream_ids (sequence of ints, preferable numpy array dtype=uint8)
        to :class:`PhSel` object.

        Parameters
        ----------
        stream_ids : np.ndarray[np.uint8]
            Sequence of stream ids 

        Returns
        -------
        PhSel
            :class:`PhSel` representation of given set of stream_ids.

        """
        stream_ids = np.asarray(stream_ids, dtype=np.uint8).reshape(-1)
        out = PhSel(*(self._stream_id_to_PhStream(stream_id) for stream_id in stream_ids))
        if convert_all:
            out = out.render_positive(self, convert_all=True)
        return out
    
    def __str__(self):
        return 'DetDef' + ''.join(f'{n}{p}' for n, p in zip(self.shape, self._params) if n != 1)
    
    def __repr__(self):
        return str(self) + f" at 0x{id(self):x}"


def check_PhSel(val:PhSel, render_positive:bool=False, detdef:DetDef=None)->PhSel:
    """
    Check val (a :class:`PhSel`) is valid for saving to HDF5 file.
    Used in TypeValidator for PhSel

    Parameters
    ----------
    val : PhSel
        Value to check.
    render_positive : bool, optional
        If :code:`True` will automatically render_positive val. The default is False.
    detdef : DetDef, optional
        :class:`DetDef` defining maximum valid index in each channel. The default is None.

    Raises
    ------
    TypeError
        val is not a :class:`PhSel`.

    Returns
    -------
    PhSel
        Verified :class:`PhSel`.

    """
    if isinstance(val, PhStream):
        val = PhSel(val)
    if not isinstance(val, PhSel):
        raise TypeError("must be PhSel")
    if detdef is not None and render_positive:
        val = val.render_positive(detdef)
    return val


def dread_PhSel(arr:bytes, dct:dict)->PhSel:
    """
    Direct read function for reading :class:`PhSel` from file, 
    takes bytes array stored in HDF5 array, and converts to :class:`PhSel`
    """
    return PhSel(arr.decode())


def dwrite_PhSel(val:PhSel)->bytes:
    """
    Direct write function for :class:`PhSel`. 
    Outputs bytes-representation of :class:`PhSel` to be written to HDF5 file.
    For use in TypeValidator object
    """
    return str(val).encode()


def node_repr_PhSel(val:PhSel)->str:
    """Generate node-string of PhSel, for use in TypeValidator object"""
    return val.attr_str


def node_read_PhSel(val:str)->PhSel:
    """Read node-string of PhSel, for use in TypeValidator object"""
    return PhSel.from_attr_str(val)

TV_PhSel = register_byteslike(PhSel, check_PhSel, dread_PhSel, dwrite_PhSel, 'phsel', node_repr_PhSel, node_read_PhSel)


def _none_or_equal(detdef:DetDef, val:int, spec:str)->None:
    """Verify that val of type spec in valid for a detdef, raises error if problem, otherwise return nothing"""
    if val is not None and getattr(detdef, spec) != val:
        raise ValueError(f"DetDef has {getattr(detdef, spec)}, but expected {val} for {spec}")


def check_DetDef(val:DetDef, ex=None, em=None, pol=None, split=None, **kwargs)->DetDef:
    """
    Check function for :class:`DetDef` to be used in TypeValidator.

    Parameters
    ----------
    val : DetDef
        :class:`DetDef` to convert/verify.
    ex : int|None, optional
        Number of expected ex channels. The default is None.
    em : TYPE, optional
        Number of expected em channels. The default is None.
    pol : TYPE, optional
        Number of expected pol channels. The default is None.
    split : TYPE, optional
        Number of expected split channels. The default is None.
    **kwargs : TYPE
        Ignored, necessary for use in TypeValidator.

    Raises
    ------
    TypeError
        Not DetDef or bad specification of one or more channels based on restrictions.

    Returns
    -------
    DetDef
        Verified :class:`DetDef` object.

    """
    if val is None:
        val = DetDef(ex=1 if ex is None else ex, em=1 if em is None else em,
                     pol=1 if pol is None else pol, split=1 if split is None else split)
    if not isinstance(val, DetDef):
        raise TypeError("det_def must be instance of DetDef")
    _none_or_equal(val, ex, 'ex')
    _none_or_equal(val, em, 'em')
    _none_or_equal(val, pol, 'pol')
    _none_or_equal(val, split, 'split')
    return val


def dread_DetDef(arr:np.ndarray[np.uint8], dct:dict)->DetDef:
    """
    Direct read function for reading :class:`DetDef` from HDF5 file.
    Takes bytes array and return :class:`DetDef`.
    For use in TypeValidator object
    """
    return DetDef(*arr)


def dwrite_DetDef(val:DetDef)->np.ndarray[np.uint8]:
    """
    Direct write function for :class:`DetDef`.
    Returns array to be written to HDF5 file
    For use in TypeValidator object.
    """
    return val.shape


TV_DetDef = register_byteslike(DetDef, check_DetDef, dread_DetDef, dwrite_DetDef)


def sort_phsels(detdef:DetDef, phsels:Sequence[PhSel], return_index:bool=False)->tuple[PhSel,...]:
    """
    Sorting function for ordering sequence of :class:`PhSel` objects.

    Parameters
    ----------
    detdef : DetDef
        Definition of detectos being used in current space, used to normalize
        for negatively defined PhSels.
    phsels : Sequence[PhSel]
        Sequence of :class:`PhSel` objects to be sorted.
    return_index : bool, optional
        If True, also return mapping of indexes in phsels to final array.
        The default is False.

    Returns
    -------
    phsels : tuple[PhSel, ...]
        Tuple of ordered phsel objects.
    sort : np.ndarray[np.int64]
        (Only in ``return_index=True``) indexes of original in sorted array.
        Ie ``[phsels[i] for i in sort]`` with return the same as first output.

    """
    stream_ids = tuple(detdef.get_stream_ids(sel) for sel in phsels)
    max_stream = max(s.shape[0] for s in stream_ids)
    stream_arr = -np.ones((max_stream, len(stream_ids)), dtype=np.int16)
    for i, s in enumerate(stream_ids):
        stream_arr[:s.size,i] = s
    sort = np.lexsort(stream_arr)
    out =  tuple(phsels[i] for i in sort)
    if return_index:
        out = out, sort
    return out


def mask_detarray(detdef:DetDef, phsel:PhSel, dets:np.ndarray[np.uint8])->np.ndarray[np.bool_]:
    r"""
    Generage a mask of dets for all photons in phsel

    Parameters
    ----------
    detdef : DetDef
        detector ranges.
    phsel : PhSel
        photon selection for which to generate the mask.
    dets : np.ndarray[np.uint8]
        detectors array to mask.

    Returns
    -------
    np.ndarray[np.bool\_]
        mask of all photons in phsel based on dets.

    """
    return np.isin(dets, detdef.get_stream_ids(phsel))


def reindex_phsel(detdef:DetDef, phsels:tuple[PhSel,...])->np.ndarray[np.int8]:
    """
    Generate a "map" array based on detdef and phsels. Allows mapping of detector
    ids defined by detdef (ie indexes in PhotonData.dets) to index defined by
    position of the given :class:`PhSel` in phsels.
    
    the map is the size of the nubmer of detector ids in detdef, detector ids
    not defined in phsels have a value of -1.
    
    The :code:`idx_map[PhotonData.dets]` will produce an array re-indexed according
    to phsels.

    Parameters
    ----------
    detdef : DetDef
        Detector definition.
    phsels : tuple[PhSel,...]
        phsels to convert.

    Returns
    -------
    idx_map : np.ndarray[np.int8]
        mapping of dets -> phsels, detector ids not covered in phsels have 
        value of -1.

    """
    idx_map = -np.ones(detdef.size, dtype=np.dtype('<i8'))
    for i, phsel in enumerate(phsels):
        idx_map[detdef.get_stream_ids(phsel)] = i
    return idx_map